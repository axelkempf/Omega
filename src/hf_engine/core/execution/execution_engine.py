from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from time import time

from strategies._base.base_strategy import TradeSetup

from hf_engine.adapter.broker.broker_interface import BrokerInterface, OrderResult
from hf_engine.adapter.broker.broker_utils import get_pip_size
from hf_engine.core.execution.execution_result import ExecutionResult
from hf_engine.core.execution.execution_tracker import ExecutionTracker
from hf_engine.core.execution.sl_tp_utils import distance_to_sl, ensure_abs_levels
from hf_engine.core.risk.lot_size_calculator import calculate_lot_size
from hf_engine.infra.logging.error_handler import safe_execute
from hf_engine.infra.logging.log_service import log_service
from hf_engine.infra.monitoring.telegram_bot import (
    send_telegram_message,
    send_watchdog_telegram_message,
)


class IdempotencyCache:
    """Einfacher TTL‑Cache für Idempotency‑Keys (thread‑safe)."""

    def __init__(self, ttl_sec: float = 180.0):
        self.ttl = float(ttl_sec)
        self._store: dict[str, float] = {}
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        now = time()
        with self._lock:
            # Cleanup abgelaufener Keys (cheap)
            if self._store:
                for k, exp in list(self._store.items()):
                    if exp < now:
                        self._store.pop(k, None)
            # Prüfen/Setzen
            if key in self._store and self._store[key] > now:
                return True
            self._store[key] = now + self.ttl
            return False


class ExecutionEngine:
    def __init__(
        self,
        broker: BrokerInterface,
        magic_number: int,
        idem_cache: IdempotencyCache | None = None,
        io_timeout_sec: float = 8.0,
    ):  # ↑ Default großzügiger
        self.broker = broker
        self.magic_number = magic_number
        self._io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ExecIO")
        self._timeout_sec = float(io_timeout_sec)
        self._idem = idem_cache or IdempotencyCache(ttl_sec=180.0)
        self.tracker = ExecutionTracker()

    # --- Idempotency / Duplicate‑Guard ---------------------------------------
    def _build_idem_key(self, setup: TradeSetup) -> str:
        md = getattr(setup, "metadata", {}) or {}
        strategy = md.get("strategy_id") or getattr(setup, "strategy", "") or "Strategy"
        symbol = getattr(setup, "symbol", "UNK")
        side = (getattr(setup, "direction", "") or "").lower()
        otype = (getattr(setup, "order_type", "") or "").lower()
        # bevorzugt deterministische Zeit aus dem Signal/Bar
        as_of = (
            md.get("as_of") or md.get("signal_time") or str(int(time() // 60))
        )  # Minute als Fallback
        # Entry binning (preisunabhängig ≈ robust), nutzt Pip falls verfügbar
        try:
            pip = get_pip_size(symbol)
            entry_bin = int(round(float(getattr(setup, "entry", 0.0)) / (pip or 1.0)))
        except Exception:
            entry_bin = int(round(float(getattr(setup, "entry", 0.0)) * 1e5))
        return f"{self.magic_number}:{strategy}:{symbol}:{as_of}:{side}:{otype}:{entry_bin}"

    def make_idempotency_key(self, setup: TradeSetup) -> str:
        return self._build_idem_key(setup)

    def _guard_idem(self, setup: TradeSetup) -> bool:
        md = getattr(setup, "metadata", {}) or {}
        key = md.get("idempotency_key") or self._build_idem_key(setup)
        if self._idem.seen(key):
            log_service.log_system(
                f"[OrderRouter] duplicate suppressed key={key}", level="INFO"
            )
            return False
        # Key für Downstream (Broker, Tracker, Logs)
        md["idempotency_key"] = key
        try:
            setup.metadata = md
        except Exception:
            pass
        return True

    def _call_broker(self, fn, *args, **kwargs):
        fut = self._io_pool.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=self._timeout_sec)
        except FuturesTimeout:
            log_service.log_system("[BrokerIO] ⏱️ Timeout", level="ERROR")
            return None
        except Exception as e:
            log_service.log_system(f"[BrokerIO] ❌ {e}", level="ERROR")
            return None

    # --- Gemeinsame Guards (Preisfenster, Mindestdistanz) ---
    def _pre_trade_price_guards(self, setup: TradeSetup) -> tuple[bool, str]:
        """
        Praxisnahe Pre-Trade-Guards für MARKET und PENDING (limit/stop).

        Konfigurierbare Policies via setup.metadata:
          - min_sl_pips: float = 3.0
          - max_deviation_pips (nur MARKET): float = 3.0
          - volatility_buffer_pips (nur MARKET, additiv): float = 0.0
          - max_spread_pips (nur MARKET, falls Broker es liefert): float = 2.0
          - min_pending_distance_pips (nur PENDING): float = 2.0
          - allow_marketable_pending: bool = False
        """
        order_type = (setup.order_type or "market").lower()
        direction = (setup.direction or "").lower()

        # 1) SL/TP-Validierung (absolute Level & Richtungslogik)
        try:
            levels = ensure_abs_levels(setup.entry, setup.sl, setup.tp, direction)
        except Exception as e:
            return False, f"SL/TP ungültig: {e}"

        # Punkt-/Tick-Größe je Symbol
        from hf_engine.core.risk.lot_size_calculator import get_price_spec

        point = get_price_spec(setup.symbol, self.broker)["point"]

        # 2) Mindestdistanz Entry ↔ SL (in Pips)
        min_sl_points = float(setup.metadata.get("min_sl_points", 3.0))
        if (distance_to_sl(setup.entry, levels.sl) / point) < min_sl_points:
            return False, f"SL zu nah (< {min_sl_points} points)"

        # Aktueller markt-relevanter Preis (bid/ask je Richtung)
        mkt_price = self.broker.get_symbol_price(setup.symbol, direction)

        # 3) Verzweigung nach Order-Typ
        if order_type == "market":
            # # 3a) Slippage-/Volatilitäts-Toleranz
            # base_dev = float(setup.metadata.get("max_deviation_pips", 3.0))
            # vola_buf = float(setup.metadata.get("volatility_buffer_pips", 0.0))
            # max_dev_pips = max(0.0, base_dev + vola_buf)

            # dev_pips = round(abs(mkt_price - setup.entry) / pip,6)
            # if dev_pips > max_dev_pips:
            #     return False, f"Preisabweichung {dev_pips:.2f}p > Limit {max_dev_pips:.2f}p (Market)"

            # # 3b) Spread-Guard (wenn Broker es anbietet)
            # max_spread_pips = float(setup.metadata.get("max_spread_pips", 2.0))
            # try:
            #     spread_abs = float(self.broker.get_symbol_spread(setup.symbol))  # abs Preis
            #     spread_pips = spread_abs / pip
            #     if spread_pips > max_spread_pips:
            #         return False, f"Spread {spread_pips:.2f}p > Limit {max_spread_pips:.2f}p"
            # except Exception:
            #     # Broker liefert evtl. keinen Spread — Guard stillschweigend überspringen
            #     pass

            return True, "OK (Market)"

        elif order_type in ("limit", "stop"):
            # 3c) Pending-Guards: korrekte Seite + Mindestabstand zum Markt
            min_pending_distance_pips = float(
                setup.metadata.get("min_pending_distance_pips", 2.0)
            )
            allow_marketable_pending = bool(
                setup.metadata.get("allow_marketable_pending", False)
            )

            dist_pts = round(abs(setup.entry - mkt_price) / point, 6)

            if order_type == "limit":
                if direction == "buy":
                    # Buy Limit MUSS UNTER Marktpreis liegen
                    if setup.entry >= mkt_price:
                        # Marktgängig?
                        if allow_marketable_pending:
                            return True, "OK (Limit→Market erlaubt)"
                        return False, "Buy Limit liegt nicht unter Marktpreis"
                else:  # sell
                    # Sell Limit MUSS ÜBER Marktpreis liegen
                    if setup.entry <= mkt_price:
                        if allow_marketable_pending:
                            return True, "OK (Limit→Market erlaubt)"
                        return False, "Sell Limit liegt nicht über Marktpreis"

            else:  # order_type == "stop"
                if direction == "buy":
                    # Buy Stop MUSS ÜBER Marktpreis liegen
                    if setup.entry <= mkt_price:
                        if allow_marketable_pending:
                            return True, "OK (Stop→Market erlaubt)"
                        return False, "Buy Stop liegt nicht über Marktpreis"
                else:  # sell
                    # Sell Stop MUSS UNTER Marktpreis liegen
                    if setup.entry >= mkt_price:
                        if allow_marketable_pending:
                            return True, "OK (Stop→Market erlaubt)"
                        return False, "Sell Stop liegt nicht unter Marktpreis"

            # Mindestabstand zum Markt für Pending (bspw. Freeze-Level/Invalid Price vermeiden)
            min_pending_distance_points = float(
                setup.metadata.get("min_pending_distance_points", 2.0)
            )
            if dist_pts < min_pending_distance_points:
                return (
                    False,
                    f"Pending-Entry zu nah am Markt (< {min_pending_distance_points:.2f} points)",
                )

            return True, "OK (Pending)"

        else:
            return False, f"Ungültiger order_type: {order_type}"

    def place_pending_order(self, setup: TradeSetup) -> int | None:
        def inner():
            ok, reason = self._pre_trade_price_guards(setup)
            if not ok:
                log_service.log_system(
                    f"[OrderGuard] {setup.symbol} → {reason}", level="WARNING"
                )
                msg = f"⚠️ *OrderGuard (Pending)* `{setup.symbol}`: {reason}"
                send_telegram_message(msg)
                send_watchdog_telegram_message(msg)
                return None

            if not self._guard_idem(setup):
                return None

            lot = calculate_lot_size(setup, self.broker)
            order_type = (setup.order_type or "stop").lower()
            if order_type not in ("stop", "limit"):
                msg = f"Ungültiger Pending-Typ: {order_type}"
                log_service.log_system(f"[Fehler] {msg}", level="ERROR")
                send_telegram_message(f"🚨 {msg}")
                send_watchdog_telegram_message(f"🚨 {msg}")
                return None

            log_service.log_system(
                f"[OrderAttempt] {setup.symbol} {setup.direction.upper()} {order_type} "
                f"@ {setup.entry} | SL={setup.sl}, TP={setup.tp}, Vol={lot}"
            )

            key = self.make_idempotency_key(setup)
            result: OrderResult = self._call_broker(
                self.broker.place_pending_order,
                symbol=setup.symbol,
                direction=setup.direction,
                entry=setup.entry,
                sl=setup.sl,
                tp=setup.tp,
                volume=lot,
                comment=(
                    ("Stop Order" if order_type == "stop" else "Limit Order")
                    + f" |ID:{key[-12:]}"
                )[:31],
                order_type=order_type,
                magic_number=self.magic_number,
                scenario=setup.metadata.get("scenario", None),
            )

            if not result or not result.success:
                msg = result.message if result else "Keine Antwort"
                log_service.log_system(
                    f"[Order Fehler] {setup.symbol}: {msg}", level="ERROR"
                )
                err_msg = f"🚨 *Order-Fehler* `{setup.symbol}`:\n{msg}"
                send_telegram_message(err_msg)
                send_watchdog_telegram_message(err_msg)
                return None

            log_service.log_system(
                f"[Order Erfolgreich] {setup.symbol}: Order-ID {result.order}"
            )
            send_telegram_message(
                f"✅ *Pending platziert*: `{setup.symbol}` | {setup.direction.upper()} @ `{round(setup.entry,5)}`"
            )
            return result.order

        return safe_execute(f"PendingOrder-{setup.symbol}", inner) or None

    def place_market_order(self, setup: TradeSetup) -> int | None:
        def inner():
            ok, reason = self._pre_trade_price_guards(setup)
            if not ok:
                log_service.log_system(
                    f"[OrderGuard] {setup.symbol} → {reason}", level="WARNING"
                )
                msg = f"⚠️ *OrderGuard (Market)* `{setup.symbol}`: {reason}"
                send_telegram_message(msg)
                send_watchdog_telegram_message(msg)
                return None

            if not self._guard_idem(setup):
                return None

            lot = calculate_lot_size(setup, self.broker)
            log_service.log_system(
                f"[OrderAttempt] {setup.symbol} {setup.direction.upper()} MARKET "
                f"@ {setup.entry} | SL={setup.sl}, TP={setup.tp}, Vol={lot}"
            )
            key = self.make_idempotency_key(setup)
            result: OrderResult = self._call_broker(
                self.broker.place_market_order,
                symbol=setup.symbol,
                direction=setup.direction,
                sl=setup.sl,
                tp=setup.tp,
                volume=lot,
                comment=("Market Order" + f" |ID:{key[-12:]}")[:31],
                magic_number=self.magic_number,
                scenario=setup.metadata.get("scenario", None),
            )

            if not result or not result.success:
                msg = result.message if result else "Keine Antwort"
                log_service.log_system(
                    f"[Order Fehler] {setup.symbol}: {msg}", level="ERROR"
                )
                err_msg = f"🚨 *Market-Order Fehler*: `{setup.symbol}`\n{msg}"
                send_telegram_message(err_msg)
                send_watchdog_telegram_message(err_msg)
                return None

            # Tracking – korrekte, vom Broker vergebene Order-ID verwenden
            oid = getattr(result, "order_id", None)
            if oid is None:
                oid = getattr(result, "order", None)
            try:
                order_id_int = int(oid) if oid is not None else None
            except Exception:
                order_id_int = None

            if order_id_int is not None:
                self.tracker.mark_trade_open(
                    symbol=setup.symbol,
                    strategy=setup.metadata.get("setup", setup.strategy),
                    order_id=order_id_int,
                    direction=setup.direction,
                    entry_price=setup.entry,
                    volume=lot,
                    sl=setup.sl,
                    tp=setup.tp,
                    scenario=setup.metadata.get("scenario", None),
                )
            else:
                log_service.log_system(
                    "[OrderTracking] Keine gültige order_id erhalten – Eintrag wird übersprungen",
                    level="WARNING",
                )

            log_service.log_system(
                f"[Order Erfolgreich] {setup.symbol}: Order-ID {result.order}"
            )
            # Kurze Kennungen für Strategie und Szenario für kompakte Telegram-Nachricht
            strategy_name = getattr(setup, "strategy", "") or ""
            strategy_clean = strategy_name.replace("_", " ").replace("-", " ").strip()
            strategy_parts = [p for p in strategy_clean.split(" ") if p]
            strategy_tag = "".join(part[0].upper() for part in strategy_parts)

            scenario_raw = str(setup.metadata.get("scenario") or "")
            scenario_digits = "".join(ch for ch in scenario_raw if ch.isdigit())
            scenario_tag = f"S{scenario_digits}" if scenario_digits else ""

            timeframe_raw = str(setup.metadata.get("timeframe") or "")
            timeframe_tag = timeframe_raw.strip().upper()

            tag = ""
            if strategy_tag:
                tag = strategy_tag
            if scenario_tag:
                tag = f"{tag} {scenario_tag}" if tag else scenario_tag
            if timeframe_tag:
                tag = f"{tag} {timeframe_tag}" if tag else timeframe_tag

            header = "📉 *Market ausgeführt*"
            if tag:
                header += f" `{tag}`"

            send_telegram_message(
                header + "\n"
                f"• Symbol: `{setup.symbol}`  • Richtung: *{setup.direction.upper()}*\n"
                f"• Entry: `{round(setup.entry,5)}`  • SL: `{round(setup.sl,5)}`  • TP: `{round(setup.tp,5)}`\n"
                f"• Volumen: `{lot}`"
            )
            return result.order

        return safe_execute(f"MarketOrder-{setup.symbol}", inner) or None

    def cancel_pending_orders(self, symbol: str):
        safe_execute(
            f"CancelPending-{symbol}", self.broker.cancel_all_pending_orders, symbol
        )
        log_service.log_system(f"[Cancel] Alle Pending Orders für {symbol} gelöscht")
        send_telegram_message(f"🧹 *Pending Orders gelöscht*: `{symbol}`")

    def close_position(self, ticket_id: int, symbol: str) -> ExecutionResult:
        result = safe_execute(
            f"CloseFull-{symbol}", self.broker.close_position_full, ticket_id
        )
        if result and getattr(result, "success", False):
            log_service.log_system(f"[Close] {symbol} Position {ticket_id} geschlossen")
            send_telegram_message(f"❌ *Position geschlossen*: `{symbol}`")
            return (
                result
                if isinstance(result, ExecutionResult)
                else ExecutionResult(True, "OK")
            )
        msg = (
            result.message
            if hasattr(result, "message")
            else (str(result) or "Keine Antwort")
        )
        log_service.log_system(f"[Close Fehler] {symbol}: {msg}", level="ERROR")
        err_msg = f"🚨 *Close-Fehler*: `{symbol}`\n{msg}"
        send_telegram_message(err_msg)
        send_watchdog_telegram_message(err_msg)
        return ExecutionResult(False, msg)

    def close_position_partial(
        self, ticket_id: int, volume: float, symbol: str
    ) -> ExecutionResult:
        result = safe_execute(
            f"ClosePartial-{symbol}",
            self.broker.close_position_partial,
            ticket_id,
            volume,
        )
        if result and getattr(result, "success", False):
            log_service.log_system(
                f"[Teil-Close] {symbol} {volume} Lots von {ticket_id} geschlossen"
            )
            send_telegram_message(
                f"🔻 *Teilweise geschlossen*: `{symbol}` ({volume} Lots)"
            )
            return (
                result
                if isinstance(result, ExecutionResult)
                else ExecutionResult(True, "OK")
            )
        msg = (
            result.message
            if hasattr(result, "message")
            else (str(result) or "Keine Antwort")
        )
        log_service.log_system(f"[Teil-Close Fehler] {symbol}: {msg}", level="ERROR")
        err_msg = f"🚨 *Teil-Close Fehler*: `{symbol}`\n{msg}"
        send_telegram_message(err_msg)
        send_watchdog_telegram_message(err_msg)
        return ExecutionResult(False, msg)

    def modify_stop_loss(
        self, ticket_id: int, new_sl: float, symbol: str = ""
    ) -> ExecutionResult:
        result = safe_execute(
            f"ModifySL-{symbol}", self.broker.modify_sl, ticket_id, new_sl=new_sl
        )
        if result and getattr(result, "success", False):
            log_service.log_system(f"[SL-Update] {symbol}: SL={new_sl}")
            send_telegram_message(f"✏️ *SL aktualisiert*: `{symbol}`\nSL={new_sl}")
            return (
                result
                if isinstance(result, ExecutionResult)
                else ExecutionResult(True, "OK")
            )
        msg = (
            result.message
            if hasattr(result, "message")
            else (str(result) or "Keine Antwort")
        )
        log_service.log_system(f"[SL-Fehler] {symbol}: {msg}", level="ERROR")
        err_msg = f"🚨 *SL Fehler*: `{symbol}`\n{msg}"
        send_telegram_message(err_msg)
        send_watchdog_telegram_message(err_msg)
        return ExecutionResult(False, msg)

    def modify_take_profit(
        self, ticket_id: int, new_tp: float, symbol: str = ""
    ) -> ExecutionResult:
        result = safe_execute(
            f"ModifyTP-{symbol}", self.broker.modify_tp, ticket_id, new_tp=new_tp
        )
        if result and getattr(result, "success", False):
            log_service.log_system(f"[TP-Update] {symbol}: TP={new_tp}")
            send_telegram_message(f"✏️ *TP aktualisiert*: `{symbol}`\nTP={new_tp}")
            return (
                result
                if isinstance(result, ExecutionResult)
                else ExecutionResult(True, "OK")
            )
        msg = (
            result.message
            if hasattr(result, "message")
            else (str(result) or "Keine Antwort")
        )
        log_service.log_system(f"[TP-Fehler] {symbol}: {msg}", level="ERROR")
        err_msg = f"🚨 *TP Fehler*: `{symbol}`\n{msg}"
        send_telegram_message(err_msg)
        send_watchdog_telegram_message(err_msg)
        return ExecutionResult(False, msg)
