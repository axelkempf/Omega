# hf_engine/core/controlling/strategy_runner.py
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from hf_engine.adapter.broker.broker_interface import BrokerInterface
from hf_engine.adapter.data.mt5_data_provider import MT5DataProvider
from hf_engine.core.controlling.event_bus import Event, EventType
from hf_engine.core.controlling.position_monitor_controller import (
    PositionMonitorController,
)
from hf_engine.core.controlling.session_runner import SessionRunner
from hf_engine.core.execution.execution_engine import ExecutionEngine
from hf_engine.core.execution.execution_tracker import ExecutionTracker
from hf_engine.core.risk.risk_manager import RiskManager
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.error_handler import safe_execute
from hf_engine.infra.logging.log_service import log_service
from strategies._base.base_position_manager import strategy_position_manager_factory


class StrategyRunner:
    def __init__(
        self,
        strategy,
        broker: BrokerInterface,
        data_provider: MT5DataProvider,
        magic_number,
        controller=None,
        symbol_mapper=None,
    ):
        self.strategy = strategy
        self.broker = broker
        self.data_provider = data_provider
        self.session_runner = SessionRunner(strategy)
        self.magic_number = magic_number

        self.execution_tracker = ExecutionTracker()
        io_to = float(self.strategy.config.get("io_timeout_sec", 12.0))
        self.execution_engine = ExecutionEngine(
            broker, magic_number, io_timeout_sec=io_to
        )
        self.risk_manager = RiskManager(broker)
        self.controller = controller
        self.symbol_mapper = symbol_mapper

        self._last_barclose_run: Optional[float] = None
        self._tf: Optional[str] = getattr(
            strategy, "timeframe", None
        ) or strategy.config.get("timeframe")
        # Verhindert überlappende run_daily_cycle() Aufrufe pro Runner
        self._cycle_lock = threading.Lock()

        self.position_monitor_controller = PositionMonitorController()
        self.position_monitor_controller.start()

        # Konfigurierbarer Pool (Fallback 8)
        try:
            workers = int(self.strategy.config.get("runner_max_workers", 1))
        except Exception:
            workers = 8
        self._pool = ThreadPoolExecutor(max_workers=max(2, workers))

        # Tunables (konfigurierbar mit Fallback)
        try:
            self._signal_timeout_sec = float(
                self.strategy.config.get("signal_timeout_sec", 6.0)
            )
        except Exception:
            self._signal_timeout_sec = 2.0
        try:
            cfg_order_to = float(self.strategy.config.get("order_timeout_sec", 14.0))
        except Exception:
            cfg_order_to = 10.0
        # WICHTIG: Außen muss >= innen sein, sonst bekommen wir Ghost-Orders
        self._order_timeout_sec = max(cfg_order_to, io_to + 2.0)

        # Startup reconciliation: resume managers for already open positions
        try:
            self._resume_open_positions()
        except Exception as e:
            # Never fail constructor because of resume flow
            log_service.log_system(
                f"[Resume] Initialer Resume-Check fehlgeschlagen: {self.strategy.name()} – {e}",
                level="ERROR",
            )

    # --- Logging Helpers -------------------------------------------------
    def _short_repr(self, value: Any, max_len: int = 120) -> str:
        """Create a safe, short representation for logging fields."""
        try:
            text = repr(value)
        except Exception:
            return "<unrepr-able>"
        if len(text) > max_len:
            return f"{text[:max_len]}…(+{len(text) - max_len} chars)"
        return text

    def _log_signal_flow(
        self,
        symbol: Optional[str],
        step: str,
        details: str | None = None,
        *,
        level: str = "INFO",
    ) -> None:
        """Emit a structured log line for signal related processing."""
        try:
            strat_name = (
                self.strategy.name()
                if hasattr(self.strategy, "name")
                else self.strategy.__class__.__name__
            )
        except Exception:
            strat_name = self.strategy.__class__.__name__

        tf = (self._tf or "-").upper()
        sym = symbol or "-"
        msg = f"[SignalFlow|{strat_name}|{sym}|tf={tf}] {step}"
        if details:
            msg = f"{msg} - {details}"
        try:
            log_service.log_system(msg, level=level)
        except Exception:
            # Logging must never raise inside trading loop
            pass

    def _format_setup_details(self, setup) -> str:
        """Collect relevant setup attributes for compact logging."""
        parts: List[str] = []
        for attr in (
            "direction",
            "order_type",
            "volume",
            "entry",
            "entry_price",
            "sl",
            "stop_loss",
            "tp",
            "take_profit",
        ):
            if hasattr(setup, attr):
                value = getattr(setup, attr, None)
                if value is not None:
                    parts.append(f"{attr}={self._short_repr(value)}")

        metadata = getattr(setup, "metadata", None)
        if isinstance(metadata, dict):
            for key in (
                "strategy_id",
                "signal_id",
                "reason",
                "confidence",
                "direction",
                "order_type",
                "idempotency_key",
            ):
                if key in metadata:
                    parts.append(f"meta.{key}={self._short_repr(metadata[key])}")

        return ", ".join(parts) if parts else setup.__class__.__name__

    def _normalize_signals(self, signals) -> tuple[List[Any], int]:
        """
        Normalize user supplied signals into a clean list.

        Returns a tuple (normalized_signals, dropped_none_count).
        """
        if signals is None:
            return [], 0

        if isinstance(signals, str):
            iterable = [signals]
        elif isinstance(signals, (list, tuple, set)):
            iterable = list(signals)
        else:
            try:
                iterable = list(signals)
            except TypeError:
                iterable = [signals]

        normalized: List[Any] = []
        dropped = 0
        for item in iterable:
            if item is None:
                dropped += 1
                continue
            normalized.append(item)
        return normalized, dropped

    # --- robustes Timeframe-Matching für BAR_CLOSE-Events ---
    def _tf_matches(self, event_tf: Optional[str]) -> bool:
        """
        Vergleicht das BAR_CLOSE-Timeframe mit dem Strategie-Timeframe.
        - Case-insensitiv
        - Verträgt fehlende / leere Angaben defensiv
        """
        try:
            raw_strat_tf = self.strategy.config.get("timeframe", "")
            strat_tf = str(raw_strat_tf or "").strip().upper()
            ev_tf = str(event_tf or "").strip().upper()
            if not strat_tf:
                log_service.log_system(
                    f"[{self.strategy.name()}] _tf_matches skip: Strategie-Timeframe fehlt",
                    level="WARNING",
                )
                return False
            if not ev_tf:
                log_service.log_system(
                    f"[{self.strategy.name()}] _tf_matches skip: Event-Timeframe fehlt",
                    level="DEBUG",
                )
                return False
            return ev_tf == strat_tf
        except Exception as e:
            log_service.log_system(
                f"[{self.strategy.name()}] _tf_matches error: {e}", level="DEBUG"
            )
            return True

    def _ensure_utc(self, dt: datetime) -> datetime:
        # Naive Zeitstempel als UTC interpretieren; aware → nach UTC konvertieren
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _parse_iso_to_utc(self, value: datetime | str, fallback: datetime) -> datetime:
        """ISO‑String robust nach UTC parsen (handhabt 'Z' und naive Inputs)."""
        try:
            if isinstance(value, datetime):
                dt = value
            else:
                txt = str(value).strip()
                if txt.endswith("Z"):
                    txt = txt[:-1] + "+00:00"
                dt = datetime.fromisoformat(txt)
            return self._ensure_utc(dt)
        except Exception as e:
            log_service.log_system(
                f"[{self.strategy.name()}] closed_at parse failed ('{value}'): {e}",
                level="DEBUG",
            )
            return fallback

    def _get_symbols(self) -> List[str]:
        try:
            symbols = list(self.strategy.config.get("symbols", []))
            if not symbols:
                log_service.log_system(
                    f"[{self.strategy.name()}] ⚠️ Keine Symbole in strategy.config['symbols'] definiert.",
                    level="WARNING",
                )
            return symbols
        except Exception as e:
            log_service.log_system(
                f"[{self.strategy.name()}] Symbols lesen fehlgeschlagen: {e}",
                level="ERROR",
            )
            return []

    def on_event(self, event: Event):
        now_dt = now_utc()  # immer aware UTC

        if event.type == EventType.TIMER_TICK:
            # Heartbeat only – keine schweren Workloads hier
            return

        elif event.type == EventType.BAR_CLOSE:
            ev_tf = (event.payload or {}).get("timeframe")
            if self._tf_matches(ev_tf):
                self._last_barclose_run = time.time()
                closed_at = (event.payload or {}).get("closed_at")
                when = (
                    self._parse_iso_to_utc(closed_at, now_dt) if closed_at else now_dt
                )
                ready_symbols = None
                raw_symbols = (event.payload or {}).get("symbols")
                if raw_symbols:
                    ready_symbols = {
                        str(sym).strip()
                        for sym in raw_symbols
                        if isinstance(sym, str) and sym.strip()
                    }
                self.run_daily_cycle(when, symbols_filter=ready_symbols, closed_at=when)

        elif event.type == EventType.SESSION_OPEN:
            self.run_daily_cycle(now_dt)

        elif event.type == EventType.AUTO_CLOSE:
            try:
                for symbol in self._get_symbols():
                    self.execution_engine.cancel_pending_orders(symbol)
            except Exception as e:
                log_service.log_system(
                    f"[{self.strategy.name()}] AUTO_CLOSE cleanup error: {e}",
                    level="ERROR",
                )

        elif event.type == EventType.NEWS:
            # PRE/LIVE: Pending Orders löschen (optional)
            try:
                phase = (event.payload or {}).get("phase", "").upper()
                affected = (event.payload or {}).get(
                    "symbols", []
                ) or self._get_symbols()
                if phase in ("PRE", "LIVE"):
                    for s in affected:
                        self.execution_engine.cancel_pending_orders(s)
            except Exception as e:
                log_service.log_system(
                    f"[{self.strategy.name()}] NEWS handler error: {e}",
                    level="ERROR",
                )

    def _ensure_broker(self) -> bool:
        ok = self.broker.ensure_connection()
        if not ok:
            log_service.log_system(
                f"[{self.strategy.name()}] ❌ Brokerverbindung gestört."
            )
            try:
                if self.controller and hasattr(self.controller, "event_bus"):
                    self.controller.event_bus.publish(
                        Event(EventType.BROKER_STATUS, {"status": "DEGRADED"})
                    )
            except Exception as e:
                log_service.log_system(
                    f"[{self.strategy.name()}] Broker-Status Publish fail: {e}",
                    level="DEBUG",
                )
        return ok

    def _symbol_closed_ready(self, symbol: str, closed_at: datetime) -> bool:
        if closed_at is None:
            return True
        if self.data_provider is None:
            return True
        try:
            rec = self.data_provider.get_ohlc_for_closed_candle(
                symbol, self._tf, offset=1
            )
        except Exception:
            return False
        if not isinstance(rec, dict):
            return False
        raw_time = rec.get("time")
        if raw_time is None:
            return False
        dt = self._parse_iso_to_utc(raw_time, closed_at)
        return dt >= closed_at

    def _tf_to_timedelta(self) -> timedelta:
        """timedelta passend zum Strategy‑Timeframe. Fallback 0."""
        tf = (self._tf or "").upper()
        mapping = {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
            "W1": timedelta(weeks=1),
            # Optional: Monatskerze – grobe Annahme 30 Tage
            "MN1": timedelta(days=30),
        }
        return mapping.get(tf, timedelta(0))

    def run_daily_cycle(
        self,
        now: datetime,
        symbols_filter: Optional[Set[str]] = None,
        closed_at: Optional[datetime] = None,
    ):
        waited_for_lock = False
        if not self._cycle_lock.acquire(blocking=False):
            waited_for_lock = True
            self._log_signal_flow(
                "*",
                "cycle:wait",
                "Vorheriger Durchlauf noch aktiv – warte auf Freigabe",
                level="DEBUG",
            )
            self._cycle_lock.acquire()

        try:
            if waited_for_lock:
                self._log_signal_flow(
                    "*",
                    "cycle:resume",
                    "Vorheriger Durchlauf beendet – starte neuen Zyklus",
                    level="DEBUG",
                )

            if not self._ensure_broker():
                return
            # Für BAR_CLOSE-Events liegt 'closed_at' auf Bar‑Open. Für die
            # Session‑Prüfung shiften wir auf Bar‑Close (closed_at + tf_delta).
            check_dt = (closed_at + self._tf_to_timedelta()) if closed_at else now
            if not self.session_runner.is_trade_window_open(check_dt):
                self._log_signal_flow(
                    "*",
                    "cycle:skip",
                    f"Handelsfenster geschlossen (check_dt={check_dt.replace(microsecond=0).isoformat()})",
                    level="DEBUG",
                )
                return

            symbols = self._get_symbols()
            filter_set = None
            if symbols_filter:
                filter_set = {sym.upper() for sym in symbols_filter}
                symbols = [sym for sym in symbols if sym.upper() in filter_set]

            if not symbols:
                self._log_signal_flow(
                    "*",
                    "cycle:skip",
                    (
                        "Keine Symbole für Strategie konfiguriert"
                        if not symbols_filter
                        else "Keine bereitgestellten Symbole im Event"
                    ),
                    level="WARNING",
                )
                return

            self._log_signal_flow(
                "*",
                "cycle:start",
                f"now={now.replace(microsecond=0).isoformat()} check_dt={check_dt.replace(microsecond=0).isoformat()} symbols={symbols}",
                level="DEBUG",
            )

            futures: Dict = {}
            # Event-spezifische Symbole gelten als ready; vermeidet doppelte Provider-Abfragen
            skip_readiness_check = bool(closed_at and filter_set)

            # 1) Signale parallel ermitteln (nicht blockieren)
            for symbol in symbols:
                if closed_at:
                    if skip_readiness_check:
                        self._log_signal_flow(
                            symbol,
                            "signals:ready",
                            "Readiness aus Event-Payload übernommen",
                            level="DEBUG",
                        )
                    else:
                        if not self._symbol_closed_ready(symbol, closed_at):
                            self._log_signal_flow(
                                symbol,
                                "signals:skip",
                                f"closed candle {closed_at.isoformat()} nicht verfügbar",
                                level="DEBUG",
                            )
                            continue
                detail_suffix = (
                    " (skip_ready_check)" if closed_at and skip_readiness_check else ""
                )
                self._log_signal_flow(
                    symbol,
                    "signals:request",
                    f"as_of={now.replace(microsecond=0).isoformat()}{detail_suffix}",
                    level="DEBUG",
                )
                fut = self._pool.submit(
                    safe_execute,
                    "SignalGen",
                    self.strategy.generate_signal,
                    symbol,
                    now,
                    self.broker,
                    self.data_provider,
                )
                futures[fut] = symbol

            # Zeitbudget für die Gesamtsammlung (defensiv >= 0.5s pro Symbol)
            global_collect_timeout = max(self._signal_timeout_sec, 0.5) * len(symbols)

            remaining = set(futures.keys())
            try:
                for fut in as_completed(remaining, timeout=global_collect_timeout):
                    symbol = futures[fut]
                    remaining.discard(fut)
                    try:
                        signals = fut.result(timeout=self._signal_timeout_sec)
                    except TimeoutError:
                        self._log_signal_flow(
                            symbol,
                            "signals:timeout",
                            f"timeout={self._signal_timeout_sec}s",
                            level="WARNING",
                        )
                        log_service.log_system(
                            f"[SignalGen] ⏱️ Timeout für {symbol}", level="WARNING"
                        )
                        continue
                    except Exception as e:
                        self._log_signal_flow(
                            symbol,
                            "signals:error",
                            self._short_repr(e),
                            level="ERROR",
                        )
                        log_service.log_system(
                            f"[SignalGen] ❌ Fehler für {symbol}: {e}", level="ERROR"
                        )
                        continue

                    normalized_signals, dropped = self._normalize_signals(signals)
                    if dropped:
                        self._log_signal_flow(
                            symbol,
                            "signals:normalize",
                            f"ignored_none={dropped}",
                            level="DEBUG",
                        )

                    if not normalized_signals:
                        self._log_signal_flow(
                            symbol,
                            "signals:empty",
                            "generate_signal lieferte kein Setup",
                            level="INFO",
                        )
                        continue

                    total_setups = len(normalized_signals)
                    self._log_signal_flow(
                        symbol,
                        "signals:received",
                        f"count={total_setups}",
                        level="INFO",
                    )

                    for idx, setup in enumerate(normalized_signals, start=1):
                        # Setup defensiv normalisieren
                        try:
                            setattr(setup, "symbol", symbol)
                        except Exception as e:
                            log_service.log_system(
                                f"[SetupNormalize] Symbol setzen fehlgeschlagen: {e}",
                                level="DEBUG",
                            )

                        if (
                            not hasattr(setup, "metadata")
                            or getattr(setup, "metadata", None) is None
                        ):
                            try:
                                setup.metadata = {}
                            except Exception as e:
                                log_service.log_system(
                                    f"[SetupNormalize] Metadata init fehlgeschlagen: {e}",
                                    level="DEBUG",
                                )

                        # deterministischer Kontext
                        try:
                            md = getattr(setup, "metadata", {}) or {}
                            strat_name = (
                                self.strategy.name()
                                if hasattr(self.strategy, "name")
                                else self.strategy.__class__.__name__
                            )
                            as_of = now.replace(microsecond=0).isoformat()
                            md.setdefault("strategy_id", strat_name)
                            md.setdefault("as_of", as_of)
                            if getattr(setup, "direction", None) is not None:
                                md.setdefault("direction", setup.direction)
                            if getattr(setup, "order_type", None) is not None:
                                md.setdefault("order_type", setup.order_type)
                            setup.metadata = md
                        except Exception as e:
                            log_service.log_system(
                                f"[SetupNormalize] Kontext setzen fehlgeschlagen: {e}",
                                level="DEBUG",
                            )

                        # start_capital sicherstellen
                        if not hasattr(setup, "start_capital"):
                            start_cap = None
                            try:
                                if hasattr(self.broker, "get_equity") and callable(
                                    self.broker.get_equity
                                ):
                                    start_cap = float(self.broker.get_equity())
                                elif hasattr(self.broker, "get_balance") and callable(
                                    self.broker.get_balance
                                ):
                                    start_cap = float(self.broker.get_balance())
                            except Exception:
                                start_cap = None
                            if start_cap is None:
                                try:
                                    start_cap = float(
                                        getattr(setup, "metadata", {}).get(
                                            "start_capital", 100000.0
                                        )
                                    )
                                except Exception:
                                    start_cap = 100000.0
                            try:
                                setup.start_capital = start_cap
                            except Exception as e:
                                log_service.log_system(
                                    f"[SetupNormalize] start_capital setzen fehlgeschlagen: {e}",
                                    level="DEBUG",
                                )

                        setup_details = self._format_setup_details(setup)
                        self._log_signal_flow(
                            symbol,
                            "risk_check:start",
                            f"setup={idx}/{total_setups} {setup_details}",
                            level="DEBUG",
                        )

                        risk_result = safe_execute(
                            "RiskCheck",
                            self.risk_manager.validate_pre_trade_conditions,
                            setup,
                        )
                        if risk_result is None:
                            ok, msg = False, "Fehler im Risk-Check"
                        else:
                            ok, msg = risk_result

                        if not ok:
                            self._log_signal_flow(
                                symbol,
                                "risk_check:fail",
                                f"reason={msg} | setup={idx}/{total_setups} {setup_details}",
                                level="WARNING",
                            )
                            log_service.log_system(
                                f"[RiskCheckFail] {setup.symbol} → {msg}",
                                level="WARNING",
                            )
                            continue

                        self._log_signal_flow(
                            symbol,
                            "risk_check:pass",
                            f"code={msg} | setup={idx}/{total_setups} {setup_details}",
                            level="INFO",
                        )

                        # 2) Order asynchron absenden
                        def _place(s):
                            order_type = (
                                getattr(s, "order_type", None) or "market"
                            ).lower()
                            if order_type == "market":
                                return safe_execute(
                                    "MarketOrder",
                                    self.execution_engine.place_market_order,
                                    s,
                                )
                            return safe_execute(
                                "PendingOrder",
                                self.execution_engine.place_pending_order,
                                s,
                            )

                        # Idempotency-Key vorab deterministisch erzeugen (und im Setup ablegen),
                        # damit ExecutionEngine & Reconcile denselben Key sehen.
                        try:
                            idem_key = self.execution_engine.make_idempotency_key(setup)
                            setup.metadata["idempotency_key"] = idem_key
                        except Exception:
                            idem_key = None

                        order_fut = self._pool.submit(_place, setup)
                        self._log_signal_flow(
                            symbol,
                            "order_submit:attempt",
                            f"setup={idx}/{total_setups} {setup_details}, idem_key={idem_key or '-'}",
                        )
                        try:
                            ticket = order_fut.result(timeout=self._order_timeout_sec)
                        except TimeoutError:
                            self._log_signal_flow(
                                symbol,
                                "order_submit:timeout",
                                f"waited={self._order_timeout_sec}s idem_key={idem_key or '-'}",
                                level="WARNING",
                            )
                            # -> Late-Fill Watcher starten (nicht blockierend)
                            threading.Thread(
                                target=self._reconcile_late_fill,
                                args=(setup, idem_key, 60.0, 2.0),
                                daemon=True,
                            ).start()
                            continue
                        except Exception as e:
                            self._log_signal_flow(
                                symbol,
                                "order_submit:error",
                                f"{self._short_repr(e)} | setup={idx}/{total_setups} {setup_details}",
                                level="ERROR",
                            )
                            log_service.log_system(
                                f"[Order] ❌ Platzierungsfehler {setup.symbol}: {e}",
                                level="ERROR",
                            )
                            continue

                        if ticket:
                            try:
                                setup.metadata["ticket_id"] = ticket
                            except Exception:
                                pass

                            self._log_signal_flow(
                                symbol,
                                "order_submit:success",
                                f"ticket={ticket} idem_key={idem_key or '-'}",
                            )

                            def register_monitor(setup_obj):
                                try:
                                    manager = strategy_position_manager_factory(
                                        setup_obj, self.broker, self.data_provider
                                    )
                                    self.position_monitor_controller.add_manager(
                                        manager
                                    )
                                except Exception as e:
                                    log_service.log_system(
                                        f"[MonitorError] Fehler beim Registrieren für {setup_obj.symbol}: {e}",
                                        level="ERROR",
                                    )

                            threading.Thread(
                                target=register_monitor, args=(setup,), daemon=True
                            ).start()
                        else:
                            self._log_signal_flow(
                                symbol,
                                "order_submit:no_ticket",
                                f"idem_key={idem_key or '-'} result={self._short_repr(ticket)}",
                                level="WARNING",
                            )

            except TimeoutError:
                # globaler Sammel-Timeout: restliche Futures abbrechen
                log_service.log_system(
                    "[SignalGen] ⏱️ Globaler Timeout: nicht alle Futures terminiert",
                    level="WARNING",
                )
            finally:
                # Übrig gebliebene Futures canceln
                try:
                    for fut in list(remaining):
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                except Exception:
                    pass

            if self.controller and hasattr(self.controller, "last_heartbeat"):
                self.controller.last_heartbeat[self] = time.time()
        finally:
            self._cycle_lock.release()

    def _reconcile_late_fill(
        self, setup, idem_key: str | None, wait_sec: float = 60.0, poll_sec: float = 2.0
    ):
        """
        Sucht nach verspätet ausgeführten Orders/Positionen und registriert nachträglich den Manager.
        """
        from time import sleep, time

        t_end = time() + max(5.0, float(wait_sec))
        symbol = getattr(setup, "symbol", "")
        found_ticket = None

        # Helper: Position-Manager registrieren
        def _register_now(ticket_id: int | None):
            try:
                if ticket_id:
                    setup.metadata["ticket_id"] = ticket_id
            except Exception:
                pass
            try:
                manager = strategy_position_manager_factory(
                    setup, self.broker, self.data_provider
                )
                self.position_monitor_controller.add_manager(manager)
                log_service.log_system(
                    f"[Reconcile] Manager nachträglich registriert: {symbol}"
                )
            except Exception as e:
                log_service.log_system(
                    f"[Reconcile] Registrierungsfehler {symbol}: {e}", level="ERROR"
                )

        # Hauptschleife: Polling
        while time() < t_end and not found_ticket:
            try:
                # 1) Offene Positionen prüfen
                positions = (
                    self.broker.get_own_position(symbol, magic_number=self.magic_number)
                    or []
                )
                if positions:
                    # Nehme die „jüngste“ / erste passende
                    pos = positions[0]
                    found_ticket = getattr(pos, "ticket", None)
                    break
            except Exception:
                pass
            try:
                # 2) Pending Orders prüfen
                pendings = (
                    self.broker.get_own_pending_orders(
                        symbol, magic_number=self.magic_number
                    )
                    or []
                )
                if pendings:
                    ord = pendings[0]
                    found_ticket = getattr(ord, "ticket", None)
                    break
            except Exception:
                pass
            sleep(max(0.5, float(poll_sec)))

        if found_ticket:
            log_service.log_system(
                f"[Reconcile] Späten Fill erkannt: {symbol} (Ticket {found_ticket})"
            )
            _register_now(found_ticket)
        else:
            log_service.log_system(
                f"[Reconcile] Kein Late-Fill gefunden: {symbol}", level="DEBUG"
            )

    def shutdown(self):
        try:
            self.position_monitor_controller.stop_all()
        finally:
            self._pool.shutdown(wait=False, cancel_futures=True)
            log_service.log_system(
                f"[Runner] Shutdown für {self.strategy.name()} eingeleitet"
            )

    # --- Startup Reconciliation / Resume -------------------------------------
    def _resume_open_positions(self) -> None:
        """
        Sucht beim Start nach bereits offenen Positionen (pro Magic-Number) und
        registriert passende Position Manager im Step-Modus.

        Idempotent genug für wiederholte Aufrufe; nutzt ticket_id als Schlüssel.
        """
        # Brokerverbindung prüfen
        try:
            if not self._ensure_broker():
                return
        except Exception:
            return

        # Symbolliste der Strategie (Filter)
        try:
            configured_symbols = {
                s.strip().upper()
                for s in self._get_symbols()
                if isinstance(s, str) and s.strip()
            }
        except Exception:
            configured_symbols = set()

        # Offene Positionen vom Broker laden
        try:
            positions = (
                self.broker.get_all_own_positions(magic_number=self.magic_number) or []
            )
        except Exception as e:
            log_service.log_system(
                f"[Resume|{self.strategy.name()}] ❌ Laden offener Positionen fehlgeschlagen: {e}",
                level="ERROR",
            )
            return

        # Frühzeitiges Logging
        try:
            log_service.log_system(
                f"[Resume|{self.strategy.name()}] Starte Reconciliation – offene Positionen: {len(positions)} (Magic={self.magic_number})",
                level="INFO",
            )
        except Exception:
            pass

        if not positions:
            # Keine offenen Broker-Positionen – Downtime‑Reconcile trotzdem ausführen
            positions = []

        # Ableitungen für Manager/Setup
        from strategies._base.base_position_manager import (
            strategy_position_manager_factory,
        )
        from strategies._base.base_strategy import TradeSetup

        # Strategie‑Modulpfad für den Positionsmanager ermitteln
        try:
            mod = getattr(self.strategy.__class__, "__module__", "") or ""
            # Erwartet z. B. "strategies.mean_reversion_z_score.live.strategy"
            # -> wir brauchen "mean_reversion_z_score.live" für factory
            stripped = mod.removeprefix("strategies.")
            strategy_module = (
                stripped.rsplit(".", 1)[0] if "." in stripped else stripped
            )
        except Exception:
            strategy_module = ""

        # Timeframe / Max-Haltedauer aus Config (Fallbacks)
        tf = None
        try:
            tf = (self._tf or self.strategy.config.get("timeframe") or "").upper()
        except Exception:
            tf = None
        try:
            default_max_hold = int(
                self.strategy.config.get("max_holding_minutes", 0) or 0
            )
        except Exception:
            default_max_hold = 0

        # Mapping-/Helper
        def _as_float(val, default=0.0) -> float:
            try:
                return float(val)
            except Exception:
                return float(default)

        def _getattr_any(obj, *names, default=None):
            for n in names:
                try:
                    if hasattr(obj, n):
                        v = getattr(obj, n)
                        if v is not None:
                            return v
                except Exception:
                    pass
            return default

        def _resolve_max_hold(
            symbol: str | None, timeframe: str | None, direction: str | None
        ) -> int:
            """
            Versucht, max_holding_minutes analog zur Signal-Phase aus
            strategy.config['param_overrides'] aufzulösen.
            Fällt zurück auf default_max_hold, falls nichts gefunden/konfiguriert.
            """
            try:
                cfg = getattr(self.strategy, "config", {}) or {}
                po = cfg.get("param_overrides", {}) or {}
                if not po:
                    return default_max_hold

                sym = (symbol or "").upper()
                tf_norm = (timeframe or "").upper()
                dir_raw = (direction or "").lower()
                dir_map = {"long": "buy", "short": "sell", "buy": "buy", "sell": "sell"}
                dir_key = dir_map.get(dir_raw, dir_raw)

                for sym_key, tf_key in (
                    (sym, tf_norm),
                    (sym, "*"),
                    ("*", tf_norm),
                    ("*", "*"),
                ):
                    node = po.get(sym_key, {}).get(tf_key, {})
                    val = (node.get(dir_key, {}) or {}).get("max_holding_minutes", None)
                    if val is not None:
                        try:
                            return int(val or 0)
                        except Exception:
                            continue
            except Exception:
                pass
            return default_max_hold

        # Bereits registrierte Manager anhand Heartbeat-Map erkennen
        existing_ids = set()
        try:
            with self.position_monitor_controller._lock:
                existing_ids = set(self.position_monitor_controller._managers.keys())
        except Exception:
            existing_ids = set()

        resumed = 0
        tracker = None
        try:
            tracker = self.execution_tracker
        except Exception:
            tracker = None

        def _find_tracker_info_by_order(order_id: int):
            """Suche in den letzten Tagen nach Tracker‑Record mit passender order_id.
            Liefert (entry_time_dt, scenario_str) oder (None, None)."""
            if not tracker:
                return None, None
            from datetime import timedelta

            base = now_utc()
            for d in range(0, 5):
                day_dt = base - timedelta(days=d)
                try:
                    day_map = tracker.get_day_data(date=day_dt) or {}
                except Exception:
                    continue
                # Durchsuche alle Records dieses Tages
                for _key, rec in (day_map or {}).items():
                    try:
                        if int(rec.get("order_id")) != int(order_id):
                            continue
                    except Exception:
                        continue
                    # Treffer
                    raw_et = rec.get("entry_time")
                    scenario = rec.get("scenario")
                    try:
                        et_dt = self._parse_iso_to_utc(raw_et, base) if raw_et else None
                    except Exception:
                        et_dt = None
                    return et_dt, scenario
            return None, None

        # Zählwerte für Diagnose/Logs
        open_count = len(positions or [])
        skipped_symbol = 0
        already_active = 0
        errors_count = 0

        for pos in positions:
            try:
                # Broker‑Symbol lesen und auf logisches Symbol abbilden
                symbol_b = _getattr_any(pos, "symbol", default=None)
                if not symbol_b:
                    continue
                try:
                    symbol_logical = (
                        self.symbol_mapper.to_logical_from_broker(symbol_b)
                        if self.symbol_mapper
                        else symbol_b
                    )
                except Exception:
                    symbol_logical = symbol_b
                if (
                    configured_symbols
                    and symbol_logical.strip().upper() not in configured_symbols
                ):
                    # Nicht von dieser Strategie verwaltetes Symbol überspringen
                    skipped_symbol += 1
                    continue
                # Ab hier ausschließlich mit logischem Symbol weiterarbeiten
                symbol = symbol_logical

                ticket = _getattr_any(pos, "position_id", "ticket", default=None)
                if ticket is None:
                    # Ohne Ticket keine eindeutige Manager-ID
                    continue
                try:
                    ticket = int(ticket)
                except Exception:
                    continue
                if ticket in existing_ids:
                    # Bereits registriert
                    already_active += 1
                    continue

                # Richtung/Levels robust ermitteln
                direction_obj = _getattr_any(pos, "direction", default=None)
                direction = None
                try:
                    # Enum oder String
                    direction = (
                        direction_obj.value
                        if hasattr(direction_obj, "value")
                        else str(direction_obj).lower()
                    )
                except Exception:
                    direction = None
                if direction not in ("buy", "sell"):
                    # Fallback über Broker-Helfer (falls vorhanden)
                    try:
                        direction = str(self.broker.position_direction(pos)).lower()
                    except Exception:
                        direction = None

                entry = _as_float(_getattr_any(pos, "entry", "price_open", default=0.0))
                sl = _as_float(_getattr_any(pos, "sl", "stop_loss", default=0.0))
                tp = _as_float(_getattr_any(pos, "tp", "take_profit", default=0.0))

                # Tracker‑Info (Entry‑Zeit & Szenario) ermitteln
                entry_time_dt, scenario = _find_tracker_info_by_order(ticket)

                # Szenario‑Fallback für bekannte Strategie
                if not scenario and (strategy_module or "").startswith(
                    "mean_reversion_z_score.live"
                ):
                    if direction in ("buy", "long"):
                        scenario = "szenario_3_long"
                    elif direction in ("sell", "short"):
                        scenario = "szenario_3_short"

                # Setup‑Suffix möglichst kompatibel rekonstruieren, damit Tracker‑Close den Open findet
                setup_suffix = f"{self.strategy.name()}|MN{self.magic_number}|{tf or '-'}|{direction or '-'}|{scenario or 'resume'}"

                try:
                    risk_cfg = self.strategy.config.get("risk", {}) or {}
                    start_capital = float(risk_cfg.get("start_capital", 0.0) or 0.0)
                    risk_pct = float(risk_cfg.get("risk_per_trade_pct", 0.0) or 0.0)
                except Exception:
                    start_capital, risk_pct = 0.0, 0.0

                # Max-Haltedauer analog zur Signal-Phase aus param_overrides ableiten (falls vorhanden)
                resolved_max_hold = _resolve_max_hold(symbol, tf, direction)

                setup = TradeSetup(
                    symbol=symbol,
                    direction=direction or "buy",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    strategy=self.strategy.name(),
                    strategy_module=strategy_module or "",
                    start_capital=start_capital,
                    risk_pct=risk_pct,
                    order_type="market",
                    session_times=self.strategy.config.get("session", {}),
                    magic_number=self.magic_number,
                    entry_time=entry_time_dt,
                    metadata={
                        "ticket_id": ticket,
                        "setup": setup_suffix,
                        "timeframe": tf,
                        "max_holding_minutes": resolved_max_hold,
                        "resume": True,
                        # Zusatz für Manager, falls TradeSetup.entry_time nicht genutzt werden kann
                        "entry_time": (
                            entry_time_dt.isoformat().replace("+00:00", "Z")
                            if entry_time_dt
                            else (
                                _getattr_any(pos, "open_time", default=None)
                                .isoformat()
                                .replace("+00:00", "Z")
                                if _getattr_any(pos, "open_time", default=None)
                                is not None
                                else None
                            )
                        ),
                        "scenario": scenario,
                    },
                )

                # Manager instanziieren/registrieren
                try:
                    manager = strategy_position_manager_factory(
                        setup, self.broker, self.data_provider
                    )
                    self.position_monitor_controller.add_manager(manager)
                    resumed += 1
                    try:
                        et_dbg = (
                            entry_time_dt.replace(microsecond=0).isoformat()
                            if entry_time_dt
                            else "-"
                        )
                    except Exception:
                        et_dbg = "-"
                    log_service.log_system(
                        f"[Resume|{self.strategy.name()}] ▶️ Manager registriert: {symbol} ticket={ticket} dir={direction} sl={sl} tp={tp} entry_time={et_dbg} scenario={scenario or '-'} (broker_sym={symbol_b})",
                        level="INFO",
                    )
                except Exception as e:
                    log_service.log_system(
                        f"[Resume|{self.strategy.name()}] ❌ Manager-Registrierung fehlgeschlagen für {symbol}/{ticket}: {e}",
                        level="ERROR",
                    )
            except Exception as e:
                log_service.log_system(
                    f"[Resume|{self.strategy.name()}] ⚠️ Überspringe Position wegen Fehler: {e}",
                    level="WARNING",
                )
                errors_count += 1

        if resumed:
            log_service.log_system(
                f"[Resume|{self.strategy.name()}] ✅ {resumed} Position-Manager wieder aufgenommen (open={open_count}, already_active={already_active}, skipped_symbol={skipped_symbol}, errors={errors_count})",
                level="INFO",
            )
        else:
            log_service.log_system(
                f"[Resume|{self.strategy.name()}] ℹ️ Keine neuen Manager (open={open_count}, already_active={already_active}, skipped_symbol={skipped_symbol}, errors={errors_count})",
                level="INFO",
            )

        # --- Downtime‑Schließungen finalisieren ---------------------------------
        # Finde Tracker‑Einträge, die noch 'open' sind, deren Tickets aber nicht mehr offen sind.
        try:
            from datetime import timedelta

            from hf_engine.adapter.broker.broker_utils import get_pip_size

            # Menge aktuell offener Tickets (vom Broker) bilden
            open_tickets: set[int] = set()
            try:
                for p in positions or []:
                    tid = _getattr_any(p, "position_id", "ticket", default=None)
                    if tid is None:
                        continue
                    try:
                        open_tickets.add(int(tid))
                    except Exception:
                        pass
            except Exception:
                open_tickets = set()

            # Ohne Tracker keine Aktion
            if not tracker:
                return

            base_now = now_utc()
            closed_count = 0

            for d in range(0, 5):  # bis zu 5 Tage rückwärts prüfen
                day_dt = base_now - timedelta(days=d)
                try:
                    day_map = tracker.get_day_data(date=day_dt) or {}
                except Exception:
                    continue

                for full_key, rec in (day_map or {}).items():
                    try:
                        status = str(rec.get("status") or "").lower().strip()
                        if status != "open":
                            continue
                        raw_oid = rec.get("order_id")
                        if raw_oid is None:
                            continue
                        try:
                            oid = int(raw_oid)
                        except Exception:
                            continue
                        # Wenn Ticket noch offen ist → nichts tun
                        if oid in open_tickets:
                            continue

                        # Zusätzlicher Broker‑Check: existiert Position noch?
                        try:
                            still_pos = self.broker.get_position_by_ticket(oid)
                        except Exception:
                            still_pos = None
                        if still_pos:
                            # Broker kennt die Position; konservativ überspringen
                            continue

                        # Key zerlegen: "SYMBOL::suffix"
                        if "::" in full_key:
                            symbol, suffix = full_key.split("::", 1)
                        else:
                            symbol, suffix = full_key, None

                        # Rekonstruktion versuchen (für vollständigen Trade‑Log)
                        trade = None
                        try:
                            pip = get_pip_size(symbol)
                            trade = self.broker.reconstruct_trade_from_deal_ticket(
                                int(oid), float(pip or 0.0001)
                            )
                        except Exception:
                            trade = None

                        # Tracker schließen
                        exit_time = None
                        try:
                            if trade and trade.get("exit_time"):
                                exit_time = trade.get("exit_time")
                        except Exception:
                            exit_time = None
                        try:
                            tracker.mark_trade_closed(
                                symbol,
                                suffix,
                                exit_price=(trade.get("exit_price") if trade else None),
                                direction=(
                                    rec.get("direction")
                                    if isinstance(rec, dict)
                                    else None
                                ),
                                exit_time=exit_time or base_now,
                            )
                        except Exception as e:
                            log_service.log_system(
                                f"[Resume|{self.strategy.name()}] ⚠️ Tracker‑Close fehlgeschlagen für {symbol}/{oid}: {e}",
                                level="WARNING",
                            )

                        # Trade‑Log schreiben, falls verfügbar
                        if trade:
                            try:
                                # Minimale Ergänzungen für CSV/DB
                                trade["symbol"] = trade.get("symbol") or symbol
                                trade["strategy"] = self.strategy.name()
                                log_service.log_trade(trade)
                            except Exception:
                                pass

                        closed_count += 1
                        log_service.log_system(
                            f"[Resume|{self.strategy.name()}] 🧹 Finalisiere Downtime‑Close: symbol={symbol} ticket={oid}",
                            level="INFO",
                        )
                    except Exception as e:
                        # Schutz: Einzelner defekter Record darf den gesamten Durchlauf nicht stoppen
                        try:
                            fk = str(full_key)
                        except Exception:
                            fk = "?"
                        log_service.log_system(
                            f"[Resume|{self.strategy.name()}] ⚠️ Fehler beim Reconcile eines Records ({fk}): {e}",
                            level="WARNING",
                        )
            if closed_count:
                log_service.log_system(
                    f"[Resume|{self.strategy.name()}] ✅ {closed_count} im Downtime geschlossene Positionen finalisiert",
                    level="INFO",
                )
        except Exception as e:
            log_service.log_system(
                f"[Resume|{self.strategy.name()}] ⚠️ Downtime‑Reconcile Fehler: {e}",
                level="WARNING",
            )
