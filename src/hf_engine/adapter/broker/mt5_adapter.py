# mt5_adapter.py
"""
MT5Adapter implementiert das BrokerInterface für MetaTrader5.

Diese Klasse abstrahiert alle Interaktionen mit der MetaTrader5-API und ermöglicht es, den Trading-Code
unabhängig vom konkreten Broker zu entwickeln. Sie eignet sich sowohl für Live-Trading als auch für Simulationszwecke,
sofern MetaTrader5-Datenquellen verwendet werden.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

from hf_engine.adapter.broker.broker_interface import BrokerInterface, OrderResult
from hf_engine.core.execution.execution_tracker import ExecutionTracker
from hf_engine.infra.config.environment import TIMEZONE
from hf_engine.infra.config.symbol_mapper import SymbolMapper
from hf_engine.infra.config.time_utils import from_utc_to_broker, now_utc, to_utc
from hf_engine.infra.logging.log_service import log_service

_DEAL_HISTORY_LOOKBACK_DAYS = 5


class MT5Adapter(BrokerInterface):
    """
    Adapter für die MT5-Python-API mit defensiven Checks, einheitlicher Symbol-Mapping-Schicht
    und minimaler Parametrisierung für Order-Defaults.
    """

    def __init__(
        self,
        account_id: int,
        password: str,
        server: str,
        terminal_path: str,
        magic_number: int = 0,
        data_path: Optional[str] = None,
        symbol_mapper: Optional[SymbolMapper] = None,
        # Parametrisierbare, aber konservative Defaults
        default_deviation: int = 10,
        default_filling_market: int = mt5.ORDER_FILLING_IOC,
        default_filling_pending: int = mt5.ORDER_FILLING_RETURN,
    ) -> None:
        self.account_id: int = int(account_id)
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        self.magic_number: int = int(magic_number)
        self.tracker = ExecutionTracker()
        self.symbol_mapper = symbol_mapper or SymbolMapper({})

        self.default_deviation = default_deviation
        self.default_filling_market = default_filling_market
        self.default_filling_pending = default_filling_pending

        # Individueller Data-Pfad je Account (für saubere Trennung)
        self.data_path = os.path.abspath(
            data_path or f"./mt5_data/account_{account_id}"
        )
        os.makedirs(self.data_path, exist_ok=True)

        # Logging
        log_service.log_system(f"[MT5Adapter] 🚀 Starte MT5 für Account {account_id}")
        log_service.log_system(f"[MT5Adapter] Terminalpfad: {terminal_path}")
        log_service.log_system(f"[MT5Adapter] Datenpfad: {self.data_path}")

        # Initialisierung
        self._connect_mt5()

    # -----------------------
    # Internals & Utilities
    # -----------------------

    def _connect_mt5(self) -> None:
        connected = mt5.initialize(
            path=self.terminal_path,
            login=self.account_id,
            password=self.password,
            server=self.server,
            portable=True,
            data_path=self.data_path,
        )
        if not connected:
            code, msg = mt5.last_error()
            log_service.log_system(f"[MT5Adapter] ❌ Verbindungsfehler: {code} – {msg}")
            raise RuntimeError(f"MT5 konnte nicht initialisiert werden: {code} – {msg}")

        log_service.log_system(
            f"[MT5Adapter] ✅ Erfolgreich verbunden mit Account {self.account_id}"
        )

        info = mt5.account_info()
        if not info or info.login != self.account_id:
            raise RuntimeError(
                f"MT5Adapter: Account-Zuordnung inkonsistent (erwartet: {self.account_id}, "
                f"erhalten: {info.login if info else 'None'})"
            )

    @staticmethod
    def _mt5_failed(result: Any) -> bool:
        return (result is None) or (
            getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE
        )

    # --- Kommentar-Schutz ---------------------------------------------------
    @staticmethod
    def _sanitize_comment(comment: Optional[str]) -> str:
        """
        MT5-Beschränkungen:
          - max. 31 Zeichen
          - nur 7-bit ASCII (keine Emojis/Umlaut)
        """
        if not comment:
            return ""
        # Whitespace komprimieren
        c = " ".join(str(comment).split())
        # ASCII-only (ersetzt Nicht-ASCII durch '?')
        c = c.encode("ascii", "replace").decode("ascii")
        # Hart kürzen auf 31 Zeichen
        return c[:31]

    def _ensure_symbol_selected(self, symbol_b: str) -> None:
        if not mt5.symbol_select(symbol_b, True):
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"Symbol konnte nicht selektiert werden: {symbol_b} ({code} – {msg})"
            )

    def _round_price(self, symbol_b: str, price: float) -> float:
        info = mt5.symbol_info(symbol_b)
        if not info:
            return round(price, 5)
        ts = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        if ts > 0:
            steps = round(price / ts)
            return round(steps * ts, int(info.digits or 5))
        return round(price, int(info.digits or 5))

    def _get_tick(self, symbol_b: str):
        self._ensure_symbol_selected(symbol_b)
        tick = mt5.symbol_info_tick(symbol_b)
        if not tick:
            code, msg = mt5.last_error()
            log_service.log_system(
                f"[MT5Adapter] ⚠️ Tick-Daten nicht verfügbar für {symbol_b} ({code} – {msg})",
                level="WARNING",
            )
            raise RuntimeError(
                f"Tick-Daten nicht verfügbar für {symbol_b} ({code} – {msg})"
            )
        return tick

    def _validate_account(self) -> None:
        info = mt5.account_info()
        if not info or info.login != self.account_id:
            raise RuntimeError(
                f"[MT5Adapter] ❌ Account-Verifizierung fehlgeschlagen: "
                f"erwartet={self.account_id}, erhalten={info.login if info else 'None'}"
            )

    # -----------------------
    # Lifecycle
    # -----------------------

    def shutdown(self) -> None:
        log_service.log_system(
            f"[MT5Adapter] 📴 Shutdown für Account {self.account_id}"
        )
        try:
            mt5.shutdown()
        except Exception as e:
            log_service.log_system(f"[MT5Adapter] ⚠️ Fehler beim MT5 Shutdown: {e}")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception as e:
            log_service.log_system(f"[MT5Adapter] ⚠️ Fehler im __del__: {e}")

    def ensure_connection(self) -> bool:
        try:
            account = mt5.account_info()
            if account is None or account.login != self.account_id:
                log_service.log_system(
                    f"[MT5Adapter] ❌ MT5 nicht korrekt verbunden mit Account {self.account_id}",
                    level="ERROR",
                )
                return False
            return True
        except Exception:
            return False

    def set_magic_number(self, new_magic: int) -> None:
        new_magic = int(new_magic)
        log_service.log_system(
            f"[MT5Adapter] 🔁 Setze neuen Magic Number: {new_magic} für Account {self.account_id}"
        )
        self.magic_number = new_magic

    # -----------------------
    # Account & Symbol
    # -----------------------

    def get_account_equity(self) -> float:
        info = mt5.account_info()
        if not info:
            raise RuntimeError("[MT5Adapter] ❌ AccountInfo nicht verfügbar")
        return float(info.equity)

    def get_account_currency(self) -> str:
        self._validate_account()
        info = mt5.account_info()
        return info.currency if info else "USD"

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)
        info = mt5.symbol_info(symbol_b)
        if not info:
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"Symbolinfo nicht verfügbar für {symbol_b} ({code} – {msg})"
            )
        return {
            "digits": info.digits,
            "contract_size": float(info.trade_contract_size),
            "tick_size": float(getattr(info, "trade_tick_size", 0.0) or 0.0),
            "tick_value": float(getattr(info, "trade_tick_value", 0.0) or 0.0),
            "currency_profit": getattr(info, "currency_profit", None),
            "volume_min": float(getattr(info, "volume_min", 0.01) or 0.01),
            "volume_step": float(getattr(info, "volume_step", 0.01) or 0.01),
            "volume_max": float(getattr(info, "volume_max", 0.0) or 0.0),
        }

    def get_symbol_tick(self, symbol: str) -> Dict[str, float]:
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)
        tick = self._get_tick(symbol_b)
        return {"bid": tick.bid, "ask": tick.ask, "last": tick.last}

    def get_symbol_spread(self, symbol: str) -> float:
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)
        tick = self._get_tick(symbol_b)
        info = mt5.symbol_info(symbol_b)
        digits = int(info.digits) if info and info.digits is not None else 5
        return round((tick.ask or 0.0) - (tick.bid or 0.0), digits)

    def get_min_lot_size(self, symbol: str) -> float:
        symbol_b = self.symbol_mapper.to_broker(symbol)
        info = mt5.symbol_info(symbol_b)
        if info is None:
            raise ValueError(
                f"[MT5] ❌ SymbolInfo konnte nicht abgerufen werden für: {symbol_b}"
            )
        return float(info.volume_min)

    def get_valid_volume(self, symbol: str, volume: float) -> float:
        symbol_b = self.symbol_mapper.to_broker(symbol)
        log_service.log_system(
            f"[MT5Adapter] ℹ️ Hole Volumen: Symbol={symbol_b}, Ursprungs-Volume={volume}"
        )
        info = mt5.symbol_info(symbol_b)
        if not info:
            raise RuntimeError(f"⚠️ Symbol-Info nicht verfügbar für {symbol_b}")

        step = info.volume_step
        min_vol = info.volume_min
        if step is None or min_vol is None:
            raise RuntimeError(
                f"⚠️ Ungültige volume_step ({step}) oder volume_min ({min_vol}) für {symbol_b}"
            )

        # konservativ runden
        try:
            rounded = max(min_vol, round(volume / step) * step)
            return float(round(rounded, 2))
        except Exception as e:
            raise RuntimeError(
                f"❌ Fehler beim Runden des Volumens für {symbol_b}: {e}"
            )

    # -----------------------
    # Preise & Risiko
    # -----------------------

    def get_symbol_price(self, symbol: str, direction: str) -> float:
        """
        Liefert den aktuellen Preis abhängig von der Richtung ('buy' -> ask, 'sell' -> bid).
        """
        symbol_b = self.symbol_mapper.to_broker(symbol)
        tick = self._get_tick(symbol_b)
        direction = (direction or "").lower()
        price = tick.ask if direction == "buy" else tick.bid
        if not price:
            raise RuntimeError(f"❌ Kein Preis verfügbar für {symbol_b}")
        return float(price)

    def calculate_risk_amount(
        self, symbol: str, entry_price: float, sl_price: Optional[float], volume: float
    ) -> Optional[float]:
        """
        Schätzt den monetären Risiko-Betrag je Trade basierend auf Pip-Entfernung und Broker-Metadaten.
        Keine Validierung von SL/TP-Logik – diese ist ausgelagert.
        """
        if sl_price is None:
            return None

        symbol_b = self.symbol_mapper.to_broker(symbol)
        info = mt5.symbol_info(symbol_b)
        if info is None:
            raise ValueError(f"Symbol not found: {symbol_b}")

        # Pip-Größe bestimmen (generisch): 0.01 bei 3/1 Nachkommastellen (JPY), sonst 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        pip_size = 0.01 if digits in (1, 3) else 0.0001

        # Pip-Value über Tick-Relation ableiten, Fallback ~10 pro Lot
        try:
            tick_value = float(info.trade_tick_value or 0.0)
            tick_size = float(info.trade_tick_size or 0.0)
            if tick_value > 0.0 and tick_size > 0.0:
                pip_value_per_lot = tick_value * (pip_size / tick_size)
            else:
                pip_value_per_lot = 10.0
        except Exception:
            pip_value_per_lot = 10.0

        pip_distance = abs(entry_price - sl_price) / pip_size
        risk_amount = float(pip_distance) * float(volume) * float(pip_value_per_lot)
        return risk_amount

    # -----------------------
    # Orders
    # -----------------------

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        volume: float = 1.0,
        comment: str = "Market Order",
        magic_number: Optional[int] = None,
        scenario: Optional[str] = "scenario",
    ) -> OrderResult:
        """
        Sendet eine Market-Order. Keine SL/TP-Logikvalidierung hier – die erfolgt upstream.
        """
        self._validate_account()

        symbol_b = self.symbol_mapper.to_broker(symbol)
        self._ensure_symbol_selected(symbol_b)

        order_type = (
            mt5.ORDER_TYPE_BUY
            if (direction or "").lower() == "buy"
            else mt5.ORDER_TYPE_SELL
        )
        price = self.get_symbol_price(symbol_b, direction)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol_b,
            "volume": float(volume),
            "type": order_type,
            "price": self._round_price(symbol_b, price),
            "sl": self._round_price(symbol_b, sl) if sl is not None else 0.0,
            "tp": self._round_price(symbol_b, tp) if tp is not None else 0.0,
            "deviation": self.default_deviation,
            "magic": (
                int(magic_number) if magic_number is not None else self.magic_number
            ),
            "comment": self._sanitize_comment(comment),
            "type_filling": self.default_filling_market,
        }

        result = mt5.order_send(request)
        if self._mt5_failed(result):
            err_msg = getattr(result, "comment", None) if result else None
            err_code = getattr(result, "retcode", None) if result else None
            if not err_msg:
                code, msg = mt5.last_error()
                err_msg, err_code = msg, code
            return OrderResult(False, f"{err_msg} (Code {err_code})")

        return OrderResult(
            True,
            f"✅ Market Order ausgeführt: {symbol_b}",
            order=getattr(result, "order", None),
        )

    def place_pending_order(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        comment: str = "",
        order_type: str = "stop",
        magic_number: Optional[int] = None,
        scenario: Optional[str] = "scenario",
    ) -> OrderResult:
        self._validate_account()

        symbol_b = self.symbol_mapper.to_broker(symbol)
        self._ensure_symbol_selected(symbol_b)

        if (order_type or "").lower() == "market":
            return OrderResult(
                False, "❌ Market Orders bitte über 'place_market_order()' senden."
            )

        if (order_type or "").lower() == "stop":
            mt5_order_type = (
                mt5.ORDER_TYPE_BUY_STOP
                if (direction or "").lower() == "buy"
                else mt5.ORDER_TYPE_SELL_STOP
            )
        elif (order_type or "").lower() == "limit":
            mt5_order_type = (
                mt5.ORDER_TYPE_BUY_LIMIT
                if (direction or "").lower() == "buy"
                else mt5.ORDER_TYPE_SELL_LIMIT
            )
        else:
            return OrderResult(False, f"❌ Unbekannter Order-Typ: {order_type}")

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol_b,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": self._round_price(symbol_b, entry),
            "sl": self._round_price(symbol_b, sl),
            "tp": self._round_price(symbol_b, tp),
            "deviation": self.default_deviation,
            "magic": (
                int(magic_number) if magic_number is not None else self.magic_number
            ),
            "comment": self._sanitize_comment(comment),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.default_filling_pending,
        }

        result = mt5.order_send(request)
        if self._mt5_failed(result):
            err_msg = getattr(result, "comment", None) or "MT5 Fehler"
            err_code = getattr(result, "retcode", None)
            log_service.log_system(
                f"[MT5Adapter] ❌ Pending Order fehlgeschlagen: {err_msg} (Code {err_code})"
            )
            return OrderResult(False, f"{err_msg} (Code {err_code})")

        return OrderResult(
            True,
            f"✅ Order ({order_type.upper()}) platziert: {symbol_b}",
            order=getattr(result, "order", None),
        )

    def cancel_all_pending_orders(self, symbol: str) -> bool:
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)

        orders = mt5.orders_get(symbol=symbol_b) or []
        success = True

        for order in orders:
            if order.magic != self.magic_number:
                continue
            if order.type not in {
                mt5.ORDER_TYPE_BUY_LIMIT,
                mt5.ORDER_TYPE_SELL_LIMIT,
                mt5.ORDER_TYPE_BUY_STOP,
                mt5.ORDER_TYPE_SELL_STOP,
                mt5.ORDER_TYPE_BUY_STOP_LIMIT,
                mt5.ORDER_TYPE_SELL_STOP_LIMIT,
            }:
                continue

            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "symbol": order.symbol,
                "magic": order.magic,
                "comment": "AutoCancel",
            }
            result = mt5.order_send(request)
            if self._mt5_failed(result):
                log_service.log_system(
                    f"[MT5] ❌ Löschen fehlgeschlagen für Order {order.ticket}: "
                    f"{getattr(result, 'comment', '')} ({getattr(result, 'retcode', '')})"
                )
                success = False
            else:
                log_service.log_system(
                    f"[MT5] 🗑️ Pending Order {order.ticket} gelöscht ({symbol_b})"
                )

        return success

    def cancel_pending_order_by_ticket(self, ticket_id: int, magic_number: int) -> bool:
        self._validate_account()
        order = self.get_pending_order_by_ticket(ticket_id, magic_number)
        if not order:
            log_service.log_system(
                f"[MT5] ⚠️ Keine Pending-Order mit Ticket {ticket_id} und Magic {magic_number} gefunden"
            )
            return False

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order.ticket,
            "symbol": order.symbol,
            "magic": order.magic,
            "comment": "AutoCancel",
        }
        result = mt5.order_send(request)
        if self._mt5_failed(result):
            log_service.log_system(
                f"[MT5] ❌ Löschen fehlgeschlagen für Order {order.ticket}: "
                f"{getattr(result, 'comment', '')} ({getattr(result, 'retcode', '')})"
            )
            return False

        log_service.log_system(
            f"[MT5] 🗑️ Pending Order {order.ticket} gelöscht ({order.symbol})"
        )
        return True

    def modify_tp(self, position_id: int, new_tp: float) -> OrderResult:
        self._validate_account()
        position = mt5.positions_get(ticket=position_id)
        if not position:
            log_service.log_system(
                f"[MT5Adapter] ❌ Position {position_id} nicht gefunden für TP-Anpassung."
            )
            return OrderResult(False, f"Position {position_id} nicht gefunden")

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": pos.sl,
            "tp": self._round_price(pos.symbol, new_tp),
            "magic": self.magic_number,
            "comment": "Auto TP Update",
        }
        result = mt5.order_send(request)
        if self._mt5_failed(result):
            return OrderResult(
                False, f"TP-Update fehlgeschlagen: {getattr(result, 'comment', '')}"
            )
        return OrderResult(
            True, f"✅ TP aktualisiert auf {self._round_price(pos.symbol, new_tp)}"
        )

    def modify_sl(self, position_id: int, new_sl: float) -> OrderResult:
        self._validate_account()
        position = mt5.positions_get(ticket=position_id)
        if not position:
            log_service.log_system(
                f"[MT5Adapter] ❌ Position {position_id} nicht gefunden für SL-Anpassung."
            )
            return OrderResult(False, "Position nicht gefunden")

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": self._round_price(pos.symbol, new_sl),
            "tp": pos.tp,
            "magic": self.magic_number,
            "comment": "Auto SL Update",
        }
        result = mt5.order_send(request)
        if self._mt5_failed(result):
            return OrderResult(
                False, f"SL-Update fehlgeschlagen: {getattr(result, 'comment', '')}"
            )
        return OrderResult(
            True, f"✅ SL angepasst auf {self._round_price(pos.symbol, new_sl)}"
        )

    def close_position_full(self, position_id: int) -> OrderResult:
        self._validate_account()
        position = mt5.positions_get(ticket=position_id)
        if not position:
            return OrderResult(False, f"Keine Position mit ID {position_id} gefunden.")

        pos = position[0]
        symbol_b = pos.symbol
        direction = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
        tick = self._get_tick(symbol_b)
        price = tick.bid if direction == "sell" else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if direction == "sell" else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol_b,
            "volume": float(pos.volume),
            "type": close_type,
            "price": price,
            "deviation": self.default_deviation,
            "magic": int(pos.magic),
            "comment": "Position Close Full",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.default_filling_market,
            "position": pos.ticket,
        }
        result = mt5.order_send(request)
        if not self._mt5_failed(result):
            return OrderResult(True, f"Position {position_id} vollständig geschlossen.")
        else:
            return OrderResult(
                False,
                f"Fehler beim Schließen: {getattr(result, 'comment', '')} ({getattr(result, 'retcode', '')})",
            )

    def close_position_partial(self, position_id: int, volume: float) -> OrderResult:
        self._validate_account()
        position = mt5.positions_get(ticket=position_id)
        if not position:
            return OrderResult(False, f"Keine Position mit ID {position_id} gefunden.")

        pos = position[0]
        symbol_b = pos.symbol
        direction = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
        tick = self._get_tick(symbol_b)
        price = tick.bid if direction == "sell" else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if direction == "sell" else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol_b,
            "volume": float(volume),
            "type": close_type,
            "price": price,
            "deviation": self.default_deviation,
            "magic": int(pos.magic),
            "comment": "Position Close Partial",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.default_filling_market,
            "position": pos.ticket,
        }
        result = mt5.order_send(request)
        if not self._mt5_failed(result):
            return OrderResult(
                True, f"{volume} Lots von Position {position_id} geschlossen."
            )
        else:
            return OrderResult(
                False,
                f"Fehler beim Teil-Schließen: {getattr(result, 'comment', '')} ({getattr(result, 'retcode', '')})",
            )

    # -----------------------
    # Positions & Orders Query
    # -----------------------

    def position_direction(self, pos) -> str:
        return "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell"

    def pending_order_direction(self, order) -> str:
        """
        Richtung einer Pending Order ('buy' oder 'sell').
        """
        buy_types = {
            mt5.ORDER_TYPE_BUY_LIMIT,
            mt5.ORDER_TYPE_BUY_STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT,
        }
        sell_types = {
            mt5.ORDER_TYPE_SELL_LIMIT,
            mt5.ORDER_TYPE_SELL_STOP,
            mt5.ORDER_TYPE_SELL_STOP_LIMIT,
        }
        if order.type in buy_types:
            return "buy"
        if order.type in sell_types:
            return "sell"
        raise ValueError(f"Unbekannter Pending-Order-Typ: {order.type}")

    def get_own_position(self, symbol: str, magic_number: Optional[int] = None) -> List:
        """
        Liefert eigene Positionen zu Symbol & Magic.
        """
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)
        try:
            all_positions = mt5.positions_get()
        except Exception as e:
            log_service.log_exception(f"[MT5Adapter] ❌ Fehler bei positions_get()", e)
            return []

        if not all_positions:
            # log_service.log_system(f"[MT5Adapter] ⚠️ MT5 gab None/leer zurück – keine Positionen vorhanden")
            return []

        magic = int(magic_number) if magic_number is not None else self.magic_number
        return [
            pos
            for pos in all_positions
            if pos.symbol == symbol_b and pos.magic == magic
        ]

    def get_position_by_ticket(self, ticket: Optional[int]) -> Optional[object]:
        """
        Gibt eine einzelne Position mit dem exakten Ticket zurück – oder None.
        """
        if ticket is None:
            return None

        self._validate_account()
        try:
            all_positions = mt5.positions_get()
        except Exception as e:
            log_service.log_exception("[MT5Adapter] ❌ Fehler bei positions_get()", e)
            return None

        if not all_positions:
            # log_service.log_system("[MT5Adapter] ⚠️ MT5 gab None/leer zurück – keine Positionen vorhanden")
            return None

        for pos in all_positions:
            if pos.ticket == ticket:
                return pos
        return None

    def get_all_own_positions(self, magic_number: Optional[int] = None) -> List:
        self._validate_account()
        positions = mt5.positions_get()
        if not positions:
            return []
        magic = int(magic_number) if magic_number is not None else self.magic_number
        return [p for p in positions if p.magic == magic]

    def get_own_pending_orders(self, symbol: str, magic_number: Optional[int] = None):
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)
        orders = mt5.orders_get(symbol=symbol_b) or []
        magic = int(magic_number) if magic_number is not None else self.magic_number
        return [
            o
            for o in orders
            if o.magic == magic
            and o.type
            in {
                mt5.ORDER_TYPE_BUY_LIMIT,
                mt5.ORDER_TYPE_SELL_LIMIT,
                mt5.ORDER_TYPE_BUY_STOP,
                mt5.ORDER_TYPE_SELL_STOP,
                mt5.ORDER_TYPE_BUY_STOP_LIMIT,
                mt5.ORDER_TYPE_SELL_STOP_LIMIT,
            }
        ]

    def get_pending_order_by_ticket(self, ticket_id: int, magic_number: int):
        """
        Gibt die eigene Pending Order mit der angegebenen Ticket-ID zurück,
        sofern sie existiert und vom richtigen Typ + Magic Number ist.
        """
        self._validate_account()
        order_list = mt5.orders_get(ticket=ticket_id)
        if not order_list:
            return None

        order = order_list[0]
        if order.magic != int(magic_number):
            return None

        if order.type not in {
            mt5.ORDER_TYPE_BUY_LIMIT,
            mt5.ORDER_TYPE_SELL_LIMIT,
            mt5.ORDER_TYPE_BUY_STOP,
            mt5.ORDER_TYPE_SELL_STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT,
            mt5.ORDER_TYPE_SELL_STOP_LIMIT,
        }:
            return None
        return order

    # -----------------------
    # Performance-Kennzahlen
    # -----------------------

    def get_current_r_multiple(
        self, symbol: str, initial_sl: float, ticket_id: Optional[int] = None
    ) -> Optional[float]:
        """
        Aktuelles R = (aktueller Buchgewinn) / (initiales Risiko).
        Falls ticket_id None ist, wird die erste eigene Position zum Symbol verwendet (falls vorhanden).
        """
        self._validate_account()
        symbol_b = self.symbol_mapper.to_broker(symbol)

        pos = self.get_position_by_ticket(ticket_id)
        if pos is None:
            own_positions = self.get_own_position(symbol_b)
            pos = own_positions[0] if own_positions else None

        if not pos:
            raise RuntimeError(f"Keine offene Position für {symbol_b} gefunden.")

        direction = self.position_direction(pos)
        price_current = self.get_symbol_price(symbol_b, direction)

        sl = abs(pos.price_open - initial_sl)
        if sl == 0:
            return None

        if direction == "buy":
            price_move = price_current - pos.price_open
        else:
            price_move = pos.price_open - price_current

        current_r = price_move / sl
        return round(float(current_r), 2)

    def get_closed_trade_profit(self, ticket_id: int) -> float:
        self._validate_account()
        deals = mt5.history_deals_get(ticket=ticket_id)
        if deals is None:
            return 0.0
        return float(sum(float(deal.profit) for deal in deals))

    def get_realized_r_multiple(self, magic_number: int) -> float:
        self._validate_account()

        # 1) Tag in UTC definieren (Systemstandard)
        now = now_utc()
        from_time_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
        to_time_utc = now.replace(hour=23, minute=59, second=59, microsecond=0)

        # 2) Für MT5 → Broker-TZ umrechnen
        from_time_b = from_utc_to_broker(from_time_utc)
        to_time_b = from_utc_to_broker(to_time_utc)

        deals = mt5.history_deals_get(from_time_b, to_time_b)
        if deals is None:
            return 0.0

        # Gruppiere Deals nach Position (jede Position = ein abgeschlossener Trade)
        trades_by_position: Dict[int, List[Any]] = {}
        for deal in deals:
            if deal.magic != int(magic_number):
                continue
            trades_by_position.setdefault(deal.position_id, []).append(deal)

        realized_r_total = 0.0

        for deal_list in trades_by_position.values():
            # MT5-Konvention: entry==0 -> IN (Entry), entry==1 -> OUT (Exit)
            entry_deal = next(
                (d for d in deal_list if getattr(d, "entry", None) == 0), None
            )
            exit_deal = next(
                (d for d in deal_list if getattr(d, "entry", None) == 1), None
            )
            if not entry_deal or not exit_deal:
                continue

            # SL aus zugehörigem Order rekonstruieren
            orders = mt5.history_orders_get(ticket=entry_deal.order)
            if not orders or not hasattr(orders[0], "sl") or orders[0].sl == 0:
                continue

            sl_distance = abs(entry_deal.price - orders[0].sl)
            if sl_distance == 0:
                continue

            # Buy/Sell anhand deal.type robust bestimmen (Deal- oder Order-Konstante)
            is_buy = getattr(mt5, "DEAL_TYPE_BUY", None)
            if is_buy is not None:
                is_entry_buy = entry_deal.type == mt5.DEAL_TYPE_BUY
            else:
                is_entry_buy = entry_deal.type == mt5.ORDER_TYPE_BUY  # Fallback

            if is_entry_buy:
                price_move = exit_deal.price - entry_deal.price
            else:
                price_move = entry_deal.price - exit_deal.price

            realized_r_total += price_move / sl_distance

        return round(float(realized_r_total), 2)

    def get_floating_r_multiple(self, strategy: str) -> float:
        self._validate_account()
        # Tracker arbeitet in UTC-Tagen; hole direkt die Tages-Map in UTC
        today_data = self.tracker.get_day_data(date=now_utc())
        open_trades = {
            key: data
            for key, data in today_data.items()
            if data.get("status") == "open" and key.endswith(f"::{strategy}")
        }

        if not open_trades:
            return 0.0

        positions = mt5.positions_get()
        if not positions:
            return 0.0

        floating_r_total = 0.0
        for pos in positions:
            # Schlüssel wie im Tracker: EURUSD::MyStrategy
            logical_symbol = self.symbol_mapper.to_logical_from_broker(pos.symbol)
            key = f"{logical_symbol}::{strategy}"

            trade = open_trades.get(key)
            if not trade:
                continue

            initial_risk = trade.get("risk")
            if not initial_risk:
                continue

            r = float(pos.profit) / float(initial_risk)
            floating_r_total += r

        return round(float(floating_r_total), 2)

    def get_total_r_multiple(self, magic_number: int, strategy: str) -> float:
        return round(
            self.get_realized_r_multiple(int(magic_number))
            + self.get_floating_r_multiple(strategy),
            2,
        )

    # -----------------------
    # Trade-Rekonstruktion
    # -----------------------

    def reconstruct_trade_from_deal_ticket(
        self, ticket_id: int, pip_size: float = 0.0001
    ) -> Optional[dict]:
        """
        Rekonstruiert einen abgeschlossenen Trade (Entry/Exit) anhand der Deals des Tages.
        Achtung: Nutzt Tagesfenster in UTC -> konvertiert auf Brokerzeit für MT5-Abfragen.
        """
        self._validate_account()

        # 1) UTC-Zeitfenster mit Lookback über mehrere Tage (z. B. Overnight-Trades)
        now = now_utc()
        lookback_days = max(1, int(_DEAL_HISTORY_LOOKBACK_DAYS))
        from_time_base = now - timedelta(days=lookback_days)
        from_time_utc = from_time_base.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        to_time_utc = now.replace(hour=23, minute=59, second=59, microsecond=0)

        # 2) Broker-Zeit für MT5
        from_time_b = from_utc_to_broker(from_time_utc)
        to_time_b = from_utc_to_broker(to_time_utc)

        history = mt5.history_deals_get(from_time_b, to_time_b)
        if history is None:
            return None

        related_deals = [d for d in history if d.position_id == ticket_id]
        related_deals.sort(key=lambda d: d.time)

        # Korrekt: entry==0 -> Entry / entry==1 -> Exit
        entry_deal = next(
            (d for d in related_deals if getattr(d, "entry", None) == 0), None
        )
        exit_deal = next(
            (d for d in related_deals if getattr(d, "entry", None) == 1), None
        )
        if not entry_deal or not exit_deal:
            return None

        # Exit-Typ erkennen
        comment = (exit_deal.comment or "").lower()
        if "tp" in comment:
            exit_type = "TakeProfit"
        elif "sl" in comment:
            exit_type = "StopLoss"
        elif "close" in comment or "manual" in comment:
            exit_type = "ManualClose"
        else:
            exit_type = "Unknown"

        # Richtung bestimmen (Deal- oder Order-Konstante)
        is_entry_buy = (
            hasattr(mt5, "DEAL_TYPE_BUY") and entry_deal.type == mt5.DEAL_TYPE_BUY
        ) or (entry_deal.type == mt5.ORDER_TYPE_BUY)
        direction = "BUY" if is_entry_buy else "SELL"

        entry_price = float(entry_deal.price)
        exit_price = float(exit_deal.price)
        volume = float(entry_deal.volume)
        profit = float(exit_deal.profit)
        commission = float(exit_deal.commission)
        swap = float(exit_deal.swap)

        # Duration
        # In UTC normalisieren, damit mit OHLC‑Serien (UTC) konsistent
        entry_time = to_utc(datetime.fromtimestamp(entry_deal.time, tz=TIMEZONE))
        exit_time = to_utc(datetime.fromtimestamp(exit_deal.time, tz=TIMEZONE))
        duration_sec = (exit_time - entry_time).total_seconds()
        duration_min = round(duration_sec / 60)

        # Pips (pip_size ggf. symbolabhängig setzen – hier param.)
        price_diff = (
            exit_price - entry_price if is_entry_buy else entry_price - exit_price
        )
        pips = price_diff / float(pip_size)

        # Return relativ in %
        return_relative = (price_diff / entry_price) * 100.0

        return {
            "symbol": entry_deal.symbol,
            "position_id": entry_deal.position_id,
            "entry_ticket": entry_deal.ticket,
            "exit_ticket": exit_deal.ticket,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "volume": volume,
            "pips": round(pips, 1),
            "return_abs": round(profit, 2),
            "return_relative": round(return_relative, 4),
            "duration_sec": int(duration_sec),
            "duration_min": duration_min,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "exit_type": exit_type,
            "commission": round(commission, 2),
            "swap": round(swap, 2),
            "entry_comment": entry_deal.comment,
            "exit_comment": exit_deal.comment,
        }
