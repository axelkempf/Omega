from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from backtest_engine.data.tick import Tick
from backtest_engine.logging.entry_log import EntryLogger
from backtest_engine.logging.trade_logger import TradeLogger


@dataclass
class TradeSignal:
    """
    Data structure representing a trading signal.

    Attributes:
        direction (str): "long" or "short"
        entry_price (float): Entry price
        stop_loss (float): Stop loss level
        take_profit (float): Take profit level
        symbol (str): Trading symbol
        timestamp (datetime): Timestamp of the signal
        type (str): Order type, default "market"
    """

    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    timestamp: datetime
    type: str = "market"
    # ---- Neu: Metadaten & Labeling direkt am Signal mitführen ----
    reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    scenario: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class StrategyWrapper:
    """
    Institutional-ready wrapper for trading strategies with cooldown and logging.

    Args:
        strategy: The trading strategy instance (must implement on_data or check_entry).
        cooldown_candles (int): Minimum candles between entries.
        cooldown_minutes (int): Minimum minutes between entries.
        portfolio (Portfolio): Reference portfolio (optional).
        trade_logging_fields (list): List of custom fields for trade logger (optional).
        enable_logging (bool): Enable entry logging.

    Attributes:
        strategy: The wrapped strategy.
        last_entry_time: Timestamp of last entry.
        last_entry_index: Index of last entry.
        cooldown_candles: Minimum candles between entries.
        cooldown_minutes: Minimum minutes between entries.
        logger: Optional entry logger.
        trade_logger: Optional trade logger.
        cross_symbol: True if strategy is cross-symbol enabled.
    """

    def __init__(
        self,
        strategy: Any,
        cooldown_candles: int = 0,
        cooldown_minutes: int = 0,
        cooldown_minutes_trade: int = 0,
        portfolio: Optional[Portfolio] = None,
        trade_logging_fields: Optional[List[str]] = None,
        enable_logging: bool = False,
        logging_mode: str = "trades_only",  # "off" | "trades_only" | "all"
        entry_timestamp_mode: str = "open",
    ) -> None:
        self.strategy = strategy
        self.last_entry_time: Optional[datetime] = None
        self.last_entry_index: Optional[int] = None
        self.last_exit_time: Optional[datetime] = None
        self.cooldown_candles = cooldown_candles
        self.cooldown_minutes = cooldown_minutes
        self.cooldown_minutes_trade = cooldown_minutes_trade
        self.logger = EntryLogger() if enable_logging else None
        self.logging_mode = logging_mode
        self.strategy.portfolio = portfolio
        self.trade_logger = TradeLogger(fields=trade_logging_fields or [])
        self.cross_symbol: bool = getattr(strategy, "cross_symbol", False)
        self.entry_timestamp_mode = entry_timestamp_mode.lower()

    def _tf_minutes(self) -> int:
        tf = str(self.get_primary_timeframe()).upper()
        if tf.startswith("M"):
            return int(tf[1:])
        if tf.startswith("H"):
            return int(tf[1:]) * 60
        if tf.startswith("D"):
            return int(tf[1:]) * 1440
        return 0

    def _effective_time(self, open_ts):
        """
        Decision-Time: für mode=='close' = Open + TF-Dauer, sonst Open.
        (Nur für Entscheidungen/Logs; Execution bleibt immer Open.)
        """
        if self.entry_timestamp_mode == "close":
            from datetime import timedelta

            minutes = self._tf_minutes()
            return open_ts + timedelta(minutes=minutes) if minutes > 0 else open_ts
        return open_ts

    def evaluate(
        self, index: int, slice_map: Dict[str, SymbolDataSlice]
    ) -> Optional[Union[TradeSignal, List[TradeSignal]]]:
        """
        Evaluates the wrapped strategy for a trade signal, with cooldown logic and multi-leg support.

        Args:
            index (int): Current index in time series.
            slice_map (dict): Map of symbol name to SymbolDataSlice.

        Returns:
            Optional[Union[TradeSignal, List[TradeSignal]]]: Trade signal(s) if entry allowed, else None.
        """
        # Cross-Symbol Handling: Pick reference (primary) symbol for time logic
        if self.cross_symbol:
            primary_symbol = getattr(
                self.strategy, "primary_symbol", list(slice_map.keys())[0]
            )
            symbol_slice = slice_map[primary_symbol]
            symbol_name = primary_symbol
        else:
            symbol_name = list(slice_map.keys())[0]
            symbol_slice = slice_map[symbol_name]

        current_candle = symbol_slice.latest(
            self.get_primary_timeframe(), price_type="bid"
        )
        if not current_candle:
            return None
        open_time = current_candle.timestamp  # Engine-Zeit
        decision_time = self._effective_time(open_time)  # Close (oder Open)

        # Cooldown nach letztem Exit (Trade-basiert)
        if self.last_exit_time and self.cooldown_minutes_trade > 0:
            if decision_time - self.last_exit_time < timedelta(
                minutes=self.cooldown_minutes_trade
            ):
                if self.logger and self.logging_mode == "all":
                    self.logger.log(
                        decision_time,
                        is_candidate=False,
                        entry_allowed=False,
                        blocker="Cooldown(trade_minutes)",
                    )
                return None

        # Cooldown (time-based)
        if self.last_entry_time and self.cooldown_minutes > 0:
            if decision_time - self.last_entry_time < timedelta(
                minutes=self.cooldown_minutes
            ):
                if self.logger and self.logging_mode == "all":
                    self.logger.log(
                        decision_time,
                        is_candidate=False,
                        entry_allowed=False,
                        blocker="Cooldown(minutes)",
                    )
                return None

        # Cooldown (candle-based)
        if self.last_entry_index is not None and self.cooldown_candles > 0:
            if index - self.last_entry_index < self.cooldown_candles:
                if self.logger and self.logging_mode == "all":
                    self.logger.log(
                        decision_time,
                        is_candidate=False,
                        entry_allowed=False,
                        blocker="Cooldown(candles)",
                    )
                return None

        # Strategy Invocation
        if hasattr(self.strategy, "on_data"):
            # on_data expects either full slice_map or per-symbol
            raw_signal = self.strategy.on_data(slice_map)
        else:
            # Legacy check_entry API
            raw_signal = self.strategy.check_entry(symbol_slice, index=index)

        # No Signal
        if not raw_signal:
            # No-Signal Logs nur im "all"-Modus; default spart I/O
            if self.logger and self.logging_mode == "all":
                self.logger.log(
                    decision_time,
                    is_candidate=False,
                    entry_allowed=False,
                    blocker="No Signal",
                )
            return None

        # Multi-Leg (list of signals)
        if isinstance(raw_signal, list):
            signals: List[TradeSignal] = []
            for raw_sig in raw_signal:
                signals.append(
                    self._build_signal(raw_sig, decision_time, open_time, symbol_name)
                )
            # Entry-Cooldowns (minutes) auf Decision-Time führen:
            self.last_entry_time = decision_time
            self.last_entry_index = index
            return signals

        # Single-Leg (dict)
        signal = self._build_signal(raw_signal, decision_time, open_time, symbol_name)
        # Entry-Cooldowns (minutes) auf Decision-Time führen:
        self.last_entry_time = decision_time
        self.last_entry_index = index
        return signal

    def _build_signal(
        self,
        raw_sig: Dict[str, Any],
        decision_time: datetime,
        execution_time_open: datetime,
        default_symbol: str,
    ) -> TradeSignal:
        """
        Helper to build TradeSignal, log entries, and handle direction mapping.

        Args:
            raw_sig (dict): Raw signal dict from strategy.
            current_time (datetime): Timestamp for signal.
            default_symbol (str): Symbol to fallback on if not provided.

        Returns:
            TradeSignal: Clean, typed signal instance.
        """
        direction_map = {"buy": "long", "sell": "short"}
        mapped_direction = direction_map.get(raw_sig["direction"], raw_sig["direction"])
        signal_reason = raw_sig.get("reason", "-")
        tags = (
            list(raw_sig.get("tags", []))
            if isinstance(raw_sig.get("tags", []), (list, tuple))
            else []
        )
        scenario = raw_sig.get("scenario")
        # 'meta' ist der Standard; 'metadata' als Backcompat akzeptieren
        meta: Dict[str, Any] = {}
        if isinstance(raw_sig.get("meta"), dict):
            meta.update(raw_sig["meta"])
        if isinstance(raw_sig.get("metadata"), dict):
            # falls beide vorhanden sind, zusammenführen (metadata überschreibt nicht)
            for k, v in raw_sig["metadata"].items():
                meta.setdefault(k, v)

        # Logging (optional)
        if self.logger and self.logging_mode != "off":
            self.logger.log(
                decision_time,
                is_candidate=True,
                entry_allowed=True,
                signal_reason=signal_reason,
                tags=tags,
            )
        if self.trade_logger:
            try:
                specs = getattr(self.strategy, "symbol_specs", None)
                ps = (
                    specs.get(raw_sig.get("symbol", default_symbol)).pip_size
                    if specs
                    else None
                )
                dec = max(
                    0, min(6, int(round(-__import__("math").log10(ps))) if ps else 5)
                )
            except Exception:
                dec = 5
            # TF in Meta für nachgelagerte Exit-Close-Berechnung
            meta.setdefault("tf", str(self.get_primary_timeframe()))
            entry_data = {
                "timestamp": decision_time,
                "symbol": raw_sig.get("symbol", default_symbol),
                "direction": mapped_direction,
                "entry_price": round(raw_sig["entry"], dec),
                "stop_loss": round(raw_sig["sl"], dec),
                "take_profit": round(raw_sig["tp"], dec),
                "tags": "|".join(tags),
            }
            entry_data.update(meta)
            self.trade_logger.log(entry_data)

        # Engine/Simulator sollen IMMER Open-Zeit als Entry-Zeit bekommen:
        # Decision-Time & Semantik zusätzlich im Meta ablegen.
        meta.setdefault("decision_time", decision_time)
        meta.setdefault("timestamp_semantics", self.entry_timestamp_mode)
        meta.setdefault("bar_open", execution_time_open)
        meta.setdefault("tf", str(self.get_primary_timeframe()))

        return TradeSignal(
            direction=mapped_direction,
            entry_price=raw_sig["entry"],
            stop_loss=raw_sig["sl"],
            take_profit=raw_sig["tp"],
            symbol=raw_sig.get("symbol", default_symbol),
            timestamp=execution_time_open,
            type=raw_sig.get("type", "market"),
            reason=signal_reason,
            tags=tags,
            scenario=scenario,
            meta=meta,
        )

    def evaluate_tick(
        self, tick: Tick, slice_map: Dict[str, SymbolDataSlice]
    ) -> Optional[TradeSignal]:
        """
        Evaluates the strategy on a single tick (if strategy supports tick evaluation).

        Args:
            tick (Tick): Tick data.
            slice_map (dict): Map of symbol to SymbolDataSlice.

        Returns:
            Optional[TradeSignal]: Signal if present and not blocked by cooldown.
        """
        if not hasattr(self.strategy, "evaluate_tick"):
            return None

        if self.last_entry_time and self.cooldown_minutes > 0:
            if tick.timestamp - self.last_entry_time < timedelta(
                minutes=self.cooldown_minutes
            ):
                if self.logger:
                    self.logger.log(
                        tick.timestamp,
                        is_candidate=False,
                        entry_allowed=False,
                        blocker="Cooldown(minutes)",
                    )
                return None

        raw_signal = self.strategy.evaluate_tick(tick, slice_map)
        if raw_signal:
            tags = (
                list(raw_signal.get("tags", []))
                if isinstance(raw_signal.get("tags", []), (list, tuple))
                else []
            )
            signal_reason = raw_signal.get("reason", "tick_signal")
            scenario = raw_signal.get("scenario")
            meta: Dict[str, Any] = {}
            if isinstance(raw_signal.get("meta"), dict):
                meta.update(raw_signal["meta"])
            if isinstance(raw_signal.get("metadata"), dict):
                for k, v in raw_signal["metadata"].items():
                    meta.setdefault(k, v)
            if self.logger:
                self.logger.log(
                    tick.timestamp,
                    is_candidate=True,
                    entry_allowed=True,
                    signal_reason=signal_reason,
                    tags=tags,
                )
            self.last_entry_time = tick.timestamp
            return TradeSignal(
                direction=raw_signal["direction"],
                entry_price=raw_signal["entry"],
                stop_loss=raw_signal["sl"],
                take_profit=raw_signal["tp"],
                symbol=raw_signal.get("symbol", ""),
                timestamp=tick.timestamp,
                type=raw_signal.get("type", "market"),
                reason=signal_reason,
                tags=tags,
                scenario=scenario,
                meta=meta,
            )
        else:
            if self.logger:
                self.logger.log(
                    tick.timestamp,
                    is_candidate=False,
                    entry_allowed=False,
                    blocker="No Signal (tick)",
                )
            return None

    def get_primary_timeframe(self) -> Any:
        """
        Returns the primary timeframe of the wrapped strategy.

        Returns:
            Any: The primary timeframe object.

        Raises:
            AttributeError: If not implemented by the strategy.
        """
        if hasattr(self.strategy, "get_primary_timeframe"):
            return self.strategy.get_primary_timeframe()
        raise AttributeError("Wrapped strategy has no get_primary_timeframe()")

    def on_data(self, slice_map: Dict[str, SymbolDataSlice]) -> Any:
        """
        Pass-through for strategy's on_data method.

        Args:
            slice_map (dict): Map of symbol to SymbolDataSlice.

        Returns:
            Any: Strategy output.
        """
        return self.strategy.on_data(slice_map)

    def on_signal(self, *args, **kwargs) -> Any:
        """
        Pass-through for strategy's on_signal method, if present.

        Returns:
            Any: Strategy output.
        """
        if hasattr(self.strategy, "on_signal"):
            return self.strategy.on_signal(*args, **kwargs)
        return None
