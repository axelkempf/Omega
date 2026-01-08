"""Strategy wrapper module for institutional-ready trading strategies.

This module provides a type-safe wrapper that adds cooldown logic, logging,
and multi-leg signal support to trading strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Final, List, Optional, Protocol, Union, runtime_checkable

from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from backtest_engine.data.tick import Tick
from backtest_engine.bt_logging.entry_log import EntryLogger
from backtest_engine.bt_logging.trade_logger import TradeLogger

# Type aliases for clarity
SignalDict = Dict[str, Any]
SliceMap = Dict[str, SymbolDataSlice]


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol defining the minimum interface for a trading strategy."""

    portfolio: Optional[Portfolio]

    def on_data(
        self, slice_map: SliceMap
    ) -> Optional[Union[SignalDict, List[SignalDict]]]:
        """Process data and optionally return signal(s)."""
        ...

    def get_primary_timeframe(self) -> Any:
        """Return the primary timeframe for the strategy."""
        ...


@dataclass(slots=True)
class TradeSignal:
    """Data structure representing a trading signal.

    Attributes:
        direction: "long" or "short"
        entry_price: Entry price
        stop_loss: Stop loss level
        take_profit: Take profit level
        symbol: Trading symbol
        timestamp: Timestamp of the signal
        type: Order type, default "market"
        reason: Optional reason/description for the signal
        tags: List of tags for categorization
        scenario: Optional scenario identifier
        meta: Additional metadata dictionary
    """

    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    timestamp: datetime
    type: str = "market"
    reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    scenario: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# Constants for logging modes
LOGGING_MODE_OFF: Final[str] = "off"
LOGGING_MODE_TRADES_ONLY: Final[str] = "trades_only"
LOGGING_MODE_ALL: Final[str] = "all"

# Constants for timestamp modes
TIMESTAMP_MODE_OPEN: Final[str] = "open"
TIMESTAMP_MODE_CLOSE: Final[str] = "close"

# Direction mappings
DIRECTION_MAP: Final[Dict[str, str]] = {"buy": "long", "sell": "short"}


class StrategyWrapper:
    """Institutional-ready wrapper for trading strategies with cooldown and logging.

    Provides:
    - Time-based and candle-based cooldown logic
    - Entry and trade logging
    - Multi-leg signal support
    - Cross-symbol strategy handling

    Attributes:
        strategy: The wrapped strategy instance
        last_entry_time: Timestamp of last entry
        last_entry_index: Index of last entry
        last_exit_time: Timestamp of last exit
        cooldown_candles: Minimum candles between entries
        cooldown_minutes: Minimum minutes between entries
        cooldown_minutes_trade: Minimum minutes after exit before new entry
        logger: Optional entry logger
        trade_logger: Trade logger instance
        logging_mode: Logging verbosity mode
        cross_symbol: True if strategy handles multiple symbols
        entry_timestamp_mode: "open" or "close" for entry timing
    """

    __slots__ = (
        "strategy",
        "last_entry_time",
        "last_entry_index",
        "last_exit_time",
        "cooldown_candles",
        "cooldown_minutes",
        "cooldown_minutes_trade",
        "logger",
        "logging_mode",
        "trade_logger",
        "cross_symbol",
        "entry_timestamp_mode",
    )

    def __init__(
        self,
        strategy: Any,
        cooldown_candles: int = 0,
        cooldown_minutes: int = 0,
        cooldown_minutes_trade: int = 0,
        portfolio: Optional[Portfolio] = None,
        trade_logging_fields: Optional[List[str]] = None,
        enable_logging: bool = False,
        logging_mode: str = LOGGING_MODE_TRADES_ONLY,
        entry_timestamp_mode: str = TIMESTAMP_MODE_OPEN,
    ) -> None:
        """Initialize strategy wrapper.

        Args:
            strategy: The trading strategy instance (must implement on_data or check_entry)
            cooldown_candles: Minimum candles between entries
            cooldown_minutes: Minimum minutes between entries
            cooldown_minutes_trade: Minimum minutes after trade exit before new entry
            portfolio: Reference portfolio (optional)
            trade_logging_fields: List of custom fields for trade logger
            enable_logging: Enable entry logging
            logging_mode: "off", "trades_only", or "all"
            entry_timestamp_mode: "open" or "close" for entry timing
        """
        self.strategy: Any = strategy
        self.last_entry_time: Optional[datetime] = None
        self.last_entry_index: Optional[int] = None
        self.last_exit_time: Optional[datetime] = None
        self.cooldown_candles: int = cooldown_candles
        self.cooldown_minutes: int = cooldown_minutes
        self.cooldown_minutes_trade: int = cooldown_minutes_trade
        self.logger: Optional[EntryLogger] = EntryLogger() if enable_logging else None
        self.logging_mode: str = logging_mode
        self.strategy.portfolio = portfolio
        self.trade_logger: TradeLogger = TradeLogger(fields=trade_logging_fields or [])
        self.cross_symbol: bool = bool(getattr(strategy, "cross_symbol", False))
        self.entry_timestamp_mode: str = entry_timestamp_mode.lower()

    def _tf_minutes(self) -> int:
        """Extract minutes from timeframe string.

        Returns:
            Number of minutes for the timeframe (0 if unparseable)
        """
        tf: str = str(self.get_primary_timeframe()).upper()
        if tf.startswith("M"):
            return int(tf[1:])
        if tf.startswith("H"):
            return int(tf[1:]) * 60
        if tf.startswith("D"):
            return int(tf[1:]) * 1440
        return 0

    def _effective_time(self, open_ts: datetime) -> datetime:
        """Calculate decision time based on timestamp mode.

        For mode=='close': returns Open + TF duration
        For mode=='open': returns Open timestamp

        Args:
            open_ts: Bar open timestamp

        Returns:
            Effective decision time
        """
        if self.entry_timestamp_mode == TIMESTAMP_MODE_CLOSE:
            minutes: int = self._tf_minutes()
            return open_ts + timedelta(minutes=minutes) if minutes > 0 else open_ts
        return open_ts

    def evaluate(
        self, index: int, slice_map: SliceMap
    ) -> Optional[Union[TradeSignal, List[TradeSignal]]]:
        """Evaluate the wrapped strategy for trade signals with cooldown logic.

        Args:
            index: Current index in time series
            slice_map: Map of symbol name to SymbolDataSlice

        Returns:
            Trade signal(s) if entry allowed, else None
        """
        # Cross-Symbol Handling: Pick reference (primary) symbol for time logic
        symbol_name: str
        symbol_slice: SymbolDataSlice

        if self.cross_symbol:
            primary_symbol: str = str(
                getattr(self.strategy, "primary_symbol", list(slice_map.keys())[0])
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

        open_time: datetime = current_candle.timestamp
        decision_time: datetime = self._effective_time(open_time)

        # Cooldown nach letztem Exit (Trade-basiert)
        if self.last_exit_time and self.cooldown_minutes_trade > 0:
            if decision_time - self.last_exit_time < timedelta(
                minutes=self.cooldown_minutes_trade
            ):
                if self.logger and self.logging_mode == LOGGING_MODE_ALL:
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
                if self.logger and self.logging_mode == LOGGING_MODE_ALL:
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
                if self.logger and self.logging_mode == LOGGING_MODE_ALL:
                    self.logger.log(
                        decision_time,
                        is_candidate=False,
                        entry_allowed=False,
                        blocker="Cooldown(candles)",
                    )
                return None

        # Strategy Invocation
        raw_signal: Optional[Union[SignalDict, List[SignalDict]]]
        if hasattr(self.strategy, "on_data"):
            raw_signal = self.strategy.on_data(slice_map)
        else:
            # Legacy check_entry API
            raw_signal = self.strategy.check_entry(symbol_slice, index=index)

        # No Signal
        if not raw_signal:
            if self.logger and self.logging_mode == LOGGING_MODE_ALL:
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
            self.last_entry_time = decision_time
            self.last_entry_index = index
            return signals

        # Single-Leg (dict)
        signal: TradeSignal = self._build_signal(
            raw_signal, decision_time, open_time, symbol_name
        )
        self.last_entry_time = decision_time
        self.last_entry_index = index
        return signal

    def _build_signal(
        self,
        raw_sig: SignalDict,
        decision_time: datetime,
        execution_time_open: datetime,
        default_symbol: str,
    ) -> TradeSignal:
        """Build TradeSignal from raw dict, log entries, and handle direction mapping.

        Args:
            raw_sig: Raw signal dict from strategy
            decision_time: Decision timestamp (for logging)
            execution_time_open: Bar open timestamp (for execution)
            default_symbol: Symbol to fallback on if not provided

        Returns:
            Clean, typed TradeSignal instance
        """
        mapped_direction: str = DIRECTION_MAP.get(
            raw_sig["direction"], raw_sig["direction"]
        )
        signal_reason: str = str(raw_sig.get("reason", "-"))

        tags: List[str]
        raw_tags = raw_sig.get("tags", [])
        if isinstance(raw_tags, (list, tuple)):
            tags = list(raw_tags)
        else:
            tags = []

        scenario: Optional[str] = raw_sig.get("scenario")

        # 'meta' is standard; 'metadata' accepted for backcompat
        meta: Dict[str, Any] = {}
        if isinstance(raw_sig.get("meta"), dict):
            meta.update(raw_sig["meta"])
        if isinstance(raw_sig.get("metadata"), dict):
            for k, v in raw_sig["metadata"].items():
                meta.setdefault(k, v)

        # Logging (optional)
        if self.logger and self.logging_mode != LOGGING_MODE_OFF:
            self.logger.log(
                decision_time,
                is_candidate=True,
                entry_allowed=True,
                signal_reason=signal_reason,
                tags=tags,
            )

        if self.trade_logger:
            dec: int = self._get_price_decimals(raw_sig, default_symbol)
            meta.setdefault("tf", str(self.get_primary_timeframe()))
            entry_data: Dict[str, Any] = {
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

        # Engine/Simulator always gets Open time as entry time
        meta.setdefault("decision_time", decision_time)
        meta.setdefault("timestamp_semantics", self.entry_timestamp_mode)
        meta.setdefault("bar_open", execution_time_open)
        meta.setdefault("tf", str(self.get_primary_timeframe()))

        return TradeSignal(
            direction=mapped_direction,
            entry_price=float(raw_sig["entry"]),
            stop_loss=float(raw_sig["sl"]),
            take_profit=float(raw_sig["tp"]),
            symbol=str(raw_sig.get("symbol", default_symbol)),
            timestamp=execution_time_open,
            type=str(raw_sig.get("type", "market")),
            reason=signal_reason,
            tags=tags,
            scenario=scenario,
            meta=meta,
        )

    def _get_price_decimals(self, raw_sig: SignalDict, default_symbol: str) -> int:
        """Determine decimal places for price rounding based on symbol specs.

        Args:
            raw_sig: Raw signal dict
            default_symbol: Default symbol name

        Returns:
            Number of decimal places (default 5)
        """
        try:
            specs = getattr(self.strategy, "symbol_specs", None)
            if specs:
                symbol = raw_sig.get("symbol", default_symbol)
                spec = specs.get(symbol)
                if spec and spec.pip_size:
                    return max(0, min(6, int(round(-math.log10(spec.pip_size)))))
        except Exception:
            pass
        return 5

    def evaluate_tick(self, tick: Tick, slice_map: SliceMap) -> Optional[TradeSignal]:
        """Evaluate the strategy on a single tick (if strategy supports it).

        Args:
            tick: Tick data
            slice_map: Map of symbol to SymbolDataSlice

        Returns:
            Signal if present and not blocked by cooldown
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

        raw_signal: Optional[SignalDict] = self.strategy.evaluate_tick(tick, slice_map)

        if raw_signal:
            raw_tags = raw_signal.get("tags", [])
            tags: List[str] = (
                list(raw_tags) if isinstance(raw_tags, (list, tuple)) else []
            )
            signal_reason: str = str(raw_signal.get("reason", "tick_signal"))
            scenario: Optional[str] = raw_signal.get("scenario")

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
                direction=str(raw_signal["direction"]),
                entry_price=float(raw_signal["entry"]),
                stop_loss=float(raw_signal["sl"]),
                take_profit=float(raw_signal["tp"]),
                symbol=str(raw_signal.get("symbol", "")),
                timestamp=tick.timestamp,
                type=str(raw_signal.get("type", "market")),
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
        """Return the primary timeframe of the wrapped strategy.

        Returns:
            The primary timeframe object

        Raises:
            AttributeError: If not implemented by the strategy
        """
        if hasattr(self.strategy, "get_primary_timeframe"):
            return self.strategy.get_primary_timeframe()
        raise AttributeError("Wrapped strategy has no get_primary_timeframe()")

    def on_data(self, slice_map: SliceMap) -> Any:
        """Pass-through for strategy's on_data method.

        Args:
            slice_map: Map of symbol to SymbolDataSlice

        Returns:
            Strategy output
        """
        return self.strategy.on_data(slice_map)

    def on_signal(self, *args: Any, **kwargs: Any) -> Any:
        """Pass-through for strategy's on_signal method, if present.

        Returns:
            Strategy output or None
        """
        if hasattr(self.strategy, "on_signal"):
            return self.strategy.on_signal(*args, **kwargs)
        return None
