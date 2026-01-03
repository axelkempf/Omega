from datetime import datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from backtest_engine.data.candle import Candle
from backtest_engine.data.tick import Tick
from backtest_engine.strategy.strategy_wrapper import TradeSignal


class AnchoredSessionFilterWrapper:
    """
    Wraps a trading strategy with session-based filtering according to an anchored calendar.

    Args:
        wrapped_strategy: The strategy instance to wrap.
        anchored_calendar (dict): Dict mapping date string to list of (start, end) time windows.

    Attributes:
        strategy: The wrapped strategy.
        calendar: The anchored session calendar.
        cross_symbol: Whether cross-symbol logic is active.
        logger: Optional, used for session logging.
    """

    def __init__(self, wrapped_strategy: Any, anchored_calendar: dict) -> None:
        self.strategy = wrapped_strategy
        self.calendar = anchored_calendar
        self.cross_symbol: bool = getattr(wrapped_strategy, "cross_symbol", False)
        self.logger: Optional[Any] = getattr(wrapped_strategy, "logger", None)
        self.logging_mode: str = getattr(
            wrapped_strategy, "logging_mode", "trades_only"
        )

    # --- Permanent: Session-Gating immer auf Kerzen-CLOSE (Open + TF-Dauer) ---
    def _tf_minutes(self) -> int:
        tf = str(self.strategy.get_primary_timeframe()).upper()
        if tf.startswith("M"):
            return int(tf[1:])
        if tf.startswith("H"):
            return int(tf[1:]) * 60
        if tf.startswith("D"):
            return int(tf[1:]) * 1440
        return 0

    def _effective_time_close(self, open_ts: datetime) -> datetime:
        mins = self._tf_minutes()
        return open_ts + timedelta(minutes=mins) if mins > 0 else open_ts

    def _is_allowed(self, dt: datetime) -> bool:
        """
        Checks if a datetime is within any allowed session window for its date.

        Args:
            dt (datetime): The datetime to check.

        Returns:
            bool: True if allowed, else False.
        """
        date_str = dt.date().isoformat()
        time_now = dt.time()
        windows = self.calendar.get(date_str, [])
        for start_str, end_str in windows:
            start = time.fromisoformat(start_str)
            end = time.fromisoformat(end_str)
            if start <= end:
                if start <= time_now <= end:
                    return True
            else:
                if time_now >= start or time_now <= end:
                    return True
        return False

    def evaluate(
        self, index: int, slice_map: Dict[str, SymbolDataSlice]
    ) -> Optional[List[TradeSignal]]:
        """
        Evaluates the wrapped strategy only if the current time is allowed.

        Args:
            index (int): Index in the time series.
            slice_map (dict): Mapping from symbol to SymbolDataSlice.

        Returns:
            Optional[List[TradeSignal]]: Strategy output if allowed, else None.
        """
        if self.cross_symbol:
            primary_symbol = getattr(
                self.strategy, "primary_symbol", list(slice_map.keys())[0]
            )
            symbol_slice = slice_map[primary_symbol]
            current_candle = symbol_slice.latest(
                self.strategy.get_primary_timeframe(), price_type="bid"
            )
        else:
            symbol = list(slice_map.keys())[0]
            symbol_slice = slice_map[symbol]
            current_candle = symbol_slice.latest(
                self.strategy.get_primary_timeframe(), price_type="bid"
            )

        if not current_candle:
            return None

        # Gate immer nach CLOSE-Zeit (Open + TF-Dauer)
        current_time = self._effective_time_close(current_candle.timestamp)
        if not self._is_allowed(current_time):
            if self.logger and self.logging_mode == "all":
                self.logger.log(
                    current_time,
                    is_candidate=True,
                    entry_allowed=False,
                    blocker="AnchoredSession",
                )
            return None

        # Pass through to wrapped strategy
        if self.cross_symbol:
            result = self.strategy.evaluate(index, slice_map)
        else:
            result = self.strategy.evaluate(index, {symbol: symbol_slice})

        if result is None:
            return None
        if isinstance(result, list):
            return result
        return [result]

    def evaluate_tick(
        self, tick: Tick, slice_map: Dict[str, SymbolDataSlice]
    ) -> Optional[Any]:
        """
        Evaluates the wrapped strategy on tick-level if session is open.

        Args:
            tick (Tick): The tick data object.
            slice_map (dict): Mapping from symbol to SymbolDataSlice.

        Returns:
            Optional[Any]: Strategy output if allowed, else None.
        """
        dt = tick.timestamp
        if not self._is_allowed(dt):
            if self.logger and self.logging_mode == "all":
                self.logger.log(
                    dt,
                    is_candidate=True,
                    entry_allowed=False,
                    blocker="AnchoredSession",
                )
            return None
        return self.strategy.evaluate_tick(tick, slice_map)

    def on_signal(self, *args, **kwargs) -> Any:
        """
        Pass-through for on_signal hook to the wrapped strategy.
        """
        return self.strategy.on_signal(*args, **kwargs)


class UniversalSessionFilterWrapper:
    """
    Wraps a trading strategy with universal time-window-based session filtering.

    Args:
        wrapped_strategy: The strategy instance to wrap.
        session_filter (dict): Dict with "fixed_times" and/or "sessions" for filtering.

    Attributes:
        strategy: The wrapped strategy.
        cross_symbol: Whether cross-symbol logic is active.
        fixed_times: List of (start, end) times for trading windows.
        logger: Optional, used for session logging.
    """

    def __init__(self, wrapped_strategy: Any, session_filter: dict) -> None:
        self.strategy = wrapped_strategy
        self.cross_symbol: bool = getattr(wrapped_strategy, "cross_symbol", False)
        self.fixed_times: List[Tuple[time, time]] = [
            (time.fromisoformat(s), time.fromisoformat(e))
            for s, e in session_filter.get("fixed_times", [])
        ]
        self.session_names: List[str] = session_filter.get("sessions", [])
        self.logger: Optional[Any] = getattr(wrapped_strategy, "logger", None)
        self.logging_mode: str = getattr(
            wrapped_strategy, "logging_mode", "trades_only"
        )

        # --- Permanent: Session-Gating immer auf Kerzen-CLOSE ---

    def _tf_minutes(self) -> int:
        tf = str(self.strategy.get_primary_timeframe()).upper()
        if tf.startswith("M"):
            return int(tf[1:])
        if tf.startswith("H"):
            return int(tf[1:]) * 60
        if tf.startswith("D"):
            return int(tf[1:]) * 1440
        return 0

    def _effective_time_close(self, open_ts: datetime) -> datetime:
        mins = self._tf_minutes()
        return open_ts + timedelta(minutes=mins) if mins > 0 else open_ts

    def _in_fixed_time(self, t: time) -> bool:
        """
        Checks if a time is inside any of the configured fixed time windows.

        Args:
            t (time): Time to check.

        Returns:
            bool: True if allowed, else False.
        """
        if not self.fixed_times:
            return True
        for start, end in self.fixed_times:
            if start <= end:
                if start <= t <= end:
                    return True
            else:
                if t >= start or t <= end:
                    return True
        return False

    def _is_allowed(self, dt: datetime) -> bool:
        """
        Checks if a datetime is allowed by fixed time windows.

        Args:
            dt (datetime): The datetime to check.

        Returns:
            bool: True if allowed, else False.
        """
        return self._in_fixed_time(dt.time())

    def evaluate(
        self, index: int, slice_map: Dict[str, SymbolDataSlice]
    ) -> Optional[List[TradeSignal]]:
        """
        Evaluates the wrapped strategy only if the current time is within allowed windows.

        Args:
            index (int): Index in the time series.
            slice_map (dict): Mapping from symbol to SymbolDataSlice.

        Returns:
            Optional[List[TradeSignal]]: Strategy output if allowed, else None.
        """
        if self.cross_symbol:
            primary_symbol = getattr(
                self.strategy, "primary_symbol", list(slice_map.keys())[0]
            )
            symbol_slice = slice_map[primary_symbol]
            current_candle = symbol_slice.latest(
                self.strategy.get_primary_timeframe(), price_type="bid"
            )
        else:
            symbol = list(slice_map.keys())[0]
            symbol_slice = slice_map[symbol]
            current_candle = symbol_slice.latest(
                self.strategy.get_primary_timeframe(), price_type="bid"
            )

        if current_candle is None:
            return None

        # Gate immer nach CLOSE-Zeit (Open + TF-Dauer)
        eff_time = self._effective_time_close(current_candle.timestamp)
        if not self._is_allowed(eff_time):
            if self.logger and self.logging_mode == "all":
                self.logger.log(
                    eff_time,
                    is_candidate=True,
                    entry_allowed=False,
                    blocker="UniversalSession",
                )
            return None

        if self.cross_symbol:
            result = self.strategy.evaluate(index, slice_map)
        else:
            result = self.strategy.evaluate(index, {symbol: symbol_slice})

        if result is None:
            return None
        if isinstance(result, list):
            return result
        return [result]

    def evaluate_tick(
        self, tick: Tick, slice_map: Dict[str, SymbolDataSlice]
    ) -> Optional[Any]:
        """
        Evaluates the wrapped strategy on tick-level if session is open.

        Args:
            tick (Tick): The tick data object.
            slice_map (dict): Mapping from symbol to SymbolDataSlice.

        Returns:
            Optional[Any]: Strategy output if allowed, else None.
        """
        dt = tick.timestamp
        if not self._is_allowed(dt):
            if self.logger and self.logging_mode == "all":
                self.logger.log(
                    dt,
                    is_candidate=True,
                    entry_allowed=False,
                    blocker="UniversalSession",
                )
            return None
        return self.strategy.evaluate_tick(tick, slice_map)

    def on_signal(self, *args, **kwargs) -> Any:
        """
        Pass-through for on_signal hook to the wrapped strategy.
        """
        return self.strategy.on_signal(*args, **kwargs)
