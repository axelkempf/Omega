from typing import Any, Callable, Dict, List, Optional

from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.indicator_cache import get_cached_indicator_cache
from backtest_engine.core.multi_symbol_slice import MultiSymbolSlice
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from backtest_engine.data.candle import Candle
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper


class EventEngine:
    """
    Event Engine für Single-Symbol Backtests.

    Args:
        bid_candles: Liste der Bid-Kerzen.
        ask_candles: Liste der Ask-Kerzen.
        strategy: Strategie-Wrapper.
        executor: Execution Simulator.
        portfolio: Portfolio-Objekt.
        multi_candle_data: Dict mit Multi-TF Candle-Listen.
        symbol: Symbol-Name (z.B. 'EURUSD').
        on_progress: Optionaler Callback für Fortschrittsanzeige.
    """

    def __init__(
        self,
        bid_candles: List[Candle],
        ask_candles: List[Candle],
        strategy: StrategyWrapper,
        executor: ExecutionSimulator,
        portfolio: Portfolio,
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        symbol: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        original_start_dt: Optional[
            Any
        ] = None,  # Typ je nach Timestamp-Klasse anpassen!
    ):
        self.bid_candles = bid_candles
        self.ask_candles = ask_candles
        self.strategy = strategy
        self.executor = executor
        self.portfolio = portfolio
        self.multi_candle_data = multi_candle_data
        self.symbol = symbol
        self.on_progress = on_progress
        self.original_start_dt = original_start_dt

    def run(self):
        """
        Hauptschleife für die Event Engine (Single Symbol).
        """
        total = len(self.bid_candles)

        if self.original_start_dt is None:
            raise ValueError("original_start_dt muss gesetzt werden!")

        start_index = next(
            (
                i
                for i, c in enumerate(self.bid_candles)
                if c.timestamp >= self.original_start_dt
            ),
            None,
        )
        if start_index is None:
            raise ValueError("Kein Startindex gefunden – überprüfe original_start_dt!")

        # Wiederverwendbaren IndicatorCache beziehen (spart DF-Build & Recompute)
        ind_cache = get_cached_indicator_cache(self.multi_candle_data)

        # Reusable SymbolDataSlice (vermeidet Objektkonstruktion pro Bar) + Cache
        symbol_slice = SymbolDataSlice(
            multi_candle_data=self.multi_candle_data,
            index=start_index,
            indicator_cache=ind_cache,
        )
        slice_map = {self.symbol: symbol_slice}

        for i in range(start_index, total):
            # lokales Binding für Speed
            bid_candle = self.bid_candles[i]
            ask_candle = self.ask_candles[i]
            timestamp = bid_candle.timestamp
            # Index weiterschieben statt neues Objekt zu bauen
            symbol_slice.set_index(i)

            # === ENTRY ===
            signals = self.strategy.evaluate(i, slice_map)
            if signals:
                if not isinstance(signals, list):
                    signals = [signals]
                for signal in signals:
                    self.executor.process_signal(signal)

            # === EXITS ===
            if self.executor.active_positions:
                self.executor.evaluate_exits(bid_candle, ask_candle)

            # === POSITIONSMANAGEMENT ===
            if self.executor.active_positions:
                strategy_instance = getattr(self.strategy.strategy, "strategy", None)
                pm = getattr(strategy_instance, "position_manager", None)
                if pm:
                    if not getattr(pm, "portfolio", None):
                        pm.attach_portfolio(self.portfolio)

                    open_pos = self.portfolio.get_open_positions(self.symbol)
                    all_pos = self.executor.active_positions
                    pm.manage_positions(
                        open_positions=open_pos,
                        symbol_slice=symbol_slice,
                        bid_candle=bid_candle,
                        ask_candle=ask_candle,
                        all_positions=all_pos,
                    )

            # === PORTFOLIO ===
            self.portfolio.update(timestamp)

            # === Fortschritt melden ===
            if callable(self.on_progress):
                self.on_progress((i - start_index) + 1, (total - start_index))


class CrossSymbolEventEngine:
    """
    Event Engine für Multi-Symbol Backtests.

    Args:
        candle_lookups: Dict[symbol][side][timestamp] = Candle
        common_timestamps: List of synchronisierte Timestamps.
        strategy: Strategy-Wrapper oder Strategie-Objekt.
        executor: ExecutionSimulator.
        portfolio: Portfolio-Objekt.
        primary_tf: Haupt-Timeframe als String.
        on_progress: Optionaler Fortschritt-Callback.
    """

    def __init__(
        self,
        candle_lookups: Dict[str, Dict[str, Dict[Any, Candle]]],
        common_timestamps: List[Any],
        strategy: Any,
        executor: ExecutionSimulator,
        portfolio: Portfolio,
        primary_tf: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        original_start_dt: Optional[Any] = None,
    ):
        self.candle_lookups = candle_lookups
        self.common_timestamps = common_timestamps
        self.strategy = strategy
        self.executor = executor
        self.portfolio = portfolio
        self.primary_tf = primary_tf
        self.on_progress = on_progress
        self.original_start_dt = original_start_dt

    def run(self):
        """
        Hauptschleife für die Multi-Symbol Event Engine.
        """
        total = len(self.common_timestamps)

        if self.original_start_dt is not None:
            start_index = next(
                (
                    i
                    for i, ts in enumerate(self.common_timestamps)
                    if ts >= self.original_start_dt
                ),
                None,
            )
            if start_index is None:
                raise ValueError(
                    "Kein Timestamp >= original_start_dt nach Warmup-Cutoff gefunden."
                )
        else:
            start_index = 0

        multi_slice = MultiSymbolSlice(
            self.candle_lookups, self.common_timestamps[start_index], self.primary_tf
        )
        for idx in range(start_index, total):
            ts = self.common_timestamps[idx]
            multi_slice.set_timestamp(ts)

            signals = self.strategy.evaluate(idx, multi_slice)
            if signals:
                if not isinstance(signals, list):
                    signals = [signals]
                for signal in signals:
                    self.executor.process_signal(signal)

            for symbol in self.candle_lookups:
                bid = self.candle_lookups[symbol]["bid"].get(ts)
                ask = self.candle_lookups[symbol]["ask"].get(ts)
                if bid and ask:
                    self.executor.evaluate_exits(bid, ask)

            strategy_instance = getattr(
                self.strategy.strategy, "strategy", self.strategy
            )
            pm = getattr(strategy_instance, "position_manager", None)
            if pm:
                if not getattr(pm, "portfolio", None):
                    pm.attach_portfolio(self.portfolio)
                all_pos = self.executor.active_positions
                for sym in self.candle_lookups:
                    bid_candle = self.candle_lookups[sym]["bid"].get(ts)
                    ask_candle = self.candle_lookups[sym]["ask"].get(ts)
                    if bid_candle and ask_candle:
                        open_pos = [
                            p for p in all_pos if p.symbol == sym and p.status == "open"
                        ]
                        pm.manage_positions(
                            open_positions=open_pos,
                            symbol_slice=multi_slice,
                            bid_candle=bid_candle,
                            ask_candle=ask_candle,
                            all_positions=all_pos,
                        )

            self.portfolio.update(ts)

            if callable(self.on_progress):
                self.on_progress((idx - start_index) + 1, (total - start_index))
