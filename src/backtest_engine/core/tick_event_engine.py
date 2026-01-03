from datetime import datetime
from typing import Callable, Dict, List, Optional

from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.symbol_data_slicer import SymbolDataSlice
from backtest_engine.data.tick import Tick
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper


class TickEventEngine:
    """
    Simuliert die Tick-basierte Eventschleife für eine Strategie.

    Args:
        ticks: Liste aller Ticks (sortiert).
        strategy: Strategie-Objekt (muss .symbol besitzen).
        executor: ExecutionSimulator-Instanz.
        portfolio: Portfolio-Instanz.
        multi_candle_data: Dict[TF][{"bid": [...], "ask": [...]}]
        on_progress: Optionaler Fortschritts-Callback.
    """

    def __init__(
        self,
        ticks: List[Tick],
        strategy: StrategyWrapper,
        executor: ExecutionSimulator,
        portfolio: Portfolio,
        multi_candle_data: Dict[str, Dict[str, List]],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        self.ticks = ticks
        self.strategy = strategy
        self.executor = executor
        self.portfolio = portfolio
        self.multi_candle_data = multi_candle_data
        self.on_progress = on_progress

        self.symbol = strategy.symbol
        self.warmup_bars = 0
        self.original_start_dt: Optional[datetime] = None

    def run(self):
        """
        Führt das Event-Loop über alle Ticks aus.
        """
        total = len(self.ticks)

        if not self.original_start_dt:
            raise ValueError(
                "❌ TickEventEngine benötigt `original_start_dt`, wurde aber nicht gesetzt."
            )

        start_index = next(
            (
                i
                for i, t in enumerate(self.ticks)
                if t.timestamp >= self.original_start_dt
            ),
            self.warmup_bars,
        )

        print(f"⏱ Warmup bis Tick #{start_index} ({self.ticks[start_index].timestamp})")

        for i in range(start_index, total):
            tick = self.ticks[i]

            # Slice für Kontextdaten erzeugen (z. B. M1, M5, H1)
            matching_index = self._find_nearest_candle_index(
                self.multi_candle_data, tick.timestamp
            )

            symbol_slice = SymbolDataSlice(
                multi_candle_data=self.multi_candle_data, index=matching_index
            )

            # Entry evaluieren
            signal = self.strategy.evaluate_tick(tick, {self.symbol: symbol_slice})
            if signal:
                self.executor.process_signal_tick(signal, tick)

            # Exits prüfen
            self.executor.evaluate_exits_tick(tick)
            self.portfolio.update(tick.timestamp)

            if self.on_progress and i % 10000 == 0:
                self.on_progress(i + 1, total)

    def _find_nearest_candle_index(
        self, multi_candle_data: Dict[str, Dict[str, List]], timestamp: datetime
    ) -> int:
        """
        Sucht im Primary Timeframe die Candle mit passendem/nahem Timestamp.
        Optional: Binäre Suche für große Datensätze möglich.

        Args:
            multi_candle_data: Dict[TF][{"bid": [...], ...}]
            timestamp: Der Tick-Timestamp.

        Returns:
            Index der Candle im Primary Timeframe, die <= Tick ist.
        """
        primary_tf = sorted(multi_candle_data.keys())[0]  # z. B. "M1"
        candles = multi_candle_data[primary_tf]["bid"]

        for i in range(len(candles)):
            if candles[i].timestamp > timestamp:
                return max(i - 1, 0)
        return len(candles) - 1
