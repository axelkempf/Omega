from typing import Callable, Dict, List, Optional

from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.multi_symbol_slice import MultiSymbolSlice
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.data.candle import Candle
from backtest_engine.data.tick import Tick
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper


class StrategyEnvironment:
    """
    Kapselt Strategie, Portfolio, Executor und die zugehörigen Multi-TF-Daten.

    Args:
        name: Name der Strategie.
        strategy: StrategyWrapper-Instanz.
        multi_candle_data: Dict oder Slice-Objekt mit Multi-TF Candle-Daten.
    """

    def __init__(
        self,
        name: str,
        strategy: StrategyWrapper,
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
    ):
        self.name = name
        self.strategy = strategy
        self.portfolio = Portfolio()
        self.executor = ExecutionSimulator(self.portfolio)
        self.multi_candle_data = multi_candle_data

    def evaluate_candle(self, i: int, bid_candle: Candle, ask_candle: Candle):
        """
        Evaluates one candle-step für diese Strategie.
        """
        self.multi_candle_data.set_index(i)

        # Sicherheitsprüfung pro Symbol/TF
        for symbol in self.multi_candle_data.slices:
            slice = self.multi_candle_data.get(symbol)
            for tf in slice.tf_bid_candles:
                if i >= len(slice.tf_bid_candles[tf]):
                    print(
                        f"⚠️ Index {i} überschreitet ask-Candle-Länge für {symbol}, TF {tf}"
                    )
                    return
            for tf in slice.tf_ask_candles:
                if i >= len(slice.tf_ask_candles[tf]):
                    print(
                        f"⚠️ Index {i} überschreitet ask-Candle-Länge für {symbol}, TF {tf}"
                    )
                    return

        signals = self.strategy.evaluate(i, self.multi_candle_data)
        if signals:
            if not isinstance(signals, list):
                signals = [signals]
            for signal in signals:
                self._handle_signal(signal, bid_candle)

    def evaluate_tick(self, tick: Tick):
        """
        Evaluates einen Tick für diese Strategie.
        """
        signal = self.strategy.evaluate_tick(tick)
        if signal:
            self.executor.process_signal_tick(signal, tick)
        self.executor.evaluate_exits_tick(tick)
        self.portfolio.update(tick.timestamp)

    def _handle_signal(self, signal, candle):
        """
        Verarbeitet das Signal (kann von Kindklassen überschrieben werden).
        """
        self.strategy.on_signal(signal, candle)


class MultiStrategyController:
    """
    Koordiniert mehrere Strategien auf Multi-Symbol/Multi-TF Daten.

    Args:
        strategy: StrategyWrapper oder Strategiemanager.
    """

    def __init__(self, strategy: StrategyWrapper):
        self.strategy = strategy
        self.multi_candle_data: Optional[MultiSymbolSlice] = None

    def set_multi_candle_data(self, data: MultiSymbolSlice):
        """
        Setzt das Multi-Candle-Datenobjekt für die Steuerung.
        """
        if not hasattr(data, "set_index"):
            raise TypeError("multi_candle_data ist kein Slice-Objekt mit set_index().")
        self.multi_candle_data = data

    def run_on_candles(
        self,
        bid_candles: List[Candle],
        ask_candles: List[Candle],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Läuft die Strategie über alle Candle-Paare.
        """
        for i, (bid, ask) in enumerate(zip(bid_candles, ask_candles)):
            if callable(on_progress):
                on_progress(i, len(bid_candles))
            self.evaluate_candle(i, bid, ask, self.multi_candle_data)

    def evaluate_candle(
        self,
        i: int,
        bid_candle: Candle,
        ask_candle: Candle,
        multi_slice: MultiSymbolSlice,
    ):
        """
        Evaluates einen Candle-Step für den MultiSlice-Kontext.
        """
        signals = self.strategy.evaluate(i, multi_slice)
        if signals:
            if not isinstance(signals, list):
                signals = [signals]
            for signal in signals:
                self._handle_signal(signal, bid_candle)

    def _handle_signal(self, signal, candle):
        """
        Verarbeitet das Signal (leitet weiter an die Strategie).
        """
        self.strategy.on_signal(signal, candle)
