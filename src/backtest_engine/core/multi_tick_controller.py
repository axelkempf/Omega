from typing import Any, Callable, Dict, List, Optional

from backtest_engine.core.multi_strategy_controller import StrategyEnvironment
from backtest_engine.core.tick_event_engine import TickEventEngine
from backtest_engine.data.tick import Tick


class MultiTickController:
    """
    Steuert die Ausführung mehrerer Tick-Strategie-Environments
    auf unterschiedlichen Symbolen.

    Args:
        envs: Liste der StrategyEnvironments (müssen Symbol zugeordnet haben)
        tick_data_map: Mapping {symbol: [Tick, ...]}
        multi_candle_data: Multi-TF candle data {TF: {"bid": [...], "ask": [...]}}
        on_progress: Optionaler Fortschritts-Callback
    """

    def __init__(
        self,
        envs: List[StrategyEnvironment],
        tick_data_map: Dict[str, List[Tick]],
        multi_candle_data: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        self.envs = envs
        self.tick_data_map = tick_data_map
        self.multi_candle_data: Dict[str, Dict[str, List[Any]]] = multi_candle_data or {}
        self.on_progress = on_progress

    def run(self) -> None:
        """
        Führt für jede Strategieumgebung den Tick-Backtest durch.
        """
        for env in self.envs:
            symbol = getattr(env.strategy, "symbol", None)
            if not symbol:
                raise ValueError(f"❌ Strategy '{env.name}' hat kein .symbol gesetzt.")

            ticks = self.tick_data_map.get(symbol)
            if not ticks:
                print(f"⚠️ Keine Tickdaten für Symbol: {symbol}")
                continue

            print(
                f"▶️ Starte Tick-Backtest für {env.name} auf {symbol} ({len(ticks)} Ticks)"
            )
            engine = TickEventEngine(
                ticks=ticks,
                strategy=env.strategy,
                executor=env.executor,
                portfolio=env.portfolio,
                multi_candle_data=self.multi_candle_data,
                on_progress=self.on_progress,
            )
            engine.original_start_dt = ticks[0].timestamp
            engine.run()
