# ui_engine/strategies/mt5_manager.py

import os

from ui_engine.config import LIVE_CONFIG_DIR
from ui_engine.controller import get_status, start_strategy, stop_strategy
from ui_engine.strategies.base import BaseStrategyManager


class MT5StrategyManager(BaseStrategyManager):
    def start(self) -> bool:
        config_path = os.path.join(
            LIVE_CONFIG_DIR, f"strategy_config_{self.resolved_id}.json"
        )
        return start_strategy(self.resolved_id, config_path)

    def stop(self) -> bool:
        return stop_strategy(self.resolved_id)

    def status(self) -> str:
        return get_status(self.resolved_id)
