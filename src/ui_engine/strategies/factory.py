# ui_engine/strategies/factory.py

from ui_engine.registry.strategy_alias import resolve_alias
from ui_engine.strategies.base import BaseStrategyManager
from ui_engine.strategies.mt5_manager import MT5StrategyManager


def get_strategy_manager(alias: str) -> BaseStrategyManager:
    resolved = resolve_alias(alias)
    return MT5StrategyManager(alias, resolved)
