"""
Base Position Manager Module.

This module provides the abstract base class for all position managers
and a factory function to dynamically load strategy-specific managers.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from strategies._base.base_strategy import TradeSetup

if TYPE_CHECKING:
    from typing import Protocol

    class BrokerProtocol(Protocol):
        """Protocol for broker interface (type hints only)."""

        def get_position(self, ticket: int) -> dict[str, Any] | None: ...

        def close_position(self, ticket: int) -> bool: ...

        def modify_sl(self, ticket: int, new_sl: float) -> bool: ...

    class DataProviderProtocol(Protocol):
        """Protocol for data provider interface (type hints only)."""

        def get_current_price(self, symbol: str) -> dict[str, float]: ...

        def get_candles(
            self, symbol: str, timeframe: str, count: int
        ) -> list[dict[str, Any]]: ...


class BasePositionManager(ABC):
    """
    Abstract base class for position managers in step-based monitoring mode.

    Subclasses MUST implement `monitor_step() -> bool`.
    Optionally, `stop_monitoring()` and/or `cancel()` can be overridden.

    Attributes:
        setup: Trade setup configuration containing entry, SL, TP, etc.
        broker: Broker interface for order management.
        data_provider: Market data provider for price/candle data.
        symbol: Trading symbol being managed.
        strategy_name: Name of the strategy for logging/identification.

    Example:
        >>> class MyPositionManager(BasePositionManager):
        ...     def monitor_step(self) -> bool:
        ...         # Check position status, adjust SL/TP, etc.
        ...         return False  # Still monitoring
    """

    setup: TradeSetup | Any
    broker: BrokerProtocol | Any
    data_provider: DataProviderProtocol | Any
    symbol: str
    strategy_name: str

    def __init__(
        self,
        setup: TradeSetup | Any,
        broker: BrokerProtocol | Any,
        data_provider: DataProviderProtocol | Any,
    ) -> None:
        """
        Initialize the position manager.

        Args:
            setup: Strategy configuration/setup (TradeSetup or compatible object).
            broker: Broker interface for order management.
            data_provider: Market data provider for price/candle data.
        """
        self.setup = setup
        self.broker = broker
        self.data_provider = data_provider

        # Defensively extract symbol and strategy name
        self.symbol = self._extract_symbol(setup)
        self.strategy_name = self._extract_strategy_name(setup)

    def _extract_symbol(self, setup: TradeSetup | Any) -> str:
        """
        Extract symbol from setup object.

        Args:
            setup: Trade setup object.

        Returns:
            Symbol string, or empty string if not found.
        """
        if hasattr(setup, "symbol"):
            return str(setup.symbol) if setup.symbol else ""

        metadata = getattr(setup, "metadata", {})
        if isinstance(metadata, dict):
            return str(metadata.get("symbol", ""))

        return ""

    def _extract_strategy_name(self, setup: TradeSetup | Any) -> str:
        """
        Extract strategy name from setup object.

        Args:
            setup: Trade setup object.

        Returns:
            Strategy name string, or "unknown" if not found.
        """
        if hasattr(setup, "strategy"):
            return str(setup.strategy) if setup.strategy else "unknown"

        metadata = getattr(setup, "metadata", {})
        if isinstance(metadata, dict):
            return str(metadata.get("strategy", "unknown"))

        return "unknown"

    @abstractmethod
    def monitor_step(self) -> bool:
        """
        Execute exactly one monitoring step.

        This method should check position status, adjust stop loss/take profit,
        handle break-even moves, partial closes, etc.

        Returns:
            True if monitoring is complete (position can be removed from manager).
            False if still actively monitoring.
        """
        ...

    def stop_monitoring(self) -> None:
        """
        Optional graceful shutdown hook.

        Override in subclasses to perform cleanup when monitoring is stopped
        externally (e.g., system shutdown, strategy switch).
        """
        pass

    def cancel(self) -> None:
        """
        Optional alias/hook for cancellation.

        Override in subclasses for specific cancellation logic.
        Default implementation calls stop_monitoring().
        """
        self.stop_monitoring()


def strategy_position_manager_factory(
    setup: TradeSetup | Any,
    broker: BrokerProtocol | Any,
    data_provider: DataProviderProtocol | Any,
) -> BasePositionManager:
    """
    Factory function to load and instantiate the appropriate position manager.

    Dynamically imports the position manager class from the strategy module
    specified in the setup's `strategy_module` attribute.

    Args:
        setup: Strategy configuration/setup with `strategy_module` attribute.
        broker: Broker interface for order management.
        data_provider: Market data provider.

    Returns:
        Instantiated position manager for the strategy.

    Raises:
        ImportError: If the module or StrategyPositionManager class cannot be found.
        AttributeError: If setup lacks required `strategy_module` attribute.

    Example:
        >>> setup = TradeSetup(strategy_module="ema_rsi_trend_follow.live", ...)
        >>> manager = strategy_position_manager_factory(setup, broker, data_provider)
    """
    strategy_module: str = setup.strategy_module
    module_path = f"strategies.{strategy_module}.position_manager"

    try:
        module = importlib.import_module(module_path)
        manager_class: type[BasePositionManager] = getattr(
            module, "StrategyPositionManager"
        )
        return manager_class(setup, broker, data_provider)
    except ImportError as e:
        raise ImportError(
            f"❌ Position Manager module not found for {strategy_module}: {e}"
        ) from e
    except AttributeError as e:
        raise ImportError(
            f"❌ StrategyPositionManager class not found in {module_path}: {e}"
        ) from e
