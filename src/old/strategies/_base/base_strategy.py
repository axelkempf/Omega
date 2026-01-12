"""
Base Strategy Module.

This module provides the abstract base class for all trading strategies
and the TradeSetup dataclass for standardized trade setup representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import TYPE_CHECKING, Any, Literal

from strategies._base.domain_types import Direction, OrderType

if TYPE_CHECKING:
    from typing import Protocol

    class BrokerProtocol(Protocol):
        """Protocol for broker interface (used for type hints only)."""

        def get_margin_info(self, symbol: str) -> dict[str, Any]: ...

    class DataProviderProtocol(Protocol):
        """Protocol for data provider interface (used for type hints only)."""

        def get_candles(
            self, symbol: str, timeframe: str, count: int
        ) -> list[dict[str, Any]]: ...


class SessionTimes:
    """Type for session time configuration."""

    start: time | None
    end: time | None


@dataclass
class TradeSetup:
    """
    Standard data container for a planned trade setup.

    Contains all order, risk, and strategy metadata required to execute
    and track a trade throughout its lifecycle.

    Attributes:
        symbol: Trading instrument symbol (e.g., "EURUSD").
        direction: Trade direction ("buy" or "sell").
        start_capital: Account balance at trade creation.
        risk_pct: Risk percentage per trade (e.g., 0.02 for 2%).
        entry: Entry price for the trade.
        sl: Stop loss distance in points.
        tp: Take profit price.
        strategy: Strategy name identifier.
        strategy_module: Python module path for strategy.
        confidence: Signal confidence score (0.0 to 1.0).
        metadata: Additional strategy-specific metadata.
        magic_number: MT5 magic number for order identification.
        order_type: Order type ("market", "stop", "limit").
        be_trigger_atr: Break-even trigger in ATR multiples.
        be_trigger_r_multiple: Break-even trigger in R-multiples.
        be_buffer_pips: Buffer for break-even adjustment in pips.
        partial_close_trigger_r_multiple: R-multiple to trigger partial close.
        partial_close_volume_pct: Volume percentage to close partially.
        atr_timeframe: Timeframe for ATR calculation.
        atr_period: Period for ATR calculation.
        auto_close_time: Time to auto-close position (hours).
        session_end: Session end time for auto-close.
        r_multiple_goal: Target R-multiple string descriptor.
        session_times: Dictionary with session start/end times.
        entry_time: Timestamp when entry signal was generated.

    Example:
        >>> setup = TradeSetup(
        ...     symbol="EURUSD",
        ...     direction="buy",
        ...     start_capital=10000.0,
        ...     risk_pct=0.02,
        ...     entry=1.1000,
        ...     sl=50,
        ...     tp=1.1100,
        ...     strategy="mean_reversion_z_score",
        ...     strategy_module="mean_reversion_z_score.live",
        ... )
    """

    # Required fields
    symbol: str
    direction: Direction
    start_capital: float
    risk_pct: float
    entry: float
    sl: float  # Stop loss in points
    tp: float
    strategy: str
    strategy_module: str

    # Optional fields with defaults
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Extended strategy functionality (optional)
    magic_number: int | None = None
    order_type: OrderType = "stop"
    be_trigger_atr: float | None = None
    be_trigger_r_multiple: float | None = None
    be_buffer_pips: float | None = None
    partial_close_trigger_r_multiple: float | None = None
    partial_close_volume_pct: float | None = None
    atr_timeframe: str | None = None
    atr_period: float | None = None
    auto_close_time: float | None = None
    session_end: float | None = None
    r_multiple_goal: str | None = None
    session_times: dict[str, Any] | None = None
    entry_time: datetime | None = None

    def __post_init__(self) -> None:
        """Validate trade setup after initialization."""
        if self.direction not in ("buy", "sell"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.risk_pct <= 0 or self.risk_pct > 1:
            raise ValueError(f"Invalid risk_pct: {self.risk_pct} (must be 0 < x <= 1)")


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Defines the minimum interface requirements for strategies including
    name identification and signal generation.

    Attributes:
        config: Strategy configuration dictionary.

    Example:
        >>> class MyStrategy(Strategy):
        ...     def name(self) -> str:
        ...         return "my_strategy"
        ...
        ...     def generate_signal(
        ...         self, symbol: str, date: datetime, broker=None, data_provider=None
        ...     ) -> list[TradeSetup]:
        ...         return []
    """

    config: dict[str, Any]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the strategy with optional configuration.

        Args:
            config: Configuration dictionary for the strategy.
                    Defaults to empty dict if not provided.
        """
        self.config = config or {}

    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the strategy.

        Used for logging, evaluation, and identification purposes.

        Returns:
            Strategy name as string.
        """
        ...

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        date: datetime,
        broker: BrokerProtocol | None = None,
        data_provider: DataProviderProtocol | None = None,
    ) -> list[TradeSetup]:
        """
        Generate potential entry setups for a given symbol and timestamp.

        Args:
            symbol: Trading symbol to generate signals for (e.g., "EURUSD").
            date: Timestamp/bar time for signal evaluation.
            broker: Optional broker object for margin/position checks.
            data_provider: Optional market data provider for historical data.

        Returns:
            List of TradeSetup objects representing potential trades.
            Empty list if no valid signals are detected.
        """
        ...
