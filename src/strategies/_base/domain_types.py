# src/strategies/_base/domain_types.py
"""
Shared Domain Types for Trading Strategies.

This module provides centralized type definitions used across all trading strategies,
both for live trading and backtesting. Using these shared types ensures consistency
and reduces code duplication.

Example usage:
    from strategies._base.domain_types import (
        CandleProtocol,
        PositionProtocol,
        MarketSignal,
        CandleDict,
        TradeSignalDict,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

# =============================================================================
# Protocols (Structural Subtyping)
# =============================================================================


@runtime_checkable
class CandleProtocol(Protocol):
    """
    Protocol for OHLC candle data.

    This protocol defines the minimal interface for candle objects used in backtesting.
    Any object with these attributes can be used wherever CandleProtocol is expected.

    Attributes:
        timestamp: Candle timestamp (datetime or any timestamp-like object).
        open: Opening price.
        high: Highest price during the period.
        low: Lowest price during the period.
        close: Closing price.

    Example:
        >>> class MyCandle:
        ...     def __init__(self, o, h, l, c, ts):
        ...         self.open, self.high, self.low, self.close = o, h, l, c
        ...         self.timestamp = ts
        >>> candle = MyCandle(1.1000, 1.1050, 1.0980, 1.1020, datetime.now())
        >>> isinstance(candle, CandleProtocol)  # True
    """

    timestamp: Any
    open: float
    high: float
    low: float
    close: float


@runtime_checkable
class PositionProtocol(Protocol):
    """
    Protocol for position data in backtesting.

    Defines the minimal interface for position objects used by position managers.

    Attributes:
        status: Position status (e.g., "open", "pending", "closed").
        direction: Trade direction ("buy" or "sell").
        entry_time: Time when the position was opened.
        trigger_time: Time when pending order was triggered.
        entry_price: Entry price of the position.
        stop_loss: Current stop loss price.
        initial_stop_loss: Original stop loss price (before adjustments).
        order_type: Order type (e.g., "market", "limit", "stop").
        is_closed: Whether the position is closed.
        reason: Reason for closing (if closed).
        exit_time: Time when position was closed.
        exit_price: Exit price of the position.
    """

    status: str
    direction: str
    entry_time: Any
    trigger_time: Any
    entry_price: float
    stop_loss: float
    initial_stop_loss: float
    order_type: str
    is_closed: bool
    reason: str
    exit_time: Any
    exit_price: float

    def close(self, timestamp: Any, exit_price: float, reason: str) -> None:
        """
        Close the position.

        Args:
            timestamp: Time of closing.
            exit_price: Price at which position was closed.
            reason: Reason for closing (e.g., "sl_hit", "tp_hit", "manual").
        """
        ...


@runtime_checkable
class PortfolioProtocol(Protocol):
    """
    Protocol for portfolio/account management in backtesting.

    Defines minimal interface for portfolio objects that position managers interact with.
    """

    def get_open_positions(self) -> list[Any]:
        """Return list of currently open positions."""
        ...


# =============================================================================
# TypedDicts (Structured Dictionaries)
# =============================================================================


class CandleDict(TypedDict, total=False):
    """
    TypedDict for candle data passed as dictionaries.

    Used when candle data is represented as a dict rather than an object.
    All fields are optional (total=False) to allow partial data.

    Attributes:
        open: Opening price.
        high: Highest price.
        low: Lowest price.
        close: Closing price.
        volume: Trading volume (optional).
        timestamp: Candle timestamp (optional).
    """

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Any


class MarketSignal(TypedDict):
    """
    TypedDict for market entry signals (required fields).

    Used for signals that trigger immediate market orders.

    Attributes:
        direction: Trade direction ("buy" or "sell").
        sl: Stop loss price.
        tp: Take profit price.
        order_type: Always "market" for this type.

    Example:
        >>> signal: MarketSignal = {
        ...     "direction": "buy",
        ...     "sl": 1.0950,
        ...     "tp": 1.1100,
        ...     "order_type": "market",
        ... }
    """

    direction: Literal["buy", "sell"]
    sl: float
    tp: float
    order_type: Literal["market"]


class TradeSignalDict(TypedDict, total=False):
    """
    TypedDict for trade signals with optional metadata.

    Extended signal format supporting various order types and metadata.

    Attributes:
        direction: Trade direction ("buy" or "sell").
        entry: Entry price (for limit/stop orders).
        closing: Alternative name for entry/closing price.
        sl: Stop loss price.
        tp: Take profit price.
        order_type: Type of order ("market", "limit", "stop").
        scenario: Scenario identifier string.
        indicators: Dictionary of indicator values at signal time.
        confidence: Signal confidence score (0.0 to 1.0).
        metadata: Additional arbitrary metadata.
    """

    direction: Literal["buy", "sell"]
    entry: float
    closing: float
    sl: float
    tp: float
    order_type: Literal["market", "limit", "stop"]
    scenario: str
    indicators: dict[str, Any]
    confidence: float
    metadata: dict[str, Any]


class SignalDict(TypedDict, total=False):
    """
    TypedDict for scenario evaluator signals.

    Standard return format for scenario evaluators in strategies.
    Compatible with base_scenarios.py SignalDict.

    Attributes:
        direction: Trade direction ("buy" or "sell").
        entry: Entry price.
        sl: Stop loss price.
        tp: Take profit price.
        order_type: Order type string.
        scenario: Scenario name/identifier.
    """

    direction: str
    entry: float
    sl: float
    tp: float
    order_type: str
    scenario: str


# =============================================================================
# Type Aliases
# =============================================================================

Direction = Literal["buy", "sell"]
"""Type alias for trade direction."""

OrderType = Literal["market", "limit", "stop"]
"""Type alias for order types."""

PositionStatus = Literal["open", "pending", "closed", "cancelled"]
"""Type alias for position status values."""

CloseReason = Literal[
    "sl_hit",
    "tp_hit",
    "breakeven_hit",
    "timeout",
    "manual",
    "partial_close",
    "pending_expired",
    "session_end",
]
"""Type alias for position close reasons."""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Protocols
    "CandleProtocol",
    "PositionProtocol",
    "PortfolioProtocol",
    # TypedDicts
    "CandleDict",
    "MarketSignal",
    "TradeSignalDict",
    "SignalDict",
    # Type Aliases
    "Direction",
    "OrderType",
    "PositionStatus",
    "CloseReason",
]
