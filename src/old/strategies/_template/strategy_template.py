"""Strategy Template with Full Type Hints.

This module provides a fully-typed strategy template that can be used
as a starting point for new trading strategies. All methods include
complete type annotations following the project's typing conventions.

Usage:
    1. Copy this file to your strategy directory
    2. Rename the class to match your strategy name
    3. Implement your trading logic in on_data()
    4. Run mypy to verify type correctness

See Also:
    - docs/typing.md for typing conventions
    - strategies/_base/domain_types.py for shared types
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

# Import shared types from the central domain_types module
from strategies._base.domain_types import Direction

if TYPE_CHECKING:
    # Heavy imports only for type checking to avoid circular dependencies
    pass


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------


class CandleProtocol(Protocol):
    """Protocol for OHLC candle data.

    Any object that has these properties can be used as a candle,
    regardless of its actual class (duck typing).
    """

    @property
    def open(self) -> float:
        """Opening price."""
        ...

    @property
    def high(self) -> float:
        """Highest price."""
        ...

    @property
    def low(self) -> float:
        """Lowest price."""
        ...

    @property
    def close(self) -> float:
        """Closing price."""
        ...


class DataSliceProtocol(Protocol):
    """Protocol for multi-timeframe data slice."""

    def latest(self, timeframe: str) -> CandleProtocol | None:
        """Get the latest candle for a timeframe.

        Args:
            timeframe: Timeframe string (e.g., "M5", "H1").

        Returns:
            Latest candle or None if not available.
        """
        ...


class MarketSignal(TypedDict, total=False):
    """TypedDict for market order signals.

    Required fields: direction, entry, sl, tp, reason
    Optional fields: type, tags
    """

    direction: Direction
    entry: float
    sl: float
    tp: float
    reason: str
    type: str  # "market", "limit", "stop"
    tags: list[str]


# ---------------------------------------------------------------------------
# Strategy Implementation
# ---------------------------------------------------------------------------


class StrategyTemplate:
    """Universal strategy template for hf_engine.

    This template demonstrates the recommended structure for trading
    strategies with full type annotations. It uses on_data() with
    a SymbolDataSlice for multi-timeframe analysis.

    Attributes:
        primary_tf: Primary timeframe for signal generation.
        confirmation_tf: Higher timeframe for trend confirmation.
        min_body_pips: Minimum candle body size in pips.
        tp_factor: Take profit multiplier (risk:reward ratio).
        sl_pips: Stop loss distance in pips.

    Example:
        >>> strategy = StrategyTemplate(primary_tf="M5", tp_factor=2.0)
        >>> signal = strategy.on_data(data_slice)
        >>> if signal:
        ...     print(f"Signal: {signal['direction']} at {signal['entry']}")
    """

    # Class-level type annotations for attributes
    primary_tf: str
    confirmation_tf: str
    min_body_pips: float
    tp_factor: float
    sl_pips: float

    def __init__(
        self,
        primary_tf: str = "M5",
        confirmation_tf: str = "H1",
        min_body_pips: float = 3.0,
        tp_factor: float = 2.0,
        sl_pips: float = 5.0,
    ) -> None:
        """Initialize the strategy with parameters.

        Args:
            primary_tf: Primary timeframe for signal generation.
            confirmation_tf: Higher timeframe for trend confirmation.
            min_body_pips: Minimum candle body size in pips.
            tp_factor: Take profit multiplier (risk:reward ratio).
            sl_pips: Stop loss distance in pips.
        """
        self.primary_tf = primary_tf
        self.confirmation_tf = confirmation_tf
        self.min_body_pips = min_body_pips
        self.tp_factor = tp_factor
        self.sl_pips = sl_pips

    def on_data(self, slice: DataSliceProtocol) -> MarketSignal | None:
        """Process new market data and generate trading signal.

        This is the main entry point called by the engine when new
        data arrives. Override this method to implement your strategy logic.

        Args:
            slice: Multi-timeframe data slice with latest candles.

        Returns:
            MarketSignal dict if a trade should be placed, None otherwise.
        """
        # Load current candles for each timeframe
        candle = slice.latest(self.primary_tf)
        confirm = slice.latest(self.confirmation_tf)

        if candle is None or confirm is None:
            return None  # Incomplete data slice

        # Calculate body size in pips
        body_pips = self._calculate_body_pips(candle)

        # Check for bullish setup with confirmation
        if self._is_bullish_setup(candle, confirm, body_pips):
            return self._create_long_signal(candle, body_pips)

        # Check for bearish setup with confirmation
        if self._is_bearish_setup(candle, confirm, body_pips):
            return self._create_short_signal(candle, body_pips)

        return None

    def _calculate_body_pips(self, candle: CandleProtocol) -> float:
        """Calculate candle body size in pips.

        Args:
            candle: OHLC candle data.

        Returns:
            Body size in pips (for forex pairs with 4/5 decimal places).
        """
        return abs(candle.close - candle.open) * 10_000

    def _is_bullish_setup(
        self,
        candle: CandleProtocol,
        confirm: CandleProtocol,
        body_pips: float,
    ) -> bool:
        """Check if conditions for a bullish trade are met.

        Args:
            candle: Primary timeframe candle.
            confirm: Confirmation timeframe candle.
            body_pips: Body size in pips.

        Returns:
            True if bullish setup conditions are met.
        """
        return (
            candle.close > candle.open
            and confirm.close > confirm.open
            and body_pips >= self.min_body_pips
        )

    def _is_bearish_setup(
        self,
        candle: CandleProtocol,
        confirm: CandleProtocol,
        body_pips: float,
    ) -> bool:
        """Check if conditions for a bearish trade are met.

        Args:
            candle: Primary timeframe candle.
            confirm: Confirmation timeframe candle.
            body_pips: Body size in pips.

        Returns:
            True if bearish setup conditions are met.
        """
        return (
            candle.close < candle.open
            and confirm.close < confirm.open
            and body_pips >= self.min_body_pips
        )

    def _create_long_signal(
        self,
        candle: CandleProtocol,
        body_pips: float,
    ) -> MarketSignal:
        """Create a long (buy) market signal.

        Args:
            candle: Candle data for entry calculation.
            body_pips: Body size for reason string.

        Returns:
            MarketSignal dict for a long position.
        """
        entry = candle.close
        sl = entry - self.sl_pips * 0.0001
        tp = entry + (entry - sl) * self.tp_factor

        return MarketSignal(
            direction="buy",
            entry=entry,
            sl=sl,
            tp=tp,
            reason=f"Bullish body {body_pips:.1f} pips + H1 confirmation",
            type="market",
            tags=["bull", "confirmation", "momentum"],
        )

    def _create_short_signal(
        self,
        candle: CandleProtocol,
        body_pips: float,
    ) -> MarketSignal:
        """Create a short (sell) market signal.

        Args:
            candle: Candle data for entry calculation.
            body_pips: Body size for reason string.

        Returns:
            MarketSignal dict for a short position.
        """
        entry = candle.close
        sl = entry + self.sl_pips * 0.0001
        tp = entry - (sl - entry) * self.tp_factor

        return MarketSignal(
            direction="sell",
            entry=entry,
            sl=sl,
            tp=tp,
            reason=f"Bearish body {body_pips:.1f} pips + H1 confirmation",
            type="market",
            tags=["bear", "confirmation", "momentum"],
        )


# ---------------------------------------------------------------------------
# Signal Tags Reference
# ---------------------------------------------------------------------------
#
# Use consistent tags for filtering and analysis:
#
# | Concept      | Tags                                                 |
# | ------------ | ---------------------------------------------------- |
# | Rejection    | "rejection", "wick", "pinbar"                        |
# | Breakout     | "breakout", "range", "structure"                     |
# | Momentum     | "momentum", "trend", "impulse"                       |
# | Pairs/Spread | "spread", "hedge", "zscore", "cointegration"         |
# | Confirmation | "confirmation", "multi_tf"                           |
# | Time-based   | "london", "ny", "asia"                               |
# | Order Type   | "market", "limit", "stop"                            |
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Example: Limit Order Signal
# ---------------------------------------------------------------------------
#
# For strategies that use limit orders (e.g., wick rejection):
#
# def _create_limit_long_signal(
#     self,
#     candle: CandleProtocol,
# ) -> MarketSignal:
#     """Create a limit order for wick rejection entry."""
#     return MarketSignal(
#         direction="buy",
#         entry=candle.low + 0.0010,     # Entry level as limit
#         sl=candle.low - 0.0015,
#         tp=candle.high + 0.0030,
#         type="limit",
#         reason="Wick entry near support",
#         tags=["limit", "rejection"],
#     )
#
# ---------------------------------------------------------------------------
