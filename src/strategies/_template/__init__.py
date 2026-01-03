"""Strategy Template Package.

This package provides a fully-typed strategy template that serves as a
reference implementation for new trading strategies. All modules follow
the project's typing conventions documented in docs/typing.md.

Usage:
    To create a new strategy:
    1. Copy the _template directory to strategies/your_strategy_name/
    2. Rename classes and update docstrings
    3. Implement your trading logic
    4. Verify typing with: mypy src/strategies/your_strategy_name/ --strict

Exports:
    StrategyTemplate: Example strategy class with full type hints.
    MarketSignal: TypedDict for market order signals.
    CandleProtocol: Protocol for OHLC candle data.
    DataSliceProtocol: Protocol for multi-timeframe data slices.
"""

from __future__ import annotations

from strategies._template.strategy_template import (
    CandleProtocol,
    DataSliceProtocol,
    MarketSignal,
    StrategyTemplate,
)

__all__ = [
    "StrategyTemplate",
    "MarketSignal",
    "CandleProtocol",
    "DataSliceProtocol",
]
