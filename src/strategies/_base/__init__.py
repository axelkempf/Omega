# src/strategies/_base/__init__.py
"""
Base module for trading strategies.

Provides shared base classes, protocols, and type definitions used across
all trading strategies.

Exports:
    - Strategy: Abstract base class for all strategies
    - TradeSetup: Dataclass for trade setup definitions
    - BaseSzenario: Base class for scenario implementations
    - SignalDict: TypedDict for scenario signals
    - Domain Types: CandleProtocol, PositionProtocol, MarketSignal, etc.
"""

from strategies._base.base_scenarios import BaseSzenario, SignalDict
from strategies._base.base_strategy import Strategy, TradeSetup
from strategies._base.domain_types import (
    CandleDict,
    CandleProtocol,
    CloseReason,
    Direction,
    MarketSignal,
    OrderType,
    PortfolioProtocol,
    PositionProtocol,
    PositionStatus,
    TradeSignalDict,
)

__all__ = [
    # Base Classes
    "Strategy",
    "TradeSetup",
    "BaseSzenario",
    # TypedDicts
    "SignalDict",
    "CandleDict",
    "MarketSignal",
    "TradeSignalDict",
    # Protocols
    "CandleProtocol",
    "PositionProtocol",
    "PortfolioProtocol",
    # Type Aliases
    "Direction",
    "OrderType",
    "PositionStatus",
    "CloseReason",
]
