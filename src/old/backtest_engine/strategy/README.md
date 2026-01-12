# Strategy Module

The **Strategy Module** defines the core interfaces and base classes for implementing trading strategies within the backtest engine. It provides the structure for signal generation, session management, and strategy validation.

## Features

- **Strategy Interface**: `strategy_wrapper.py` defines the `StrategyWrapper` class, which serves as the base for all strategies. It handles data access, signal generation, and interaction with the portfolio.
- **Session Management**: `session_filter.py` and `session_time_utils.py` provide tools for filtering trades based on trading sessions (e.g., London, New York).
- **Validation**: `validators.py` ensures that strategy configurations and signals meet required constraints.

## Key Components

| File | Description |
|------|-------------|
| `strategy_wrapper.py` | Base class for all strategies. Defines `on_candle`, `on_tick`, and signal generation methods. |
| `session_filter.py` | Filters trading signals based on defined time sessions. |
| `session_time_utils.py` | Utilities for parsing and handling session time strings. |
| `validators.py` | Validates strategy parameters and signals. |

## Implementing a Strategy

To create a new strategy, inherit from `StrategyWrapper` and implement the `on_candle` method:

```python
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper, TradeSignal

class MyStrategy(StrategyWrapper):
    def on_candle(self, slice: SymbolDataSlice) -> List[TradeSignal]:
        # Your logic here
        if self.should_buy():
            return [TradeSignal(direction="long", ...)]
        return []
```
