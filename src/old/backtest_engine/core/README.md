# Core Module

The **Core Module** contains the fundamental building blocks of the backtest engine. It handles the event loop, trade execution simulation, portfolio management, and data slicing for strategies.

## Features

- **Event Engine**: `event_engine.py` and `tick_event_engine.py` drive the backtest simulation, processing market data candle-by-candle or tick-by-tick.
- **Execution Simulation**: `execution_simulator.py` simulates order execution, including slippage and fees (`slippage_and_fee.py`).
- **Portfolio Management**: `portfolio.py` tracks positions, equity, and trade history.
- **Data Slicing**: `symbol_data_slicer.py` and `multi_symbol_slice.py` provide strategies with the correct view of market data at each time step, preventing lookahead bias.
- **Indicator Caching**: `indicator_cache.py` optimizes performance by caching calculated indicators.

## Key Components

| File | Description |
|------|-------------|
| `event_engine.py` | Main event loop for single-symbol backtests. |
| `execution_simulator.py` | Simulates order fills based on price action. |
| `portfolio.py` | Manages account state, open positions, and closed trades. |
| `symbol_data_slicer.py` | Provides a window of past data to the strategy at each step. |
| `multi_strategy_controller.py` | Orchestrates backtests involving multiple strategies or symbols. |

## Architecture

The core follows an event-driven architecture where the `EventEngine` iterates through time, updating the `SymbolDataSlice` and invoking the strategy. The strategy generates signals, which are processed by the `ExecutionSimulator` and updated in the `Portfolio`.
