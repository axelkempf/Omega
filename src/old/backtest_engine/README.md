# Backtest Engine

The `backtest_engine` is a robust, event-driven framework designed for simulating trading strategies against historical data. It supports single-run backtests, multi-stage optimization, and walk-forward analysis.

## Core Components

### 🏃 Runner & Execution
- **`runner.py`**: The primary entry point for running a single backtest based on a JSON configuration.
- **`batch_runner.py`**: Utilities for executing multiple backtests in batch.
- **`core/`**: Contains the core simulation logic, event loop, and engine definitions.

### 🧠 Optimization
- **`optimizer/`**: A multi-stage optimization subsystem.
  - `grid_searcher.py`: Combinatorial search for parameter exploration.
  - `optuna_optimizer.py`: Bayesian optimization using Optuna.
  - `walkforward.py`: Walk-forward validation to test strategy robustness over shifting time windows.
  - `robust_zone_analyzer.py`: Identifies stable parameter zones.

### 📊 Data & Analysis
- **`data/`**: Handles data loading, processing, and caching (CSV/Parquet).
- **`analysis/`**: Tools for analyzing backtest results, including metric adjustments and stability scoring.
- **`rating/`**: Scoring mechanisms to evaluate strategy performance (e.g., Sharpe, Sortino, custom scores).
- **`report/`**: Generates performance reports and visualizations.

## Usage

To run a single backtest:

```bash
python src/backtest_engine/runner.py configs/backtest/<your_config>.json
```

> [!TIP]
> The engine is designed to be deterministic. Ensure you set fixed seeds in your configuration if you require exact reproducibility.

## Key Features

- **Event-Driven**: Simulates realistic market conditions by processing events tick-by-tick or bar-by-bar.
- **No Lookahead Bias**: Strict separation of data to prevent future information leakage.
- **Performance**: Optimized for speed using efficient data structures and optional parallel processing in optimization stages.
