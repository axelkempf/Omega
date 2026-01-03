# Metrics Layer

This directory contains logic for calculating performance metrics and analyzing trade results.

## Overview

The Metrics Layer provides tools to evaluate the performance of trading strategies based on historical trade logs. It calculates standard financial metrics used for reporting and optimization.

## Key Components

### `performance_metrics.py`
A library of functions to calculate metrics from a pandas DataFrame of trades.
- **Metrics**:
    - **Win Rate**: Percentage of profitable trades.
    - **Profit Factor**: Gross Profit / Gross Loss.
    - **Drawdown**: Maximum peak-to-valley decline in equity.
    - **Sharpe Ratio**: Risk-adjusted return.
    - **Expectancy**: Average profit per trade.
- **Input**: Expects a DataFrame matching the `trade_log.csv` schema.

## Usage

This layer is primarily used by the Backtest Engine and the UI Dashboard to display performance statistics.

```python
from hf_engine.infra.metrics.performance_metrics import load_trade_log, calculate_metrics

df = load_trade_log()
metrics = calculate_metrics(df)
print(f"Win Rate: {metrics['win_rate']:.2%}")
```
