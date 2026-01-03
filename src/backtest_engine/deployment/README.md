# Deployment Module

The **Deployment Module** assists in the transition from backtesting to live trading by selecting the best-performing strategies based on their ratings.

## Features

- **Strategy Selection**: `deployment_selector.py` aggregates ratings from multiple symbols and timeframes to recommend strategies for deployment.
- **Filtering**: Allows filtering based on minimum scores and successful time windows to ensure only robust strategies are selected.

## Key Components

| File | Description |
|------|-------------|
| `deployment_selector.py` | Analyzes rating files to select the best strategies for deployment. |

## Usage

```python
from backtest_engine.deployment.deployment_selector import select_best_strategies

recommendations = select_best_strategies(
    rating_files=["var/results/EURUSD_ratings.json", "var/results/GBPUSD_ratings.json"],
    min_score=1.5,
    min_windows=4
)
```
