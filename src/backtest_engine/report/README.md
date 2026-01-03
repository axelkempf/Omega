# Report Module

The **Report Module** handles the calculation of performance metrics, result exportation, and visualization of backtest outcomes.

## Features

- **Metrics Calculation**: `metrics.py` computes a wide range of performance metrics (e.g., Win Rate, Profit Factor, Drawdown, Sharpe Ratio) from portfolio data.
- **Visualization**: `visualizer.py` and `overlay_plot.py` generate charts, including equity curves and trade overlays on price data.
- **Exporting**: `exporter.py` and `result_saver.py` handle saving results to CSV, JSON, and other formats for persistence and further analysis.

## Key Components

| File | Description |
|------|-------------|
| `metrics.py` | Calculates comprehensive performance metrics from a portfolio. |
| `visualizer.py` | Generates standard plots like equity curves. |
| `overlay_plot.py` | Creates advanced plots overlaying trades on price charts. |
| `exporter.py` | Exports backtest results to various file formats. |
| `result_saver.py` | Manages the saving of optimization and backtest results. |

## Usage

```python
from backtest_engine.report.metrics import calculate_metrics
from backtest_engine.report.visualizer import plot_equity_curve

# Calculate metrics
metrics = calculate_metrics(portfolio)

# Plot equity curve
plot_equity_curve(
    positions=portfolio.closed_positions,
    strategy_name="MyStrategy",
    symbol="EURUSD"
)
```
