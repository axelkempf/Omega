# Analysis Module

The **Analysis Module** provides a suite of tools for evaluating backtest results, with a strong focus on walk-forward analysis, robustness metrics, and equity curve visualization. It is designed to help traders and researchers validate strategy performance beyond simple backtesting.

## Features

- **Walk-Forward Analysis**: `walkforward_analyzer.py` processes optimization results to determine strategy stability over time.
- **Equity Curve Combination**: Tools like `combine_equity_curves.py` and `final_combo_equity_plotter.py` allow for aggregating and visualizing performance across multiple periods or strategies.
- **Metric Adjustments**: `metric_adjustments.py` implements advanced statistical adjustments (e.g., Wilson score, shrinkage) to provide more realistic performance estimates.
- **Backfill & Validation**: Utilities to backfill equity curves and validate walk-forward matrices.

## Key Components

| File | Description |
|------|-------------|
| `walkforward_analyzer.py` | Core engine for analyzing walk-forward optimization results and generating combined scores. |
| `metric_adjustments.py` | Statistical functions for risk-adjusted and shrinkage-adjusted metrics. |
| `combine_equity_curves.py` | Merges multiple equity curves into a single continuous performance view. |
| `final_combo_equity_plotter.py` | Visualizes the final combined equity curves for analysis. |

## Usage

Most tools in this module are designed to be run as standalone scripts or imported by the `backtest_engine`.

### Example: Walk-Forward Analysis

```bash
python -m backtest_engine.analysis.walkforward_analyzer --input var/results/optimization_run
```

> [!NOTE]
> Ensure that your optimization results follow the expected directory structure before running the analyzer.
