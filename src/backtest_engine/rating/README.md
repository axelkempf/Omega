# Rating Module

The **Rating Module** provides a comprehensive set of scoring metrics to evaluate strategy performance, robustness, and stability. These scores are used to rank strategies and determine their suitability for deployment.

## Features

- **Performance Scoring**: `strategy_rating.py` evaluates basic performance metrics against defined thresholds.
- **Robustness Metrics**:
    - `robustness_score_1.py`: General robustness score.
    - `stability_score.py`: Measures the stability of returns over time.
    - `ulcer_index_score.py`: Calculates the Ulcer Index, a measure of downside risk.
- **Stress Testing Scores**:
    - `cost_shock_score.py`: Evaluates performance under increased transaction costs.
    - `data_jitter_score.py`: Tests sensitivity to noise in market data.
    - `timing_jitter_score.py`: Tests sensitivity to slight changes in entry/exit timing.
    - `trade_dropout_score.py`: Simulates random trade failures.
    - `tp_sl_stress_score.py`: Tests sensitivity to changes in Take Profit and Stop Loss levels.
- **Statistical Significance**: `p_values.py` calculates p-values to assess the statistical significance of the results.

## Key Components

| File | Description |
|------|-------------|
| `strategy_rating.py` | Core function for rating strategy performance based on thresholds. |
| `stability_score.py` | Calculates stability metrics (e.g., R-squared of equity curve). |
| `stress_penalty.py` | Aggregates penalties from various stress tests. |
| `p_values.py` | Computes statistical significance of strategy performance. |

## Usage

These scoring functions are typically used within the `Analysis` or `Optimizer` modules to rank and filter strategies.

```python
from backtest_engine.rating.strategy_rating import rate_strategy_performance

rating = rate_strategy_performance(
    summary={
        "Winrate (%)": 55,
        "Avg R-Multiple": 1.5,
        "Net Profit": 1200,
        "profit_factor": 1.8,
        "drawdown_eur": 500
    }
)
print(f"Score: {rating['Score']}, Deploy: {rating['Deployment']}")
```
