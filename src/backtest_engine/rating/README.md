# Rating Module

The **Rating Module** provides a comprehensive set of scoring metrics to evaluate strategy performance, robustness, and stability. These scores are used to rank strategies and determine their suitability for deployment.

## Migration Status

This module is being prepared for **Wave 1 of the Rust/Julia Migration**. The individual rating functions are pure mathematical computations ideal for high-performance Rust implementations.

**Note:** The `strategy_rating.py` file has been removed as part of migration preparation. The `rate_strategy_performance` functionality has been moved inline to the consuming modules (`walkforward.py`) to reduce dependencies before Rust migration.

## Features

- **Robustness Metrics**:
    - `robustness_score_1.py`: General robustness score based on parameter jitter.
    - `stability_score.py`: Measures the stability of returns over time (yearly profits).
    - `ulcer_index_score.py`: Calculates the Ulcer Index, a measure of downside risk.
- **Stress Testing Scores**:
    - `cost_shock_score.py`: Evaluates performance under increased transaction costs.
    - `data_jitter_score.py`: Tests sensitivity to noise in market data.
    - `timing_jitter_score.py`: Tests sensitivity to slight changes in entry/exit timing.
    - `trade_dropout_score.py`: Simulates random trade failures.
    - `tp_sl_stress_score.py`: Tests sensitivity to changes in Take Profit and Stop Loss levels.
    - `stress_penalty.py`: Shared penalty computation logic used by stress tests.
- **Statistical Significance**: `p_values.py` calculates p-values to assess the statistical significance of the results.

## Key Components

| File | Description |
|------|-------------|
| `robustness_score_1.py` | Robustness score based on parameter jitter tolerance. |
| `stability_score.py` | Calculates stability metrics based on yearly profit distribution. |
| `stress_penalty.py` | Aggregates penalties from various stress tests. |
| `cost_shock_score.py` | Score for cost shock sensitivity. |
| `data_jitter_score.py` | Score for data noise sensitivity. |
| `timing_jitter_score.py` | Score for timing shift sensitivity. |
| `trade_dropout_score.py` | Score for trade dropout simulation. |
| `tp_sl_stress_score.py` | Score for TP/SL stress testing. |
| `ulcer_index_score.py` | Ulcer Index calculation and scoring. |
| `p_values.py` | Computes statistical significance of strategy performance. |

## Usage

These scoring functions are typically used within the `Analysis` or `Optimizer` modules to rank and filter strategies.

```python
from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import compute_stability_score_from_yearly_profits
from backtest_engine.rating.cost_shock_score import compute_multi_factor_cost_shock_score

# Example: Robustness score
base_metrics = {"profit": 1000, "avg_r": 0.8, "winrate": 55, "drawdown": 200}
jitter_metrics = [
    {"profit": 950, "avg_r": 0.75, "winrate": 53, "drawdown": 220},
    {"profit": 1020, "avg_r": 0.82, "winrate": 56, "drawdown": 190},
]
score = compute_robustness_score_1(base_metrics, jitter_metrics)
print(f"Robustness Score: {score}")

# Example: Stability score
profits_by_year = {2020: 4000, 2021: 3500, 2022: 4200, 2023: 3800}
stability_score = compute_stability_score_from_yearly_profits(profits_by_year)
print(f"Stability Score: {stability_score}")
```

## FFI Migration Documentation

For detailed FFI specifications and migration runbooks, see:
- `docs/ffi/rating_modules.md` - FFI interface specifications
- `docs/runbooks/rating_modules_migration.md` - Migration runbook
