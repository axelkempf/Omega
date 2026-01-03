# Optimizer Module

The **Optimizer Module** provides advanced tools for strategy parameter optimization, including grid search, Bayesian optimization (Optuna), and rigorous walk-forward validation.

## Features

- **Bayesian Optimization**: `optuna_optimizer.py` leverages Optuna for efficient parameter search using TPE or NSGA-II samplers.
- **Grid Search**: `grid_searcher.py` performs exhaustive or randomized grid searches for parameter exploration.
- **Walk-Forward Optimization**: `walkforward.py` implements the walk-forward validation method to test strategy robustness over shifting time windows.
- **Robustness Analysis**: `robust_zone_analyzer.py` identifies stable parameter zones, reducing the risk of overfitting.
- **Final Selection**: `final_param_selector.py` applies stress tests (dropout, cost shock) to select the most robust parameters for deployment.

## Key Components

| File | Description |
|------|-------------|
| `optuna_optimizer.py` | Optuna-based optimizer for single and multi-objective optimization. |
| `walkforward.py` | Orchestrates the walk-forward optimization process (Train/Test splits). |
| `grid_searcher.py` | Traditional grid search implementation. |
| `robust_zone_analyzer.py` | Analyzes parameter space to find clusters of stable performance. |
| `final_param_selector.py` | Selects final parameters based on stability and stress testing. |

## Usage

The optimizer is typically invoked via the `backtest_engine` runner or specific scripts.

### Example: Optuna Optimization

```python
from backtest_engine.optimizer.optuna_optimizer import run_optuna_optimization

study = run_optuna_optimization(
    strategy_class=MyStrategy,
    config=config,
    n_trials=100
)
```
