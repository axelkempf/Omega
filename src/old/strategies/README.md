# Strategies

This directory houses the trading logic and algorithms. It provides a structured environment for developing, testing, and deploying trading strategies.

## Structure

| Directory | Purpose |
|-----------|---------|
| **`_base/`** | Contains the abstract base classes and interfaces that all strategies must implement. Defines the contract for `on_tick`, `on_bar`, and signal generation. |
| **`_template/`** | A clean starting point for new strategies. Copy this folder when creating a new strategy to ensure you have the correct structure and required files. |
| **`<strategy_name>/`** | Individual strategy implementations (e.g., `mean_reversion_z_score`). |

## Developing a New Strategy

1.  **Copy the Template**: Duplicate the `_template/` directory and rename it to your strategy name.
2.  **Implement Logic**: Edit `strategy.py` to implement your trading rules.
    - `on_init()`: Initialize indicators and state.
    - `on_bar()` / `on_tick()`: Process market data and generate signals.
3.  **Configuration**: Create a corresponding JSON configuration file in `configs/backtest/` or `configs/live/`.

## Best Practices

- **Statelessness**: Where possible, keep strategies stateless or explicitly manage state to ensure resume capabilities.
- **Indicator Separation**: Use the `indicators/` module within your strategy or shared libraries to keep the core logic clean.
- **Configurability**: Expose key parameters (e.g., lookback periods, thresholds) via the configuration file rather than hardcoding them.

> [!TIP]
> Use the `backtest_engine` to rigorously test your strategy before deploying it to the `hf_engine`. The strategy interface is designed to be compatible with both.
