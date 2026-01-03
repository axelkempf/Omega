# Strategy Manager

The **Strategy Manager** module handles the lifecycle management of trading strategies. It provides the core functionality to start, stop, and monitor the status of different trading strategies, ensuring a standardized way to interact with the underlying trading engines.

## Features

-   **Standardized Lifecycle**: Defines a common interface (`BaseStrategyManager`) for all strategy operations (`start`, `stop`, `status`).
-   **Factory-Based Instantiation**: Uses a factory pattern to create strategy managers, abstracting the creation logic.
-   **MT5 Integration**: Includes a robust implementation (`MT5StrategyManager`) for managing strategies running on MetaTrader 5.
-   **Config Integration**: Automatically resolves configuration paths based on strategy IDs.

## Architecture

-   `base.py`: Defines the `BaseStrategyManager` abstract base class.
-   `factory.py`: Provides `get_strategy_manager` to instantiate managers.
-   `mt5_manager.py`: Concrete implementation for MT5 strategies, interfacing with the main controller.

## Usage

To manage a strategy, obtain an instance via the factory:

```python
from ui_engine.strategies.factory import get_strategy_manager

# Get a manager for a specific account alias
strategy = get_strategy_manager("account_10928521")

# Start the strategy
# This will look for the config file: strategy_config_10928521.json
if strategy.start():
    print("Strategy started successfully")

# Check current status
current_status = strategy.status()
print(f"Status: {current_status}")

# Stop the strategy
strategy.stop()
```

> [!IMPORTANT]
> The `MT5StrategyManager` relies on the `LIVE_CONFIG_DIR` setting to locate configuration files. Ensure that `strategy_config_{id}.json` exists in the configured directory for the strategy to start.
