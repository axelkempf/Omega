# Datafeeds Manager

The **Datafeeds Manager** module is responsible for managing the lifecycle and connectivity of various data feed providers within the UI Engine. It provides a unified interface to start, stop, and monitor data feeds, abstracting the underlying implementation details of specific providers like MetaTrader 5 (MT5) or DxFeed.

## Features

-   **Unified Interface**: All data feed managers inherit from `BaseDatafeedManager`, ensuring a consistent API for `start`, `stop`, and `status` operations.
-   **Factory Pattern**: The `get_datafeed_manager` factory function simplifies the instantiation of the correct manager based on the provided alias.
-   **Multi-Provider Support**:
    -   **MT5**: Manages data feeds sourced from MetaTrader 5.
    -   **DxFeed**: Support for DxFeed integration (via `DXFeedDatafeedManager`).
-   **Alias Resolution**: Integrates with the registry to resolve logical aliases to technical IDs.

## Architecture

The module is structured around a base abstract class and concrete implementations:

-   `base.py`: Defines the `BaseDatafeedManager` abstract base class.
-   `factory.py`: Contains the logic to instantiate the appropriate manager.
-   `mt5_manager.py`: Implementation for MT5-based data feeds.
-   `dxfeed_manager.py`: Implementation for DxFeed-based data feeds.

## Usage

To obtain a data feed manager instance, use the factory function:

```python
from ui_engine.datafeeds.factory import get_datafeed_manager

# Get a manager for the 'datafeed' alias (mapped to MT5)
manager = get_datafeed_manager("datafeed")

# Start the data feed
success = manager.start()

# Check status
status = manager.status()

# Stop the data feed
manager.stop()
```

> [!NOTE]
> The factory automatically resolves aliases using `ui_engine.registry.strategy_alias`. If an alias is not explicitly mapped to a provider in `DATAFEED_PROVIDER`, it defaults to the MT5 manager.
