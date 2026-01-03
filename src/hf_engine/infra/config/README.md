# Configuration Layer

This directory contains the central configuration and environment management logic for the trading engine.

## Overview

The Configuration Layer ensures that the application is correctly configured across different environments (Dev, Staging, Prod). It handles environment variables, file paths, symbol mapping, and timezones in a centralized and consistent manner.

## Key Components

### `environment.py`
Loads and validates configuration from `.env` files and environment variables.
- **Features**:
    - **Type Safety**: Helpers for retrieving boolean, integer, and string values.
    - **Validation**: Enforces required variables based on enabled features (e.g., Telegram, MT5).
    - **Timezones**: Provides consistent `ZoneInfo` objects for System and Broker timezones.

### `paths.py`
Defines the directory structure and file paths used by the application.
- **Centralization**: All paths (logs, data, results, temp files) are defined here.
- **Consistency**: Ensures all components use the same locations for artifacts.

### `symbol_mapper.py`
Handles the translation between internal symbol names (e.g., "EURUSD") and broker-specific tickers (e.g., "EURUSD.r", "EURUSD+").
- **Usage**: Used by the Broker Adapter and Data Provider to ensure compatibility with different brokers.

### `time_utils.py`
Utilities for robust time handling.
- **UTC Enforcement**: Helpers to get current UTC time (`now_utc()`).
- **Conversion**: Functions to convert between UTC and Broker time.

## Usage

Components should import configuration from this layer rather than reading environment variables or hardcoding paths directly.

```python
from hf_engine.infra.config.environment import ENV
from hf_engine.infra.config.paths import LOG_DIR

if ENV.TELEGRAM_ENABLED:
    # ...
```
