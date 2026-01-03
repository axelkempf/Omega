# Logging Layer

This directory contains the logging infrastructure, designed for high-frequency trading requirements: speed, reliability, and structured data.

## Overview

The Logging Layer provides a robust logging service that supports multiple outputs (Console, File, SQLite, CSV) and ensures thread safety. It is designed to handle high volumes of logs without blocking the main execution thread.

## Key Components

### `log_service.py`
The central logging service.
- **Backends**:
    - **Console**: Standard output for real-time monitoring.
    - **File**: Rotating file handlers for persistent logs.
    - **SQLite**: Stores logs in a local SQLite database (WAL mode) for advanced querying and filtering.
    - **CSV**: Dedicated CSV logging for trade history (`trade_log.csv`).
- **Features**:
    - **Thread Safety**: Uses locks to prevent race conditions.
    - **Structured Data**: Logs include context (strategy, symbol, etc.).

### `error_handler.py`
Utilities for safe execution and error reporting.
- **`safe_execute`**: A decorator/wrapper that catches exceptions, logs them with stack traces, and prevents the application from crashing.
- **Alerting**: Can trigger external alerts (e.g., Telegram) on critical errors.

## Trade Log Schema

The `trade_log.csv` follows a strict schema defined in `CSV_HEADERS`:
- `datetime`: UTC timestamp
- `strategy`: Strategy name
- `symbol`: Traded symbol
- `direction`: BUY/SELL
- `entry_price`, `exit_price`, `profit_abs`, `profit_pct`, etc.

## Usage

```python
from hf_engine.infra.logging.log_service import log_service

# General logging
log_service.logger.info("System started")

# Trade logging
log_service.log_trade(
    strategy="MyStrategy",
    symbol="EURUSD",
    direction="BUY",
    # ...
)
```
