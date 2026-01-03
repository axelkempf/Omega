# Logging Module

The **Logging Module** provides specialized logging capabilities for the backtest engine, focusing on trade execution and entry signal evaluation. It ensures that every decision made by the strategy is recorded for later analysis and debugging.

## Features

- **Trade Logging**: `trade_logger.py` captures executed trades and saves them to CSV, allowing for post-trade analysis.
- **Entry Evaluation Logging**: `entry_log.py` records the decision-making process for entry signals, including why a trade was taken or blocked (e.g., risk limits, filters).
- **Structured Data**: Logs are stored in structured formats (CSV, JSON) to facilitate easy parsing and analysis.

## Key Components

| File | Description |
|------|-------------|
| `trade_logger.py` | Logs executed trades to a CSV file in the configured log directory. |
| `entry_log.py` | Captures detailed information about entry signal evaluations, including candidates, blockers, and tags. |

## Usage

### Trade Logger

```python
from backtest_engine.logging.trade_logger import TradeLogger

logger = TradeLogger()
logger.log({
    "timestamp": datetime.now(),
    "symbol": "EURUSD",
    "action": "BUY",
    "price": 1.1050
})
logger.save()
```

### Entry Logger

```python
from backtest_engine.logging.entry_log import EntryLogger, EntryEvaluation

entry_logger = EntryLogger()
entry_logger.log(EntryEvaluation(
    timestamp=datetime.now(),
    is_candidate=True,
    entry_allowed=False,
    blocker="MaxSpreadExceeded"
))
```
