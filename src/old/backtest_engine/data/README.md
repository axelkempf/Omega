# Data Module

The **Data Module** is responsible for loading, processing, and managing market data used in backtests. It supports both CSV and Parquet formats and handles various data types including candles, ticks, and news events.

## Features

- **Data Loading**: `data_handler.py` loads market data from CSV or Parquet files, with caching mechanisms for performance.
- **Data Structures**: Defines core data structures like `Candle` (`candle.py`) and `Tick` (`tick.py`).
- **Format Conversion**: Tools like `convert_csv_candles_to_parquet.py` and `csv_converter.py` facilitate data format transformations.
- **Market Context**: Includes utilities for handling market hours (`market_hours.py`), trading holidays (`trading_holidays.py`), and news events (`news_filter.py`).

## Key Components

| File | Description |
|------|-------------|
| `data_handler.py` | Main interface for loading and caching market data. |
| `candle.py` | Dataclass definition for OHLCV candle data. |
| `market_hours.py` | Utilities to check for valid trading hours. |
| `news_filter.py` | Filters out trading periods based on high-impact news events. |
| `convert_csv_candles_to_parquet.py` | Script to convert CSV data to the more efficient Parquet format. |

## Data Formats

The engine primarily works with:
- **CSV**: Raw data format, typically used for import.
- **Parquet**: Optimized columnar format for fast loading during backtests.

## Usage

```python
from backtest_engine.data.data_handler import load_candle_data

# Load M5 candles for EURUSD
candles = load_candle_data(
    symbol="EURUSD",
    timeframe="M5",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```
