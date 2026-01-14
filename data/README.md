# data/ – Market Data Directory

This directory holds all **market data** for backtesting and analysis.
Contents are git-ignored; only the directory structure (README files) is tracked.

## Structure

| Subdirectory | Purpose |
|--------------|---------|
| `csv/` | Processed CSV candle data (BID/ASK per symbol/timeframe) |
| `parquet/` | Optimized Parquet format for fast loading |
| `raw/` | Original unprocessed data exports |
| `news/` | Economic calendar and news data |

## Naming Convention

```
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.csv
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.csv
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_BID.parquet
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_ASK.parquet
```

### Examples

```
data/csv/EURUSD/EURUSD_M5_BID.csv
data/csv/EURUSD/EURUSD_H1_ASK.csv
data/parquet/EURUSD/EURUSD_D1_BID.parquet
```

## Timeframes

| Code | Meaning |
|------|---------|
| `M1` | 1 Minute |
| `M5` | 5 Minutes |
| `M15` | 15 Minutes |
| `M30` | 30 Minutes |
| `H1` | 1 Hour |
| `H4` | 4 Hours |
| `D1` | Daily |

## CSV Schema

| Column | Description |
|--------|-------------|
| `UTC time` | Candle Open-Time (UTC, timezone-aware) |
| `Open` | Opening price |
| `High` | Highest price |
| `Low` | Lowest price |
| `Close` | Closing price |
| `Volume` | Tick volume |

## Auto-Creation

Directories are created automatically by `src/hf_engine/infra/config/paths.py` on import.

## CI Validation

The directory structure is validated in CI via `tests/test_directory_structure.py`.
