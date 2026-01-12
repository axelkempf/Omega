# data/csv/ – CSV Candle Data

Processed OHLCV candle data in CSV format.

## Structure

```
data/csv/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_{TYPE}.csv
```

Where:
- `{SYMBOL}` – e.g., `EURUSD`, `GBPUSD`, `US500`
- `{TIMEFRAME}` – e.g., `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`
- `{TYPE}` – `BID` or `ASK`

## Schema

`UTC time` ist die **Open-Time** (Beginn) der Kerze und ist **UTC timezone-aware**.

```csv
UTC time,Open,High,Low,Close,Volume
2024-01-01 00:00:00+00:00,1.10450,1.10475,1.10430,1.10460,125
```

## Converting to Parquet

For faster loading, convert CSVs to Parquet:

```bash
python -m src.backtest_engine.data.convert_csv_candles_to_parquet
```

## Data Sources

Obtain historical data from your broker (e.g., MetaTrader 5 export) or data vendors.
