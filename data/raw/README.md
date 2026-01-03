# data/raw/ – Unprocessed Data Exports

Original data files as exported from brokers or data vendors.

## Purpose

- Preserve original data for reproducibility
- Store various export formats before standardization
- Keep backups of source data

## Typical Structure

```
data/raw/{SYMBOL}/{TIMEFRAME}/
```

## Processing

Raw data is processed into standardized CSV/Parquet format using utilities in:
- `src/backtest_engine/data/csv_converter.py`
- `src/backtest_engine/data/merge_csv.py`
