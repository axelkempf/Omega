# data/parquet/ – Optimized Candle Data

Parquet-format OHLCV data for fast DataFrame loading.

## Structure

```
data/parquet/{SYMBOL}/{SYMBOL}_{TIMEFRAME}_{TYPE}.parquet
```

## Benefits

- **Faster loading** – columnar format, compressed
- **Type preservation** – datetime/numeric types preserved
- **Smaller size** – typically 5-10x smaller than CSV

## Generation

Convert from CSV using:

```bash
python -m src.backtest_engine.data.convert_csv_candles_to_parquet
```

## Usage

```python
from hf_engine.infra.config.paths import PARQUET_DIR
import pandas as pd

df = pd.read_parquet(PARQUET_DIR / "EURUSD" / "EURUSD_H1_BID.parquet")
```
