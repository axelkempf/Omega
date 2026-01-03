# data/news/ – Economic Calendar Data

News and economic event data for fundamental analysis filtering.

## Key File

- `news_calender.csv` – Economic calendar with event times and impact levels

## Usage

The `NewsFilter` in `src/backtest_engine/data/news_filter.py` uses this data to filter out
high-impact news periods from backtests.

## Utilities

- `csv_cleaner.py` – Utility script for cleaning/processing news CSV files
