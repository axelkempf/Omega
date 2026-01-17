#!/usr/bin/env python3
"""Generate test candle fixtures with >=500 bars per timeframe."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE_PRICE = 1.1000
DEFAULT_SPREAD = 0.0001
DEFAULT_BARS = 600


def _price_at(index: int, amplitude: float, period: float) -> float:
    return BASE_PRICE + amplitude * math.sin(index / period)


def _build_rows(
    bars: int,
    minutes: int,
    amplitude: float,
    period: float,
    start: datetime,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(bars):
        timestamp = start + timedelta(minutes=minutes * idx)
        open_price = _price_at(idx, amplitude, period)
        close_price = _price_at(idx + 1, amplitude, period)
        high = max(open_price, close_price) + 0.0002
        low = min(open_price, close_price) - 0.0002
        volume = 100.0 + float(idx % 10)
        rows.append(
            {
                "UTC time": timestamp,
                "Open": round(open_price, 5),
                "High": round(high, 5),
                "Low": round(low, 5),
                "Close": round(close_price, 5),
                "Volume": round(volume, 1),
            }
        )
    return rows


def _apply_spread(
    rows: list[dict[str, object]],
    spread: float,
) -> list[dict[str, object]]:
    adjusted: list[dict[str, object]] = []
    for row in rows:
        adjusted.append(
            {
                "UTC time": row["UTC time"],
                "Open": round(float(row["Open"]) + spread, 5),
                "High": round(float(row["High"]) + spread, 5),
                "Low": round(float(row["Low"]) + spread, 5),
                "Close": round(float(row["Close"]) + spread, 5),
                "Volume": row["Volume"],
            }
        )
    return adjusted


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("UTC time,Open,High,Low,Close,Volume\n")
        for row in rows:
            ts = row["UTC time"].strftime("%Y-%m-%d %H:%M:%S%z")
            ts = f"{ts[:-2]}:{ts[-2:]}"
            handle.write(
                f"{ts},{row['Open']:.5f},{row['High']:.5f},{row['Low']:.5f},"
                f"{row['Close']:.5f},{row['Volume']:.1f}\n"
            )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True)
    df.to_parquet(path, index=False)


def _write_pair(
    root: Path,
    timeframe: str,
    minutes: int,
    amplitude: float,
    period: float,
    bars: int,
    spread: float,
    start: datetime,
) -> None:
    bid_rows = _build_rows(bars, minutes, amplitude, period, start)
    ask_rows = _apply_spread(bid_rows, spread)

    csv_dir = root / "csv" / "EURUSD"
    parquet_dir = root / "parquet" / "EURUSD"
    csv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    bid_csv = csv_dir / f"EURUSD_{timeframe}_BID.csv"
    ask_csv = csv_dir / f"EURUSD_{timeframe}_ASK.csv"
    bid_parquet = parquet_dir / f"EURUSD_{timeframe}_BID.parquet"
    ask_parquet = parquet_dir / f"EURUSD_{timeframe}_ASK.parquet"

    _write_csv(bid_csv, bid_rows)
    _write_csv(ask_csv, ask_rows)
    _write_parquet(bid_parquet, bid_rows)
    _write_parquet(ask_parquet, ask_rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_root = repo_root / "python" / "tests" / "fixtures" / "data"
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    _write_pair(
        fixture_root,
        timeframe="M1",
        minutes=1,
        amplitude=0.0006,
        period=30.0,
        bars=DEFAULT_BARS,
        spread=DEFAULT_SPREAD,
        start=start,
    )
    _write_pair(
        fixture_root,
        timeframe="M5",
        minutes=5,
        amplitude=0.0008,
        period=25.0,
        bars=DEFAULT_BARS,
        spread=DEFAULT_SPREAD,
        start=start,
    )

    print("Test fixtures regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
