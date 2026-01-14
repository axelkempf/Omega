from datetime import datetime
from typing import List, Optional

import pandas as pd
from backtest_engine.data.market_hours import is_valid_trading_time
from backtest_engine.data.tick import Tick

from hf_engine.infra.config.paths import PARQUET_DIR, RAW_DATA_DIR


class TickDataHandler:
    """
    Lädt Tick-Daten (Bid/Ask/Volume) aus Parquet- oder CSV-Dateien.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.csv_dir = RAW_DATA_DIR / "csv" / symbol
        self.parquet_dir = PARQUET_DIR / symbol

        self.tick_csv = self.csv_dir / f"{symbol}_ticks.csv"
        self.tick_parquet = self.parquet_dir / f"{symbol}_ticks.parquet"

    def load_ticks(
        self, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
    ) -> List[Tick]:
        """
        Lädt und filtert Ticks (Bid, Ask, Volume) für ein Symbol und Zeitraum.

        Args:
            start_dt: Erstes erlaubtes Datum/Zeit (UTC).
            end_dt: Letztes erlaubtes Datum/Zeit (UTC).

        Returns:
            Liste von Tick-Objekten.
        """
        if self.tick_parquet.exists():
            df = pd.read_parquet(self.tick_parquet)
        elif self.tick_csv.exists():
            df = pd.read_csv(self.tick_csv)
        else:
            raise FileNotFoundError(
                f"Keine Tickdaten gefunden für Symbol {self.symbol}"
            )

        df["UTC time"] = pd.to_datetime(df["UTC time"])
        if df["UTC time"].dt.tz is None:
            df["UTC time"] = df["UTC time"].dt.tz_localize("UTC")
        else:
            df["UTC time"] = df["UTC time"].dt.tz_convert("UTC")

        if start_dt:
            df = df[df["UTC time"] >= start_dt]
        if end_dt:
            df = df[df["UTC time"] <= end_dt]

        df = df[df["UTC time"].apply(is_valid_trading_time)]

        ticks: List[Tick] = []
        for _, row in df.iterrows():
            ticks.append(
                Tick(
                    timestamp=row["UTC time"],
                    bid=row["Bid"],
                    ask=row["Ask"],
                    volume=row.get("Volume", 0.0),
                )
            )
        return ticks
