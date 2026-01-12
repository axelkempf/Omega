from pathlib import Path

import pandas as pd

# ============================================================
# H1 -> H4 & D1 Converter mit 17:00 New York Rollover
# - Input:  CSV mit Header: UTC time,Open,High,Low,Close,Volume
#           (UTC time ist UTC, idealerweise tz-aware; sonst wird UTC gesetzt)
# - Output: Zwei CSVs: *_H4.csv und *_D1.csv
# - Daily-Tag endet stets um 17:00 America/New_York (NY-Close).
#   => Im Sommer 21:00 UTC, im Winter 22:00 UTC (DST automatisch!)
# - H4-Bars sind an exakt diesen Rollover verankert (Start 21/22 UTC)
# ============================================================

NY_TZ = "America/New_York"
ROLLOVER_H = 17  # 17:00 New York

AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sorgt dafür, dass df.index tz-aware UTC ist und sortiert."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")
    if idx.tz is None:
        # Naive -> als UTC interpretieren
        df.index = idx.tz_localize("UTC")
    else:
        # ggf. in UTC konvertieren
        df.index = idx.tz_convert("UTC")
    return df.sort_index()


def _read_h1_csv(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "UTC time" not in df.columns:
        raise ValueError("Spalte 'UTC time' fehlt im Input-CSV.")
    # Parse Datetime
    df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True, errors="coerce")
    df = df.dropna(subset=["UTC time"]).copy()
    df = df.set_index("UTC time")
    # Sicherstellen, dass Index UTC & sortiert ist
    df = _ensure_utc_index(df)
    # Relevante Spalten prüfen
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten im Input: {missing}")
    return df[needed].copy()


def resample_h4(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt H4-Bars, verankert am 17:00 NY Rollover.
    Idee:
      1) Index nach NY konvertieren
      2) Um 17h nach hinten schieben -> floor('4H') -> wieder 17h addieren
      3) resultierenden Startzeitpunkt nach UTC zurück und gruppieren
    """
    idx = df_h1.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")
    # 1) NY-Zeit
    ny_index = idx.tz_convert(NY_TZ)
    # 2) Verschiebung, um 17:00 auf Mitternacht zu legen
    shifted = (ny_index - pd.Timedelta(hours=ROLLOVER_H)).floor("4h")
    # 3) Startzeit (NY) wieder auf echte Bar-Starts (zurück) +17h
    bin_start_ny = shifted + pd.Timedelta(hours=ROLLOVER_H)
    # 4) In UTC konvertieren
    bin_start_utc = bin_start_ny.tz_convert("UTC")

    g = df_h1.groupby(bin_start_utc)
    h4 = g.agg(AGG).dropna(subset=["Open", "High", "Low", "Close"])
    h4.index.name = "UTC time"  # Label = Bar-Start in UTC
    h4 = h4.reset_index()
    return h4


def resample_d1(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt D1-Bars im identischen Format zum Input:
    UTC time,Open,High,Low,Close,Volume
    - UTC time = NY-Close-Zeitpunkt (21/22 UTC je nach DST)
    """
    idx = df_h1.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")
    ny_index = idx.tz_convert(NY_TZ)

    # Tages-Label = Schlussdatum (Shift +7h)
    session_close_date = (ny_index + pd.Timedelta(hours=24 - ROLLOVER_H)).date

    g = df_h1.groupby(session_close_date)
    d1 = g.agg(AGG).dropna(subset=["Open", "High", "Low", "Close"])

    # UTC time = Bar-Close (17:00 NY -> 21/22 UTC)
    close_ny = pd.to_datetime(d1.index).tz_localize(NY_TZ) + pd.Timedelta(
        hours=ROLLOVER_H
    )
    close_utc = close_ny.tz_convert("UTC")

    # Nur die 6 Spalten
    d1 = d1.copy()
    d1.insert(0, "UTC time", close_utc)
    d1 = d1[["UTC time", "Open", "High", "Low", "Close", "Volume"]]

    d1 = d1.sort_values("UTC time").reset_index(drop=True)
    return d1


def convert_h1_to_h4_d1(input_csv: str, out_h4_csv: str, out_d1_csv: str) -> None:
    df_h1 = _read_h1_csv(input_csv)

    # H4
    h4 = resample_h4(df_h1)
    Path(out_h4_csv).parent.mkdir(parents=True, exist_ok=True)
    h4.to_csv(out_h4_csv, index=False)
    print(f"✅ H4 exportiert: {out_h4_csv} (Bars: {len(h4)})")

    # D1
    d1 = resample_d1(df_h1)
    Path(out_d1_csv).parent.mkdir(parents=True, exist_ok=True)
    d1.to_csv(out_d1_csv, index=False)
    print(f"✅ D1 exportiert: {out_d1_csv} (Bars: {len(d1)})")


if __name__ == "__main__":
    # Use relative paths or config-driven paths instead of hardcoded absolute paths
    from hf_engine.infra.config.paths import DATA_DIR

    INPUT = DATA_DIR / "csv" / "USDCHF" / "USDCHF_H1_BID.csv"
    OUT_H4 = DATA_DIR / "csv" / "USDCHF" / "USDCHF_H4_BID.csv"
    OUT_D1 = DATA_DIR / "csv" / "USDCHF" / "USDCHF_D1_BID.csv"

    convert_h1_to_h4_d1(str(INPUT), str(OUT_H4), str(OUT_D1))
