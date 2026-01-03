import glob
import os
from typing import List

import pandas as pd


def merge_csv_files(input_files: List[str], output_file: str) -> None:
    """
    Führt mehrere Candle-CSV-Dateien zusammen, normalisiert und sortiert sie nach UTC-Zeit.

    - 'Gmt time' wird als 'UTC time' übernommen (UTC-parsing mit dayfirst=True).
    - Duplikate werden entfernt.
    - Ausgabe erfolgt als CSV.

    Args:
        input_files: Liste von CSV-Dateinamen (beliebige Reihenfolge).
        output_file: Name der zusammengeführten Output-CSV.
    """
    all_dfs = []
    for file in input_files:
        print(f"📥 Lade: {file}")
        try:
            df = pd.read_csv(file)
            # Parse Gmt time als UTC
            df["UTC time"] = pd.to_datetime(
                df["Gmt time"], utc=True, dayfirst=True, errors="coerce"
            )
            df.drop(columns=["Gmt time"], inplace=True)
            # Spaltenreihenfolge: UTC time ganz nach vorne
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index("UTC time")))
            df = df[cols]
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Fehler beim Laden von {file}: {e}")

    if not all_dfs:
        print(f"⚠️ Keine Daten zum Zusammenführen gefunden.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.dropna(subset=["UTC time"])
    merged = merged.sort_values(by="UTC time")
    merged = merged.drop_duplicates(subset="UTC time")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        merged.to_csv(output_file, index=False)
        print(f"✅ Zusammengeführt: {output_file}")
    except Exception as e:
        print(f"❌ Fehler beim Schreiben nach {output_file}: {e}")


if __name__ == "__main__":
    # Beispiel für GBPUSD in verschiedenen Zeitrahmen
    patterns = [
        ("M30", "GBPUSD_Candlestick_30_M_BID_*.csv", "GBPUSD_M30_BID.csv"),
        ("M30", "GBPUSD_Candlestick_30_M_ASK_*.csv", "GBPUSD_M30_ASK.csv"),
        ("M5", "GBPUSD_Candlestick_5_M_BID_*.csv", "GBPUSD_M5_BID.csv"),
        ("M5", "GBPUSD_Candlestick_5_M_ASK_*.csv", "GBPUSD_M5_ASK.csv"),
        ("M15", "GBPUSD_Candlestick_15_M_BID_*.csv", "GBPUSD_M15_BID.csv"),
        ("M15", "GBPUSD_Candlestick_15_M_ASK_*.csv", "GBPUSD_M15_ASK.csv"),
        ("H1", "GBPUSD_Candlestick_1_Hour_BID_*.csv", "GBPUSD_H1_BID.csv"),
        ("H1", "GBPUSD_Candlestick_1_Hour_ASK_*.csv", "GBPUSD_H1_ASK.csv"),
        ("H4", "GBPUSD_Candlestick_4_Hour_BID_*.csv", "GBPUSD_H4_BID.csv"),
        ("H4", "GBPUSD_Candlestick_4_Hour_ASK_*.csv", "GBPUSD_H4_ASK.csv"),
        ("D1", "GBPUSD_Candlestick_1_D_BID_*.csv", "GBPUSD_D1_BID.csv"),
        ("D1", "GBPUSD_Candlestick_1_D_ASK_*.csv", "GBPUSD_D1_ASK.csv"),
    ]

    # Use relative paths or config-driven paths instead of hardcoded absolute paths
    from hf_engine.infra.config.paths import DATA_DIR

    base_raw = DATA_DIR / "raw" / "GBPUSD"
    base_out = DATA_DIR / "csv" / "GBPUSD"

    for tf, pattern, out_file in patterns:
        input_files = sorted(glob.glob(os.path.join(str(base_raw), tf, pattern)))
        output_file = os.path.join(str(base_out), out_file)
        merge_csv_files(input_files, output_file)
