import os
from typing import Optional

import pandas as pd


def convert_all_csv_to_parquet(
    base_dir: str = "data/history",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Konvertiert alle CSV-Dateien im angegebenen Verzeichnis ins Parquet-Format.

    Args:
        base_dir: Quellverzeichnis mit CSV-Dateien.
        output_dir: Zielverzeichnis für Parquet-Dateien. Standard: base_dir.
        overwrite: Falls True, werden existierende Parquet-Dateien überschrieben.

    Prints:
        Fortschritt und Status für jede Datei.
    """
    if output_dir is None:
        output_dir = base_dir

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(base_dir):
        if not file.endswith(".csv"):
            continue

        full_path = os.path.join(base_dir, file)
        parquet_file = file.replace(".csv", ".parquet")
        parquet_path = os.path.join(output_dir, parquet_file)

        if os.path.exists(parquet_path) and not overwrite:
            print(f"🟡 Überspringe: {parquet_file} (bereits vorhanden)")
            continue

        print(f"🔄 Konvertiere: {file}")

        try:
            df = pd.read_csv(
                full_path,
                parse_dates=["UTC time"],
                dtype={
                    "Open": float,
                    "High": float,
                    "Low": float,
                    "Close": float,
                    "Volume": float,
                },
            )
        except Exception as e:
            print(f"❌ Fehler beim Lesen von {file}: {e}")
            continue

        df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True, errors="coerce")

        try:
            df.to_parquet(parquet_path, index=False)
            print(f"✅ Gespeichert: {parquet_path}")
        except Exception as e:
            print(f"❌ Fehler beim Schreiben von {parquet_file}: {e}")


if __name__ == "__main__":
    convert_all_csv_to_parquet(
        base_dir="/Users/axelkempf/kempf_capital_algorithmus/data/csv/EURUSD",
        output_dir="/Users/axelkempf/kempf_capital_algorithmus/data/parquet/EURUSD",
    )
