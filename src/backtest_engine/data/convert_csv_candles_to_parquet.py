import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def convert_all_csv_to_parquet(
    base_dir: Union[str, Path] = "data/history",
    output_dir: Optional[Union[str, Path]] = None,
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
    base_dir = Path(base_dir) if not isinstance(base_dir, Path) else base_dir
    if output_dir is None:
        output_dir = base_dir
    else:
        output_dir = (
            Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in base_dir.iterdir():
        if file_path.suffix != ".csv":
            continue

        full_path = file_path
        parquet_file = file_path.stem + ".parquet"
        parquet_path = output_dir / parquet_file

        if parquet_path.exists() and not overwrite:
            print(f"🟡 Überspringe: {parquet_file} (bereits vorhanden)")
            continue

        print(f"🔄 Konvertiere: {file_path.name}")

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
            print(f"❌ Fehler beim Lesen von {file_path.name}: {e}")
            continue

        df["UTC time"] = pd.to_datetime(df["UTC time"], utc=True, errors="coerce")

        try:
            df.to_parquet(parquet_path, index=False)
            print(f"✅ Gespeichert: {parquet_path}")
        except Exception as e:
            print(f"❌ Fehler beim Schreiben von {parquet_file}: {e}")


if __name__ == "__main__":
    # Use relative paths or config-driven paths instead of hardcoded absolute paths
    from hf_engine.infra.config.paths import DATA_DIR

    convert_all_csv_to_parquet(
        base_dir=DATA_DIR / "csv" / "EURUSD",
        output_dir=DATA_DIR / "parquet" / "EURUSD",
    )
