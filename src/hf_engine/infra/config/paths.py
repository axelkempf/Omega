"""
Zentrale Definition aller Systempfade für Logging, Archivierung und Datenmanagement.
Vermeidet harte Strings im Code und stellt sicher, dass notwendige Verzeichnisse existieren.
"""

from pathlib import Path
from typing import List

# Root-Verzeichnis (automatisch ausgehend vom Speicherort der Datei)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]

# Basisverzeichnisse
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"

# Laufzeitdaten getrennt halten ("var" als Sammelpunkt, git-ignored)
VAR_DIR: Path = PROJECT_ROOT / "var"
LOGS_DIR: Path = VAR_DIR / "logs"
RESULTS_DIR: Path = VAR_DIR / "results"
ARCHIVE_DIR: Path = VAR_DIR / "archive"
TMP_DIR: Path = VAR_DIR / "tmp"

# Unterordner – Konfiguration
LIVE_CONFIG_DIR: Path = CONFIG_DIR / "live"
BACKTEST_CONFIG_DIR: Path = CONFIG_DIR / "backtest"

# Unterordner – Logs
ENTRY_LOGS_DIR: Path = LOGS_DIR / "entry_logs"
TRADE_LOGS_DIR: Path = LOGS_DIR / "trade_logs"
SYSTEM_LOGS_DIR: Path = LOGS_DIR / "system"
OPTUNA_LOGS_DIR: Path = LOGS_DIR / "optuna"

# Unterordner – Daten
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PARQUET_DIR: Path = DATA_DIR / "parquet"
CSV_DATA_DIR: Path = DATA_DIR / "csv"
NEWS_DATA_DIR: Path = DATA_DIR / "news"

# Unterordner – Ergebnisse
BACKTEST_RESULTS_DIR: Path = RESULTS_DIR / "backtests"
WALKFORWARD_RESULTS_DIR: Path = RESULTS_DIR / "walkforwards"

# → Pfade zu spezifischen Dateien
TRADE_LOG_CSV: Path = TRADE_LOGS_DIR / "trade_log.csv"
EXECUTION_TRACK_PATH: Path = TRADE_LOGS_DIR / "executions.json"
DB_PATH: Path = SYSTEM_LOGS_DIR / "engine_logs.db"
NEWS_CALENDER: Path = NEWS_DATA_DIR / "news_calender.csv"

# --- Interne Hilfsfunktionen ---


def ensure_directories(dirs: List[Path]) -> None:
    """Stellt sicher, dass alle angegebenen Verzeichnisse existieren."""
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Erstellen des Verzeichnisses: {d}") from e


# --- Verzeichnis-Erstellung ---

DIRECTORIES: List[Path] = [
    CONFIG_DIR,
    DATA_DIR,
    VAR_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    ARCHIVE_DIR,
    TMP_DIR,
    ENTRY_LOGS_DIR,
    TRADE_LOGS_DIR,
    SYSTEM_LOGS_DIR,
    OPTUNA_LOGS_DIR,
    RAW_DATA_DIR,
    PARQUET_DIR,
    CSV_DATA_DIR,
    NEWS_DATA_DIR,
    BACKTEST_RESULTS_DIR,
    WALKFORWARD_RESULTS_DIR,
]

ensure_directories(DIRECTORIES)
