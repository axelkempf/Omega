from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from hf_engine.infra.config.paths import TRADE_LOGS_DIR


class TradeLogger:
    """
    Loggt alle Trades (als Dict) in einer CSV-Datei im Logverzeichnis.

    Felder werden beim ersten Log automatisch übernommen, falls nicht gesetzt.
    """

    def __init__(
        self, fields: Optional[List[str]] = None, log_dir: Optional[Path] = None
    ):
        """
        Args:
            fields: Liste der erwarteten Felder im Log (wird von erstem Datensatz übernommen, falls None).
            log_dir: Zielverzeichnis für Logs (default: TRADE_LOGS_DIR).
        """
        self.fields: List[str] = fields or []
        self.records: List[Dict[str, Any]] = []
        self.log_dir: Path = log_dir or TRADE_LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, data: Dict[str, Any]) -> None:
        """
        Fügt einen Trade-Datensatz dem Log hinzu.

        Args:
            data: Dict mit Trade-Feldern.
        """
        if not self.fields:
            self.fields = list(sorted(data.keys()))
        row = {field: data.get(field, None) for field in self.fields}
        self.records.append(row)

    def save(self, strategy_name: str, symbol: str, timeframe: str = "unknown"):
        """
        Speichert alle Trades als CSV-Datei im Logverzeichnis.
        Der Dateiname enthält Strategie, Symbol, Timeframe und UTC-Timestamp.

        Args:
            strategy_name: Name der Strategie.
            symbol: Symbol (z.B. 'EURUSD').
            timeframe: (Optional) Timeframe (z.B. 'M15'). Default: "unknown".
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}.csv"
        path = self.log_dir / filename
        pd.DataFrame(self.records).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ Trade-Log gespeichert: {path}")
