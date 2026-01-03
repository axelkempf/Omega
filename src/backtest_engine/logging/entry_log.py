import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class EntryEvaluation:
    """
    Hält alle Infos zu einem Entry-Evaluationspunkt:
    - timestamp: Zeitpunkt der Auswertung
    - is_candidate: War ein Entry-Kandidat?
    - entry_allowed: Wurde Entry erlaubt?
    - signal_reason: Grund des Signals (Text)
    - blocker: Grund für Verhinderung (Text)
    - tags: Zusätzliche Markierungen/Lables (Liste von Strings)
    """

    timestamp: datetime
    is_candidate: bool
    entry_allowed: bool
    signal_reason: str = "-"
    blocker: str = "-"
    tags: List[str] = field(default_factory=list)


class EntryLogger:
    """
    Loggt EntryEvaluations und bietet Export als DataFrame, CSV oder JSON.
    """

    def __init__(self):
        self.logs: List[EntryEvaluation] = []

    def log(
        self,
        timestamp: datetime,
        is_candidate: bool,
        entry_allowed: bool,
        signal_reason: str = "-",
        blocker: str = "-",
        tags: Optional[List[str]] = None,
    ):
        """
        Fügt einen Entry-Eintrag dem Log hinzu.

        Args:
            timestamp: Zeitpunkt der Evaluation.
            is_candidate: Entry-Kandidat?
            entry_allowed: Entry erlaubt?
            signal_reason: Grund des Signals.
            blocker: Grund für Blockierung.
            tags: Liste von String-Tags (optional).
        """
        self.logs.append(
            EntryEvaluation(
                timestamp=timestamp,
                is_candidate=is_candidate,
                entry_allowed=entry_allowed,
                signal_reason=signal_reason,
                blocker=blocker,
                tags=tags or [],
            )
        )

    def to_csv(self, path: str) -> None:
        """
        Exportiert das Log als CSV.

        Args:
            path: Zielpfad der CSV.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        """
        Exportiert das Log als JSON.

        Args:
            path: Zielpfad der JSON-Datei.
        """
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.logs], f, indent=2, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Wandelt alle Logs in ein DataFrame um.
        Tags werden als boolesche Spalten für jedes einzigartige Tag dargestellt.

        Returns:
            pd.DataFrame mit allen Feldern und Tag-Spalten.
        """
        df = pd.DataFrame([asdict(e) for e in self.logs])
        if "tags" in df and not df["tags"].empty:
            all_tags = set(tag for tags in df["tags"] for tag in tags)
            for tag in all_tags:
                df[f"tag_{tag}"] = df["tags"].apply(lambda lst: tag in lst)
        return df
