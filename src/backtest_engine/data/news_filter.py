from bisect import bisect_right
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd


class NewsFilter:
    """
    Unterdrückt Trading 60 Minuten vor und 30 Minuten nach News-Events für betroffene Währungen.

    Regeln:
      - GBP News: blockiert Paare mit GBP
      - EUR News: blockiert Paare mit EUR
      - AUD News: blockiert Paare mit AUD und NZD
      - NZD News: blockiert Paare mit NZD und AUD
      - JPY News: blockiert Paare mit JPY
      - CHF News: blockiert Paare mit CHF
      - CAD News: blockiert Paare mit CAD
      - USD News: blockiert **alle** Paare
    """

    # ---- Klassenweiter Cache für bereits geparste Fenster ----
    @staticmethod
    @lru_cache(maxsize=8)
    def _load_windows_cached(csv_path_str: str, date_col: str, currency_col: str):
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            raise FileNotFoundError(f"News CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=[date_col]).sort_values(date_col)
        block_map = {
            "GBP": {"GBP"},
            "EUR": {"EUR"},
            "AUD": {"AUD", "NZD"},
            "NZD": {"NZD", "AUD"},
            "JPY": {"JPY"},
            "CHF": {"CHF"},
            "CAD": {"CAD"},
            "USD": None,
        }
        windows: List[Tuple[datetime, datetime, Optional[Set[str]]]] = []
        for _, row in df.iterrows():
            ts = row[date_col]
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            cur = str(row[currency_col]).upper()
            blocked = block_map.get(cur, set())
            start = ts - timedelta(minutes=60)
            end = ts + timedelta(minutes=30)
            windows.append((start, end, blocked))
        # Rückgabe ist hashbar/Cache-geeignet
        return tuple(windows), block_map

    def __init__(
        self,
        csv_path: Union[str, Path],
        date_col: str = "Start",
        currency_col: str = "Currency",
    ):
        """
        Initialisiert NewsFilter.

        Args:
            csv_path: Pfad zur CSV-Datei mit News-Events.
            date_col: Spaltenname für Event-Startzeitpunkt.
            currency_col: Spaltenname für Währungskennung.
        """
        self.csv_path = Path(csv_path)
        # Lade (und cache) die Fenster + Mapping einmalig je CSV/Spalten
        cached_windows, block_map = self._load_windows_cached(
            str(self.csv_path), date_col, currency_col
        )
        # lokale, mutierbare Kopie für Instanz
        self.windows: List[Tuple[datetime, datetime, Optional[Set[str]]]] = list(
            cached_windows
        )
        self.block_map = block_map
        self.windows.sort(key=lambda w: w[0])
        self._starts = [w[0] for w in self.windows]
        self._ends = [w[1] for w in self.windows]
        self._blks = [w[2] for w in self.windows]

    def is_trading_allowed(
        self, timestamp, symbol: str, *, mode: str = "open", tf: str | None = None
    ) -> bool:
        """
        Prüft für ein Symbol (z.B. 'EURUSD') und Zeitpunkt, ob Trading erlaubt ist.

        Args:
            timestamp: Zeitpunkt (datetime oder pd.Timestamp, mit oder ohne TZ).
            symbol: FX-Symbol (z.B. 'EURUSD', 'AUDNZD').
            Optional: mode="close" + tf="M5|M15|H1|D1" ⇒ Zeitstempel wird erst auf Bar-Close gemappt.

        Returns:
            True, wenn Trading erlaubt; sonst False.
        """
        if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
            try:
                timestamp = timestamp.tz_localize(None)
            except Exception:
                timestamp = timestamp.tz_convert(None)

        # --- Optional: Open→Close-Zeit-Semantik ---
        if mode.lower() == "close" and tf:
            TF = tf.upper()
            minutes = (
                int(TF[1:])
                if TF.startswith("M")
                else (
                    int(TF[1:]) * 60
                    if TF.startswith("H")
                    else int(TF[1:]) * 1440 if TF.startswith("D") else 0
                )
            )
            if minutes > 0:
                from datetime import timedelta

                timestamp = timestamp + timedelta(minutes=minutes)

        base, quote = symbol[:3].upper(), symbol[-3:].upper()

        i = bisect_right(self._starts, timestamp) - 1
        if i < 0:
            return True

        j = i
        while j >= 0 and self._ends[j] >= timestamp:
            blocked = self._blks[j]
            if blocked is None:
                return False
            if blocked and (base in blocked or quote in blocked):
                return False
            j -= 1
        return True
