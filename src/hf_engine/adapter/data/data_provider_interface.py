# core/data/data_provider_interface.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Final, Optional, TypedDict

import pandas as pd

# ---- Gemeinsame, verbindliche Verträge (Schema & Typen) ---------------------

DATAFRAME_COLUMNS: Final = ("open", "high", "low", "close", "volume")


class OHLC(TypedDict):
    """
    Repräsentiert EINE vollständig geschlossene Kerze.
    - time:   Abschlusszeitpunkt der Kerze, TZ‑AWARE in UTC
    - open:   Eröffnungskurs
    - high:   Höchstkurs
    - low:    Tiefstkurs
    - close:  Schlusskurs
    - volume: Volumen (Einheit providerabhängig, z. B. Lots/Ticks)
    """

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCSeries(TypedDict):
    """
    Repräsentiert eine Serie abgeschlossener Kerzen als spaltenorientierte Struktur.
    Alle Listen müssen gleich lang und zeitlich aufsteigend sortiert sein.
    - time:   Liste TZ‑AWARE UTC‑Zeitpunkte (Bar‑Close)
    - open/high/low/close/volume: Listen numerischer Werte
    """

    time: list[datetime]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]


# ---- Interface --------------------------------------------------------------


class DataProviderInterface(ABC):
    """
    Abstrakte Schnittstelle für Marktdatenquellen.

    ZEITZONEN-POLICY (verbindlich):
    - Alle Datums-/Zeitparameter sind TZ‑AWARE und in UTC zu übergeben.
    - Alle zurückgegebenen Zeiten sind TZ‑AWARE und in UTC.
    - Implementierungen, deren Upstream-Quelle in anderer TZ (z. B. UTC+3) arbeitet,
      führen die Konvertierung intern durch. Der Aufrufer sieht ausschließlich UTC.

    DATAFRAME-SCHEMA (verbindlich):
    - Index:   'time' (TZ‑AWARE UTC, aufsteigend, Bar‑Close)
    - Columns: ('open','high','low','close','volume') = DATAFRAME_COLUMNS
    - Dtypes:  float (volume ggf. float/int; Implementierung sollte bestmöglich
                in float casten, um Einheitlichkeit zu wahren)

    HINWEIS:
    - 'timeframe' ist ein vom System definierter String (z. B. "M1","M5","H1","D1").
      Validierung erfolgt in der konkreten Implementierung oder höherer Ebene.
    """

    # ------------------------- Historische Balken -----------------------------

    @abstractmethod
    def get_rates_range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Liefert OHLCV‑Bars innerhalb [start, end).

        Anforderungen:
        - start/end: TZ‑AWARE UTC, end > start
        - Rückgabe: DataFrame mit Index 'time' (UTC) und Spalten DATAFRAME_COLUMNS.
        - Bars sind vollständig geschlossen und lückenfrei soweit vom Provider möglich.

        Fehlerbehandlung:
        - Bei leerer Ergebnismenge: DataFrame mit korrektem Schema, aber 0 Zeilen.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rates_from_pos(
        self,
        symbol: str,
        timeframe: str,
        start_pos: int,
        count: int,
    ) -> pd.DataFrame:
        """
        Liefert 'count' OHLCV‑Bars beginnend bei einer providerdefinierten Position.

        Anforderungen:
        - start_pos: >= 0 (0 = ältester verfügbarer Bar gemäß Providerdefinition)
        - count:     > 0
        - Rückgabe:  DataFrame mit Schema wie oben (UTC‑Index 'time').

        Einsatz:
        - Nützlich, wenn der Provider effiziente, positionsbasierte Zugriffe bietet.
        """
        raise NotImplementedError

    # ----------------------------- Ticks -------------------------------------

    @abstractmethod
    def get_tick_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Liefert Tick‑Daten im Bereich [start, end).

        Anforderungen:
        - start/end: TZ‑AWARE UTC, end > start
        - Rückgabe:  DataFrame mit mindestens:
                     Index 'time' (UTC),
                     Spalten z. B. ('bid','ask','last','volume') – je nach Quelle.
                     Die konkrete Implementierung MUSS ihre Tick‑Spalten im
                     Modul-/Klassendocstring dokumentieren.
        - Reihenfolge: Zeitlich aufsteigend.

        Bei leerem Ergebnis: Leerer DataFrame mit dokumentiertem Schema.
        """
        raise NotImplementedError

    # ---------------------------- OHLC Helpers -------------------------------

    @abstractmethod
    def get_ohlc_for_closed_candle(
        self,
        symbol: str,
        timeframe: str,
        offset: int = 1,
    ) -> Optional[OHLC]:
        """
        Gibt die OHLCV‑Daten EINER bereits geschlossenen Kerze zurück.

        Parameter:
        - offset: 1 = zuletzt geschlossene Kerze, 2 = die davor, usw. (>=1)

        Rückgabe:
        - OHLC (inkl. 'time' als UTC) oder None, falls nicht verfügbar.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ohlc_series(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> Optional[OHLCSeries]:
        """
        Gibt eine Serie von 'count' vollständig geschlossenen Kerzen zurück.

        Anforderungen:
        - count > 0
        - Reihenfolge: Zeitlich aufsteigend
        - Alle Listen im Ergebnis sind gleich lang.

        Rückgabe:
        - OHLCSeries (inkl. UTC‑'time') oder None bei Nichtverfügbarkeit.
        """
        raise NotImplementedError

    # ------------------------- Optionale Hilfsroutine ------------------------

    @staticmethod
    def ensure_utc(dt: datetime) -> datetime:
        """
        Stellt sicher, dass ein datetime TZ‑AWARE in UTC ist.
        Implementierungen können diese Utility für Eingangs-/Ausgangsvalidierung verwenden.
        """
        if dt.tzinfo is None:
            # Falls ohne TZ geliefert, als UTC interpretieren (strikt und explizit).
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
