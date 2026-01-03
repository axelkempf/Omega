# news_filter.py
from __future__ import annotations

import csv
import os
import threading
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from hf_engine.infra.config.paths import NEWS_CALENDER as NEWS_PATH
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.log_service import log_service

logger = log_service.logger
# ------------------------------- FX Zuordnung ---------------------------------

_SYMBOL_TO_CURRENCIES = {
    "AUDCAD": ["AUD", "CAD"],
    "AUDCHF": ["AUD", "CHF"],
    "AUDJPY": ["AUD", "JPY"],
    "AUDNZD": ["AUD", "NZD"],
    "AUDUSD": ["AUD", "USD"],
    "CADJPY": ["CAD", "JPY"],
    "CADCHF": ["CAD", "CHF"],
    "CHFJPY": ["CHF", "JPY"],
    "EURAUD": ["EUR", "AUD"],
    "EURCAD": ["EUR", "CAD"],
    "EURCHF": ["EUR", "CHF"],
    "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "EURNZD": ["EUR", "NZD"],
    "EURUSD": ["EUR", "USD"],
    "GBPAUD": ["GBP", "AUD"],
    "GBPCAD": ["GBP", "CAD"],
    "GBPCHF": ["GBP", "CHF"],
    "GBPJPY": ["GBP", "JPY"],
    "GBPNZD": ["GBP", "NZD"],
    "GBPUSD": ["GBP", "USD"],
    "NZDCAD": ["NZD", "CAD"],
    "NZDCHF": ["NZD", "CHF"],
    "NZDJPY": ["NZD", "JPY"],
    "NZDUSD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "USDCHF": ["USD", "CHF"],
    "USDJPY": ["USD", "JPY"],
    "XAUUSD": ["XAU", "USD"],
    "US500": ["USD"],
}
# Fallback-Menge erlaubter Währungscodes (für generisches Parsen 3+3)
_KNOWN_CCY = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD", "XAU"}

# Block‑Logik: welche Währungen sperren einen Trade auf welche anderen
# "*" => alle Symbole sperren (z.B. bei USD‑High‑Impact)
_CURRENCY_BLOCK_MAP = {
    "AUD": ["AUD", "NZD"],
    "NZD": ["NZD", "AUD"],
    "USD": ["*"],
}

# Impact‑Ranking
_IMPACT_LEVELS = {"low": 1, "medium": 2, "high": 3}

_PRE_BLOCK_MINUTES = 60
_POST_BLOCK_MINUTES = 30

# ------------------------------ Datenstrukturen --------------------------------


@dataclass(frozen=True)
class NewsItem:
    datetime: datetime  # in UTC, tz‑aware
    currency: str  # "USD", "EUR", ...
    impact: str  # "low"|"medium"|"high" (lowercase)
    event: str  # Title


class _NewsCache:
    """
    Thread‑sicherer Warm‑Cache:
    - Lädt CSV nur bei Bedarf
    - Auto‑Reload, wenn mtime der Datei sich ändert
    """

    def __init__(self) -> None:
        self._events: List[NewsItem] = []
        self._times_sorted: List[datetime] = []
        self._mtime: Optional[float] = None
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._times_sorted.clear()
            self._mtime = None

    def maybe_reload(self, path: Path) -> None:
        """
        Lädt initial oder bei Dateiänderung neu.
        """
        with self._lock:
            try:
                if not path.exists():
                    if self._mtime is not None:
                        logger.warning(
                            "📰 News‑Kalender nicht gefunden (war zuvor vorhanden). Pfad=%s",
                            str(path),
                        )
                    self._events = []
                    self._times_sorted = []
                    self._mtime = None
                    return

                mtime = path.stat().st_mtime
                if self._mtime is not None and mtime == self._mtime:
                    # Warm‑Cache gültig
                    return

                events, times = _load_csv_file(path)
                self._events = events
                self._times_sorted = times
                self._mtime = mtime

                logger.info(
                    "📰 News‑Kalender geladen: %d Events (Warm‑Cache aktiv). Pfad=%s",
                    len(self._events),
                    str(path),
                )
            except Exception as e:
                # Fehler niemals stillschweigend verschlucken
                logger.exception("❌ Fehler beim Laden des News‑Kalenders: %s", e)
                # Cache in konsistenten Zustand versetzen (leer, aber definiert)
                self._events = []
                self._times_sorted = []
                self._mtime = None

    def view(self) -> Tuple[List[NewsItem], List[datetime]]:
        with self._lock:
            # Read‑Only Views (Kopien verhindern Race Conditions beim Iterieren)
            return list(self._events), list(self._times_sorted)


# Singleton‑Cache auf Modulebene (für Rückwärtskompatibilität der API)
_CACHE = _NewsCache()


# ------------------------------ CSV‑Ladelogik ----------------------------------

# Akzeptierte Datums‑/Zeitformate (UTC in CSV)
# Beispiel: "06-28-2024 10:30AM"  -> "%m-%d-%Y %I:%M%p"
#           "06-28-2024 14:30"    -> "%m-%d-%Y %H:%M"
_DT_FORMATS = [
    "%m-%d-%Y %I:%M%p",  # MDY + 12h
    "%m-%d-%Y %H:%M",  # MDY + 24h
]

_REQUIRED_HEADERS = {"Date", "Time", "Country", "Impact", "Title"}


def _parse_dt_utc(date_str: str, time_str: str) -> Optional[datetime]:
    raw = f"{date_str.strip()} {time_str.strip()}"
    for fmt in _DT_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=ZoneInfo("UTC"))
        except Exception:
            continue
    return None


def _load_csv_file(path: Path) -> Tuple[List[NewsItem], List[datetime]]:
    """
    Lädt CSV vollständig, filtert KEINE Impact‑Stufen (Filter erst bei Abfrage).
    Gibt NewsItem‑Liste (UTC) und sortierte Zeitliste zurück.
    """
    events: List[NewsItem] = []

    # Lesen als Text, BOM tolerant
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = f.readlines()

    if not lines:
        logger.warning("📰 News‑CSV leer: %s", str(path))
        return [], []

    # Header prüfen
    headers = [h.strip() for h in lines[0].strip().replace("\ufeff", "").split(",")]
    missing = _REQUIRED_HEADERS.difference(headers)
    if missing:
        logger.error(
            "❌ Ungültiger News‑CSV Header: es fehlen %s. Pfad=%s",
            ", ".join(sorted(missing)),
            str(path),
        )
        return [], []

    reader = csv.DictReader(lines[1:], fieldnames=headers)
    bad_rows = 0

    for idx, row in enumerate(reader, start=2):  # start=2 => 1‑basiert inkl. Header
        try:
            impact = str(row.get("Impact", "")).strip().lower()
            country = str(row.get("Country", "")).strip().upper()
            title = str(row.get("Title", "")).strip()
            date_s = row.get("Date", "")
            time_s = row.get("Time", "")

            if not (impact and country and title and date_s and time_s):
                bad_rows += 1
                continue

            dt = _parse_dt_utc(date_s, time_s)
            if dt is None:
                bad_rows += 1
                continue

            events.append(
                NewsItem(
                    datetime=dt,
                    currency=country,
                    impact=impact,
                    event=title,
                )
            )
        except Exception as e:
            bad_rows += 1
            # pro fehlerhafter Zeile nur warnen, kein Abbruch
            logger.warning("⚠️ Zeile %d in News‑CSV übersprungen (%s).", idx, e)

    events.sort(key=lambda x: x.datetime)
    times = [e.datetime for e in events]

    if bad_rows:
        logger.warning(
            "🟠 %d fehlerhafte/inkomplette Zeilen im News‑CSV verworfen. Pfad=%s",
            bad_rows,
            str(path),
        )

    return events, times


# ------------------------------ Kernfunktionalität -----------------------------


def _post_block_end(news_time: datetime) -> datetime:
    """
    Ende des Sperrfensters nach News:
    Immer exakt 30 Minuten nach dem Event.
    """
    return news_time + timedelta(minutes=_POST_BLOCK_MINUTES)


def _normalize_min_impact(level: str) -> int:
    return _IMPACT_LEVELS.get(str(level or "").strip().lower(), _IMPACT_LEVELS["high"])


def _currencies_for_symbol(symbol: str) -> List[str]:
    s = (symbol or "").upper().strip()
    if not s:
        return []

    if s in _SYMBOL_TO_CURRENCIES:
        return _SYMBOL_TO_CURRENCIES[s]

    # Fallback: 6‑Zeichen Symbol in 3+3 Währungscodes zerlegen (z.B. "EURUSD")
    if len(s) >= 6:
        c1, c2 = s[:3], s[3:6]
        if c1 in _KNOWN_CCY and c2 in _KNOWN_CCY:
            return [c1, c2]

    return []


def load_news_csv(path: Optional[str | os.PathLike[str]] = None) -> None:
    """
    Öffentliche Ladefunktion (rückwärtskompatibel).
    - Ohne Argument: verwendet NEWS_PATH
    - Löst Auto‑Reload aus, wenn Datei geändert wurde.
    """
    p = Path(path) if path is not None else Path(NEWS_PATH)
    _CACHE.maybe_reload(p)


def clear_news_cache() -> None:
    """
    Für Tests/Tools: leert den Warm‑Cache explizit.
    """
    _CACHE.clear()


def is_news_nearby(symbol: str, now: datetime, min_impact: str = "high") -> bool:
    """
    Prüft, ob für das Symbol aktuell eine News-Sperrphase aktiv ist.
    - Pre-Block: 60 Min vor Event
    - Post-Block: 30 Min nach Event
    - **Exklusive Grenzen:** Blockintervall ist offen: (pre_start, post_end).
      Beispiele bei Event 15:30:
        - 14:30:00 ist NICHT geblockt (letzter erlaubter Trade).
        - 16:00:00 ist NICHT geblockt (erster erlaubter Trade).
    - Impact wird hier gefiltert (alle Events werden geladen)
    """
    # Sicherstellen, dass Cache geladen/aktuell ist
    load_news_csv()

    # Zeitebene: 'now' muss tz‑aware sein; falls nicht, als UTC interpretieren
    now = now if now.tzinfo else now.replace(tzinfo=ZoneInfo("UTC"))

    currencies = _currencies_for_symbol(symbol)
    if not currencies:
        return False

    min_level = _normalize_min_impact(min_impact)
    events, times_sorted = _CACHE.view()
    if not events:
        return False

    # Suchfenster um now:
    # - Links: 60m zurück (um Post-Block von kurz vor now zu erfassen)
    # - Rechts: 60m nach vorne (damit Pre-Block zuverlässig greift)
    left_dt = now - timedelta(minutes=_PRE_BLOCK_MINUTES)
    right_dt = now + timedelta(minutes=_PRE_BLOCK_MINUTES)
    li = max(bisect_left(times_sorted, left_dt) - 1, 0)
    for i in range(li, len(events)):
        news = events[i]
        nt = news.datetime
        if nt > right_dt:
            break

        if _IMPACT_LEVELS.get(news.impact, 0) < min_level:
            continue

        block_currencies = _CURRENCY_BLOCK_MAP.get(news.currency, [news.currency])
        if "*" not in block_currencies and not any(
            c in currencies for c in block_currencies
        ):
            continue

        pre_start = nt - timedelta(minutes=_PRE_BLOCK_MINUTES)
        post_end = _post_block_end(nt)
        # Exklusive Grenzen: (pre_start, post_end)
        if pre_start < now < post_end:
            return True

    return False


def get_news_events() -> list[NewsItem]:
    """
    Liefert eine *Kopie* der geladenen News-Events (UTC, sortiert).
    Public-API für Scheduler/Monitoring.
    """
    events, _ = _CACHE.view()
    return list(events)


def currencies_for_symbol(symbol: str) -> list[str]:
    """
    Public-Wrapper um die Symbol->Währungsableitung.
    """
    return _currencies_for_symbol(symbol)


def post_block_end(dt: datetime) -> datetime:
    """
    Public-Wrapper für die Post-Block-Endzeit.
    """
    return _post_block_end(dt)


def has_warm_cache() -> bool:
    return bool(_CACHE.view()[0])


# ------------------------------ Mini‑Selftest (optional) -----------------------
if __name__ == "__main__":
    # Nur für manuelle, lokale Checks – im Produktionsbetrieb nicht relevant.
    try:
        load_news_csv()
        _now = now_utc()
        for sym in ("EURUSD", "USDJPY", "GBPNZD"):
            block = is_news_nearby(sym, _now, min_impact="high")
            print(sym, "blocked:", block)
    except Exception as _e:
        # Immer klar loggen, niemals stumm abbrechen
        logger.exception("❌ Fehler im News‑Filter‑Selftest: %s", _e)
