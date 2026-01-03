# time_utils.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from hf_engine.infra.config.environment import TIMEZONE, VANTAGE_TIMEZONE

__all__ = ["to_utc", "from_utc_to_broker", "now_utc"]


# Annahmen (explizit dokumentiert):
# - TIMEZONE repräsentiert die systemweite Referenzzeitzone (UTC).
# - VANTAGE_TIMEZONE ist die Broker-/Datafeed-Zeitzone.
# - Naive Datetimes (tzinfo is None oder utcoffset(None)) werden IMMER als Broker-Zeit interpretiert.
# - Alle Rückgaben sind tz-aware.


def _ensure_datetime(value: datetime) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"Expected 'datetime', got '{type(value).__name__}'")


def _is_naive(dt: datetime) -> bool:
    # tzinfo kann gesetzt sein, aber utcoffset(None) dennoch None liefern (Pseudo-naiv)
    return dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None


def to_utc(dt: datetime) -> datetime:
    """
    Konvertiert beliebige naive/aware Zeit in UTC (TIMEZONE).
    - Naive Eingabe wird als Broker-Zeit (VANTAGE_TIMEZONE) interpretiert.
    - Aware Eingabe wird nach TIMEZONE konvertiert.
    Rückgabe ist tz-aware und in TIMEZONE.

    Beispiele:
        >>> to_utc(datetime(2025, 1, 1, 12, 0))                 # naive -> Broker annehmen -> UTC
        >>> to_utc(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))  # aware UTC -> UTC
    """
    _ensure_datetime(dt)

    if _is_naive(dt):
        # Naive Zeiten als Broker-Zeit interpretieren
        dt = dt.replace(tzinfo=VANTAGE_TIMEZONE)

    # Nach systemweiter Referenz (UTC) konvertieren
    return dt.astimezone(TIMEZONE)


def from_utc_to_broker(dt: datetime) -> datetime:
    """
    Konvertiert UTC-Zeit nach Broker-Zeit (VANTAGE_TIMEZONE).
    - Naive Eingabe wird als UTC interpretiert.
    - Aware Eingabe wird unabhängig von ihrer TZ korrekt transformiert.
    Rückgabe ist tz-aware in VANTAGE_TIMEZONE.
    """
    _ensure_datetime(dt)

    # Naive Datetimes als UTC interpretieren (systemweite Referenz)
    if _is_naive(dt):
        dt = dt.replace(tzinfo=timezone.utc)

    # Unabhängig von der Eingabe-TZ sauber in Broker-Zeit transformieren
    return dt.astimezone(VANTAGE_TIMEZONE)


def now_utc() -> datetime:
    """
    Systemweit konsistentes 'Jetzt' in UTC (tz-aware).
    Einheitlicher Single-Source-of-Truth für Zeitabfragen.
    """
    return datetime.now(timezone.utc)
