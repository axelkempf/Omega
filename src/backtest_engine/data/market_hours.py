from datetime import time
from typing import Tuple
from zoneinfo import ZoneInfo

import pandas as pd

# Einheitliche Referenz auf Sydney-Zeitzone (DST-aware)
SYDNEY_TZ = ZoneInfo("Australia/Sydney")


def _as_utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """Hilfsfunktion: Timestamp robust nach UTC konvertieren (naiv => UTC)."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def is_valid_trading_time(ts: pd.Timestamp) -> bool:
    """
    FX-Session-Filter in **Sydney-Lokalzeit** (DST-aware):
      - Markt **öffnet**: Montag 07:00 Sydney
      - Markt **schließt**: Samstag 07:00 Sydney
    => Gültig:
       * Mo:   ab 07:00
       * Di–Fr: immer
       * Sa:   bis 07:00
       * So:   nie
    """
    ts_utc = _as_utc_ts(ts)
    ts_syd = ts_utc.tz_convert(SYDNEY_TZ)
    wd = ts_syd.weekday()  # 0=Mo ... 6=So  (lokal in Sydney)
    hour = ts_syd.hour
    minute = ts_syd.minute

    if wd == 0:  # Montag
        return (hour, minute) >= (7, 0)
    if wd in (1, 2, 3, 4):  # Di–Fr
        return True
    if wd == 5:  # Samstag
        return (hour, minute) < (7, 0)
    # Sonntag
    return False


def is_valid_trading_time_vectorized(ts_series: pd.Series) -> pd.Series:
    """
    Vektorisierte Variante für pandas.Series (deutlich schneller als apply).
    Erwartet Serie mit datetime64[ns, tz] oder naive -> wird als UTC interpretiert.
    Logik identisch zu is_valid_trading_time(), aber vektorisiert in Sydney-Zeit.
    """
    # Zu UTC (tz-aware) normalisieren
    s = pd.to_datetime(ts_series, utc=True)
    # Nach Sydney konvertieren (DST korrekt)
    s_syd = s.dt.tz_convert(SYDNEY_TZ)
    wd = s_syd.dt.weekday
    hr = s_syd.dt.hour
    mi = s_syd.dt.minute

    # Masken pro Wochentag
    mon = wd == 0
    tue_fri = wd.isin([1, 2, 3, 4])
    sat = wd == 5
    sun = wd == 6

    mon_ok = mon & ((hr > 7) | ((hr == 7) & (mi >= 0)))
    sat_ok = sat & ((hr < 7) | ((hr == 7) & (mi < 0)))  # bis 06:59:59
    return mon_ok | tue_fri | sat_ok  # Sonntag immer False
