# hf_engine/strategies/mean_reversion_z_score/live/utils.py
"""Utility functions for mean_reversion_z_score live strategy."""

from __future__ import annotations

from datetime import datetime
from datetime import time as dtime
from datetime import timedelta, timezone

from strategies._base.domain_types import MarketSignal


def _safe_round(value: float, ndigits: int = 5) -> float:
    """Round a float value safely with fallback on failure."""
    try:
        return round(float(value), ndigits)
    except Exception:
        return float(value)


def signal_long_market(sl: float, tp: float) -> MarketSignal:
    """
    Liefert ein standardisiertes Market-Buy-Signal.

    Args:
        sl: Stop-loss price level.
        tp: Take-profit price level.

    Returns:
        Standardized market buy signal dictionary.
    """
    return {
        "direction": "buy",
        "sl": _safe_round(sl, 5),
        "tp": _safe_round(tp, 5),
        "order_type": "market",
    }


def signal_short_market(sl: float, tp: float) -> MarketSignal:
    """
    Liefert ein standardisiertes Market-Sell-Signal.

    Args:
        sl: Stop-loss price level.
        tp: Take-profit price level.

    Returns:
        Standardized market sell signal dictionary.
    """
    return {
        "direction": "sell",
        "sl": _safe_round(sl, 5),
        "tp": _safe_round(tp, 5),
        "order_type": "market",
    }


def _tf_to_timedelta(tf: str) -> timedelta:
    tf = tf.upper()
    if tf.startswith("M"):
        return timedelta(minutes=int(tf[1:]))
    if tf.startswith("H"):
        return timedelta(hours=int(tf[1:]))
    if tf.startswith("D"):
        return timedelta(days=int(tf[1:]))
    raise ValueError(f"Unsupported timeframe: {tf}")


def _get_next_bar(ts: datetime, tf: str) -> datetime:
    """
    Calculate the next bar open time for the given timestamp and timeframe.

    Args:
        ts: Current timestamp.
        tf: Timeframe string (M5, M15, M30, H1, H4).

    Returns:
        Datetime of the next bar open.

    Raises:
        ValueError: If timeframe is not supported.
    """
    tf = tf.upper()
    base = ts.replace(second=0, microsecond=0)

    if tf == "M5":
        minute = ((base.minute // 5) + 1) * 5
        if minute >= 60:
            base += timedelta(hours=1)
            minute = 0
        return base.replace(minute=minute)

    if tf == "M15":
        minute = ((base.minute // 15) + 1) * 15
        if minute >= 60:
            base += timedelta(hours=1)
            minute = 0
        return base.replace(minute=minute)

    if tf == "M30":
        if base.minute < 30:
            return base.replace(minute=30)
        base += timedelta(hours=1)
        return base.replace(minute=0)

    if tf == "H1":
        return base.replace(minute=0) + timedelta(hours=1)

    if tf == "H4":
        hour = base.hour
        next_4h = ((hour // 4) + 1) * 4
        if next_4h >= 24:
            next_4h = 0
            base += timedelta(days=1)
        return base.replace(hour=next_4h, minute=0)

    raise ValueError(f"Unsupported timeframe: {tf}")


def get_next_entry_time(
    last_entry: datetime, tf: str, cooldown_minutes: int
) -> datetime:
    """
    Gibt den frühestmöglichen erlaubten Zeitpunkt für den nächsten Trade zurück.

    Änderung: Der Entry‑Cooldown wird auf das Bar‑Open normalisiert, d. h.
    die Cooldown‑Dauer wird nicht mehr zum exakten Entry‑Zeitstempel (inkl. Sekunden)
    addiert, sondern zum Open der jeweiligen Bar. Dadurch entfallen Sekunden‑Offsets
    (z. B. 14:30:04 + 10min → 14:40:00 bei M5).

    Logik:
    - time_after_cooldown = bar_open(last_entry) + cooldown
    - time_after_next_candle = nächster Bar‑Beginn nach last_entry
    - Rückgabe = max(time_after_cooldown, time_after_next_candle)
    """
    tf = tf.upper()

    # Mapping: Mindestwartezeit nach letztem Entry
    wait_durations = {
        "M5": timedelta(minutes=cooldown_minutes),
        "M15": timedelta(minutes=cooldown_minutes),
        "M30": timedelta(minutes=cooldown_minutes),
        "H1": timedelta(minutes=cooldown_minutes),
        "H4": timedelta(minutes=cooldown_minutes),
    }

    min_wait = wait_durations.get(tf, timedelta(minutes=5))  # Default fallback

    # Normalisiere auf Bar‑Open: bar_open = next_bar(last_entry) - tf_dauer
    try:
        bar_open = _get_next_bar(last_entry, tf) - _tf_to_timedelta(tf)
    except Exception:
        # Fallback: falls TF nicht unterstützt, verwende Sekundengenauigkeit wie zuvor
        bar_open = last_entry.replace(second=0, microsecond=0)

    time_after_cooldown = bar_open + min_wait

    # Zusätzlich: Falls die Cooldown-Dauer nicht exakt auf eine Bar‑Grenze fällt,
    # auf den nächsten Bar‑Open 'snappen', damit Entries exakt zum Bar‑Open erlaubt sind.
    try:
        tf_delta = _tf_to_timedelta(tf)
        current_bar_open = _get_next_bar(time_after_cooldown, tf) - tf_delta
        if time_after_cooldown != current_bar_open:
            time_after_cooldown = _get_next_bar(time_after_cooldown, tf)
    except Exception:
        # Keine Snapping‑Anpassung möglich (unbekanntes TF) – belasse Sekunden‑Rundung
        pass

    time_after_next_candle = _get_next_bar(last_entry, tf)

    # Nimm das spätere der beiden
    return max(time_after_cooldown, time_after_next_candle)


def get_next_entry_after_exit(last_exit: datetime, tf: str) -> datetime:
    """
    Erzwingt eine vollständige Bar Pause nach dem Exit.
    Beispielsweise bei M5: Exit 14:28 → früheste Rückkehr 14:35.
    """
    next_bar_open = _get_next_bar(last_exit, tf)
    return next_bar_open + _tf_to_timedelta(tf)


def in_session_utc(now_dt: datetime, session_start: dtime, session_end: dtime) -> bool:
    """
    Prüft, ob now_dt (als aware datetime) innerhalb eines täglichen Sessionfensters in UTC liegt.

    Args:
        now_dt: Timezone-aware datetime to check.
        session_start: Session start time in UTC.
        session_end: Session end time in UTC.

    Returns:
        True if now_dt is within the session window.

    Raises:
        ValueError: If now_dt is not timezone-aware.
    """
    if now_dt.tzinfo is None:
        raise ValueError("now_dt must be timezone-aware")

    t = now_dt.astimezone(timezone.utc).time()

    # Voller Handelstag, falls Start == Ende
    if session_start == session_end:
        return True

    if session_start < session_end:
        # normales Fenster:     [start, end]
        return session_start <= t <= session_end
    else:
        # wrap-around-Fenster:  [start, 24h) U [00:00, end]
        return t >= session_start or t <= session_end
