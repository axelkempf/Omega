# session_state.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, Optional

from hf_engine.infra.config.time_utils import from_utc_to_broker, now_utc
from hf_engine.infra.logging.log_service import log_service

# Thread-safe, in‑memory Blockspeicher (UTC)
_session_blocks: Dict[str, datetime] = {}
_lock = RLock()

_logger = log_service.logger


def _parse_hhmm(until_time: str) -> tuple[int, int]:
    """Strictes Parsen von 'HH:MM' mit Wertebereichsprüfung."""
    parts = until_time.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format '{until_time}'. Expected 'HH:MM'.")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid time digits in '{until_time}'.") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(
            f"Invalid time '{until_time}'. Hour 0–23, minute 0–59 required."
        )
    return hour, minute


def block_strategy_for_session(
    strategy_name: str, until_time: str, *, time_in: str = "UTC"
) -> None:
    """
    Blockiert eine Strategie bis zur angegebenen Uhrzeit der laufenden/folgenden Session.

    Args:
        strategy_name: Name der Strategie.
        until_time: Zielzeit als 'HH:MM'.
        time_in: Referenzzeitzone der Eingabezeit:
                 - "UTC"    -> until_time wird als UTC interpretiert (Standard).
                 - "BROKER" -> until_time bezieht sich auf Broker-Zeitzone.
    """
    hour, minute = _parse_hhmm(until_time)

    with _lock:
        now_u = now_utc()

        if time_in.upper() == "UTC":
            expiry_u = now_u.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if expiry_u <= now_u:
                expiry_u += timedelta(days=1)

        elif time_in.upper() == "BROKER":
            # Berechnung in Broker-Zone, anschließende Rückkonvertierung nach UTC.
            now_b = from_utc_to_broker(now_u)
            if now_b.tzinfo is None:
                # Defensive: Falls from_utc_to_broker naiv liefern sollte, erzwinge keine Annahmen
                raise RuntimeError(
                    "from_utc_to_broker returned a naive datetime; tzinfo required."
                )
            expiry_b = now_b.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if expiry_b <= now_b:
                expiry_b += timedelta(days=1)
            expiry_u = expiry_b.astimezone(timezone.utc)

        else:
            raise ValueError(f"Unsupported time_in '{time_in}'. Use 'UTC' or 'BROKER'.")

        _session_blocks[strategy_name] = expiry_u
        _logger.info(
            "[SessionState] Strategy '%s' blocked until %s (UTC) [time_in=%s]",
            strategy_name,
            expiry_u.isoformat(),
            time_in.upper(),
        )


def is_strategy_blocked(strategy_name: str) -> bool:
    """
    Prüft, ob eine Strategie aktuell blockiert ist.
    Abgelaufene Sperren werden dabei automatisch entfernt.
    """
    with _lock:
        expiry = _session_blocks.get(strategy_name)
        if not expiry:
            return False

        now_u = now_utc()
        if now_u >= expiry:
            _session_blocks.pop(strategy_name, None)
            _logger.debug(
                "[SessionState] Strategy '%s' unblocked (expired at %s UTC).",
                strategy_name,
                expiry.isoformat(),
            )
            return False
        return True


def get_block_expiry_utc(strategy_name: str) -> Optional[datetime]:
    """Gibt die Ablaufzeit der Sperre in UTC zurück (oder None)."""
    with _lock:
        return _session_blocks.get(strategy_name)


def clear_block(strategy_name: str) -> bool:
    """Hebt eine gesetzte Sperre manuell auf. Gibt True zurück, wenn eine Sperre existierte."""
    with _lock:
        existed = strategy_name in _session_blocks
        if existed:
            expiry = _session_blocks.pop(strategy_name)
            _logger.info(
                "[SessionState] Strategy '%s' manual unblock (was until %s UTC).",
                strategy_name,
                expiry.isoformat(),
            )
        return existed


def list_blocks_snapshot() -> Dict[str, datetime]:
    """Snapshot aller aktiven Sperren (UTC). Nur für Diagnose/Monitoring."""
    with _lock:
        return dict(_session_blocks)


__all__ = [
    "block_strategy_for_session",
    "is_strategy_blocked",
    "get_block_expiry_utc",
    "clear_block",
    "list_blocks_snapshot",
]
