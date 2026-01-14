from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional

from strategies._base.base_strategy import Strategy

from hf_engine.infra.logging.log_service import log_service

# Konfigurations-Keys zentral (vermeidet Tippfehler)
SESSION_START_KEY = "session_start"
SESSION_END_KEY = "session_end"
AUTO_CLOSE_KEY = "auto_close"


def is_within_session(current: time, start: time, end: time) -> bool:
    """
    Prüft, ob `current` zwischen `start` und `end` liegt.
    Unterstützt Sessions über Mitternacht (z. B. 22:00–02:00).
    """
    if start < end:
        return start <= current <= end
    if start == end:
        return True
    # über Mitternacht
    return current >= start or current <= end


def _parse_time(value: Any) -> Optional[time]:
    """
    Defensive Normalisierung:
    - akzeptiert bereits `datetime.time`
    - akzeptiert Strings "HH:MM" oder "HH:MM:SS"
    - alles andere -> None
    """
    if isinstance(value, time):
        return value

    if isinstance(value, str):
        try:
            parts = value.strip().split(":")
            if len(parts) == 2:
                h, m = int(parts[0]), int(parts[1])
                return time(h, m)
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                return time(h, m, s)
        except (
            Exception
        ) as exc:  # bewusst breit: Input kann aus externen Configs kommen
            log_service.log_system(
                f"[SessionRunner][WARN] Ungültiges Zeitformat '{value}': {exc}"
            )

    return None


class SessionRunner:
    """
    Kapselt die zeitbasierte Handelslogik einer Strategie.
    Erwartet, dass `strategy.session_times()` ein Dict mit Keys liefert:
      - 'session_start': time | 'HH:MM' | 'HH:MM:SS'
      - 'session_end'  : time | 'HH:MM' | 'HH:MM:SS'
      - 'auto_close'   : time | 'HH:MM' | 'HH:MM:SS' (optional)
    """

    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        sessions = strategy.session_times() or {}
        if not isinstance(sessions, dict):
            log_service.log_system(
                f"[SessionRunner][ERROR] session_times() von {self.strategy.name()} lieferte keinen dict – fallback auf leeres Dict."
            )
            sessions = {}
        # defensive Kopie, um externe Mutationen zu vermeiden
        self.sessions: Dict[str, time] = self._normalize_sessions(dict(sessions))
        self._validate_core_times()

    def _normalize_sessions(self, raw: Dict[str, Any]) -> Dict[str, time]:
        """
        Normalisiert und filtert bekannte Keys auf echte `time`-Objekte.
        Unerkannte Keys bleiben unangetastet (keine Abwärtskompatibilitätsbrüche).
        """
        normalized: Dict[str, time] = {}

        for key in (SESSION_START_KEY, SESSION_END_KEY, AUTO_CLOSE_KEY):
            if key in raw:
                parsed = _parse_time(raw.get(key))
                if parsed is not None:
                    normalized[key] = parsed
                else:
                    log_service.log_system(
                        f"[SessionRunner][WARN] '{key}' in {self.strategy.name()} ist nicht als Zeit interpretierbar."
                    )
        # Bewusst nur bekannte Keys normieren; Rest ignorieren
        return {**raw, **normalized}

    def _validate_core_times(self) -> None:
        """
        Frühzeitige Validierung der Kernzeiten, um spätere Überraschungen zu vermeiden.
        Loggt Warnungen, bricht aber nicht hart ab (kompatibles Verhalten).
        """
        start = self.sessions.get(SESSION_START_KEY)
        end = self.sessions.get(SESSION_END_KEY)

        if not isinstance(start, time) or not isinstance(end, time):
            log_service.log_system(
                f"[SessionRunner][WARN] Unvollständige Session-Definition für {self.strategy.name()} "
                f"(start={start!r}, end={end!r})."
            )
            return

        # Hinweis bei identischem Start/Ende (edge case)
        if start == end:
            log_service.log_system(
                f"[SessionRunner][INFO] Session-Start und -Ende identisch ({start}) für {self.strategy.name()} – "
                "Fenster gilt nur exakt zu diesem Zeitpunkt."
            )

    def _get_time(self, key: str) -> Optional[time]:
        value = self.sessions.get(key)
        if isinstance(value, time):
            return value

        # konsistente, aber nicht-unterbrechende Log-Meldung
        log_service.log_system(
            f"[SessionRunner][WARN] '{key}' fehlt oder ist kein gültiges `time`-Objekt für {self.strategy.name()}"
        )
        return None

    def is_trade_window_open(self, current_dt: datetime) -> bool:
        """
        Prüft, ob das Handelsfenster geöffnet ist.
        Erwartung: `current_dt` und Sessionzeiten befinden sich in derselben (impliziten) Zeitzone.
        """
        start = self._get_time(SESSION_START_KEY)
        end = self._get_time(SESSION_END_KEY)

        if not start or not end:
            log_service.log_system(
                f"[SessionRunner][ERROR] Kein gültiges Handelsfenster für {self.strategy.name()} – returning False."
            )
            return False

        now_t = current_dt.time()
        return is_within_session(now_t, start, end)

    def is_auto_close_time(self, current_dt: datetime) -> bool:
        """
        True, sobald `current_dt.time()` den Auto-Close‑Zeitpunkt erreicht/überschreitet.
        (Ein einzelner Zeitpunkt, kein Fenster; über Mitternacht nicht relevant.)
        """
        auto_close = self._get_time(AUTO_CLOSE_KEY)
        if not auto_close:
            return False
        return current_dt.time() >= auto_close

    def get_session_times(self) -> Dict[str, time]:
        """
        Liefert eine Kopie der geladenen Sessionzeiten (für Tests/Monitoring),
        um ungewollte externe Mutationen zu verhindern.
        """
        return dict(self.sessions)
