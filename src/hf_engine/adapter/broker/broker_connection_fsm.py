# hf_engine/adapter/broker/broker_connection_fsm.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from threading import Lock
from typing import Callable, Optional

try:
    from hf_engine.infra.logging.log_service import log_service  # dein Wrapper

    _LOGGER = log_service.logger
except Exception:  # Fallback (z. B. in isolierten Tests)
    import logging

    _LOGGER = logging.getLogger("HFEngine.broker_connection_fsm")
    if not _LOGGER.handlers:
        logging.basicConfig(level=logging.INFO)


class BrokerStatus(Enum):
    INITIALIZING = auto()
    CONNECTED = auto()
    DEGRADED = auto()
    DISCONNECTED = auto()


@dataclass
class ReconnectPolicy:
    base_delay_sec: float = 1.0
    max_delay_sec: float = 30.0
    jitter_frac: float = 0.25  # 0.0 .. 1.0
    max_retries: int = 10
    timeout_sec: float = 5.0
    cooldown_after_fail_sec: float = 60.0  # Sperrzeit nach Ausschöpfung der Retries


class BrokerConnectionFSM:
    """
    Schlanke FSM zur Sicherstellung einer Brokerverbindung mit
    Exponential Backoff + Jitter, Timeout und Cooldown nach Max-Retries.
    """

    def __init__(self, broker, policy: Optional[ReconnectPolicy] = None):
        self.broker = broker
        self.policy = policy or ReconnectPolicy()
        self.status = BrokerStatus.DISCONNECTED
        self._retries = 0
        self._last_fail_time = 0.0
        self._lock = Lock()
        try:
            self._log = _LOGGER.getChild("BrokerConnectionFSM")
        except Exception:
            self._log = _LOGGER

    # -------- internals --------
    def _set_status(self, new_status: BrokerStatus) -> None:
        if new_status != self.status:
            self._log.debug(
                "Statuswechsel: %s -> %s", self.status.name, new_status.name
            )
            self.status = new_status
            if new_status == BrokerStatus.CONNECTED:
                # Bei erfolgreichem Connect Retries zurücksetzen
                self._retries = 0

    def _call_with_timeout(self, func: Callable[[], bool], timeout_sec: float) -> bool:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return bool(future.result(timeout=timeout_sec))
            except concurrent.futures.TimeoutError:
                self._log.warning(
                    "Broker-ensure_connection Timeout nach %.2fs", timeout_sec
                )
                # Wichtig: Future abbrechen, um Threads nicht zu leaken
                future.cancel()
                return False

    # -------- public --------

    def set_status(self, status: BrokerStatus) -> None:
        self._set_status(status)

    def ensure(self) -> bool:
        """
        Stellt sicher, dass die Brokerverbindung besteht.
        Rückgabe:
            True  -> Verbindung steht
            False -> (noch) nicht verbunden; Backoff/Retry/cooldown wird gemanagt
        """
        sleep_time: float = 0.0  # außerhalb des Locks schlafen

        with self._lock:
            now = time.time()

            # Cooldown nach ausgereizten Retries
            if (
                self._retries >= self.policy.max_retries
                and (now - self._last_fail_time) < self.policy.cooldown_after_fail_sec
            ):
                if self.status != BrokerStatus.DISCONNECTED:
                    self._set_status(BrokerStatus.DISCONNECTED)
                self._log.debug(
                    "Im Cooldown: %.1fs verbleibend",
                    self.policy.cooldown_after_fail_sec - (now - self._last_fail_time),
                )
                return False

            # Bereits verbunden?
            if self.status == BrokerStatus.CONNECTED:
                return True

            # Verbindung (erneut) aufbauen
            try:
                ok = self._call_with_timeout(
                    self.broker.ensure_connection, self.policy.timeout_sec
                )
            except Exception as e:  # Grobe Fehlerklassifizierung
                # Auth-/Konfigfehler sollten nicht gespamt werden:
                self._log.exception("Fehler beim ensure_connection: %s", e)
                ok = False

            if ok:
                self._set_status(BrokerStatus.CONNECTED)
                self._log.info(
                    "Broker verbunden (Retries seit letztem Erfolg: %d)", self._retries
                )
                return True

            # Fehlschlag
            self._last_fail_time = now
            self._retries += 1

            # Bei ersten Fehlschlägen: DEGRADED signalisieren
            if self._retries < self.policy.max_retries:
                self._set_status(BrokerStatus.DEGRADED)
            else:
                # Retry-Limit erreicht -> DISCONNECTED und Cooldown aktiv
                self._set_status(BrokerStatus.DISCONNECTED)
                self._log.warning(
                    "Retry-Limit erreicht (%d). Wechsel zu DISCONNECTED, Cooldown %.0fs.",
                    self.policy.max_retries,
                    self.policy.cooldown_after_fail_sec,
                )
                return False

            # Exponential Backoff + Jitter berechnen (ohne im Lock zu schlafen)
            delay = min(
                self.policy.base_delay_sec * (2 ** (self._retries - 1)),
                self.policy.max_delay_sec,
            )
            jitter_component = delay * self.policy.jitter_frac
            # symmetrischer Jitter um 'delay'
            sleep_time = max(
                0.0, delay + random.uniform(-jitter_component, jitter_component)
            )

            self._log.debug(
                "Reconnect-Versuch %d in %.2fs (delay=%.2fs, jitter_frac=%.2f)",
                self._retries,
                sleep_time,
                delay,
                self.policy.jitter_frac,
            )

        # Wichtig: Schlaf außerhalb des Locks
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        return False
