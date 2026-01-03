# hf_engine/core/controlling/position_monitor_controller.py
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from hf_engine.infra.logging.log_service import log_service


class PositionMonitorController:
    """
    Überwacht registrierte Positions-Manager ausschließlich im Step-Modus.
    Erwartet, dass jeder Manager:
      - ein Attribut 'ticket_id' (int) besitzt
      - eine Methode 'monitor_step() -> bool' implementiert
      - optional 'stop_monitoring()' oder 'cancel()' für Shutdown implementiert
    """

    def __init__(self, interval_sec: float = 1.0, job_timeout_sec: float = 300.0):
        self._interval = float(interval_sec)
        self._timeout = float(job_timeout_sec)
        self._managers: Dict[int, object] = {}  # ticket_id -> manager
        self._hb: Dict[int, float] = {}  # ticket_id -> last heartbeat ts
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---------------- Lifecycle ----------------

    def start(self) -> None:
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                log_service.log_system("[PosMon] ⚠️ bereits gestartet")
                return
            self._stop.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="PositionMonitor"
            )
            self._thread.start()
        log_service.log_system("[PosMon] ✅ gestartet")

    def stop_all(self) -> None:
        # Signal & Join
        self._stop.set()
        t = None
        with self._lock:
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

        # Best-effort graceful shutdown der Manager
        with self._lock:
            managers_snapshot = list(self._managers.items())
        for tid, m in managers_snapshot:
            try:
                if hasattr(m, "stop_monitoring"):
                    m.stop_monitoring()
                elif hasattr(m, "cancel"):
                    m.cancel()
            except Exception as e:
                log_service.log_system(
                    f"[PosMon] ⚠️ Fehler beim Stop von {tid}: {e}", level="ERROR"
                )

        with self._lock:
            self._managers.clear()
            self._hb.clear()
            self._running = False
            self._thread = None

        log_service.log_system("[PosMon] 🛑 gestoppt")

    # ---------------- Manager-Verwaltung ----------------

    def add_manager(self, manager) -> None:
        tid = getattr(manager, "ticket_id", None)
        if tid is None:
            log_service.log_system(
                "[PosMon] ❌ Manager ohne 'ticket_id'", level="ERROR"
            )
            return

        # Hard check: monitor_step muss existieren
        if not hasattr(manager, "monitor_step"):
            log_service.log_system(
                f"[PosMon] ❌ Manager {tid} implementiert kein 'monitor_step()'",
                level="ERROR",
            )
            return

        # Optional: initialisieren/aktivieren (keine Legacy-Threads!)
        try:
            if hasattr(manager, "start_monitoring"):
                manager.start_monitoring()
        except Exception as e:
            log_service.log_system(
                f"[PosMon] ❌ start_monitoring() Fehler für {tid}: {e}", level="ERROR"
            )
            return

        with self._lock:
            self._managers[tid] = manager
            self._hb[tid] = time.time()
        log_service.log_system(f"[PosMon] ➕ Manager {tid} registriert")

    def remove_manager(self, ticket_id: int, reason: str = "done") -> None:
        mgr = None
        with self._lock:
            mgr = self._managers.pop(ticket_id, None)
            self._hb.pop(ticket_id, None)

        # Best-effort graceful stop für diesen Manager
        if mgr:
            try:
                if hasattr(mgr, "stop_monitoring"):
                    mgr.stop_monitoring()
                elif hasattr(mgr, "cancel"):
                    mgr.cancel()
            except Exception as e:
                log_service.log_system(
                    f"[PosMon] ⚠️ Fehler beim Entfernen von {ticket_id}: {e}",
                    level="ERROR",
                )

        log_service.log_system(f"[PosMon] 🧹 Manager {ticket_id} entfernt ({reason})")

    # ---------------- Überwachungsschleife ----------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            start = time.time()
            to_remove = []

            # Snapshot, um Iteration gegen Mutationen zu schützen
            with self._lock:
                items = list(self._managers.items())

            for tid, m in items:
                prev_hb = None
                with self._lock:
                    prev_hb = self._hb.get(tid, start)

                try:
                    # ausschliesslich Step-Modus
                    done = bool(m.monitor_step())

                    now2 = time.time()
                    # Timeout prüfen VOR Heartbeat-Update (Timeout gewinnt)
                    if now2 - prev_hb > self._timeout:
                        to_remove.append((tid, "timeout"))
                        continue

                    # Heartbeat nur aktualisieren, wenn kein Timeout und keine Exception
                    with self._lock:
                        self._hb[tid] = now2

                    if done:
                        to_remove.append((tid, "done"))

                except Exception as e:
                    log_service.log_system(
                        f"[PosMon] ⚠️ Fehler in monitor_step() von {tid}: {e}",
                        level="ERROR",
                    )
                    to_remove.append((tid, "error"))

            # Entfernen ausserhalb der Iteration
            for tid, reason in to_remove:
                self.remove_manager(tid, reason)

            # Schlagweite einhalten
            elapsed = time.time() - start
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                self._stop.wait(timeout=sleep_for)
