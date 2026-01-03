# hf_engine/core/controlling/event_bus.py
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional

from hf_engine.infra.logging.log_service import log_service


class EventType(Enum):
    TIMER_TICK = auto()
    BAR_CLOSE = auto()
    NEWS = auto()
    BROKER_STATUS = auto()
    SHUTDOWN = auto()
    SESSION_OPEN = auto()
    SESSION_CLOSE = auto()
    AUTO_CLOSE = auto()


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    """
    Leichtgewichtiger, threadbasierter Event-Bus.
    - API unverändert
    - Verbesserungen: robustes Logging, thread-sichere Subscriber-Liste,
      Slow-Handler-Warnung, Backpressure-/High-Water-Monitoring,
      defensives Re-Trying bei vollem Puffer
    """

    def __init__(
        self,
        maxsize: int = 10000,
        *,
        warn_highwater_ratio: float = 0.80,  # Warnung ab 80% Queue-Füllstand
        slow_handler_warning_ms: int = 250,  # Warnung bei langsamen Handlern
        drop_warn_every: int = 100,  # Jede 100. Drop meldet Warnung
        handler_workers: int = 8,
        drop_retry_ms: int = 50,
    ):
        self.q: Queue[Event] = Queue(maxsize=maxsize)
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._sub_lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # telemetry
        self.dropped_events: int = 0
        self.processed_events: int = 0
        self.queue_highwater: int = 0

        # config
        self._warn_highwater_ratio = max(0.0, min(1.0, warn_highwater_ratio))
        self._slow_handler_warning_ms = max(1, slow_handler_warning_ms)
        self._drop_warn_every = max(1, drop_warn_every)
        self._handler_workers = max(1, int(handler_workers))
        self._handler_pool: ThreadPoolExecutor | None = None
        self._pending_tasks = 0
        self._pending_lock = threading.Lock()
        self._drop_retry_timeout = max(0.0, float(drop_retry_ms) / 1000.0)
        self._drop_first_logged = False

        # logger (kompatibel zu benutzerdefiniertem log_service)
        self._logger = self._resolve_logger()

    # -------------------- Thread-Pool Management --------------------

    def _ensure_pool(self) -> None:
        if self._handler_pool is None:
            self._handler_pool = ThreadPoolExecutor(
                max_workers=self._handler_workers,
                thread_name_prefix="EventBusHandler",
            )

    def _shutdown_pool(self) -> None:
        pool, self._handler_pool = self._handler_pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
        with self._pending_lock:
            self._pending_tasks = 0

    def _increment_pending(self) -> None:
        with self._pending_lock:
            self._pending_tasks += 1

    def _decrement_pending(self) -> None:
        with self._pending_lock:
            self._pending_tasks = max(0, self._pending_tasks - 1)

    def _submit_handler(self, handler: Callable[[Event], None], ev: Event) -> None:
        try:
            self._ensure_pool()
            self._increment_pending()
            assert self._handler_pool is not None  # für Typchecker
            self._handler_pool.submit(self._run_handler, handler, ev)
        except Exception as exc:
            self._decrement_pending()
            self._log(
                "error",
                f"[EventBus] Handler konnte nicht gestartet werden ({self._handler_name(handler)}): {exc}",
                exc_info=True,
            )

    def _run_handler(self, handler: Callable[[Event], None], ev: Event) -> None:
        start = time.perf_counter()
        try:
            handler(ev)
        except Exception as exc:
            self._log(
                "error",
                f"[EventBus] Handler-Exception in {self._handler_name(handler)} für {ev.type.name}: {exc}",
                exc_info=True,
            )
        finally:
            duration = time.perf_counter() - start
            if duration * 1000.0 >= self._slow_handler_warning_ms:
                self._log(
                    "warning",
                    f"[EventBus] Langsamer Handler {self._handler_name(handler)} für {ev.type.name}: {duration*1000.0:.1f} ms",
                )
            self._decrement_pending()

    # -------------------- Public API (unverändert) --------------------

    def publish(self, event: Event) -> bool:
        """
        Non-blocking Publish. Liefert False, wenn die Queue voll ist.
        Versucht optional einen kurzen Retry (drop_retry_ms), bevor das Event verworfen wird.
        """
        try:
            self.q.put_nowait(event)
            qsize = self.q.qsize()
            if qsize > self.queue_highwater:
                self.queue_highwater = qsize
                self._maybe_warn_highwater(qsize)
            return True
        except Full:
            if self._drop_retry_timeout > 0.0:
                try:
                    self.q.put(event, timeout=self._drop_retry_timeout)
                    qsize = self.q.qsize()
                    if qsize > self.queue_highwater:
                        self.queue_highwater = qsize
                        self._maybe_warn_highwater(qsize)
                    return True
                except Full:
                    pass
                except Exception as retry_exc:
                    self._log(
                        "debug",
                        f"[EventBus] Retry beim Publish fehlgeschlagen: {retry_exc}",
                        exc_info=True,
                    )
            self._register_drop(event)
            return False
        except Exception as exc:
            self._log("error", f"[EventBus] publish-Fehler: {exc}", exc_info=True)
            return False

    def subscribe(self, etype: EventType, handler: Callable[[Event], None]) -> None:
        """
        Registriert einen Handler für einen EventType.
        """
        if not callable(handler):
            raise ValueError("Handler muss callable(Event) sein.")
        with self._sub_lock:
            self._subscribers.setdefault(etype, []).append(handler)

    def start(self) -> None:
        if self._running:
            return
        self._ensure_pool()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EventBus")
        self._thread.start()
        self._log("info", "[EventBus] gestartet")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # Best effort: sofortiger Wakeup per SHUTDOWN-Event
        try:
            self.q.put_nowait(Event(EventType.SHUTDOWN, {}))
        except Full:
            # Fallback: kein Problem – Loop beendet sich nach Timeout durch _running=False
            self._log(
                "warning",
                "[EventBus] SHUTDOWN-Event konnte nicht enqueued werden (Queue voll)",
            )
        except Exception as exc:
            self._log(
                "error", f"[EventBus] stop-Fehler beim Enqueue: {exc}", exc_info=True
            )

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._shutdown_pool()
        self._log("info", "[EventBus] gestoppt")

    # -------------------- Internals --------------------

    def _loop(self):
        while self._running:
            try:
                ev = self.q.get(timeout=0.5)
            except Empty:
                continue
            except Exception as exc:
                self._log(
                    "error", f"[EventBus] get-Fehler aus Queue: {exc}", exc_info=True
                )
                continue

            if ev.type == EventType.SHUTDOWN:
                # Sofortiger Ausstieg
                break

            # Telemetrie: zähle das Event, sobald es aus der Queue entnommen wurde.
            # Wichtig: vor dem asynchronen Handler-Submit, sonst kann ein schneller
            # Handler (ThreadPool) bereits fertig sein, bevor processed_events erhöht ist.
            self.processed_events += 1

            # Snapshot der Handler-Liste unter Lock (Thread-Sicherheit)
            with self._sub_lock:
                handlers = list(self._subscribers.get(ev.type, []))

            for h in handlers:
                self._submit_handler(h, ev)

    # -------------------- Helpers --------------------

    def _register_drop(self, event: Event) -> None:
        self.dropped_events += 1
        try:
            ev_name = event.type.name if hasattr(event, "type") else repr(event)
        except Exception:
            ev_name = "<unknown>"
        qsize = self.q.qsize()
        msg = (
            f"[EventBus] Queue voll – dropped_events={self.dropped_events}, "
            f"event={ev_name}, qsize={qsize}, maxsize={self.q.maxsize}"
        )
        if not self._drop_first_logged:
            self._drop_first_logged = True
            self._log("warning", f"{msg} (erste Meldung)")
            return
        if self.dropped_events % self._drop_warn_every == 0:
            self._log("warning", msg)

    def _maybe_warn_highwater(self, qsize: int) -> None:
        try:
            maxsize = self.q.maxsize or 0
            if maxsize > 0 and (qsize / maxsize) >= self._warn_highwater_ratio:
                self._log(
                    "warning",
                    f"[EventBus] Hoher Queue-Füllstand: qsize={qsize}, "
                    f"maxsize={maxsize}, highwater={self.queue_highwater}",
                )
        except Exception:
            # Telemetrie darf nie stören
            pass

    def _handler_name(self, h: Callable[[Event], None]) -> str:
        return getattr(h, "__qualname__", getattr(h, "__name__", repr(h)))

    def _resolve_logger(self):
        """
        Versucht das benutzerdefinierte Logging zu nutzen.
        Fällt geräuschlos zurück, falls API abweicht.
        """
        try:
            get_logger = getattr(log_service, "get_logger", None)
            if callable(get_logger):
                return get_logger("EventBus")
        except Exception:
            pass
        # Falls log_service selbst die Logging-Methoden bereitstellt:
        return log_service

    def _log(self, level: str, msg: str, **kwargs) -> None:
        """
        Sicheres Logging ohne harte Annahmen über log_service-API.
        Unterstützt u.a. .info/.warning/.error und optional exc_info.
        """
        try:
            logger = self._logger
            method = getattr(logger, level, None)
            if callable(method):
                method(msg, **kwargs)
                return
            # Fallback: versuche direkt auf log_service.* zu loggen
            method = getattr(log_service, level, None)
            if callable(method):
                method(msg, **kwargs)
        except Exception:
            # Logging darf niemals die Engine stören
            pass
