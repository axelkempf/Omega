# hf_engine/core/controlling/multi_strategy_controller.py
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

import hf_engine.core.risk.news_filter as news_filter
from hf_engine.adapter.broker.broker_connection_fsm import (
    BrokerConnectionFSM,
    BrokerStatus,
    ReconnectPolicy,
)
from hf_engine.core.controlling.event_bus import Event, EventBus, EventType
from hf_engine.core.controlling.session_runner import SessionRunner
from hf_engine.core.controlling.strategy_runner import StrategyRunner
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.log_service import log_service
from hf_engine.infra.monitoring.telegram_bot import (
    send_telegram_message,
    send_watchdog_telegram_message,
)

_TF_MIN_MAP = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def _time_hm(t):
    return (t.hour, t.minute)


@dataclass(frozen=True)
class _BarWatchSpec:
    provider: Any
    timeframe: str
    symbols: Tuple[str, ...]

    @property
    def anchor_symbol(self) -> str:
        return self.symbols[0]


class MultiStrategyController:
    """
    Orchestriert mehrere Strategien mit jeweils eigener Broker-Anbindung und Runner-Logik.
    Jede Strategie läuft in einem eigenen Thread unabhängig voneinander.
    Verbesserungen:
      - Robuste Fehlerlogs statt stummer excepts
      - Dynamische Heartbeat-Schwellen je Timeframe
      - Automatischer Thread-Restart für Runner
      - Hintergrundthreads registriert + optionaler Neustart
      - Latenz-Metriken (p50/p95/p99) im HealthMonitor
      - Sauberer Shutdown (Join auf Threads)
    """

    def __init__(
        self,
        runners: List[StrategyRunner],
        event_bus: EventBus | None = None,
        broker_watchdog_interval_sec: float = 0.3,
        *,
        barclose_poll_interval_sec: float = 0.3,
    ):
        self.runners = runners
        self.threads: list[threading.Thread] = []  # Runner-Threads (backward-compat)
        self._runner_threads: Dict[StrategyRunner, threading.Thread] = {}
        self._bg_threads: Dict[str, threading.Thread] = {}
        self._bg_specs: Dict[str, Tuple[Callable, tuple]] = {}
        self._restart_counts: Dict[str, int] = {}

        self.running = False
        self.watchdog_interval = 60  # Baseline für generische Checks
        self.event_bus = event_bus or EventBus()
        self._bus_started = False

        # Heartbeats & Latenzen threadsicher
        self._hb_lock = threading.RLock()
        self.last_heartbeat = {runner: time.time() for runner in self.runners}

        self._latency_lock = threading.RLock()
        self._latency_samples: deque[float] = deque(maxlen=5000)

        self._stop = threading.Event()
        self.barclose_poll_interval = max(0.05, float(barclose_poll_interval_sec))
        self._barclose_specs: List[_BarWatchSpec] = []
        self.broker_watchdog_interval = max(0.05, float(broker_watchdog_interval_sec))
        self.watchdog_status_interval = 1800.0  # 30 Minuten

        # FSM auf gemeinsamem Broker (Annahme: gleicher Broker für alle Runner)
        self._broker = runners[0].broker if runners else None
        self._fsm = BrokerConnectionFSM(self._broker, ReconnectPolicy())
        self._fsm.set_status(BrokerStatus.INITIALIZING)

        self._ensure_bus_started()
        for r in self.runners:
            self._subscribe_runner_events(r)

        # BROKER_STATUS Events verarbeiten
        def _on_broker_status(ev: Event):
            st = (ev.payload or {}).get("status", "").upper()
            try:
                new = BrokerStatus[st]
            except Exception as e:
                log_service.log_system(
                    f"[BrokerFSM] Ungültiger Status '{st}': {e}", level="ERROR"
                )
                return
            prev = self._fsm.status
            self._fsm.set_status(new)
            if new != prev:
                log_service.log_system(f"[BrokerFSM] Status {prev.name} → {new.name}")
                try:
                    if new == BrokerStatus.DISCONNECTED:
                        msg = "🚨 *Broker DISCONNECTED* – Reconnect wird versucht."
                        send_telegram_message(msg)
                        send_watchdog_telegram_message(msg)
                    elif new == BrokerStatus.DEGRADED:
                        msg = "⚠️ *Broker DEGRADED* – Latenz/Fehler, Reconnect einschleifen."
                        send_telegram_message(msg)
                        send_watchdog_telegram_message(msg)
                    elif new == BrokerStatus.CONNECTED:
                        send_telegram_message("✅ *Broker CONNECTED*")
                except Exception as e:
                    # Telegram darf kein kritischer Pfad sein
                    log_service.log_system(
                        f"[BrokerFSM] Telegram-Notify fehlgeschlagen: {e}",
                        level="WARNING",
                    )

        self.event_bus.subscribe(EventType.BROKER_STATUS, _on_broker_status)

    # ---------- Interna / Utilities ----------

    def _start_bg(self, name: str, target: Callable, *args) -> None:
        """Startet und registriert einen Background-Thread idempotent."""
        if name in self._bg_threads and self._bg_threads[name].is_alive():
            return
        t = threading.Thread(target=target, args=args, daemon=True, name=name)
        self._bg_threads[name] = t
        self._bg_specs[name] = (target, args)
        t.start()

    def _restart_bg(self, name: str) -> None:
        """Versucht, einen Background-Thread neu zu starten (mit einfacher Rate-Limitierung)."""
        spec = self._bg_specs.get(name)
        if not spec:
            return
        cnt = self._restart_counts.get(name, 0)
        if cnt >= 5:
            log_service.log_system(
                f"[HealthMonitor] ⚠️ Neustart-Limit erreicht für {name}", level="WARNING"
            )
            return
        self._restart_counts[name] = cnt + 1
        log_service.log_system(
            f"[HealthMonitor] ♻️ Starte {name} neu (#{self._restart_counts[name]})",
            level="WARNING",
        )
        target, args = spec
        self._start_bg(name, target, *args)

    def _safe_publish(self, event: Event) -> bool:
        """Publish mit Backpressure-Logging."""
        try:
            ok = self.event_bus.publish(event)
            if not ok:
                log_service.log_system(
                    "[EventBus] ⚠️ Event verworfen (Backpressure)", level="WARNING"
                )
            return ok
        except Exception as e:
            log_service.log_system(
                f"[EventBus] Fehler beim Publish {event.type}: {e}", level="ERROR"
            )
            return False

    def _ensure_bus_started(self) -> None:
        """Startet den EventBus einmalig (idempotent)."""
        if self._bus_started:
            return
        try:
            self.event_bus.start()
        except Exception as e:
            # Kein harter Fehler: falls EventBus kein start() braucht.
            log_service.log_system(
                f"[EventBus] start() nicht verfügbar oder fehlgeschlagen: {e}",
                level="DEBUG",
            )
        self._bus_started = True

    def _rebuild_barclose_specs(self) -> None:
        """
        Baut die BarWatch-Spezifikation aus den aktuell registrierten Runnern.
        Gruppiert nach (DataProvider, Timeframe) und vereint alle Symbole.
        """
        specs: List[_BarWatchSpec] = []
        seen: Set[Tuple[int, str, str]] = set()

        for runner in self.runners:
            provider = getattr(runner, "data_provider", None)
            if provider is None:
                continue
            tf = getattr(
                runner.strategy, "timeframe", None
            ) or runner.strategy.config.get("timeframe")
            if not isinstance(tf, str):
                continue
            tf_norm = tf.strip().upper()
            if not tf_norm or tf_norm not in _TF_MIN_MAP:
                continue
            symbols = []
            try:
                raw_symbols = runner.strategy.config.get("symbols", [])
                if isinstance(raw_symbols, (list, tuple, set)):
                    for sym in raw_symbols:
                        if isinstance(sym, str) and sym.strip():
                            cleaned = sym.strip()
                            if cleaned not in symbols:
                                symbols.append(cleaned)
            except Exception:
                continue
            if not symbols:
                continue
            for sym in symbols:
                sym_clean = sym.strip()
                if not sym_clean:
                    continue
                dedup_key = (id(provider), tf_norm, sym_clean.upper())
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                specs.append(
                    _BarWatchSpec(
                        provider=provider,
                        timeframe=tf_norm,
                        symbols=(sym_clean,),
                    )
                )

        self._barclose_specs = specs

    def _tf_minutes_for_runner(self, runner: StrategyRunner) -> int | None:
        tf = getattr(runner.strategy, "timeframe", None) or runner.strategy.config.get(
            "timeframe"
        )
        if isinstance(tf, str):
            tf_norm = tf.strip().upper()
            if tf_norm:
                return _TF_MIN_MAP.get(tf_norm)
        return None

    def _heartbeat_threshold_sec(self, runner: StrategyRunner) -> float:
        """
        Dynamik:
          - wenn TF bekannt: 2 × TF‑Dauer (konservativ)
          - sonst: 2 × watchdog_interval (Default)
        """
        tf_min = self._tf_minutes_for_runner(runner)
        if tf_min:
            return max(60.0, float(tf_min) * 60.0 * 2.0)
        return float(self.watchdog_interval) * 2.0

    # ---------- Event-Wiring ----------

    def _subscribe_runner_events(self, runner: StrategyRunner) -> None:
        """Registriert Event‑Handler für einen Runner (idempotent genug für Tests)."""

        def _deliver(ev: Event):
            start = time.time()
            try:
                runner.on_event(ev)
                with self._hb_lock:
                    self.last_heartbeat[runner] = time.time()
            except Exception as e:
                log_service.log_system(
                    f"[Deliver] Fehler in {runner.strategy.name()} bei {ev.type}: {e}",
                    level="ERROR",
                )
            finally:
                try:
                    with self._latency_lock:
                        self._latency_samples.append(time.time() - start)
                except Exception:
                    # Latenzmetriken dürfen nie stören
                    pass

        self.event_bus.subscribe(EventType.TIMER_TICK, _deliver)
        self.event_bus.subscribe(EventType.BAR_CLOSE, _deliver)

    # ---------- Worker Loops ----------

    def _run_strategy_loop(self, runner: StrategyRunner):
        thread_name = threading.current_thread().name
        log_service.log_system(
            f"[{thread_name}] gestartet für {runner.strategy.name()}"
        )

        self._ensure_bus_started()

        while self.running and not self._stop.is_set():
            # Business-Logik findet im Event-Handler statt; Loop dient als Lebenszeichen
            time.sleep(0.5)

    def _monitor_health(self):
        """Überwacht Threads, EventBus-Backpressure und berichtet Latenz-Quantile."""
        self._ensure_bus_started()
        last_latency_report = 0.0
        while self.running and not self._stop.is_set():
            # Runner-Threads: Neustart bei Ausfall
            for r, t in list(self._runner_threads.items()):
                if not t.is_alive():
                    log_service.log_system(
                        f"[HealthMonitor] ⚠️ Runner-Thread abgestürzt: {t.name}",
                        level="WARNING",
                    )
                    try:
                        nt = threading.Thread(
                            target=self._run_strategy_loop,
                            args=(r,),
                            daemon=True,
                            name=f"RunnerThread-{r.strategy.name()}",
                        )
                        self._runner_threads[r] = nt
                        self.threads.append(nt)  # Backward-compat
                        nt.start()
                        log_service.log_system(
                            f"[HealthMonitor] ♻️ Runner neu gestartet: {nt.name}",
                            level="WARNING",
                        )
                    except Exception as e:
                        log_service.log_system(
                            f"[HealthMonitor] ❌ Runner-Neustart fehlgeschlagen: {e}",
                            level="ERROR",
                        )

            # Background-Threads: optionaler Neustart
            for name, th in list(self._bg_threads.items()):
                if not th.is_alive():
                    log_service.log_system(
                        f"[HealthMonitor] ⚠️ Background-Thread down: {name}",
                        level="WARNING",
                    )
                    self._restart_bg(name)

            # EventBus Backpressure
            try:
                dropped = getattr(self.event_bus, "dropped_events", 0)
                if dropped > 0:
                    hw = getattr(self.event_bus, "queue_highwater", 0)
                    log_service.log_system(
                        f"[EventBus] ⚠️ Dropped={dropped} HighWater={hw}",
                        level="WARNING",
                    )
            except Exception:
                pass

            # Latenz-Report alle ~60s
            now_ts = time.time()
            if now_ts - last_latency_report >= 60.0:
                last_latency_report = now_ts
                try:
                    with self._latency_lock:
                        if self._latency_samples:
                            samples = sorted(self._latency_samples)
                            n = len(samples)

                            def q(p: float) -> float:
                                if n == 1:
                                    return samples[0]
                                idx = min(n - 1, max(0, int(p * (n - 1))))
                                return samples[idx]

                            p50, p95, p99 = q(0.50), q(0.95), q(0.99)
                            log_service.log_system(
                                f"[Latency] p50={p50:.3f}s p95={p95:.3f}s p99={p99:.3f}s over n={n}"
                            )
                except Exception:
                    pass

            time.sleep(5)

    def _watchdog_loop(self):
        """Warnung bei ausbleibenden Heartbeats je Runner mit TF-adaptivem Threshold."""
        self._ensure_bus_started()
        while self.running and not self._stop.is_set():
            now = now_utc()
            for runner in self.runners:
                try:
                    with self._hb_lock:
                        last = self.last_heartbeat.get(runner, 0.0)
                    threshold = self._heartbeat_threshold_sec(runner)
                    if time.time() - last > threshold:
                        log_service.log_system(
                            f"[Watchdog] ⚠️ Keine Aktivität von {runner.strategy.name()} "
                            f"[TZ:{now:%Y-%m-%d %H:%M:%S}] (>{threshold:.0f}s)",
                            level="WARNING",
                        )
                except Exception as e:
                    log_service.log_system(f"[Watchdog] Fehler: {e}", level="ERROR")
            time.sleep(self.watchdog_interval)

    def _watchdog_telegram_loop(self):
        """
        Sendet alle 30 Minuten pro Runner einen Watchdog‑Status per Telegram:
        - OK, wenn Heartbeat innerhalb des erwarteten Fensters liegt
        - INAKTIV, wenn das Threshold überschritten wird.
        """
        self._ensure_bus_started()
        interval = float(self.watchdog_status_interval)
        while self.running and not self._stop.is_set():
            now = now_utc()
            runner_status: list[tuple[str, bool, int, int]] = []
            any_inactive = False

            for runner in self.runners:
                try:
                    with self._hb_lock:
                        last = self.last_heartbeat.get(runner, 0.0)
                    delay = max(0.0, time.time() - last)
                    threshold = self._heartbeat_threshold_sec(runner)
                    ok = delay <= threshold
                    name = runner.strategy.name()

                    runner_status.append((name, ok, int(delay), int(threshold)))
                    if not ok:
                        any_inactive = True
                except Exception as e:
                    log_service.log_system(
                        f"[WatchdogTelegram] Fehler im Loop für Runner {getattr(runner.strategy, 'name', lambda: 'unknown')()}: {e}",
                        level="ERROR",
                    )

            try:
                if not runner_status:
                    time.sleep(interval)
                    continue

                if any_inactive:
                    lines = [
                        "🚨 [Watchdog] Terminal-Status – INAKTIVE Runner "
                        f"[TZ:{now:%Y-%m-%d %H:%M:%S}]"
                    ]
                    for name, ok, delay, threshold in runner_status:
                        if not ok:
                            lines.append(
                                f"- {name}: letzter Heartbeat vor {delay}s (> {threshold}s Schwelle)"
                            )
                else:
                    lines = [
                        "✅ [Watchdog] Terminal-Status – alle Runner OK "
                        f"[TZ:{now:%Y-%m-%d %H:%M:%S}]"
                    ]

                msg = "\n".join(lines)
                try:
                    send_watchdog_telegram_message(msg, parse_mode=None)
                except Exception as e:
                    log_service.log_system(
                        f"[WatchdogTelegram] Versand fehlgeschlagen: {e}",
                        level="WARNING",
                    )
            except Exception as e:
                log_service.log_system(
                    f"[WatchdogTelegram] Fehler bei Aggregation/Versand: {e}",
                    level="ERROR",
                )

            time.sleep(interval)

    def _ticker(self, interval_sec: int):
        """Periodischer TIMER_TICK."""
        self._ensure_bus_started()
        while self.running and not self._stop.is_set():
            self._safe_publish(Event(EventType.TIMER_TICK, {"note": "interval"}))
            time.sleep(interval_sec)

    def _barclose_detector_loop(self):
        """
        Publiziert BAR_CLOSE Events, sobald der DataProvider eine neue, tatsächlich
        geschlossene Kerze liefert. Rein datengetrieben – kein zeitbasierter Fallback.
        """
        self._ensure_bus_started()
        if not self._barclose_specs:
            log_service.log_system(
                "[BarCloseDetector] Keine DataProvider/Timeframes registriert – kein BAR_CLOSE.",
                level="DEBUG",
            )
            while self.running and not self._stop.is_set():
                time.sleep(1.0)
            return

        last_seen: Dict[Tuple[int, str, str], datetime] = {}

        while self.running and not self._stop.is_set():
            loop_start = time.time()
            for spec in self._barclose_specs:
                symbol = spec.anchor_symbol
                key = (id(spec.provider), spec.timeframe, symbol.upper())
                try:
                    record = spec.provider.get_ohlc_for_closed_candle(
                        symbol, spec.timeframe, offset=1
                    )
                except Exception as e:
                    log_service.log_system(
                        f"[BarCloseDetector] Fehler beim Abruf {spec.anchor_symbol}/{spec.timeframe}: {e}",
                        level="WARNING",
                    )
                    continue

                if not isinstance(record, dict):
                    continue
                raw_time = record.get("time")
                if raw_time is None:
                    continue

                closed_at = self._parse_iso_utc(raw_time)
                if closed_at is None:
                    continue

                prev = last_seen.get(key)
                if prev is None:
                    # Initiale Beobachtung – kein Event, nur Merker.
                    last_seen[key] = closed_at
                    continue

                if closed_at <= prev:
                    continue

                payload = {
                    "timeframe": spec.timeframe,
                    "closed_at": closed_at.isoformat(),
                    "symbols": [symbol],
                }
                event = Event(EventType.BAR_CLOSE, payload)
                if self._safe_publish(event):
                    last_seen[key] = closed_at
            elapsed = time.time() - loop_start
            sleep_for = max(0.0, self.barclose_poll_interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _parse_iso_utc(self, value: datetime | str) -> datetime | None:
        try:
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                txt = value.strip()
                if txt.endswith("Z"):
                    txt = txt[:-1] + "+00:00"
                dt = datetime.fromisoformat(txt)
            else:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=ZoneInfo("UTC"))
            return dt.astimezone(ZoneInfo("UTC"))
        except Exception:
            return None

    def _session_scheduler(self):
        """Publiziert SESSION_OPEN/SESSION_CLOSE/AUTO_CLOSE basierend auf Strategy‑Sessionzeiten."""
        self._ensure_bus_started()
        last_emit = {"open": set(), "close": set(), "auto": set()}
        # Einfache Cache-Strategie (geringe Last): Session-Zeiten pro Runner
        session_cache: Dict[int, Dict[str, datetime.time]] = {}

        while self.running and not self._stop.is_set():
            try:
                now = now_utc().replace(second=0, microsecond=0)
                hm = (now.hour, now.minute)

                for r in self.runners:
                    rid = id(r)
                    sess = session_cache.get(rid)
                    if not sess:
                        try:
                            sess = SessionRunner(r.strategy).get_session_times()
                        except Exception as e:
                            log_service.log_system(
                                f"[SessionScheduler] get_session_times() Fehler: {e}",
                                level="ERROR",
                            )
                            sess = None
                        if sess:
                            session_cache[rid] = sess
                        else:
                            continue

                    # session_start/session_end/auto_close sind datetime.time
                    st, en, ac = (
                        sess.get("session_start"),
                        sess.get("session_end"),
                        sess.get("auto_close"),
                    )

                    # SESSION_OPEN
                    if st and _time_hm(st) == hm:
                        key = (rid, "open", now.date())
                        if key not in last_emit["open"]:
                            self._safe_publish(
                                Event(
                                    EventType.SESSION_OPEN,
                                    {
                                        "strategy": r.strategy.name(),
                                        "at": now.isoformat(),
                                    },
                                )
                            )
                            last_emit["open"].add(key)
                    # SESSION_CLOSE
                    if en and _time_hm(en) == hm:
                        key = (rid, "close", now.date())
                        if key not in last_emit["close"]:
                            self._safe_publish(
                                Event(
                                    EventType.SESSION_CLOSE,
                                    {
                                        "strategy": r.strategy.name(),
                                        "at": now.isoformat(),
                                    },
                                )
                            )
                            last_emit["close"].add(key)
                    # AUTO_CLOSE
                    if ac and _time_hm(ac) == hm:
                        key = (rid, "auto", now.date())
                        if key not in last_emit["auto"]:
                            self._safe_publish(
                                Event(
                                    EventType.AUTO_CLOSE,
                                    {
                                        "strategy": r.strategy.name(),
                                        "at": now.isoformat(),
                                    },
                                )
                            )
                            last_emit["auto"].add(key)
            except Exception as e:
                log_service.log_system(f"[SessionScheduler] Fehler: {e}", level="ERROR")

            time.sleep(1)

    def _broker_watchdog(self):
        """Überwacht Broker-FSM und emittiert Status-Events entsprechend Policy."""
        self._ensure_bus_started()
        consecutive_failures = 0
        while self.running and not self._stop.is_set():
            ok = False
            try:
                ok = self._fsm.ensure()  # nutzt Policy (Backoff/Jitter)
            except Exception as e:
                log_service.log_system(
                    f"[BrokerWatchdog] ensure() Fehler: {e}", level="ERROR"
                )
                ok = False

            try:
                if ok:
                    if (
                        consecutive_failures > 0
                        or self._fsm.status != BrokerStatus.CONNECTED
                    ):
                        self._safe_publish(
                            Event(EventType.BROKER_STATUS, {"status": "CONNECTED"})
                        )
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    status = (
                        "DEGRADED"
                        if consecutive_failures < self._fsm.policy.max_retries
                        else "DISCONNECTED"
                    )
                    self._safe_publish(
                        Event(EventType.BROKER_STATUS, {"status": status})
                    )
            except Exception as e:
                log_service.log_system(
                    f"[BrokerWatchdog] Status-Publish Fehler: {e}", level="ERROR"
                )

            time.sleep(self.broker_watchdog_interval)

    def _news_scheduler(self):
        """
        Publiziert NEWS Events aus dem Warm‑Cache:
        - PRE:   60 min vor Event
        - LIVE:  exakter Event‑Zeitpunkt
        - POST:  dynamisches Ende gemäß news_filter.post_block_end()
        Payload: {"phase": "PRE|LIVE|POST", "currency": "...", "title": "...", "at": iso, "symbols": [...]}
        """
        self._ensure_bus_started()

        # Warm-Cache versuchen zu laden (idempotent)
        try:
            news_filter.load_news_csv()
        except Exception as e:
            log_service.log_system(
                f"[NewsScheduler] Laden des News‑Caches fehlgeschlagen: {e}",
                level="WARNING",
            )
            return  # Scheduler sauber beenden (kein Crash/Restart-Loop)

        emitted = set()

        while self.running and not self._stop.is_set():
            try:
                now = now_utc()

                # Alle relevanten Symbole aus allen Runnern sammeln
                all_symbols = set()
                for r in self.runners:
                    try:
                        all_symbols.update(r.strategy.config.get("symbols", []))
                    except Exception:
                        pass

                # News-Events lesen (Public-API)
                events = news_filter.get_news_events()
                if not events:
                    # Kein Cache? Dann sparsam pollen.
                    time.sleep(5)
                    continue

                for ev in events:
                    at = ev.datetime  # UTC
                    pre = at - timedelta(minutes=60)
                    post = news_filter.post_block_end(at)

                    cur = ev.currency
                    title = ev.event

                    # Zuordnung: welche unserer Symbole sind von dieser Währung betroffen?
                    affected_symbols = []
                    for s in all_symbols:
                        ccy = set(news_filter.currencies_for_symbol(s))
                        if not ccy:
                            continue
                        # USD breiter blocken (wie vorher)
                        if cur == "USD":
                            affected_symbols.append(s)
                        elif cur in ccy:
                            affected_symbols.append(s)

                    if not affected_symbols:
                        continue

                    base_key = (cur, title, at.isoformat())

                    if pre <= now < at and (base_key, "PRE") not in emitted:
                        self._safe_publish(
                            Event(
                                EventType.NEWS,
                                {
                                    "phase": "PRE",
                                    "currency": cur,
                                    "title": title,
                                    "at": at.isoformat(),
                                    "symbols": affected_symbols,
                                },
                            )
                        )
                        emitted.add((base_key, "PRE"))

                    # LIVE exakt am Timestamp (±1s Toleranz)
                    if (
                        abs((now - at).total_seconds()) < 1.0
                        and (base_key, "LIVE") not in emitted
                    ):
                        self._safe_publish(
                            Event(
                                EventType.NEWS,
                                {
                                    "phase": "LIVE",
                                    "currency": cur,
                                    "title": title,
                                    "at": at.isoformat(),
                                    "symbols": affected_symbols,
                                },
                            )
                        )
                        emitted.add((base_key, "LIVE"))

                    if at < now <= post and (base_key, "POST") not in emitted:
                        self._safe_publish(
                            Event(
                                EventType.NEWS,
                                {
                                    "phase": "POST",
                                    "currency": cur,
                                    "title": title,
                                    "at": at.isoformat(),
                                    "symbols": affected_symbols,
                                },
                            )
                        )
                        emitted.add((base_key, "POST"))

            except Exception as e:
                log_service.log_system(f"[NewsScheduler] Fehler: {e}", level="ERROR")

            time.sleep(1)

    # ---------- Lifecycle ----------

    def start_all(self, interval_sec: int = 10):
        if self.running:
            log_service.log_system(
                "[MultiRunner] Bereits gestartet – Ignoriere erneuten start_all()",
                level="WARNING",
            )
            return

        self.running = True
        self._stop.clear()
        self._ensure_bus_started()
        self._rebuild_barclose_specs()

        # Runner-Threads starten
        for runner in self.runners:
            t = threading.Thread(
                target=self._run_strategy_loop,
                args=(runner,),
                daemon=True,
                name=f"RunnerThread-{runner.strategy.name()}",
            )
            self.threads.append(t)  # beibehalten für Kompatibilität
            self._runner_threads[runner] = t
            t.start()

        # Background-Threads (registriert + restartfähig)
        self._start_bg("TimerTicker", self._ticker, interval_sec)
        self._start_bg("BarCloseDetector", self._barclose_detector_loop)
        self._start_bg("SessionScheduler", self._session_scheduler)
        self._start_bg("BrokerWatchdog", self._broker_watchdog)
        self._start_bg("HealthMonitor", self._monitor_health)
        self._start_bg("Watchdog", self._watchdog_loop)
        self._start_bg("WatchdogTelegram", self._watchdog_telegram_loop)
        self._start_bg("NewsScheduler", self._news_scheduler)

        log_service.log_system("[Controller] 🧭 Watchdog gestartet")
        log_service.log_system("[MultiRunner] ✅ Alle Strategie-Runner gestartet")

    def stop_all(self):
        self.running = False
        self._stop.set()
        # EventBus stoppen (best effort)
        try:
            self.event_bus.stop()
        except Exception as e:
            log_service.log_system(
                f"[Controller] EventBus.stop() fehlgeschlagen: {e}", level="DEBUG"
            )

        # Runner-Threads beenden
        for t in list(self.threads):
            try:
                if t.is_alive():
                    t.join(timeout=2.5)
            except Exception:
                pass

        # Background-Threads beenden (best effort)
        for name, t in list(self._bg_threads.items()):
            try:
                if t.is_alive():
                    t.join(timeout=2.5)
            except Exception:
                pass

        log_service.log_system("[MultiRunner] 🛑 Controller gestoppt")
