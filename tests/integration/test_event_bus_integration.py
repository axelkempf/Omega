"""Integration tests for the EventBus under load and slow handlers."""

from __future__ import annotations

import threading
import time

import pytest

from hf_engine.core.controlling.event_bus import Event, EventBus, EventType

pytestmark = [pytest.mark.integration, pytest.mark.trading_safety]


@pytest.fixture
def event_bus():
    """Provide a started EventBus instance and ensure cleanup."""

    bus = EventBus(
        maxsize=16,
        warn_highwater_ratio=0.5,
        slow_handler_warning_ms=5,
        drop_retry_ms=0,
        handler_workers=4,
    )
    bus.start()
    yield bus
    bus.stop()


def test_publish_delivers_to_subscribers(event_bus: EventBus) -> None:
    received: list[int] = []
    done = threading.Event()

    def handler(ev: Event) -> None:
        received.append(ev.payload.get("value", 0))
        done.set()

    event_bus.subscribe(EventType.TIMER_TICK, handler)

    assert event_bus.publish(Event(EventType.TIMER_TICK, {"value": 42}))
    # Increased timeout for CI environments with slower thread scheduling
    assert done.wait(timeout=5.0), "handler should be called within timeout"
    assert received == [42]
    assert event_bus.processed_events >= 1


def test_slow_handler_emits_warning(
    monkeypatch: pytest.MonkeyPatch, event_bus: EventBus
) -> None:
    logged: list[tuple[str, str]] = []

    monkeypatch.setattr(
        event_bus,
        "_log",
        lambda level, msg, **kwargs: logged.append((level, msg)),
    )

    done = threading.Event()

    def slow_handler(ev: Event) -> None:
        time.sleep(0.03)
        done.set()

    event_bus.subscribe(EventType.BAR_CLOSE, slow_handler)

    assert event_bus.publish(Event(EventType.BAR_CLOSE, {"bar": 1}))
    # Increased timeout for CI environments
    assert done.wait(timeout=5.0), "slow handler should complete within timeout"

    event_bus.stop()

    assert any("Langsamer Handler" in msg for _, msg in logged)


def test_backpressure_warns_and_registers_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus(
        maxsize=2,
        warn_highwater_ratio=0.5,
        drop_warn_every=1,
        drop_retry_ms=0,
        handler_workers=1,
    )

    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        bus, "_log", lambda level, msg, **kwargs: logged.append((level, msg))
    )

    assert bus.publish(Event(EventType.NEWS, {"id": 1}))
    assert bus.publish(Event(EventType.NEWS, {"id": 2}))
    dropped = bus.publish(Event(EventType.NEWS, {"id": 3}))

    assert dropped is False
    assert any("Hoher Queue-Füllstand" in msg for _, msg in logged)
    assert any("Queue voll" in msg or "dropped_events" in msg for _, msg in logged)

    bus.stop()


def test_graceful_shutdown_cleans_up_workers(event_bus: EventBus) -> None:
    processed: list[Event] = []
    done = threading.Event()

    def handler(ev: Event) -> None:
        processed.append(ev)
        done.set()

    event_bus.subscribe(EventType.BROKER_STATUS, handler)

    assert event_bus.publish(Event(EventType.BROKER_STATUS, {"state": "ok"}))
    # Increased timeout for CI environments
    assert done.wait(timeout=5.0), "handler should process event within timeout"

    event_bus.stop()

    assert not event_bus._running
    assert event_bus._handler_pool is None
    assert event_bus._pending_tasks == 0
    assert event_bus._thread is None or not event_bus._thread.is_alive()
    assert processed
