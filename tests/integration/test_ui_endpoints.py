"""Integration tests for FastAPI UI endpoints (process control & logs)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pytest
from fastapi.testclient import TestClient

from ui_engine import main

pytestmark = [pytest.mark.integration]


class _StubManager:
    def __init__(self, name: str):
        self.name = name
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True

    def stop(self) -> bool:
        self.started = False
        return True

    def status(self) -> str:
        return "Running" if self.started else "Stopped"


@pytest.fixture
def client_with_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Tuple[TestClient, _StubManager, Dict[str, bool], Path, Path]:
    """Provide a TestClient with patched managers, logging, and tmp paths."""

    stub_manager = _StubManager("stub")
    health_state = {"ok": True}

    monkeypatch.setattr(
        main, "restart_unresponsive_strategies", lambda interval=30: None
    )
    monkeypatch.setattr(main, "get_strategy_manager", lambda name: stub_manager)
    monkeypatch.setattr(main, "get_datafeed_manager", lambda name: stub_manager)
    monkeypatch.setattr(main, "start_datafeed_server", lambda: True)
    monkeypatch.setattr(main, "check_datafeed_health", lambda: health_state["ok"])
    monkeypatch.setattr(main, "resolve_alias", lambda alias: alias)

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "LOG_DIR", log_dir)

    tmp_dir = tmp_path / "var" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "TMP_DIR", tmp_dir)

    client = TestClient(main.app)
    yield client, stub_manager, health_state, log_dir, tmp_dir
    client.close()


def test_health_endpoint_returns_status(client_with_stubs):
    client, _, health_state, _, _ = client_with_stubs

    health_state["ok"] = True
    resp_ok = client.get("/datafeed/health")
    assert resp_ok.status_code == 200
    assert resp_ok.json()["status"] == "ok"

    health_state["ok"] = False
    resp_offline = client.get("/datafeed/health")
    assert resp_offline.status_code == 200
    assert resp_offline.json()["status"] == "offline"


def test_start_strategy_endpoint(client_with_stubs):
    client, manager, _, _, _ = client_with_stubs

    resp = client.post("/start/demo")

    assert resp.status_code == 200
    assert resp.json()["status"] == "Running"
    assert manager.started is True


def test_stop_strategy_endpoint(client_with_stubs):
    client, manager, _, _, _ = client_with_stubs
    manager.started = True

    resp = client.post("/stop/demo")

    assert resp.status_code == 200
    assert resp.json()["status"] == "Stopped"
    assert manager.started is False


def test_status_endpoint(client_with_stubs):
    client, manager, _, _, _ = client_with_stubs
    manager.started = True

    resp = client.get("/status/demo")

    assert resp.status_code == 200
    assert resp.json()["status"] == "Running"


def test_logs_endpoint_reads_tail(client_with_stubs):
    client, _, _, log_dir, _ = client_with_stubs
    log_file = log_dir / "account.log"
    log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    resp = client.get("/logs/account", params={"lines": 2})

    assert resp.status_code == 200
    assert "line2" in resp.text and "line3" in resp.text


def test_datafeed_stop_creates_signal_file(client_with_stubs):
    client, _, _, _, tmp_dir = client_with_stubs

    resp = client.post("/datafeed/stop")

    stop_file = tmp_dir / "stop_15582434.signal"
    assert resp.status_code == 200
    assert stop_file.exists()
    assert stop_file.read_text().strip() == "stop"


def test_datafeed_start_returns_started(client_with_stubs):
    client, _, _, _, _ = client_with_stubs

    resp = client.post("/datafeed/start")

    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
