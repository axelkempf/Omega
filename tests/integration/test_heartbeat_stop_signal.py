"""Integration tests for heartbeat and stop signal mechanisms in engine_launcher."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.utils.trading_test_utils import (
    create_heartbeat_file,
    create_stop_signal_file,
    wait_for_file_content,
)

pytestmark = [pytest.mark.integration, pytest.mark.trading_safety]


@pytest.fixture
def patched_paths(monkeypatch, temp_heartbeat_dir):
    """Patch _shutdown_paths to use temporary directory for isolation.

    We patch the function itself because SHUTDOWN_SIGNAL_DIR is read at import
    time and _shutdown_paths captures it in a closure-like manner.
    """
    from src import engine_launcher as launcher

    original_shutdown_paths = launcher._shutdown_paths

    def _patched_shutdown_paths(account_id: str) -> dict:
        return {
            "stop_file": temp_heartbeat_dir / f"stop_{account_id}.signal",
            "heartbeat_file": temp_heartbeat_dir / f"heartbeat_{account_id}.txt",
            "datafeed_conf": temp_heartbeat_dir / f"datafeed_conf_{account_id}.json",
        }

    monkeypatch.setattr(launcher, "_shutdown_paths", _patched_shutdown_paths)
    return launcher


def test_heartbeat_file_is_created_and_updated(patched_paths, temp_heartbeat_dir):
    launcher = patched_paths
    account_id = "testacct"

    stop_event = launcher.start_heartbeat(account_id, interval_sec=0.1)
    hb_file = temp_heartbeat_dir / f"heartbeat_{account_id}.txt"

    # CI environments (Ubuntu) may have slower disk I/O and thread scheduling.
    # Increase timeout and use longer polling to ensure robustness.
    first_content = wait_for_file_content(hb_file, timeout=5.0)
    assert (
        first_content is not None and first_content.strip() != ""
    ), f"heartbeat file should exist and have content, got: {first_content!r}"

    second_content = None
    # Wait longer and poll less frequently for CI robustness
    for _ in range(40):
        time.sleep(0.1)
        try:
            candidate = hb_file.read_text()
        except Exception:
            continue
        if candidate.strip() and candidate != first_content:
            second_content = candidate
            break

    assert (
        second_content is not None
    ), f"heartbeat should update file content (first={first_content!r})"
    assert second_content.strip() != ""

    stop_event.set()


def test_stop_signal_triggers_event(patched_paths, temp_heartbeat_dir):
    launcher = patched_paths
    account_id = "testacct"

    stop_event = launcher.start_heartbeat(account_id, interval_sec=0.1)
    create_stop_signal_file(temp_heartbeat_dir, account_id)

    # allow monitor thread to observe stop file (longer timeout for CI)
    assert stop_event.wait(timeout=5.0)


def test_cleanup_files_removes_existing(patched_paths, temp_heartbeat_dir):
    launcher = patched_paths
    sample = temp_heartbeat_dir / "sample.txt"
    sample.write_text("data")

    assert sample.exists()
    launcher._cleanup_files(sample)
    assert not sample.exists()


def test_stop_and_heartbeat_paths_are_isolated(patched_paths, temp_heartbeat_dir):
    launcher = patched_paths
    account_id = "iso"
    paths = launcher._shutdown_paths(account_id)

    assert paths["stop_file"].parent == temp_heartbeat_dir
    assert paths["heartbeat_file"].parent == temp_heartbeat_dir


def test_heartbeat_respects_custom_interval(patched_paths, temp_heartbeat_dir):
    launcher = patched_paths
    account_id = "custom"

    stop_event = launcher.start_heartbeat(account_id, interval_sec=0.05)
    hb_file = temp_heartbeat_dir / f"heartbeat_{account_id}.txt"

    # Use longer timeout for CI robustness
    first_content = wait_for_file_content(hb_file, timeout=5.0)
    second_content = None
    # Wait longer and poll less frequently for CI robustness
    for _ in range(40):
        time.sleep(0.1)
        try:
            candidate = hb_file.read_text()
        except Exception:
            continue
        if candidate.strip() and candidate != first_content:
            second_content = candidate
            break

    assert first_content is not None, "heartbeat file should have initial content"
    assert (
        second_content is not None
    ), f"heartbeat should refresh on custom interval (first={first_content!r})"
    assert second_content != first_content
    stop_event.set()


def test_monitor_stop_file_handles_missing(monkeypatch, temp_heartbeat_dir):
    from src import engine_launcher as launcher

    stop_event = launcher.threading.Event()
    # ensure no exception when file missing
    t = launcher.threading.Thread(
        target=launcher._monitor_stop_file,
        args=(temp_heartbeat_dir / "missing.signal", stop_event, 0.1),
        daemon=True,
    )
    t.start()
    time.sleep(0.2)
    stop_event.set()
    t.join(timeout=1.0)
    assert stop_event.is_set()
