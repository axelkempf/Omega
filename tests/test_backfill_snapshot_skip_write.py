"""Test für die Skip-Write-Optimierung in _prepare_backfill_snapshot."""

import json
from pathlib import Path

import pytest
from analysis.backfill_reporting_defaults import BACKFILL_REPORTING_DEFAULTS
from analysis.backfill_walkforward_equity_curves import (
    BACKFILL_SNAPSHOT_NAME,
    _prepare_backfill_snapshot,
)


def test_backfill_snapshot_skips_write_when_unchanged(tmp_path: Path) -> None:
    """Verifiziert, dass _prepare_backfill_snapshot nicht schreibt, wenn bereits aktuell."""
    run_dir = tmp_path / "run_20250101_000000"
    baseline = run_dir / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)

    base_cfg = {
        "symbol": "EURUSD",
        "timeframes": {"primary": "M15", "additional": []},
        "reporting": {"dev_mode": True},  # Wird überschrieben werden
        "start_date": "2019-01-01",
        "end_date": "2019-12-31",
    }
    blob = {"base_config": base_cfg, "param_grid": {"foo": [1, 2]}}
    frozen_path = baseline / "frozen_snapshot.json"
    frozen_path.write_text(json.dumps(blob), encoding="utf-8")

    # Erster Aufruf: erzeugt Backfill-Snapshot
    _prepare_backfill_snapshot(run_dir, start_date=None, end_date=None)

    backfill_path = baseline / BACKFILL_SNAPSHOT_NAME
    assert backfill_path.exists(), "Backfill-Snapshot sollte erzeugt worden sein"

    # Merke mtime
    mtime_1 = backfill_path.stat().st_mtime

    # Zweiter Aufruf mit gleichen Parametern: sollte NICHT schreiben
    _prepare_backfill_snapshot(run_dir, start_date=None, end_date=None)

    mtime_2 = backfill_path.stat().st_mtime
    assert (
        mtime_2 == mtime_1
    ), "Backfill-Snapshot sollte nicht neu geschrieben worden sein"


def test_backfill_snapshot_writes_when_dates_change(tmp_path: Path) -> None:
    """Verifiziert, dass _prepare_backfill_snapshot schreibt, wenn Dates sich ändern."""
    run_dir = tmp_path / "run_20250101_000000"
    baseline = run_dir / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)

    base_cfg = {
        "symbol": "EURUSD",
        "timeframes": {"primary": "M15", "additional": []},
        "reporting": {
            "dev_mode": True
        },  # Falsch, wird beim ersten Aufruf überschrieben
        "start_date": "2019-01-01",
        "end_date": "2019-12-31",
    }
    blob = {"base_config": base_cfg, "param_grid": {"foo": [1, 2]}}
    frozen_path = baseline / "frozen_snapshot.json"
    frozen_path.write_text(json.dumps(blob), encoding="utf-8")

    # Erster Aufruf: erzeugt Backfill-Snapshot (korrigiert Reporting)
    _prepare_backfill_snapshot(run_dir, start_date=None, end_date=None)

    backfill_path = baseline / BACKFILL_SNAPSHOT_NAME
    assert backfill_path.exists(), "Backfill-Snapshot sollte erzeugt worden sein"
    mtime_1 = backfill_path.stat().st_mtime

    # Zweiter Aufruf mit geändertem Datum: sollte schreiben
    import time

    time.sleep(0.01)  # Sicherstellen, dass mtime sich unterscheidet
    _prepare_backfill_snapshot(run_dir, start_date="2020-01-01", end_date=None)

    mtime_2 = backfill_path.stat().st_mtime
    assert mtime_2 > mtime_1, "Backfill-Snapshot sollte neu geschrieben worden sein"

    # Prüfe, dass Datum geändert wurde
    backfill_blob = json.loads(backfill_path.read_text())
    backfill_cfg = backfill_blob.get("base_config") or {}
    assert backfill_cfg.get("start_date") == "2020-01-01"
