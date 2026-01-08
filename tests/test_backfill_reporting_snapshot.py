import json
from pathlib import Path

import pytest

from src.backtest_engine.analysis.backfill_reporting_defaults import (
    BACKFILL_REPORTING_DEFAULTS,
)
from src.backtest_engine.analysis.backfill_walkforward_equity_curves import (
    BACKFILL_SNAPSHOT_NAME,
    _load_backfill_reporting_defaults,
    _prepare_backfill_snapshot,
    load_snapshot,
)


@pytest.mark.parametrize(
    "start_date,end_date", [(None, None), ("2020-01-01", "2020-12-31")]
)
def test_backfill_snapshot_overrides_reporting(
    tmp_path: Path, start_date, end_date
) -> None:
    """Verifiziert, dass Backfill-Snapshot immer mit zentralen Reporting-Defaults erzeugt wird."""
    # Die Defaults sind jetzt eine Konstante – kein JSON-Laden mehr nötig
    reporting_defaults = _load_backfill_reporting_defaults()
    assert reporting_defaults == BACKFILL_REPORTING_DEFAULTS
    assert reporting_defaults.get("dev_mode") is False, "dev_mode sollte False sein"

    run_dir = tmp_path / "run_20250101_000000"
    baseline = run_dir / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)

    base_cfg = {
        "symbol": "EURUSD",
        "timeframes": {"primary": "M15", "additional": []},
        "reporting": {"dev_mode": True},
        "start_date": "2019-01-01",
        "end_date": "2019-12-31",
    }
    blob = {"base_config": base_cfg, "param_grid": {"foo": [1, 2]}}
    frozen_path = baseline / "frozen_snapshot.json"
    frozen_path.write_text(json.dumps(blob), encoding="utf-8")

    _prepare_backfill_snapshot(run_dir, start_date=start_date, end_date=end_date)

    backfill_path = baseline / BACKFILL_SNAPSHOT_NAME
    assert backfill_path.exists(), "Backfill-Snapshot wurde nicht erzeugt."

    backfill_blob = json.loads(backfill_path.read_text())
    backfill_cfg = backfill_blob.get("base_config") or {}
    assert isinstance(backfill_cfg, dict)
    assert backfill_cfg.get("reporting") == reporting_defaults

    # Start/End-Datum sollte nur geändert werden, wenn gesetzt
    expected_start = start_date or base_cfg["start_date"]
    expected_end = end_date or base_cfg["end_date"]
    assert backfill_cfg.get("start_date") == expected_start
    assert backfill_cfg.get("end_date") == expected_end

    # load_snapshot soll das Reporting aus dem Backfill-Snapshot beibehalten
    upgraded_cfg, _ = load_snapshot(run_dir)
    assert upgraded_cfg.get("reporting") == reporting_defaults
