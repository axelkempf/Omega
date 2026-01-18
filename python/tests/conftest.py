"""Pytest configuration for the Python wrapper tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
OLD_SRC_DIR = REPO_ROOT / "src" / "old"
FIXTURE_DATA_ROOT = REPO_ROOT / "python" / "tests" / "fixtures" / "data"


def _ensure_importable(module_name: str, path: Path) -> None:
    if importlib.util.find_spec(module_name) is None and path.exists():
        sys.path.insert(0, str(path))


_ensure_importable("bt", PYTHON_DIR)
_ensure_importable("backtest_engine", OLD_SRC_DIR)


@pytest.fixture(autouse=True)
def _set_v2_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    if FIXTURE_DATA_ROOT.exists():
        monkeypatch.setenv(
            "OMEGA_DATA_PARQUET_ROOT", str(FIXTURE_DATA_ROOT / "parquet")
            )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Sample backtest result for output tests."""
    return {
        "meta": {
            "start_timestamp": 1_700_000_000_000_000_000,
            "end_timestamp": 1_700_000_060_000_000_000,
            "extra": {
                "run_id": "test-run",
                "engine": {"name": "omega-v2", "version": "0.1.0"},
                "config": {"hash": "abc123"},
                "dataset": {
                    "symbol": "EURUSD",
                    "timeframe": "M1",
                    "manifest_sha256": "deadbeef",
                    "governance": {"alignment": {}, "gaps": {}},
                },
                "account": {"account_currency": "EUR", "initial_balance": 10000.0},
            },
        },
        "trades": [
            {
                "entry_time_ns": 1_700_000_000_000_000_000,
                "exit_time_ns": 1_700_000_060_000_000_000,
                "direction": "long",
                "symbol": "EURUSD",
                "entry_price": 1.1000,
                "exit_price": 1.1010,
                "stop_loss": 1.0950,
                "take_profit": 1.1050,
                "size": 0.1,
                "result": 10.0,
                "r_multiple": 1.0,
                "reason": "take_profit",
                "scenario_id": 1,
            }
        ],
        "metrics": {"total_trades": 1},
        "metric_definitions": {
            "total_trades": {
                "unit": "count",
                "description": "Total trades",
                "domain": ">=0",
                "source": "trades",
                "type": "number",
            }
        },
        "equity_curve": [
            {
                "timestamp_ns": 1_700_000_000_000_000_000,
                "equity": 10000.0,
                "balance": 10000.0,
                "drawdown": 0.0,
                "high_water": 10000.0,
            }
        ],
    }
