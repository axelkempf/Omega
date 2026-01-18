"""Integration and golden file tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bt import run_backtest
from bt.config import load_config, validate_config
from bt.output import write_artifacts


def test_load_config_defaults(tmp_path: Path) -> None:
    """Defaults are applied when loading configs."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "strategy_name": "mean_reversion_z_score",
                "symbol": "EURUSD",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
                "timeframes": {"primary": "M1"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["schema_version"] == "2.0"
    assert config["run_mode"] == "dev"
    assert config["data_mode"] == "candle"
    assert config["execution_variant"] == "v2"
    assert config["warmup_bars"] == 500
    assert config["timeframes"]["additional"] == []
    assert config["timeframes"]["additional_source"] == "separate_parquet"


def test_validate_config_missing_field() -> None:
    """Missing required fields raise validation errors."""
    with pytest.raises(ValueError, match="strategy_name"):
        validate_config(
            {
                "symbol": "EURUSD",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
            }
        )


def test_validate_config_date_order() -> None:
    """Start date must be before end date."""
    with pytest.raises(ValueError, match="start_date"):
        validate_config(
            {
                "strategy_name": "mean_reversion_z_score",
                "symbol": "EURUSD",
                "start_date": "2025-01-02",
                "end_date": "2025-01-01",
            }
        )


def test_write_artifacts(sample_result: dict[str, Any], tmp_path: Path) -> None:
    """Artifact writer emits all required files."""
    write_artifacts(sample_result, tmp_path)

    for filename in ("meta.json", "trades.json", "equity.csv", "metrics.json"):
        assert (tmp_path / filename).exists(), f"Missing {filename}"

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "test-run"
    assert "generated_at" in meta
    assert "generated_at_ns" in meta
    assert meta["engine"]["name"] == "omega-v2"
    assert meta["config"]["hash"] == "abc123"
    assert meta["dataset"]["manifest_sha256"] == "deadbeef"
    assert meta["dataset"]["start_time"].endswith("+00:00")
    assert meta["dataset"]["end_time"].endswith("+00:00")

    trades = json.loads((tmp_path / "trades.json").read_text(encoding="utf-8"))
    assert trades[0]["entry_time"].endswith("+00:00")
    assert trades[0]["exit_time"].endswith("+00:00")

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in metrics
    assert "definitions" in metrics

    equity_lines = (tmp_path / "equity.csv").read_text(encoding="utf-8").splitlines()
    assert equity_lines[0].startswith("timestamp,timestamp_ns,eq")


def test_run_backtest_requires_input() -> None:
    """run_backtest enforces input arguments."""
    with pytest.raises(ValueError, match="config_path"):
        run_backtest()
