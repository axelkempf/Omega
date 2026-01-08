from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtest_engine.config.models import BacktestConfig
from configs.backtest._config_validator import validate_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "rel_path",
    [
        "configs/backtest/mean_reversion_z_score.json",
    ],
)
def test_backtest_config_models_validate_existing_configs(rel_path: str) -> None:
    path = _REPO_ROOT / rel_path
    assert path.is_file(), f"Missing config file: {path}"

    raw = json.loads(path.read_text(encoding="utf-8"))

    # Validator should accept shipped example configs.
    assert validate_config(raw) == []

    model = BacktestConfig.model_validate(raw)
    legacy = model.to_legacy_dict()

    # Legacy dict must remain JSON-serializable and keep key aliases.
    assert isinstance(legacy["start_date"], str)
    assert legacy["start_date"] == raw["start_date"]
    assert legacy["end_date"] == raw["end_date"]

    assert "strategy" in legacy
    assert isinstance(legacy["strategy"], dict)
    assert "class" in legacy["strategy"]


def test_validate_config_reports_structured_errors() -> None:
    bad = {
        "start_date": "2025-01-02",
        "end_date": "2025-01-01",
        "mode": "candle",
        "symbol": "EURUSD",
        "timeframes": {"primary": "M15", "additional": []},
        "strategy": {
            "module": "mean_reversion_z_score.backtest.backtest_strategy",
            "class": "MeanReversionZScoreStrategy",
            "parameters": {},
        },
    }

    errors = validate_config(bad)
    assert errors
    assert any("start_date" in e or "end_date" in e for e in errors)
