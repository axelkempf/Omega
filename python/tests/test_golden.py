"""Golden file regression tests for Omega V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bt import run_backtest

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "golden"
DATA_ROOT = Path(__file__).resolve().parent / "fixtures" / "data"
ARTIFACTS = ("meta.json", "trades.json", "equity.csv", "metrics.json")


@pytest.mark.parametrize(
    "scenario_dir",
    sorted([path for path in FIXTURE_ROOT.iterdir() if path.is_dir()]),
)
def test_golden_smoke(scenario_dir: Path, tmp_path: Path) -> None:
    """Run smoke golden regression against committed fixtures."""
    if not _ffi_available():
        pytest.skip("omega_bt module not available")
    if not (DATA_ROOT / "parquet").exists():
        pytest.skip("Fixture data root not available")

    config_path = scenario_dir / "config.json"
    if not config_path.exists():
        pytest.skip(f"Missing config file: {config_path}")

    output_dir = tmp_path / scenario_dir.name
    run_backtest(config_path=config_path, output_dir=output_dir)

    for artifact in ARTIFACTS:
        assert (output_dir / artifact).exists(), (
            f"Missing {artifact} for {scenario_dir.name}"
        )

    _assert_meta_equal(output_dir / "meta.json", scenario_dir / "meta.json")
    _assert_json_equal(output_dir / "trades.json", scenario_dir / "trades.json")
    _assert_json_equal(output_dir / "metrics.json", scenario_dir / "metrics.json")
    _assert_csv_equal(output_dir / "equity.csv", scenario_dir / "equity.csv")


def _assert_meta_equal(actual_path: Path, expected_path: Path) -> None:
    actual = _normalize_meta(_load_json(actual_path))
    expected = _normalize_meta(_load_json(expected_path))
    assert actual == expected


def _normalize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(meta)
    normalized["generated_at"] = "normalized"
    normalized["generated_at_ns"] = 0
    return normalized


def _assert_json_equal(actual_path: Path, expected_path: Path) -> None:
    assert _load_json(actual_path) == _load_json(expected_path)


def _assert_csv_equal(actual_path: Path, expected_path: Path) -> None:
    actual = actual_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    expected = expected_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert actual == expected


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffi_available() -> bool:
    try:
        import omega_bt  # noqa: F401
    except Exception:
        return False
    return True
