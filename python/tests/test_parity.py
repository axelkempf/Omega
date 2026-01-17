"""V1 vs V2 parity tests."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

import pytest

from bt import run_backtest as run_v2_backtest

try:
    from src.old.backtest_engine.runner import run_backtest_and_return_portfolio
except Exception:
    run_backtest_and_return_portfolio = None

FIXTURE_DATA_ROOT = Path(__file__).resolve().parent / "fixtures" / "data"


@pytest.fixture
def parity_config() -> dict[str, Any]:
    """Config for parity testing."""
    return {
        "schema_version": "1.0.0",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2025-01-01",
        "end_date": "2025-01-01",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v1_parity",
        "rng_seed": 42,
        "timeframes": {"primary": "M1"},
        "warmup_bars": 10,
        "strategy_params": {
            "enabled_scenarios": [1],
            "direction_filter": "long",
            "window_length": 5,
            "ema_length": 5,
            "atr_length": 5,
            "atr_mult": 1.0,
            "b_b_length": 5,
            "std_factor": 1.0,
            "z_score_long": 1.0,
            "z_score_short": 1.0,
            "htf_filter": "none",
            "extra_htf_filter": "none",
        },
    }


class TestV1V2Parity:
    """Tests for V1 Python vs V2 parity."""

    def test_trade_count_parity(
        self,
        parity_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trade count must match between V1 and V2."""
        _patch_v1_data_paths(monkeypatch)
        v1_result = _run_v1_backtest(parity_config)
        v2_result = _run_v2_backtest(parity_config, tmp_path / "v2-count")

        assert len(v1_result.get("trades", [])) == len(v2_result.get("trades", []))

    def test_trade_events_parity(
        self,
        parity_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Entry/Exit events must match."""
        _patch_v1_data_paths(monkeypatch)
        v1_result = _run_v1_backtest(parity_config)
        v2_result = _run_v2_backtest(parity_config, tmp_path / "v2-events")

        for i, (v1, v2) in enumerate(
            zip(v1_result.get("trades", []), v2_result.get("trades", []))
        ):
            assert v1["entry_time_ns"] == v2["entry_time_ns"], (
                f"Trade {i}: entry_time mismatch"
            )
            assert v1["exit_time_ns"] == v2["exit_time_ns"], (
                f"Trade {i}: exit_time mismatch"
            )
            assert v1["direction"] == v2["direction"], (
                f"Trade {i}: direction mismatch"
            )
            assert v1["reason"] == v2["reason"], f"Trade {i}: reason mismatch"

    def test_price_parity_quantized(
        self,
        parity_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prices must match after tick-size quantization."""
        tick_size = Decimal("0.00001")

        _patch_v1_data_paths(monkeypatch)
        v1_result = _run_v1_backtest(parity_config)
        v2_result = _run_v2_backtest(parity_config, tmp_path / "v2-price")

        for i, (v1, v2) in enumerate(
            zip(v1_result.get("trades", []), v2_result.get("trades", []))
        ):
            v1_entry = _quantize_price(v1["entry_price"], tick_size)
            v2_entry = _quantize_price(v2["entry_price"], tick_size)
            assert v1_entry == v2_entry, f"Trade {i}: entry_price mismatch"

            v1_exit = _quantize_price(v1["exit_price"], tick_size)
            v2_exit = _quantize_price(v2["exit_price"], tick_size)
            assert v1_exit == v2_exit, f"Trade {i}: exit_price mismatch"

    def test_pnl_tolerance(
        self,
        parity_config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PnL must be within tolerance after rounding."""
        _patch_v1_data_paths(monkeypatch)
        v1_result = _run_v1_backtest(parity_config)
        v2_result = _run_v2_backtest(parity_config, tmp_path / "v2-pnl")

        for i, (v1, v2) in enumerate(
            zip(v1_result.get("trades", []), v2_result.get("trades", []))
        ):
            v1_pnl = round(v1["result"], 2)
            v2_pnl = round(v2["result"], 2)
            assert abs(v1_pnl - v2_pnl) <= 0.05, (
                f"Trade {i}: PnL mismatch V1={v1_pnl}, V2={v2_pnl}"
            )

        v1_total = round(sum(t["result"] for t in v1_result.get("trades", [])), 2)
        v2_total = round(sum(t["result"] for t in v2_result.get("trades", [])), 2)
        assert abs(v1_total - v2_total) <= 0.01, (
            f"Aggregate PnL mismatch: V1={v1_total}, V2={v2_total}"
        )


class TestDeterminism:
    """Tests for DEV-mode determinism."""

    def test_identical_runs(
        self, parity_config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Two identical DEV runs must produce identical results."""
        if not _ffi_available():
            pytest.skip("omega_bt module not available")

        result1 = run_v2_backtest(
            config_dict=parity_config, output_dir=tmp_path / "run1"
        )
        result2 = run_v2_backtest(
            config_dict=parity_config, output_dir=tmp_path / "run2"
        )

        _normalize_result(result1)
        _normalize_result(result2)

        assert result1 == result2, "DEV-mode runs are not deterministic"


def _run_v2_backtest(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if not _ffi_available():
        pytest.skip("omega_bt module not available")
    return run_v2_backtest(config_dict=config, output_dir=output_dir)


def _run_v1_backtest(config: dict[str, Any]) -> dict[str, Any]:
    if run_backtest_and_return_portfolio is None:
        pytest.skip("V1 runner not available")
    if not _data_available(config):
        pytest.skip("Parity data not available")

    v1_config = _v1_config_from_v2(config)
    portfolio, _ = run_backtest_and_return_portfolio(v1_config)

    trades = [_position_to_trade(pos) for pos in portfolio.closed_positions]
    return {"trades": trades}


def _v1_config_from_v2(config: dict[str, Any]) -> dict[str, Any]:
    timeframes = config.get("timeframes", {}) or {}
    costs = config.get("costs", {}) or {}
    account = config.get("account", {}) or {}
    params = config.get("strategy_parameters") or config.get("strategy_params") or {}

    mapped_params = dict(params)
    if "bb_length" in mapped_params and "b_b_length" not in mapped_params:
        mapped_params["b_b_length"] = mapped_params["bb_length"]
    if "htf_ema_length" in mapped_params and "htf_ema" not in mapped_params:
        mapped_params["htf_ema"] = mapped_params["htf_ema_length"]
    if "use_htf_filter" in mapped_params and "htf_filter" not in mapped_params:
        htf_filter_value = "both" if mapped_params["use_htf_filter"] else "none"
        mapped_params["htf_filter"] = htf_filter_value

    return {
        "symbol": config.get("symbol"),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
        "mode": "candle",
        "timeframes": {
            "primary": timeframes.get("primary", "M1"),
            "additional": timeframes.get("additional") or timeframes.get("htf") or [],
        },
        "warmup_bars": config.get("warmup_bars", 500),
        "initial_balance": account.get("initial_balance", 10000.0),
        "account_currency": account.get("account_currency", "EUR"),
        "risk_per_trade": account.get("risk_per_trade", 0.01),
        "execution": {
            "spread_multiplier": costs.get("spread_multiplier", 1.0),
            "fee_multiplier": costs.get("commission_multiplier", 1.0),
            "slippage_multiplier": costs.get("slippage_multiplier", 1.0),
            "random_seed": config.get("rng_seed"),
        },
        "strategy": {
            "module": "strategies.mean_reversion_z_score.backtest.backtest_strategy",
            "class": "MeanReversionZScoreStrategy",
            "parameters": mapped_params,
        },
    }


def _position_to_trade(position: Any) -> dict[str, Any]:
    entry_ns = _dt_to_ns(position.entry_time)
    exit_ns = _dt_to_ns(position.exit_time)
    return {
        "entry_time_ns": entry_ns,
        "exit_time_ns": exit_ns,
        "direction": position.direction,
        "entry_price": position.entry_price,
        "exit_price": position.exit_price,
        "result": position.result or 0.0,
        "reason": position.reason or "unknown",
    }


def _dt_to_ns(value: datetime | None) -> int:
    if value is None:
        return 0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1_000_000_000)


def _quantize_price(price: float, tick_size: Decimal) -> Decimal:
    return (Decimal(str(price)) / tick_size).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) * tick_size


def _normalize_result(result: dict[str, Any]) -> None:
    meta = result.get("meta")
    if isinstance(meta, dict):
        meta.pop("generated_at", None)
        meta.pop("generated_at_ns", None)
        meta.pop("runtime_seconds", None)


def _ffi_available() -> bool:
    try:
        import omega_bt  # noqa: F401
    except Exception:
        return False
    return True


def _data_available(config: dict[str, Any]) -> bool:
    symbol = config.get("symbol")
    if not symbol:
        return False

    timeframes = config.get("timeframes", {}) or {}
    tfs: Iterable[str] = [timeframes.get("primary", "")]
    tfs = list(tfs) + list(timeframes.get("additional") or timeframes.get("htf") or [])

    data_root = _parquet_root() / symbol
    for tf in tfs:
        if not tf:
            continue
        bid = data_root / f"{symbol}_{tf}_BID.parquet"
        ask = data_root / f"{symbol}_{tf}_ASK.parquet"
        if not bid.exists() or not ask.exists():
            return False
    return True


def _parquet_root() -> Path:
    env_root = os.environ.get("OMEGA_DATA_PARQUET_ROOT")
    if env_root:
        return Path(env_root)
    return FIXTURE_DATA_ROOT / "parquet"


def _patch_v1_data_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from hf_engine.infra.config import paths as hf_paths
    except Exception:
        return

    fixture_root = FIXTURE_DATA_ROOT
    if not fixture_root.exists():
        return
    monkeypatch.setattr(hf_paths, "RAW_DATA_DIR", fixture_root)
    monkeypatch.setattr(hf_paths, "PARQUET_DIR", fixture_root / "parquet")
