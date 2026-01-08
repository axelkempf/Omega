"""
Parity tests for Rust vs Python rating implementations.

These tests explicitly compare Python and Rust implementations to ensure
numerical equivalence within acceptable floating-point tolerance.

The tests toggle OMEGA_USE_RUST_RATING to compare both paths directly.
"""

from __future__ import annotations

import importlib
import math
import os
from contextlib import contextmanager
from typing import Generator

import pytest

# Tolerance for floating-point comparison (16th decimal place variance observed)
REL_TOL = 1e-14
ABS_TOL = 1e-15


@contextmanager
def force_python_path() -> Generator[None, None, None]:
    """Force Python implementation by disabling Rust."""
    original = os.environ.get("OMEGA_USE_RUST_RATING")
    os.environ["OMEGA_USE_RUST_RATING"] = "false"
    try:
        yield
    finally:
        if original is not None:
            os.environ["OMEGA_USE_RUST_RATING"] = original
        else:
            os.environ.pop("OMEGA_USE_RUST_RATING", None)


@contextmanager
def force_rust_path() -> Generator[None, None, None]:
    """Force Rust implementation."""
    original = os.environ.get("OMEGA_USE_RUST_RATING")
    os.environ["OMEGA_USE_RUST_RATING"] = "true"
    try:
        yield
    finally:
        if original is not None:
            os.environ["OMEGA_USE_RUST_RATING"] = original
        else:
            os.environ.pop("OMEGA_USE_RUST_RATING", None)


def assert_close(
    python_result: float,
    rust_result: float,
    description: str,
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
) -> None:
    """Assert Python and Rust results are numerically equivalent."""
    assert math.isclose(
        python_result, rust_result, rel_tol=rel_tol, abs_tol=abs_tol
    ), (
        f"{description}: Python={python_result}, Rust={rust_result}, "
        f"diff={abs(python_result - rust_result)}"
    )


class TestRobustnessScore1Parity:
    """Parity tests for compute_robustness_score_1."""

    def test_basic_metrics_parity(self) -> None:
        """Test robustness_score_1 produces identical results."""
        base_metrics = {
            "profit": 1000.0,
            "avg_r": 0.5,
            "winrate": 0.55,
            "drawdown": 200.0,
        }
        jitter_metrics = [
            {"profit": 800.0, "avg_r": 0.4, "winrate": 0.50, "drawdown": 250.0},
            {"profit": 1200.0, "avg_r": 0.6, "winrate": 0.58, "drawdown": 180.0},
            {"profit": 900.0, "avg_r": 0.45, "winrate": 0.52, "drawdown": 220.0},
        ]

        with force_python_path():
            from src.backtest_engine.rating import robustness_score_1 as rs1_mod

            importlib.reload(rs1_mod)
            python_score = rs1_mod.compute_robustness_score_1(base_metrics, jitter_metrics)

        with force_rust_path():
            from src.backtest_engine.rating import robustness_score_1 as rs1_mod

            importlib.reload(rs1_mod)
            rust_score = rs1_mod.compute_robustness_score_1(base_metrics, jitter_metrics)

        assert_close(python_score, rust_score, "robustness_score_1")

    def test_edge_case_single_jitter(self) -> None:
        """Test with single jitter metric (edge case)."""
        base_metrics = {
            "profit": 500.0,
            "avg_r": 0.3,
            "winrate": 0.50,
            "drawdown": 100.0,
        }
        jitter_metrics = [
            {"profit": 450.0, "avg_r": 0.25, "winrate": 0.48, "drawdown": 120.0},
        ]

        with force_python_path():
            from src.backtest_engine.rating import robustness_score_1 as rs1_mod

            importlib.reload(rs1_mod)
            python_score = rs1_mod.compute_robustness_score_1(base_metrics, jitter_metrics)

        with force_rust_path():
            from src.backtest_engine.rating import robustness_score_1 as rs1_mod

            importlib.reload(rs1_mod)
            rust_score = rs1_mod.compute_robustness_score_1(base_metrics, jitter_metrics)

        assert_close(python_score, rust_score, "robustness_score_1 single jitter")


class TestStabilityScoreParity:
    """Parity tests for stability score calculations."""

    def test_wmape_calculation_parity(self) -> None:
        """Test stability score with WMAPE produces identical results."""
        # Mapping from year -> profit (required by the function signature)
        profits_by_year = {
            2019: 1000.0,
            2020: 1200.0,
            2021: 800.0,
            2022: 1100.0,
            2023: 950.0,
        }

        with force_python_path():
            from src.backtest_engine.rating import stability_score as ss_mod

            importlib.reload(ss_mod)
            py_score, py_wmape = ss_mod.compute_stability_score_and_wmape_from_yearly_profits(
                profits_by_year
            )

        with force_rust_path():
            from src.backtest_engine.rating import stability_score as ss_mod

            importlib.reload(ss_mod)
            rs_score, rs_wmape = ss_mod.compute_stability_score_and_wmape_from_yearly_profits(
                profits_by_year
            )

        assert_close(py_score, rs_score, "stability_score")
        assert_close(py_wmape, rs_wmape, "stability_wmape")

    def test_simple_stability_score_parity(self) -> None:
        """Test simple stability score function."""
        profits_by_year = {
            2018: 500.0,
            2019: 600.0,
            2020: 550.0,
            2021: 700.0,
            2022: 650.0,
            2023: 800.0,
        }

        with force_python_path():
            from src.backtest_engine.rating import stability_score as ss_mod

            importlib.reload(ss_mod)
            python_score = ss_mod.compute_stability_score_from_yearly_profits(profits_by_year)

        with force_rust_path():
            from src.backtest_engine.rating import stability_score as ss_mod

            importlib.reload(ss_mod)
            rust_score = ss_mod.compute_stability_score_from_yearly_profits(profits_by_year)

        assert_close(python_score, rust_score, "simple stability_score")


class TestStressPenaltyParity:
    """Parity tests for stress penalty calculations."""

    def test_penalty_calculation_parity(self) -> None:
        """Test stress penalty produces identical results."""
        base_metrics = {
            "profit": 1000.0,
            "drawdown": 200.0,
            "sharpe": 1.5,
        }
        stress_metrics = [
            {"profit": 800.0, "drawdown": 300.0, "sharpe": 1.0},
            {"profit": 750.0, "drawdown": 350.0, "sharpe": 0.8},
        ]

        with force_python_path():
            from src.backtest_engine.rating import stress_penalty as sp_mod

            importlib.reload(sp_mod)
            py_penalty = sp_mod.compute_penalty_profit_drawdown_sharpe(
                base_metrics, stress_metrics
            )
            py_score = sp_mod.score_from_penalty(py_penalty)

        with force_rust_path():
            from src.backtest_engine.rating import stress_penalty as sp_mod

            importlib.reload(sp_mod)
            rs_penalty = sp_mod.compute_penalty_profit_drawdown_sharpe(
                base_metrics, stress_metrics
            )
            rs_score = sp_mod.score_from_penalty(rs_penalty)

        assert_close(py_penalty, rs_penalty, "stress_penalty")
        assert_close(py_score, rs_score, "score_from_penalty")


class TestCostShockScoreParity:
    """Parity tests for cost shock score calculations."""

    def test_single_factor_parity(self) -> None:
        """Test single-factor cost shock score."""
        base_metrics = {
            "profit": 1000.0,
            "drawdown": 200.0,
            "sharpe": 1.5,
        }
        shocked_metrics = {
            "profit": 750.0,
            "drawdown": 280.0,
            "sharpe": 1.0,
        }

        with force_python_path():
            from src.backtest_engine.rating import cost_shock_score as cs_mod

            importlib.reload(cs_mod)
            python_score = cs_mod.compute_cost_shock_score(base_metrics, shocked_metrics)

        with force_rust_path():
            from src.backtest_engine.rating import cost_shock_score as cs_mod

            importlib.reload(cs_mod)
            rust_score = cs_mod.compute_cost_shock_score(base_metrics, shocked_metrics)

        assert_close(python_score, rust_score, "cost_shock_score single factor")

    def test_multi_factor_parity(self) -> None:
        """Test multi-factor cost shock score."""
        base_metrics = {
            "profit": 1000.0,
            "drawdown": 200.0,
            "sharpe": 1.5,
        }
        shocked_metrics_list = [
            {"profit": 900.0, "drawdown": 220.0, "sharpe": 1.3},
            {"profit": 800.0, "drawdown": 250.0, "sharpe": 1.1},
            {"profit": 700.0, "drawdown": 300.0, "sharpe": 0.9},
        ]

        with force_python_path():
            from src.backtest_engine.rating import cost_shock_score as cs_mod

            importlib.reload(cs_mod)
            python_score = cs_mod.compute_multi_factor_cost_shock_score(
                base_metrics, shocked_metrics_list
            )

        with force_rust_path():
            from src.backtest_engine.rating import cost_shock_score as cs_mod

            importlib.reload(cs_mod)
            rust_score = cs_mod.compute_multi_factor_cost_shock_score(
                base_metrics, shocked_metrics_list
            )

        assert_close(python_score, rust_score, "cost_shock_score multi factor")


class TestTradeDropoutScoreParity:
    """Parity tests for trade dropout score calculations."""

    def test_single_run_parity(self) -> None:
        """Test single-run trade dropout score."""
        base_metrics = {
            "profit": 1000.0,
            "drawdown": 200.0,
            "sharpe": 1.5,
        }
        dropout_metrics = {
            "profit": 850.0,
            "drawdown": 250.0,
            "sharpe": 1.2,
        }

        with force_python_path():
            from src.backtest_engine.rating import trade_dropout_score as td_mod

            importlib.reload(td_mod)
            python_score = td_mod.compute_trade_dropout_score(base_metrics, dropout_metrics)

        with force_rust_path():
            from src.backtest_engine.rating import trade_dropout_score as td_mod

            importlib.reload(td_mod)
            rust_score = td_mod.compute_trade_dropout_score(base_metrics, dropout_metrics)

        assert_close(python_score, rust_score, "trade_dropout_score single run")

    def test_multi_run_parity(self) -> None:
        """Test multi-run trade dropout score."""
        base_metrics = {
            "profit": 1000.0,
            "drawdown": 200.0,
            "sharpe": 1.5,
        }
        dropout_metrics_list = [
            {"profit": 950.0, "drawdown": 210.0, "sharpe": 1.4},
            {"profit": 900.0, "drawdown": 230.0, "sharpe": 1.3},
            {"profit": 850.0, "drawdown": 250.0, "sharpe": 1.2},
            {"profit": 800.0, "drawdown": 280.0, "sharpe": 1.0},
            {"profit": 750.0, "drawdown": 320.0, "sharpe": 0.8},
        ]

        with force_python_path():
            from src.backtest_engine.rating import trade_dropout_score as td_mod

            importlib.reload(td_mod)
            python_score = td_mod.compute_multi_run_trade_dropout_score(
                base_metrics, dropout_metrics_list
            )

        with force_rust_path():
            from src.backtest_engine.rating import trade_dropout_score as td_mod

            importlib.reload(td_mod)
            rust_score = td_mod.compute_multi_run_trade_dropout_score(
                base_metrics, dropout_metrics_list
            )

        assert_close(python_score, rust_score, "trade_dropout_score multi run")


@pytest.mark.skip(
    reason="Ulcer Index stays Python-only in Wave 1 (complex timestamp resampling)"
)
class TestUlcerIndexParity:
    """Parity tests for ulcer index calculations.

    Note: Ulcer Index calculation involves complex timestamp resampling
    (weekly closes) that is not yet implemented in Rust. The Rust function
    exists but the Python module doesn't dispatch to it yet.
    This test is skipped for Wave 1.
    """

    def test_ulcer_index_parity(self) -> None:
        """Test ulcer index from equity values (no timestamps - fallback path)."""
        equity_values = [
            10000.0,
            10200.0,
            10150.0,
            10300.0,
            10250.0,
            10400.0,
            10350.0,
            10500.0,
            10450.0,
            10600.0,
        ]

        with force_python_path():
            from src.backtest_engine.rating import ulcer_index_score as ui_mod

            importlib.reload(ui_mod)
            py_ulcer, py_score = ui_mod.compute_ulcer_index_and_score(equity_values)

        with force_rust_path():
            from src.backtest_engine.rating import ulcer_index_score as ui_mod

            importlib.reload(ui_mod)
            rs_ulcer, rs_score = ui_mod.compute_ulcer_index_and_score(equity_values)

        assert_close(py_ulcer, rs_ulcer, "ulcer_index")
        assert_close(py_score, rs_score, "ulcer_score")


@pytest.mark.skipif(
    os.environ.get("OMEGA_USE_RUST_RATING", "auto").lower() == "false",
    reason="Rust not enabled",
)
class TestRustAvailability:
    """Verify Rust is actually being used when enabled."""

    def test_rust_module_available(self) -> None:
        """Verify omega_rust module can be imported."""
        try:
            import omega_rust

            assert hasattr(omega_rust, "compute_robustness_score_1")
            assert hasattr(omega_rust, "compute_stability_score")
            assert hasattr(omega_rust, "compute_penalty_profit_drawdown_sharpe")
        except ImportError:
            pytest.skip("omega_rust module not available")

    def test_bridge_reports_enabled(self) -> None:
        """Verify bridge reports Rust as enabled."""
        with force_rust_path():
            from src.backtest_engine.rating import _rust_bridge as bridge_mod

            importlib.reload(bridge_mod)
            assert bridge_mod.is_rust_enabled(), "Rust should be enabled when forced"

    def test_bridge_reports_disabled(self) -> None:
        """Verify bridge reports Rust as disabled when forced off."""
        with force_python_path():
            from src.backtest_engine.rating import _rust_bridge as bridge_mod

            importlib.reload(bridge_mod)
            assert not bridge_mod.is_rust_enabled(), "Rust should be disabled when forced off"
