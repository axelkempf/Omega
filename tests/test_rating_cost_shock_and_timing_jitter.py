from copy import deepcopy

import numpy as np

from backtest_engine.rating.cost_shock_score import (
    COST_SHOCK_FACTORS,
    apply_cost_shock_inplace,
    compute_cost_shock_score,
    compute_multi_factor_cost_shock_score,
)
from backtest_engine.rating.timing_jitter_score import (
    apply_timing_jitter_month_shift_inplace,
    compute_timing_jitter_score,
)


def test_apply_cost_shock_inplace_sets_execution_multipliers_without_mutating_sections():
    cfg = {
        "slippage": {"pips": 1.0, "enabled": True},
        "fees": {"fixed": 2, "note": "x", "lot_size": 100_000.0},
        "commission": {"per_lot": 3.0},
    }
    apply_cost_shock_inplace(cfg, factor=2.0)

    # Sections are not scaled directly (avoids double application; multipliers are used downstream).
    assert cfg["slippage"]["pips"] == 1.0
    assert cfg["slippage"]["enabled"] is True
    assert cfg["fees"]["fixed"] == 2
    assert cfg["fees"]["lot_size"] == 100_000.0
    assert cfg["fees"]["note"] == "x"
    assert cfg["commission"]["per_lot"] == 3.0
    # Multipliers are the primary mechanism (prevents silent no-ops when costs come from YAML defaults)
    assert cfg["execution"]["slippage_multiplier"] == 2.0
    assert cfg["execution"]["fee_multiplier"] == 2.0


def test_apply_cost_shock_inplace_factor_leq_zero_is_noop():
    cfg = {"fees": {"fixed": 2.0}}
    before = deepcopy(cfg)
    apply_cost_shock_inplace(cfg, factor=0.0)
    assert cfg == before


def test_apply_cost_shock_inplace_works_without_explicit_cost_sections():
    cfg = {}
    apply_cost_shock_inplace(cfg, factor=1.5)
    assert cfg["execution"]["slippage_multiplier"] == 1.5
    assert cfg["execution"]["fee_multiplier"] == 1.5


def test_compute_cost_shock_score_wraps_stress_penalty_and_is_bounded():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    shocked = {"profit": 50.0, "drawdown": 15.0, "sharpe": 1.0}

    score = compute_cost_shock_score(base, shocked, penalty_cap=0.5)
    assert np.isclose(score, 0.5)


def test_compute_multi_factor_cost_shock_score_averages_individual_scores():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    shocked = [
        {"profit": 80.0, "drawdown": 12.0, "sharpe": 1.8},
        {"profit": 60.0, "drawdown": 15.0, "sharpe": 1.5},
        {"profit": 40.0, "drawdown": 20.0, "sharpe": 1.0},
    ]

    score = compute_multi_factor_cost_shock_score(base, shocked, penalty_cap=0.5)

    assert 0.0 <= score <= 1.0
    assert score < 1.0  # average should reflect degradation


def test_compute_multi_factor_cost_shock_score_empty_returns_one():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    score = compute_multi_factor_cost_shock_score(base, [])
    assert score == 1.0


def test_cost_shock_factors_are_deterministic_and_expected():
    assert COST_SHOCK_FACTORS == (1.25, 1.50, 2.00)
    assert len(COST_SHOCK_FACTORS) == 3


def test_apply_timing_jitter_month_shift_inplace_shift_zero_is_noop():
    cfg = {
        "start_date": "2020-01-01",
        "end_date": "2020-02-01",
        "timeframes": {"primary": "D1"},
    }
    before = deepcopy(cfg)
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=0)
    assert cfg == before


def test_apply_timing_jitter_month_shift_inplace_invalid_dates_are_ignored():
    cfg = {"start_date": "not-a-date", "end_date": "2020-02-01"}
    before = deepcopy(cfg)
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=1)
    assert cfg == before


def test_apply_timing_jitter_month_shift_inplace_shifts_date_only_windows_in_months():
    cfg = {
        "start_date": "2020-01-01",
        "end_date": "2020-02-01",
        "timeframes": {"primary": "H1"},
    }
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=1)
    assert cfg["start_date"] == "2019-12-01"
    assert cfg["end_date"] == "2020-01-01"


def test_compute_timing_jitter_score_is_bounded_and_uses_penalty():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    jitters = [
        {"profit": 50.0, "drawdown": 15.0, "sharpe": 1.0},
        {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0},
    ]
    score = compute_timing_jitter_score(base, jitters, penalty_cap=0.5)
    assert 0.0 <= score <= 1.0


def test_cost_shock_fee_multiplier_scales_commission_model_using_execution_costs_yaml():
    import pytest

    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    from backtest_engine.sizing.commission import CommissionModel, Side
    from backtest_engine.sizing.rate_provider import StaticRateProvider
    from backtest_engine.sizing.symbol_specs_registry import (
        SymbolSpec,
        SymbolSpecsRegistry,
    )

    exec_costs = yaml.safe_load(
        Path("configs/execution_costs.yaml").read_text(encoding="utf-8")
    )
    specs = SymbolSpecsRegistry(
        {
            "EURUSD": SymbolSpec(
                symbol="EURUSD",
                contract_size=100_000.0,
                tick_size=0.0001,
                tick_value=10.0,
                quote_currency="USD",
                base_currency="EUR",
            )
        }
    )
    rp = StaticRateProvider({}, strict=False)

    base = CommissionModel("USD", rp, exec_costs, specs, multiplier=1.0)
    shocked = CommissionModel("USD", rp, exec_costs, specs, multiplier=2.0)

    fee_base = base.fee_for_order("EURUSD", lots=1.0, price=1.0, side=Side.ENTRY)
    fee_shocked = shocked.fee_for_order("EURUSD", lots=1.0, price=1.0, side=Side.ENTRY)
    assert fee_shocked == round(fee_base * 2.0, 2)
