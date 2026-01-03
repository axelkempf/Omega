import numpy as np

from backtest_engine.rating.stress_penalty import (
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
)


def test_stress_penalty_cap_zero_returns_zero_and_full_score():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    stress = [{"profit": 50.0, "drawdown": 15.0, "sharpe": 1.0}]

    penalty = compute_penalty_profit_drawdown_sharpe(base, stress, penalty_cap=0.0)
    assert penalty == 0.0

    score = score_from_penalty(penalty, penalty_cap=0.0)
    assert score == 1.0


def test_stress_penalty_empty_stress_metrics_returns_zero():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}

    penalty = compute_penalty_profit_drawdown_sharpe(base, [], penalty_cap=0.5)
    assert penalty == 0.0


def test_stress_penalty_basic_case_and_score_match_expected():
    # 50% profit drop, 50% drawdown increase, 50% sharpe drop -> mean penalty 0.5
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    stressed = {"profit": 50.0, "drawdown": 15.0, "sharpe": 1.0}

    penalty = compute_penalty_profit_drawdown_sharpe(base, [stressed], penalty_cap=0.5)
    assert np.isclose(penalty, 0.5)

    score = score_from_penalty(penalty, penalty_cap=0.5)
    assert np.isclose(score, 0.5)


def test_stress_penalty_is_clipped_to_cap():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    stressed = {"profit": 0.0, "drawdown": 10_000.0, "sharpe": 0.0}

    penalty = compute_penalty_profit_drawdown_sharpe(base, [stressed], penalty_cap=0.25)
    assert np.isclose(penalty, 0.25)


def test_score_from_penalty_handles_non_finite_penalty():
    score = score_from_penalty(float("nan"), penalty_cap=0.5)
    assert 0.0 <= score <= 1.0
