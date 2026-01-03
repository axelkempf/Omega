import numpy as np

from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import (
    compute_stability_score_and_wmape_from_yearly_profits,
)


def test_compute_robustness_score_1_penalty_cap_zero_returns_one():
    base = {"profit": 100.0, "avg_r": 0.5, "winrate": 60.0, "drawdown": 10.0}
    score = compute_robustness_score_1(base, [{"profit": 0.0}], penalty_cap=0.0)
    assert score == 1.0


def test_compute_robustness_score_1_empty_jitter_returns_one_minus_cap():
    base = {"profit": 100.0, "avg_r": 0.5, "winrate": 60.0, "drawdown": 10.0}
    score = compute_robustness_score_1(base, [], penalty_cap=0.25)
    assert np.isclose(score, 0.75)


def test_compute_robustness_score_1_basic_case_matches_expected():
    base = {"profit": 100.0, "avg_r": 0.5, "winrate": 60.0, "drawdown": 10.0}
    # 50% profit drop, 50% avg_r drop, 50% winrate drop, 50% drawdown increase -> penalty 0.5 -> score 0.5
    jitter = {"profit": 50.0, "avg_r": 0.25, "winrate": 30.0, "drawdown": 15.0}
    score = compute_robustness_score_1(base, [jitter], penalty_cap=0.5)
    assert np.isclose(score, 0.5)


def test_stability_score_empty_mapping_returns_safe_defaults():
    score, wmape = compute_stability_score_and_wmape_from_yearly_profits({})
    assert score == 1.0
    assert wmape == 0.0


def test_stability_score_perfect_constant_daily_rate_is_one():
    # 2020 has 366 days, 2021 has 365 days
    profits = {2020: 366.0, 2021: 365.0}
    score, wmape = compute_stability_score_and_wmape_from_yearly_profits(profits)
    assert np.isclose(wmape, 0.0)
    assert np.isclose(score, 1.0)


def test_stability_score_penalizes_uneven_yearly_profits():
    profits = {2020: 1000.0, 2021: 0.0}
    score, wmape = compute_stability_score_and_wmape_from_yearly_profits(profits)
    assert 0.0 < score < 1.0
    assert wmape > 0.0


def test_stability_score_uses_provided_segment_durations_for_partial_years():
    # Constant daily profit rate (1.0 per day) across a full year + a partial year.
    profits = {2020: 366.0, 2021: 100.0}
    durations = {2020: 366.0, 2021: 100.0}

    score, wmape = compute_stability_score_and_wmape_from_yearly_profits(
        profits, durations_by_year=durations
    )
    assert np.isclose(wmape, 0.0)
    assert np.isclose(score, 1.0)
