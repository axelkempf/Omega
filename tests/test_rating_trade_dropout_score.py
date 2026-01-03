import numpy as np
import pandas as pd

from backtest_engine.rating.trade_dropout_score import (
    compute_multi_run_trade_dropout_score,
    compute_trade_dropout_score,
    simulate_trade_dropout_metrics,
    simulate_trade_dropout_metrics_multi,
)


def test_simulate_trade_dropout_metrics_returns_base_metrics_when_dropout_leq_zero():
    base = {"profit": 10.0, "drawdown": 2.0, "sharpe": 1.5}
    df = pd.DataFrame({"result": [1.0, -1.0], "r_multiple": [0.1, -0.1]})

    out = simulate_trade_dropout_metrics(df, dropout_frac=0.0, base_metrics=base)
    assert out == base


def test_simulate_trade_dropout_metrics_handles_missing_pnl_column():
    base = {"profit": 10.0, "drawdown": 2.0, "sharpe": 1.5}
    df = pd.DataFrame({"oops": [1.0, -1.0]})

    out = simulate_trade_dropout_metrics(df, dropout_frac=0.5, base_metrics=base)
    assert out == base


def test_simulate_trade_dropout_metrics_is_deterministic_with_seed():
    df = pd.DataFrame(
        {
            "result": [1.0, -2.0, 3.0, -4.0, 5.0, 6.0],
            "r_multiple": [0.1, -0.2, 0.3, -0.4, 0.5, 0.6],
        }
    )

    a = simulate_trade_dropout_metrics(df, dropout_frac=0.5, seed=123)
    b = simulate_trade_dropout_metrics(df, dropout_frac=0.5, seed=123)
    assert a == b
    assert set(a.keys()) == {"profit", "drawdown", "sharpe"}
    assert all(np.isfinite(list(a.values())))


def test_simulate_trade_dropout_metrics_is_net_of_fees_when_total_fee_present():
    # Gross results are 0, fees are positive -> net results are negative.
    # With dropout_frac=0.5 and n=4 we always keep 2 trades -> profit = -2.
    df = pd.DataFrame(
        {
            "exit_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "result": [0.0, 0.0, 0.0, 0.0],
            "total_fee": [1.0, 1.0, 1.0, 1.0],
            "r_multiple": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = simulate_trade_dropout_metrics(df, dropout_frac=0.5, seed=123)
    assert np.isclose(out["profit"], -2.0)
    assert np.isclose(out["drawdown"], 2.0)


def test_simulate_trade_dropout_metrics_is_invariant_to_row_order_when_exit_time_present():
    df = pd.DataFrame(
        {
            "exit_time": ["2020-01-03", "2020-01-01", "2020-01-04", "2020-01-02"],
            "result": [1.0, 2.0, 3.0, 4.0],
            "total_fee": [0.5, 0.5, 0.5, 0.5],
            "r_multiple": [0.1, 0.2, 0.3, 0.4],
        }
    )
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    a = simulate_trade_dropout_metrics(df, dropout_frac=0.5, seed=123)
    b = simulate_trade_dropout_metrics(df_shuffled, dropout_frac=0.5, seed=123)
    assert a == b


def test_compute_trade_dropout_score_matches_expected_wrapper_behavior():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    stressed = {"profit": 50.0, "drawdown": 15.0, "sharpe": 1.0}

    score = compute_trade_dropout_score(base, stressed, penalty_cap=0.5)
    # Same 50%/50%/50% case -> score 0.5
    assert np.isclose(score, 0.5)


def test_simulate_trade_dropout_metrics_multi_returns_n_results():
    df = pd.DataFrame(
        {
            "result": [1.0, -2.0, 3.0, -4.0, 5.0, 6.0],
            "r_multiple": [0.1, -0.2, 0.3, -0.4, 0.5, 0.6],
        }
    )

    results = simulate_trade_dropout_metrics_multi(
        df, dropout_frac=0.3, n_runs=5, seed=123
    )

    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)
    assert all({"profit", "drawdown", "sharpe"} <= set(r.keys()) for r in results)


def test_simulate_trade_dropout_metrics_multi_is_deterministic():
    df = pd.DataFrame(
        {
            "result": [1.0, -2.0, 3.0, -4.0, 5.0, 6.0],
            "r_multiple": [0.1, -0.2, 0.3, -0.4, 0.5, 0.6],
        }
    )

    a = simulate_trade_dropout_metrics_multi(df, dropout_frac=0.3, n_runs=3, seed=42)
    b = simulate_trade_dropout_metrics_multi(df, dropout_frac=0.3, n_runs=3, seed=42)

    assert a == b


def test_simulate_trade_dropout_metrics_multi_zero_runs_returns_empty():
    df = pd.DataFrame(
        {
            "result": [1.0, -2.0, 3.0, -4.0, 5.0, 6.0],
            "r_multiple": [0.1, -0.2, 0.3, -0.4, 0.5, 0.6],
        }
    )

    results = simulate_trade_dropout_metrics_multi(df, dropout_frac=0.3, n_runs=0)

    assert results == []


def test_compute_multi_run_trade_dropout_score_averages_individual_scores():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}
    dropout_list = [
        {"profit": 80.0, "drawdown": 12.0, "sharpe": 1.8},
        {"profit": 60.0, "drawdown": 15.0, "sharpe": 1.5},
        {"profit": 40.0, "drawdown": 20.0, "sharpe": 1.0},
    ]

    score = compute_multi_run_trade_dropout_score(base, dropout_list, penalty_cap=0.5)

    assert 0.0 < score < 1.0


def test_compute_multi_run_trade_dropout_score_empty_returns_one():
    base = {"profit": 100.0, "drawdown": 10.0, "sharpe": 2.0}

    score = compute_multi_run_trade_dropout_score(base, [], penalty_cap=0.5)

    assert score == 1.0
