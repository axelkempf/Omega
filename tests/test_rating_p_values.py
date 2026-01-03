import pandas as pd

from backtest_engine.rating.p_values import (
    bootstrap_p_value_mean_gt_zero,
    compute_p_values,
)


def test_bootstrap_p_value_returns_one_for_empty_or_too_small_inputs():
    assert bootstrap_p_value_mean_gt_zero([]) == 1.0
    assert bootstrap_p_value_mean_gt_zero([1.0]) == 1.0


def test_bootstrap_p_value_is_zero_for_all_positive_samples():
    x = [1.0, 2.0, 3.0, 4.0]
    p = bootstrap_p_value_mean_gt_zero(x, n_boot=200, seed=123)
    assert p == 0.0


def test_bootstrap_p_value_is_one_for_all_negative_samples():
    x = [-1.0, -2.0, -3.0, -4.0]
    p = bootstrap_p_value_mean_gt_zero(x, n_boot=200, seed=123)
    assert p == 1.0


def test_compute_p_values_returns_standard_keys_and_bounds():
    trades = pd.DataFrame({"result": [1.0, -1.0, 2.0], "r_multiple": [0.1, -0.1, 0.2]})
    out = compute_p_values(trades, n_boot=200, seed_r=1, seed_pnl=2)

    assert set(out.keys()) == {"p_mean_r_gt_0", "p_net_profit_gt_0"}
    assert 0.0 <= float(out["p_mean_r_gt_0"]) <= 1.0
    assert 0.0 <= float(out["p_net_profit_gt_0"]) <= 1.0


def test_compute_p_values_net_profit_is_net_of_fees_by_default_when_total_fee_present():
    # Gross results are positive, but fees make net results negative -> p should be 1.0 under net-of-fees.
    trades = pd.DataFrame(
        {
            "result": [1.0, 1.0, 1.0, 1.0],
            "total_fee": [2.0, 2.0, 2.0, 2.0],
            "r_multiple": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out_net = compute_p_values(trades, n_boot=200, seed_r=1, seed_pnl=2)
    assert out_net["p_net_profit_gt_0"] == 1.0

    out_gross = compute_p_values(
        trades, n_boot=200, seed_r=1, seed_pnl=2, net_of_fees_pnl=False
    )
    assert out_gross["p_net_profit_gt_0"] == 0.0


def test_compute_p_values_empty_df_returns_ones():
    out = compute_p_values(pd.DataFrame())
    assert out["p_mean_r_gt_0"] == 1.0
    assert out["p_net_profit_gt_0"] == 1.0
