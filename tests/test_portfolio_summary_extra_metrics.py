from backtest_engine.core.portfolio import Portfolio


def test_get_summary_does_not_include_extra_robust_metrics_by_default():
    p = Portfolio(initial_balance=1000.0)
    summary = p.get_summary()

    assert "Cost Shock Score" not in summary
    assert "Timing Jitter Score" not in summary
    assert "Trade Dropout Score" not in summary
    assert "Ulcer Index" not in summary
    assert "Ulcer Index Score" not in summary


def test_get_summary_includes_extra_metrics_when_enabled_and_present():
    p = Portfolio(initial_balance=1000.0)
    p.enable_backtest_robust_metrics = True
    p.backtest_robust_metrics = {
        "robustness_1": 0.11,
        "robustness_1_num_samples": 5,
        "cost_shock_score": 0.22,
        "timing_jitter_score": 0.33,
        "trade_dropout_score": 0.44,
        "ulcer_index": 1.11,
        "ulcer_index_score": 0.77,
        "data_jitter_score": 0.88,
        "data_jitter_num_samples": 5,
        "p_mean_gt": 0.55,
        "stability_score": 0.66,
        "tp_sl_stress_score": 0.77,
    }

    summary = p.get_summary()

    assert summary["Robustness 1"] == 0.11
    assert summary["Robustness 1 Num Samples"] == 5
    assert summary["Cost Shock Score"] == 0.22
    assert summary["Timing Jitter Score"] == 0.33
    assert summary["Trade Dropout Score"] == 0.44
    assert summary["Ulcer Index"] == 1.11
    assert summary["Ulcer Index Score"] == 0.77
    assert summary["Data Jitter Score"] == 0.88
    assert summary["Data Jitter Num Samples"] == 5
    assert summary["p_mean_gt"] == 0.55
    assert summary["Stability Score"] == 0.66
    assert summary["TP/SL Stress Score"] == 0.77


def test_get_summary_includes_defaults_when_enabled_but_missing_dict():
    p = Portfolio(initial_balance=1000.0)
    p.enable_backtest_robust_metrics = True
    # backtest_robust_metrics missing / wrong type
    p.backtest_robust_metrics = None

    summary = p.get_summary()

    assert summary["Robustness 1"] == 0.0
    assert summary["Robustness 1 Num Samples"] == 0
    assert summary["Cost Shock Score"] == 0.0
    assert summary["Timing Jitter Score"] == 0.0
    assert summary["Trade Dropout Score"] == 0.0
    assert summary["Ulcer Index"] == 0.0
    assert summary["Ulcer Index Score"] == 0.0
    assert summary["Data Jitter Score"] == 0.0
    assert summary["Data Jitter Num Samples"] == 0
    assert summary["p_mean_gt"] == 1.0
    assert summary["Stability Score"] == 1.0
    assert summary["TP/SL Stress Score"] == 1.0
