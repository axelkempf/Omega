import numpy as np
import pandas as pd

import backtest_engine.analysis.combined_walkforward_matrix_analyzer as cwm
import backtest_engine.analysis.walkforward_analyzer as wfa


def test_add_total_adust_metrics_to_portfolios_adds_columns_and_clips_negative_pod(
    monkeypatch,
):
    df = pd.DataFrame(
        {
            "avg_r": [0.5, 0.2],
            "winrate": [60.0, 40.0],
            "total_profit_over_dd": [2.0, -1.0],
            "total_trades": [10, 20],
            "duration_days": [30.0, 800.0],
        }
    )

    captured = {}

    def _fake_shrinkage_adjusted(*, average_r, n_trades, n_years, n_categories=1.0):
        captured["n_years"] = np.asarray(n_years)
        captured["n_categories"] = np.asarray(n_categories)
        return np.asarray(average_r) * 0.0 + 123.0

    def _fake_wilson(*, winrate, n_trades):
        # winrate must be decimals, not percent
        assert np.allclose(np.asarray(winrate), df["winrate"].to_numpy() / 100.0)
        return np.asarray(winrate)  # identity

    def _fake_risk_adjusted(
        *, profit_over_drawdown, n_trades, n_years, n_categories=1.0
    ):
        # negative PoD must be clipped to 0 before passing in
        assert np.all(np.asarray(profit_over_drawdown) >= 0.0)
        captured["n_categories_risk"] = np.asarray(n_categories)
        return np.asarray(profit_over_drawdown) * 0.0 + 7.0

    monkeypatch.setattr(cwm, "shrinkage_adjusted", _fake_shrinkage_adjusted)
    monkeypatch.setattr(cwm, "wilson_score_lower_bound", _fake_wilson)
    monkeypatch.setattr(cwm, "risk_adjusted", _fake_risk_adjusted)

    out = cwm._add_total_adust_metrics_to_portfolios(df)

    assert "winrate_adust" in out.columns
    assert "avg_r_adust" in out.columns
    assert "profit_over_dd_adust" in out.columns

    assert np.allclose(out["avg_r_adust"].to_numpy(), np.array([123.0, 123.0]))
    assert np.allclose(out["profit_over_dd_adust"].to_numpy(), np.array([7.0, 7.0]))
    # identity wilson -> winrate_adust should match raw percent
    assert np.allclose(out["winrate_adust"].to_numpy(), df["winrate"].to_numpy())

    # n_years derived from duration_days, but never below 1.0
    assert captured["n_years"].shape[0] == 2
    assert captured["n_years"][0] == 1.0
    assert captured["n_years"][1] > 1.0

    # groups_count is absent in this test df → n_categories must default to 1.0
    assert captured["n_categories"].shape[0] == 2
    assert np.allclose(captured["n_categories"], np.array([1.0, 1.0]))
    assert captured["n_categories_risk"].shape[0] == 2
    assert np.allclose(captured["n_categories_risk"], np.array([1.0, 1.0]))


def test_add_yearly_composite_scores_stores_adust_columns(monkeypatch):
    years = ["2020"]
    df = pd.DataFrame(
        {
            "winrate_combined_2020": [50.0, 70.0],
            "avg_r_combined_2020": [0.1, 0.2],
            "profit_over_dd_combined_2020": [2.0, -5.0],
            "trades_combined_2020": [10, 20],
        }
    )

    def _wilson(*, winrate, n_trades):
        return np.asarray([0.1, 0.2], dtype=float)

    def _shrink(*, average_r, n_trades, n_years):
        assert n_years == 1.0
        return np.asarray([0.3, 0.4], dtype=float)

    def _risk(*, profit_over_drawdown, n_trades, n_years):
        assert n_years == 1.0
        # input must be clipped non-negative
        assert np.all(np.asarray(profit_over_drawdown) >= 0.0)
        return np.asarray([1.0, 3.0], dtype=float)

    monkeypatch.setattr(wfa, "wilson_score_lower_bound", _wilson)
    monkeypatch.setattr(wfa, "shrinkage_adjusted", _shrink)
    monkeypatch.setattr(wfa, "risk_adjusted", _risk)

    out = wfa._add_yearly_composite_scores(df, years)

    assert "comp_score_2020" in out.columns
    assert "2020_winrate_adust" in out.columns
    assert "2020_avg_r_adust" in out.columns
    assert "2020_profit_over_dd_adust" in out.columns

    assert np.allclose(out["2020_winrate_adust"].to_numpy(), np.array([10.0, 20.0]))
    assert np.allclose(out["2020_avg_r_adust"].to_numpy(), np.array([0.3, 0.4]))
    assert np.allclose(
        out["2020_profit_over_dd_adust"].to_numpy(), np.array([1.0, 3.0])
    )

    # comp = (wr_adj + avg_r_adj + pod/(1+pod)) * 0.33
    pod_term = np.array([1.0 / 2.0, 3.0 / 4.0])
    expected = (np.array([0.1, 0.2]) + np.array([0.3, 0.4]) + pod_term) * 0.33
    assert np.allclose(out["comp_score_2020"].to_numpy(), expected)
