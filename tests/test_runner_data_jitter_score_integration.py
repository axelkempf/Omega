import pandas as pd
import pytest

from backtest_engine import runner as r


class _DummyPortfolio:
    def __init__(self, kind: str):
        self.kind = kind

    def trades_to_dataframe(self) -> pd.DataFrame:
        # Minimal shape; actual computation is monkeypatched in this test.
        return pd.DataFrame({"r_multiple": [1.0, -1.0]})


def test_data_jitter_score_is_not_forced_to_cap_when_metrics_equal(monkeypatch):
    """Regression test: data jitter should use avg_r + winrate from metrics.

    Before the fix, the jitter runs used _metrics_from_portfolio(), which does not
    include avg_r/winrate. That implicitly set them to 0.0 and pushed the score to
    the penalty cap (0.5) even when profit/drawdown were unchanged.

    This test simulates "no impact" jitter runs (base == jitter metrics) and
    expects a perfect score of 1.0.
    """

    base_portfolio = _DummyPortfolio("base")

    def fake_calculate_metrics(portfolio):
        # Identical base/jitter metrics => no robustness penalty.
        return {
            "net_profit_after_fees_eur": 100.0,
            "avg_r_multiple": 0.8,
            "winrate_percent": 55.0,
            "drawdown_eur": 10.0,
            "fees_total_eur": 1.0,
            "total_trades": 10,
            "sharpe_trade": 0.5,
        }

    # Patch metrics function used by runner._compute_backtest_robust_metrics
    import backtest_engine.report.metrics as report_metrics

    monkeypatch.setattr(report_metrics, "calculate_metrics", fake_calculate_metrics)

    # Make jitter backtests cheap and deterministic
    def fake_run_backtest_and_return_portfolio(
        cfg, preloaded_data=None, prealigned=None
    ):
        return _DummyPortfolio("jitter"), None

    monkeypatch.setattr(
        r, "run_backtest_and_return_portfolio", fake_run_backtest_and_return_portfolio
    )

    # Avoid disk I/O / heavy computations inside robust metrics
    monkeypatch.setattr(r, "_load_base_preloaded_data", lambda *args, **kwargs: {})

    import backtest_engine.rating.data_jitter_score as dj

    seen = {"atr_period": None, "sigma_atr": []}

    def fake_precompute_atr_cache(*args, **kwargs):
        seen["atr_period"] = kwargs.get("period")
        return {}

    def fake_build_jittered_preloaded_data(*args, **kwargs):
        seen["sigma_atr"].append(kwargs.get("sigma_atr"))
        return {}

    monkeypatch.setattr(
        dj, "build_jittered_preloaded_data", fake_build_jittered_preloaded_data
    )
    monkeypatch.setattr(dj, "precompute_atr_cache", fake_precompute_atr_cache)

    import backtest_engine.rating.tp_sl_stress_score as tp_sl

    monkeypatch.setattr(
        tp_sl, "load_primary_candle_arrays_from_parquet", lambda *a, **k: None
    )
    monkeypatch.setattr(tp_sl, "compute_tp_sl_stress_score", lambda *a, **k: 1.0)

    import backtest_engine.rating.p_values as pvals

    monkeypatch.setattr(pvals, "compute_p_mean_r_gt_0", lambda *a, **k: 0.9)

    # Skip expensive stability reruns (year-by-year)
    monkeypatch.setattr(r, "_stability_score_yearly_reruns", lambda *a, **k: 1.0)

    import backtest_engine.rating.timing_jitter_score as tj

    monkeypatch.setattr(
        tj, "get_timing_jitter_backward_shift_months", lambda *a, **k: [1]
    )

    import backtest_engine.rating.trade_dropout_score as td

    monkeypatch.setattr(td, "simulate_trade_dropout_metrics_multi", lambda *a, **k: [])

    cfg = {
        "symbol": "EURUSD",
        "mode": "candle",
        "timeframes": {"primary": "H1", "additional": []},
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "reporting": {
            "enable_backtest_robust_metrics": True,
            # keep everything else default; we don't set robust_metrics_mode=r1_only
            # because that explicitly skips the data-jitter stage.
            "robust_metrics_mode": "full",
            "robust_dropout_runs": 1,
        },
        # Make parameter jitter a no-op for this test
        "robust_jitter_repeats": 0,
        "robust_data_jitter_repeats": 2,
        # Backward-compat: the repository configs use plural naming for these keys.
        "robust_data_jitter_atr_periods": 5,
        "robust_data_jitter_sigma_atrs": 0.123,
        "robust_data_jitter_penalty_cap": 0.5,
    }

    rob = r._compute_backtest_robust_metrics(cfg, base_portfolio)

    assert rob["data_jitter_failures"] == 0
    assert rob["data_jitter_num_samples"] == 2
    assert rob["data_jitter_score"] == pytest.approx(1.0)

    assert seen["atr_period"] == 5
    assert seen["sigma_atr"] == [0.123, 0.123]
