import numpy as np
import pandas as pd

from analysis.combined_walkforward_matrix_analyzer import (
    _EQUITY_CACHE,
    _TRADES_CACHE,
    _compute_additional_categorical_metrics,
    _compute_category_champions,
    get_equity_cached,
    get_trades_cached,
)


def _make_equity_series(
    rng: np.random.Generator, start: str = "2024-01-01", periods: int = 180
) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    rets = rng.normal(loc=0.0, scale=5.0, size=periods)
    eq = 10_000.0 + np.cumsum(rets)
    return pd.Series(eq, index=idx)


def _make_trades_df(rng: np.random.Generator, n: int = 30) -> pd.DataFrame:
    entry = pd.date_range("2024-02-01", periods=n, freq="12h", tz="UTC")
    # ensure exit > entry
    exit_ = entry + pd.to_timedelta(rng.integers(1, 48, size=n), unit="h")
    directions = np.where(rng.random(n) > 0.5, "long", "short")
    # commission is often negative in exports; keep mixed sign to exercise abs handling
    commission = rng.normal(loc=-0.2, scale=0.05, size=n)
    result = rng.normal(loc=1.0, scale=3.0, size=n)
    r_multiple = rng.normal(loc=0.1, scale=0.8, size=n)

    return pd.DataFrame(
        {
            "entry_time": entry.astype(str),
            "exit_time": exit_.astype(str),
            "direction": directions,
            "commission": commission,
            "result": result,
            "r_multiple": r_multiple,
        }
    )


def _make_candidate_matrix(n_portfolios: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for i in range(n_portfolios):
        eq = _make_equity_series(rng, start="2024-01-01", periods=220)
        trades_df = _make_trades_df(rng, n=40)

        total_profit = float(200 + 10 * i)
        total_max_dd = float(20 + i)

        rows.append(
            {
                "final_combo_pair_id": f"final_{i}",
                "total_profit": total_profit,
                "total_max_dd": total_max_dd,
                "total_profit_over_dd": total_profit / total_max_dd,
                "final_score": float(rng.uniform(0.0, 1.0)),
                "comp_score": float(rng.uniform(0.0, 1.0)),
                "stability_score_monthly": float(rng.uniform(0.0, 1.0)),
                "total_trades": int(rng.integers(50, 400)),
                "sharpe_trade": float(rng.uniform(0.0, 3.0)),
                "sortino_trade": float(rng.uniform(0.0, 4.0)),
                "avg_r": float(rng.uniform(-0.2, 0.8)),
                "winrate": float(rng.uniform(35.0, 75.0)),
                "equity_returns_volatility": float(rng.uniform(0.5, 3.0)),
                "equity_returns_skew": float(rng.normal(0.0, 1.0)),
                "equity_returns_kurtosis": float(rng.uniform(2.0, 8.0)),
                "equity_returns_autocorr": float(rng.uniform(-0.5, 0.5)),
                "identical_trades_absolut_percentage": float(rng.uniform(0.0, 0.2)),
                "_equity_internal": eq,
                "_trades_internal": trades_df,
            }
        )

    return pd.DataFrame(rows)


def test_compute_additional_categorical_metrics_smoke_chunked():
    matrix = _make_candidate_matrix(n_portfolios=7)
    out = _compute_additional_categorical_metrics(
        matrix.copy(), chunk_size=1, show_progress=False
    )

    expected_cols = {
        "worst_weekly_profit",
        "average_trade_duration_hours",
        "commission",
        "profit_without_commission",
        "fee_drag",
        "time_in_market_hours",
        "duration_days",
        "ulcer_index_weekly",
        "yearly_pnl_dispersion",
        "max_trades_simult",
        "long_short_overlap_episodes",
        "equity_curvature",
        "equity_log_vol",
        "dd_slope_stability",
        "time_in_highs",
    }
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == len(matrix)
    assert pd.api.types.is_integer_dtype(out["max_trades_simult"].dtype)
    assert pd.api.types.is_integer_dtype(out["long_short_overlap_episodes"].dtype)


def test_compute_category_champions_unique_portfolios_and_categories():
    matrix = _make_candidate_matrix(n_portfolios=40)
    champions = _compute_category_champions(matrix.copy())

    assert not champions.empty
    assert "category" in champions.columns
    assert champions["category"].nunique() == len(champions)
    assert champions["final_combo_pair_id"].nunique() == len(champions)
    assert len(champions) <= 11


def test_negative_cache_entries_are_sticky():
    _EQUITY_CACHE.clear()
    _TRADES_CACHE.clear()

    missing_eq = "this/path/does/not/exist/equity.csv"
    s1 = get_equity_cached(missing_eq)
    s2 = get_equity_cached(missing_eq)
    assert s1 is None and s2 is None
    assert missing_eq in _EQUITY_CACHE
    assert _EQUITY_CACHE[missing_eq] is None

    missing_tr = "this/path/does/not/exist/trades.json"
    t1 = get_trades_cached(missing_tr)
    t2 = get_trades_cached(missing_tr)
    assert t1 == [] and t2 == []
    assert missing_tr in _TRADES_CACHE
    assert _TRADES_CACHE[missing_tr] == []
