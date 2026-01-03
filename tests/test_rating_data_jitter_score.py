import numpy as np
import pandas as pd

from backtest_engine.rating.data_jitter_score import (
    _stable_data_jitter_seed,
    build_jittered_preloaded_data,
    compute_atr_series,
    compute_data_jitter_score,
    precompute_atr_cache,
)


def _sample_bid_ask() -> dict:
    bid = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [1, 1, 1],
        }
    )
    ask = bid + 0.0002
    return {("M15", "bid"): bid, ("M15", "ask"): ask}


def _shift_from_close(jittered: pd.DataFrame, base: pd.DataFrame) -> np.ndarray:
    return jittered["Close"].to_numpy() - base["Close"].to_numpy()


def test_shift_jitter_preserves_candle_constraints():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)
    jittered = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=123, min_price=1e-6
    )

    for key, df in jittered.items():
        high_ge = (
            df["High"].to_numpy() >= np.maximum(df["Open"], df["Close"]).to_numpy()
        )
        low_le = df["Low"].to_numpy() <= np.minimum(df["Open"], df["Close"]).to_numpy()
        assert np.all(high_ge)
        assert np.all(low_le)


def test_jitter_is_deterministic_with_same_seed():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)

    jittered_a = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=999, min_price=1e-6
    )
    jittered_b = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=999, min_price=1e-6
    )

    for key in jittered_a.keys():
        np.testing.assert_allclose(jittered_a[key]["Close"], jittered_b[key]["Close"])


def test_jitter_differs_with_different_seeds():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)

    jittered_a = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=1, min_price=1e-6
    )
    jittered_b = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=2, min_price=1e-6
    )

    close_a = jittered_a[("M15", "bid")]["Close"].to_numpy()
    close_b = jittered_b[("M15", "bid")]["Close"].to_numpy()
    assert not np.allclose(close_a, close_b)


def test_compute_data_jitter_score_empty_list_returns_one_minus_cap():
    base_metrics = {"profit": 1.0, "avg_r": 1.0, "winrate": 1.0, "drawdown": 1.0}
    score = compute_data_jitter_score(base_metrics, [], penalty_cap=0.5)
    assert score == 0.5


def test_atr_series_warmup_uses_expanding_mean():
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [1.0, 3.0, 5.0],
            "Low": [0.0, 0.0, 0.0],
            "Close": [0.5, 0.5, 0.5],
            "Volume": [1, 1, 1],
        }
    )
    atr = compute_atr_series(df, period=3)
    expected = pd.Series([1.0, 2.0, 3.0])
    np.testing.assert_allclose(atr.reset_index(drop=True), expected)


def test_bid_ask_receive_identical_epsilon():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)

    jittered = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=42, min_price=1e-6
    )

    bid_diff = (
        jittered[("M15", "bid")]["Close"].to_numpy()
        - base[("M15", "bid")]["Close"].to_numpy()
    )
    ask_diff = (
        jittered[("M15", "ask")]["Close"].to_numpy()
        - base[("M15", "ask")]["Close"].to_numpy()
    )

    np.testing.assert_allclose(bid_diff, ask_diff)


def test_jitter_shift_is_cumulative_prefix_sum():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)

    jittered = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=2025, min_price=1e-6
    )

    shift = _shift_from_close(jittered[("M15", "bid")], base[("M15", "bid")])
    delta = np.empty_like(shift)
    delta[0] = shift[0]
    delta[1:] = np.diff(shift)
    reconstructed = np.cumsum(delta)

    np.testing.assert_allclose(shift, reconstructed)


def test_jitter_delta_signs_alternate():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)

    jittered = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.05, seed=77, min_price=1e-6
    )

    shift = _shift_from_close(jittered[("M15", "bid")], base[("M15", "bid")])
    delta = np.diff(np.insert(shift, 0, shift[0]))
    non_zero = delta[np.abs(delta) > 1e-12]
    if len(non_zero) > 1:
        assert np.all(non_zero[:-1] * non_zero[1:] < 0)


def test_jitter_respects_min_price_with_global_offset():
    base = _sample_bid_ask()
    atr_cache = precompute_atr_cache(base, period=3)

    jittered = build_jittered_preloaded_data(
        base, atr_cache=atr_cache, sigma_atr=0.2, seed=31415, min_price=10.0
    )

    for _, df in jittered.items():
        assert (df[["Open", "High", "Low", "Close"]] >= 10.0).all().all()


def test_stable_seed_is_deterministic_and_32bit():
    seed_a = _stable_data_jitter_seed(1234, 0)
    seed_b = _stable_data_jitter_seed(1234, 0)
    seed_c = _stable_data_jitter_seed(1234, 1)

    assert seed_a == seed_b
    assert seed_a != seed_c
    assert 0 <= seed_a < 2**32
