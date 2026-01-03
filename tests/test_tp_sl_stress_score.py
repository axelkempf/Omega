from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine.rating.tp_sl_stress_score import (
    align_primary_candles,
    compute_tp_sl_stress_score,
    load_primary_candle_arrays_from_parquet,
)


def test_align_primary_candles_returns_none_on_missing_columns():
    bid = pd.DataFrame({"time": ["2020-01-01"], "High": [1.0], "Low": [0.5]})
    ask = pd.DataFrame({"UTC time": ["2020-01-01"], "High": [1.1], "Low": [0.6]})
    assert align_primary_candles(bid, ask) is None


def test_align_primary_candles_aligns_on_utc_time_and_sorts():
    bid = pd.DataFrame(
        {
            "UTC time": ["2020-01-02", "2020-01-01"],
            "High": [2.0, 1.5],
            "Low": [1.0, 1.0],
        }
    )
    ask = pd.DataFrame(
        {
            "UTC time": ["2020-01-01", "2020-01-02"],
            "High": [1.6, 2.1],
            "Low": [1.1, 1.2],
        }
    )
    arrays = align_primary_candles(bid, ask)
    assert arrays is not None
    assert set(arrays.keys()) == {
        "times_ns",
        "bid_high",
        "bid_low",
        "ask_high",
        "ask_low",
    }
    assert arrays["times_ns"].shape[0] == 2
    # sorted ascending
    assert arrays["times_ns"][0] < arrays["times_ns"][1]


def test_load_primary_candle_arrays_from_parquet_prefers_uppercase_then_lowercase(
    tmp_path, monkeypatch
):
    sym = "EURUSD"
    tf = "M1"
    sym_dir = tmp_path / sym
    sym_dir.mkdir(parents=True)

    # Create empty files to satisfy Path.exists()
    (sym_dir / f"{sym}_{tf}_BID.parquet").write_bytes(b"x")
    (sym_dir / f"{sym}_{tf}_ask.parquet").write_bytes(b"x")

    bid_df = pd.DataFrame({"UTC time": ["2020-01-01"], "High": [1.5], "Low": [1.0]})
    ask_df = pd.DataFrame({"UTC time": ["2020-01-01"], "High": [1.6], "Low": [1.1]})

    def _fake_read_parquet(path: Path):
        if str(path).endswith("BID.parquet"):
            return bid_df
        return ask_df

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    arrays = load_primary_candle_arrays_from_parquet(sym, tf, parquet_dir=tmp_path)
    assert arrays is not None
    assert arrays["times_ns"].shape[0] == 1


def test_compute_tp_sl_stress_score_returns_one_when_inputs_missing():
    assert compute_tp_sl_stress_score(None, None) == 1.0
    assert compute_tp_sl_stress_score(pd.DataFrame(), None) == 1.0


def test_compute_tp_sl_stress_score_simple_trade_penalty_delay():
    # Build arrays for 3 daily candles
    times = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    times_ns = times.astype("int64").to_numpy()

    # Long trade: TP reached only on last candle
    arrays = {
        "times_ns": times_ns,
        "bid_high": np.array([1.0, 1.1, 1.30], dtype=float),
        "bid_low": np.array([0.9, 0.9, 0.9], dtype=float),
        "ask_high": np.array([1.0, 1.1, 1.30], dtype=float),
        "ask_low": np.array([0.9, 0.9, 0.9], dtype=float),
    }

    trades = pd.DataFrame(
        {
            "reason": ["take_profit"],
            "direction": ["long"],
            "take_profit": [1.20],
            "stop_loss": [0.80],
            "initial_stop_loss": [0.80],
            "meta": [{"prices": {"spread": 0.01}}],
            # Exit time intentionally set to entry time -> orig_exit_idx == entry idx
            "entry_time": [times[0]],
            "exit_time": [times[0]],
        }
    )

    score = compute_tp_sl_stress_score(trades, arrays, debug=False)
    # TP should be found (delayed by 1 bar relative to orig_exit_idx) -> penalty 0.1 -> score 0.9
    assert np.isclose(score, 0.9)
