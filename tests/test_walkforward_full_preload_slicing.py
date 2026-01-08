import pandas as pd


def test_safe_slice_df_by_time_supports_utc_time_column():
    """Ensure walkforward full-preload slicing works with the repo's 'UTC time' schema."""
    from backtest_engine.optimizer.walkforward import _safe_slice_df_by_time

    times = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"UTC time": times, "Close": range(10)})

    # Pass naive datetime boundaries (as used by walkforward window generation)
    out = _safe_slice_df_by_time(df, "2021-01-03", "2021-01-05")

    assert not out.empty
    assert out["UTC time"].min() == pd.Timestamp("2021-01-03", tz="UTC")
    assert out["UTC time"].max() == pd.Timestamp("2021-01-05", tz="UTC")
