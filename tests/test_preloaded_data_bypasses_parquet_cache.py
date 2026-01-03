from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from dateutil import tz

from backtest_engine.data.candle import Candle
from backtest_engine.data.data_handler import CSVDataHandler, reset_candle_build_caches


def _make_df(start: str, values: list[float]) -> pd.DataFrame:
    ts0 = pd.Timestamp(start).tz_convert("UTC")
    times = [ts0 + pd.Timedelta(hours=i) for i in range(len(values))]
    return pd.DataFrame(
        {
            "UTC time": times,
            "Open": values,
            "High": [v + 0.1 for v in values],
            "Low": [v - 0.1 for v in values],
            "Close": values,
            "Volume": [1.0 for _ in values],
        }
    )


def test_preloaded_data_wins_over_parquet_build_cache(monkeypatch: pytest.MonkeyPatch):
    """Regression: preloaded_data darf nicht durch _PARQUET_BUILD_CACHE überschrieben werden.

    Hintergrund:
    - Robust Metrics (Data Jitter) generieren preloaded/jittered DataFrames und rufen
      anschließend den normalen Candle-Load-Pfad.
    - Wenn ein Base-Backtest zuvor Candles aus Parquet geladen hat, liegt ggf. ein
      _PARQUET_BUILD_CACHE-Eintrag vor.
    - Der Loader muss in diesem Fall trotzdem die preloaded DataFrames verwenden.
    """

    reset_candle_build_caches()

    # Handler so konfigurieren, dass er preloaded_data für TF=H1 nutzt.
    pre_df = _make_df("2021-01-01T00:00:00Z", [200.0, 201.0, 202.0])
    handler = CSVDataHandler(
        symbol="TEST",
        timeframe="H1",
        preloaded_data={("H1", "bid"): pre_df},
        normalize_to_timeframe=False,
    )

    start_dt = datetime(2021, 1, 1, tzinfo=tz.UTC)
    end_dt = datetime(2021, 1, 1, 2, tzinfo=tz.UTC)

    # Simuliere einen vorhandenen Parquet-Cache-Eintrag (Base-Candles), der früher
    # fälschlicherweise auch dann zurückgegeben wurde, wenn preloaded_data existiert.
    import backtest_engine.data.data_handler as dh_mod

    fake_path = dh_mod.PARQUET_DIR / "TEST" / "TEST_H1_BID.parquet"
    cache_key_parquet = (
        str(fake_path),
        "bid",
        start_dt.isoformat(),
        end_dt.isoformat(),
        False,
    )
    dh_mod._PARQUET_BUILD_CACHE[cache_key_parquet] = [
        Candle(
            timestamp=pd.Timestamp("2021-01-01T00:00:00Z"),
            open=111.0,
            high=111.0,
            low=111.0,
            close=111.0,
            volume=1.0,
            candle_type="bid",
        )
    ]

    candles = handler._load_parquet(fake_path, "bid", start_dt, end_dt)

    # Erwartung: preloaded_data gewinnt – NICHT der Parquet-Cache.
    assert candles, "Es müssen Candles aus preloaded_data gebaut werden"
    assert candles[0].open == 200.0
    assert candles[0].close == 200.0
