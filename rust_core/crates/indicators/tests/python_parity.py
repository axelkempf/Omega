from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.old.backtest_engine.core.indicator_cache import IndicatorCache  # noqa: E402


def make_candle(close: float) -> dict[str, float]:
    return {
        "open": close - 0.01,
        "high": close + 0.02,
        "low": close - 0.02,
        "close": close,
        "volume": 1.0,
    }


def series_to_list(series) -> list[float | None]:
    values = series.to_numpy(dtype="float64")
    out: list[float | None] = []
    for v in values:
        if not np.isfinite(v):
            out.append(None)
        else:
            out.append(float(v))
    return out


def main() -> None:
    closes = [1.00, 1.10, 1.05, 1.20, 1.15, 1.30]
    m1_candles = [make_candle(c) for c in closes]

    h1_c1 = make_candle(1.05)
    h1_c2 = make_candle(1.25)
    h1_candles = [h1_c1] * 3 + [h1_c2] * 3

    multi = {
        "M1": {"bid": m1_candles, "ask": m1_candles},
        "H1": {"bid": h1_candles, "ask": h1_candles},
    }

    cache = IndicatorCache(multi)

    ema = cache.ema("M1", "bid", period=3)
    ema_step = cache.ema_stepwise("H1", "bid", period=3)
    bb_upper, bb_mid, bb_lower = cache.bollinger_stepwise(
        "H1", "bid", period=3, std_factor=2.0
    )
    kz_step = cache.kalman_zscore_stepwise(
        "H1", "bid", window=3, R=0.5, Q=0.1
    )

    garch_local = cache.garch_volatility_local(
        "M1",
        "bid",
        idx=5,
        lookback=4,
        alpha=0.1,
        beta=0.8,
        omega=None,
        use_log_returns=True,
        scale=100.0,
        min_periods=2,
        sigma_floor=1e-6,
    )
    kgz_local = cache.kalman_garch_zscore_local(
        "M1",
        "bid",
        idx=5,
        lookback=4,
        R=0.01,
        Q=1.0,
        alpha=0.1,
        beta=0.8,
        omega=None,
        use_log_returns=True,
        scale=100.0,
        min_periods=2,
        sigma_floor=1e-6,
    )

    kgz_value = None
    if isinstance(kgz_local, float) and math.isfinite(kgz_local):
        kgz_value = float(kgz_local)

    payload = {
        "ema": series_to_list(ema),
        "ema_stepwise": series_to_list(ema_step),
        "bollinger_stepwise": {
            "upper": series_to_list(bb_upper),
            "middle": series_to_list(bb_mid),
            "lower": series_to_list(bb_lower),
        },
        "kalman_zscore_stepwise": series_to_list(kz_step),
        "garch_volatility_local": series_to_list(garch_local),
        "kalman_garch_zscore_local": kgz_value,
    }

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
