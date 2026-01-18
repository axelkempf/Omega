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


def ema_with_warmup(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("EMA period must be > 0")
    if len(values) < period:
        return [None] * len(values)

    alpha = 2.0 / (period + 1.0)
    out: list[float | None] = [None] * len(values)
    sma = float(np.mean(values[:period]))
    out[period - 1] = sma
    prev = sma

    for i in range(period, len(values)):
        v = values[i]
        if not np.isfinite(v):
            out[i] = prev
            continue
        prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev

    return out


def expand_and_ffill(
    reduced: list[float | None], indices: list[int], length: int
) -> list[float | None]:
    out: list[float | None] = [None] * length
    for i, idx in enumerate(indices):
        if idx < length and i < len(reduced):
            out[idx] = reduced[i]

    last: float | None = None
    for i, value in enumerate(out):
        if value is not None and math.isfinite(value):
            last = value
        elif last is not None:
            out[i] = last
    return out


def zscore_rolling(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: list[float | None] = [None] * len(values)
    if len(values) < window:
        return out

    for i in range(window - 1, len(values)):
        window_vals = values[i - window + 1 : i + 1]
        if not all(np.isfinite(v) for v in window_vals):
            continue
        mean = float(np.mean(window_vals))
        denom = window - 1
        if denom <= 0:
            continue
        variance = float(
            np.sum([(v - mean) ** 2 for v in window_vals]) / denom
        )
        std = math.sqrt(variance)
        if std > 0.0 and math.isfinite(std):
            out[i] = (values[i] - mean) / std
        else:
            out[i] = 0.0
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

    ema = ema_with_warmup(closes, period=3)

    h1_new_idx = cache._stepwise_indices("H1", "bid")
    reduced = [h1_candles[i]["close"] for i in h1_new_idx]
    reduced_ema = ema_with_warmup(reduced, period=3)
    ema_step = expand_and_ffill(reduced_ema, h1_new_idx, len(h1_candles))
    bb_step_upper, bb_step_mid, bb_step_lower = cache.bollinger_stepwise(
        "H1", "bid", period=3, std_factor=2.0
    )
    kz_step = cache.kalman_zscore_stepwise(
        "H1", "bid", window=3, R=0.5, Q=0.1
    )

    atr = cache.atr("M1", "bid", period=3)
    bb_upper, bb_mid, bb_lower = cache.bollinger(
        "M1", "bid", period=3, std_factor=2.0
    )
    zscore_roll = zscore_rolling(closes, window=3)
    zscore_ema = cache.zscore(
        "M1", "bid", window=3, mean_source="ema", ema_period=2
    )
    kalman_zscore = cache.kalman_zscore(
        "M1", "bid", window=3, R=0.5, Q=0.1
    )
    garch_vol = cache.garch_volatility(
        "M1",
        "bid",
        alpha=0.1,
        beta=0.8,
        omega=None,
        use_log_returns=True,
        scale=100.0,
        min_periods=2,
        sigma_floor=1e-6,
    )
    kalman_garch_z = cache.kalman_garch_zscore(
        "M1",
        "bid",
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
        "ema": ema,
        "ema_stepwise": ema_step,
        "bollinger_stepwise": {
            "upper": series_to_list(bb_step_upper),
            "middle": series_to_list(bb_step_mid),
            "lower": series_to_list(bb_step_lower),
        },
        "kalman_zscore_stepwise": series_to_list(kz_step),
        "atr": series_to_list(atr),
        "bollinger": {
            "upper": series_to_list(bb_upper),
            "middle": series_to_list(bb_mid),
            "lower": series_to_list(bb_lower),
        },
        "zscore_rolling": zscore_roll,
        "zscore_ema": series_to_list(zscore_ema),
        "kalman_zscore": series_to_list(kalman_zscore),
        "garch_volatility": series_to_list(garch_vol),
        "kalman_garch_zscore": series_to_list(kalman_garch_z),
        "garch_volatility_local": series_to_list(garch_local),
        "kalman_garch_zscore_local": kgz_value,
    }

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
