#!/usr/bin/env python3
"""Benchmark: Python vs Rust IndicatorCache Performance."""

import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_data(n_bars: int = 100_000):
    """Create mock OHLCV data for benchmarking."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="5min")
    prices = 1.1 + np.cumsum(np.random.randn(n_bars) * 0.0001)

    class MockCandle:
        __slots__ = ("open", "high", "low", "close", "timestamp")

        def __init__(self, o, h, l, c, ts):
            self.open = o
            self.high = h
            self.low = l
            self.close = c
            self.timestamp = ts

    # Format expected by IndicatorCache: Dict[tf, Dict[side, List[Candle]]]
    data = {"M5": {"bid": [], "ask": []}, "H1": {"bid": [], "ask": []}}

    for i in range(n_bars):
        o = prices[i]
        h = o + abs(np.random.randn() * 0.0005)
        l = o - abs(np.random.randn() * 0.0005)
        c = o + np.random.randn() * 0.0003
        data["M5"]["bid"].append(MockCandle(o, h, l, c, dates[i]))
        data["M5"]["ask"].append(
            MockCandle(o + 0.0001, h + 0.0001, l + 0.0001, c + 0.0001, dates[i])
        )

    # H1 als Subset (jede 12. Kerze)
    data["H1"]["bid"] = data["M5"]["bid"][::12]
    data["H1"]["ask"] = data["M5"]["ask"][::12]

    return data


def run_benchmark(n_bars: int = 100_000):
    """Run benchmark comparing Python vs Rust indicator calculations."""
    from importlib import reload

    import backtest_engine.core.indicator_cache as ic_module
    from backtest_engine.core.indicator_cache import (
        _check_rust_indicator_cache_available,
    )

    print(f"Test-Daten: {n_bars:,} Bars")
    print(f"Rust verfügbar: {_check_rust_indicator_cache_available()}")
    print()

    data = create_mock_data(n_bars)

    indicators = [
        ("EMA(20)", lambda c: c.ema("M5", "bid", 20)),
        ("SMA(50)", lambda c: c.sma("M5", "bid", 50)),
        ("RSI(14)", lambda c: c.rsi("M5", "bid", 14)),
        ("ATR(14)", lambda c: c.atr("M5", "bid", 14)),
        ("Bollinger(20)", lambda c: c.bollinger("M5", "bid", 20)),
        ("DMI(14)", lambda c: c.dmi("M5", "bid", 14)),
        ("MACD(12,26,9)", lambda c: c.macd("M5", "bid", 12, 26, 9)),
        ("ROC(14)", lambda c: c.roc("M5", "bid", 14)),
        ("Choppiness(14)", lambda c: c.choppiness("M5", "bid", 14)),
        ("Kalman Mean", lambda c: c.kalman_mean("M5", "bid", 0.01, 1.0)),
        ("Kalman ZScore", lambda c: c.kalman_zscore("M5", "bid", 100, 0.01, 1.0)),
        ("ZScore(100)", lambda c: c.zscore("M5", "bid", 100)),
        ("EMA Stepwise(20)", lambda c: c.ema_stepwise("H1", "bid", 20)),
        ("Kalman ZScore Stepwise", lambda c: c.kalman_zscore_stepwise("H1", "bid", 100, 0.01, 1.0)),
        ("GARCH Volatility", lambda c: c.garch_volatility("M5", "bid", 0.05, 0.90)),
        ("Kalman GARCH ZScore", lambda c: c.kalman_garch_zscore("M5", "bid", 0.01, 1.0, 0.05, 0.90)),
    ]

    # Benchmark ohne Rust
    os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
    reload(ic_module)

    cache_py = ic_module.IndicatorCache(data)

    print("=== Python (ohne Rust) ===")
    py_times = {}
    for name, fn in indicators:
        cache_py._ind_cache.clear()
        start = time.perf_counter()
        _ = fn(cache_py)
        elapsed = time.perf_counter() - start
        py_times[name] = elapsed
        print(f"  {name}: {elapsed*1000:.2f}ms")

    # Benchmark mit Rust
    os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
    reload(ic_module)

    cache_rust = ic_module.IndicatorCache(data)

    print()
    print("=== Rust (aktiviert) ===")
    rust_times = {}
    for name, fn in indicators:
        cache_rust._ind_cache.clear()
        start = time.perf_counter()
        _ = fn(cache_rust)
        elapsed = time.perf_counter() - start
        rust_times[name] = elapsed
        print(f"  {name}: {elapsed*1000:.2f}ms")

    print()
    print("=== Speedup (Python/Rust) ===")
    total_py = sum(py_times.values())
    total_rust = sum(rust_times.values())
    for name in py_times:
        speedup = py_times[name] / rust_times[name] if rust_times[name] > 0 else 0
        status = "✅" if speedup > 1.5 else "⚠️" if speedup > 1.0 else "❌"
        print(f"  {status} {name}: {speedup:.1f}x")

    print()
    print(f"Gesamt Python: {total_py*1000:.1f}ms")
    print(f"Gesamt Rust:   {total_rust*1000:.1f}ms")
    print(f"Gesamt Speedup: {total_py/total_rust:.1f}x")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    run_benchmark(n)
