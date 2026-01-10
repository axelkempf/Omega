#!/usr/bin/env python3
"""Benchmark: Python vs Rust IndicatorCache Performance with Memory Tracking."""

import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    time_ms: float
    peak_memory_kb: float
    current_memory_kb: float


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


def benchmark_indicator(
    cache, name: str, fn, clear_cache: bool = True
) -> BenchmarkResult:
    """Benchmark a single indicator with memory tracking."""
    if clear_cache:
        cache._ind_cache.clear()

    # Force garbage collection before measurement
    gc.collect()

    # Start memory tracking
    tracemalloc.start()

    start = time.perf_counter()
    _ = fn(cache)
    elapsed = time.perf_counter() - start

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=name,
        time_ms=elapsed * 1000,
        peak_memory_kb=peak / 1024,
        current_memory_kb=current / 1024,
    )


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
        (
            "Kalman ZScore Stepwise",
            lambda c: c.kalman_zscore_stepwise("H1", "bid", 100, 0.01, 1.0),
        ),
        ("GARCH Volatility", lambda c: c.garch_volatility("M5", "bid", 0.05, 0.90)),
        (
            "Kalman GARCH ZScore",
            lambda c: c.kalman_garch_zscore("M5", "bid", 0.01, 1.0, 0.05, 0.90),
        ),
    ]

    # Benchmark ohne Rust
    os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
    reload(ic_module)

    cache_py = ic_module.IndicatorCache(data)

    print("=== Python (ohne Rust) ===")
    print(
        f"{'Indikator':<25} {'Zeit (ms)':>10} {'Peak RAM (KB)':>14} {'Alloc (KB)':>12}"
    )
    print("-" * 65)
    py_results: dict[str, BenchmarkResult] = {}
    for name, fn in indicators:
        result = benchmark_indicator(cache_py, name, fn)
        py_results[name] = result
        print(
            f"  {name:<23} {result.time_ms:>10.2f} {result.peak_memory_kb:>14.1f} {result.current_memory_kb:>12.1f}"
        )

    # Benchmark mit Rust
    os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
    reload(ic_module)

    cache_rust = ic_module.IndicatorCache(data)

    print()
    print("=== Rust (aktiviert) ===")
    print(
        f"{'Indikator':<25} {'Zeit (ms)':>10} {'Peak RAM (KB)':>14} {'Alloc (KB)':>12}"
    )
    print("-" * 65)
    rust_results: dict[str, BenchmarkResult] = {}
    for name, fn in indicators:
        result = benchmark_indicator(cache_rust, name, fn)
        rust_results[name] = result
        print(
            f"  {name:<23} {result.time_ms:>10.2f} {result.peak_memory_kb:>14.1f} {result.current_memory_kb:>12.1f}"
        )

    print()
    print("=== Vergleich (Python vs Rust) ===")
    print(f"{'Indikator':<25} {'Speedup':>8} {'RAM Diff':>12} {'Status':>8}")
    print("-" * 55)

    total_py_time = sum(r.time_ms for r in py_results.values())
    total_rust_time = sum(r.time_ms for r in rust_results.values())
    total_py_mem = sum(r.peak_memory_kb for r in py_results.values())
    total_rust_mem = sum(r.peak_memory_kb for r in rust_results.values())

    for name in py_results:
        py_r = py_results[name]
        rust_r = rust_results[name]
        speedup = py_r.time_ms / rust_r.time_ms if rust_r.time_ms > 0 else 0
        mem_diff = rust_r.peak_memory_kb - py_r.peak_memory_kb
        mem_pct = (
            (mem_diff / py_r.peak_memory_kb * 100) if py_r.peak_memory_kb > 0 else 0
        )

        # Status: Zeit-basiert
        time_status = "✅" if speedup > 1.5 else "⚠️" if speedup > 1.0 else "❌"
        # Memory: negativ = Rust spart Speicher
        mem_status = "💾" if mem_diff < 0 else ""

        print(
            f"  {name:<23} {speedup:>7.1f}x {mem_diff:>+10.1f}KB {time_status}{mem_status:>5}"
        )

    print()
    print("=== Zusammenfassung ===")
    print(f"{'Metrik':<25} {'Python':>12} {'Rust':>12} {'Diff':>12}")
    print("-" * 55)
    print(
        f"  {'Gesamt Zeit':<23} {total_py_time:>10.1f}ms {total_rust_time:>10.1f}ms {total_py_time/total_rust_time:>10.1f}x"
    )
    print(
        f"  {'Gesamt Peak RAM':<23} {total_py_mem:>10.1f}KB {total_rust_mem:>10.1f}KB {total_rust_mem - total_py_mem:>+10.1f}KB"
    )
    print()
    print(f"Zeit-Speedup:    {total_py_time/total_rust_time:.1f}x schneller")
    mem_ratio = total_rust_mem / total_py_mem if total_py_mem > 0 else 1.0
    if mem_ratio < 1.0:
        print(f"RAM-Effizienz:   {(1-mem_ratio)*100:.1f}% weniger Speicher")
    else:
        print(f"RAM-Overhead:    {(mem_ratio-1)*100:.1f}% mehr Speicher (FFI-Kosten)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    run_benchmark(n)
