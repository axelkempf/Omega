from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest_engine.core.indicator_cache import IndicatorCache


@dataclass
class OpResult:
    first_call_seconds: float
    first_peak_mb: float
    cached_call_seconds: float
    cached_peak_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "first_call_seconds": round(self.first_call_seconds, 6),
            "first_peak_mb": round(self.first_peak_mb, 6),
            "cached_call_seconds": round(self.cached_call_seconds, 6),
            "cached_peak_mb": round(self.cached_peak_mb, 6),
        }


def _generate_candles(num_bars: int) -> Dict[str, Dict[str, list[dict[str, float]]]]:
    rng = np.random.default_rng(42)
    base_prices = rng.normal(loc=1.2, scale=0.01, size=num_bars)
    deltas = rng.normal(loc=0.0, scale=0.002, size=num_bars)
    opens = base_prices
    closes = base_prices + deltas
    highs = np.maximum(opens, closes) + rng.random(num_bars) * 0.001
    lows = np.minimum(opens, closes) - rng.random(num_bars) * 0.001
    volumes = rng.integers(1_000, 10_000, size=num_bars)

    def _candles() -> list[dict[str, float]]:
        return [
            {
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]),
            }
            for i in range(num_bars)
        ]

    dataset = _candles()
    return {"M15": {"bid": dataset, "ask": dataset}}


def _measure(func: Callable[[], Any]) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


def _profile_block(func: Callable[[], Any]) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(15)
    return stream.getvalue()


def benchmark_indicator_cache(num_bars: int, repetitions: int) -> Dict[str, Any]:
    data = _generate_candles(num_bars)
    init_duration, init_peak = _measure(lambda: IndicatorCache(data))
    cache = IndicatorCache(data)

    def _op(name: str, fn: Callable[[], Any]) -> OpResult:
        first_duration, first_peak = _measure(fn)
        cached_duration, cached_peak = _measure(fn)
        return OpResult(first_duration, first_peak, cached_duration, cached_peak)

    operations: Dict[str, Callable[[], Any]] = {
        "ema": lambda: cache.ema("M15", "bid", 50),
        "ema_stepwise": lambda: cache.ema_stepwise("M15", "bid", 50),
        "sma": lambda: cache.sma("M15", "bid", 50),
        "rsi": lambda: cache.rsi("M15", "bid", 14),
        "macd": lambda: cache.macd("M15", "bid", 12, 26, 9),
        "roc": lambda: cache.roc("M15", "bid", 14),
        "dmi": lambda: cache.dmi("M15", "bid", 14),
        "bollinger": lambda: cache.bollinger("M15", "bid", 20, 2.0),
        "bollinger_stepwise": lambda: cache.bollinger_stepwise("M15", "bid", 20, 2.0),
        "atr": lambda: cache.atr("M15", "bid", 14),
    }

    op_results: Dict[str, Dict[str, float]] = {
        name: _op(name, fn).to_dict() for name, fn in operations.items()
    }

    def _profile_all() -> None:
        cache_profile = IndicatorCache(data)
        ops_profile = {
            "ema": lambda: cache_profile.ema("M15", "bid", 50),
            "ema_stepwise": lambda: cache_profile.ema_stepwise("M15", "bid", 50),
            "sma": lambda: cache_profile.sma("M15", "bid", 50),
            "rsi": lambda: cache_profile.rsi("M15", "bid", 14),
            "macd": lambda: cache_profile.macd("M15", "bid", 12, 26, 9),
            "roc": lambda: cache_profile.roc("M15", "bid", 14),
            "dmi": lambda: cache_profile.dmi("M15", "bid", 14),
            "bollinger": lambda: cache_profile.bollinger("M15", "bid", 20, 2.0),
            "bollinger_stepwise": lambda: cache_profile.bollinger_stepwise(
                "M15", "bid", 20, 2.0
            ),
            "atr": lambda: cache_profile.atr("M15", "bid", 14),
        }
        for _ in range(repetitions):
            for fn in ops_profile.values():
                fn()

    profile_summary = _profile_block(_profile_all)

    return {
        "meta": {
            "num_bars": num_bars,
            "repetitions": repetitions,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "init_seconds": round(init_duration, 6),
        "init_peak_mb": round(init_peak, 6),
        "operations": op_results,
        "profile_top15": profile_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance baseline for IndicatorCache (synthetic data)"
    )
    parser.add_argument(
        "-n",
        "--num-bars",
        type=int,
        default=50_000,
        help="Number of synthetic bars to generate",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per operation for profiling",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_indicator_cache.json"),
        help="Path to the JSON output file",
    )
    args = parser.parse_args()

    results = benchmark_indicator_cache(args.num_bars, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved performance baseline to {args.output}")


if __name__ == "__main__":
    main()
