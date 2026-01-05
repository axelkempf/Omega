from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.core.event_engine import EventEngine
from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.indicator_cache import clear_indicator_cache_pool
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from backtest_engine.data.candle import Candle
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper


class _NoopStrategy:
    def __init__(self, primary_tf: str = "M15") -> None:
        self.primary_tf = primary_tf

    def get_primary_timeframe(self) -> str:
        return self.primary_tf

    def on_data(self, slice_map):
        return None


def _generate_candles(
    num_bars: int, spread: float = 0.0002
) -> Tuple[list[Candle], list[Candle]]:
    rng = np.random.default_rng(7)
    start = datetime(2020, 1, 1)
    ts = [start + timedelta(minutes=15 * i) for i in range(num_bars)]
    prices = rng.normal(loc=1.2, scale=0.01, size=num_bars)
    deltas = rng.normal(loc=0.0, scale=0.002, size=num_bars)
    bid_closes = prices + deltas
    bid_opens = prices
    bid_highs = np.maximum(bid_opens, bid_closes) + rng.random(num_bars) * 0.001
    bid_lows = np.minimum(bid_opens, bid_closes) - rng.random(num_bars) * 0.001
    vols = rng.integers(500, 5000, size=num_bars)

    bid = [
        Candle(
            timestamp=ts[i],
            open=float(bid_opens[i]),
            high=float(bid_highs[i]),
            low=float(bid_lows[i]),
            close=float(bid_closes[i]),
            volume=float(vols[i]),
            candle_type="bid",
        )
        for i in range(num_bars)
    ]
    ask = [
        Candle(
            timestamp=ts[i],
            open=float(bid_opens[i] + spread),
            high=float(bid_highs[i] + spread),
            low=float(bid_lows[i] + spread),
            close=float(bid_closes[i] + spread),
            volume=float(vols[i]),
            candle_type="ask",
        )
        for i in range(num_bars)
    ]
    return bid, ask


def _measure(func) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


def _profile(func) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(20)
    return stream.getvalue()


def benchmark_event_engine(num_bars: int) -> Dict[str, Any]:
    clear_indicator_cache_pool()
    bid, ask = _generate_candles(num_bars)
    multi = {"M15": {"bid": bid, "ask": ask}}

    portfolio = Portfolio(initial_balance=10000.0)
    exec_sim = ExecutionSimulator(
        portfolio=portfolio, slippage_model=SlippageModel(), fee_model=FeeModel()
    )
    strategy = _NoopStrategy(primary_tf="M15")
    wrapper = StrategyWrapper(strategy, portfolio=portfolio)

    engine = EventEngine(
        bid_candles=bid,
        ask_candles=ask,
        strategy=wrapper,
        executor=exec_sim,
        portfolio=portfolio,
        multi_candle_data=multi,
        symbol="EURUSD",
        original_start_dt=bid[0].timestamp,
    )

    first_run_seconds, first_peak_mb = _measure(engine.run)
    second_run_seconds, second_peak_mb = _measure(engine.run)
    profile_summary = _profile(engine.run)

    return {
        "meta": {
            "num_bars": num_bars,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "first_run_seconds": round(first_run_seconds, 6),
        "first_peak_mb": round(first_peak_mb, 6),
        "second_run_seconds": round(second_run_seconds, 6),
        "second_peak_mb": round(second_peak_mb, 6),
        "profile_top20": profile_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance baseline for EventEngine (synthetic data)"
    )
    parser.add_argument(
        "-n", "--num-bars", type=int, default=20000, help="Number of synthetic bars"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_event_engine.json"),
        help="Path to JSON output",
    )
    args = parser.parse_args()

    results = benchmark_event_engine(args.num_bars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved performance baseline to {args.output}")


if __name__ == "__main__":
    main()
