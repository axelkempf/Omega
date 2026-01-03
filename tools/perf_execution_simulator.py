from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import cProfile
import io
import pstats
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from backtest_engine.data.candle import Candle
from backtest_engine.sizing.rate_provider import StaticRateProvider
from backtest_engine.strategy.strategy_wrapper import TradeSignal


def _generate_candles(num_bars: int, spread: float = 0.0002) -> Tuple[list[Candle], list[Candle]]:
    """Generate monotonic-up candles so TPs hit deterministically."""
    start = datetime(2020, 1, 1)
    ts = [start + timedelta(minutes=15 * i) for i in range(num_bars)]
    base = np.linspace(1.2, 1.2 + 0.01, num_bars)
    bid_opens = base
    bid_closes = base + 0.0002
    bid_highs = bid_closes + 0.0004
    bid_lows = bid_opens - 0.0004
    vols = np.full(num_bars, 2000.0)

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


def _run_sim(sim: ExecutionSimulator, bid: list[Candle], ask: list[Candle], num_signals: int) -> None:
    for idx, (b, a) in enumerate(zip(bid, ask)):
        if idx < num_signals:
            entry = b.close
            signal = TradeSignal(
                direction="long",
                entry_price=entry,
                stop_loss=entry - 0.0006,
                take_profit=entry + 0.0006,
                symbol="EURUSD",
                timestamp=b.timestamp,
                type="market",
            )
            sim.process_signal(signal)
        sim.evaluate_exits(b, a)


def benchmark_execution_simulator(num_bars: int, num_signals: int) -> Dict[str, Any]:
    bid, ask = _generate_candles(num_bars)
    rp = StaticRateProvider({"EURUSD": 1.10})
    rp.account_currency = "USD"  # type: ignore[attr-defined]
    sim = ExecutionSimulator(
        portfolio=Portfolio(initial_balance=10000.0),
        slippage_model=SlippageModel(fixed_pips=0.0, random_pips=0.0),
        fee_model=FeeModel(per_million=5.0),
        rate_provider=rp,
    )

    first_seconds, first_peak = _measure(lambda: _run_sim(sim, bid, ask, num_signals))
    second_seconds, second_peak = _measure(lambda: _run_sim(sim, bid, ask, num_signals))
    profile_summary = _profile(lambda: _run_sim(sim, bid, ask, num_signals))

    return {
        "meta": {
            "num_bars": num_bars,
            "num_signals": num_signals,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "first_run_seconds": round(first_seconds, 6),
        "first_peak_mb": round(first_peak, 6),
        "second_run_seconds": round(second_seconds, 6),
        "second_peak_mb": round(second_peak, 6),
        "profile_top20": profile_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance baseline for ExecutionSimulator (synthetic data)")
    parser.add_argument("-n", "--num-bars", type=int, default=20000, help="Number of synthetic bars")
    parser.add_argument("-s", "--num-signals", type=int, default=1000, help="Number of market signals")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_execution_simulator.json"),
        help="Path to JSON output",
    )
    args = parser.parse_args()

    results = benchmark_execution_simulator(args.num_bars, args.num_signals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved performance baseline to {args.output}")


if __name__ == "__main__":
    main()
