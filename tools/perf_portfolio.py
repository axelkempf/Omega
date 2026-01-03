from __future__ import annotations

import argparse
import json
sys_path_inserted = False
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import cProfile
import io
import pstats
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys_path_inserted = True

from backtest_engine.core.portfolio import Portfolio, PortfolioPosition


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
    stats.print_stats(25)
    return stream.getvalue()


def _make_positions(n: int, seed: int) -> list[PortfolioPosition]:
    rng = np.random.default_rng(seed)
    base_time = datetime(2020, 1, 1)
    dirs = np.where(rng.random(n) > 0.5, "long", "short")
    entries = 1.2000 + rng.normal(0.0, 0.0004, size=n)
    sl = entries - rng.uniform(0.0003, 0.0006, size=n)
    tp = entries + rng.uniform(0.0004, 0.0008, size=n)
    sizes = rng.uniform(0.05, 1.5, size=n)
    times = [base_time + timedelta(minutes=int(i)) for i in range(n)]

    positions: list[PortfolioPosition] = []
    for i in range(n):
        pos = PortfolioPosition(
            entry_time=times[i],
            direction=str(dirs[i]),
            symbol="EURUSD",
            entry_price=float(entries[i]),
            stop_loss=float(sl[i]),
            take_profit=float(tp[i]),
            size=float(sizes[i]),
            initial_stop_loss=float(sl[i]),
            initial_take_profit=float(tp[i]),
        )
        # Simuliere Trigger/Exit-Ergebnisse deterministisch
        exit_price = pos.take_profit if i % 3 != 0 else pos.stop_loss
        reason = "take_profit" if i % 3 != 0 else "stop_loss"
        pos.close(time=times[i] + timedelta(minutes=15), price=exit_price, reason=reason)
        positions.append(pos)
    return positions


def _run_portfolio(events: int, seed: int) -> Dict[str, Any]:
    positions = _make_positions(events, seed)
    portfolio = Portfolio(initial_balance=100_000.0)

    def _process():
        for pos in positions:
            portfolio.register_entry(pos)
            portfolio.register_fee(amount=pos.size * 0.5, time=pos.entry_time, kind="entry", position=pos)
            portfolio.register_exit(pos)
            portfolio.register_fee(amount=pos.size * 0.5, time=pos.exit_time, kind="exit", position=pos)
            portfolio.update(current_time=pos.exit_time or pos.entry_time)

    first_sec, first_peak = _measure(_process)
    second_sec, second_peak = _measure(_process)
    profile = _profile(_process)

    return {
        "events": int(events),
        "first_run_seconds": round(first_sec, 6),
        "first_peak_mb": round(first_peak, 6),
        "second_run_seconds": round(second_sec, 6),
        "second_peak_mb": round(second_peak, 6),
        "profile_top25": profile,
        "summary": portfolio.get_summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance baseline for Portfolio hot paths")
    parser.add_argument("-e", "--events", type=int, default=20000, help="Anzahl Trade-/Fee-Events")
    parser.add_argument("-s", "--seed", type=int, default=123, help="Seed für deterministische Daten")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_portfolio.json"),
        help="Pfad zur JSON-Ausgabe",
    )
    args = parser.parse_args()

    result = _run_portfolio(events=args.events, seed=args.seed)
    result["meta"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "sys_path_inserted": sys_path_inserted,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved portfolio baseline to {args.output}")


if __name__ == "__main__":
    main()
