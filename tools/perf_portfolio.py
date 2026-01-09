from __future__ import annotations

import argparse
import json

sys_path_inserted = False
import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        pos.close(
            time=times[i] + timedelta(minutes=15), price=exit_price, reason=reason
        )
        positions.append(pos)
    return positions


def _run_portfolio(events: int, seed: int) -> Dict[str, Any]:
    positions = _make_positions(events, seed)
    portfolio = Portfolio(initial_balance=100_000.0)

    def _process():
        for pos in positions:
            portfolio.register_entry(pos)
            portfolio.register_fee(
                amount=pos.size * 0.5, time=pos.entry_time, kind="entry", position=pos
            )
            portfolio.register_exit(pos)
            portfolio.register_fee(
                amount=pos.size * 0.5, time=pos.exit_time, kind="exit", position=pos
            )
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


def _run_portfolio_batch(events: int, seed: int) -> Dict[str, Any]:
    """Batch-Processing für FFI-Overhead-Reduktion.

    Statt einzelner FFI-Calls werden alle Operationen in einem Batch gesammelt
    und in einem einzigen Call verarbeitet.
    """
    positions = _make_positions(events, seed)
    portfolio = Portfolio(initial_balance=100_000.0)

    def _build_batch_ops() -> List[Dict[str, Any]]:
        """Baut Liste aller Batch-Operationen."""
        ops: List[Dict[str, Any]] = []
        for pos in positions:
            # Entry mit Fee in einem Op
            ops.append(
                {
                    "type": "entry",
                    "position": pos,
                    "fee": pos.size * 0.5,
                    "fee_kind": "entry",
                }
            )
        return ops

    def _process_batch():
        batch_ops = _build_batch_ops()
        result = portfolio.process_batch(batch_ops)
        return result

    # Batch Processing - Entry Phase
    first_sec, first_peak = _measure(_process_batch)

    # Zweiter Durchlauf für Warm-Cache Messung
    portfolio2 = Portfolio(initial_balance=100_000.0)

    def _process_batch2():
        batch_ops = _build_batch_ops()
        return portfolio2.process_batch(batch_ops)

    second_sec, second_peak = _measure(_process_batch2)
    profile = _profile(_process_batch2)

    # Ergebnis der Batch-Verarbeitung
    final_result = portfolio.process_batch([])  # Leerer Batch für Status

    return {
        "events": int(events),
        "mode": "batch",
        "first_run_seconds": round(first_sec, 6),
        "first_peak_mb": round(first_peak, 6),
        "second_run_seconds": round(second_sec, 6),
        "second_peak_mb": round(second_peak, 6),
        "profile_top25": profile,
        "batch_stats": {
            "entries_registered": len(positions),
            "total_fees": portfolio.total_fees,
            "final_equity": portfolio.equity,
            "final_cash": portfolio.cash,
        },
        "summary": portfolio.get_summary(),
    }


def _run_comparison(events: int, seed: int) -> Dict[str, Any]:
    """Vergleich Sequential vs Batch Processing."""
    # Sequential
    seq_result = _run_portfolio(events, seed)

    # Batch
    batch_result = _run_portfolio_batch(events, seed)

    # Speedup berechnen
    seq_time = seq_result["first_run_seconds"]
    batch_time = batch_result["first_run_seconds"]
    speedup = seq_time / batch_time if batch_time > 0 else float("inf")

    return {
        "events": int(events),
        "sequential": {
            "first_run_seconds": seq_result["first_run_seconds"],
            "first_peak_mb": seq_result["first_peak_mb"],
        },
        "batch": {
            "first_run_seconds": batch_result["first_run_seconds"],
            "first_peak_mb": batch_result["first_peak_mb"],
        },
        "speedup": round(speedup, 3),
        "speedup_pct": round((speedup - 1) * 100, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance baseline for Portfolio hot paths"
    )
    parser.add_argument(
        "-e", "--events", type=int, default=20000, help="Anzahl Trade-/Fee-Events"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=123, help="Seed für deterministische Daten"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_portfolio.json"),
        help="Pfad zur JSON-Ausgabe",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Nutze Batch-Processing statt sequenzieller Operationen",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Vergleiche Sequential vs Batch Processing",
    )
    args = parser.parse_args()

    if args.compare:
        result = _run_comparison(events=args.events, seed=args.seed)
        output_path = args.output.parent / "p0-02_portfolio_batch_comparison.json"
    elif args.batch:
        result = _run_portfolio_batch(events=args.events, seed=args.seed)
        output_path = args.output.parent / "p0-01_portfolio_batch.json"
    else:
        result = _run_portfolio(events=args.events, seed=args.seed)
        output_path = args.output

    result["meta"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "mode": (
            "batch" if args.batch else ("compare" if args.compare else "sequential")
        ),
        "sys_path_inserted": sys_path_inserted,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved portfolio baseline to {output_path}")

    if args.compare:
        print(f"\n📊 Comparison Results ({args.events} events):")
        print(f"   Sequential: {result['sequential']['first_run_seconds']:.4f}s")
        print(f"   Batch:      {result['batch']['first_run_seconds']:.4f}s")
        print(
            f"   Speedup:    {result['speedup']:.2f}x ({result['speedup_pct']:+.1f}%)"
        )


if __name__ == "__main__":
    main()
