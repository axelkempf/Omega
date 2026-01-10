#!/usr/bin/env python3
"""
Event Engine Performance Benchmark.

Compares Python vs Rust backend performance with proper isolation:
- Memory tracking via tracemalloc
- Separate process spawning to avoid cache contamination
- Multiple runs for statistical significance
"""

import gc
import json
import os
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    backend: str
    time_seconds: float
    peak_memory_mb: float
    current_memory_mb: float
    bars_processed: int
    total_trades: int
    final_balance: float


def run_single_backtest(use_rust: bool) -> BenchmarkResult:
    """
    Run a single backtest with memory tracking.

    Uses subprocess isolation to avoid cache contamination between runs.
    """
    env = os.environ.copy()
    env["OMEGA_USE_RUST_EVENT_ENGINE"] = "true" if use_rust else "false"
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)

    # Run backtest in subprocess for clean state
    script = """
import gc
import json
import sys
import time
import tracemalloc

gc.collect()
tracemalloc.start()

from src.backtest_engine.runner import run_backtest_and_return_portfolio
from pathlib import Path

config_path = Path("configs/backtest/mean_reversion_z_score.json")
with open(config_path) as f:
    config = json.load(f)

start = time.perf_counter()
portfolio, _ = run_backtest_and_return_portfolio(config)
elapsed = time.perf_counter() - start

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

result = {
    "time_seconds": elapsed,
    "peak_memory_mb": peak / 1024 / 1024,
    "current_memory_mb": current / 1024 / 1024,
    "bars_processed": getattr(portfolio, "bars_processed", 0),
    "total_trades": len(portfolio.closed_positions),
    "final_balance": portfolio.cash,
}
print(json.dumps(result))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    if result.returncode != 0:
        print(f"Error running backtest: {result.stderr}")
        raise RuntimeError(f"Backtest failed: {result.stderr}")

    # Parse JSON from last line of output
    output_lines = result.stdout.strip().split("\n")
    json_line = output_lines[-1]
    data = json.loads(json_line)

    return BenchmarkResult(
        backend="rust" if use_rust else "python",
        time_seconds=data["time_seconds"],
        peak_memory_mb=data["peak_memory_mb"],
        current_memory_mb=data["current_memory_mb"],
        bars_processed=data["bars_processed"],
        total_trades=data["total_trades"],
        final_balance=data["final_balance"],
    )


def run_benchmark(num_runs: int = 3) -> dict[str, Any]:
    """
    Run benchmark comparing Python and Rust backends.

    Each run is in a separate subprocess to ensure:
    - No shared cache between Python and Rust runs
    - No data preloading advantage
    - Clean memory state
    """
    print("=" * 60)
    print("Event Engine Performance Benchmark")
    print("=" * 60)
    print(f"Runs per backend: {num_runs}")
    print()

    # Python backend runs
    print("=== Python Backend ===")
    print(f"{'Run':<5} {'Time (s)':>10} {'Peak RAM (MB)':>15} {'Trades':>8}")
    print("-" * 45)

    py_results: list[BenchmarkResult] = []
    for i in range(num_runs):
        result = run_single_backtest(use_rust=False)
        py_results.append(result)
        print(
            f"  {i+1:<3} {result.time_seconds:>10.3f} {result.peak_memory_mb:>15.1f} {result.total_trades:>8}"
        )

    # Rust backend runs
    print()
    print("=== Rust Backend ===")
    print(f"{'Run':<5} {'Time (s)':>10} {'Peak RAM (MB)':>15} {'Trades':>8}")
    print("-" * 45)

    rust_results: list[BenchmarkResult] = []
    for i in range(num_runs):
        result = run_single_backtest(use_rust=True)
        rust_results.append(result)
        print(
            f"  {i+1:<3} {result.time_seconds:>10.3f} {result.peak_memory_mb:>15.1f} {result.total_trades:>8}"
        )

    # Calculate statistics
    py_avg_time = sum(r.time_seconds for r in py_results) / len(py_results)
    py_avg_mem = sum(r.peak_memory_mb for r in py_results) / len(py_results)
    py_min_time = min(r.time_seconds for r in py_results)

    rust_avg_time = sum(r.time_seconds for r in rust_results) / len(rust_results)
    rust_avg_mem = sum(r.peak_memory_mb for r in rust_results) / len(rust_results)
    rust_min_time = min(r.time_seconds for r in rust_results)

    # Comparison
    print()
    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<25} {'Python':>12} {'Rust':>12} {'Diff':>12}")
    print("-" * 55)

    speedup_avg = py_avg_time / rust_avg_time if rust_avg_time > 0 else 0
    speedup_best = py_min_time / rust_min_time if rust_min_time > 0 else 0
    mem_diff = rust_avg_mem - py_avg_mem
    mem_pct = (mem_diff / py_avg_mem * 100) if py_avg_mem > 0 else 0

    print(
        f"  {'Avg Time (s)':<23} {py_avg_time:>12.3f} {rust_avg_time:>12.3f} {speedup_avg:>11.2f}x"
    )
    print(
        f"  {'Best Time (s)':<23} {py_min_time:>12.3f} {rust_min_time:>12.3f} {speedup_best:>11.2f}x"
    )
    print(
        f"  {'Peak RAM (MB)':<23} {py_avg_mem:>12.1f} {rust_avg_mem:>12.1f} {mem_diff:>+11.1f}"
    )
    print(
        f"  {'Trades':<23} {py_results[0].total_trades:>12} {rust_results[0].total_trades:>12} {'✅ match' if py_results[0].total_trades == rust_results[0].total_trades else '❌ diff':>12}"
    )

    # Verification
    print()
    print("=" * 60)
    print("Result Verification")
    print("=" * 60)

    trades_match = py_results[0].total_trades == rust_results[0].total_trades
    balance_tolerance = (
        max(py_results[0].final_balance, rust_results[0].final_balance) * 0.001
    )
    balance_match = (
        abs(py_results[0].final_balance - rust_results[0].final_balance)
        < balance_tolerance
    )

    print(
        f"  Trades match:        {'✅' if trades_match else '❌'} (Python: {py_results[0].total_trades}, Rust: {rust_results[0].total_trades})"
    )
    print(
        f"  Balance match (<0.1%): {'✅' if balance_match else '❌'} (Python: {py_results[0].final_balance:.2f}, Rust: {rust_results[0].final_balance:.2f})"
    )

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = trades_match and balance_match

    if all_pass:
        print(f"✅ All verifications PASSED")
        print(f"✅ Rust backend is {speedup_avg:.2f}x faster (avg)")
        print(f"✅ Rust backend is {speedup_best:.2f}x faster (best)")
        if mem_diff < 0:
            print(
                f"✅ Rust uses {abs(mem_diff):.1f}MB less RAM ({abs(mem_pct):.1f}% reduction)"
            )
        else:
            print(
                f"⚠️  Rust uses {mem_diff:.1f}MB more RAM ({mem_pct:.1f}% overhead - FFI costs)"
            )
    else:
        print("❌ Some verifications FAILED")

    return {
        "python": {
            "avg_time": py_avg_time,
            "best_time": py_min_time,
            "avg_memory_mb": py_avg_mem,
            "trades": py_results[0].total_trades,
            "balance": py_results[0].final_balance,
        },
        "rust": {
            "avg_time": rust_avg_time,
            "best_time": rust_min_time,
            "avg_memory_mb": rust_avg_mem,
            "trades": rust_results[0].total_trades,
            "balance": rust_results[0].final_balance,
        },
        "comparison": {
            "speedup_avg": speedup_avg,
            "speedup_best": speedup_best,
            "memory_diff_mb": mem_diff,
            "memory_diff_pct": mem_pct,
        },
        "verification": {
            "trades_match": trades_match,
            "balance_match": balance_match,
            "all_pass": all_pass,
        },
    }


if __name__ == "__main__":
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    results = run_benchmark(num_runs)
