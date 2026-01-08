from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from math import fsum
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel


def _measure(func) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


def _bench_slippage(n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    prices = 1.2 + rng.normal(0.0, 0.0004, size=n)
    directions = np.where(rng.random(n) > 0.5, "long", "short")
    model = SlippageModel(fixed_pips=0.05, random_pips=0.02)

    def _run():
        acc = 0.0
        for price, direction in zip(prices, directions):
            acc += model.apply(float(price), direction)
        return acc

    sec, peak = _measure(_run)
    return {
        "seconds": round(sec, 6),
        "peak_mb": round(peak, 6),
        "ops_per_sec": round(n / sec, 2),
    }


def _bench_fee(n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    volumes = rng.uniform(0.01, 5.0, size=n)
    prices = 1.2 + rng.normal(0.0, 0.0004, size=n)
    model = FeeModel(per_million=7.0, lot_size=100_000.0, min_fee=0.2)

    def _run():
        # fsum verhindert unnötige Optimierungen des Loops
        return fsum(
            model.calculate(float(v), float(p)) for v, p in zip(volumes, prices)
        )

    sec, peak = _measure(_run)
    return {
        "seconds": round(sec, 6),
        "peak_mb": round(peak, 6),
        "ops_per_sec": round(n / sec, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmarks für SlippageModel und FeeModel"
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=100_000, help="Anzahl Aufrufe je Modell"
    )
    parser.add_argument("-s", "--seed", type=int, default=321, help="RNG-Seed")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_slippage_and_fee.json"),
        help="Pfad zur JSON-Ausgabe",
    )
    args = parser.parse_args()

    slip = _bench_slippage(args.iterations, args.seed)
    fee = _bench_fee(args.iterations, args.seed + 1)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": int(args.iterations),
            "seed": int(args.seed),
        },
        "slippage_model": slip,
        "fee_model": fee,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved slippage/fee microbench to {args.output}")


if __name__ == "__main__":
    main()
