from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.core.multi_symbol_slice import MultiSymbolSlice


def _measure(func) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


def _generate_lookup(
    symbols: int, timeframes: int, bars: int, seed: int
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    rng = np.random.default_rng(seed)
    tf_names = [f"TF{idx+1}" for idx in range(timeframes)]
    lookup: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for s in range(symbols):
        sym = f"SYM{s+1:03d}"
        sym_map: Dict[str, Dict[int, Dict[str, float]]] = {"bid": {}, "ask": {}}
        base = 1.10 + 0.01 * s
        deltas = rng.normal(0.0, 0.0005, size=bars)
        for price_type in ("bid", "ask"):
            price = base + (0.0002 if price_type == "ask" else 0.0)
            sym_map[price_type] = {
                idx: {
                    "price": float(price + delta),
                    "tf": tf_names[idx % len(tf_names)],
                }
                for idx, delta in enumerate(deltas)
            }
        lookup[sym] = sym_map
    return lookup


def _run_lookup(
    symbols: int, timeframes: int, bars: int, seed: int
) -> Dict[str, float]:
    lookup = _generate_lookup(symbols, timeframes, bars, seed)
    slice_obj = MultiSymbolSlice(candle_lookups=lookup, timestamp=0, primary_tf="TF1")

    def _loop():
        acc = 0.0
        for idx in range(bars):
            slice_obj.set_timestamp(idx)
            for sym in slice_obj:
                bid = slice_obj.get(sym, "bid") or {}
                ask = slice_obj[sym].latest(price_type="ask") or {}
                acc += float(bid.get("price", 0.0))
                acc += float(ask.get("price", 0.0))
        return acc

    sec, peak = _measure(_loop)
    return {
        "seconds": round(sec, 6),
        "peak_mb": round(peak, 6),
        "ops_per_sec": round(symbols * bars / sec, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MultiSymbolSlice get/set operations"
    )
    parser.add_argument("-s", "--symbols", type=int, default=5, help="Anzahl Symbole")
    parser.add_argument(
        "-t",
        "--timeframes",
        type=int,
        default=3,
        help="Anzahl Timeframes (synthetisch)",
    )
    parser.add_argument(
        "-b", "--bars", type=int, default=50_000, help="Anzahl Schritte/Timestamps"
    )
    parser.add_argument("-r", "--seed", type=int, default=777, help="Seed für RNG")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_multi_symbol_slice.json"),
        help="Pfad zur JSON-Ausgabe",
    )
    args = parser.parse_args()

    stats = _run_lookup(args.symbols, args.timeframes, args.bars, args.seed)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbols": int(args.symbols),
            "timeframes": int(args.timeframes),
            "bars": int(args.bars),
            "seed": int(args.seed),
        },
        "results": stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved multi-symbol slice benchmark to {args.output}")


if __name__ == "__main__":
    main()
