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

from backtest_engine.core.symbol_data_slicer import SymbolDataSlice


def _measure(func) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


def _make_multi(tf_list, bars: int, seed: int) -> Dict[str, Dict[str, list]]:
    rng = np.random.default_rng(seed)
    multi: Dict[str, Dict[str, list]] = {}
    for tf in tf_list:
        base = 1.20 + 0.0005 * len(tf)
        deltas = rng.normal(0.0, 0.0006, size=bars)
        bid = list(base + deltas)
        ask = list(base + 0.0002 + deltas)
        multi[tf] = {"bid": bid, "ask": ask}
    return multi


def _run_slicer(tf_list, bars: int, history_len: int, seed: int) -> Dict[str, float]:
    multi = _make_multi(tf_list, bars, seed)
    slicer = SymbolDataSlice(multi_candle_data=multi, index=0, indicator_cache=None)

    def _loop():
        acc = 0.0
        for idx in range(bars):
            slicer.set_index(idx)
            for tf in tf_list:
                latest_bid = slicer.latest(tf, "bid") or 0.0
                latest_ask = slicer.latest(tf, "ask") or 0.0
                acc += float(latest_bid) + float(latest_ask)
                hist = slicer.history(tf, "bid", length=history_len)
                if hist:
                    acc += float(hist[-1])
        return acc

    sec, peak = _measure(_loop)
    return {
        "seconds": round(sec, 6),
        "peak_mb": round(peak, 6),
        "ops_per_sec": round((bars * len(tf_list)) / sec, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SymbolDataSlice iteration/history cache"
    )
    parser.add_argument(
        "-b", "--bars", type=int, default=30_000, help="Anzahl Schritte"
    )
    parser.add_argument(
        "-t",
        "--timeframes",
        nargs="+",
        default=["M5", "M15", "H1"],
        help="Liste synthetischer Timeframes",
    )
    parser.add_argument(
        "-l", "--history-length", type=int, default=50, help="History-Länge für Cache"
    )
    parser.add_argument("-s", "--seed", type=int, default=2024, help="RNG-Seed")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_symbol_data_slicer.json"),
        help="Pfad zur JSON-Ausgabe",
    )
    args = parser.parse_args()

    stats = _run_slicer(args.timeframes, args.bars, args.history_length, args.seed)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bars": int(args.bars),
            "timeframes": list(args.timeframes),
            "history_length": int(args.history_length),
            "seed": int(args.seed),
        },
        "results": stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved SymbolDataSlice benchmark to {args.output}")


if __name__ == "__main__":
    main()
