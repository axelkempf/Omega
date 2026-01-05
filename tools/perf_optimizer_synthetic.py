from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.optimizer.final_param_selector import run_final_parameter_selection
from backtest_engine.optimizer.robust_zone_analyzer import run_robust_zone_analysis


def _measure(func) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


def _build_fixture(
    root: Path,
    param_grid: Dict[str, Dict[str, Any]],
    rows: int,
    windows: int,
    seed: int,
) -> Path:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    records = []
    for w in range(windows):
        for _ in range(rows):
            row: Dict[str, Any] = {"window_id": w}
            for name, spec in param_grid.items():
                if spec["type"] == "int":
                    row[name] = int(rng.integers(spec["low"], spec["high"] + 1))
                elif spec["type"] == "float":
                    row[name] = float(rng.uniform(spec["low"], spec["high"]))
                elif spec["type"] == "categorical":
                    choices = spec.get("choices", []) or ["a"]
                    row[name] = str(choices[int(rng.integers(0, len(choices)))])
            net_profit = float(rng.uniform(800.0, 2000.0))
            drawdown = float(rng.uniform(200.0, 700.0))
            commission = float(net_profit * rng.uniform(0.05, 0.25))
            row.update(
                {
                    "Net Profit": net_profit,
                    "Drawdown": drawdown,
                    "Commission": commission,
                    "Sharpe (trade)": float(rng.uniform(0.5, 2.0)),
                    "Avg R-Multiple": float(rng.uniform(0.5, 1.5)),
                    "Winrate (%)": float(rng.uniform(45.0, 70.0)),
                    "total_trades": int(rng.integers(20, 80)),
                }
            )
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(root / "all_top_out_of_sample.csv", index=False)

    manifest = {
        "walkforward_options": {
            "min_trades": 10,
            "train_days": 90,
            "test_days": 30,
        }
    }
    (root / "walkforward_run_config.json").write_text(json.dumps(manifest, indent=2))

    # Minimales Template für Metadaten-Referenz
    config_template = root / "config_template.json"
    config_template.write_text(json.dumps({"strategy": {"parameters": {}}}, indent=2))
    return config_template


def benchmark_optimizer(rows: int, windows: int, seed: int, output: Path) -> None:
    fixture_root = output.parent / "p0-01_optimizer_fixture"
    param_grid = {
        "ema_fast": {"type": "int", "low": 5, "high": 20, "step": 1},
        "ema_slow": {"type": "int", "low": 21, "high": 80, "step": 1},
    }
    config_template = _build_fixture(fixture_root, param_grid, rows, windows, seed)

    base_config = {
        "start_date": "2020-01-01",
        "end_date": "2020-03-31",
        "timeframes": {"primary": "M15"},
        "reporting": {"dev_mode": True, "dev_seed": 42},
    }

    robust_sec, robust_peak = _measure(
        lambda: run_robust_zone_analysis(
            walkforward_root=str(fixture_root),
            param_grid=param_grid,
            analyze_alpha=0.10,
            analyze_min_coverage=0.10,
            analyze_min_sharpe_trade=0.30,
            enable_plots=False,
        )
    )

    final_sec, final_peak = _measure(
        lambda: run_final_parameter_selection(
            walkforward_root=str(fixture_root),
            base_config=base_config,
            config_template_path=str(config_template),
            param_grid=param_grid,
            search_mode="grid",
            preload_mode="window",
            jitter_frac=0.02,
            jitter_repeats=2,
            max_grid_candidates=100,
            n_jobs=1,
        )
    )

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows": int(rows),
            "windows": int(windows),
            "seed": int(seed),
            "fixture_root": str(fixture_root),
        },
        "robust_zone_analysis": {
            "seconds": round(robust_sec, 6),
            "peak_mb": round(robust_peak, 6),
        },
        "final_param_selector": {
            "seconds": round(final_sec, 6),
            "peak_mb": round(final_peak, 6),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved optimizer baseline to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic baseline for robust_zone_analyzer + final_param_selector"
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=200,
        help="Zeilen pro Window im Synthetic-OOS-CSV",
    )
    parser.add_argument("-w", "--windows", type=int, default=5, help="Anzahl Windows")
    parser.add_argument("-s", "--seed", type=int, default=99, help="RNG-Seed")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_optimizer.json"),
        help="Pfad zur JSON-Ausgabe",
    )
    args = parser.parse_args()
    benchmark_optimizer(args.rows, args.windows, args.seed, args.output)


if __name__ == "__main__":
    main()
