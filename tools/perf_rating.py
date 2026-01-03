from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import cProfile
import io
import pstats
import tracemalloc

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.rating.cost_shock_score import (
    COST_SHOCK_FACTORS,
    compute_multi_factor_cost_shock_score,
)
from backtest_engine.rating.data_jitter_score import compute_data_jitter_score
from backtest_engine.rating.p_values import compute_p_values
from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import (
    compute_stability_score_from_yearly_profits,
)
from backtest_engine.rating.strategy_rating import rate_strategy_performance
from backtest_engine.rating.timing_jitter_score import compute_timing_jitter_score
from backtest_engine.rating.tp_sl_stress_score import compute_tp_sl_stress_score
from backtest_engine.rating.trade_dropout_score import (
    compute_multi_run_trade_dropout_score,
    simulate_trade_dropout_metrics_multi,
)
from backtest_engine.rating.ulcer_index_score import compute_ulcer_index_and_score


def _gen_equity(n: int) -> list[tuple[datetime, float]]:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(123)
    steps = rng.normal(loc=5.0, scale=20.0, size=n)
    equity = np.cumsum(steps) + 10_000.0
    ts = [start + timedelta(days=i) for i in range(n)]
    return list(zip(ts, equity.astype(float)))


def _make_trades_df(n: int = 200) -> pd.DataFrame:
    base_time = pd.Timestamp("2020-01-01T00:00:00Z")
    times = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n)]
    directions = np.where(np.arange(n) % 2 == 0, "long", "short")
    rng = np.random.default_rng(999)
    tp = 1.2000 + rng.normal(0.0004, 0.00005, size=n)
    sl = 1.2000 - rng.normal(0.0004, 0.00005, size=n)
    pnl = rng.normal(15.0, 25.0, size=n)
    fees = rng.normal(0.5, 0.1, size=n)
    r_mult = rng.normal(0.8, 0.5, size=n)

    return pd.DataFrame(
        {
            "reason": "take_profit",
            "direction": directions,
            "take_profit": tp,
            "stop_loss": sl,
            "entry_time": times,
            "exit_time": [t + pd.Timedelta(minutes=30) for t in times],
            "meta": [{"prices": {"spread": 0.0002}} for _ in range(n)],
            "result": pnl,
            "total_fee": fees,
            "r_multiple": r_mult,
        }
    )


def _make_tp_sl_arrays(n: int = 400) -> Dict[str, np.ndarray]:
    base_time = pd.Timestamp("2020-01-01T00:00:00Z")
    times = np.array([(base_time + pd.Timedelta(minutes=i)).value for i in range(n)], dtype=np.int64)
    bid_low = np.linspace(1.1990, 1.2050, n)
    bid_high = bid_low + 0.0008
    ask_low = bid_low + 0.0002
    ask_high = ask_low + 0.0008
    return {
        "times_ns": times,
        "bid_high": bid_high,
        "bid_low": bid_low,
        "ask_high": ask_high,
        "ask_low": ask_low,
    }


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
    stats.print_stats(15)
    return stream.getvalue()


def benchmark_rating(n_equity: int, n_jitter: int) -> Dict[str, Any]:
    equity_curve = _gen_equity(n_equity)
    base_metrics = {
        "profit": 12_000.0,
        "avg_r": 0.9,
        "winrate": 0.55,
        "drawdown": 1_200.0,
        "sharpe": 1.4,
    }
    jitter_metrics = []
    rng = np.random.default_rng(7)
    for _ in range(n_jitter):
        jitter_metrics.append(
            {
                "profit": float(12_000.0 * rng.uniform(0.7, 1.1)),
                "avg_r": float(0.9 * rng.uniform(0.7, 1.1)),
                "winrate": float(0.55 * rng.uniform(0.7, 1.1)),
                "drawdown": float(1_200.0 * rng.uniform(0.8, 1.2)),
                "sharpe": float(1.4 * rng.uniform(0.6, 1.1)),
            }
        )

    summary = {
        "Winrate (%)": 55,
        "Avg R-Multiple": 0.95,
        "Net Profit": 1500,
        "profit_factor": 1.4,
        "drawdown_eur": 800,
    }

    trades_df = _make_trades_df()
    tp_sl_arrays = _make_tp_sl_arrays()

    shocked_metrics_list = [
        {
            "profit": base_metrics["profit"] * f,
            "drawdown": base_metrics["drawdown"] * (1.0 + 0.1 * idx),
            "sharpe": base_metrics["sharpe"] * (1.0 - 0.05 * idx),
        }
        for idx, f in enumerate(COST_SHOCK_FACTORS)
    ]

    dropout_metrics_list = simulate_trade_dropout_metrics_multi(
        trades_df,
        dropout_frac=0.2,
        base_metrics=base_metrics,
        n_runs=3,
        seed=77,
    )

    ops = {
        "ulcer_index": lambda: compute_ulcer_index_and_score(equity_curve, ulcer_cap=10.0),
        "robustness_1": lambda: compute_robustness_score_1(base_metrics, jitter_metrics),
        "strategy_rating": lambda: rate_strategy_performance(summary),
        "data_jitter": lambda: compute_data_jitter_score(base_metrics, jitter_metrics),
        "timing_jitter": lambda: compute_timing_jitter_score(base_metrics, jitter_metrics),
        "cost_shock": lambda: compute_multi_factor_cost_shock_score(
            base_metrics, shocked_metrics_list
        ),
        "trade_dropout": lambda: compute_multi_run_trade_dropout_score(
            base_metrics, dropout_metrics_list
        ),
        "tp_sl_stress": lambda: compute_tp_sl_stress_score(trades_df, tp_sl_arrays),
        "stability": lambda: compute_stability_score_from_yearly_profits(
            {2020: 4_000.0, 2021: 3_000.0, 2022: 5_000.0, 2023: 4_200.0}
        ),
        "p_values": lambda: compute_p_values(trades_df, n_boot=200),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, fn in ops.items():
        dur, peak = _measure(fn)
        results[name] = {"seconds": round(dur, 6), "peak_mb": round(peak, 6)}

    profile_summary = _profile(lambda: [fn() for fn in ops.values() for _ in range(50)])

    return {
        "meta": {
            "n_equity_points": n_equity,
            "n_jitter": n_jitter,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "operations": results,
        "profile_top15": profile_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance baseline for rating functions (synthetic data)")
    parser.add_argument("-e", "--n-equity", type=int, default=365, help="Number of equity points")
    parser.add_argument("-j", "--n-jitter", type=int, default=100, help="Number of jitter metric samples")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/performance_baselines/p0-01_rating.json"),
        help="Path to JSON output",
    )
    args = parser.parse_args()

    results = benchmark_rating(args.n_equity, args.n_jitter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved performance baseline to {args.output}")


if __name__ == "__main__":
    main()
