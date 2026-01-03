from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import tracemalloc

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.optimizer import walkforward as wf


def _measure(func) -> Tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak / 1_000_000.0


class _DummyPortfolio:
    def __init__(self) -> None:
        self.trades = []

    def trades_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


def _patch_walkforward() -> None:
    def fake_backtest(config: Dict[str, Any], preloaded_data=None):
        return _DummyPortfolio(), {"trades": []}

    class _FakeTrial:
        def __init__(self, number: int, params: Dict[str, Any]) -> None:
            self.number = number
            self.params = params
            self.values = [1.0, 0.5, 60.0, 100.0]
            self.value = 1.0
            self.user_attrs = {"params": params, "total_trades": 20}
            self.state = type("State", (), {"name": "COMPLETE"})()
            self.duration = timedelta(seconds=0.05)

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials = [
                _FakeTrial(0, {"ema_fast": 10, "ema_slow": 30}),
                _FakeTrial(1, {"ema_fast": 12, "ema_slow": 33}),
            ]
            self.directions = ["maximize", "maximize", "maximize", "minimize"]

    def fake_optimize(*args, **kwargs):
        return _FakeStudy()

    def fake_metrics(portfolio: Any) -> Dict[str, Any]:
        return {
            "net_profit_after_fees_eur": 1000.0,
            "fees_total_eur": 50.0,
            "avg_r_multiple": 1.1,
            "winrate_percent": 55.0,
            "drawdown_eur": 200.0,
            "drawdown_percent": 5.0,
            "total_trades": 40,
            "active_days": 20,
            "sharpe_trade": 1.2,
            "sortino_trade": 1.5,
        }

    def fake_rating(summary: Dict[str, Any], thresholds=None) -> Dict[str, Any]:
        return {"Score": 0.8}

    wf.run_backtest_and_return_portfolio = fake_backtest  # type: ignore[attr-defined]
    wf.optimize_strategy_with_optuna_pareto = fake_optimize  # type: ignore[attr-defined]
    wf.calculate_metrics = fake_metrics  # type: ignore[attr-defined]
    wf.rate_strategy_performance = fake_rating  # type: ignore[attr-defined]


def benchmark_walkforward(output: Path, seed: int, windows: int) -> None:
    _patch_walkforward()

    root = output.parent / "p0-01_walkforward_fixture"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    window_info = {
        "window_id": 0,
        "train_start": datetime(2020, 1, 1),
        "train_end": datetime(2020, 1, 15),
        "test_start": datetime(2020, 1, 16),
        "test_end": datetime(2020, 1, 31),
    }
    base_config = {
        "start_date": "2020-01-01",
        "end_date": "2020-03-01",
        "strategy": {"parameters": {}},
        "timeframes": {"primary": "M15"},
        "reporting": {"dev_mode": True, "dev_seed": seed},
    }
    param_grid = {
        "ema_fast": {"type": "int", "low": 5, "high": 20, "step": 1},
        "ema_slow": {"type": "int", "low": 21, "high": 50, "step": 1},
    }
    rating_thresholds = {"profit": 0, "winrate": 0, "drawdown": 1_000}

    def _run():
        wf.run_walkforward_window(
            window_info=window_info,
            base_config=base_config,
            param_grid=param_grid,
            n_trials=5,
            rating_thresholds=rating_thresholds,
            walkforward_root=str(root),
            data_preload={},
            min_trades=5,
            min_days_active=1,
            kfold_splits=2,
            robustness_jitter_frac=0.01,
            robustness_repeats=1,
            preload_mode="window",
            export_artifacts=True,
        )

    sec, peak = _measure(_run)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": int(seed),
            "windows": int(windows),
            "fixture_root": str(root),
        },
        "walkforward_window": {"seconds": round(sec, 6), "peak_mb": round(peak, 6)},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved walkforward baseline to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stubbed walkforward baseline (synthetic, monkeypatched)")
    parser.add_argument("-o", "--output", type=Path, default=Path("reports/performance_baselines/p0-01_walkforward_stub.json"))
    parser.add_argument("-s", "--seed", type=int, default=4242, help="Seed für Dev-Mode")
    parser.add_argument("-w", "--windows", type=int, default=1, help="Anzahl Windows (synthetisch)")
    args = parser.parse_args()
    benchmark_walkforward(args.output, args.seed, args.windows)


if __name__ == "__main__":
    main()
