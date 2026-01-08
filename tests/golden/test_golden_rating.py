"""Golden-File Tests für Determinismus der Rating-Module (P6-10).

Ziel:
- Sicherstellen, dass die Rating-/Robustness-Module (inkl. seed-basierter Simulationen)
  bei fixierten Inputs reproduzierbare Outputs liefern.

Hinweis:
- Wir speichern bewusst nur Scalar-Outputs und Hashes abgeleiteter Artefakte
  (z.B. DataFrame-Hashes) als Golden-Reference, um die Files klein und stabil zu halten.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import pytest

from backtest_engine.rating.cost_shock_score import (
    apply_cost_shock_inplace,
    compute_multi_factor_cost_shock_score,
)
from backtest_engine.rating.data_jitter_score import (
    _stable_data_jitter_seed,
    build_jittered_preloaded_data,
    compute_data_jitter_score,
    precompute_atr_cache,
)
from backtest_engine.rating.p_values import compute_p_values
from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import (
    compute_stability_score_and_wmape_from_yearly_profits,
)
from backtest_engine.rating.stress_penalty import (
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
)
from backtest_engine.rating.timing_jitter_score import (
    apply_timing_jitter_month_shift_inplace,
    compute_timing_jitter_score,
    get_timing_jitter_backward_shift_months,
)
from backtest_engine.rating.tp_sl_stress_score import compute_tp_sl_stress_score
from backtest_engine.rating.trade_dropout_score import (
    compute_multi_run_trade_dropout_score,
    compute_trade_dropout_score,
    simulate_trade_dropout_metrics,
    simulate_trade_dropout_metrics_multi,
)
from backtest_engine.rating.ulcer_index_score import compute_ulcer_index_and_score
from tests.golden.conftest import (
    GoldenRatingResult,
    assert_golden_match,
    compute_dataframe_hash,
    compute_dict_hash,
    create_metadata,
)


def _rate_strategy_performance(
    summary: Dict[str, Any], thresholds: Dict[str, float] | None = None
) -> Dict[str, Any]:
    """
    Bewertet die Strategie-Performance anhand fester Schwellenwerte.
    Gibt Score (0–1), Deployment-Entscheidung und Fehlerschlüssel zurück.

    Note: Diese Funktion war ursprünglich in strategy_rating.py und wurde
    für die Wave-1 Rust/Julia Migration vorbereitet. Die Funktion wurde
    inline hier verschoben, da strategy_rating.py vollständig entfernt wurde.
    """
    thresholds = thresholds or {
        "min_winrate": 45,
        "min_avg_r": 0.6,
        "min_profit": 500,
        "min_profit_factor": 1.2,
        "max_drawdown": 1000,
    }
    deployment = True
    checks: list[str] = []

    if summary.get("Winrate (%)", 0) < thresholds["min_winrate"]:
        deployment = False
        checks.append("Winrate")
    if summary.get("Avg R-Multiple", 0) < thresholds["min_avg_r"]:
        deployment = False
        checks.append("Avg R")
    if summary.get("Net Profit", 0) < thresholds["min_profit"]:
        deployment = False
        checks.append("Net Profit")
    if summary.get("profit_factor", 0) < thresholds["min_profit_factor"]:
        deployment = False
        checks.append("Profit Factor")
    if summary.get("drawdown_eur", 0) > thresholds["max_drawdown"]:
        deployment = False
        checks.append("Drawdown")

    score = 1 - len(checks) / 5
    return {
        "Score": round(score, 2),
        "Deployment": deployment,
        "Deployment_Fails": "|".join(checks),
    }


def _synthetic_ohlc_df(
    *, n: int, start: str = "2024-01-01", freq: str = "15min"
) -> pd.DataFrame:
    rng = np.random.default_rng(12345)

    # Kleine, aber deterministische Preisbewegung.
    returns = rng.normal(0.0, 0.001, int(n))
    close = 1.1000 * np.cumprod(1.0 + returns)

    high = close * (1.0 + np.abs(rng.normal(0.0, 0.0005, int(n))))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.0005, int(n))))

    open_ = np.roll(close, 1)
    open_[0] = 1.1000

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    idx = pd.date_range(start, periods=int(n), freq=freq, tz="UTC")

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100, 10000, int(n)),
        },
        index=idx,
    )


def _synthetic_trades_df(*, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(222)
    times = pd.date_range("2024-01-01", periods=int(n), freq="1D", tz="UTC")

    # Gewinne/Verluste und R-Multiples (inkl. Fees) – bewusst klein.
    result = rng.normal(10.0, 30.0, int(n))
    total_fee = np.full(int(n), 0.15)

    r_multiple = rng.normal(0.2, 0.8, int(n))

    return pd.DataFrame(
        {
            "exit_time": times,
            "result": result,
            "total_fee": total_fee,
            "r_multiple": r_multiple,
        }
    )


@pytest.mark.usefixtures("set_seed")
class TestGoldenRatingDeterminism:
    def test_rating_modules_golden(
        self, golden_manager, regenerate_golden_files, deterministic_seed
    ):
        seed = int(deterministic_seed)

        base_metrics: Dict[str, float] = {
            "profit": 10_000.0,
            "avg_r": 0.80,
            "winrate": 0.55,
            "drawdown": 2_000.0,
            "sharpe": 1.50,
        }

        jitter_metrics: list[dict[str, float]] = [
            {"profit": 9_200.0, "avg_r": 0.72, "winrate": 0.52, "drawdown": 2_200.0},
            {"profit": 9_700.0, "avg_r": 0.78, "winrate": 0.54, "drawdown": 2_050.0},
            {"profit": 8_800.0, "avg_r": 0.70, "winrate": 0.50, "drawdown": 2_350.0},
        ]

        stress_metrics: list[dict[str, float]] = [
            {"profit": 8_000.0, "drawdown": 2_500.0, "sharpe": 1.10},
            {"profit": 6_500.0, "drawdown": 3_200.0, "sharpe": 0.90},
        ]

        robustness_score = float(
            compute_robustness_score_1(base_metrics, jitter_metrics, penalty_cap=0.5)
        )

        stress_penalty = float(
            compute_penalty_profit_drawdown_sharpe(
                base_metrics,
                stress_metrics,
                penalty_cap=0.5,
            )
        )
        stress_score = float(score_from_penalty(stress_penalty, penalty_cap=0.5))

        # Cost-shock: reine Config-Mutation + Aggregation über deterministische Inputs.
        cfg = {"execution": {"slippage_multiplier": 1.0, "fee_multiplier": 1.0}}
        shocked_cfg = deepcopy(cfg)
        apply_cost_shock_inplace(shocked_cfg, factor=1.5)

        shocked_metrics_list = [
            {"profit": 9_000.0, "drawdown": 2_200.0, "sharpe": 1.35},
            {"profit": 8_500.0, "drawdown": 2_450.0, "sharpe": 1.20},
            {"profit": 8_000.0, "drawdown": 2_700.0, "sharpe": 1.10},
        ]
        cost_shock_score = float(
            compute_multi_factor_cost_shock_score(
                base_metrics,
                shocked_metrics_list,
                penalty_cap=0.5,
            )
        )

        # Timing jitter: deterministische Shift-Ermittlung + in-place Shift.
        timing_cfg: Dict[str, Any] = {
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
        }
        shifts = get_timing_jitter_backward_shift_months(
            start_date=timing_cfg["start_date"],
            end_date=timing_cfg["end_date"],
        )
        shifted_cfg = deepcopy(timing_cfg)
        apply_timing_jitter_month_shift_inplace(
            shifted_cfg, shift_months_backward=shifts[0]
        )

        timing_score = float(
            compute_timing_jitter_score(
                base_metrics,
                [
                    {"profit": 8_800.0, "drawdown": 2_250.0, "sharpe": 1.25},
                    {"profit": 8_400.0, "drawdown": 2_400.0, "sharpe": 1.20},
                ],
                penalty_cap=0.5,
            )
        )

        # Stability score: deterministisch aus Year-Profit Map.
        stability_score, wmape = compute_stability_score_and_wmape_from_yearly_profits(
            {2021: 2000.0, 2022: 3000.0, 2023: 5000.0},
        )

        # Ulcer Index: timestamped path (pandas resample) + deterministisch.
        equity_times = pd.date_range("2024-01-01", periods=90, freq="1D", tz="UTC")
        equity_vals = 100_000.0 + np.cumsum(np.sin(np.linspace(0, 7, 90)) * 100.0)
        ulcer_index, ulcer_score = compute_ulcer_index_and_score(
            list(zip(equity_times.to_pydatetime().tolist(), equity_vals.tolist())),
            ulcer_cap=10.0,
        )

        # p-values: bewusst kleines n_boot für schnelle Golden-Tests.
        trades_df = _synthetic_trades_df(n=40)
        pvals = compute_p_values(trades_df, n_boot=300, seed_r=123, seed_pnl=456)

        # Strategy rating: threshold-based und deterministisch.
        rating = _rate_strategy_performance(
            {
                "Winrate (%)": 52,
                "Avg R-Multiple": 0.75,
                "Net Profit": 1200,
                "profit_factor": 1.35,
                "drawdown_eur": 800,
            }
        )

        # Trade dropout: seed-basierte Simulation.
        dropout_metrics = simulate_trade_dropout_metrics(
            trades_df,
            dropout_frac=0.25,
            base_metrics={
                "profit": base_metrics["profit"],
                "drawdown": base_metrics["drawdown"],
                "sharpe": base_metrics["sharpe"],
            },
            seed=777,
        )
        dropout_score = float(
            compute_trade_dropout_score(base_metrics, dropout_metrics)
        )

        dropout_multi = simulate_trade_dropout_metrics_multi(
            trades_df,
            dropout_frac=0.25,
            base_metrics={
                "profit": base_metrics["profit"],
                "drawdown": base_metrics["drawdown"],
                "sharpe": base_metrics["sharpe"],
            },
            n_runs=3,
            seed=777,
        )
        dropout_multi_score = float(
            compute_multi_run_trade_dropout_score(base_metrics, dropout_multi)
        )

        # TP/SL Stress: arrays + minimal Trades.
        ts = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        times_ns = ts.astype("int64").to_numpy()

        bid_high = np.array(
            [
                1.1005,
                1.1010,
                1.1012,
                1.1013,
                1.1014,
                1.1016,
                1.1018,
                1.1025,  # TP hit late
                1.1026,
                1.1027,
            ],
            dtype=float,
        )
        bid_low = np.array([1.0990] * 10, dtype=float)
        ask_high = bid_high + 0.0002
        ask_low = bid_low + 0.0002

        arrays = {
            "times_ns": times_ns,
            "bid_high": bid_high,
            "bid_low": bid_low,
            "ask_high": ask_high,
            "ask_low": ask_low,
        }

        tp_sl_trades = pd.DataFrame(
            [
                {
                    "reason": "take_profit",
                    "direction": "long",
                    "take_profit": 1.1020,
                    "stop_loss": 1.0980,
                    "entry_time": ts[2],
                    "exit_time": ts[4],
                    "meta": {"prices": {"spread": 0.0002}},
                }
            ]
        )
        tp_sl_score = float(
            compute_tp_sl_stress_score(tp_sl_trades, arrays, debug=False)
        )

        # Data jitter: base_preloaded_data + ATR cache + jittered hashes.
        base_df = _synthetic_ohlc_df(n=64)
        base_preloaded_data: Mapping[Tuple[str, str], pd.DataFrame] = {
            ("M15", "bid"): base_df,
            ("M15", "ask"): base_df.copy(),
        }
        atr_cache = precompute_atr_cache(base_preloaded_data, period=14)

        stable_seeds = [int(_stable_data_jitter_seed(seed, i)) for i in range(3)]
        jitter_seed = int(stable_seeds[1])
        jittered = build_jittered_preloaded_data(
            base_preloaded_data,
            atr_cache=atr_cache,
            sigma_atr=0.10,
            seed=jitter_seed,
            fraq=0.10,
        )
        jitter_bid_hash = compute_dataframe_hash(
            jittered[("M15", "bid")], columns=["Open", "High", "Low", "Close"]
        )
        jitter_ask_hash = compute_dataframe_hash(
            jittered[("M15", "ask")], columns=["Open", "High", "Low", "Close"]
        )
        atr_hash = compute_dict_hash(
            {"atr_M15": atr_cache["M15"].round(8).head(32).tolist()}
        )

        data_jitter_score = float(
            compute_data_jitter_score(base_metrics, jitter_metrics, penalty_cap=0.5)
        )

        outputs: Dict[str, Any] = {
            "robustness_score_1": robustness_score,
            "stress_penalty": stress_penalty,
            "stress_score": stress_score,
            "cost_shock": {
                "cfg_execution": shocked_cfg.get("execution"),
                "score_multi": cost_shock_score,
            },
            "timing_jitter": {
                "shifts_months": shifts,
                "shifted_cfg": shifted_cfg,
                "score": timing_score,
            },
            "stability": {"score": float(stability_score), "wmape": float(wmape)},
            "ulcer": {
                "ulcer_index": float(ulcer_index),
                "ulcer_score": float(ulcer_score),
            },
            "p_values": {k: float(v) for k, v in dict(pvals).items()},
            "strategy_rating": rating,
            "trade_dropout": {
                "one": {k: float(v) for k, v in dropout_metrics.items()},
                "one_score": dropout_score,
                "multi": [{k: float(v) for k, v in m.items()} for m in dropout_multi],
                "multi_score": dropout_multi_score,
            },
            "tp_sl_stress": tp_sl_score,
            "data_jitter": {
                "stable_seeds": stable_seeds,
                "seed_used": jitter_seed,
                "atr_head_hash": atr_hash,
                "jitter_bid_hash": jitter_bid_hash,
                "jitter_ask_hash": jitter_ask_hash,
                "score": data_jitter_score,
            },
        }

        current = GoldenRatingResult(
            metadata=create_metadata(seed, "P6-10 rating module determinism"),
            outputs=outputs,
            outputs_hash=compute_dict_hash(outputs),
        )

        assert_golden_match(
            golden_manager,
            "rating_modules_v1",
            current,
            regenerate=bool(regenerate_golden_files),
        )
