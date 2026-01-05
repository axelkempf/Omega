# -*- coding: utf-8 -*-
"""
Benchmark Suite für Rating-Module (P3-04).

Testet Score-Berechnungen aus backtest_engine.rating:
- robustness_score_1: Parameter Jitter Robustheit
- cost_shock_score: Execution Cost Stress
- trade_dropout_score: Trade Dropout Simulation
- stability_score: Yearly Profit Stability
- stress_penalty: Basis-Penalty-Berechnung
- timing_jitter_score: Timing Jitter Robustheit
- tp_sl_stress_score: TP/SL Stress Testing

Verwendung:
    pytest tests/benchmarks/test_bench_rating.py -v
    pytest tests/benchmarks/test_bench_rating.py --benchmark-json=output.json
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from backtest_engine.rating.cost_shock_score import (
    compute_cost_shock_score,
    compute_multi_factor_cost_shock_score,
)
from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import (
    compute_stability_score_and_wmape_from_yearly_profits,
    compute_stability_score_from_yearly_profits,
)
from backtest_engine.rating.stress_penalty import (
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
)
from backtest_engine.rating.trade_dropout_score import (
    simulate_trade_dropout_metrics,
)

from .conftest import (
    BENCHMARK_SEED,
    generate_base_metrics,
    generate_synthetic_trades_df,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Synthetic Metrics Generators
# ══════════════════════════════════════════════════════════════════════════════


def generate_jitter_metrics(
    base_metrics: Dict[str, float],
    n_repeats: int = 10,
    *,
    seed: int = BENCHMARK_SEED,
    noise_factor: float = 0.2,
) -> List[Dict[str, float]]:
    """Generiert Jitter-Metrics für Robustness-Tests."""
    rng = np.random.default_rng(seed)
    jitter_metrics = []

    for _ in range(n_repeats):
        jittered = {}
        for key, value in base_metrics.items():
            # Zufällige Variation um ±noise_factor
            noise = rng.uniform(-noise_factor, noise_factor)
            if key == "drawdown":
                # Drawdown kann auch steigen
                jittered[key] = max(0.0, value * (1 + noise))
            else:
                jittered[key] = value * (1 + noise)
        jitter_metrics.append(jittered)

    return jitter_metrics


def generate_yearly_profits(
    n_years: int = 5,
    *,
    seed: int = BENCHMARK_SEED,
    avg_profit: float = 2000.0,
    std_profit: float = 800.0,
) -> Dict[int, float]:
    """Generiert Yearly Profits für Stability-Score."""
    rng = np.random.default_rng(seed)
    base_year = 2020
    return {
        base_year + i: float(rng.normal(avg_profit, std_profit))
        for i in range(n_years)
    }


def generate_yearly_durations(
    profits_by_year: Dict[int, float],
    *,
    seed: int = BENCHMARK_SEED,
) -> Dict[int, float]:
    """Generiert Durations passend zu Yearly Profits."""
    rng = np.random.default_rng(seed)
    durations = {}
    for year in profits_by_year.keys():
        # Zwischen 200 und 365 Trading-Tagen
        durations[year] = float(rng.integers(200, 366))
    return durations


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Robustness Score 1 (Parameter Jitter)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestRobustnessScore1Benchmarks:
    """Benchmarks für compute_robustness_score_1."""

    def test_robustness_score_10_repeats(self, benchmark: Any) -> None:
        """Benchmark: Robustness Score mit 10 Jitter-Repeats."""
        base = generate_base_metrics()
        jitter = generate_jitter_metrics(base, n_repeats=10)

        def compute() -> float:
            return compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_robustness_score_50_repeats(self, benchmark: Any) -> None:
        """Benchmark: Robustness Score mit 50 Jitter-Repeats."""
        base = generate_base_metrics()
        jitter = generate_jitter_metrics(base, n_repeats=50)

        def compute() -> float:
            return compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_robustness_score_100_repeats(self, benchmark: Any) -> None:
        """Benchmark: Robustness Score mit 100 Jitter-Repeats."""
        base = generate_base_metrics()
        jitter = generate_jitter_metrics(base, n_repeats=100)

        def compute() -> float:
            return compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Cost Shock Score
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestCostShockScoreBenchmarks:
    """Benchmarks für Cost Shock Score Berechnungen."""

    def test_single_cost_shock_score(self, benchmark: Any) -> None:
        """Benchmark: Einzelner Cost Shock Score."""
        base = generate_base_metrics()
        shocked = {
            "profit": base["profit"] * 0.7,
            "drawdown": base["drawdown"] * 1.3,
            "sharpe": base["sharpe"] * 0.8,
        }

        def compute() -> float:
            return compute_cost_shock_score(base, shocked, penalty_cap=0.5)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_multi_factor_cost_shock_3_factors(self, benchmark: Any) -> None:
        """Benchmark: Multi-Factor Cost Shock (3 Faktoren)."""
        base = generate_base_metrics()
        factors = [1.25, 1.50, 2.00]
        shocked_list = [
            {
                "profit": base["profit"] / f,
                "drawdown": base["drawdown"] * f,
                "sharpe": base["sharpe"] / f,
            }
            for f in factors
        ]

        def compute() -> float:
            return compute_multi_factor_cost_shock_score(
                base, shocked_list, penalty_cap=0.5
            )

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_multi_factor_cost_shock_5_factors(self, benchmark: Any) -> None:
        """Benchmark: Multi-Factor Cost Shock (5 Faktoren)."""
        base = generate_base_metrics()
        factors = [1.1, 1.25, 1.5, 1.75, 2.0]
        shocked_list = [
            {
                "profit": base["profit"] / f,
                "drawdown": base["drawdown"] * f,
                "sharpe": base["sharpe"] / f,
            }
            for f in factors
        ]

        def compute() -> float:
            return compute_multi_factor_cost_shock_score(
                base, shocked_list, penalty_cap=0.5
            )

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Trade Dropout Score
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestTradeDropoutScoreBenchmarks:
    """Benchmarks für Trade Dropout Simulation."""

    def test_trade_dropout_100_trades_10pct(self, benchmark: Any) -> None:
        """Benchmark: Trade Dropout (100 Trades, 10% Dropout)."""
        trades_df = generate_synthetic_trades_df(100)
        base = generate_base_metrics()

        def compute() -> Dict[str, float]:
            return simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=0.1,
                base_metrics=base,
                seed=BENCHMARK_SEED,
            )

        result = benchmark(compute)
        assert "profit" in result
        assert "drawdown" in result

    def test_trade_dropout_500_trades_10pct(self, benchmark: Any) -> None:
        """Benchmark: Trade Dropout (500 Trades, 10% Dropout)."""
        trades_df = generate_synthetic_trades_df(500)
        base = generate_base_metrics()

        def compute() -> Dict[str, float]:
            return simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=0.1,
                base_metrics=base,
                seed=BENCHMARK_SEED,
            )

        result = benchmark(compute)
        assert "profit" in result

    def test_trade_dropout_500_trades_25pct(self, benchmark: Any) -> None:
        """Benchmark: Trade Dropout (500 Trades, 25% Dropout)."""
        trades_df = generate_synthetic_trades_df(500)
        base = generate_base_metrics()

        def compute() -> Dict[str, float]:
            return simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=0.25,
                base_metrics=base,
                seed=BENCHMARK_SEED,
            )

        result = benchmark(compute)
        assert "profit" in result

    def test_trade_dropout_2000_trades_10pct(self, benchmark: Any) -> None:
        """Benchmark: Trade Dropout (2000 Trades, 10% Dropout)."""
        trades_df = generate_synthetic_trades_df(2000)
        base = generate_base_metrics()

        def compute() -> Dict[str, float]:
            return simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=0.1,
                base_metrics=base,
                seed=BENCHMARK_SEED,
            )

        result = benchmark(compute)
        assert "profit" in result


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Stability Score
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestStabilityScoreBenchmarks:
    """Benchmarks für Stability Score Berechnungen."""

    def test_stability_score_5_years(self, benchmark: Any) -> None:
        """Benchmark: Stability Score (5 Jahre)."""
        profits = generate_yearly_profits(5)

        def compute() -> float:
            return compute_stability_score_from_yearly_profits(profits)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_stability_score_10_years(self, benchmark: Any) -> None:
        """Benchmark: Stability Score (10 Jahre)."""
        profits = generate_yearly_profits(10)

        def compute() -> float:
            return compute_stability_score_from_yearly_profits(profits)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0

    def test_stability_score_with_durations_5_years(
        self, benchmark: Any
    ) -> None:
        """Benchmark: Stability Score mit Durations (5 Jahre)."""
        profits = generate_yearly_profits(5)
        durations = generate_yearly_durations(profits)

        def compute() -> tuple:
            return compute_stability_score_and_wmape_from_yearly_profits(
                profits, durations_by_year=durations
            )

        result = benchmark(compute)
        score, wmape = result
        assert 0.0 <= score <= 1.0
        assert wmape >= 0.0

    def test_stability_score_20_years(self, benchmark: Any) -> None:
        """Benchmark: Stability Score (20 Jahre)."""
        profits = generate_yearly_profits(20)

        def compute() -> float:
            return compute_stability_score_from_yearly_profits(profits)

        result = benchmark(compute)
        assert 0.0 <= result <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Stress Penalty (Basis-Funktion)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestStressPenaltyBenchmarks:
    """Benchmarks für Basis Stress-Penalty Berechnungen."""

    def test_penalty_computation_10_stress(self, benchmark: Any) -> None:
        """Benchmark: Penalty-Berechnung (10 Stress-Metriken)."""
        base = generate_base_metrics()
        stress_metrics = generate_jitter_metrics(base, n_repeats=10)

        def compute() -> float:
            return compute_penalty_profit_drawdown_sharpe(
                base, stress_metrics, penalty_cap=0.5
            )

        result = benchmark(compute)
        assert 0.0 <= result <= 0.5

    def test_penalty_computation_50_stress(self, benchmark: Any) -> None:
        """Benchmark: Penalty-Berechnung (50 Stress-Metriken)."""
        base = generate_base_metrics()
        stress_metrics = generate_jitter_metrics(base, n_repeats=50)

        def compute() -> float:
            return compute_penalty_profit_drawdown_sharpe(
                base, stress_metrics, penalty_cap=0.5
            )

        result = benchmark(compute)
        assert 0.0 <= result <= 0.5

    def test_score_from_penalty(self, benchmark: Any) -> None:
        """Benchmark: Score from Penalty Transformation."""
        penalties = [0.1, 0.2, 0.3, 0.4, 0.5]

        def compute_all() -> List[float]:
            return [score_from_penalty(p, penalty_cap=0.5) for p in penalties]

        result = benchmark(compute_all)
        assert len(result) == 5
        assert all(0.0 <= s <= 1.0 for s in result)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Combined Rating Pipeline
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestCombinedRatingBenchmarks:
    """Benchmarks für kombinierte Rating-Pipelines."""

    def test_full_rating_pipeline_small(self, benchmark: Any) -> None:
        """Benchmark: Vollständige Rating-Pipeline (klein)."""
        base = generate_base_metrics()
        jitter_metrics = generate_jitter_metrics(base, n_repeats=10)
        shocked_metrics = [
            {
                "profit": base["profit"] * 0.8,
                "drawdown": base["drawdown"] * 1.2,
                "sharpe": base["sharpe"] * 0.9,
            }
        ]
        yearly_profits = generate_yearly_profits(5)
        trades_df = generate_synthetic_trades_df(100)

        def compute_all_scores() -> Dict[str, float]:
            robustness = compute_robustness_score_1(
                base, jitter_metrics, penalty_cap=0.5
            )
            cost_shock = compute_multi_factor_cost_shock_score(
                base, shocked_metrics, penalty_cap=0.5
            )
            stability = compute_stability_score_from_yearly_profits(
                yearly_profits
            )
            dropout = simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=0.1,
                base_metrics=base,
                seed=BENCHMARK_SEED,
            )

            return {
                "robustness": robustness,
                "cost_shock": cost_shock,
                "stability": stability,
                "dropout_profit": dropout.get("profit", 0.0),
            }

        result = benchmark(compute_all_scores)
        assert "robustness" in result
        assert "cost_shock" in result
        assert "stability" in result

    def test_full_rating_pipeline_medium(self, benchmark: Any) -> None:
        """Benchmark: Vollständige Rating-Pipeline (medium)."""
        base = generate_base_metrics()
        jitter_metrics = generate_jitter_metrics(base, n_repeats=50)
        shocked_metrics = [
            {
                "profit": base["profit"] / f,
                "drawdown": base["drawdown"] * f,
                "sharpe": base["sharpe"] / f,
            }
            for f in [1.25, 1.5, 2.0]
        ]
        yearly_profits = generate_yearly_profits(10)
        trades_df = generate_synthetic_trades_df(500)

        def compute_all_scores() -> Dict[str, float]:
            robustness = compute_robustness_score_1(
                base, jitter_metrics, penalty_cap=0.5
            )
            cost_shock = compute_multi_factor_cost_shock_score(
                base, shocked_metrics, penalty_cap=0.5
            )
            stability = compute_stability_score_from_yearly_profits(
                yearly_profits
            )
            dropout = simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=0.15,
                base_metrics=base,
                seed=BENCHMARK_SEED,
            )

            return {
                "robustness": robustness,
                "cost_shock": cost_shock,
                "stability": stability,
                "dropout_profit": dropout.get("profit", 0.0),
            }

        result = benchmark(compute_all_scores)
        assert all(k in result for k in ["robustness", "cost_shock", "stability"])


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Vectorized vs Loop Comparisons
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_rating
class TestVectorizedPerformance:
    """Benchmarks für vektorisierte vs. Loop-basierte Berechnungen."""

    def test_numpy_mean_vs_python_loop(self, benchmark: Any) -> None:
        """Benchmark: NumPy mean vs Python Loop."""
        values = list(np.random.default_rng(BENCHMARK_SEED).normal(0, 1, 1000))

        def numpy_mean() -> float:
            return float(np.mean(values))

        result = benchmark(numpy_mean)
        assert isinstance(result, float)

    def test_penalty_accumulation_vectorized(self, benchmark: Any) -> None:
        """Benchmark: Penalty-Akkumulation (vektorisiert)."""
        n = 100
        base = generate_base_metrics()
        stress_list = generate_jitter_metrics(base, n_repeats=n)

        # Extrahiere Werte als Arrays
        base_profit = base["profit"]
        stress_profits = np.array([m["profit"] for m in stress_list])

        def vectorized_drops() -> float:
            drops = np.maximum(0, (base_profit - stress_profits) / base_profit)
            return float(np.mean(drops))

        result = benchmark(vectorized_drops)
        assert isinstance(result, float)

    def test_dataframe_operations_for_trades(self, benchmark: Any) -> None:
        """Benchmark: DataFrame-Operationen für Trade-Analyse."""
        trades_df = generate_synthetic_trades_df(1000)

        def df_analysis() -> Dict[str, float]:
            return {
                "total_profit": float(trades_df["result"].sum()),
                "avg_r": float(trades_df["r_multiple"].mean()),
                "win_rate": float(
                    (trades_df["r_multiple"] > 0).sum() / len(trades_df)
                ),
                "max_dd": float(
                    (
                        trades_df["result"].cumsum()
                        - trades_df["result"].cumsum().cummax()
                    ).min()
                ),
            }

        result = benchmark(df_analysis)
        assert "total_profit" in result
        assert "avg_r" in result
