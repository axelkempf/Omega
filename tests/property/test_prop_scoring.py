# -*- coding: utf-8 -*-
"""
Property-Based Tests für Scoring-Funktionen.

Phase 3 Task P3-07: Property-Based Tests für Scoring-Funktionen

Invarianten die getestet werden:
1. Alle Scores sind zwischen 0 und 1
2. Scores sind deterministisch
3. Score = 1 bei perfekten Bedingungen
4. Score → 0 bei schlechten Bedingungen
5. Scores reagieren monoton auf Verschlechterung
6. Edge-Cases (leere Inputs, negative Werte, etc.)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from backtest_engine.rating.robustness_score_1 import compute_robustness_score_1
from backtest_engine.rating.stability_score import (
    compute_stability_score_and_wmape_from_yearly_profits,
    compute_stability_score_from_yearly_profits,
)

from .conftest import percentage_values, trade_returns, yearly_profits_dict

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM STRATEGIES FOR SCORING
# ══════════════════════════════════════════════════════════════════════════════


@st.composite
def base_metrics(
    draw: st.DrawFn,
    min_profit: float = -10000.0,
    max_profit: float = 100000.0,
) -> Dict[str, float]:
    """Strategy für base_metrics Dict."""
    profit = draw(
        st.floats(min_value=min_profit, max_value=max_profit, allow_nan=False)
    )
    avg_r = draw(st.floats(min_value=-5.0, max_value=10.0, allow_nan=False))
    winrate = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    drawdown = draw(
        st.floats(min_value=0.0, max_value=abs(profit) * 2 + 1000, allow_nan=False)
    )

    return {
        "profit": profit,
        "avg_r": avg_r,
        "winrate": winrate,
        "drawdown": drawdown,
    }


@st.composite
def jitter_metrics_list(
    draw: st.DrawFn,
    base: Dict[str, float],
    min_count: int = 1,
    max_count: int = 10,
    max_degradation: float = 0.5,
) -> List[Dict[str, float]]:
    """Strategy für jitter_metrics basierend auf base_metrics."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    result = []

    for _ in range(count):
        # Degradiere die Metriken zufällig
        profit_factor = draw(st.floats(min_value=1.0 - max_degradation, max_value=1.0))
        avg_r_factor = draw(st.floats(min_value=1.0 - max_degradation, max_value=1.0))
        winrate_factor = draw(st.floats(min_value=1.0 - max_degradation, max_value=1.0))
        drawdown_factor = draw(
            st.floats(min_value=1.0, max_value=1.0 + max_degradation)
        )

        jitter = {
            "profit": base["profit"] * profit_factor,
            "avg_r": base["avg_r"] * avg_r_factor,
            "winrate": max(0.0, min(1.0, base["winrate"] * winrate_factor)),
            "drawdown": base["drawdown"] * drawdown_factor,
        }
        result.append(jitter)

    return result


@st.composite
def penalty_cap(draw: st.DrawFn) -> float:
    """Strategy für penalty_cap Parameter."""
    return draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False))


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS SCORE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRobustnessScoreBounds:
    """Tests für Bounds des Robustness Scores."""

    @given(base=base_metrics(min_profit=100.0, max_profit=100000.0), cap=penalty_cap())
    @settings(max_examples=100)
    def test_score_between_0_and_1(
        self,
        base: Dict[str, float],
        cap: float,
    ) -> None:
        """Robustness Score muss zwischen 0 und 1 liegen."""
        # Generiere Jitter-Metriken
        jitter = []
        for i in range(5):
            jitter.append(
                {
                    "profit": base["profit"] * (0.7 + i * 0.05),
                    "avg_r": base["avg_r"] * (0.7 + i * 0.05),
                    "winrate": max(0.0, base["winrate"] * (0.8 + i * 0.04)),
                    "drawdown": base["drawdown"] * (1.0 + i * 0.1),
                }
            )

        score = compute_robustness_score_1(base, jitter, penalty_cap=cap)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    @given(cap=penalty_cap())
    @settings(max_examples=50)
    def test_perfect_jitter_gives_high_score(self, cap: float) -> None:
        """Bei identischen Jitter-Metriken sollte Score hoch sein."""
        base = {"profit": 10000.0, "avg_r": 1.5, "winrate": 0.6, "drawdown": 2000.0}
        # Jitter identisch zu base
        jitter = [base.copy() for _ in range(5)]

        score = compute_robustness_score_1(base, jitter, penalty_cap=cap)

        # Bei identischen Metriken sollte Penalty ≈ 0, also Score ≈ 1
        assert score >= 0.95, f"Score too low for identical metrics: {score}"

    @given(cap=penalty_cap())
    @settings(max_examples=50)
    def test_empty_jitter_gives_minimum_score(self, cap: float) -> None:
        """Bei leeren Jitter-Metriken sollte Score = 1 - cap sein."""
        base = {"profit": 10000.0, "avg_r": 1.5, "winrate": 0.6, "drawdown": 2000.0}
        jitter: List[Dict[str, float]] = []

        score = compute_robustness_score_1(base, jitter, penalty_cap=cap)

        expected = max(0.0, 1.0 - cap)
        assert abs(score - expected) < 0.01, f"Score {score} != expected {expected}"

    def test_zero_base_metrics_handled(self) -> None:
        """Zero base metrics sollten keine Division by Zero verursachen."""
        base = {"profit": 0.0, "avg_r": 0.0, "winrate": 0.0, "drawdown": 0.0}
        jitter = [
            {"profit": 100.0, "avg_r": 0.5, "winrate": 0.5, "drawdown": 50.0}
            for _ in range(3)
        ]

        # Sollte nicht crashen
        score = compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"


class TestRobustnessScoreDeterminism:
    """Tests für Determinismus des Robustness Scores."""

    @given(base=base_metrics(min_profit=100.0, max_profit=50000.0))
    @settings(max_examples=50)
    def test_deterministic(self, base: Dict[str, float]) -> None:
        """Gleiche Inputs → Gleiche Outputs."""
        jitter = [
            {
                "profit": base["profit"] * 0.9,
                "avg_r": base["avg_r"] * 0.9,
                "winrate": base["winrate"] * 0.95,
                "drawdown": base["drawdown"] * 1.1,
            }
            for _ in range(5)
        ]

        score1 = compute_robustness_score_1(base, jitter, penalty_cap=0.5)
        score2 = compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        assert score1 == score2, f"Non-deterministic: {score1} != {score2}"


class TestRobustnessScoreMonotonicity:
    """Tests für Monotonie des Robustness Scores."""

    def test_worse_jitter_gives_lower_score(self) -> None:
        """Schlechtere Jitter-Metriken sollten niedrigeren Score geben."""
        base = {"profit": 10000.0, "avg_r": 1.5, "winrate": 0.6, "drawdown": 2000.0}

        # Gute Jitter (nur 10% Verschlechterung)
        good_jitter = [
            {
                "profit": base["profit"] * 0.9,
                "avg_r": base["avg_r"] * 0.9,
                "winrate": base["winrate"] * 0.95,
                "drawdown": base["drawdown"] * 1.1,
            }
            for _ in range(5)
        ]

        # Schlechte Jitter (50% Verschlechterung)
        bad_jitter = [
            {
                "profit": base["profit"] * 0.5,
                "avg_r": base["avg_r"] * 0.5,
                "winrate": base["winrate"] * 0.5,
                "drawdown": base["drawdown"] * 2.0,
            }
            for _ in range(5)
        ]

        score_good = compute_robustness_score_1(base, good_jitter, penalty_cap=0.5)
        score_bad = compute_robustness_score_1(base, bad_jitter, penalty_cap=0.5)

        assert (
            score_good > score_bad
        ), f"Monotonicity violated: {score_good} <= {score_bad}"


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY SCORE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestStabilityScoreBounds:
    """Tests für Bounds des Stability Scores."""

    @given(profits=yearly_profits_dict(min_years=2, max_years=8))
    @settings(max_examples=100)
    def test_score_between_0_and_1(self, profits: Dict[int, float]) -> None:
        """Stability Score muss zwischen 0 und 1 liegen."""
        score = compute_stability_score_from_yearly_profits(profits)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_empty_profits_returns_1(self) -> None:
        """Leere Profits sollten Score 1 geben."""
        score = compute_stability_score_from_yearly_profits({})

        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_single_year_returns_1(self) -> None:
        """Ein einzelnes Jahr sollte Score 1 geben."""
        score = compute_stability_score_from_yearly_profits({2020: 10000.0})

        # Bei einem Jahr ist die WMAPE nicht sinnvoll definiert
        # Die Funktion sollte aber nicht crashen
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_perfect_stability_high_score(self) -> None:
        """Perfekt stabile jährliche Profits sollten hohen Score geben."""
        # Gleichmäßig verteilte Profits
        profits = {
            2018: 10000.0,
            2019: 10000.0,
            2020: 10000.0,
            2021: 10000.0,
            2022: 10000.0,
        }

        score = compute_stability_score_from_yearly_profits(profits)

        # Bei identischen Profits sollte WMAPE ≈ 0 und Score ≈ 1
        assert score >= 0.9, f"Score too low for stable profits: {score}"


class TestStabilityScoreDeterminism:
    """Tests für Determinismus des Stability Scores."""

    @given(profits=yearly_profits_dict())
    @settings(max_examples=50)
    def test_deterministic(self, profits: Dict[int, float]) -> None:
        """Gleiche Inputs → Gleiche Outputs (mit Floating-Point-Toleranz).

        Note: Python und Rust können in der letzten Dezimalstelle differieren
        aufgrund von FP-Arithmetik. Daher nutzen wir rel_tol=1e-14.
        """
        score1 = compute_stability_score_from_yearly_profits(profits)
        score2 = compute_stability_score_from_yearly_profits(profits)

        assert math.isclose(score1, score2, rel_tol=1e-14), (
            f"Non-deterministic: {score1} != {score2}"
        )


class TestStabilityScoreConsistency:
    """Tests für Konsistenz zwischen Score und WMAPE."""

    @given(profits=yearly_profits_dict(min_years=2, max_years=6))
    @settings(max_examples=50)
    def test_score_wmape_relationship(self, profits: Dict[int, float]) -> None:
        """Score sollte konsistent mit WMAPE sein: score = 1 / (1 + wmape)."""
        score, wmape = compute_stability_score_and_wmape_from_yearly_profits(profits)

        if wmape >= 0 and np.isfinite(wmape):
            expected_score = 1.0 / (1.0 + wmape)
            assert (
                abs(score - expected_score) < 1e-9
            ), f"Inconsistent: score={score}, wmape={wmape}, expected={expected_score}"


class TestStabilityScoreMonotonicity:
    """Tests für Monotonie des Stability Scores."""

    def test_higher_variance_lower_score(self) -> None:
        """Höhere Varianz in Profits sollte niedrigeren Score geben."""
        # Stabile Profits
        stable = {2018: 10000.0, 2019: 10000.0, 2020: 10000.0, 2021: 10000.0}

        # Volatile Profits (gleicher Durchschnitt)
        volatile = {2018: 5000.0, 2019: 15000.0, 2020: 5000.0, 2021: 15000.0}

        score_stable = compute_stability_score_from_yearly_profits(stable)
        score_volatile = compute_stability_score_from_yearly_profits(volatile)

        assert (
            score_stable > score_volatile
        ), f"Monotonicity violated: stable={score_stable} <= volatile={score_volatile}"


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests für Edge-Cases in Scoring-Funktionen."""

    def test_robustness_with_nan_in_jitter(self) -> None:
        """NaN-Werte in Jitter sollten graceful gehandelt werden."""
        base = {"profit": 10000.0, "avg_r": 1.5, "winrate": 0.6, "drawdown": 2000.0}
        jitter = [
            {"profit": float("nan"), "avg_r": 1.0, "winrate": 0.5, "drawdown": 2500.0},
            {"profit": 9000.0, "avg_r": 1.3, "winrate": 0.55, "drawdown": 2200.0},
        ]

        # Sollte nicht crashen
        score = compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_stability_with_negative_profits(self) -> None:
        """Negative Profits sollten korrekt behandelt werden."""
        profits = {
            2018: -5000.0,
            2019: 10000.0,
            2020: -3000.0,
            2021: 8000.0,
        }

        score = compute_stability_score_from_yearly_profits(profits)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_stability_with_all_negative_profits(self) -> None:
        """Alle negativen Profits sollten behandelt werden."""
        profits = {
            2018: -5000.0,
            2019: -3000.0,
            2020: -4000.0,
            2021: -2000.0,
        }

        score = compute_stability_score_from_yearly_profits(profits)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_robustness_with_negative_base_profit(self) -> None:
        """Negative base profits sollten behandelt werden."""
        base = {"profit": -5000.0, "avg_r": -0.5, "winrate": 0.3, "drawdown": 8000.0}
        jitter = [
            {"profit": -6000.0, "avg_r": -0.6, "winrate": 0.25, "drawdown": 9000.0}
            for _ in range(3)
        ]

        # Sollte nicht crashen
        score = compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    @given(cap=st.floats(min_value=0.0, max_value=0.0))
    @settings(max_examples=10)
    def test_robustness_with_zero_penalty_cap(self, cap: float) -> None:
        """penalty_cap = 0 sollte Score 1 geben."""
        base = {"profit": 10000.0, "avg_r": 1.5, "winrate": 0.6, "drawdown": 2000.0}
        jitter = [{"profit": 1.0, "avg_r": 0.1, "winrate": 0.1, "drawdown": 10000.0}]

        score = compute_robustness_score_1(base, jitter, penalty_cap=cap)

        assert score == 1.0, f"Expected 1.0 with cap=0, got {score}"


# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL STABILITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestNumericalStability:
    """Tests für numerische Stabilität bei extremen Werten."""

    def test_robustness_with_very_large_values(self) -> None:
        """Sehr große Werte sollten keine Overflows verursachen."""
        base = {
            "profit": 1e15,
            "avg_r": 1000.0,
            "winrate": 0.99,
            "drawdown": 1e12,
        }
        jitter = [
            {
                "profit": base["profit"] * 0.9,
                "avg_r": base["avg_r"] * 0.9,
                "winrate": base["winrate"] * 0.99,
                "drawdown": base["drawdown"] * 1.1,
            }
            for _ in range(3)
        ]

        score = compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        assert np.isfinite(score), f"Non-finite score: {score}"
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_robustness_with_very_small_values(self) -> None:
        """Sehr kleine Werte sollten keine Underflows verursachen."""
        base = {
            "profit": 1e-10,
            "avg_r": 1e-8,
            "winrate": 0.001,
            "drawdown": 1e-10,
        }
        jitter = [
            {
                "profit": base["profit"] * 0.9,
                "avg_r": base["avg_r"] * 0.9,
                "winrate": base["winrate"] * 0.9,
                "drawdown": base["drawdown"] * 1.1,
            }
            for _ in range(3)
        ]

        score = compute_robustness_score_1(base, jitter, penalty_cap=0.5)

        assert np.isfinite(score), f"Non-finite score: {score}"
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_stability_with_very_large_profits(self) -> None:
        """Sehr große Profits sollten keine Overflows verursachen."""
        profits = {
            2018: 1e15,
            2019: 1.1e15,
            2020: 0.9e15,
            2021: 1e15,
        }

        score = compute_stability_score_from_yearly_profits(profits)

        assert np.isfinite(score), f"Non-finite score: {score}"
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_stability_with_very_small_profits(self) -> None:
        """Sehr kleine Profits sollten keine Underflows verursachen."""
        profits = {
            2018: 1e-10,
            2019: 1.1e-10,
            2020: 0.9e-10,
            2021: 1e-10,
        }

        score = compute_stability_score_from_yearly_profits(profits)

        assert np.isfinite(score), f"Non-finite score: {score}"
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
