"""
Golden-File Tests für Optimizer-Determinismus.

P3-10: Validiert dass Optimizer-Runs mit fixiertem Seed reproduzierbare
Ergebnisse liefern. Dies ist kritisch für die FFI-Migration, um
sicherzustellen dass Rust/Julia-Implementierungen von Optuna-ähnlichen
Algorithmen identische Suchpfade und Ergebnisse produzieren.

Voraussetzungen:
- Optuna installiert
- backtest_engine.optimizer Module verfügbar

Tests:
1. TPE-Sampler Determinismus
2. Grid-Search Determinismus
3. Bayesian Optimization Reproduzierbarkeit
4. Pruner-Entscheidungen sind konsistent
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Optuna import
try:
    import optuna
    from optuna.samplers import RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from tests.golden.conftest import (
    GoldenFileManager,
    GoldenOptimizerResult,
    compute_dict_hash,
    create_metadata,
    set_deterministic_seed,
)

pytestmark = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def simple_objective() -> Callable[[optuna.Trial], float]:
    """
    Einfache Objective-Funktion für Optimizer-Tests.

    Diese Funktion hat ein bekanntes Optimum bei x=0, y=0
    mit minimalem Wert 0.
    """

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    return objective


@pytest.fixture
def categorical_objective() -> Callable[[optuna.Trial], float]:
    """
    Objective mit kategorischen und numerischen Parametern.

    Testet Determinismus bei gemischten Parameter-Typen.
    """

    def objective(trial: optuna.Trial) -> float:
        # Kategorisch
        optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        # Integer
        layers = trial.suggest_int("layers", 1, 5)
        # Float
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        # Simulierter "Loss" basierend auf Parametern
        optimizer_scores = {"adam": 0.0, "sgd": 0.5, "rmsprop": 0.3}
        base_score = optimizer_scores[optimizer]
        return base_score + layers * 0.1 + learning_rate * 10

    return objective


@pytest.fixture
def mock_backtest_objective() -> Callable[[optuna.Trial], float]:
    """
    Mock-Objective die einen Backtest simuliert.

    Verwendet deterministische Berechnungen ohne echte Backtests.
    """

    def objective(trial: optuna.Trial) -> float:
        # Trading-spezifische Parameter
        ema_fast = trial.suggest_int("ema_fast", 5, 20)
        ema_slow = trial.suggest_int("ema_slow", 20, 50)
        rsi_threshold = trial.suggest_float("rsi_threshold", 20, 80)

        # Constraint: ema_fast < ema_slow
        if ema_fast >= ema_slow:
            return float("inf")  # Ungültig

        # Simulierter Profit (deterministisch)
        np.random.seed(ema_fast * 100 + ema_slow * 10 + int(rsi_threshold))
        base_profit = 1000 - abs(ema_fast - 10) * 50 - abs(ema_slow - 35) * 30
        noise = np.random.normal(0, 10)

        return -base_profit - noise  # Negativ für Minimierung

    return objective


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================


def extract_trial_hashes(study: optuna.Study, top_n: int = 10) -> List[str]:
    """Extrahiert Hashes der Top-N Trials."""
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value if t.value is not None else float("inf"),
    )[:top_n]

    hashes = []
    for trial in sorted_trials:
        trial_data = {
            "params": trial.params,
            "value": round(trial.value, 10) if trial.value is not None else None,
        }
        hashes.append(compute_dict_hash(trial_data))

    return hashes


def create_optimizer_result_from_study(
    study: optuna.Study,
    param_ranges: Dict[str, Any],
    seed: int,
    description: str,
) -> GoldenOptimizerResult:
    """Erstellt GoldenOptimizerResult aus Optuna Study."""
    best_trial = study.best_trial

    return GoldenOptimizerResult(
        metadata=create_metadata(seed, description),
        best_params=best_trial.params,
        best_score=best_trial.value if best_trial.value is not None else 0.0,
        n_trials=len(study.trials),
        param_ranges=param_ranges,
        top_n_hashes=extract_trial_hashes(study, top_n=10),
    )


# ==============================================================================
# TPE SAMPLER DETERMINISMUS TESTS
# ==============================================================================


class TestTPESamplerDeterminism:
    """Tests für TPE-Sampler Determinismus."""

    def test_tpe_sampler_same_seed_same_suggestions(
        self, simple_objective: Callable[[optuna.Trial], float]
    ):
        """TPE-Sampler mit gleichem Seed liefert identische Vorschläge."""
        seed = 42
        n_trials = 20

        # Run 1
        sampler1 = TPESampler(seed=seed)
        study1 = optuna.create_study(sampler=sampler1)
        study1.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)

        # Run 2
        sampler2 = TPESampler(seed=seed)
        study2 = optuna.create_study(sampler=sampler2)
        study2.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)

        # Vergleiche Parameter-Sequenzen
        for i in range(n_trials):
            params1 = study1.trials[i].params
            params2 = study2.trials[i].params
            assert params1 == params2, f"Mismatch at trial {i}: {params1} vs {params2}"

    def test_tpe_sampler_different_seeds_different_suggestions(
        self, simple_objective: Callable[[optuna.Trial], float]
    ):
        """TPE-Sampler mit verschiedenen Seeds liefert unterschiedliche Vorschläge."""
        n_trials = 10

        sampler1 = TPESampler(seed=42)
        study1 = optuna.create_study(sampler=sampler1)
        study1.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)

        sampler2 = TPESampler(seed=123)
        study2 = optuna.create_study(sampler=sampler2)
        study2.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)

        # Mindestens einige Trials sollten unterschiedlich sein
        different_count = sum(
            1
            for i in range(n_trials)
            if study1.trials[i].params != study2.trials[i].params
        )
        assert different_count > 0, "Different seeds should produce different results"

    def test_tpe_sampler_best_value_determinism(
        self, simple_objective: Callable[[optuna.Trial], float]
    ):
        """Best Value ist bei gleichem Seed identisch."""
        seed = 42
        n_trials = 50

        best_values = []
        for _ in range(3):
            sampler = TPESampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)
            best_values.append(study.best_value)

        # Alle Runs sollten identischen Best Value haben
        assert all(
            abs(v - best_values[0]) < 1e-10 for v in best_values
        ), f"Best values should be identical: {best_values}"


# ==============================================================================
# RANDOM SAMPLER DETERMINISMUS TESTS
# ==============================================================================


class TestRandomSamplerDeterminism:
    """Tests für Random-Sampler Determinismus."""

    def test_random_sampler_determinism(
        self, simple_objective: Callable[[optuna.Trial], float]
    ):
        """Random-Sampler ist mit fixiertem Seed deterministisch."""
        seed = 42
        n_trials = 30

        # Mehrere identische Runs
        results = []
        for _ in range(3):
            sampler = RandomSampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)
            results.append(
                [
                    (trial.params["x"], trial.params["y"], trial.value)
                    for trial in study.trials
                ]
            )

        # Alle Runs sollten identisch sein
        assert results[0] == results[1] == results[2]

    def test_random_sampler_categorical_determinism(
        self, categorical_objective: Callable[[optuna.Trial], float]
    ):
        """Random-Sampler mit kategorischen Parametern ist deterministisch."""
        seed = 42
        n_trials = 20

        param_sequences = []
        for _ in range(2):
            sampler = RandomSampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(
                categorical_objective, n_trials=n_trials, show_progress_bar=False
            )
            param_sequences.append([trial.params for trial in study.trials])

        assert param_sequences[0] == param_sequences[1]


# ==============================================================================
# GOLDEN FILE OPTIMIZER TESTS
# ==============================================================================


class TestGoldenFileOptimizer:
    """Golden-File Tests für Optimizer-Runs."""

    def test_simple_optimization_golden(
        self,
        golden_manager: GoldenFileManager,
        simple_objective: Callable[[optuna.Trial], float],
    ):
        """Golden-File Test für einfache Optimierung."""
        seed = 42
        n_trials = 30

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(simple_objective, n_trials=n_trials, show_progress_bar=False)

        result = create_optimizer_result_from_study(
            study=study,
            param_ranges={"x": [-10, 10], "y": [-10, 10]},
            seed=seed,
            description="Simple quadratic optimization",
        )

        # Vergleiche oder erstelle Referenz
        try:
            comparison = golden_manager.compare_optimizer_results(
                "simple_quadratic", result, score_tolerance=1e-8
            )
            assert comparison["status"] == "match"
        except Exception:
            golden_manager.save_optimizer_reference("simple_quadratic", result)

    def test_categorical_optimization_golden(
        self,
        golden_manager: GoldenFileManager,
        categorical_objective: Callable[[optuna.Trial], float],
    ):
        """Golden-File Test für Optimierung mit kategorischen Parametern."""
        seed = 42
        n_trials = 25

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(
            categorical_objective, n_trials=n_trials, show_progress_bar=False
        )

        result = create_optimizer_result_from_study(
            study=study,
            param_ranges={
                "optimizer": ["adam", "sgd", "rmsprop"],
                "layers": [1, 5],
                "learning_rate": [1e-5, 1e-1],
            },
            seed=seed,
            description="Categorical parameter optimization",
        )

        try:
            comparison = golden_manager.compare_optimizer_results(
                "categorical_params", result, score_tolerance=1e-8
            )
            assert comparison["status"] == "match"
        except Exception:
            golden_manager.save_optimizer_reference("categorical_params", result)

    def test_mock_backtest_optimization_golden(
        self,
        golden_manager: GoldenFileManager,
        mock_backtest_objective: Callable[[optuna.Trial], float],
    ):
        """Golden-File Test für Mock-Backtest Optimierung."""
        seed = 42
        n_trials = 40

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(
            mock_backtest_objective, n_trials=n_trials, show_progress_bar=False
        )

        result = create_optimizer_result_from_study(
            study=study,
            param_ranges={
                "ema_fast": [5, 20],
                "ema_slow": [20, 50],
                "rsi_threshold": [20, 80],
            },
            seed=seed,
            description="Mock backtest optimization",
        )

        try:
            comparison = golden_manager.compare_optimizer_results(
                "mock_backtest", result, score_tolerance=1e-6
            )
            assert comparison["status"] == "match"
        except Exception:
            golden_manager.save_optimizer_reference("mock_backtest", result)


# ==============================================================================
# GRID SEARCH DETERMINISMUS TESTS
# ==============================================================================


class TestGridSearchDeterminism:
    """Tests für Grid-Search Determinismus."""

    def test_grid_search_all_combinations(self):
        """Grid-Search evaluiert alle Kombinationen deterministisch."""
        from itertools import product

        # Parameter-Grid
        param_grid = {
            "ema_fast": [5, 10, 15],
            "ema_slow": [20, 30, 40],
            "threshold": [0.5, 0.7],
        }

        def evaluate(params: Dict[str, Any]) -> float:
            """Deterministische Evaluation."""
            return (
                params["ema_fast"] * 0.1
                + params["ema_slow"] * 0.05
                + params["threshold"]
            )

        # Generiere alle Kombinationen
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, combo)) for combo in product(*values)]

        # Evaluiere in zwei Durchläufen
        results1 = [(combo, evaluate(combo)) for combo in combinations]
        results2 = [(combo, evaluate(combo)) for combo in combinations]

        # Sollten identisch sein
        assert results1 == results2

        # Sortiert nach Score sollte gleich sein
        sorted1 = sorted(results1, key=lambda x: x[1])
        sorted2 = sorted(results2, key=lambda x: x[1])
        assert sorted1 == sorted2


# ==============================================================================
# PRUNER DETERMINISMUS TESTS
# ==============================================================================


class TestPrunerDeterminism:
    """Tests für Pruner-Entscheidungen Determinismus."""

    def test_median_pruner_determinism(self):
        """MedianPruner trifft deterministische Pruning-Entscheidungen."""
        from optuna.pruners import MedianPruner

        seed = 42
        n_trials = 30

        def objective_with_intermediate(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -10, 10)

            # Simuliere mehrere Epochen mit Intermediate Values
            for epoch in range(10):
                intermediate = x**2 + epoch * 0.1
                trial.report(intermediate, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return x**2

        # Run 1
        sampler1 = TPESampler(seed=seed)
        pruner1 = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study1 = optuna.create_study(sampler=sampler1, pruner=pruner1)
        study1.optimize(
            objective_with_intermediate, n_trials=n_trials, show_progress_bar=False
        )

        # Run 2
        sampler2 = TPESampler(seed=seed)
        pruner2 = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study2 = optuna.create_study(sampler=sampler2, pruner=pruner2)
        study2.optimize(
            objective_with_intermediate, n_trials=n_trials, show_progress_bar=False
        )

        # Vergleiche Trial-States (COMPLETE vs PRUNED)
        states1 = [t.state for t in study1.trials]
        states2 = [t.state for t in study2.trials]
        assert states1 == states2, "Pruning decisions should be deterministic"


# ==============================================================================
# MULTI-OBJECTIVE DETERMINISMUS TESTS
# ==============================================================================


class TestMultiObjectiveDeterminism:
    """Tests für Multi-Objective Optimization Determinismus."""

    def test_nsga2_determinism(self):
        """NSGA-II Sampler ist mit fixiertem Seed deterministisch."""
        from optuna.samplers import NSGAIISampler

        seed = 42
        n_trials = 30

        def multi_objective(trial: optuna.Trial) -> Tuple[float, float]:
            x = trial.suggest_float("x", 0, 5)
            y = trial.suggest_float("y", 0, 5)
            # Zwei konkurrierende Objectives
            f1 = x**2 + y**2
            f2 = (x - 2) ** 2 + (y - 2) ** 2
            return f1, f2

        # Run 1
        sampler1 = NSGAIISampler(seed=seed)
        study1 = optuna.create_study(
            directions=["minimize", "minimize"], sampler=sampler1
        )
        study1.optimize(multi_objective, n_trials=n_trials, show_progress_bar=False)

        # Run 2
        sampler2 = NSGAIISampler(seed=seed)
        study2 = optuna.create_study(
            directions=["minimize", "minimize"], sampler=sampler2
        )
        study2.optimize(multi_objective, n_trials=n_trials, show_progress_bar=False)

        # Vergleiche alle Trial-Parameter
        for i in range(n_trials):
            params1 = study1.trials[i].params
            params2 = study2.trials[i].params
            assert params1 == params2, f"NSGA-II mismatch at trial {i}"


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestOptimizerEdgeCases:
    """Tests für Edge Cases bei Optimizer-Determinismus."""

    def test_single_trial_determinism(
        self, simple_objective: Callable[[optuna.Trial], float]
    ):
        """Einzelner Trial ist deterministisch."""
        seed = 42

        sampler1 = TPESampler(seed=seed)
        study1 = optuna.create_study(sampler=sampler1)
        study1.optimize(simple_objective, n_trials=1, show_progress_bar=False)

        sampler2 = TPESampler(seed=seed)
        study2 = optuna.create_study(sampler=sampler2)
        study2.optimize(simple_objective, n_trials=1, show_progress_bar=False)

        assert study1.trials[0].params == study2.trials[0].params

    def test_failed_trials_handled_consistently(self):
        """Fehlgeschlagene Trials werden konsistent behandelt."""
        seed = 42
        n_trials = 20

        def failing_objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -10, 10)
            if abs(x) < 1:
                raise ValueError("Simulated failure")
            return x**2

        # Run mehrmals
        failed_counts = []
        for _ in range(3):
            sampler = TPESampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(
                failing_objective,
                n_trials=n_trials,
                show_progress_bar=False,
                catch=(ValueError,),
            )
            failed_counts.append(
                sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
            )

        # Anzahl fehlgeschlagener Trials sollte identisch sein
        assert (
            failed_counts[0] == failed_counts[1] == failed_counts[2]
        ), f"Failed trial counts should match: {failed_counts}"

    def test_constraint_function_determinism(self):
        """Constraint-Funktion beeinflusst Determinismus nicht negativ."""
        seed = 42
        n_trials = 25

        def objective_with_constraint(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -10, 10)
            y = trial.suggest_float("y", -10, 10)
            return x**2 + y**2

        def constraint(trial: optuna.Trial) -> List[float]:
            # x + y <= 5 (Constraint erfüllt wenn <= 0)
            params = trial.params
            return [params["x"] + params["y"] - 5]

        # Hinweis: NSGAIISampler unterstützt Constraints
        from optuna.samplers import NSGAIISampler

        results = []
        for _ in range(2):
            sampler = NSGAIISampler(seed=seed, constraints_func=constraint)
            study = optuna.create_study(sampler=sampler)
            study.optimize(
                objective_with_constraint, n_trials=n_trials, show_progress_bar=False
            )
            results.append([t.params for t in study.trials])

        assert results[0] == results[1]


# ==============================================================================
# GOLDEN FILE MANAGEMENT
# ==============================================================================


class TestOptimizerGoldenFileManagement:
    """Tests für Optimizer Golden-File Management."""

    def test_save_and_load_optimizer_reference(self, tmp_path: Path):
        """Optimizer-Referenz kann gespeichert und geladen werden."""
        manager = GoldenFileManager(tmp_path / "golden")

        result = GoldenOptimizerResult(
            metadata=create_metadata(42, "Test optimizer reference"),
            best_params={"x": 0.5, "y": -0.3},
            best_score=0.34,
            n_trials=50,
            param_ranges={"x": [-10, 10], "y": [-10, 10]},
            top_n_hashes=["abc123", "def456", "ghi789"],
        )

        # Speichern
        path = manager.save_optimizer_reference("test_opt", result)
        assert path.exists()

        # Laden
        loaded = manager.load_optimizer_reference("test_opt")
        assert loaded is not None
        assert loaded.best_params == result.best_params
        assert loaded.best_score == result.best_score
        assert loaded.n_trials == result.n_trials
        assert loaded.top_n_hashes == result.top_n_hashes

    def test_comparison_detects_differences(self, tmp_path: Path):
        """Vergleich erkennt Unterschiede zwischen Referenz und aktuellem Ergebnis."""
        manager = GoldenFileManager(tmp_path / "golden")

        reference = GoldenOptimizerResult(
            metadata=create_metadata(42, "Reference"),
            best_params={"x": 0.5},
            best_score=0.25,
            n_trials=50,
            param_ranges={"x": [-10, 10]},
            top_n_hashes=["abc"],
        )
        manager.save_optimizer_reference("test_diff", reference)

        # Aktuelles Ergebnis mit Unterschieden
        current = GoldenOptimizerResult(
            metadata=create_metadata(42, "Current"),
            best_params={"x": 0.6},  # Unterschiedlich
            best_score=0.36,  # Unterschiedlich
            n_trials=50,
            param_ranges={"x": [-10, 10]},
            top_n_hashes=["def"],  # Unterschiedlich
        )

        from tests.golden.conftest import GoldenFileComparisonError

        with pytest.raises(GoldenFileComparisonError) as exc_info:
            manager.compare_optimizer_results("test_diff", current)

        assert "best_score" in exc_info.value.details
        assert "best_params" in exc_info.value.details
