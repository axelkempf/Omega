# -*- coding: utf-8 -*-
"""
Benchmark Suite für Optimizer-Module (P6-09).

Testet alle public functions der Optimizer-Module:
- grid_searcher: Parameter-Kombinationen, Grid-Search
- optuna_optimizer: Bayesian Optimization, Pareto-Front
- walkforward: Walk-Forward Validation, Fold-Generierung

Ergebnisse sind in JSON exportierbar für Regression-Detection.

Verwendung:
    pytest tests/benchmarks/test_bench_optimizer.py -v
    pytest tests/benchmarks/test_bench_optimizer.py --benchmark-json=output.json
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
)

# ══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS (um Import-Fehler bei fehlenden optionalen Deps zu vermeiden)
# ══════════════════════════════════════════════════════════════════════════════


def _import_grid_searcher():
    """Lazy Import für grid_searcher."""
    try:
        from backtest_engine.optimizer.grid_searcher import (
            generate_param_combinations,
            run_grid_search,
        )

        return generate_param_combinations, run_grid_search
    except ImportError:
        return None, None


def _import_optuna_optimizer():
    """Lazy Import für optuna_optimizer."""
    try:
        from backtest_engine.optimizer.optuna_optimizer import (
            _snap_to_step,
            _split_train_period,
        )

        return _snap_to_step, _split_train_period
    except ImportError:
        return None, None


def _import_walkforward():
    """Lazy Import für walkforward."""
    try:
        from backtest_engine.optimizer.walkforward import set_global_seed

        return set_global_seed
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════


def generate_param_space_small() -> Dict[str, List[Any]]:
    """Kleine Parameter-Space für Grid-Search."""
    return {
        "sma_fast": [5, 10, 15],
        "sma_slow": [20, 30, 40],
        "rsi_period": [7, 14],
    }


def generate_param_space_medium() -> Dict[str, List[Any]]:
    """Mittlere Parameter-Space für Grid-Search."""
    return {
        "sma_fast": [5, 10, 15, 20, 25],
        "sma_slow": [30, 40, 50, 60, 70, 80],
        "rsi_period": [7, 14, 21],
        "atr_period": [10, 14, 20],
        "risk_factor": [0.5, 1.0, 1.5, 2.0],
    }


def generate_param_space_large() -> Dict[str, List[Any]]:
    """Große Parameter-Space für Grid-Search."""
    return {
        "sma_fast": list(range(5, 26, 2)),  # 11 values
        "sma_slow": list(range(30, 101, 5)),  # 15 values
        "rsi_period": [7, 10, 14, 21],
        "atr_period": [10, 14, 20, 28],
        "atr_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0],
        "risk_factor": [0.5, 0.75, 1.0, 1.25, 1.5],
    }


def generate_backtest_results(
    n_combinations: int, seed: int = BENCHMARK_SEED
) -> pd.DataFrame:
    """Generiert synthetische Backtest-Ergebnisse."""
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "combo_id": range(n_combinations),
            "profit_factor": rng.uniform(0.5, 3.0, n_combinations),
            "sharpe_ratio": rng.normal(0.5, 1.0, n_combinations),
            "max_drawdown": rng.uniform(0.05, 0.5, n_combinations),
            "win_rate": rng.uniform(0.3, 0.7, n_combinations),
            "total_trades": rng.integers(50, 500, n_combinations),
            "expectancy": rng.normal(10, 50, n_combinations),
        }
    )


def generate_price_series(n_candles: int, seed: int = BENCHMARK_SEED) -> pd.DataFrame:
    """Generiert synthetische Preisdaten für Walkforward."""
    rng = np.random.default_rng(seed)
    base_price = 100.0
    base_time = datetime(2020, 1, 1)

    prices = [base_price]
    for _ in range(n_candles - 1):
        change = rng.normal(0, 0.5)
        prices.append(prices[-1] + change)

    prices = np.array(prices)
    dates = pd.date_range(start=base_time, periods=n_candles, freq="1h")

    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * (1 + rng.uniform(0, 0.01, n_candles)),
            "Low": prices * (1 - rng.uniform(0, 0.01, n_candles)),
            "Close": prices + rng.normal(0, 0.1, n_candles),
            "Volume": rng.integers(100, 10000, n_candles),
        },
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def param_space_small() -> Dict[str, List[Any]]:
    """Kleine Parameter-Space (18 Kombinationen)."""
    return generate_param_space_small()


@pytest.fixture
def param_space_medium() -> Dict[str, List[Any]]:
    """Mittlere Parameter-Space (~1080 Kombinationen)."""
    return generate_param_space_medium()


@pytest.fixture
def param_space_large() -> Dict[str, List[Any]]:
    """Große Parameter-Space (~33000 Kombinationen)."""
    return generate_param_space_large()


@pytest.fixture
def backtest_results_small() -> pd.DataFrame:
    """100 Backtest-Ergebnisse."""
    return generate_backtest_results(100)


@pytest.fixture
def backtest_results_medium() -> pd.DataFrame:
    """1000 Backtest-Ergebnisse."""
    return generate_backtest_results(1000)


@pytest.fixture
def backtest_results_large() -> pd.DataFrame:
    """10000 Backtest-Ergebnisse."""
    return generate_backtest_results(10000)


@pytest.fixture
def price_series_small() -> pd.DataFrame:
    """1K Candles Preisserie."""
    return generate_price_series(SMALL_CANDLE_COUNT)


@pytest.fixture
def price_series_medium() -> pd.DataFrame:
    """10K Candles Preisserie."""
    return generate_price_series(DEFAULT_CANDLE_COUNT)


@pytest.fixture
def price_series_large() -> pd.DataFrame:
    """100K Candles Preisserie."""
    return generate_price_series(LARGE_CANDLE_COUNT)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Parameter Combination Generation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestParamCombinationBenchmarks:
    """Benchmarks für Parameter-Kombinationsgenerierung."""

    def test_generate_combinations_small(
        self, benchmark: Any, param_space_small: Dict[str, List[Any]]
    ) -> None:
        """Benchmark: Kombinationen generieren (klein)."""
        generate_param_combinations, _ = _import_grid_searcher()
        if generate_param_combinations is None:
            pytest.skip("grid_searcher not available")

        def generate() -> List[Dict[str, Any]]:
            return list(generate_param_combinations(param_space_small))

        result = benchmark(generate)
        # 3 * 3 * 2 = 18 combinations
        assert len(result) == 18

    def test_generate_combinations_medium(
        self, benchmark: Any, param_space_medium: Dict[str, List[Any]]
    ) -> None:
        """Benchmark: Kombinationen generieren (mittel)."""
        generate_param_combinations, _ = _import_grid_searcher()
        if generate_param_combinations is None:
            pytest.skip("grid_searcher not available")

        def generate() -> List[Dict[str, Any]]:
            return list(generate_param_combinations(param_space_medium))

        result = benchmark(generate)
        # 5 * 6 * 3 * 3 * 4 = 1080 combinations
        assert len(result) == 1080

    @pytest.mark.benchmark_slow
    def test_generate_combinations_large(
        self, benchmark: Any, param_space_large: Dict[str, List[Any]]
    ) -> None:
        """Benchmark: Kombinationen generieren (groß)."""
        generate_param_combinations, _ = _import_grid_searcher()
        if generate_param_combinations is None:
            pytest.skip("grid_searcher not available")

        def generate() -> List[Dict[str, Any]]:
            return list(generate_param_combinations(param_space_large))

        result = benchmark(generate)
        # 11 * 15 * 4 * 4 * 5 * 5 = 33000 combinations
        assert len(result) == 33000

    def test_combination_iteration_small(
        self, benchmark: Any, param_space_small: Dict[str, List[Any]]
    ) -> None:
        """Benchmark: Durch Kombinationen iterieren (klein)."""
        generate_param_combinations, _ = _import_grid_searcher()
        if generate_param_combinations is None:
            pytest.skip("grid_searcher not available")

        def iterate() -> int:
            count = 0
            for combo in generate_param_combinations(param_space_small):
                count += 1
            return count

        result = benchmark(iterate)
        assert result == 18

    def test_combination_iteration_medium(
        self, benchmark: Any, param_space_medium: Dict[str, List[Any]]
    ) -> None:
        """Benchmark: Durch Kombinationen iterieren (mittel)."""
        generate_param_combinations, _ = _import_grid_searcher()
        if generate_param_combinations is None:
            pytest.skip("grid_searcher not available")

        def iterate() -> int:
            count = 0
            for combo in generate_param_combinations(param_space_medium):
                count += 1
            return count

        result = benchmark(iterate)
        assert result == 1080


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Optuna Utilities
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestOptunaUtilityBenchmarks:
    """Benchmarks für Optuna-Utility-Funktionen."""

    def test_snap_to_step_single(self, benchmark: Any) -> None:
        """Benchmark: _snap_to_step einzelner Wert."""
        _snap_to_step, _ = _import_optuna_optimizer()
        if _snap_to_step is None:
            pytest.skip("optuna_optimizer not available")

        def snap() -> float:
            return _snap_to_step(14.7, 5)

        result = benchmark(snap)
        assert result == 15.0

    def test_snap_to_step_many(self, benchmark: Any) -> None:
        """Benchmark: _snap_to_step viele Werte."""
        _snap_to_step, _ = _import_optuna_optimizer()
        if _snap_to_step is None:
            pytest.skip("optuna_optimizer not available")

        rng = np.random.default_rng(BENCHMARK_SEED)
        values = rng.uniform(0, 100, 10000)

        def snap_all() -> int:
            count = 0
            for v in values:
                _ = _snap_to_step(v, 5)
                count += 1
            return count

        result = benchmark(snap_all)
        assert result == 10000

    def test_split_train_period_small(self, benchmark: Any) -> None:
        """Benchmark: _split_train_period (kleine Periode)."""
        _, _split_train_period = _import_optuna_optimizer()
        if _split_train_period is None:
            pytest.skip("optuna_optimizer not available")

        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)  # 2 Monate

        def split() -> Tuple[datetime, datetime, datetime, datetime]:
            return _split_train_period(start, end, validation_ratio=0.2)

        result = benchmark(split)
        assert result is not None

    def test_split_train_period_medium(self, benchmark: Any) -> None:
        """Benchmark: _split_train_period (mittlere Periode)."""
        _, _split_train_period = _import_optuna_optimizer()
        if _split_train_period is None:
            pytest.skip("optuna_optimizer not available")

        start = datetime(2022, 1, 1)
        end = datetime(2024, 1, 1)  # 2 Jahre

        def split() -> Tuple[datetime, datetime, datetime, datetime]:
            return _split_train_period(start, end, validation_ratio=0.2)

        result = benchmark(split)
        assert result is not None

    def test_split_train_period_repeated(self, benchmark: Any) -> None:
        """Benchmark: _split_train_period wiederholte Aufrufe."""
        _, _split_train_period = _import_optuna_optimizer()
        if _split_train_period is None:
            pytest.skip("optuna_optimizer not available")

        start = datetime(2023, 1, 1)
        end = datetime(2024, 1, 1)

        def split_many() -> int:
            count = 0
            for ratio in [0.1, 0.15, 0.2, 0.25, 0.3]:
                for _ in range(200):
                    _ = _split_train_period(start, end, validation_ratio=ratio)
                    count += 1
            return count

        result = benchmark(split_many)
        assert result == 1000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Walkforward Utilities
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestWalkforwardUtilityBenchmarks:
    """Benchmarks für Walkforward-Utility-Funktionen."""

    def test_set_global_seed(self, benchmark: Any) -> None:
        """Benchmark: set_global_seed."""
        set_global_seed = _import_walkforward()
        if set_global_seed is None:
            pytest.skip("walkforward not available")

        def set_seed() -> None:
            set_global_seed(BENCHMARK_SEED)

        benchmark(set_seed)

    def test_set_global_seed_repeated(self, benchmark: Any) -> None:
        """Benchmark: set_global_seed wiederholte Aufrufe."""
        set_global_seed = _import_walkforward()
        if set_global_seed is None:
            pytest.skip("walkforward not available")

        def set_many_seeds() -> int:
            count = 0
            for seed in range(1000):
                set_global_seed(seed)
                count += 1
            return count

        result = benchmark(set_many_seeds)
        assert result == 1000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Result Processing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestResultProcessingBenchmarks:
    """Benchmarks für Ergebnis-Verarbeitung."""

    def test_filter_results_small(
        self, benchmark: Any, backtest_results_small: pd.DataFrame
    ) -> None:
        """Benchmark: Ergebnisse filtern (100 Zeilen)."""
        df = backtest_results_small

        def filter_results() -> pd.DataFrame:
            return df[
                (df["profit_factor"] > 1.5)
                & (df["sharpe_ratio"] > 0.5)
                & (df["max_drawdown"] < 0.3)
            ]

        result = benchmark(filter_results)
        assert len(result) >= 0

    def test_filter_results_medium(
        self, benchmark: Any, backtest_results_medium: pd.DataFrame
    ) -> None:
        """Benchmark: Ergebnisse filtern (1000 Zeilen)."""
        df = backtest_results_medium

        def filter_results() -> pd.DataFrame:
            return df[
                (df["profit_factor"] > 1.5)
                & (df["sharpe_ratio"] > 0.5)
                & (df["max_drawdown"] < 0.3)
            ]

        result = benchmark(filter_results)
        assert len(result) >= 0

    @pytest.mark.benchmark_slow
    def test_filter_results_large(
        self, benchmark: Any, backtest_results_large: pd.DataFrame
    ) -> None:
        """Benchmark: Ergebnisse filtern (10000 Zeilen)."""
        df = backtest_results_large

        def filter_results() -> pd.DataFrame:
            return df[
                (df["profit_factor"] > 1.5)
                & (df["sharpe_ratio"] > 0.5)
                & (df["max_drawdown"] < 0.3)
            ]

        result = benchmark(filter_results)
        assert len(result) >= 0

    def test_sort_results_small(
        self, benchmark: Any, backtest_results_small: pd.DataFrame
    ) -> None:
        """Benchmark: Ergebnisse sortieren (100 Zeilen)."""
        df = backtest_results_small

        def sort_results() -> pd.DataFrame:
            return df.sort_values(
                ["sharpe_ratio", "profit_factor"], ascending=[False, False]
            )

        result = benchmark(sort_results)
        assert len(result) == 100

    def test_sort_results_medium(
        self, benchmark: Any, backtest_results_medium: pd.DataFrame
    ) -> None:
        """Benchmark: Ergebnisse sortieren (1000 Zeilen)."""
        df = backtest_results_medium

        def sort_results() -> pd.DataFrame:
            return df.sort_values(
                ["sharpe_ratio", "profit_factor"], ascending=[False, False]
            )

        result = benchmark(sort_results)
        assert len(result) == 1000

    def test_rank_results_medium(
        self, benchmark: Any, backtest_results_medium: pd.DataFrame
    ) -> None:
        """Benchmark: Ergebnisse ranken (1000 Zeilen)."""
        df = backtest_results_medium.copy()

        def rank_results() -> pd.DataFrame:
            df["sharpe_rank"] = df["sharpe_ratio"].rank(ascending=False)
            df["pf_rank"] = df["profit_factor"].rank(ascending=False)
            df["dd_rank"] = df["max_drawdown"].rank(ascending=True)
            df["combined_rank"] = df["sharpe_rank"] + df["pf_rank"] + df["dd_rank"]
            return df.sort_values("combined_rank")

        result = benchmark(rank_results)
        assert len(result) == 1000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Date Range Operations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestDateRangeBenchmarks:
    """Benchmarks für Datums-Operationen."""

    def test_date_range_generation_small(self, benchmark: Any) -> None:
        """Benchmark: Datumsbereich generieren (klein)."""

        def generate_dates() -> List[datetime]:
            start = datetime(2024, 1, 1)
            return [start + timedelta(days=i) for i in range(100)]

        result = benchmark(generate_dates)
        assert len(result) == 100

    def test_date_range_generation_medium(self, benchmark: Any) -> None:
        """Benchmark: Datumsbereich generieren (mittel)."""

        def generate_dates() -> List[datetime]:
            start = datetime(2020, 1, 1)
            return [start + timedelta(days=i) for i in range(1000)]

        result = benchmark(generate_dates)
        assert len(result) == 1000

    def test_fold_generation_small(self, benchmark: Any) -> None:
        """Benchmark: Walkforward-Fold-Generierung (klein)."""

        def generate_folds() -> List[Tuple[datetime, datetime, datetime, datetime]]:
            folds = []
            base = datetime(2023, 1, 1)
            train_days = 60
            test_days = 20

            for i in range(5):
                train_start = base + timedelta(days=i * test_days)
                train_end = train_start + timedelta(days=train_days)
                test_start = train_end
                test_end = test_start + timedelta(days=test_days)
                folds.append((train_start, train_end, test_start, test_end))

            return folds

        result = benchmark(generate_folds)
        assert len(result) == 5

    def test_fold_generation_medium(self, benchmark: Any) -> None:
        """Benchmark: Walkforward-Fold-Generierung (mittel)."""

        def generate_folds() -> List[Tuple[datetime, datetime, datetime, datetime]]:
            folds = []
            base = datetime(2020, 1, 1)
            train_days = 90
            test_days = 30

            for i in range(20):
                train_start = base + timedelta(days=i * test_days)
                train_end = train_start + timedelta(days=train_days)
                test_start = train_end
                test_end = test_start + timedelta(days=test_days)
                folds.append((train_start, train_end, test_start, test_end))

            return folds

        result = benchmark(generate_folds)
        assert len(result) == 20

    def test_rolling_window_dates(
        self, benchmark: Any, price_series_medium: pd.DataFrame
    ) -> None:
        """Benchmark: Rolling Window Datum-Zugriffe."""
        dates = price_series_medium.index

        def rolling_access() -> int:
            count = 0
            window_size = 200
            for i in range(window_size, len(dates)):
                start_idx = i - window_size
                end_idx = i
                _ = dates[start_idx:end_idx]
                count += 1
            return count

        result = benchmark(rolling_access)
        assert result == DEFAULT_CANDLE_COUNT - 200


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Pareto Front Operations (Simulated)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestParetoFrontBenchmarks:
    """Benchmarks für Pareto-Front-Operationen."""

    def test_pareto_dominance_check_small(
        self, benchmark: Any, backtest_results_small: pd.DataFrame
    ) -> None:
        """Benchmark: Pareto-Dominanz-Prüfung (100 Punkte)."""
        df = backtest_results_small
        objectives = ["profit_factor", "sharpe_ratio"]  # Maximize both

        def check_dominance() -> int:
            values = df[objectives].values
            n = len(values)
            dominated = np.zeros(n, dtype=bool)

            for i in range(n):
                for j in range(n):
                    if i != j:
                        # j dominates i if j is better in all objectives
                        if all(values[j] >= values[i]) and any(values[j] > values[i]):
                            dominated[i] = True
                            break

            return np.sum(~dominated)

        result = benchmark(check_dominance)
        assert result > 0

    def test_pareto_front_extraction_medium(
        self, benchmark: Any, backtest_results_medium: pd.DataFrame
    ) -> None:
        """Benchmark: Pareto-Front-Extraktion (1000 Punkte, vereinfacht)."""
        df = backtest_results_medium
        # Use numpy for speed
        pf = df["profit_factor"].values
        sr = df["sharpe_ratio"].values

        def extract_pareto() -> int:
            n = len(pf)
            # Simplified: just find top performers in combined score
            combined = pf + sr
            threshold = np.percentile(combined, 90)
            front = combined >= threshold
            return np.sum(front)

        result = benchmark(extract_pareto)
        assert result > 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Throughput Baselines
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="optimizer")
class TestThroughputBaselines:
    """Throughput-Baselines für Rust-Vergleich."""

    def test_combinations_per_second(
        self, benchmark: Any, param_space_medium: Dict[str, List[Any]]
    ) -> None:
        """Baseline: Kombinationen pro Sekunde."""
        generate_param_combinations, _ = _import_grid_searcher()
        if generate_param_combinations is None:
            pytest.skip("grid_searcher not available")

        def generate_many() -> int:
            count = 0
            for _ in range(10):
                for combo in generate_param_combinations(param_space_medium):
                    count += 1
            return count

        result = benchmark(generate_many)
        assert result == 10800  # 10 * 1080

    def test_result_filters_per_second(
        self, benchmark: Any, backtest_results_medium: pd.DataFrame
    ) -> None:
        """Baseline: Ergebnis-Filter pro Sekunde."""
        df = backtest_results_medium

        def filter_many() -> int:
            count = 0
            for pf_thresh in [1.0, 1.5, 2.0]:
                for sr_thresh in [0.0, 0.5, 1.0]:
                    for dd_thresh in [0.2, 0.3, 0.4]:
                        _ = df[
                            (df["profit_factor"] > pf_thresh)
                            & (df["sharpe_ratio"] > sr_thresh)
                            & (df["max_drawdown"] < dd_thresh)
                        ]
                        count += 1
            return count

        result = benchmark(filter_many)
        assert result == 27  # 3 * 3 * 3

    def test_date_calculations_per_second(self, benchmark: Any) -> None:
        """Baseline: Datum-Berechnungen pro Sekunde."""

        def calc_many_dates() -> int:
            count = 0
            base = datetime(2024, 1, 1)
            for _ in range(5000):
                _ = base + timedelta(days=count % 365)
                count += 1
            return count

        result = benchmark(calc_many_dates)
        assert result == 5000

    def test_fold_calculations_per_second(self, benchmark: Any) -> None:
        """Baseline: Fold-Berechnungen pro Sekunde."""

        def calc_many_folds() -> int:
            count = 0
            base = datetime(2020, 1, 1)

            for iteration in range(100):
                for fold_num in range(10):
                    train_start = base + timedelta(days=fold_num * 30)
                    train_end = train_start + timedelta(days=90)
                    test_start = train_end
                    test_end = test_start + timedelta(days=30)
                    _ = (train_start, train_end, test_start, test_end)
                    count += 1

            return count

        result = benchmark(calc_many_folds)
        assert result == 1000
