# -*- coding: utf-8 -*-
"""
Benchmark Suite für SymbolDataSlicer (P6-08).

Testet alle public functions des SymbolDataSlicer-Moduls:
- CandleSet Operations (get_latest, get_all)
- SymbolDataSlice Operations (set_index, get, series_ref, latest, history)
- History View Zugriffe

Ergebnisse sind in JSON exportierbar für Regression-Detection.

Verwendung:
    pytest tests/benchmarks/test_bench_symbol_data_slicer.py -v
    pytest tests/benchmarks/test_bench_symbol_data_slicer.py --benchmark-json=output.json
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest_engine.core.symbol_data_slicer import CandleSet, SymbolDataSlice

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════


def generate_candle_dataframe(
    n_candles: int, seed: int = BENCHMARK_SEED, symbol: str = "EURUSD"
) -> pd.DataFrame:
    """Generiert einen OHLCV-DataFrame."""
    rng = np.random.default_rng(seed)
    base_price = 1.1000 if "USD" in symbol else 150.0
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    opens = [base_price]
    for _ in range(n_candles - 1):
        change = rng.normal(0, 0.0001 * base_price)
        opens.append(opens[-1] + change)

    opens = np.array(opens)
    highs = opens * (1 + rng.uniform(0.0001, 0.001, n_candles))
    lows = opens * (1 - rng.uniform(0.0001, 0.001, n_candles))
    closes = lows + rng.uniform(0.3, 0.7, n_candles) * (highs - lows)
    volumes = rng.integers(100, 10000, n_candles)

    dates = pd.date_range(start=base_time, periods=n_candles, freq="1h")

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


def generate_multi_timeframe_data(
    n_candles: int, seed: int = BENCHMARK_SEED
) -> Dict[str, pd.DataFrame]:
    """Generiert Multi-Timeframe Daten."""
    return {
        "M1": generate_candle_dataframe(n_candles, seed),
        "M5": generate_candle_dataframe(n_candles // 5, seed + 1),
        "M15": generate_candle_dataframe(n_candles // 15, seed + 2),
        "H1": generate_candle_dataframe(n_candles // 60, seed + 3),
        "H4": generate_candle_dataframe(n_candles // 240, seed + 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def candle_df_small() -> pd.DataFrame:
    """1K Candles DataFrame."""
    return generate_candle_dataframe(SMALL_CANDLE_COUNT)


@pytest.fixture
def candle_df_medium() -> pd.DataFrame:
    """10K Candles DataFrame."""
    return generate_candle_dataframe(DEFAULT_CANDLE_COUNT)


@pytest.fixture
def candle_df_large() -> pd.DataFrame:
    """100K Candles DataFrame."""
    return generate_candle_dataframe(LARGE_CANDLE_COUNT)


@pytest.fixture
def candle_set_small(candle_df_small: pd.DataFrame) -> CandleSet:
    """CandleSet mit 1K Candles."""
    return CandleSet(candle_df_small)


@pytest.fixture
def candle_set_medium(candle_df_medium: pd.DataFrame) -> CandleSet:
    """CandleSet mit 10K Candles."""
    return CandleSet(candle_df_medium)


@pytest.fixture
def candle_set_large(candle_df_large: pd.DataFrame) -> CandleSet:
    """CandleSet mit 100K Candles."""
    return CandleSet(candle_df_large)


@pytest.fixture
def symbol_slice_small(candle_df_small: pd.DataFrame) -> SymbolDataSlice:
    """SymbolDataSlice mit 1K Candles."""
    return SymbolDataSlice(candle_df_small)


@pytest.fixture
def symbol_slice_medium(candle_df_medium: pd.DataFrame) -> SymbolDataSlice:
    """SymbolDataSlice mit 10K Candles."""
    return SymbolDataSlice(candle_df_medium)


@pytest.fixture
def symbol_slice_large(candle_df_large: pd.DataFrame) -> SymbolDataSlice:
    """SymbolDataSlice mit 100K Candles."""
    return SymbolDataSlice(candle_df_large)


@pytest.fixture
def multi_tf_data_small() -> Dict[str, pd.DataFrame]:
    """Multi-TF Daten basierend auf 1K M1 Candles."""
    return generate_multi_timeframe_data(SMALL_CANDLE_COUNT)


@pytest.fixture
def multi_tf_data_medium() -> Dict[str, pd.DataFrame]:
    """Multi-TF Daten basierend auf 10K M1 Candles."""
    return generate_multi_timeframe_data(DEFAULT_CANDLE_COUNT)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: CandleSet Operations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestCandleSetBenchmarks:
    """Benchmarks für CandleSet-Operationen."""

    def test_get_latest_small(
        self, benchmark: Any, candle_set_small: CandleSet
    ) -> None:
        """Benchmark: get_latest() (1K Candles)."""

        def get_latest() -> pd.Series:
            return candle_set_small.get_latest()

        result = benchmark(get_latest)
        assert result is not None

    def test_get_latest_medium(
        self, benchmark: Any, candle_set_medium: CandleSet
    ) -> None:
        """Benchmark: get_latest() (10K Candles)."""

        def get_latest() -> pd.Series:
            return candle_set_medium.get_latest()

        result = benchmark(get_latest)
        assert result is not None

    @pytest.mark.benchmark_slow
    def test_get_latest_large(
        self, benchmark: Any, candle_set_large: CandleSet
    ) -> None:
        """Benchmark: get_latest() (100K Candles)."""

        def get_latest() -> pd.Series:
            return candle_set_large.get_latest()

        result = benchmark(get_latest)
        assert result is not None

    def test_get_all_small(self, benchmark: Any, candle_set_small: CandleSet) -> None:
        """Benchmark: get_all() (1K Candles)."""

        def get_all() -> pd.DataFrame:
            return candle_set_small.get_all()

        result = benchmark(get_all)
        assert len(result) == SMALL_CANDLE_COUNT

    def test_get_all_medium(self, benchmark: Any, candle_set_medium: CandleSet) -> None:
        """Benchmark: get_all() (10K Candles)."""

        def get_all() -> pd.DataFrame:
            return candle_set_medium.get_all()

        result = benchmark(get_all)
        assert len(result) == DEFAULT_CANDLE_COUNT

    def test_get_latest_repeated(
        self, benchmark: Any, candle_set_medium: CandleSet
    ) -> None:
        """Benchmark: Wiederholte get_latest() Aufrufe."""

        def repeated_latest() -> int:
            count = 0
            for _ in range(1000):
                _ = candle_set_medium.get_latest()
                count += 1
            return count

        result = benchmark(repeated_latest)
        assert result == 1000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: SymbolDataSlice Index Operations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestIndexOperationsBenchmarks:
    """Benchmarks für Index-Operationen."""

    def test_set_index_sequential_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: Sequentielles set_index (1K Indices)."""

        def set_indices() -> int:
            for i in range(SMALL_CANDLE_COUNT):
                symbol_slice_small.set_index(i)
            return SMALL_CANDLE_COUNT

        result = benchmark(set_indices)
        assert result == SMALL_CANDLE_COUNT

    def test_set_index_sequential_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: Sequentielles set_index (10K Indices)."""

        def set_indices() -> int:
            for i in range(DEFAULT_CANDLE_COUNT):
                symbol_slice_medium.set_index(i)
            return DEFAULT_CANDLE_COUNT

        result = benchmark(set_indices)
        assert result == DEFAULT_CANDLE_COUNT

    def test_set_index_random_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: Random set_index (1K Random Indices)."""
        rng = np.random.default_rng(BENCHMARK_SEED)
        indices = rng.integers(0, SMALL_CANDLE_COUNT, size=1000)

        def set_random_indices() -> int:
            for idx in indices:
                symbol_slice_small.set_index(idx)
            return len(indices)

        result = benchmark(set_random_indices)
        assert result == 1000

    def test_set_index_random_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: Random set_index (5K Random Indices)."""
        rng = np.random.default_rng(BENCHMARK_SEED)
        indices = rng.integers(0, DEFAULT_CANDLE_COUNT, size=5000)

        def set_random_indices() -> int:
            for idx in indices:
                symbol_slice_medium.set_index(idx)
            return len(indices)

        result = benchmark(set_random_indices)
        assert result == 5000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: SymbolDataSlice Data Access
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestDataAccessBenchmarks:
    """Benchmarks für Daten-Zugriff."""

    def test_get_column_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: get() für einzelne Spalte (1K Candles)."""
        symbol_slice_small.set_index(SMALL_CANDLE_COUNT - 1)

        def get_column() -> pd.Series:
            return symbol_slice_small.get("Close")

        result = benchmark(get_column)
        assert result is not None

    def test_get_column_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: get() für einzelne Spalte (10K Candles)."""
        symbol_slice_medium.set_index(DEFAULT_CANDLE_COUNT - 1)

        def get_column() -> pd.Series:
            return symbol_slice_medium.get("Close")

        result = benchmark(get_column)
        assert result is not None

    def test_series_ref_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: series_ref() (1K Candles)."""
        symbol_slice_small.set_index(SMALL_CANDLE_COUNT - 1)

        def get_series_ref() -> pd.Series:
            return symbol_slice_small.series_ref("Close")

        result = benchmark(get_series_ref)
        assert result is not None

    def test_series_ref_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: series_ref() (10K Candles)."""
        symbol_slice_medium.set_index(DEFAULT_CANDLE_COUNT - 1)

        def get_series_ref() -> pd.Series:
            return symbol_slice_medium.series_ref("Close")

        result = benchmark(get_series_ref)
        assert result is not None

    def test_latest_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: latest() (1K Candles)."""
        symbol_slice_small.set_index(SMALL_CANDLE_COUNT - 1)

        def get_latest() -> pd.Series:
            return symbol_slice_small.latest()

        result = benchmark(get_latest)
        assert result is not None

    def test_latest_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: latest() (10K Candles)."""
        symbol_slice_medium.set_index(DEFAULT_CANDLE_COUNT - 1)

        def get_latest() -> pd.Series:
            return symbol_slice_medium.latest()

        result = benchmark(get_latest)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: History Access
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestHistoryAccessBenchmarks:
    """Benchmarks für History-Zugriff."""

    def test_history_small_window(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history() mit kleinem Fenster (20 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history() -> pd.DataFrame:
            return symbol_slice_medium.history(20)

        result = benchmark(get_history)
        assert len(result) == 20

    def test_history_medium_window(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history() mit mittlerem Fenster (200 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history() -> pd.DataFrame:
            return symbol_slice_medium.history(200)

        result = benchmark(get_history)
        assert len(result) == 200

    def test_history_large_window(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history() mit großem Fenster (1000 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history() -> pd.DataFrame:
            return symbol_slice_medium.history(1000)

        result = benchmark(get_history)
        assert len(result) == 1000

    def test_history_view_small(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history_view() (20 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history_view() -> pd.DataFrame:
            return symbol_slice_medium.history_view(20)

        result = benchmark(get_history_view)
        assert len(result) == 20

    def test_history_view_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history_view() (200 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history_view() -> pd.DataFrame:
            return symbol_slice_medium.history_view(200)

        result = benchmark(get_history_view)
        assert len(result) == 200

    def test_history_repeated_access(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: Wiederholter History-Zugriff."""
        symbol_slice_medium.set_index(5000)

        def repeated_history() -> int:
            count = 0
            for _ in range(100):
                _ = symbol_slice_medium.history(50)
                count += 1
            return count

        result = benchmark(repeated_history)
        assert result == 100


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Combined Backtest Pattern
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestBacktestPatternBenchmarks:
    """Benchmarks für typische Backtest-Muster."""

    def test_typical_indicator_calculation_pattern(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: Typisches Indikator-Berechnungsmuster."""

        def indicator_pattern() -> int:
            count = 0
            for i in range(100, min(1000, DEFAULT_CANDLE_COUNT)):
                symbol_slice_medium.set_index(i)
                # SMA-ähnlicher Zugriff
                _ = symbol_slice_medium.history(20)
                # RSI-ähnlicher Zugriff
                _ = symbol_slice_medium.history(14)
                # MACD-ähnlicher Zugriff
                _ = symbol_slice_medium.history(26)
                count += 1
            return count

        result = benchmark(indicator_pattern)
        assert result == 900

    def test_typical_signal_check_pattern(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: Typisches Signal-Check-Muster."""

        def signal_pattern() -> int:
            count = 0
            for i in range(100, min(500, DEFAULT_CANDLE_COUNT)):
                symbol_slice_medium.set_index(i)
                # Aktuellen Preis holen
                _ = symbol_slice_medium.latest()
                # Close-Serie für Vergleich
                _ = symbol_slice_medium.get("Close")
                count += 1
            return count

        result = benchmark(signal_pattern)
        assert result == 400

    def test_full_backtest_simulation(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: Volle Backtest-Simulation (vereinfacht)."""

        def backtest_sim() -> int:
            count = 0
            for i in range(200, min(1200, DEFAULT_CANDLE_COUNT)):
                symbol_slice_medium.set_index(i)
                # Candle-Daten
                latest = symbol_slice_medium.latest()
                # History für Indikator
                hist = symbol_slice_medium.history(20)
                # Simpler "Indikator"
                if len(hist) > 0:
                    sma = hist["Close"].mean()
                    if latest["Close"] > sma:
                        count += 1  # "Signal"
            return count

        result = benchmark(backtest_sim)
        assert result >= 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Multi-Timeframe Access
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestMultiTimeframeBenchmarks:
    """Benchmarks für Multi-Timeframe-Zugriff."""

    def test_multi_tf_slice_creation(
        self, benchmark: Any, multi_tf_data_medium: Dict[str, pd.DataFrame]
    ) -> None:
        """Benchmark: Multi-TF SymbolDataSlice-Erstellung."""

        def create_slices() -> int:
            slices = {}
            for tf, df in multi_tf_data_medium.items():
                slices[tf] = SymbolDataSlice(df)
            return len(slices)

        result = benchmark(create_slices)
        assert result == 5

    def test_multi_tf_access_pattern(
        self, benchmark: Any, multi_tf_data_medium: Dict[str, pd.DataFrame]
    ) -> None:
        """Benchmark: Multi-TF Zugriffsmuster."""
        slices = {tf: SymbolDataSlice(df) for tf, df in multi_tf_data_medium.items()}

        def multi_tf_access() -> int:
            count = 0
            for _ in range(100):
                for tf, sl in slices.items():
                    # Set to mid-point
                    mid = len(multi_tf_data_medium[tf]) // 2
                    sl.set_index(mid)
                    _ = sl.latest()
                    count += 1
            return count

        result = benchmark(multi_tf_access)
        assert result == 500


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Throughput Baselines
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="symbol_data_slicer")
class TestThroughputBaselines:
    """Throughput-Baselines für Rust-Vergleich."""

    def test_index_updates_per_second(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Baseline: Index-Updates pro Sekunde."""

        def many_updates() -> int:
            for i in range(10000):
                symbol_slice_medium.set_index(i % DEFAULT_CANDLE_COUNT)
            return 10000

        result = benchmark(many_updates)
        assert result == 10000

    def test_latest_calls_per_second(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Baseline: latest() Aufrufe pro Sekunde."""
        symbol_slice_medium.set_index(5000)

        def many_latest() -> int:
            count = 0
            for _ in range(5000):
                _ = symbol_slice_medium.latest()
                count += 1
            return count

        result = benchmark(many_latest)
        assert result == 5000

    def test_history_calls_per_second(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Baseline: history() Aufrufe pro Sekunde."""
        symbol_slice_medium.set_index(5000)

        def many_history() -> int:
            count = 0
            for _ in range(1000):
                _ = symbol_slice_medium.history(20)
                count += 1
            return count

        result = benchmark(many_history)
        assert result == 1000

    def test_combined_ops_per_second(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Baseline: Kombinierte Operationen pro Sekunde."""

        def combined_ops() -> int:
            count = 0
            for i in range(500):
                idx = (i * 17) % (DEFAULT_CANDLE_COUNT - 100)  # Pseudo-random
                symbol_slice_medium.set_index(idx + 50)
                _ = symbol_slice_medium.latest()
                _ = symbol_slice_medium.history(20)
                count += 1
            return count

        result = benchmark(combined_ops)
        assert result == 500
