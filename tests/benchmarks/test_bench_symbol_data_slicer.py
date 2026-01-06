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

from collections import deque
from typing import Any, Dict, List

import numpy as np
import pytest

from backtest_engine.core.symbol_data_slicer import CandleSet, SymbolDataSlice

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
    generate_multi_tf_candle_data,
)

TIMEFRAMES: List[str] = ["M1", "M5", "M15", "H1", "H4"]
PRIMARY_TF = "M5"
PRICE_TYPE = "bid"


def _make_tf_bid_ask(
    multi_tf_data: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> tuple[Dict[str, List[Dict[str, float]]], Dict[str, List[Dict[str, float]]]]:
    tf_bid = {tf: sides.get("bid", []) for tf, sides in multi_tf_data.items()}
    tf_ask = {tf: sides.get("ask", []) for tf, sides in multi_tf_data.items()}
    return tf_bid, tf_ask


@pytest.fixture(scope="session")
def multi_tf_candle_data_small() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    return generate_multi_tf_candle_data(
        SMALL_CANDLE_COUNT, seed=BENCHMARK_SEED, timeframes=TIMEFRAMES
    )


@pytest.fixture(scope="session")
def multi_tf_candle_data_medium() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    return generate_multi_tf_candle_data(
        DEFAULT_CANDLE_COUNT, seed=BENCHMARK_SEED, timeframes=TIMEFRAMES
    )


@pytest.fixture(scope="session")
def multi_tf_candle_data_large() -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    return generate_multi_tf_candle_data(
        LARGE_CANDLE_COUNT, seed=BENCHMARK_SEED, timeframes=TIMEFRAMES
    )


@pytest.fixture
def candle_set_small(
    multi_tf_candle_data_small: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> CandleSet:
    tf_bid, tf_ask = _make_tf_bid_ask(multi_tf_candle_data_small)
    return CandleSet(tf_bid_candles=tf_bid, tf_ask_candles=tf_ask)


@pytest.fixture
def candle_set_medium(
    multi_tf_candle_data_medium: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> CandleSet:
    tf_bid, tf_ask = _make_tf_bid_ask(multi_tf_candle_data_medium)
    return CandleSet(tf_bid_candles=tf_bid, tf_ask_candles=tf_ask)


@pytest.fixture
def candle_set_large(
    multi_tf_candle_data_large: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> CandleSet:
    tf_bid, tf_ask = _make_tf_bid_ask(multi_tf_candle_data_large)
    return CandleSet(tf_bid_candles=tf_bid, tf_ask_candles=tf_ask)


@pytest.fixture
def symbol_slice_small(
    multi_tf_candle_data_small: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> SymbolDataSlice:
    return SymbolDataSlice(multi_tf_candle_data_small, index=0)


@pytest.fixture
def symbol_slice_medium(
    multi_tf_candle_data_medium: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> SymbolDataSlice:
    return SymbolDataSlice(multi_tf_candle_data_medium, index=0)


@pytest.fixture
def symbol_slice_large(
    multi_tf_candle_data_large: Dict[str, Dict[str, List[Dict[str, float]]]],
) -> SymbolDataSlice:
    return SymbolDataSlice(multi_tf_candle_data_large, index=0)


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

        idx = SMALL_CANDLE_COUNT // 2

        def get_latest() -> Dict[str, float] | None:
            return candle_set_small.get_latest(PRIMARY_TF, idx, PRICE_TYPE)

        result = benchmark(get_latest)
        assert result is not None

    def test_get_latest_medium(
        self, benchmark: Any, candle_set_medium: CandleSet
    ) -> None:
        """Benchmark: get_latest() (10K Candles)."""

        idx = DEFAULT_CANDLE_COUNT // 2

        def get_latest() -> Dict[str, float] | None:
            return candle_set_medium.get_latest(PRIMARY_TF, idx, PRICE_TYPE)

        result = benchmark(get_latest)
        assert result is not None

    @pytest.mark.benchmark_slow
    def test_get_latest_large(
        self, benchmark: Any, candle_set_large: CandleSet
    ) -> None:
        """Benchmark: get_latest() (100K Candles)."""

        idx = LARGE_CANDLE_COUNT // 2

        def get_latest() -> Dict[str, float] | None:
            return candle_set_large.get_latest(PRIMARY_TF, idx, PRICE_TYPE)

        result = benchmark(get_latest)
        assert result is not None

    def test_get_all_small(self, benchmark: Any, candle_set_small: CandleSet) -> None:
        """Benchmark: get_all() (1K Candles)."""

        def get_all() -> List[Dict[str, float]]:
            return candle_set_small.get_all(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_all)
        assert len(result) == SMALL_CANDLE_COUNT

    def test_get_all_medium(self, benchmark: Any, candle_set_medium: CandleSet) -> None:
        """Benchmark: get_all() (10K Candles)."""

        def get_all() -> List[Dict[str, float]]:
            return candle_set_medium.get_all(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_all)
        assert len(result) == DEFAULT_CANDLE_COUNT

    def test_get_latest_repeated(
        self, benchmark: Any, candle_set_medium: CandleSet
    ) -> None:
        """Benchmark: Wiederholte get_latest() Aufrufe."""

        idx = DEFAULT_CANDLE_COUNT // 2

        def repeated_latest() -> int:
            count = 0
            for _ in range(1000):
                _ = candle_set_medium.get_latest(PRIMARY_TF, idx, PRICE_TYPE)
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
        """Benchmark: get() für ein Timeframe (1K Candles)."""
        symbol_slice_small.set_index(SMALL_CANDLE_COUNT - 1)

        def get_timeframe() -> List[Dict[str, float]]:
            return symbol_slice_small.get(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_timeframe)
        assert len(result) == SMALL_CANDLE_COUNT

    def test_get_column_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: get() für ein Timeframe (10K Candles)."""
        symbol_slice_medium.set_index(DEFAULT_CANDLE_COUNT - 1)

        def get_timeframe() -> List[Dict[str, float]]:
            return symbol_slice_medium.get(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_timeframe)
        assert len(result) == DEFAULT_CANDLE_COUNT

    def test_series_ref_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: series_ref() (1K Candles)."""
        symbol_slice_small.set_index(SMALL_CANDLE_COUNT - 1)

        def get_series_ref() -> List[Dict[str, float]]:
            return symbol_slice_small.series_ref(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_series_ref)
        assert len(result) == SMALL_CANDLE_COUNT

    def test_series_ref_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: series_ref() (10K Candles)."""
        symbol_slice_medium.set_index(DEFAULT_CANDLE_COUNT - 1)

        def get_series_ref() -> List[Dict[str, float]]:
            return symbol_slice_medium.series_ref(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_series_ref)
        assert len(result) == DEFAULT_CANDLE_COUNT

    def test_latest_small(
        self, benchmark: Any, symbol_slice_small: SymbolDataSlice
    ) -> None:
        """Benchmark: latest() (1K Candles)."""
        symbol_slice_small.set_index(SMALL_CANDLE_COUNT - 1)

        def get_latest() -> Dict[str, float] | None:
            return symbol_slice_small.latest(PRIMARY_TF, PRICE_TYPE)

        result = benchmark(get_latest)
        assert result is not None

    def test_latest_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: latest() (10K Candles)."""
        symbol_slice_medium.set_index(DEFAULT_CANDLE_COUNT - 1)

        def get_latest() -> Dict[str, float] | None:
            return symbol_slice_medium.latest(PRIMARY_TF, PRICE_TYPE)

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

        def get_history() -> List[Dict[str, float]]:
            return symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=20)

        result = benchmark(get_history)
        assert len(result) == 20

    def test_history_medium_window(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history() mit mittlerem Fenster (200 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history() -> List[Dict[str, float]]:
            return symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=200)

        result = benchmark(get_history)
        assert len(result) == 200

    def test_history_large_window(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history() mit großem Fenster (1000 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history() -> List[Dict[str, float]]:
            return symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=1000)

        result = benchmark(get_history)
        assert len(result) == 1000

    def test_history_view_small(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history_view() (20 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history_view() -> deque[Dict[str, float]]:
            return symbol_slice_medium.history_view(PRIMARY_TF, PRICE_TYPE, length=20)

        result = benchmark(get_history_view)
        assert len(result) == 20

    def test_history_view_medium(
        self, benchmark: Any, symbol_slice_medium: SymbolDataSlice
    ) -> None:
        """Benchmark: history_view() (200 Candles)."""
        symbol_slice_medium.set_index(5000)

        def get_history_view() -> deque[Dict[str, float]]:
            return symbol_slice_medium.history_view(PRIMARY_TF, PRICE_TYPE, length=200)

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
                _ = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=50)
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
                _ = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=20)
                # RSI-ähnlicher Zugriff
                _ = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=14)
                # MACD-ähnlicher Zugriff
                _ = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=26)
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
                _ = symbol_slice_medium.latest(PRIMARY_TF, PRICE_TYPE)
                # Timeframe-Serie für Vergleich
                _ = symbol_slice_medium.get(PRIMARY_TF, PRICE_TYPE)
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
                latest = symbol_slice_medium.latest(PRIMARY_TF, PRICE_TYPE)
                # History für Indikator
                hist = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=20)
                # Simpler "Indikator"
                if latest and hist:
                    sma = sum(c["close"] for c in hist) / float(len(hist))
                    if latest["close"] > sma:
                        count += 1
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
        self,
        benchmark: Any,
        multi_tf_candle_data_medium: Dict[str, Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """Benchmark: Multi-TF SymbolDataSlice-Erstellung."""

        def create_slices() -> int:
            slices: Dict[str, SymbolDataSlice] = {}
            for tf, sides in multi_tf_candle_data_medium.items():
                slices[tf] = SymbolDataSlice({tf: sides}, index=0)
            return len(slices)

        result = benchmark(create_slices)
        assert result == len(multi_tf_candle_data_medium)

    def test_multi_tf_access_pattern(
        self,
        benchmark: Any,
        multi_tf_candle_data_medium: Dict[str, Dict[str, List[Dict[str, float]]]],
    ) -> None:
        """Benchmark: Multi-TF Zugriffsmuster."""
        slices = {
            tf: SymbolDataSlice({tf: sides}, index=0)
            for tf, sides in multi_tf_candle_data_medium.items()
        }

        def multi_tf_access() -> int:
            count = 0
            for _ in range(100):
                for tf, sl in slices.items():
                    # Set to mid-point
                    mid = len(multi_tf_candle_data_medium[tf][PRICE_TYPE]) // 2
                    sl.set_index(mid)
                    _ = sl.latest(tf, PRICE_TYPE)
                    count += 1
            return count

        result = benchmark(multi_tf_access)
        assert result == 100 * len(multi_tf_candle_data_medium)


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
                _ = symbol_slice_medium.latest(PRIMARY_TF, PRICE_TYPE)
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
                _ = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=20)
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
                _ = symbol_slice_medium.latest(PRIMARY_TF, PRICE_TYPE)
                _ = symbol_slice_medium.history(PRIMARY_TF, PRICE_TYPE, length=20)
                count += 1
            return count

        result = benchmark(combined_ops)
        assert result == 500
