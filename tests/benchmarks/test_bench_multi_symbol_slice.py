# -*- coding: utf-8 -*-
"""
Benchmark Suite für MultiSymbolSlice (P6-07).

Testet alle public functions des MultiSymbolSlice-Moduls:
- Multi-Symbol Lookups
- Timestamp-basierte Operationen
- SliceView Lookups
- Iteration über Symbole

Ergebnisse sind in JSON exportierbar für Regression-Detection.

Verwendung:
    pytest tests/benchmarks/test_bench_multi_symbol_slice.py -v
    pytest tests/benchmarks/test_bench_multi_symbol_slice.py --benchmark-json=output.json
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from backtest_engine.core.multi_symbol_slice import MultiSymbolSlice, SliceView
from backtest_engine.data.candle import Candle

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════


def generate_symbol_dataframes(
    symbols: List[str], candles_per_symbol: int, seed: int = BENCHMARK_SEED
) -> Dict[str, Dict[str, Dict[datetime, Candle]]]:
    """Generiert Candle-Lookups (timestamp -> Candle) für mehrere Symbole.

    MultiSymbolSlice arbeitet timestamp-basiert und liefert pro Symbol/price_type
    genau eine Candle für den aktuellen Timestamp.
    """
    rng = np.random.default_rng(seed)
    data: Dict[str, Dict[str, Dict[datetime, Candle]]] = {}

    for symbol in symbols:
        base_price = float(
            rng.uniform(1.0, 2.0) if "USD" in symbol else rng.uniform(100.0, 200.0)
        )
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        opens: List[float] = [base_price]
        for _ in range(int(candles_per_symbol) - 1):
            change = float(rng.normal(0, 0.0001 * base_price))
            opens.append(opens[-1] + change)

        bid_map: Dict[datetime, Candle] = {}
        ask_map: Dict[datetime, Candle] = {}
        for i in range(int(candles_per_symbol)):
            ts = base_time + timedelta(hours=i)
            o = float(opens[i])
            hi = o * float(1 + rng.uniform(0.0001, 0.001))
            lo = o * float(1 - rng.uniform(0.0001, 0.001))
            cl = lo + float(rng.uniform(0.3, 0.7)) * (hi - lo)
            vol = float(rng.integers(100, 10000))
            bid = Candle(timestamp=ts, open=o, high=hi, low=lo, close=cl, volume=vol, candle_type="bid")
            # Minimaler Spread für Ask
            ask = Candle(
                timestamp=ts,
                open=o * 1.00001,
                high=hi * 1.00001,
                low=lo * 1.00001,
                close=cl * 1.00001,
                volume=vol,
                candle_type="ask",
            )
            bid_map[ts] = bid
            ask_map[ts] = ask

        data[symbol] = {"bid": bid_map, "ask": ask_map}

    return data


def create_multi_symbol_slice(
    symbols: List[str], candles: int, seed: int = BENCHMARK_SEED
) -> MultiSymbolSlice:
    """Erstellt eine MultiSymbolSlice mit synthetischen Daten."""
    candle_lookups = generate_symbol_dataframes(symbols, candles, seed)
    start_ts = datetime(2024, 1, 1, 0, 0, 0)
    return MultiSymbolSlice(candle_lookups=candle_lookups, timestamp=start_ts, primary_tf="H1")


def generate_timestamps(count: int) -> List[datetime]:
    base = datetime(2024, 1, 1, 0, 0, 0)
    return [base + timedelta(hours=i) for i in range(int(count))]


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

SYMBOLS_SMALL = ["EURUSD", "GBPUSD", "USDJPY"]
SYMBOLS_MEDIUM = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "USDCHF",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
]
SYMBOLS_LARGE = SYMBOLS_MEDIUM + [
    "AUDCAD",
    "AUDCHF",
    "AUDJPY",
    "AUDNZD",
    "CADCHF",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "EURCHF",
]


@pytest.fixture
def slice_small() -> MultiSymbolSlice:
    """MultiSymbolSlice mit 3 Symbolen, 1K Candles."""
    return create_multi_symbol_slice(SYMBOLS_SMALL, SMALL_CANDLE_COUNT)


@pytest.fixture
def slice_medium() -> MultiSymbolSlice:
    """MultiSymbolSlice mit 10 Symbolen, 10K Candles."""
    return create_multi_symbol_slice(SYMBOLS_MEDIUM, DEFAULT_CANDLE_COUNT)


@pytest.fixture
def slice_large() -> MultiSymbolSlice:
    """MultiSymbolSlice mit 20 Symbolen, 100K Candles."""
    return create_multi_symbol_slice(SYMBOLS_LARGE, LARGE_CANDLE_COUNT)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Symbol Lookups
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="multi_symbol_slice")
class TestSymbolLookupBenchmarks:
    """Benchmarks für Symbol-Lookups."""

    def test_get_single_symbol_small(
        self, benchmark: Any, slice_small: MultiSymbolSlice
    ) -> None:
        """Benchmark: Single Symbol Lookup (1K Candles)."""

        def get_symbol() -> Optional[Candle]:
            return slice_small.get("EURUSD", price_type="bid")

        result = benchmark(get_symbol)
        assert result is not None
        assert isinstance(result, Candle)

    def test_get_single_symbol_medium(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Single Symbol Lookup (10K Candles)."""

        def get_symbol() -> Optional[Candle]:
            return slice_medium.get("EURUSD", price_type="bid")

        result = benchmark(get_symbol)
        assert result is not None
        assert isinstance(result, Candle)

    @pytest.mark.benchmark_slow
    def test_get_single_symbol_large(
        self, benchmark: Any, slice_large: MultiSymbolSlice
    ) -> None:
        """Benchmark: Single Symbol Lookup (100K Candles)."""

        def get_symbol() -> Optional[Candle]:
            return slice_large.get("EURUSD", price_type="bid")

        result = benchmark(get_symbol)
        assert result is not None
        assert isinstance(result, Candle)

    def test_get_all_symbols_small(
        self, benchmark: Any, slice_small: MultiSymbolSlice
    ) -> None:
        """Benchmark: Alle Symbole abfragen (3 Symbole)."""

        def get_all() -> List[Optional[Candle]]:
            return [slice_small.get(s, price_type="bid") for s in SYMBOLS_SMALL]

        result = benchmark(get_all)
        assert len(result) == 3

    def test_get_all_symbols_medium(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Alle Symbole abfragen (10 Symbole)."""

        def get_all() -> List[Optional[Candle]]:
            return [slice_medium.get(s, price_type="bid") for s in SYMBOLS_MEDIUM]

        result = benchmark(get_all)
        assert len(result) == 10

    def test_get_nonexistent_symbol(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Nicht-existentes Symbol abfragen."""

        def get_missing() -> Optional[Candle]:
            return slice_medium.get("XAUUSD", price_type="bid")

        result = benchmark(get_missing)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Iteration
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="multi_symbol_slice")
class TestIterationBenchmarks:
    """Benchmarks für Iteration über Symbole."""

    def test_keys_small(self, benchmark: Any, slice_small: MultiSymbolSlice) -> None:
        """Benchmark: Keys-Iteration (3 Symbole)."""

        def get_keys() -> List[str]:
            return list(slice_small.keys())

        result = benchmark(get_keys)
        assert len(result) == 3

    def test_keys_medium(self, benchmark: Any, slice_medium: MultiSymbolSlice) -> None:
        """Benchmark: Keys-Iteration (10 Symbole)."""

        def get_keys() -> List[str]:
            return list(slice_medium.keys())

        result = benchmark(get_keys)
        assert len(result) == 10

    def test_iter_small(self, benchmark: Any, slice_small: MultiSymbolSlice) -> None:
        """Benchmark: __iter__ (3 Symbole)."""

        def iterate() -> int:
            count = 0
            for symbol in slice_small:
                count += 1
            return count

        result = benchmark(iterate)
        assert result == 3

    def test_iter_medium(self, benchmark: Any, slice_medium: MultiSymbolSlice) -> None:
        """Benchmark: __iter__ (10 Symbole)."""

        def iterate() -> int:
            count = 0
            for symbol in slice_medium:
                count += 1
            return count

        result = benchmark(iterate)
        assert result == 10

    def test_full_iteration_with_data(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Vollständige Iteration mit Daten-Zugriff."""

        def iterate_with_data() -> int:
            total = 0
            for symbol in slice_medium:
                c = slice_medium.get(symbol, price_type="bid")
                if c is not None:
                    total += 1
            return total

        result = benchmark(iterate_with_data)
        assert result == 10


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Timestamp Operations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="multi_symbol_slice")
class TestTimestampBenchmarks:
    """Benchmarks für Timestamp-Operationen."""

    def test_set_timestamp_small(
        self, benchmark: Any, slice_small: MultiSymbolSlice
    ) -> None:
        """Benchmark: Timestamp setzen (1K Candles, sequentiell)."""
        timestamps = generate_timestamps(SMALL_CANDLE_COUNT)

        def set_timestamps() -> int:
            for ts in timestamps:
                slice_small.set_timestamp(ts)
            return SMALL_CANDLE_COUNT

        result = benchmark(set_timestamps)
        assert result == SMALL_CANDLE_COUNT

    def test_set_timestamp_medium(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Timestamp setzen (10K Candles, sequentiell)."""
        timestamps = generate_timestamps(DEFAULT_CANDLE_COUNT)

        def set_timestamps() -> int:
            for ts in timestamps:
                slice_medium.set_timestamp(ts)
            return DEFAULT_CANDLE_COUNT

        result = benchmark(set_timestamps)
        assert result == DEFAULT_CANDLE_COUNT

    def test_timestamp_random_access(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Random Timestamp-Zugriff."""
        rng = np.random.default_rng(BENCHMARK_SEED)
        timestamps = generate_timestamps(DEFAULT_CANDLE_COUNT)
        random_indices = rng.choice(len(timestamps), size=1000, replace=True)
        random_timestamps = [timestamps[int(i)] for i in random_indices]

        def random_access() -> int:
            for ts in random_timestamps:
                slice_medium.set_timestamp(ts)
            return len(random_timestamps)

        result = benchmark(random_access)
        assert result == 1000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: SliceView Operations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="multi_symbol_slice")
class TestSliceViewBenchmarks:
    """Benchmarks für SliceView-Operationen."""

    def test_slice_view_creation(self, benchmark: Any) -> None:
        """Benchmark: SliceView-Erstellung."""
        data = generate_symbol_dataframes(SYMBOLS_MEDIUM, DEFAULT_CANDLE_COUNT)
        ts = datetime(2024, 1, 1, 0, 0, 0)

        def create_views() -> int:
            views = []
            for symbol, maps in data.items():
                view = SliceView(maps["bid"].get(ts))
                views.append(view)
            return len(views)

        result = benchmark(create_views)
        assert result == 10

    def test_slice_view_latest_small(self, benchmark: Any) -> None:
        """Benchmark: SliceView.latest() (1K Candles)."""
        data = generate_symbol_dataframes(SYMBOLS_SMALL, SMALL_CANDLE_COUNT)
        ts = datetime(2024, 1, 1, 0, 0, 0)
        views = {symbol: SliceView(maps["bid"].get(ts)) for symbol, maps in data.items()}

        def get_all_latest() -> int:
            count = 0
            for _ in range(100):
                for view in views.values():
                    _ = view.latest()
                    count += 1
            return count

        result = benchmark(get_all_latest)
        assert result == 300

    def test_slice_view_latest_medium(self, benchmark: Any) -> None:
        """Benchmark: SliceView.latest() (10K Candles)."""
        data = generate_symbol_dataframes(SYMBOLS_MEDIUM, DEFAULT_CANDLE_COUNT)
        ts = datetime(2024, 1, 1, 0, 0, 0)
        views = {symbol: SliceView(maps["bid"].get(ts)) for symbol, maps in data.items()}

        def get_all_latest() -> int:
            count = 0
            for _ in range(100):
                for view in views.values():
                    _ = view.latest()
                    count += 1
            return count

        result = benchmark(get_all_latest)
        assert result == 1000

    def test_slice_view_concurrent_access(self, benchmark: Any) -> None:
        """Benchmark: Concurrent SliceView-Zugriffe."""
        data = generate_symbol_dataframes(SYMBOLS_MEDIUM, DEFAULT_CANDLE_COUNT)
        ts = datetime(2024, 1, 1, 0, 0, 0)
        views = {symbol: SliceView(maps["bid"].get(ts)) for symbol, maps in data.items()}

        def concurrent_access() -> int:
            count = 0
            for _ in range(50):
                for view in views.values():
                    _ = view.latest()
                    count += 1
            return count

        result = benchmark(concurrent_access)
        assert result == 500


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Combined Operations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="multi_symbol_slice")
class TestCombinedOperationsBenchmarks:
    """Benchmarks für kombinierte Operationen."""

    def test_typical_backtest_access_pattern(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Typisches Backtest-Zugriffsmuster."""
        timestamps = generate_timestamps(1000)

        def backtest_pattern() -> int:
            count = 0
            for ts in timestamps:
                slice_medium.set_timestamp(ts)
                for symbol in SYMBOLS_MEDIUM[:3]:  # Nur primäre Symbole
                    c = slice_medium.get(symbol, price_type="bid")
                    if c is not None:
                        count += 1
            return count

        result = benchmark(backtest_pattern)
        assert result == 3000

    def test_multi_symbol_correlation_check(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Benchmark: Multi-Symbol-Korrelationsprüfung."""

        def correlation_check() -> int:
            pairs_checked = 0
            for i, sym1 in enumerate(SYMBOLS_MEDIUM[:-1]):
                for sym2 in SYMBOLS_MEDIUM[i + 1 :]:
                    c1 = slice_medium.get(sym1, price_type="bid")
                    c2 = slice_medium.get(sym2, price_type="bid")
                    if c1 is not None and c2 is not None:
                        pairs_checked += 1
            return pairs_checked

        result = benchmark(correlation_check)
        # n*(n-1)/2 = 10*9/2 = 45 pairs
        assert result == 45


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Throughput Baselines
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="multi_symbol_slice")
class TestThroughputBaselines:
    """Throughput-Baselines für Rust-Vergleich."""

    def test_lookups_per_second(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Baseline: Lookups pro Sekunde."""

        def many_lookups() -> int:
            count = 0
            for _ in range(1000):
                for symbol in SYMBOLS_MEDIUM:
                    _ = slice_medium.get(symbol, price_type="bid")
                    count += 1
            return count

        result = benchmark(many_lookups)
        assert result == 10000

    def test_timestamp_updates_per_second(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Baseline: Timestamp-Updates pro Sekunde."""
        timestamps = generate_timestamps(5000)

        def many_updates() -> int:
            for ts in timestamps:
                slice_medium.set_timestamp(ts)
            return len(timestamps)

        result = benchmark(many_updates)
        assert result == 5000

    def test_full_access_cycle(
        self, benchmark: Any, slice_medium: MultiSymbolSlice
    ) -> None:
        """Baseline: Vollständiger Zugriffs-Zyklus."""
        timestamps = generate_timestamps(500)

        def full_cycle() -> int:
            ops = 0
            for ts in timestamps:
                slice_medium.set_timestamp(ts)
                ops += 1
                for symbol in SYMBOLS_MEDIUM:
                    c = slice_medium.get(symbol, price_type="bid")
                    if c is not None:
                        _ = c.close
                        ops += 1
            return ops

        result = benchmark(full_cycle)
        assert result == 500 + 500 * 10
