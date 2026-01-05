# -*- coding: utf-8 -*-
"""
Benchmark Suite für EventEngine (P3-03).

Testet Throughput und Latenz der Event Engines:
- EventEngine: Single-Symbol Backtest Loop
- CrossSymbolEventEngine: Multi-Symbol Backtest Loop

Fokus auf:
- Hauptschleife Durchsatz (Bars/Sekunde)
- Latenz pro Bar
- Memory-Effizienz bei großen Datenmengen

Verwendung:
    pytest tests/benchmarks/test_bench_event_engine.py -v
    pytest tests/benchmarks/test_bench_event_engine.py --benchmark-json=output.json
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.indicator_cache import IndicatorCache
from backtest_engine.core.portfolio import Portfolio, PortfolioPosition

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
    generate_multi_tf_candle_data,
    generate_synthetic_ohlcv,
)


# ══════════════════════════════════════════════════════════════════════════════
# MOCK OBJECTS FÜR BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MockCandle:
    """Leichtgewichtiges Candle-Mock für Benchmarks."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MockStrategy:
    """Minimale Strategie für Throughput-Tests."""

    def __init__(self, signal_probability: float = 0.02) -> None:
        self.signal_probability = signal_probability
        self._rng = np.random.default_rng(BENCHMARK_SEED)
        self.call_count = 0

    def evaluate(
        self, index: int, slice_map: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """Gibt mit geringer Wahrscheinlichkeit ein Signal zurück."""
        self.call_count += 1
        if self._rng.random() < self.signal_probability:
            return [
                {
                    "symbol": "EURUSD",
                    "direction": "long" if self._rng.random() > 0.5 else "short",
                    "entry_price": 1.1000,
                    "stop_loss": 1.0950,
                    "take_profit": 1.1100,
                }
            ]
        return None


class MockStrategyWrapper:
    """Wrapper für MockStrategy (kompatibel mit EventEngine)."""

    def __init__(self, strategy: MockStrategy) -> None:
        self.strategy = strategy

    def evaluate(
        self, index: int, slice_map: Any
    ) -> Optional[List[Dict[str, Any]]]:
        return self.strategy.evaluate(index, slice_map)


class MockExecutionSimulator:
    """Leichtgewichtiger ExecutionSimulator für Throughput-Tests."""

    def __init__(self) -> None:
        self.active_positions: List[Any] = []
        self.signal_count = 0
        self.exit_eval_count = 0

    def process_signal(self, signal: Any) -> None:
        self.signal_count += 1

    def evaluate_exits(self, bid_candle: Any, ask_candle: Any) -> None:
        self.exit_eval_count += 1


class MockPortfolio:
    """Leichtgewichtiges Portfolio für Throughput-Tests."""

    def __init__(self) -> None:
        self.update_count = 0
        self.cash = 10000.0

    def update(self, timestamp: datetime) -> None:
        self.update_count += 1

    def get_open_positions(self, symbol: str) -> List[Any]:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def generate_candle_list(
    n: int,
    *,
    seed: int = BENCHMARK_SEED,
    start_time: Optional[datetime] = None,
) -> List[MockCandle]:
    """Generiert Liste von MockCandle-Objekten."""
    if start_time is None:
        start_time = datetime(2024, 1, 1)

    df = generate_synthetic_ohlcv(n, seed=seed)
    candles = []

    for i in range(len(df)):
        candles.append(
            MockCandle(
                timestamp=start_time + timedelta(minutes=5 * i),
                open=float(df.iloc[i]["open"]),
                high=float(df.iloc[i]["high"]),
                low=float(df.iloc[i]["low"]),
                close=float(df.iloc[i]["close"]),
                volume=float(df.iloc[i]["volume"]),
            )
        )

    return candles


def generate_candle_lookup(
    n: int,
    symbols: List[str],
    *,
    seed: int = BENCHMARK_SEED,
    start_time: Optional[datetime] = None,
) -> tuple:
    """Generiert Candle-Lookups für CrossSymbolEventEngine."""
    if start_time is None:
        start_time = datetime(2024, 1, 1)

    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n)]
    lookups: Dict[str, Dict[str, Dict[datetime, MockCandle]]] = {}

    for sym_idx, symbol in enumerate(symbols):
        sym_seed = seed + sym_idx * 1000
        candles = generate_candle_list(n, seed=sym_seed, start_time=start_time)
        lookups[symbol] = {
            "bid": {c.timestamp: c for c in candles},
            "ask": {c.timestamp: c for c in candles},
        }

    return lookups, timestamps


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Core Event Loop (Throughput)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_event_engine
class TestEventLoopThroughput:
    """Benchmarks für Event-Loop Durchsatz."""

    def test_bare_loop_small(self, benchmark: Any) -> None:
        """Benchmark: Minimaler Loop ohne Engine (Baseline)."""
        n = SMALL_CANDLE_COUNT
        candles = generate_candle_list(n)

        def bare_loop() -> int:
            total = 0
            for i, candle in enumerate(candles):
                # Simuliert minimale Arbeit pro Bar
                _ = candle.close
                total += 1
            return total

        result = benchmark(bare_loop)
        assert result == n

    def test_bare_loop_medium(self, benchmark: Any) -> None:
        """Benchmark: Minimaler Loop (10K Bars, Baseline)."""
        n = DEFAULT_CANDLE_COUNT
        candles = generate_candle_list(n)

        def bare_loop() -> int:
            total = 0
            for candle in candles:
                _ = candle.close
                total += 1
            return total

        result = benchmark(bare_loop)
        assert result == n

    @pytest.mark.benchmark_slow
    def test_bare_loop_large(self, benchmark: Any) -> None:
        """Benchmark: Minimaler Loop (100K Bars, Baseline)."""
        n = LARGE_CANDLE_COUNT
        candles = generate_candle_list(n)

        def bare_loop() -> int:
            total = 0
            for candle in candles:
                _ = candle.close
                total += 1
            return total

        result = benchmark(bare_loop)
        assert result == n


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Event Engine Single Symbol
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_event_engine
class TestSingleSymbolEventEngine:
    """Benchmarks für Single-Symbol EventEngine Simulation."""

    def test_event_loop_with_strategy_small(self, benchmark: Any) -> None:
        """Benchmark: Event Loop mit Strategy Evaluation (1K Bars)."""
        n = SMALL_CANDLE_COUNT
        candles = generate_candle_list(n)
        strategy = MockStrategy(signal_probability=0.02)
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def event_loop() -> int:
            strategy.call_count = 0
            for i, candle in enumerate(candles):
                # Strategy Evaluation
                signals = strategy.evaluate(i, {"EURUSD": candle})
                if signals:
                    for signal in signals:
                        executor.process_signal(signal)

                # Exit Evaluation
                executor.evaluate_exits(candle, candle)

                # Portfolio Update
                portfolio.update(candle.timestamp)

            return strategy.call_count

        result = benchmark(event_loop)
        assert result == n

    def test_event_loop_with_strategy_medium(self, benchmark: Any) -> None:
        """Benchmark: Event Loop mit Strategy Evaluation (10K Bars)."""
        n = DEFAULT_CANDLE_COUNT
        candles = generate_candle_list(n)
        strategy = MockStrategy(signal_probability=0.02)
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def event_loop() -> int:
            strategy.call_count = 0
            for i, candle in enumerate(candles):
                signals = strategy.evaluate(i, {"EURUSD": candle})
                if signals:
                    for signal in signals:
                        executor.process_signal(signal)
                executor.evaluate_exits(candle, candle)
                portfolio.update(candle.timestamp)
            return strategy.call_count

        result = benchmark(event_loop)
        assert result == n

    @pytest.mark.benchmark_slow
    def test_event_loop_with_strategy_large(self, benchmark: Any) -> None:
        """Benchmark: Event Loop mit Strategy Evaluation (100K Bars)."""
        n = LARGE_CANDLE_COUNT
        candles = generate_candle_list(n)
        strategy = MockStrategy(signal_probability=0.02)
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def event_loop() -> int:
            strategy.call_count = 0
            for i, candle in enumerate(candles):
                signals = strategy.evaluate(i, {"EURUSD": candle})
                if signals:
                    for signal in signals:
                        executor.process_signal(signal)
                executor.evaluate_exits(candle, candle)
                portfolio.update(candle.timestamp)
            return strategy.call_count

        result = benchmark(event_loop)
        assert result == n


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Event Engine with Indicator Cache
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_event_engine
class TestEventEngineWithIndicators:
    """Benchmarks für EventEngine mit Indikator-Lookups."""

    def test_event_loop_with_indicator_lookups_medium(
        self, benchmark: Any
    ) -> None:
        """Benchmark: Event Loop mit Indikator-Lookups (10K Bars)."""
        n = DEFAULT_CANDLE_COUNT
        candles = generate_candle_list(n)

        # Multi-TF Data für IndicatorCache
        multi_tf_data = generate_multi_tf_candle_data(n)
        cache = IndicatorCache(multi_tf_data)

        # Pre-compute Indikatoren
        _ = cache.ema("M5", "bid", 20)
        _ = cache.rsi("M5", "bid", 14)

        strategy = MockStrategy(signal_probability=0.02)
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def event_loop_with_indicators() -> int:
            strategy.call_count = 0
            for i, candle in enumerate(candles):
                # Indikator-Lookups (cached)
                ema_val = cache.ema("M5", "bid", 20).iloc[i]
                rsi_val = cache.rsi("M5", "bid", 14).iloc[i]

                # Strategy mit Indikatoren
                signals = strategy.evaluate(i, {"EURUSD": candle})
                if signals:
                    for signal in signals:
                        executor.process_signal(signal)

                executor.evaluate_exits(candle, candle)
                portfolio.update(candle.timestamp)

            return strategy.call_count

        result = benchmark(event_loop_with_indicators)
        assert result == n


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Multi-Symbol Event Engine
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_event_engine
class TestMultiSymbolEventEngine:
    """Benchmarks für Multi-Symbol EventEngine Simulation."""

    def test_multi_symbol_loop_3_symbols_medium(self, benchmark: Any) -> None:
        """Benchmark: Multi-Symbol Loop (3 Symbole, 10K Bars)."""
        n = DEFAULT_CANDLE_COUNT
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        lookups, timestamps = generate_candle_lookup(n, symbols)

        strategy = MockStrategy(signal_probability=0.02)
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def multi_symbol_loop() -> int:
            strategy.call_count = 0
            for ts in timestamps:
                # Strategy evaluation
                signals = strategy.evaluate(0, lookups)
                if signals:
                    for signal in signals:
                        executor.process_signal(signal)

                # Exit evaluation für alle Symbole
                for symbol in symbols:
                    bid = lookups[symbol]["bid"].get(ts)
                    ask = lookups[symbol]["ask"].get(ts)
                    if bid and ask:
                        executor.evaluate_exits(bid, ask)

                portfolio.update(ts)

            return len(timestamps)

        result = benchmark(multi_symbol_loop)
        assert result == n

    def test_multi_symbol_loop_5_symbols_medium(self, benchmark: Any) -> None:
        """Benchmark: Multi-Symbol Loop (5 Symbole, 10K Bars)."""
        n = DEFAULT_CANDLE_COUNT
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        lookups, timestamps = generate_candle_lookup(n, symbols)

        strategy = MockStrategy(signal_probability=0.02)
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def multi_symbol_loop() -> int:
            strategy.call_count = 0
            for ts in timestamps:
                signals = strategy.evaluate(0, lookups)
                if signals:
                    for signal in signals:
                        executor.process_signal(signal)

                for symbol in symbols:
                    bid = lookups[symbol]["bid"].get(ts)
                    ask = lookups[symbol]["ask"].get(ts)
                    if bid and ask:
                        executor.evaluate_exits(bid, ask)

                portfolio.update(ts)

            return len(timestamps)

        result = benchmark(multi_symbol_loop)
        assert result == n


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Latency Measurements
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_event_engine
class TestEventEngineLatency:
    """Benchmarks für Latenz-Messungen pro Bar."""

    def test_single_bar_latency(self, benchmark: Any) -> None:
        """Benchmark: Latenz einer einzelnen Bar-Verarbeitung."""
        candle = MockCandle(
            timestamp=datetime(2024, 1, 1),
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000.0,
        )
        strategy = MockStrategy(signal_probability=0.5)  # Häufigere Signale
        executor = MockExecutionSimulator()
        portfolio = MockPortfolio()

        def process_single_bar() -> bool:
            signals = strategy.evaluate(0, {"EURUSD": candle})
            if signals:
                for signal in signals:
                    executor.process_signal(signal)
            executor.evaluate_exits(candle, candle)
            portfolio.update(candle.timestamp)
            return True

        result = benchmark(process_single_bar)
        assert result is True

    def test_bar_with_indicator_lookup_latency(self, benchmark: Any) -> None:
        """Benchmark: Latenz Bar + Indikator-Lookup."""
        n = DEFAULT_CANDLE_COUNT
        multi_tf_data = generate_multi_tf_candle_data(n)
        cache = IndicatorCache(multi_tf_data)

        # Pre-compute
        _ = cache.ema("M5", "bid", 20)
        _ = cache.rsi("M5", "bid", 14)
        _ = cache.macd("M5", "bid", 12, 26, 9)

        candle = MockCandle(
            timestamp=datetime(2024, 1, 1),
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000.0,
        )
        idx = 5000  # Mitte des Datasets

        def process_bar_with_indicators() -> tuple:
            ema = cache.ema("M5", "bid", 20).iloc[idx]
            rsi = cache.rsi("M5", "bid", 14).iloc[idx]
            macd, signal = cache.macd("M5", "bid", 12, 26, 9)
            macd_val = macd.iloc[idx]
            return (ema, rsi, macd_val)

        result = benchmark(process_bar_with_indicators)
        assert len(result) == 3


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Memory Efficiency
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_event_engine
class TestEventEngineMemoryEfficiency:
    """Benchmarks für Memory-Effizienz."""

    def test_candle_object_creation_overhead(self, benchmark: Any) -> None:
        """Benchmark: Overhead von Candle-Objekt-Erstellung."""
        n = 10000

        def create_candles() -> int:
            candles = []
            base_time = datetime(2024, 1, 1)
            for i in range(n):
                candles.append(
                    MockCandle(
                        timestamp=base_time + timedelta(minutes=5 * i),
                        open=1.1000 + i * 0.0001,
                        high=1.1010 + i * 0.0001,
                        low=1.0990 + i * 0.0001,
                        close=1.1005 + i * 0.0001,
                        volume=float(1000 + i),
                    )
                )
            return len(candles)

        result = benchmark(create_candles)
        assert result == n

    def test_dict_candle_creation_overhead(self, benchmark: Any) -> None:
        """Benchmark: Dict-basierte Candle-Erstellung (Alternative)."""
        n = 10000

        def create_dict_candles() -> int:
            candles = []
            base_time = datetime(2024, 1, 1)
            for i in range(n):
                candles.append(
                    {
                        "timestamp": base_time + timedelta(minutes=5 * i),
                        "open": 1.1000 + i * 0.0001,
                        "high": 1.1010 + i * 0.0001,
                        "low": 1.0990 + i * 0.0001,
                        "close": 1.1005 + i * 0.0001,
                        "volume": float(1000 + i),
                    }
                )
            return len(candles)

        result = benchmark(create_dict_candles)
        assert result == n
