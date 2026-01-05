# -*- coding: utf-8 -*-
"""
Benchmark Suite für IndicatorCache (P3-02).

Testet alle public functions des IndicatorCache-Moduls:
- DataFrame-Erstellung und Caching
- EMA, SMA, RSI, MACD, Bollinger, ATR, DMI, ROC
- Stepwise-Varianten für HTF-Bars

Ergebnisse sind in JSON exportierbar für Regression-Detection.

Verwendung:
    pytest tests/benchmarks/test_bench_indicator_cache.py -v
    pytest tests/benchmarks/test_bench_indicator_cache.py --benchmark-json=output.json
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from backtest_engine.core.indicator_cache import IndicatorCache

from .conftest import (
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
    generate_multi_tf_candle_data,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES (lokale Erzeugung für Isolation)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def indicator_cache_small() -> IndicatorCache:
    """Frischer IndicatorCache mit 1K Kerzen."""
    data = generate_multi_tf_candle_data(SMALL_CANDLE_COUNT)
    return IndicatorCache(data)


@pytest.fixture
def indicator_cache_medium() -> IndicatorCache:
    """Frischer IndicatorCache mit 10K Kerzen."""
    data = generate_multi_tf_candle_data(DEFAULT_CANDLE_COUNT)
    return IndicatorCache(data)


@pytest.fixture
def indicator_cache_large() -> IndicatorCache:
    """Frischer IndicatorCache mit 100K Kerzen."""
    data = generate_multi_tf_candle_data(LARGE_CANDLE_COUNT)
    return IndicatorCache(data)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: DataFrame Creation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestIndicatorCacheCreation:
    """Benchmarks für IndicatorCache Instanziierung und DataFrame-Build."""

    def test_cache_creation_small(self, benchmark: Any) -> None:
        """Benchmark: Cache-Erstellung mit 1K Kerzen."""
        data = generate_multi_tf_candle_data(SMALL_CANDLE_COUNT)

        def create_cache() -> IndicatorCache:
            return IndicatorCache(data)

        result = benchmark(create_cache)
        assert result is not None

    def test_cache_creation_medium(self, benchmark: Any) -> None:
        """Benchmark: Cache-Erstellung mit 10K Kerzen."""
        data = generate_multi_tf_candle_data(DEFAULT_CANDLE_COUNT)

        def create_cache() -> IndicatorCache:
            return IndicatorCache(data)

        result = benchmark(create_cache)
        assert result is not None

    @pytest.mark.benchmark_slow
    def test_cache_creation_large(self, benchmark: Any) -> None:
        """Benchmark: Cache-Erstellung mit 100K Kerzen."""
        data = generate_multi_tf_candle_data(LARGE_CANDLE_COUNT)

        def create_cache() -> IndicatorCache:
            return IndicatorCache(data)

        result = benchmark(create_cache)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: EMA (Exponential Moving Average)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestEMABenchmarks:
    """Benchmarks für EMA-Berechnungen."""

    @pytest.mark.parametrize("period", [12, 26, 50, 200])
    def test_ema_small(
        self, benchmark: Any, indicator_cache_small: IndicatorCache, period: int
    ) -> None:
        """Benchmark: EMA mit verschiedenen Perioden (1K Kerzen)."""
        cache = indicator_cache_small

        def compute_ema() -> pd.Series:
            # Cache leeren für fairen Test
            cache._ind_cache.clear()
            return cache.ema("M5", "bid", period)

        result = benchmark(compute_ema)
        assert len(result) == SMALL_CANDLE_COUNT

    def test_ema_medium_period_50(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: EMA(50) mit 10K Kerzen."""
        cache = indicator_cache_medium

        def compute_ema() -> pd.Series:
            cache._ind_cache.clear()
            return cache.ema("M5", "bid", 50)

        result = benchmark(compute_ema)
        assert len(result) == DEFAULT_CANDLE_COUNT

    @pytest.mark.benchmark_slow
    def test_ema_large_period_200(
        self, benchmark: Any, indicator_cache_large: IndicatorCache
    ) -> None:
        """Benchmark: EMA(200) mit 100K Kerzen."""
        cache = indicator_cache_large

        def compute_ema() -> pd.Series:
            cache._ind_cache.clear()
            return cache.ema("M5", "bid", 200)

        result = benchmark(compute_ema)
        assert len(result) == LARGE_CANDLE_COUNT

    def test_ema_cached_retrieval(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: Cached EMA Retrieval (Cache-Hit Szenario)."""
        cache = indicator_cache_medium
        # Erst berechnen, dann gecached abrufen
        _ = cache.ema("M5", "bid", 50)

        def retrieve_cached() -> pd.Series:
            return cache.ema("M5", "bid", 50)

        result = benchmark(retrieve_cached)
        assert len(result) == DEFAULT_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: EMA Stepwise (HTF-Bar Handling)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestEMAStepwiseBenchmarks:
    """Benchmarks für stepwise EMA (HTF-Bar Variante)."""

    def test_ema_stepwise_medium(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: EMA Stepwise mit 10K Kerzen."""
        cache = indicator_cache_medium

        def compute() -> pd.Series:
            cache._ind_cache.clear()
            return cache.ema_stepwise("H1", "bid", 20)

        result = benchmark(compute)
        assert len(result) == DEFAULT_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: SMA (Simple Moving Average)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestSMABenchmarks:
    """Benchmarks für SMA-Berechnungen."""

    @pytest.mark.parametrize("period", [20, 50, 200])
    def test_sma_medium(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache, period: int
    ) -> None:
        """Benchmark: SMA mit verschiedenen Perioden (10K Kerzen)."""
        cache = indicator_cache_medium

        def compute() -> pd.Series:
            cache._ind_cache.clear()
            return cache.sma("M5", "bid", period)

        result = benchmark(compute)
        assert len(result) == DEFAULT_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: RSI (Relative Strength Index)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestRSIBenchmarks:
    """Benchmarks für RSI-Berechnungen."""

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_rsi_medium(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache, period: int
    ) -> None:
        """Benchmark: RSI mit verschiedenen Perioden (10K Kerzen)."""
        cache = indicator_cache_medium

        def compute() -> pd.Series:
            cache._ind_cache.clear()
            return cache.rsi("M5", "bid", period)

        result = benchmark(compute)
        assert len(result) == DEFAULT_CANDLE_COUNT

    @pytest.mark.benchmark_slow
    def test_rsi_large(
        self, benchmark: Any, indicator_cache_large: IndicatorCache
    ) -> None:
        """Benchmark: RSI(14) mit 100K Kerzen."""
        cache = indicator_cache_large

        def compute() -> pd.Series:
            cache._ind_cache.clear()
            return cache.rsi("M5", "bid", 14)

        result = benchmark(compute)
        assert len(result) == LARGE_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: MACD
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestMACDBenchmarks:
    """Benchmarks für MACD-Berechnungen."""

    def test_macd_default_medium(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: MACD(12,26,9) mit 10K Kerzen."""
        cache = indicator_cache_medium

        def compute() -> tuple:
            cache._ind_cache.clear()
            return cache.macd("M5", "bid", 12, 26, 9)

        result = benchmark(compute)
        macd_line, signal_line = result
        assert len(macd_line) == DEFAULT_CANDLE_COUNT
        assert len(signal_line) == DEFAULT_CANDLE_COUNT

    def test_macd_custom_params(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: MACD(8,21,5) mit 10K Kerzen."""
        cache = indicator_cache_medium

        def compute() -> tuple:
            cache._ind_cache.clear()
            return cache.macd("M5", "bid", 8, 21, 5)

        result = benchmark(compute)
        assert len(result[0]) == DEFAULT_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: ROC (Rate of Change)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestROCBenchmarks:
    """Benchmarks für ROC-Berechnungen."""

    @pytest.mark.parametrize("period", [10, 14, 20])
    def test_roc_medium(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache, period: int
    ) -> None:
        """Benchmark: ROC mit verschiedenen Perioden (10K Kerzen)."""
        cache = indicator_cache_medium

        def compute() -> pd.Series:
            cache._ind_cache.clear()
            return cache.roc("M5", "bid", period)

        result = benchmark(compute)
        assert len(result) == DEFAULT_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: DMI (Directional Movement Index)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestDMIBenchmarks:
    """Benchmarks für DMI-Berechnungen (+DI, -DI, ADX)."""

    def test_dmi_medium(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: DMI(14) mit 10K Kerzen."""
        cache = indicator_cache_medium

        def compute() -> tuple:
            cache._ind_cache.clear()
            return cache.dmi("M5", "bid", 14)

        result = benchmark(compute)
        plus_di, minus_di, adx = result
        assert len(plus_di) == DEFAULT_CANDLE_COUNT
        assert len(minus_di) == DEFAULT_CANDLE_COUNT
        assert len(adx) == DEFAULT_CANDLE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Combined Indicator Computation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestCombinedIndicatorBenchmarks:
    """Benchmarks für typische Strategie-Kombinationen."""

    def test_typical_strategy_indicators(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: Typische Strategie (EMA12, EMA26, RSI14, MACD)."""
        cache = indicator_cache_medium

        def compute_all() -> Dict[str, Any]:
            cache._ind_cache.clear()
            return {
                "ema_fast": cache.ema("M5", "bid", 12),
                "ema_slow": cache.ema("M5", "bid", 26),
                "rsi": cache.rsi("M5", "bid", 14),
                "macd": cache.macd("M5", "bid", 12, 26, 9),
            }

        result = benchmark(compute_all)
        assert "ema_fast" in result
        assert "rsi" in result

    def test_multi_tf_strategy_indicators(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: Multi-TF Strategie (M5 + H1 Indikatoren)."""
        cache = indicator_cache_medium

        def compute_multi_tf() -> Dict[str, Any]:
            cache._ind_cache.clear()
            return {
                "m5_ema_20": cache.ema("M5", "bid", 20),
                "m5_rsi_14": cache.rsi("M5", "bid", 14),
                "h1_ema_50": cache.ema("H1", "bid", 50),
                "h1_ema_stepwise": cache.ema_stepwise("H1", "bid", 20),
            }

        result = benchmark(compute_multi_tf)
        assert "m5_ema_20" in result
        assert "h1_ema_50" in result


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Cache Efficiency
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark_indicator
class TestCacheEfficiencyBenchmarks:
    """Benchmarks für Cache-Effizienz."""

    def test_repeated_access_pattern(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: Wiederholter Zugriff (Strategie-Loop Simulation)."""
        cache = indicator_cache_medium
        # Pre-compute alle Indikatoren
        _ = cache.ema("M5", "bid", 20)
        _ = cache.rsi("M5", "bid", 14)
        _ = cache.macd("M5", "bid", 12, 26, 9)

        def simulate_strategy_loop() -> int:
            total = 0
            # Simuliert 1000 Bar-Iterationen mit Cache-Lookups
            for _ in range(1000):
                ema = cache.ema("M5", "bid", 20)
                rsi = cache.rsi("M5", "bid", 14)
                macd, signal = cache.macd("M5", "bid", 12, 26, 9)
                total += 1
            return total

        result = benchmark(simulate_strategy_loop)
        assert result == 1000

    def test_df_access_vs_series(
        self, benchmark: Any, indicator_cache_medium: IndicatorCache
    ) -> None:
        """Benchmark: DataFrame vs. einzelne Series Zugriffe."""
        cache = indicator_cache_medium

        def access_df() -> float:
            df = cache.get_df("M5", "bid")
            return float(df["close"].iloc[-1])

        result = benchmark(access_df)
        assert isinstance(result, float)
