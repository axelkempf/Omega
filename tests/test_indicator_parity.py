"""Rust↔Python Indicator Parity Tests.

Verifies that IndicatorCache produces identical results whether using
Python or Rust backend. This is critical for ensuring the Rust migration
does not change trading behavior.

Run with:
    pytest tests/test_indicator_parity.py -v
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# Numerical tolerance for floating-point comparison
TOLERANCE = 1e-10
LOOSE_TOLERANCE = 1e-6

N_BARS = 1000


def rust_available() -> bool:
    """Check if Rust IndicatorCacheRust module is available."""
    try:
        from omega_rust import IndicatorCacheRust

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_data() -> Dict[str, Dict[str, List[Any]]]:
    """Generate realistic multi-candle data for IndicatorCache."""
    np.random.seed(42)

    returns = np.random.randn(N_BARS) * 0.001 + 0.0001
    close = 1.10000 * np.exp(np.cumsum(returns))
    spread = np.abs(np.random.randn(N_BARS) * 0.0005) + 0.0001
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(np.random.randn(N_BARS) * 1000) + 100

    candles = [
        {
            "open": float(open_[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": float(volume[i]),
        }
        for i in range(N_BARS)
    ]

    return {"H1": {"bid": candles, "ask": candles}}


def compare_arrays(
    arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: float = TOLERANCE
) -> None:
    """Compare two arrays with detailed error reporting."""
    if isinstance(arr1, pd.Series):
        arr1 = arr1.to_numpy()
    if isinstance(arr2, pd.Series):
        arr2 = arr2.to_numpy()

    assert len(arr1) == len(
        arr2
    ), f"{name}: Length mismatch: {len(arr1)} vs {len(arr2)}"

    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)

    nan_mismatch = np.sum(nan1 != nan2)
    assert (
        nan_mismatch == 0
    ), f"{name}: NaN position mismatch at {nan_mismatch} positions"

    valid = ~nan1 & ~nan2
    if np.sum(valid) > 0:
        max_diff = np.max(np.abs(arr1[valid] - arr2[valid]))
        assert (
            max_diff < tolerance
        ), f"{name}: Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}"


def clear_indicator_cache_modules():
    """Clear cached imports to force fresh load."""
    for mod_name in list(sys.modules.keys()):
        if "backtest_engine" in mod_name or "indicator_cache" in mod_name:
            del sys.modules[mod_name]


@pytest.mark.skipif(not rust_available(), reason="Rust module not available")
class TestIndicatorCacheParity:
    """Test IndicatorCache produces identical results with Python vs Rust backend."""

    def test_ema_parity(self, sample_data: Dict[str, Dict[str, List[Any]]]) -> None:
        """Test EMA produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_result = py_cache.ema("H1", "bid", 20)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_result = rust_cache.ema("H1", "bid", 20)

        compare_arrays(py_result, rust_result, "EMA(20)")

    def test_sma_parity(self, sample_data: Dict[str, Dict[str, List[Any]]]) -> None:
        """Test SMA produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_result = py_cache.sma("H1", "bid", 20)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_result = rust_cache.sma("H1", "bid", 20)

        compare_arrays(py_result, rust_result, "SMA(20)")

    def test_atr_parity(self, sample_data: Dict[str, Dict[str, List[Any]]]) -> None:
        """Test ATR produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_result = py_cache.atr("H1", "bid", 14)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_result = rust_cache.atr("H1", "bid", 14)

        compare_arrays(py_result, rust_result, "ATR(14)")

    def test_roc_parity(self, sample_data: Dict[str, Dict[str, List[Any]]]) -> None:
        """Test ROC produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_result = py_cache.roc("H1", "bid", 14)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_result = rust_cache.roc("H1", "bid", 14)

        compare_arrays(py_result, rust_result, "ROC(14)")

    def test_bollinger_parity(
        self, sample_data: Dict[str, Dict[str, List[Any]]]
    ) -> None:
        """Test Bollinger Bands produce identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_upper, py_mid, py_lower = py_cache.bollinger("H1", "bid", 20, 2.0)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_upper, rust_mid, rust_lower = rust_cache.bollinger("H1", "bid", 20, 2.0)

        compare_arrays(py_upper, rust_upper, "Bollinger Upper", LOOSE_TOLERANCE)
        compare_arrays(py_mid, rust_mid, "Bollinger Middle")
        compare_arrays(py_lower, rust_lower, "Bollinger Lower", LOOSE_TOLERANCE)

    def test_dmi_parity(self, sample_data: Dict[str, Dict[str, List[Any]]]) -> None:
        """Test DMI produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_plus_di, py_minus_di, py_adx = py_cache.dmi("H1", "bid", 14)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_plus_di, rust_minus_di, rust_adx = rust_cache.dmi("H1", "bid", 14)

        compare_arrays(py_plus_di, rust_plus_di, "+DI(14)", LOOSE_TOLERANCE)
        compare_arrays(py_minus_di, rust_minus_di, "-DI(14)", LOOSE_TOLERANCE)
        compare_arrays(py_adx, rust_adx, "ADX(14)", LOOSE_TOLERANCE)

    def test_macd_parity(self, sample_data: Dict[str, Dict[str, List[Any]]]) -> None:
        """Test MACD produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_line, py_signal = py_cache.macd("H1", "bid", 12, 26, 9)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_line, rust_signal = rust_cache.macd("H1", "bid", 12, 26, 9)

        compare_arrays(py_line, rust_line, "MACD Line")
        compare_arrays(py_signal, rust_signal, "MACD Signal")

    def test_choppiness_parity(
        self, sample_data: Dict[str, Dict[str, List[Any]]]
    ) -> None:
        """Test Choppiness Index produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_result = py_cache.choppiness("H1", "bid", 14)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_result = rust_cache.choppiness("H1", "bid", 14)

        compare_arrays(py_result, rust_result, "Choppiness(14)")

    def test_kalman_mean_parity(
        self, sample_data: Dict[str, Dict[str, List[Any]]]
    ) -> None:
        """Test Kalman Mean produces identical results."""
        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        from backtest_engine.core.indicator_cache import IndicatorCache

        py_cache = IndicatorCache(sample_data)
        py_result = py_cache.kalman_mean("H1", "bid", R=0.01, Q=1.0)

        clear_indicator_cache_modules()
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        from backtest_engine.core.indicator_cache import IndicatorCache as RustCache

        rust_cache = RustCache(sample_data)
        rust_result = rust_cache.kalman_mean("H1", "bid", R=0.01, Q=1.0)

        compare_arrays(py_result, rust_result, "Kalman Mean")
