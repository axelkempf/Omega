"""Integration tests for Rust IndicatorCache in backtest pipeline.

These tests verify that:
1. The backtest pipeline correctly delegates to Rust when available
2. All integrated indicators work end-to-end
3. Python fallback works correctly when Rust is disabled

Run with:
    pytest tests/test_indicator_cache_backtest_integration.py -v

To force Python-only mode:
    OMEGA_USE_RUST_INDICATOR_CACHE=0 pytest tests/test_indicator_cache_backtest_integration.py -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Test tolerance for numerical comparison
TOLERANCE = 1e-10


def create_aligned_candle_data(n: int = 500) -> Dict[str, Dict[str, List[Any]]]:
    """Create realistic aligned multi-timeframe candle data for testing.
    
    Returns a structure compatible with IndicatorCache:
    {
        "M1": {"bid": [Candle, ...], "ask": [Candle, ...]},
        "H1": {"bid": [Candle, ...], "ask": [Candle, ...]},
    }
    """
    np.random.seed(42)
    
    # Generate M1 data
    returns = np.random.randn(n) * 0.0002 + 0.00001
    close = 1.10000 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    spread = np.abs(np.random.randn(n) * 0.0003) + 0.0001
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(np.random.randn(n) * 1000) + 100
    
    # Create simple dict-based candles (compatible with indicator_cache)
    m1_bid = []
    m1_ask = []
    for i in range(n):
        candle = {
            "open": open_[i],
            "high": high[i],
            "low": low[i],
            "close": close[i],
            "volume": volume[i],
        }
        m1_bid.append(candle)
        # Ask is slightly higher
        ask_candle = {
            "open": open_[i] + 0.0001,
            "high": high[i] + 0.0001,
            "low": low[i] + 0.0001,
            "close": close[i] + 0.0001,
            "volume": volume[i],
        }
        m1_ask.append(ask_candle)
    
    return {
        "M1": {"bid": m1_bid, "ask": m1_ask},
    }


def rust_available() -> bool:
    """Check if Rust IndicatorCache is available."""
    try:
        from omega_rust import IndicatorCacheRust
        return True
    except ImportError:
        return False


class TestIndicatorCacheBacktestIntegration:
    """Integration tests for IndicatorCache in backtest pipeline."""

    def test_indicator_cache_import(self) -> None:
        """Test that IndicatorCache can be imported."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        assert IndicatorCache is not None

    def test_indicator_cache_initialization(self) -> None:
        """Test IndicatorCache initialization with multi-candle data."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        assert cache is not None
        
    def test_ema_calculation(self) -> None:
        """Test EMA calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        ema = cache.ema("M1", "bid", period=20)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == 200
        # Check that EMA converges (not all NaN)
        assert not ema.isna().all()
        
    def test_sma_calculation(self) -> None:
        """Test SMA calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        sma = cache.sma("M1", "bid", period=20)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == 200
        # First 19 values should be NaN (need 20 periods for SMA)
        assert sma.iloc[:19].isna().all()
        # From index 19, values should be valid
        assert not sma.iloc[19:].isna().all()

    def test_atr_calculation(self) -> None:
        """Test ATR (Wilder) calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        atr = cache.atr("M1", "bid", period=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == 200
        # ATR should be positive where valid
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_bollinger_bands(self) -> None:
        """Test Bollinger Bands calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        upper, mid, lower = cache.bollinger("M1", "bid", period=20, std_factor=2.0)
        
        assert isinstance(upper, pd.Series)
        assert isinstance(mid, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # Check relationship: upper > mid > lower
        valid_mask = ~(upper.isna() | mid.isna() | lower.isna())
        assert (upper[valid_mask] >= mid[valid_mask]).all()
        assert (mid[valid_mask] >= lower[valid_mask]).all()

    def test_dmi_calculation(self) -> None:
        """Test DMI (+DI, -DI, ADX) calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        plus_di, minus_di, adx = cache.dmi("M1", "bid", period=14)
        
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)
        assert isinstance(adx, pd.Series)
        assert len(plus_di) == 200
        
    def test_macd_calculation(self) -> None:
        """Test MACD calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        macd_line, signal_line = cache.macd("M1", "bid", 12, 26, 9)
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert len(macd_line) == 200

    def test_roc_calculation(self) -> None:
        """Test ROC (Rate of Change) calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        roc = cache.roc("M1", "bid", period=14)
        
        assert isinstance(roc, pd.Series)
        assert len(roc) == 200

    def test_choppiness_calculation(self) -> None:
        """Test Choppiness Index calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        chop = cache.choppiness("M1", "bid", period=14)
        
        assert isinstance(chop, pd.Series)
        assert len(chop) == 200
        # Choppiness should be between 0 and 100
        valid = chop.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_kalman_mean_calculation(self) -> None:
        """Test Kalman Mean calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        km = cache.kalman_mean("M1", "bid", R=0.01, Q=1.0)
        
        assert isinstance(km, pd.Series)
        assert len(km) == 200
        # Kalman mean should follow price trend
        assert not km.isna().all()

    def test_kalman_zscore_calculation(self) -> None:
        """Test Kalman Z-Score calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        kz = cache.kalman_zscore("M1", "bid", window=100, R=0.01, Q=1.0)
        
        assert isinstance(kz, pd.Series)
        assert len(kz) == 200

    def test_zscore_calculation(self) -> None:
        """Test Z-Score calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        z = cache.zscore("M1", "bid", window=100, mean_source="rolling")
        
        assert isinstance(z, pd.Series)
        assert len(z) == 200

    def test_indicator_caching(self) -> None:
        """Test that indicators are properly cached."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        # First call
        ema1 = cache.ema("M1", "bid", period=20)
        # Second call (should hit cache)
        ema2 = cache.ema("M1", "bid", period=20)
        
        # Should be the same object (cached)
        assert ema1 is ema2

    def test_rsi_calculation(self) -> None:
        """Test RSI calculation through IndicatorCache."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        rsi = cache.rsi("M1", "bid", period=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == 200
        # RSI should be between 0 and 100
        valid = rsi.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()


@pytest.mark.skipif(not rust_available(), reason="Rust module not available")
class TestRustDelegation:
    """Tests that verify Rust delegation is working."""

    def test_rust_backend_enabled(self) -> None:
        """Test that Rust backend is enabled when available."""
        from backtest_engine.core.indicator_cache import (
            IndicatorCache,
            USE_RUST_INDICATOR_CACHE,
        )
        
        # With auto mode, Rust should be enabled if available
        env_val = os.environ.get("OMEGA_USE_RUST_INDICATOR_CACHE", "auto").lower()
        if env_val == "0":
            pytest.skip("Rust explicitly disabled via env var")
        
        data = create_aligned_candle_data(200)
        cache = IndicatorCache(data)
        
        # Check that _use_rust is True (Rust delegation active)
        assert cache._use_rust is True
        assert cache._rust_cache is not None

    def test_rust_atr_matches_python(self) -> None:
        """Test that Rust ATR matches Python implementation."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(500)
        
        # Calculate with Rust
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
        cache_rust = IndicatorCache(data)
        atr_rust = cache_rust.atr("M1", "bid", period=14)
        
        # Calculate with Python
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "0"
        # Need to reimport to pick up env change - create new cache
        cache_python = IndicatorCache(data)
        cache_python._use_rust = False  # Force Python
        cache_python._rust_cache = None
        cache_python._ind_cache.clear()  # Clear cached indicators
        atr_python = cache_python.atr("M1", "bid", period=14)
        
        # Reset env
        os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "auto"
        
        # Compare - allow for small floating point differences
        np.testing.assert_allclose(
            atr_rust.dropna().values, 
            atr_python.dropna().values, 
            rtol=1e-10,
            err_msg="Rust and Python ATR results differ"
        )


class TestEventEngineIntegration:
    """Tests for integration with event_engine."""

    def test_get_cached_indicator_cache(self) -> None:
        """Test that get_cached_indicator_cache returns proper cache."""
        from backtest_engine.core.indicator_cache import get_cached_indicator_cache
        
        data = create_aligned_candle_data(200)
        
        cache1 = get_cached_indicator_cache(data)
        cache2 = get_cached_indicator_cache(data)
        
        # Should return cached instance for same data
        assert cache1 is cache2

    def test_indicator_cache_with_none_candles(self) -> None:
        """Test that IndicatorCache handles None candles correctly."""
        from backtest_engine.core.indicator_cache import IndicatorCache
        
        data = create_aligned_candle_data(200)
        # Insert some None candles (simulating missing data)
        data["M1"]["bid"][50] = None
        data["M1"]["bid"][100] = None
        
        cache = IndicatorCache(data)
        
        # Should not crash and handle NaNs properly
        ema = cache.ema("M1", "bid", period=20)
        assert isinstance(ema, pd.Series)
        assert len(ema) == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
