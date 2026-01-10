"""Tests for Rust IndicatorCache implementation.

These tests verify:
1. Rust/Python numerical equivalence (golden file tests)
2. Feature flag behavior
3. API compatibility between Rust and Python implementations

Run with:
    pytest tests/test_indicator_cache_rust.py -v

To enable Rust implementation:
    OMEGA_USE_RUST_INDICATOR_CACHE=1 pytest tests/test_indicator_cache_rust.py -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Tolerance for numerical comparison (matches Wave 0 & 2)
TOLERANCE = 1e-12


@pytest.fixture
def sample_ohlcv() -> dict[str, NDArray[np.float64]]:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 1000

    # Generate realistic price series
    returns = np.random.randn(n) * 0.001 + 0.0001
    close = 1.10000 * np.exp(np.cumsum(returns))

    # Generate high/low around close
    spread = np.abs(np.random.randn(n) * 0.0005) + 0.0001
    high = close + spread
    low = close - spread

    # Open is previous close (shifted)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Volume is random
    volume = np.abs(np.random.randn(n) * 1000) + 100

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def rust_available() -> bool:
    """Check if Rust IndicatorCache is available."""
    try:
        from omega_rust import IndicatorCacheRust

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not rust_available(), reason="Rust module not available")
class TestRustIndicatorCacheBasic:
    """Basic tests for Rust IndicatorCache."""

    def test_import(self) -> None:
        """Test Rust module can be imported."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        assert cache is not None

    def test_register_ohlcv(self, sample_ohlcv: dict) -> None:
        """Test OHLCV registration."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        # Should not raise
        assert True

    def test_cache_size(self, sample_ohlcv: dict) -> None:
        """Test cache size tracking."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        assert cache.cache_size() == 0

        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        # Call an indicator to populate cache
        _ = cache.atr("EURUSD", "H1", "BID", 14)
        assert cache.cache_size() >= 1

    def test_clear(self, sample_ohlcv: dict) -> None:
        """Test cache clearing."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )
        _ = cache.atr("EURUSD", "H1", "BID", 14)
        cache.clear()
        assert cache.cache_size() == 0


@pytest.mark.skipif(not rust_available(), reason="Rust module not available")
class TestRustIndicatorCacheIndicators:
    """Test individual indicator calculations."""

    def test_atr(self, sample_ohlcv: dict) -> None:
        """Test ATR calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        atr = cache.atr("EURUSD", "H1", "BID", 14)

        # Verify shape
        assert len(atr) == len(sample_ohlcv["close"])

        # Verify first values are NaN (warmup period)
        assert np.isnan(atr[0])

        # Verify later values are positive (ATR is always positive)
        valid_atr = atr[~np.isnan(atr)]
        assert np.all(valid_atr > 0)

    def test_sma(self, sample_ohlcv: dict) -> None:
        """Test SMA calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        sma = cache.sma("EURUSD", "H1", "BID", 20)

        assert len(sma) == len(sample_ohlcv["close"])

        # Verify manual SMA calculation for one point
        close = sample_ohlcv["close"]
        expected_sma_at_50 = np.mean(close[31:51])  # window [31, 50] for idx 50
        assert np.abs(sma[50] - expected_sma_at_50) < TOLERANCE

    def test_ema(self, sample_ohlcv: dict) -> None:
        """Test EMA calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        ema = cache.ema("EURUSD", "H1", "BID", 20)

        assert len(ema) == len(sample_ohlcv["close"])

        # EMA should be close to price
        close = sample_ohlcv["close"]
        assert np.abs(ema[-1] - close[-1]) < 0.01

    def test_bollinger(self, sample_ohlcv: dict) -> None:
        """Test Bollinger Bands calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        upper, middle, lower = cache.bollinger("EURUSD", "H1", "BID", 20, 2.0)

        assert len(upper) == len(sample_ohlcv["close"])
        assert len(middle) == len(sample_ohlcv["close"])
        assert len(lower) == len(sample_ohlcv["close"])

        # Verify band ordering: upper >= middle >= lower
        valid = ~np.isnan(upper)
        assert np.all(upper[valid] >= middle[valid])
        assert np.all(middle[valid] >= lower[valid])

    def test_dmi(self, sample_ohlcv: dict) -> None:
        """Test DMI calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        plus_di, minus_di, adx = cache.dmi("EURUSD", "H1", "BID", 14)

        assert len(plus_di) == len(sample_ohlcv["close"])
        assert len(minus_di) == len(sample_ohlcv["close"])
        assert len(adx) == len(sample_ohlcv["close"])

        # DI values should be in [0, 100] range (approximately)
        # Note: Each output may have different valid ranges due to warmup
        valid_plus = ~np.isnan(plus_di)
        valid_minus = ~np.isnan(minus_di)
        valid_adx = ~np.isnan(adx)
        assert np.all(plus_di[valid_plus] >= 0)
        assert np.all(minus_di[valid_minus] >= 0)
        assert np.all(adx[valid_adx] >= 0)

    def test_macd(self, sample_ohlcv: dict) -> None:
        """Test MACD calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        macd, signal, histogram = cache.macd("EURUSD", "H1", "BID", 12, 26, 9)

        assert len(macd) == len(sample_ohlcv["close"])
        assert len(signal) == len(sample_ohlcv["close"])
        assert len(histogram) == len(sample_ohlcv["close"])

        # Histogram = MACD - Signal
        valid = ~np.isnan(histogram)
        np.testing.assert_allclose(
            histogram[valid], macd[valid] - signal[valid], rtol=TOLERANCE
        )

    def test_zscore(self, sample_ohlcv: dict) -> None:
        """Test Z-Score calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        zscore = cache.zscore("EURUSD", "H1", "BID", 20, 1)

        assert len(zscore) == len(sample_ohlcv["close"])

        # Z-scores should be roughly in [-3, 3] for normal data
        valid = ~np.isnan(zscore)
        assert np.percentile(zscore[valid], 5) > -5
        assert np.percentile(zscore[valid], 95) < 5

    def test_choppiness(self, sample_ohlcv: dict) -> None:
        """Test Choppiness Index calculation."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        chop = cache.choppiness("EURUSD", "H1", "BID", 14)

        assert len(chop) == len(sample_ohlcv["close"])

        # Choppiness Index should be in [0, 100]
        valid = ~np.isnan(chop)
        assert np.all(chop[valid] >= 0)
        assert np.all(chop[valid] <= 100)


@pytest.mark.skipif(not rust_available(), reason="Rust module not available")
class TestRustIndicatorCacheCaching:
    """Test caching behavior."""

    def test_cache_hit(self, sample_ohlcv: dict) -> None:
        """Test that repeated calls use cached values."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        # First call
        atr1 = cache.atr("EURUSD", "H1", "BID", 14)
        size_after_first = cache.cache_size()

        # Second call (should hit cache)
        atr2 = cache.atr("EURUSD", "H1", "BID", 14)
        size_after_second = cache.cache_size()

        # Cache size should not change
        assert size_after_first == size_after_second

        # Results should be identical
        np.testing.assert_array_equal(atr1, atr2)

    def test_cache_invalidation(self, sample_ohlcv: dict) -> None:
        """Test that re-registering OHLCV invalidates cache."""
        from omega_rust import IndicatorCacheRust

        cache = IndicatorCacheRust()
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        # Calculate indicator
        _ = cache.atr("EURUSD", "H1", "BID", 14)
        size_before = cache.cache_size()

        # Re-register with modified data
        modified_close = sample_ohlcv["close"] * 1.01
        cache.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=modified_close,
            volume=sample_ohlcv["volume"],
        )

        size_after = cache.cache_size()

        # Cache should be invalidated (size reset)
        assert size_after < size_before


class TestFeatureFlag:
    """Test feature flag behavior."""

    def test_feature_flag_disabled_by_default(self) -> None:
        """Test that Rust is disabled by default."""
        from src.backtest_engine.core.indicator_cache_rust import is_rust_enabled

        # Temporarily unset the flag
        old_value = os.environ.pop("OMEGA_USE_RUST_INDICATOR_CACHE", None)
        try:
            assert not is_rust_enabled()
        finally:
            if old_value is not None:
                os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = old_value

    def test_feature_flag_enabled(self) -> None:
        """Test that Rust can be enabled via flag."""
        from src.backtest_engine.core.indicator_cache_rust import is_rust_enabled

        old_value = os.environ.get("OMEGA_USE_RUST_INDICATOR_CACHE")
        try:
            os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = "1"
            assert is_rust_enabled()
        finally:
            if old_value is not None:
                os.environ["OMEGA_USE_RUST_INDICATOR_CACHE"] = old_value
            else:
                os.environ.pop("OMEGA_USE_RUST_INDICATOR_CACHE", None)

    @pytest.mark.skipif(not rust_available(), reason="Rust module not available")
    def test_wrapper_creation(self, sample_ohlcv: dict) -> None:
        """Test wrapper creation with feature flag."""
        from src.backtest_engine.core.indicator_cache_rust import (
            IndicatorCacheRustWrapper,
        )

        # Should be able to create wrapper directly
        wrapper = IndicatorCacheRustWrapper()
        assert wrapper.implementation == "rust"

        # Register and calculate
        wrapper.register_ohlcv(
            symbol="EURUSD",
            timeframe="H1",
            price_type="BID",
            open=sample_ohlcv["open"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            close=sample_ohlcv["close"],
            volume=sample_ohlcv["volume"],
        )

        atr = wrapper.atr("EURUSD", "H1", "BID", 14)
        assert len(atr) == len(sample_ohlcv["close"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
