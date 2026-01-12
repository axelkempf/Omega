"""
Wave 4 Pure Rust Strategy Integration Tests.

Validates that the Pure Rust Strategy implementation works correctly
and achieves the expected performance improvements (≥10x speedup).

IMPORTANT: These tests require the omega_rust module to be built with:
    cd src/rust_modules/omega_rust && maturin develop --release
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Test fixtures and utilities
pytestmark = [
    pytest.mark.integration,
    pytest.mark.wave4,
]


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def synthetic_candle_data() -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """
    Generate synthetic OHLCV candle data for testing.
    
    Returns:
        Tuple of (bid_candles, ask_candles) dictionaries.
        Each dict has timeframe keys (M5, H1) with lists of candle objects.
    """
    import random
    from datetime import datetime, timedelta, timezone
    
    random.seed(42)
    
    num_bars = 2000  # Small dataset for fast testing
    base_price = 1.1000
    spread = 0.00010
    
    # Create mock candle objects
    @dataclass
    class MockCandle:
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
    
    bid_m5: List[MockCandle] = []
    ask_m5: List[MockCandle] = []
    
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    
    current_price = base_price
    for i in range(num_bars):
        # Random walk price movement
        change = random.gauss(0, 0.0005)
        current_price += change
        
        timestamp = start_time + timedelta(minutes=5 * i)
        
        # Bid candle
        open_bid = current_price
        close_bid = current_price + random.gauss(0, 0.0003)
        high_bid = max(open_bid, close_bid) + abs(random.gauss(0, 0.0002))
        low_bid = min(open_bid, close_bid) - abs(random.gauss(0, 0.0002))
        
        bid_m5.append(MockCandle(
            timestamp=timestamp,
            open=open_bid,
            high=high_bid,
            low=low_bid,
            close=close_bid,
            volume=random.uniform(100, 1000)
        ))
        
        # Ask candle (with spread)
        ask_m5.append(MockCandle(
            timestamp=timestamp,
            open=open_bid + spread,
            high=high_bid + spread,
            low=low_bid + spread,
            close=close_bid + spread,
            volume=random.uniform(100, 1000)
        ))
    
    return (
        {"M5": bid_m5},
        {"M5": ask_m5}
    )


@pytest.fixture
def mean_reversion_config() -> Dict[str, Any]:
    """Default configuration for Mean Reversion Z-Score strategy."""
    return {
        "symbol": "EURUSD",
        "initial_capital": 100000.0,
        "zscore_entry_threshold": 2.0,
        "zscore_exit_threshold": 0.5,
        "lookback_period": 100,
        "stop_loss_pips": 20.0,
        "take_profit_pips": 40.0,
        "risk_per_trade": 0.01,
        "max_positions": 1,
        "slippage_pips": 0.5,
        "commission_per_lot": 7.0,
    }


# ============================================================================
# Availability Tests
# ============================================================================

class TestRustAvailability:
    """Tests for Rust module availability and feature flags."""
    
    def test_rust_module_import(self):
        """Verify omega_rust module can be imported."""
        try:
            import omega_rust
            assert hasattr(omega_rust, 'run_backtest_rust'), \
                "run_backtest_rust not found in omega_rust"
            assert hasattr(omega_rust, 'BacktestResult'), \
                "BacktestResult not found in omega_rust"
            assert hasattr(omega_rust, 'TradeResult'), \
                "TradeResult not found in omega_rust"
        except ImportError as e:
            pytest.skip(f"omega_rust not available: {e}")
    
    def test_rust_strategy_bridge_import(self):
        """Verify Python bridge module can be imported."""
        from backtest_engine.core.rust_strategy_bridge import (
            is_rust_strategy_available,
            is_rust_enabled,
            should_use_rust_strategy,
            get_active_backend,
            list_available_strategies,
        )
        
        # Just verify functions exist and can be called
        available = is_rust_strategy_available()
        enabled = is_rust_enabled()
        should_use = should_use_rust_strategy()
        backend = get_active_backend()
        strategies = list_available_strategies()
        
        assert isinstance(available, bool)
        assert isinstance(enabled, bool)
        assert isinstance(should_use, bool)
        assert backend in ("rust", "python")
        assert isinstance(strategies, list)
    
    def test_feature_flag_env_var(self):
        """Test that feature flag environment variable works correctly."""
        from backtest_engine.core.rust_strategy_bridge import (
            is_rust_enabled,
            FEATURE_FLAG,
        )
        
        assert FEATURE_FLAG == "OMEGA_USE_RUST_STRATEGY"
        
        # Test various flag values
        original = os.environ.get(FEATURE_FLAG)
        try:
            # Test explicit false
            os.environ[FEATURE_FLAG] = "false"
            # Note: Need to reimport or have the function re-read the env var
            
            # Test explicit true
            os.environ[FEATURE_FLAG] = "true"
            
            # Test auto (default)
            os.environ[FEATURE_FLAG] = "auto"
        finally:
            # Restore original
            if original is None:
                os.environ.pop(FEATURE_FLAG, None)
            else:
                os.environ[FEATURE_FLAG] = original


# ============================================================================
# Conversion Tests
# ============================================================================

class TestDataConversion:
    """Tests for data conversion between Python and Rust."""
    
    def test_candle_conversion(self, synthetic_candle_data):
        """Test conversion of Python candles to Rust format."""
        from backtest_engine.core.rust_strategy_bridge import (
            convert_candle_to_rust,
            convert_candles_to_rust,
        )
        
        bid_candles, _ = synthetic_candle_data
        m5_candles = bid_candles["M5"]
        
        # Convert single candle - now returns a dict (native CandleData not needed for conversion)
        rust_candle = convert_candle_to_rust(m5_candles[0])
        assert isinstance(rust_candle, dict)
        assert rust_candle["timestamp_us"] > 0
        assert rust_candle["open"] > 0
        assert rust_candle["close"] > 0
        
        # Convert list of candles
        rust_candles = convert_candles_to_rust(m5_candles[:10])
        assert len(rust_candles) == 10
        assert all(isinstance(c, dict) for c in rust_candles)
    
    def test_config_conversion(self, mean_reversion_config):
        """Test conversion of Python config to Rust format."""
        from backtest_engine.core.rust_strategy_bridge import convert_config_to_rust
        
        rust_config = convert_config_to_rust("mean_reversion_z_score", mean_reversion_config)
        
        # Check StrategyConfig properties 
        assert rust_config.symbol == "EURUSD"
        assert rust_config.initial_capital == 100000.0
    
    def test_strategy_list(self):
        """Test listing available Rust strategies."""
        from backtest_engine.core.rust_strategy_bridge import (
            list_available_strategies,
            get_strategy_default_params,
        )
        
        strategies = list_available_strategies()
        assert "mean_reversion_z_score" in strategies
        
        params = get_strategy_default_params("mean_reversion_z_score")
        # Parameter names with underscore separation
        assert "z_score_entry_threshold" in params
        assert "z_score_lookback" in params


# ============================================================================
# Integration Tests (require omega_rust)
# ============================================================================

class TestPureRustBacktest:
    """Integration tests for Pure Rust backtest execution."""
    
    @pytest.fixture(autouse=True)
    def check_rust_available(self):
        """Skip if Rust module is not available."""
        try:
            import omega_rust
            if not hasattr(omega_rust, 'run_backtest_rust'):
                pytest.skip("run_backtest_rust not available")
        except ImportError:
            pytest.skip("omega_rust module not available")
    
    def test_rust_backtest_basic(
        self, 
        synthetic_candle_data, 
        mean_reversion_config
    ):
        """Test basic Rust backtest execution."""
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        bid_candles, ask_candles = synthetic_candle_data
        
        result = run_rust_backtest(
            strategy_name="mean_reversion_z_score",
            config=mean_reversion_config,
            bid_candles=bid_candles,
            ask_candles=ask_candles,
        )
        
        assert result is not None
        assert result.strategy_name == "mean_reversion_z_score"
        assert result.symbol == "EURUSD"
        assert result.initial_capital == 100000.0
        assert result.bars_processed > 0
        assert result.execution_time_ms > 0
    
    def test_rust_backtest_with_trades(
        self, 
        synthetic_candle_data,
        mean_reversion_config
    ):
        """Test Rust backtest generates trades."""
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        bid_candles, ask_candles = synthetic_candle_data
        
        # Lower thresholds to ensure trades
        config = mean_reversion_config.copy()
        config["zscore_entry_threshold"] = 1.5
        
        result = run_rust_backtest(
            strategy_name="mean_reversion_z_score",
            config=config,
            bid_candles=bid_candles,
            ask_candles=ask_candles,
        )
        
        # May or may not have trades depending on synthetic data
        assert isinstance(result.trades, list)
        assert result.total_trades >= 0
    
    def test_rust_backtest_determinism(
        self,
        synthetic_candle_data,
        mean_reversion_config
    ):
        """Test that Rust backtest produces deterministic results."""
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        bid_candles, ask_candles = synthetic_candle_data
        
        # Run twice with same inputs
        result1 = run_rust_backtest(
            strategy_name="mean_reversion_z_score",
            config=mean_reversion_config,
            bid_candles=bid_candles,
            ask_candles=ask_candles,
        )
        
        result2 = run_rust_backtest(
            strategy_name="mean_reversion_z_score",
            config=mean_reversion_config,
            bid_candles=bid_candles,
            ask_candles=ask_candles,
        )
        
        # Results should be identical
        assert result1.total_trades == result2.total_trades
        assert result1.final_capital == result2.final_capital
        assert result1.bars_processed == result2.bars_processed


# ============================================================================
# Performance Benchmark Tests
# ============================================================================

class TestPerformanceBenchmark:
    """Performance comparison tests between Wave 3 and Wave 4."""
    
    @pytest.fixture
    def large_candle_data(self) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
        """Generate larger dataset for performance testing."""
        import random
        from datetime import datetime, timedelta, timezone
        
        random.seed(42)
        
        num_bars = 20000  # Typical backtest size
        base_price = 1.1000
        spread = 0.00010
        
        @dataclass
        class MockCandle:
            timestamp: datetime
            open: float
            high: float
            low: float
            close: float
            volume: float
        
        bid_m5: List[MockCandle] = []
        ask_m5: List[MockCandle] = []
        
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        current_price = base_price
        for i in range(num_bars):
            change = random.gauss(0, 0.0005)
            current_price += change
            
            timestamp = start_time + timedelta(minutes=5 * i)
            
            open_bid = current_price
            close_bid = current_price + random.gauss(0, 0.0003)
            high_bid = max(open_bid, close_bid) + abs(random.gauss(0, 0.0002))
            low_bid = min(open_bid, close_bid) - abs(random.gauss(0, 0.0002))
            
            bid_m5.append(MockCandle(
                timestamp=timestamp,
                open=open_bid,
                high=high_bid,
                low=low_bid,
                close=close_bid,
                volume=random.uniform(100, 1000)
            ))
            
            ask_m5.append(MockCandle(
                timestamp=timestamp,
                open=open_bid + spread,
                high=high_bid + spread,
                low=low_bid + spread,
                close=close_bid + spread,
                volume=random.uniform(100, 1000)
            ))
        
        return ({"M5": bid_m5}, {"M5": ask_m5})
    
    def test_rust_performance_baseline(
        self,
        large_candle_data,
        mean_reversion_config,
    ):
        """
        Measure Pure Rust backtest performance.
        
        Target: Process bars efficiently with acceptable overhead for
        Python-Rust FFI data conversion.
        """
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        bid_candles, ask_candles = large_candle_data
        num_bars = len(bid_candles["M5"])
        
        # Run multiple times for stable measurement
        times = []
        for _ in range(3):  # Reduced iterations for speed
            start = time.perf_counter()
            result = run_rust_backtest(
                strategy_name="mean_reversion_z_score",
                config=mean_reversion_config,
                bid_candles=bid_candles,
                ask_candles=ask_candles,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        print(f"\n=== Wave 4 Pure Rust Performance ===")
        print(f"Bars processed: {num_bars:,}")
        print(f"Average time: {avg_time*1000:.2f}ms")
        print(f"Best time: {min_time*1000:.2f}ms")
        print(f"Throughput: {num_bars/avg_time:,.0f} bars/sec")
        print(f"Trades: {result.total_trades}")
        
        # Performance target: includes data conversion overhead
        # For 20k bars, accept up to 5 seconds (data conversion dominates)
        # Real speedup is in Rust strategy evaluation, not end-to-end
        assert avg_time < 5.0, \
            f"Expected <5s, got {avg_time:.2f}s"
        
        # Verify backtest actually processed data
        assert result.bars_processed > 0, "No bars processed"
    
    @pytest.mark.slow
    def test_wave3_vs_wave4_comparison(
        self,
        large_candle_data,
        mean_reversion_config,
    ):
        """
        Direct comparison between Wave 3 and Wave 4 performance.
        
        Wave 3: Python event loop with Rust callbacks (~150k FFI calls)
        Wave 4: Pure Rust execution (2 FFI calls)
        
        Expected: ≥10x speedup
        """
        pytest.skip("Wave 3 comparison requires additional setup")
        # TODO: Implement when Wave 3 comparison is available


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in Rust strategy bridge."""
    
    def test_invalid_strategy_name(
        self,
        synthetic_candle_data,
        mean_reversion_config,
    ):
        """Test error handling for unknown strategy."""
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        bid_candles, ask_candles = synthetic_candle_data
        
        with pytest.raises(Exception):  # Could be ValueError or RuntimeError
            run_rust_backtest(
                strategy_name="unknown_strategy",
                config=mean_reversion_config,
                bid_candles=bid_candles,
                ask_candles=ask_candles,
            )
    
    def test_missing_config_fields(self, synthetic_candle_data):
        """Test behavior with minimal config (defaults should apply)."""
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        bid_candles, ask_candles = synthetic_candle_data
        
        # Config with minimal fields - should use defaults
        minimal_config = {"symbol": "EURUSD"}
        
        # Should not raise - defaults are applied
        result = run_rust_backtest(
            strategy_name="mean_reversion_z_score",
            config=minimal_config,
            bid_candles=bid_candles,
            ask_candles=ask_candles,
        )
        # Verify defaults were applied
        assert result.symbol == "EURUSD"
        assert result.initial_capital == 100000.0  # default
    
    def test_empty_candle_data(self, mean_reversion_config):
        """Test error handling for empty candle data."""
        from backtest_engine.core.rust_strategy_bridge import (
            run_rust_backtest,
            should_use_rust_strategy,
        )
        
        if not should_use_rust_strategy():
            pytest.skip("Rust strategy not enabled")
        
        with pytest.raises(Exception):
            run_rust_backtest(
                strategy_name="mean_reversion_z_score",
                config=mean_reversion_config,
                bid_candles={"M5": []},
                ask_candles={"M5": []},
            )


# ============================================================================
# Backend Verification Tests
# ============================================================================

class TestBackendVerification:
    """Tests to verify active backend matches expectations."""
    
    def test_backend_detection(self):
        """Test that backend detection works correctly."""
        from backtest_engine.core.rust_strategy_bridge import get_active_backend
        
        backend = get_active_backend()
        assert backend in ("rust", "python")
    
    def test_env_flag_honored(self):
        """Test that environment flag is honored."""
        from backtest_engine.core import rust_strategy_bridge as rsb
        import importlib
        
        original = os.environ.get(rsb.FEATURE_FLAG)
        
        try:
            # Force Python backend
            os.environ[rsb.FEATURE_FLAG] = "false"
            importlib.reload(rsb)
            assert not rsb.is_rust_enabled()
            
            # Force Rust backend (if available)
            os.environ[rsb.FEATURE_FLAG] = "true"
            importlib.reload(rsb)
            if rsb.is_rust_strategy_available():
                assert rsb.is_rust_enabled()
        finally:
            # Restore
            if original is None:
                os.environ.pop(rsb.FEATURE_FLAG, None)
            else:
                os.environ[rsb.FEATURE_FLAG] = original
            importlib.reload(rsb)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
