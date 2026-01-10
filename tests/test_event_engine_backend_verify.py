"""
Verification tests for Event Engine Rust backend.

Tests follow Wave 3 migration plan patterns:
- Backend availability checking
- Feature flag handling
- CI verification helpers
"""

import os
from unittest.mock import patch

import pytest


class TestEventEngineBackendVerify:
    """Backend verification tests for Event Engine."""

    def test_rust_backend_available(self):
        """Verify Rust event engine is importable."""
        from src.backtest_engine.core.event_engine import (
            _check_rust_event_engine_available,
        )

        available = _check_rust_event_engine_available()
        assert available is True, "Rust event engine should be available"

    def test_rust_module_has_required_exports(self):
        """Verify Rust module exports required classes/functions."""
        import omega_rust

        # Core classes
        assert hasattr(omega_rust, "EventEngineRust")
        assert hasattr(omega_rust, "EventEngineStats")

        # CI verification helper
        assert hasattr(omega_rust, "get_event_engine_backend")

    def test_get_event_engine_backend_returns_rust(self):
        """Verify backend identifier returns correct value."""
        import omega_rust

        backend = omega_rust.get_event_engine_backend()
        assert backend == "rust", f"Expected 'rust', got '{backend}'"

    def test_get_active_backend_default(self):
        """Verify active backend is Rust by default when available."""
        from src.backtest_engine.core.event_engine import get_active_backend

        # Clear env var if set
        with patch.dict(os.environ, {}, clear=True):
            backend = get_active_backend()
            assert (
                backend == "rust"
            ), f"Default backend should be 'rust', got '{backend}'"

    def test_feature_flag_auto(self):
        """Test OMEGA_USE_RUST_EVENT_ENGINE=auto behavior."""
        from src.backtest_engine.core.event_engine import _should_use_rust

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EVENT_ENGINE": "auto"}):
            assert _should_use_rust() is True, "Auto should use Rust when available"

    def test_feature_flag_true(self):
        """Test OMEGA_USE_RUST_EVENT_ENGINE=true forces Rust."""
        from src.backtest_engine.core.event_engine import _should_use_rust

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EVENT_ENGINE": "true"}):
            assert _should_use_rust() is True

    def test_feature_flag_false(self):
        """Test OMEGA_USE_RUST_EVENT_ENGINE=false forces Python."""
        from src.backtest_engine.core.event_engine import _should_use_rust

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EVENT_ENGINE": "false"}):
            assert _should_use_rust() is False

    def test_get_active_backend_force_python(self):
        """Verify Python backend can be forced via env var."""
        from src.backtest_engine.core.event_engine import get_active_backend

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EVENT_ENGINE": "false"}):
            backend = get_active_backend()
            assert backend == "python", f"Forced Python, got '{backend}'"


class TestEventEngineStatsAttributes:
    """Verify EventEngineStats has expected attributes."""

    def test_stats_attributes_exist(self):
        """Verify EventEngineStats has required fields."""
        import omega_rust

        stats = omega_rust.EventEngineStats()

        # Required attributes from actual implementation
        assert hasattr(stats, "bars_processed")
        assert hasattr(stats, "signals_generated")
        assert hasattr(stats, "trades_executed")
        assert hasattr(stats, "exits_processed")
        assert hasattr(stats, "loop_time_ms")
        assert hasattr(stats, "callback_time_ms")
        assert hasattr(stats, "portfolio_time_ms")

    def test_stats_initial_values(self):
        """Verify initial values are zero."""
        import omega_rust

        stats = omega_rust.EventEngineStats()

        assert stats.bars_processed == 0
        assert stats.signals_generated == 0
        assert stats.trades_executed == 0
        assert stats.exits_processed == 0
        assert stats.loop_time_ms == 0.0
        assert stats.callback_time_ms == 0.0

    def test_stats_summary_method(self):
        """Verify summary method returns string."""
        import omega_rust

        stats = omega_rust.EventEngineStats()
        summary = stats.summary()

        assert isinstance(summary, str)
        assert "bars" in summary.lower()


class TestEventEngineRustInit:
    """Test EventEngineRust initialization."""

    def test_rust_engine_init_with_data(self):
        """Verify EventEngineRust can be initialized with candle data."""
        from dataclasses import dataclass
        from datetime import datetime

        import omega_rust

        # Create mock candle objects
        @dataclass
        class MockCandle:
            timestamp: datetime
            open: float
            high: float
            low: float
            close: float
            volume: float

        candles = [
            MockCandle(datetime(2024, 1, 1, 0, 0), 1.1, 1.2, 1.0, 1.15, 1000.0),
            MockCandle(datetime(2024, 1, 1, 0, 1), 1.15, 1.18, 1.12, 1.16, 1100.0),
        ]

        engine = omega_rust.EventEngineRust(
            bid_candles=candles,
            ask_candles=candles,
            start_index=0,
            symbol="EURUSD",
        )

        assert engine is not None
        assert hasattr(engine, "run")

    def test_rust_engine_empty_candles_error(self):
        """Verify error handling for empty candle lists."""
        import omega_rust

        # Empty candles should raise error
        with pytest.raises(RuntimeError, match="start_index.*must be less than"):
            omega_rust.EventEngineRust(
                bid_candles=[],
                ask_candles=[],
                start_index=0,
                symbol="EURUSD",
            )


class TestEventEngineClassInterface:
    """Test EventEngine class interface remains compatible."""

    def test_event_engine_has_use_rust_param(self):
        """Verify EventEngine accepts use_rust parameter."""
        import inspect

        from src.backtest_engine.core.event_engine import EventEngine

        sig = inspect.signature(EventEngine.__init__)
        params = list(sig.parameters.keys())

        assert "use_rust" in params, "EventEngine should accept use_rust parameter"

    def test_event_engine_has_run_methods(self):
        """Verify EventEngine has both run methods."""
        from src.backtest_engine.core.event_engine import EventEngine

        assert hasattr(EventEngine, "run")
        assert hasattr(EventEngine, "_run_rust")
        assert hasattr(EventEngine, "_run_python")

    def test_event_engine_has_position_mgmt_callback(self):
        """Verify position management callback creation method exists."""
        from src.backtest_engine.core.event_engine import EventEngine

        assert hasattr(EventEngine, "_create_position_mgmt_callback")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
