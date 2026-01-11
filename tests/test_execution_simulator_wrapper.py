"""Tests for ExecutionSimulator Rust Wrapper (Wave 4 Phase 6).

This module tests the thin Python wrapper that delegates to Rust ExecutionSimulatorRust.

Test Categories:
1. Feature flag validation
2. Wrapper instantiation and delegation
3. Signal processing via Arrow IPC
4. Exit evaluation via Arrow IPC
5. Portfolio integration
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestFeatureFlagValidation:
    """Tests for OMEGA_USE_RUST_EXECUTION_SIMULATOR feature flag."""

    def test_feature_flag_always_accepted(self) -> None:
        """Feature flag 'always' should be accepted."""
        from backtest_engine.core.execution_simulator import _check_rust_feature_flag

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "always"}):
            result = _check_rust_feature_flag()
            assert result is True

    def test_feature_flag_default_is_always(self) -> None:
        """Default value should be 'always'."""
        from backtest_engine.core.execution_simulator import _check_rust_feature_flag

        # Remove the env var if it exists
        env = os.environ.copy()
        env.pop("OMEGA_USE_RUST_EXECUTION_SIMULATOR", None)

        with patch.dict(os.environ, env, clear=True):
            result = _check_rust_feature_flag()
            assert result is True

    def test_feature_flag_other_values_rejected(self) -> None:
        """Other values should raise ValueError."""
        from backtest_engine.core.execution_simulator import _check_rust_feature_flag

        for invalid_value in ["false", "python", "auto", "hybrid", ""]:
            with patch.dict(
                os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": invalid_value}
            ):
                with pytest.raises(ValueError) as exc_info:
                    _check_rust_feature_flag()

                assert "OMEGA_USE_RUST_EXECUTION_SIMULATOR=always" in str(
                    exc_info.value
                )
                assert "deployment-based" in str(exc_info.value)


class TestRustWrapperHelpers:
    """Tests for Arrow IPC helper functions."""

    def test_datetime_to_utc_micros_naive(self) -> None:
        """Naive datetime should be treated as UTC."""
        from backtest_engine.core.execution_simulator_rust import (
            _datetime_to_utc_micros,
        )

        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = _datetime_to_utc_micros(dt)

        # 2024-01-01 12:00:00 UTC in microseconds
        expected = int(
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1_000_000
        )
        assert result == expected

    def test_datetime_to_utc_micros_aware(self) -> None:
        """Aware datetime should be converted to UTC."""
        from backtest_engine.core.execution_simulator_rust import (
            _datetime_to_utc_micros,
        )

        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _datetime_to_utc_micros(dt)

        expected = int(dt.timestamp() * 1_000_000)
        assert result == expected

    def test_datetime_to_utc_micros_int_passthrough(self) -> None:
        """Integer should pass through unchanged."""
        from backtest_engine.core.execution_simulator_rust import (
            _datetime_to_utc_micros,
        )

        micros = 1704067200_000_000
        result = _datetime_to_utc_micros(micros)
        assert result == micros


class TestSignalBatchBuilding:
    """Tests for Arrow IPC signal batch building."""

    def test_build_signal_batch_creates_valid_ipc(self) -> None:
        """Signal batch should create valid Arrow IPC bytes."""
        from backtest_engine.core.execution_simulator_rust import _build_signal_batch

        # Create mock signal
        signal = MagicMock()
        signal.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        signal.direction = "long"
        signal.entry_price = 1.1000
        signal.stop_loss = 1.0950
        signal.take_profit = 1.1100
        signal.symbol = "EURUSD"
        signal.type = "market"
        signal.reason = "test_signal"
        signal.scenario = "test_scenario"

        ipc_bytes = _build_signal_batch(signal)

        # Should return bytes
        assert isinstance(ipc_bytes, bytes)
        assert len(ipc_bytes) > 0

    def test_build_signal_batch_handles_missing_optional_fields(self) -> None:
        """Signal batch should handle missing optional fields."""
        from backtest_engine.core.execution_simulator_rust import _build_signal_batch

        # Create signal without optional fields
        signal = MagicMock()
        signal.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        signal.direction = "short"
        signal.entry_price = 1.1000
        signal.stop_loss = 1.1050
        signal.take_profit = 1.0900
        signal.symbol = "EURUSD"
        signal.type = "limit"

        # Remove optional attributes
        del signal.reason
        del signal.scenario

        ipc_bytes = _build_signal_batch(signal)
        assert isinstance(ipc_bytes, bytes)


class TestCandleBatchBuilding:
    """Tests for Arrow IPC candle batch building."""

    def test_build_candle_batch_creates_valid_ipc(self) -> None:
        """Candle batch should create valid Arrow IPC bytes."""
        from backtest_engine.core.execution_simulator_rust import _build_candle_batch

        # Create mock candle
        candle = MagicMock()
        candle.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        candle.open = 1.1000
        candle.high = 1.1050
        candle.low = 1.0950
        candle.close = 1.1020
        candle.volume = 1000.0

        ipc_bytes = _build_candle_batch(candle)

        assert isinstance(ipc_bytes, bytes)
        assert len(ipc_bytes) > 0

    def test_build_candle_batch_handles_missing_volume(self) -> None:
        """Candle batch should handle missing volume."""
        from backtest_engine.core.execution_simulator_rust import _build_candle_batch

        # Create candle without volume
        candle = MagicMock()
        candle.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        candle.open = 1.1000
        candle.high = 1.1050
        candle.low = 1.0950
        candle.close = 1.1020

        # Remove volume attribute
        del candle.volume

        ipc_bytes = _build_candle_batch(candle)
        assert isinstance(ipc_bytes, bytes)


class TestWrapperInstantiation:
    """Tests for ExecutionSimulator wrapper instantiation."""

    @pytest.fixture
    def mock_portfolio(self) -> MagicMock:
        """Create mock portfolio."""
        portfolio = MagicMock()
        portfolio.open_positions = []
        return portfolio

    @pytest.fixture
    def symbol_specs(self) -> dict:
        """Create test symbol specs."""
        spec = MagicMock()
        spec.pip_size = 0.0001
        spec.contract_size = 100000.0
        spec.volume_min = 0.01
        spec.volume_step = 0.01
        spec.volume_max = 100.0
        spec.tick_size = 0.0001
        spec.tick_value = 10.0
        return {"EURUSD": spec}

    def test_wrapper_requires_rust_always(
        self, mock_portfolio: MagicMock, symbol_specs: dict
    ) -> None:
        """Wrapper should require OMEGA_USE_RUST_EXECUTION_SIMULATOR=always."""
        from backtest_engine.core.execution_simulator_rust import (
            ExecutionSimulatorRustWrapper,
        )

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "always"}):
            # Should not raise
            wrapper = ExecutionSimulatorRustWrapper(
                portfolio=mock_portfolio,
                risk_per_trade=100.0,
                symbol_specs=symbol_specs,
            )
            assert wrapper is not None

    def test_wrapper_rejects_non_always_flag(
        self, mock_portfolio: MagicMock, symbol_specs: dict
    ) -> None:
        """Wrapper should reject non-'always' feature flag values."""
        from backtest_engine.core.execution_simulator_rust import (
            ExecutionSimulatorRustWrapper,
        )

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "python"}):
            with pytest.raises(ValueError) as exc_info:
                ExecutionSimulatorRustWrapper(
                    portfolio=mock_portfolio,
                    risk_per_trade=100.0,
                    symbol_specs=symbol_specs,
                )

            assert "OMEGA_USE_RUST_EXECUTION_SIMULATOR=always" in str(exc_info.value)

    def test_wrapper_stores_portfolio_reference(
        self, mock_portfolio: MagicMock, symbol_specs: dict
    ) -> None:
        """Wrapper should store portfolio reference."""
        from backtest_engine.core.execution_simulator_rust import (
            ExecutionSimulatorRustWrapper,
        )

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "always"}):
            wrapper = ExecutionSimulatorRustWrapper(
                portfolio=mock_portfolio,
                risk_per_trade=100.0,
                symbol_specs=symbol_specs,
            )
            assert wrapper.portfolio is mock_portfolio

    def test_wrapper_stores_risk_per_trade(
        self, mock_portfolio: MagicMock, symbol_specs: dict
    ) -> None:
        """Wrapper should store risk_per_trade."""
        from backtest_engine.core.execution_simulator_rust import (
            ExecutionSimulatorRustWrapper,
        )

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "always"}):
            wrapper = ExecutionSimulatorRustWrapper(
                portfolio=mock_portfolio,
                risk_per_trade=250.0,
                symbol_specs=symbol_specs,
            )
            assert wrapper.risk_per_trade == 250.0


class TestExecutionSimulatorDelegation:
    """Tests for ExecutionSimulator -> Rust wrapper delegation."""

    @pytest.fixture
    def mock_portfolio(self) -> MagicMock:
        """Create mock portfolio."""
        portfolio = MagicMock()
        portfolio.open_positions = []
        return portfolio

    def test_execution_simulator_delegates_to_rust(
        self, mock_portfolio: MagicMock
    ) -> None:
        """ExecutionSimulator should delegate to Rust wrapper via __new__."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator
        from backtest_engine.core.execution_simulator_rust import (
            ExecutionSimulatorRustWrapper,
        )

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "always"}):
            simulator = ExecutionSimulator(
                portfolio=mock_portfolio,
                risk_per_trade=100.0,
            )

            # Should be an instance of the Rust wrapper
            assert isinstance(simulator, ExecutionSimulatorRustWrapper)

    def test_execution_simulator_rejects_non_always(
        self, mock_portfolio: MagicMock
    ) -> None:
        """ExecutionSimulator should reject non-'always' flag."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        with patch.dict(os.environ, {"OMEGA_USE_RUST_EXECUTION_SIMULATOR": "false"}):
            with pytest.raises(ValueError) as exc_info:
                ExecutionSimulator(
                    portfolio=mock_portfolio,
                    risk_per_trade=100.0,
                )

            assert "OMEGA_USE_RUST_EXECUTION_SIMULATOR=always" in str(exc_info.value)
