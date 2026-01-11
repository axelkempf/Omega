"""Test Phase 3: Arrow IPC candle decoding and evaluate_exits_batch."""

from datetime import datetime, timezone

import pyarrow as pa
import pytest


@pytest.fixture
def ohlcv_schema():
    """OHLCV schema matching arrow_schemas.py."""
    return pa.schema(
        [
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("open", pa.float64(), nullable=False),
            pa.field("high", pa.float64(), nullable=False),
            pa.field("low", pa.float64(), nullable=False),
            pa.field("close", pa.float64(), nullable=False),
            pa.field("volume", pa.float64(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ]
    )


def candles_to_ipc(candle_data: dict, schema: pa.Schema) -> bytes:
    """Convert candle data dict to Arrow IPC bytes."""
    table = pa.table(candle_data, schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue())


class TestPhase3ArrowExits:
    """Test evaluate_exits_batch with Arrow IPC."""

    def test_rust_module_import(self):
        """Verify omega_rust module can be imported."""
        from omega_rust import ExecutionSimulatorRust

        sim = ExecutionSimulatorRust()
        assert sim.active_position_count == 0

    def test_evaluate_exits_batch_exists(self):
        """Verify evaluate_exits_batch method exists."""
        from omega_rust import ExecutionSimulatorRust

        sim = ExecutionSimulatorRust()
        assert hasattr(sim, "evaluate_exits_batch")

    def test_stop_loss_exit_via_arrow_ipc(self, ohlcv_schema):
        """Test SL exit with Arrow IPC candle batch."""
        from omega_rust import ExecutionSimulatorRust

        # Create simulator
        sim = ExecutionSimulatorRust(risk_per_trade=100.0, pip_buffer_factor=0.0)
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0)

        # Add long market order (entry=1.1000, SL=1.0950, TP=1.1100)
        entry_ts = 1704067200_000_000
        sim.process_signal_single(
            entry_ts, "EURUSD", "long", "market", 1.1000, 1.0950, 1.1100
        )

        assert sim.open_position_count == 1
        assert sim.closed_position_count == 0

        # Create candle that hits SL
        candle_ts = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
        candle_data = {
            "timestamp": [candle_ts],
            "open": [1.1000],
            "high": [1.1010],
            "low": [1.0940],  # Below SL of 1.0950
            "close": [1.0960],
            "volume": [100.0],
            "valid": [True],
        }

        ipc_bytes = candles_to_ipc(candle_data, ohlcv_schema)
        closed_count = sim.evaluate_exits_batch(ipc_bytes, None)

        assert closed_count == 1
        assert sim.open_position_count == 0
        assert sim.closed_position_count == 1

    def test_take_profit_exit_via_arrow_ipc(self, ohlcv_schema):
        """Test TP exit with Arrow IPC candle batch."""
        from omega_rust import ExecutionSimulatorRust

        sim = ExecutionSimulatorRust(risk_per_trade=100.0, pip_buffer_factor=0.0)
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0)

        # Add long market order (entry=1.1000, SL=1.0950, TP=1.1100)
        entry_ts = 1704067200_000_000
        sim.process_signal_single(
            entry_ts, "EURUSD", "long", "market", 1.1000, 1.0950, 1.1100
        )

        # Create candle that hits TP
        candle_ts = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
        candle_data = {
            "timestamp": [candle_ts],
            "open": [1.1050],
            "high": [1.1110],  # Above TP of 1.1100
            "low": [1.1040],
            "close": [1.1090],
            "volume": [100.0],
            "valid": [True],
        }

        ipc_bytes = candles_to_ipc(candle_data, ohlcv_schema)
        closed_count = sim.evaluate_exits_batch(ipc_bytes, None)

        assert closed_count == 1
        assert sim.closed_position_count == 1

    def test_no_exit_when_sl_tp_not_hit(self, ohlcv_schema):
        """Test no exit when SL/TP not touched."""
        from omega_rust import ExecutionSimulatorRust

        sim = ExecutionSimulatorRust(risk_per_trade=100.0, pip_buffer_factor=0.0)
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0)

        entry_ts = 1704067200_000_000
        sim.process_signal_single(
            entry_ts, "EURUSD", "long", "market", 1.1000, 1.0950, 1.1100
        )

        # Candle that doesn't hit SL or TP
        candle_ts = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
        candle_data = {
            "timestamp": [candle_ts],
            "open": [1.1000],
            "high": [1.1080],  # Below TP
            "low": [1.0960],  # Above SL
            "close": [1.1070],
            "volume": [100.0],
            "valid": [True],
        }

        ipc_bytes = candles_to_ipc(candle_data, ohlcv_schema)
        closed_count = sim.evaluate_exits_batch(ipc_bytes, None)

        assert closed_count == 0
        assert sim.open_position_count == 1
        assert sim.closed_position_count == 0

    def test_invalid_candles_skipped(self, ohlcv_schema):
        """Test that candles with valid=False are skipped."""
        from omega_rust import ExecutionSimulatorRust

        sim = ExecutionSimulatorRust(risk_per_trade=100.0, pip_buffer_factor=0.0)
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0)

        entry_ts = 1704067200_000_000
        sim.process_signal_single(
            entry_ts, "EURUSD", "long", "market", 1.1000, 1.0950, 1.1100
        )

        # Candle that would hit SL but is marked invalid
        candle_ts = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
        candle_data = {
            "timestamp": [candle_ts],
            "open": [1.1000],
            "high": [1.1010],
            "low": [1.0940],  # Would hit SL
            "close": [1.0960],
            "volume": [100.0],
            "valid": [False],  # Invalid - should be skipped
        }

        ipc_bytes = candles_to_ipc(candle_data, ohlcv_schema)
        closed_count = sim.evaluate_exits_batch(ipc_bytes, None)

        # Position should remain open because invalid candle was skipped
        assert closed_count == 0
        assert sim.open_position_count == 1

    def test_multiple_candles_in_batch(self, ohlcv_schema):
        """Test processing multiple candles in a single batch."""
        from omega_rust import ExecutionSimulatorRust

        sim = ExecutionSimulatorRust(risk_per_trade=100.0, pip_buffer_factor=0.0)
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0)

        entry_ts = 1704067200_000_000
        sim.process_signal_single(
            entry_ts, "EURUSD", "long", "market", 1.1000, 1.0950, 1.1100
        )

        # Multiple candles, third one hits SL
        candle_data = {
            "timestamp": [
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            ],
            "open": [1.1000, 1.1010, 1.0970],
            "high": [1.1020, 1.1030, 1.0980],
            "low": [1.0970, 1.0960, 1.0940],  # Third hits SL
            "close": [1.1010, 1.0970, 1.0960],
            "volume": [100.0, 100.0, 100.0],
            "valid": [True, True, True],
        }

        ipc_bytes = candles_to_ipc(candle_data, ohlcv_schema)
        closed_count = sim.evaluate_exits_batch(ipc_bytes, None)

        assert closed_count == 1
        assert sim.closed_position_count == 1
