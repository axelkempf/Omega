"""
Parity Tests for Event Engine: Python vs Rust Backend Comparison.

Tests follow Wave 3 migration plan:
- Same inputs should produce identical outputs
- Determinism validation across backends
- Performance comparison metrics
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest_engine.core.event_engine import (
    EventEngine,
    get_active_backend,
    is_rust_available,
)
from backtest_engine.core.execution_simulator import ExecutionSimulator
from backtest_engine.core.portfolio import Portfolio
from backtest_engine.data.candle import Candle
from backtest_engine.strategy.strategy_wrapper import StrategyWrapper

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def deterministic_candles() -> tuple[List[Candle], List[Candle]]:
    """Generate deterministic bid and ask candle data for testing."""
    np.random.seed(42)
    n_bars = 500

    # Generate realistic price movements
    returns = np.random.normal(0, 0.001, n_bars)
    close = 1.1000 * np.cumprod(1 + returns)

    # OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    open_price = np.roll(close, 1)
    open_price[0] = 1.1000

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    timestamps = pd.date_range(
        "2024-01-01", periods=n_bars, freq="15min", tz="UTC"
    ).to_pydatetime()

    bid_candles = []
    ask_candles = []

    spread = 0.00010  # 1 pip spread

    for i in range(n_bars):
        bid_candles.append(
            Candle(
                timestamp=timestamps[i],
                open=open_price[i],
                high=high[i],
                low=low[i],
                close=close[i],
                volume=float(np.random.randint(100, 10000)),
                candle_type="bid",
            )
        )
        ask_candles.append(
            Candle(
                timestamp=timestamps[i],
                open=open_price[i] + spread,
                high=high[i] + spread,
                low=low[i] + spread,
                close=close[i] + spread,
                volume=float(np.random.randint(100, 10000)),
                candle_type="ask",
            )
        )

    return bid_candles, ask_candles


@pytest.fixture
def multi_candle_data(
    deterministic_candles: tuple[List[Candle], List[Candle]],
) -> Dict[str, Dict[str, List[Candle]]]:
    """Create multi-candle data structure for EventEngine."""
    bid_candles, ask_candles = deterministic_candles
    return {
        "M15": {
            "bid": bid_candles,
            "ask": ask_candles,
        }
    }


@pytest.fixture
def mock_strategy() -> MagicMock:
    """Create a mock strategy that returns deterministic signals."""
    np.random.seed(123)

    strategy = MagicMock(spec=StrategyWrapper)

    # Track call count for deterministic signal generation
    call_count = [0]
    signal_indices = [50, 150, 250, 350]  # Indices where signals occur

    def mock_evaluate(index: int, slice_map: Dict[str, Any]) -> Optional[Any]:
        call_count[0] += 1
        if index in signal_indices:
            # Return a mock signal
            signal = MagicMock()
            signal.direction = "long" if index % 2 == 0 else "short"
            signal.symbol = "EURUSD"
            signal.entry_price = 1.1000
            signal.stop_loss = 1.0950 if signal.direction == "long" else 1.1050
            signal.take_profit = 1.1050 if signal.direction == "long" else 1.0950
            return signal
        return None

    strategy.evaluate = mock_evaluate
    strategy.strategy = MagicMock()
    strategy.strategy.strategy = None  # No position manager

    return strategy


@pytest.fixture
def mock_executor() -> MagicMock:
    """Create a mock executor that tracks calls."""
    executor = MagicMock(spec=ExecutionSimulator)
    executor.active_positions = []
    executor.process_signal = MagicMock()
    executor.evaluate_exits = MagicMock()
    executor.evaluate_exits_from_dict = MagicMock()
    return executor


@pytest.fixture
def mock_portfolio() -> MagicMock:
    """Create a mock portfolio."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.update = MagicMock()
    portfolio.get_open_positions = MagicMock(return_value=[])
    return portfolio


# =============================================================================
# Parity Tests
# =============================================================================


class TestEventEngineParity:
    """Tests that verify Python and Rust backends produce identical results."""

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_same_signals_processed(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
    ):
        """
        Verify both backends process the same signals in the same order.
        """
        bid_candles, ask_candles = deterministic_candles

        # Track signals processed by each backend
        python_signals: List[int] = []
        rust_signals: List[int] = []

        # Create tracking executor for Python
        python_executor = MagicMock()
        python_executor.active_positions = []
        python_executor.process_signal = MagicMock(
            side_effect=lambda s: python_signals.append(1)
        )
        python_executor.evaluate_exits = MagicMock()

        # Create tracking executor for Rust
        rust_executor = MagicMock()
        rust_executor.active_positions = []
        rust_executor.process_signal = MagicMock(
            side_effect=lambda s: rust_signals.append(1)
        )
        rust_executor.evaluate_exits = MagicMock()
        rust_executor.evaluate_exits_from_dict = MagicMock()

        python_portfolio = MagicMock()
        python_portfolio.update = MagicMock()
        python_portfolio.get_open_positions = MagicMock(return_value=[])

        rust_portfolio = MagicMock()
        rust_portfolio.update = MagicMock()
        rust_portfolio.get_open_positions = MagicMock(return_value=[])

        start_dt = bid_candles[100].timestamp

        # Run Python backend
        python_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=python_executor,
            portfolio=python_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=False,
        )
        python_engine.run()

        # Reset strategy call tracking
        mock_strategy.evaluate = MagicMock(side_effect=mock_strategy.evaluate)

        # Run Rust backend
        rust_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=rust_executor,
            portfolio=rust_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=True,
        )
        rust_engine.run()

        # Verify same number of signals processed
        assert len(python_signals) == len(
            rust_signals
        ), f"Signal count mismatch: Python={len(python_signals)}, Rust={len(rust_signals)}"

        # Verify executors were called the same number of times
        assert (
            python_executor.process_signal.call_count
            == rust_executor.process_signal.call_count
        ), (
            f"process_signal call count mismatch: "
            f"Python={python_executor.process_signal.call_count}, "
            f"Rust={rust_executor.process_signal.call_count}"
        )

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_portfolio_updates_count_match(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
        mock_executor: MagicMock,
    ):
        """
        Verify both backends call portfolio.update() the same number of times.
        """
        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        # Python backend
        python_portfolio = MagicMock()
        python_portfolio.update = MagicMock()
        python_portfolio.get_open_positions = MagicMock(return_value=[])

        python_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=python_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=False,
        )
        python_engine.run()
        python_update_count = python_portfolio.update.call_count

        # Rust backend
        rust_portfolio = MagicMock()
        rust_portfolio.update = MagicMock()
        rust_portfolio.get_open_positions = MagicMock(return_value=[])

        rust_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=rust_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=True,
        )
        rust_engine.run()
        rust_update_count = rust_portfolio.update.call_count

        assert (
            python_update_count == rust_update_count
        ), f"Portfolio update count mismatch: Python={python_update_count}, Rust={rust_update_count}"

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_bars_processed_match(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
        mock_executor: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """
        Verify both backends process the same number of bars.
        """
        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        # Python backend - count strategy evaluations
        python_eval_count = [0]
        original_evaluate = mock_strategy.evaluate

        def python_counting_evaluate(index, slice_map):
            python_eval_count[0] += 1
            return original_evaluate(index, slice_map)

        mock_strategy.evaluate = python_counting_evaluate

        python_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=False,
        )
        python_engine.run()

        # Rust backend
        rust_eval_count = [0]

        def rust_counting_evaluate(index, slice_map):
            rust_eval_count[0] += 1
            return original_evaluate(index, slice_map)

        mock_strategy.evaluate = rust_counting_evaluate

        rust_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=True,
        )
        rust_engine.run()

        # Verify same number of strategy evaluations (= bars processed)
        assert (
            python_eval_count[0] == rust_eval_count[0]
        ), f"Bars processed mismatch: Python={python_eval_count[0]}, Rust={rust_eval_count[0]}"

        # Also check Rust stats if available
        if rust_engine.last_stats is not None:
            assert (
                rust_engine.last_stats.bars_processed == python_eval_count[0]
            ), f"Rust stats bars mismatch: stats={rust_engine.last_stats.bars_processed}, expected={python_eval_count[0]}"


class TestEventEnginePerformance:
    """Performance comparison tests between Python and Rust backends."""

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_rust_provides_timing_stats(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
        mock_executor: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """
        Verify Rust backend provides timing statistics.
        """
        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=True,
        )
        engine.run()

        stats = engine.last_stats
        assert stats is not None, "Rust backend should provide stats"
        assert stats.bars_processed > 0, "Should have processed bars"
        assert stats.loop_time_ms > 0, "Loop time should be recorded"
        assert stats.callback_time_ms >= 0, "Callback time should be recorded"

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    @pytest.mark.benchmark
    def test_performance_comparison(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
        mock_executor: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """
        Compare performance between Python and Rust backends.

        Note: This is an informational test - Rust should be faster but
        the exact speedup depends on workload and callback overhead.
        """
        import time

        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        # Run Python backend multiple times
        python_times = []
        for _ in range(3):
            python_portfolio = MagicMock()
            python_portfolio.update = MagicMock()
            python_portfolio.get_open_positions = MagicMock(return_value=[])

            start = time.perf_counter()
            python_engine = EventEngine(
                bid_candles=bid_candles,
                ask_candles=ask_candles,
                strategy=mock_strategy,
                executor=mock_executor,
                portfolio=python_portfolio,
                multi_candle_data=multi_candle_data,
                symbol="EURUSD",
                on_progress=None,
                original_start_dt=start_dt,
                use_rust=False,
            )
            python_engine.run()
            python_times.append(time.perf_counter() - start)

        # Run Rust backend multiple times
        rust_times = []
        for _ in range(3):
            rust_portfolio = MagicMock()
            rust_portfolio.update = MagicMock()
            rust_portfolio.get_open_positions = MagicMock(return_value=[])

            start = time.perf_counter()
            rust_engine = EventEngine(
                bid_candles=bid_candles,
                ask_candles=ask_candles,
                strategy=mock_strategy,
                executor=mock_executor,
                portfolio=rust_portfolio,
                multi_candle_data=multi_candle_data,
                symbol="EURUSD",
                on_progress=None,
                original_start_dt=start_dt,
                use_rust=True,
            )
            rust_engine.run()
            rust_times.append(time.perf_counter() - start)

        avg_python = sum(python_times) / len(python_times)
        avg_rust = sum(rust_times) / len(rust_times)

        # Log performance comparison (informational)
        print(f"\nPerformance comparison ({len(bid_candles) - 100} bars):")
        print(f"  Python avg: {avg_python * 1000:.2f}ms")
        print(f"  Rust avg:   {avg_rust * 1000:.2f}ms")
        if avg_rust > 0:
            print(f"  Speedup:    {avg_python / avg_rust:.2f}x")


class TestEventEngineRobustness:
    """Robustness tests for edge cases in both backends."""

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_empty_signals_handled(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_executor: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """
        Verify both backends handle strategies that never generate signals.
        """
        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        # Strategy that never generates signals
        no_signal_strategy = MagicMock()
        no_signal_strategy.evaluate = MagicMock(return_value=None)
        no_signal_strategy.strategy = MagicMock()
        no_signal_strategy.strategy.strategy = None

        # Python backend
        python_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=no_signal_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=False,
        )
        python_engine.run()
        python_calls = mock_executor.process_signal.call_count
        mock_executor.reset_mock()

        # Rust backend
        rust_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=no_signal_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=True,
        )
        rust_engine.run()
        rust_calls = mock_executor.process_signal.call_count

        assert python_calls == 0, "Python should not process signals for empty strategy"
        assert rust_calls == 0, "Rust should not process signals for empty strategy"

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_progress_callback_invoked(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
        mock_executor: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """
        Verify progress callback is invoked correctly by both backends.
        """
        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        python_progress_calls = []
        rust_progress_calls = []

        def python_progress(current, total):
            python_progress_calls.append((current, total))

        def rust_progress(current, total):
            rust_progress_calls.append((current, total))

        # Python backend
        python_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=python_progress,
            original_start_dt=start_dt,
            use_rust=False,
        )
        python_engine.run()

        # Rust backend
        rust_engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=rust_progress,
            original_start_dt=start_dt,
            use_rust=True,
        )
        rust_engine.run()

        # Both should have called progress the same number of times
        assert len(python_progress_calls) == len(
            rust_progress_calls
        ), f"Progress call count mismatch: Python={len(python_progress_calls)}, Rust={len(rust_progress_calls)}"

        # Final progress should be (total, total)
        if python_progress_calls:
            assert (
                python_progress_calls[-1][0] == python_progress_calls[-1][1]
            ), "Python final progress should be complete"
        if rust_progress_calls:
            assert (
                rust_progress_calls[-1][0] == rust_progress_calls[-1][1]
            ), "Rust final progress should be complete"


# =============================================================================
# CI Verification Tests
# =============================================================================


class TestCIVerification:
    """Tests for CI verification of backend usage."""

    def test_backend_selection_respects_flag(self):
        """Verify backend selection respects environment flag."""
        with patch.dict(os.environ, {"OMEGA_USE_RUST_EVENT_ENGINE": "false"}):
            # Need to reimport to pick up env change
            import importlib

            from src.backtest_engine.core import event_engine

            importlib.reload(event_engine)

            assert event_engine.get_active_backend() == "python"

        if is_rust_available():
            with patch.dict(os.environ, {"OMEGA_USE_RUST_EVENT_ENGINE": "true"}):
                import importlib

                from src.backtest_engine.core import event_engine

                importlib.reload(event_engine)

                assert event_engine.get_active_backend() == "rust"

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_rust_backend_actually_used_in_run(
        self,
        deterministic_candles: tuple[List[Candle], List[Candle]],
        multi_candle_data: Dict[str, Dict[str, List[Candle]]],
        mock_strategy: MagicMock,
        mock_executor: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """
        Verify Rust backend is actually used when requested.
        """
        bid_candles, ask_candles = deterministic_candles
        start_dt = bid_candles[100].timestamp

        engine = EventEngine(
            bid_candles=bid_candles,
            ask_candles=ask_candles,
            strategy=mock_strategy,
            executor=mock_executor,
            portfolio=mock_portfolio,
            multi_candle_data=multi_candle_data,
            symbol="EURUSD",
            on_progress=None,
            original_start_dt=start_dt,
            use_rust=True,
        )

        assert engine.active_backend == "rust", "Backend should be rust when requested"

        engine.run()

        # Rust backend should have stats
        assert engine.last_stats is not None, "Rust backend should populate last_stats"
        assert engine.last_stats.bars_processed > 0, "Should have processed bars"
