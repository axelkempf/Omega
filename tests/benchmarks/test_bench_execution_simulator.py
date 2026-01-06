# -*- coding: utf-8 -*-
"""
Benchmark Suite für ExecutionSimulator (P6-05).

Testet alle public functions des ExecutionSimulator-Moduls:
- Signalverarbeitung (Market, Limit, Stop)
- Entry-Trigger-Checks
- Position-Sizing und Quantization
- SL/TP-Checks

Ergebnisse sind in JSON exportierbar für Regression-Detection.

Verwendung:
    pytest tests/benchmarks/test_bench_execution_simulator.py -v
    pytest tests/benchmarks/test_bench_execution_simulator.py --benchmark-json=output.json
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest_engine.core.portfolio import Portfolio, PortfolioPosition
from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from backtest_engine.data.candle import Candle

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
    generate_synthetic_ohlcv,
)

# ══════════════════════════════════════════════════════════════════════════════
# MOCK OBJECTS FOR BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MockTradeSignal:
    """Minimal TradeSignal für Benchmarks."""

    timestamp: datetime
    direction: str
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    type: str = "market"
    meta: Optional[Dict[str, Any]] = None


def generate_mock_candles(n: int, seed: int = BENCHMARK_SEED) -> List[Candle]:
    """Generiert Mock-Candles für Benchmarks."""
    rng = np.random.default_rng(seed)
    df = generate_synthetic_ohlcv(n, seed=seed)
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    candles = []
    for i in range(len(df)):
        candles.append(
            Candle(
                timestamp=base_time + timedelta(minutes=i * 15),
                open=float(df.iloc[i]["open"]),
                high=float(df.iloc[i]["high"]),
                low=float(df.iloc[i]["low"]),
                close=float(df.iloc[i]["close"]),
                volume=float(df.iloc[i]["volume"]),
            )
        )
    return candles


def generate_mock_signals(
    n: int,
    candles: List[Candle],
    seed: int = BENCHMARK_SEED,
    order_type: str = "market",
) -> List[MockTradeSignal]:
    """Generiert Mock-Signals für Benchmarks."""
    rng = np.random.default_rng(seed)
    signals = []

    for i in range(min(n, len(candles) - 10)):
        candle = candles[i]
        direction = "long" if rng.random() > 0.5 else "short"
        entry_price = candle.close

        if direction == "long":
            sl_distance = rng.uniform(0.0005, 0.002)
            tp_distance = rng.uniform(0.001, 0.004)
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            sl_distance = rng.uniform(0.0005, 0.002)
            tp_distance = rng.uniform(0.001, 0.004)
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        signals.append(
            MockTradeSignal(
                timestamp=candle.timestamp,
                direction=direction,
                symbol="EURUSD",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                type=order_type,
            )
        )
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def candles_small() -> List[Candle]:
    """1K Mock-Candles."""
    return generate_mock_candles(SMALL_CANDLE_COUNT)


@pytest.fixture
def candles_medium() -> List[Candle]:
    """10K Mock-Candles."""
    return generate_mock_candles(DEFAULT_CANDLE_COUNT)


@pytest.fixture
def candles_large() -> List[Candle]:
    """100K Mock-Candles."""
    return generate_mock_candles(LARGE_CANDLE_COUNT)


@pytest.fixture
def execution_simulator_basic():
    """Basis ExecutionSimulator ohne Models."""
    from backtest_engine.core.execution_simulator import ExecutionSimulator

    portfolio = Portfolio(initial_balance=10000.0)
    return ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)


@pytest.fixture
def execution_simulator_full():
    """ExecutionSimulator mit Slippage/Fee Models."""
    from backtest_engine.core.execution_simulator import ExecutionSimulator

    portfolio = Portfolio(initial_balance=10000.0)
    slippage = SlippageModel(fixed_pips=0.5, random_pips=0.5)
    fee = FeeModel(per_million=5.0)
    return ExecutionSimulator(
        portfolio=portfolio,
        risk_per_trade=100.0,
        slippage_model=slippage,
        fee_model=fee,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Signal Processing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="execution_simulator")
class TestSignalProcessingBenchmarks:
    """Benchmarks für Signalverarbeitung."""

    def test_process_signal_market_small(
        self, benchmark: Any, candles_small: List[Candle]
    ) -> None:
        """Benchmark: Market-Orders verarbeiten (100 Signals)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(100, candles_small, order_type="market")

        def process_signals() -> int:
            portfolio = Portfolio(initial_balance=10000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)
            for signal in signals:
                sim.process_signal(signal)
            return len(sim.active_positions)

        result = benchmark(process_signals)
        assert result >= 0

    def test_process_signal_market_medium(
        self, benchmark: Any, candles_medium: List[Candle]
    ) -> None:
        """Benchmark: Market-Orders verarbeiten (500 Signals)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(500, candles_medium, order_type="market")

        def process_signals() -> int:
            portfolio = Portfolio(initial_balance=100000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)
            for signal in signals:
                sim.process_signal(signal)
            return len(sim.active_positions)

        result = benchmark(process_signals)
        assert result >= 0

    @pytest.mark.benchmark_slow
    def test_process_signal_market_large(
        self, benchmark: Any, candles_large: List[Candle]
    ) -> None:
        """Benchmark: Market-Orders verarbeiten (2000 Signals)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(2000, candles_large, order_type="market")

        def process_signals() -> int:
            portfolio = Portfolio(initial_balance=1000000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)
            for signal in signals:
                sim.process_signal(signal)
            return len(sim.active_positions)

        result = benchmark(process_signals)
        assert result >= 0

    def test_process_signal_with_slippage_fee(
        self, benchmark: Any, candles_medium: List[Candle]
    ) -> None:
        """Benchmark: Market-Orders mit Slippage und Fees."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(500, candles_medium, order_type="market")

        def process_signals() -> int:
            portfolio = Portfolio(initial_balance=100000.0)
            slippage = SlippageModel(fixed_pips=0.5, random_pips=0.5)
            fee = FeeModel(per_million=5.0)
            sim = ExecutionSimulator(
                portfolio=portfolio,
                risk_per_trade=100.0,
                slippage_model=slippage,
                fee_model=fee,
            )
            for signal in signals:
                sim.process_signal(signal)
            return len(sim.active_positions)

        result = benchmark(process_signals)
        assert result >= 0

    def test_process_signal_limit_orders(
        self, benchmark: Any, candles_medium: List[Candle]
    ) -> None:
        """Benchmark: Limit-Orders verarbeiten (pending)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(500, candles_medium, order_type="limit")

        def process_signals() -> int:
            portfolio = Portfolio(initial_balance=100000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)
            for signal in signals:
                sim.process_signal(signal)
            return len(sim.active_positions)

        result = benchmark(process_signals)
        assert result >= 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Entry Trigger Checks
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="execution_simulator")
class TestEntryTriggerBenchmarks:
    """Benchmarks für Entry-Trigger-Checks."""

    def test_check_entry_triggered_small(
        self, benchmark: Any, candles_small: List[Candle]
    ) -> None:
        """Benchmark: Entry-Trigger prüfen (100 Positionen)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        portfolio = Portfolio(initial_balance=10000.0)
        sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)

        # Erstelle pending Positionen
        pending_positions = []
        for i in range(100):
            candle = candles_small[i]
            pos = PortfolioPosition(
                entry_time=candle.timestamp,
                direction="long" if i % 2 == 0 else "short",
                symbol="EURUSD",
                entry_price=candle.close * (1.001 if i % 2 == 0 else 0.999),
                stop_loss=candle.close * 0.99,
                take_profit=candle.close * 1.02,
                size=0.1,
                order_type="limit",
                status="pending",
            )
            pending_positions.append(pos)

        check_candle = candles_small[150]

        def check_triggers() -> int:
            count = 0
            for pos in pending_positions:
                if sim.check_if_entry_triggered(pos, check_candle, check_candle):
                    count += 1
            return count

        result = benchmark(check_triggers)
        assert result >= 0

    def test_check_entry_triggered_medium(
        self, benchmark: Any, candles_medium: List[Candle]
    ) -> None:
        """Benchmark: Entry-Trigger prüfen (500 Positionen)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        portfolio = Portfolio(initial_balance=10000.0)
        sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)

        pending_positions = []
        for i in range(500):
            candle = candles_medium[i]
            pos = PortfolioPosition(
                entry_time=candle.timestamp,
                direction="long" if i % 2 == 0 else "short",
                symbol="EURUSD",
                entry_price=candle.close * (1.001 if i % 2 == 0 else 0.999),
                stop_loss=candle.close * 0.99,
                take_profit=candle.close * 1.02,
                size=0.1,
                order_type="stop",
                status="pending",
            )
            pending_positions.append(pos)

        check_candle = candles_medium[1000]

        def check_triggers() -> int:
            count = 0
            for pos in pending_positions:
                if sim.check_if_entry_triggered(pos, check_candle, check_candle):
                    count += 1
            return count

        result = benchmark(check_triggers)
        assert result >= 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Position Sizing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="execution_simulator")
class TestPositionSizingBenchmarks:
    """Benchmarks für Position-Sizing-Berechnungen."""

    def test_quantize_volume_batch(self, benchmark: Any) -> None:
        """Benchmark: Volume-Quantization (1000 Iterationen)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        portfolio = Portfolio(initial_balance=10000.0)
        sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)

        rng = np.random.default_rng(BENCHMARK_SEED)
        raw_volumes = rng.uniform(0.01, 10.0, 1000)

        def quantize_batch() -> List[float]:
            return [sim._quantize_volume("EURUSD", v) for v in raw_volumes]

        result = benchmark(quantize_batch)
        assert len(result) == 1000

    def test_pip_size_lookup_batch(self, benchmark: Any) -> None:
        """Benchmark: Pip-Size-Lookup (1000 Iterationen)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        portfolio = Portfolio(initial_balance=10000.0)
        sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)

        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"] * 200

        def lookup_batch() -> List[float]:
            return [sim._pip_size_for_symbol(s) for s in symbols]

        result = benchmark(lookup_batch)
        assert len(result) == 1000

    def test_unit_value_calculation_batch(self, benchmark: Any) -> None:
        """Benchmark: Unit-Value-Berechnung (1000 Iterationen)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator
        from backtest_engine.sizing.rate_provider import StaticRateProvider

        portfolio = Portfolio(initial_balance=10000.0)
        # Deterministic FX rates for quote->account conversions.
        # Only pairs needed by this benchmark are provided.
        rp = StaticRateProvider(
            rates={
                "JPYUSD": 0.0091,
                "CADUSD": 0.74,
            },
            strict=True,
        )
        sim = ExecutionSimulator(
            portfolio=portfolio, risk_per_trade=100.0, rate_provider=rp
        )

        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"] * 200

        def calc_batch() -> List[float]:
            # Clear cache for fair test
            sim._unit_value_cache.clear()
            return [sim._unit_value_per_price(s) for s in symbols]

        result = benchmark(calc_batch)
        assert len(result) == 1000


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Full Execution Cycle
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="execution_simulator")
class TestFullExecutionCycleBenchmarks:
    """Benchmarks für vollständige Execution-Zyklen."""

    def test_full_cycle_small(
        self, benchmark: Any, candles_small: List[Candle]
    ) -> None:
        """Benchmark: Vollständiger Execution-Zyklus (100 Trades)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(100, candles_small, order_type="market")

        def full_cycle() -> Dict[str, Any]:
            portfolio = Portfolio(initial_balance=10000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)

            # Process signals
            for signal in signals:
                sim.process_signal(signal)

            # Close all positions
            for pos in list(sim.active_positions):
                if pos.status == "open":
                    close_price = pos.entry_price * (
                        1.01 if pos.direction == "long" else 0.99
                    )
                    pos.close(
                        time=pos.entry_time + timedelta(hours=1),
                        price=close_price,
                        reason="benchmark_exit",
                    )
                    portfolio.register_exit(pos)

            return portfolio.get_summary()

        result = benchmark(full_cycle)
        assert "Total Trades" in result

    def test_full_cycle_medium_with_models(
        self, benchmark: Any, candles_medium: List[Candle]
    ) -> None:
        """Benchmark: Vollständiger Execution-Zyklus mit Models (500 Trades)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        signals = generate_mock_signals(500, candles_medium, order_type="market")

        def full_cycle() -> Dict[str, Any]:
            portfolio = Portfolio(initial_balance=100000.0)
            slippage = SlippageModel(fixed_pips=0.5, random_pips=0.5)
            fee = FeeModel(per_million=5.0)
            sim = ExecutionSimulator(
                portfolio=portfolio,
                risk_per_trade=100.0,
                slippage_model=slippage,
                fee_model=fee,
            )

            for signal in signals:
                sim.process_signal(signal)

            for pos in list(sim.active_positions):
                if pos.status == "open":
                    close_price = pos.entry_price * (
                        1.01 if pos.direction == "long" else 0.99
                    )
                    pos.close(
                        time=pos.entry_time + timedelta(hours=1),
                        price=close_price,
                        reason="benchmark_exit",
                    )
                    portfolio.register_exit(pos)

            return portfolio.get_summary()

        result = benchmark(full_cycle)
        assert "Total Trades" in result


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Throughput Metrics
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="execution_simulator")
class TestThroughputBenchmarks:
    """Throughput-Benchmarks für Performance-Baselines."""

    def test_signals_per_second_baseline(
        self, benchmark: Any, candles_large: List[Candle]
    ) -> None:
        """Benchmark: Signals pro Sekunde (Baseline für Rust-Vergleich)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        # 1000 Signals für messbare Zeit
        signals = generate_mock_signals(1000, candles_large, order_type="market")

        def process_all() -> int:
            portfolio = Portfolio(initial_balance=1000000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)
            for signal in signals:
                sim.process_signal(signal)
            return len(sim.active_positions)

        result = benchmark(process_all)
        assert result >= 0

    def test_mixed_order_types_throughput(
        self, benchmark: Any, candles_large: List[Candle]
    ) -> None:
        """Benchmark: Mixed Order Types (Market/Limit/Stop)."""
        from backtest_engine.core.execution_simulator import ExecutionSimulator

        market_signals = generate_mock_signals(
            334, candles_large, seed=1, order_type="market"
        )
        limit_signals = generate_mock_signals(
            333, candles_large, seed=2, order_type="limit"
        )
        stop_signals = generate_mock_signals(
            333, candles_large, seed=3, order_type="stop"
        )

        all_signals = market_signals + limit_signals + stop_signals

        def process_mixed() -> Tuple[int, int]:
            portfolio = Portfolio(initial_balance=1000000.0)
            sim = ExecutionSimulator(portfolio=portfolio, risk_per_trade=100.0)
            for signal in all_signals:
                sim.process_signal(signal)

            open_count = len([p for p in sim.active_positions if p.status == "open"])
            pending_count = len(
                [p for p in sim.active_positions if p.status == "pending"]
            )
            return open_count, pending_count

        result = benchmark(process_mixed)
        assert isinstance(result, tuple)


from typing import Tuple
