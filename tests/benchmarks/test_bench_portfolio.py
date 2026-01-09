# -*- coding: utf-8 -*-
"""
Benchmark Suite für Portfolio (P6-06).

Testet alle public functions des Portfolio-Moduls:
- Position-Registrierung (Entry/Exit)
- Equity-Updates und Drawdown-Berechnung
- Fee-Registrierung
- Summary-Generierung

Ergebnisse sind in JSON exportierbar für Regression-Detection.

Verwendung:
    pytest tests/benchmarks/test_bench_portfolio.py -v
    pytest tests/benchmarks/test_bench_portfolio.py --benchmark-json=output.json
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from backtest_engine.core.portfolio import Portfolio, PortfolioPosition

from .conftest import (
    BENCHMARK_SEED,
    DEFAULT_CANDLE_COUNT,
    LARGE_CANDLE_COUNT,
    SMALL_CANDLE_COUNT,
)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════


def generate_mock_positions(
    n: int, seed: int = BENCHMARK_SEED, closed: bool = False
) -> List[PortfolioPosition]:
    """Generiert Mock-Positionen für Benchmarks."""
    rng = np.random.default_rng(seed)
    positions = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(n):
        entry_time = base_time + timedelta(hours=i * 4)
        entry_price = 1.1000 + rng.normal(0, 0.01)
        direction = "long" if rng.random() > 0.5 else "short"

        if direction == "long":
            stop_loss = entry_price - rng.uniform(0.001, 0.003)
            take_profit = entry_price + rng.uniform(0.002, 0.006)
        else:
            stop_loss = entry_price + rng.uniform(0.001, 0.003)
            take_profit = entry_price - rng.uniform(0.002, 0.006)

        pos = PortfolioPosition(
            entry_time=entry_time,
            direction=direction,
            symbol="EURUSD",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=rng.uniform(0.1, 1.0),
            risk_per_trade=100.0,
            order_type="market",
            status="open",
        )
        pos.initial_stop_loss = stop_loss
        pos.initial_take_profit = take_profit

        if closed:
            exit_time = entry_time + timedelta(hours=int(rng.integers(1, 48)))
            # Random outcome: win (40%), loss (40%), BE (20%)
            outcome = rng.random()
            if outcome < 0.4:
                exit_price = take_profit  # Win
                reason = "take_profit"
            elif outcome < 0.8:
                exit_price = stop_loss  # Loss
                reason = "stop_loss"
            else:
                exit_price = entry_price  # Break-even
                reason = "manual"

            pos.close(time=exit_time, price=exit_price, reason=reason)

        positions.append(pos)

    return positions


def generate_equity_updates(
    n: int, seed: int = BENCHMARK_SEED
) -> List[Tuple[datetime, float]]:
    """Generiert Equity-Update-Zeitpunkte und Werte."""
    rng = np.random.default_rng(seed)
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    equity = 10000.0

    updates = []
    for i in range(n):
        time = base_time + timedelta(minutes=i * 15)
        # Random walk für Equity
        equity += rng.normal(0, 50)
        equity = max(1000, equity)  # Mindest-Equity
        updates.append((time, equity))

    return updates


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def portfolio_empty() -> Portfolio:
    """Leeres Portfolio."""
    return Portfolio(initial_balance=10000.0)


@pytest.fixture
def portfolio_with_positions() -> Portfolio:
    """Portfolio mit 100 offenen Positionen."""
    portfolio = Portfolio(initial_balance=100000.0)
    positions = generate_mock_positions(100, closed=False)
    for pos in positions:
        portfolio.register_entry(pos)
    return portfolio


@pytest.fixture
def positions_small() -> List[PortfolioPosition]:
    """100 Mock-Positionen."""
    return generate_mock_positions(100)


@pytest.fixture
def positions_medium() -> List[PortfolioPosition]:
    """500 Mock-Positionen."""
    return generate_mock_positions(500)


@pytest.fixture
def positions_large() -> List[PortfolioPosition]:
    """2000 Mock-Positionen."""
    return generate_mock_positions(2000)


@pytest.fixture
def closed_positions_medium() -> List[PortfolioPosition]:
    """500 geschlossene Mock-Positionen."""
    return generate_mock_positions(500, closed=True)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Position Registration
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestPositionRegistrationBenchmarks:
    """Benchmarks für Position-Registrierung."""

    def test_register_entry_small(
        self, benchmark: Any, positions_small: List[PortfolioPosition]
    ) -> None:
        """Benchmark: Entry-Registrierung (100 Positionen)."""

        def register_entries() -> int:
            portfolio = Portfolio(initial_balance=10000.0)
            for pos in positions_small:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(register_entries)
        assert result == 100

    def test_register_entry_medium(
        self, benchmark: Any, positions_medium: List[PortfolioPosition]
    ) -> None:
        """Benchmark: Entry-Registrierung (500 Positionen)."""

        def register_entries() -> int:
            portfolio = Portfolio(initial_balance=100000.0)
            for pos in positions_medium:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(register_entries)
        assert result == 500

    @pytest.mark.benchmark_slow
    def test_register_entry_large(
        self, benchmark: Any, positions_large: List[PortfolioPosition]
    ) -> None:
        """Benchmark: Entry-Registrierung (2000 Positionen)."""

        def register_entries() -> int:
            portfolio = Portfolio(initial_balance=1000000.0)
            for pos in positions_large:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(register_entries)
        assert result == 2000

    def test_register_exit_small(self, benchmark: Any) -> None:
        """Benchmark: Exit-Registrierung (100 Positionen)."""
        positions = generate_mock_positions(100, closed=True)

        def register_exits() -> int:
            portfolio = Portfolio(initial_balance=100000.0)
            # First register entries
            for pos in positions:
                pos.status = "open"
                pos.is_closed = False
                portfolio.register_entry(pos)

            # Then register exits
            for pos in positions:
                pos.status = "open"  # Reset for exit
                portfolio.register_exit(pos)

            return len(portfolio.closed_positions)

        result = benchmark(register_exits)
        assert result >= 0

    def test_register_exit_medium(self, benchmark: Any) -> None:
        """Benchmark: Exit-Registrierung (500 Positionen)."""
        positions = generate_mock_positions(500, closed=True)

        def register_exits() -> int:
            portfolio = Portfolio(initial_balance=500000.0)
            for pos in positions:
                pos.status = "open"
                pos.is_closed = False
                portfolio.register_entry(pos)

            for pos in positions:
                pos.status = "open"
                portfolio.register_exit(pos)

            return len(portfolio.closed_positions)

        result = benchmark(register_exits)
        assert result >= 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Equity Updates
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestEquityUpdateBenchmarks:
    """Benchmarks für Equity-Updates und Drawdown-Berechnung."""

    def test_update_small(self, benchmark: Any) -> None:
        """Benchmark: Portfolio-Update (1K Updates)."""
        updates = generate_equity_updates(SMALL_CANDLE_COUNT)

        def run_updates() -> float:
            portfolio = Portfolio(initial_balance=10000.0)
            for time, _ in updates:
                portfolio.update(time)
            return portfolio.max_drawdown

        result = benchmark(run_updates)
        assert result >= 0

    def test_update_medium(self, benchmark: Any) -> None:
        """Benchmark: Portfolio-Update (10K Updates)."""
        updates = generate_equity_updates(DEFAULT_CANDLE_COUNT)

        def run_updates() -> float:
            portfolio = Portfolio(initial_balance=10000.0)
            for time, _ in updates:
                portfolio.update(time)
            return portfolio.max_drawdown

        result = benchmark(run_updates)
        assert result >= 0

    @pytest.mark.benchmark_slow
    def test_update_large(self, benchmark: Any) -> None:
        """Benchmark: Portfolio-Update (100K Updates)."""
        updates = generate_equity_updates(LARGE_CANDLE_COUNT)

        def run_updates() -> float:
            portfolio = Portfolio(initial_balance=10000.0)
            for time, _ in updates:
                portfolio.update(time)
            return portfolio.max_drawdown

        result = benchmark(run_updates)
        assert result >= 0

    def test_equity_curve_growth(self, benchmark: Any) -> None:
        """Benchmark: Equity-Curve-Wachstum mit vielen Updates."""
        n_updates = 10000

        def run_updates() -> int:
            portfolio = Portfolio(initial_balance=10000.0)
            base_time = datetime(2024, 1, 1)
            for i in range(n_updates):
                portfolio.update(base_time + timedelta(minutes=i))
            return len(portfolio.equity_curve)

        result = benchmark(run_updates)
        assert result > 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Fee Registration
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestFeeRegistrationBenchmarks:
    """Benchmarks für Fee-Registrierung."""

    def test_register_fee_small(self, benchmark: Any) -> None:
        """Benchmark: Fee-Registrierung (100 Fees)."""

        def register_fees() -> float:
            portfolio = Portfolio(initial_balance=10000.0)
            base_time = datetime(2024, 1, 1)
            for i in range(100):
                portfolio.register_fee(
                    amount=5.0,
                    time=base_time + timedelta(hours=i),
                    kind="entry" if i % 2 == 0 else "exit",
                )
            return portfolio.total_fees

        result = benchmark(register_fees)
        assert result == 500.0

    def test_register_fee_medium(self, benchmark: Any) -> None:
        """Benchmark: Fee-Registrierung (500 Fees)."""

        def register_fees() -> float:
            portfolio = Portfolio(initial_balance=50000.0)
            base_time = datetime(2024, 1, 1)
            for i in range(500):
                portfolio.register_fee(
                    amount=5.0,
                    time=base_time + timedelta(hours=i),
                    kind="entry" if i % 2 == 0 else "exit",
                )
            return portfolio.total_fees

        result = benchmark(register_fees)
        assert result == 2500.0

    def test_register_fee_with_position(self, benchmark: Any) -> None:
        """Benchmark: Fee-Registrierung mit Position-Zuordnung."""
        positions = generate_mock_positions(200)

        def register_fees() -> float:
            portfolio = Portfolio(initial_balance=100000.0)
            base_time = datetime(2024, 1, 1)

            for i, pos in enumerate(positions):
                portfolio.register_fee(
                    amount=5.0,
                    time=base_time + timedelta(hours=i),
                    kind="entry",
                    position=pos,
                )
                portfolio.register_fee(
                    amount=5.0,
                    time=base_time + timedelta(hours=i + 1),
                    kind="exit",
                    position=pos,
                )

            return portfolio.total_fees

        result = benchmark(register_fees)
        assert result == 2000.0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Summary Generation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestSummaryBenchmarks:
    """Benchmarks für Summary-Generierung."""

    def test_get_summary_small(self, benchmark: Any) -> None:
        """Benchmark: Summary-Generierung (100 Trades)."""
        positions = generate_mock_positions(100, closed=True)

        portfolio = Portfolio(initial_balance=100000.0)
        for pos in positions:
            pos.status = "open"
            pos.is_closed = False
            portfolio.register_entry(pos)
            pos.status = "open"
            portfolio.register_exit(pos)

        def get_summary() -> Dict[str, float]:
            return portfolio.get_summary()

        result = benchmark(get_summary)
        assert "Total Trades" in result

    def test_get_summary_medium(self, benchmark: Any) -> None:
        """Benchmark: Summary-Generierung (500 Trades)."""
        positions = generate_mock_positions(500, closed=True)

        portfolio = Portfolio(initial_balance=500000.0)
        for pos in positions:
            pos.status = "open"
            pos.is_closed = False
            portfolio.register_entry(pos)
            pos.status = "open"
            portfolio.register_exit(pos)

        def get_summary() -> Dict[str, float]:
            return portfolio.get_summary()

        result = benchmark(get_summary)
        assert "Total Trades" in result

    @pytest.mark.benchmark_slow
    def test_get_summary_large(self, benchmark: Any) -> None:
        """Benchmark: Summary-Generierung (2000 Trades)."""
        positions = generate_mock_positions(2000, closed=True)

        portfolio = Portfolio(initial_balance=2000000.0)
        for pos in positions:
            pos.status = "open"
            pos.is_closed = False
            portfolio.register_entry(pos)
            pos.status = "open"
            portfolio.register_exit(pos)

        def get_summary() -> Dict[str, float]:
            return portfolio.get_summary()

        result = benchmark(get_summary)
        assert "Total Trades" in result


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Position Queries
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestPositionQueryBenchmarks:
    """Benchmarks für Position-Abfragen."""

    def test_get_open_positions_all(
        self, benchmark: Any, portfolio_with_positions: Portfolio
    ) -> None:
        """Benchmark: Alle offenen Positionen abrufen."""
        portfolio = portfolio_with_positions

        def get_positions() -> List[PortfolioPosition]:
            return portfolio.get_open_positions()

        result = benchmark(get_positions)
        assert len(result) == 100

    def test_get_open_positions_filtered(
        self, benchmark: Any, portfolio_with_positions: Portfolio
    ) -> None:
        """Benchmark: Gefilterte offene Positionen abrufen."""
        portfolio = portfolio_with_positions

        def get_positions() -> List[PortfolioPosition]:
            return portfolio.get_open_positions(symbol="EURUSD")

        result = benchmark(get_positions)
        assert len(result) == 100

    def test_position_to_dict_batch(self, benchmark: Any) -> None:
        """Benchmark: Position-Export (500 Positionen)."""
        positions = generate_mock_positions(500, closed=True)

        def export_all() -> List[Dict[str, Any]]:
            return [pos.to_dict() for pos in positions]

        result = benchmark(export_all)
        assert len(result) == 500


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Full Portfolio Lifecycle
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestFullLifecycleBenchmarks:
    """Benchmarks für vollständige Portfolio-Lebenszyklen."""

    def test_full_lifecycle_small(self, benchmark: Any) -> None:
        """Benchmark: Vollständiger Lebenszyklus (100 Trades)."""
        positions = generate_mock_positions(100, closed=True)

        def lifecycle() -> Dict[str, float]:
            portfolio = Portfolio(initial_balance=100000.0)
            base_time = datetime(2024, 1, 1)

            for i, pos in enumerate(positions):
                # Reset position state
                pos.status = "open"
                pos.is_closed = False

                # Entry
                portfolio.register_entry(pos)
                portfolio.register_fee(
                    5.0, base_time + timedelta(hours=i), "entry", pos
                )
                portfolio.update(base_time + timedelta(hours=i))

                # Exit
                pos.status = "open"
                portfolio.register_exit(pos)
                portfolio.register_fee(
                    5.0, base_time + timedelta(hours=i + 1), "exit", pos
                )
                portfolio.update(base_time + timedelta(hours=i + 1))

            return portfolio.get_summary()

        result = benchmark(lifecycle)
        assert result["Total Trades"] == 100

    def test_full_lifecycle_medium(self, benchmark: Any) -> None:
        """Benchmark: Vollständiger Lebenszyklus (500 Trades)."""
        positions = generate_mock_positions(500, closed=True)

        def lifecycle() -> Dict[str, float]:
            portfolio = Portfolio(initial_balance=500000.0)
            base_time = datetime(2024, 1, 1)

            for i, pos in enumerate(positions):
                pos.status = "open"
                pos.is_closed = False

                portfolio.register_entry(pos)
                portfolio.register_fee(
                    5.0, base_time + timedelta(hours=i), "entry", pos
                )
                portfolio.update(base_time + timedelta(hours=i))

                pos.status = "open"
                portfolio.register_exit(pos)
                portfolio.register_fee(
                    5.0, base_time + timedelta(hours=i + 1), "exit", pos
                )
                portfolio.update(base_time + timedelta(hours=i + 1))

            return portfolio.get_summary()

        result = benchmark(lifecycle)
        assert result["Total Trades"] == 500


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Throughput Baselines
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio")
class TestThroughputBaselines:
    """Throughput-Baselines für Rust-Vergleich."""

    def test_entries_per_second(self, benchmark: Any) -> None:
        """Baseline: Entries pro Sekunde."""
        positions = generate_mock_positions(1000)

        def register_all() -> int:
            portfolio = Portfolio(initial_balance=1000000.0)
            for pos in positions:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(register_all)
        assert result == 1000

    def test_updates_per_second(self, benchmark: Any) -> None:
        """Baseline: Updates pro Sekunde."""

        def run_updates() -> int:
            portfolio = Portfolio(initial_balance=10000.0)
            base_time = datetime(2024, 1, 1)
            for i in range(5000):
                portfolio.update(base_time + timedelta(minutes=i))
            return len(portfolio.equity_curve)

        result = benchmark(run_updates)
        assert result > 0

    def test_summaries_per_second(self, benchmark: Any) -> None:
        """Baseline: Summaries pro Sekunde."""
        positions = generate_mock_positions(200, closed=True)
        portfolio = Portfolio(initial_balance=200000.0)
        for pos in positions:
            pos.status = "open"
            pos.is_closed = False
            portfolio.register_entry(pos)
            pos.status = "open"
            portfolio.register_exit(pos)

        def generate_summaries() -> int:
            count = 0
            for _ in range(100):
                _ = portfolio.get_summary()
                count += 1
            return count

        result = benchmark(generate_summaries)
        assert result == 100


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK: Batch Processing (Wave 2)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.benchmark(group="portfolio_batch")
class TestBatchProcessingBenchmarks:
    """Benchmarks für Batch-Processing-API (Wave 2).

    Vergleicht sequenzielle Operationen mit Batch-Operationen,
    um den FFI-Overhead zu quantifizieren.
    """

    def test_batch_entry_1k(self, benchmark: Any) -> None:
        """Benchmark: Batch Entry-Registrierung (1000 Positionen)."""
        positions = generate_mock_positions(1000)

        def batch_entries() -> int:
            portfolio = Portfolio(initial_balance=1000000.0)
            ops = [{"type": "entry", "position": pos} for pos in positions]
            result = portfolio.process_batch(ops)
            return result.entries_registered

        result = benchmark(batch_entries)
        assert result == 1000

    def test_sequential_entry_1k(self, benchmark: Any) -> None:
        """Benchmark: Sequentielle Entry-Registrierung (1000 Positionen)."""
        positions = generate_mock_positions(1000)

        def sequential_entries() -> int:
            portfolio = Portfolio(initial_balance=1000000.0)
            for pos in positions:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(sequential_entries)
        assert result == 1000

    def test_batch_entry_5k(self, benchmark: Any) -> None:
        """Benchmark: Batch Entry-Registrierung (5000 Positionen)."""
        positions = generate_mock_positions(5000)

        def batch_entries() -> int:
            portfolio = Portfolio(initial_balance=5000000.0)
            ops = [{"type": "entry", "position": pos} for pos in positions]
            result = portfolio.process_batch(ops)
            return result.entries_registered

        result = benchmark(batch_entries)
        assert result == 5000

    def test_sequential_entry_5k(self, benchmark: Any) -> None:
        """Benchmark: Sequentielle Entry-Registrierung (5000 Positionen)."""
        positions = generate_mock_positions(5000)

        def sequential_entries() -> int:
            portfolio = Portfolio(initial_balance=5000000.0)
            for pos in positions:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(sequential_entries)
        assert result == 5000

    @pytest.mark.benchmark_slow
    def test_batch_entry_20k(self, benchmark: Any) -> None:
        """Benchmark: Batch Entry-Registrierung (20000 Positionen).

        Primärer Test für FFI-Overhead-Reduktion (Ziel: 2-3x Speedup).
        """
        positions = generate_mock_positions(20000, seed=42)

        def batch_entries() -> int:
            portfolio = Portfolio(initial_balance=20000000.0)
            ops = [{"type": "entry", "position": pos} for pos in positions]
            result = portfolio.process_batch(ops)
            return result.entries_registered

        result = benchmark(batch_entries)
        assert result == 20000

    @pytest.mark.benchmark_slow
    def test_sequential_entry_20k(self, benchmark: Any) -> None:
        """Benchmark: Sequentielle Entry-Registrierung (20000 Positionen).

        Baseline für Vergleich mit Batch-Processing.
        """
        positions = generate_mock_positions(20000, seed=42)

        def sequential_entries() -> int:
            portfolio = Portfolio(initial_balance=20000000.0)
            for pos in positions:
                portfolio.register_entry(pos)
            return len(portfolio.open_positions)

        result = benchmark(sequential_entries)
        assert result == 20000

    def test_batch_mixed_ops_1k(self, benchmark: Any) -> None:
        """Benchmark: Batch mit gemischten Operationen (Entry + Update + Fee)."""
        positions = generate_mock_positions(1000)
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        def batch_mixed() -> Dict[str, int]:
            portfolio = Portfolio(initial_balance=1000000.0)
            ops: List[Dict[str, Any]] = []

            for i, pos in enumerate(positions):
                ops.append({"type": "entry", "position": pos, "fee": 5.0})
                ops.append({"type": "update", "time": base_time + timedelta(hours=i)})

            result = portfolio.process_batch(ops)
            return {
                "entries": result.entries_registered,
                "updates": result.updates_performed,
                "fees": result.fees_registered,
            }

        result = benchmark(batch_mixed)
        assert result["entries"] == 1000
        assert result["updates"] == 1000
        assert result["fees"] == 1000

    def test_sequential_mixed_ops_1k(self, benchmark: Any) -> None:
        """Benchmark: Sequentielle gemischte Operationen (Entry + Update + Fee)."""
        positions = generate_mock_positions(1000)
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        def sequential_mixed() -> Dict[str, int]:
            portfolio = Portfolio(initial_balance=1000000.0)
            entries, updates, fees = 0, 0, 0

            for i, pos in enumerate(positions):
                portfolio.register_entry(pos)
                entries += 1
                portfolio.register_fee(5.0, base_time + timedelta(hours=i), "entry", pos)
                fees += 1
                portfolio.update(base_time + timedelta(hours=i))
                updates += 1

            return {"entries": entries, "updates": updates, "fees": fees}

        result = benchmark(sequential_mixed)
        assert result["entries"] == 1000
        assert result["updates"] == 1000
        assert result["fees"] == 1000

    def test_batch_update_only_10k(self, benchmark: Any) -> None:
        """Benchmark: Batch Updates nur (10000 Updates)."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        def batch_updates() -> int:
            portfolio = Portfolio(initial_balance=10000.0)
            ops = [
                {"type": "update", "time": base_time + timedelta(minutes=i)}
                for i in range(10000)
            ]
            result = portfolio.process_batch(ops)
            return result.updates_performed

        result = benchmark(batch_updates)
        assert result == 10000

    def test_sequential_update_only_10k(self, benchmark: Any) -> None:
        """Benchmark: Sequentielle Updates (10000 Updates)."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        def sequential_updates() -> int:
            portfolio = Portfolio(initial_balance=10000.0)
            for i in range(10000):
                portfolio.update(base_time + timedelta(minutes=i))
            return len(portfolio.equity_curve)

        result = benchmark(sequential_updates)
        assert result > 0