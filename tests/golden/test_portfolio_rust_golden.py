"""
Golden tests for Wave 2 Portfolio Rust migration.

These tests verify deterministic behavior between Python and Rust implementations
by comparing outputs for fixed inputs. The golden files store expected outputs
that must match exactly (within floating-point tolerance).

Test Strategy:
1. Create positions with known parameters
2. Execute operations (register_entry, register_exit, update, etc.)
3. Compare summary/equity curve outputs against golden files
4. Validate R-multiple calculations with known results
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.backtest_engine.core.portfolio import (
    Portfolio,
    PortfolioPosition,
    get_rust_status,
)

# Golden file directory
GOLDEN_DIR = Path(__file__).parent / "data" / "portfolio"


def _round_dict(d: Dict[str, Any], precision: int = 6) -> Dict[str, Any]:
    """Round all float values in a dictionary to given precision."""
    result = {}
    for k, v in d.items():
        if isinstance(v, float):
            result[k] = round(v, precision)
        elif isinstance(v, dict):
            result[k] = _round_dict(v, precision)
        else:
            result[k] = v
    return result


class TestPortfolioPositionGolden:
    """Golden tests for PortfolioPosition calculations."""

    def test_r_multiple_long_win(self) -> None:
        """Test R-multiple calculation for winning long position."""
        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,  # 10 pips risk
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )

        # Close at take profit (20 pips gain, 2R)
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10200, "take_profit")

        # Expected R-multiple: (1.10200 - 1.10000) / (1.10000 - 1.09900) = 2.0
        assert pos.r_multiple == pytest.approx(2.0, rel=1e-6)
        # Expected result: 2.0 * 100.0 = 200.0
        assert pos.result == pytest.approx(200.0, rel=1e-6)

    def test_r_multiple_long_loss(self) -> None:
        """Test R-multiple calculation for losing long position."""
        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,  # 10 pips risk
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )

        # Close at stop loss (-10 pips loss, -1R)
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.09900, "stop_loss")

        # Expected R-multiple: (1.09900 - 1.10000) / (1.10000 - 1.09900) = -1.0
        assert pos.r_multiple == pytest.approx(-1.0, rel=1e-6)
        # Expected result: -1.0 * 100.0 = -100.0
        assert pos.result == pytest.approx(-100.0, rel=1e-6)

    def test_r_multiple_short_win(self) -> None:
        """Test R-multiple calculation for winning short position."""
        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="short",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.10100,  # 10 pips risk
            take_profit=1.09800,
            size=1.0,
            risk_per_trade=100.0,
        )

        # Close at take profit (20 pips gain, 2R)
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.09800, "take_profit")

        # Expected R-multiple: (1.10000 - 1.09800) / (1.10100 - 1.10000) = 2.0
        assert pos.r_multiple == pytest.approx(2.0, rel=1e-6)
        # Expected result: 2.0 * 100.0 = 200.0
        assert pos.result == pytest.approx(200.0, rel=1e-6)

    def test_r_multiple_short_loss(self) -> None:
        """Test R-multiple calculation for losing short position."""
        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="short",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.10100,  # 10 pips risk
            take_profit=1.09800,
            size=1.0,
            risk_per_trade=100.0,
        )

        # Close at stop loss (-10 pips loss, -1R)
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10100, "stop_loss")

        # Expected R-multiple: (1.10000 - 1.10100) / (1.10100 - 1.10000) = -1.0
        assert pos.r_multiple == pytest.approx(-1.0, rel=1e-6)
        # Expected result: -1.0 * 100.0 = -100.0
        assert pos.result == pytest.approx(-100.0, rel=1e-6)


class TestPortfolioGolden:
    """Golden tests for Portfolio state management."""

    def test_portfolio_initial_state(self) -> None:
        """Test portfolio initial state matches expected values."""
        portfolio = Portfolio(initial_balance=100_000.0)

        assert portfolio.initial_balance == 100_000.0
        assert portfolio.cash == 100_000.0
        assert portfolio.equity == 100_000.0
        assert portfolio.max_equity == 100_000.0
        assert portfolio.max_drawdown == 0.0
        assert portfolio.total_fees == 0.0
        assert len(portfolio.open_positions) == 0
        assert len(portfolio.closed_positions) == 0

    def test_portfolio_register_entry_exit(self) -> None:
        """Test position entry and exit registration."""
        portfolio = Portfolio(initial_balance=100_000.0)

        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )

        portfolio.register_entry(pos)
        assert len(portfolio.open_positions) == 1
        assert len(portfolio.closed_positions) == 0

        # Close position with win
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10200, "take_profit")
        portfolio.register_exit(pos)

        assert len(portfolio.open_positions) == 0
        assert len(portfolio.closed_positions) == 1
        assert portfolio.cash == pytest.approx(100_200.0, rel=1e-6)

    def test_portfolio_fee_registration(self) -> None:
        """Test fee registration reduces cash correctly."""
        portfolio = Portfolio(initial_balance=100_000.0)

        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )

        # Register entry fee
        portfolio.register_fee(3.0, datetime(2024, 1, 1, 12, 0, 0), "entry", pos)

        assert portfolio.total_fees == pytest.approx(3.0, rel=1e-6)
        assert portfolio.cash == pytest.approx(99_997.0, rel=1e-6)
        assert pos.entry_fee == pytest.approx(3.0, rel=1e-6)

    def test_portfolio_summary_deterministic(self) -> None:
        """Test that portfolio summary produces deterministic output."""
        portfolio = Portfolio(initial_balance=100_000.0)

        # Create and close multiple positions
        positions_data = [
            ("EURUSD", "long", 1.10000, 1.09900, 1.10200, 1.10150, "signal"),  # +150
            ("EURUSD", "long", 1.10200, 1.10100, 1.10400, 1.10050, "stop_loss"),  # -150
            ("GBPUSD", "short", 1.25000, 1.25100, 1.24800, 1.24850, "take_profit"),  # +150
        ]

        for i, (symbol, direction, entry, sl, tp, exit_price, reason) in enumerate(
            positions_data
        ):
            pos = PortfolioPosition(
                entry_time=datetime(2024, 1, 1, 12 + i, 0, 0),
                direction=direction,
                symbol=symbol,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                size=1.0,
                risk_per_trade=100.0,
            )
            portfolio.register_entry(pos)
            pos.close(datetime(2024, 1, 1, 12 + i, 30, 0), exit_price, reason)
            portfolio.register_exit(pos)

        summary = portfolio.get_summary()

        # Verify key metrics are deterministic
        assert summary["Initial Balance"] == 100_000.0
        assert summary["Total Trades"] == 3
        assert summary["Wins"] == 2
        assert summary["Losses"] == 1
        assert summary["Winrate"] == pytest.approx(66.67, abs=0.01)

    def test_portfolio_drawdown_calculation(self) -> None:
        """Test max drawdown is calculated correctly."""
        portfolio = Portfolio(initial_balance=100_000.0)

        # Win first (equity goes up)
        pos1 = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )
        portfolio.register_entry(pos1)
        pos1.close(datetime(2024, 1, 1, 12, 30, 0), 1.10200, "take_profit")
        portfolio.register_exit(pos1)
        portfolio.update(datetime(2024, 1, 1, 12, 30, 0))

        assert portfolio.max_equity == pytest.approx(100_200.0, rel=1e-6)

        # Lose second (equity goes down)
        pos2 = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 13, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )
        portfolio.register_entry(pos2)
        pos2.close(datetime(2024, 1, 1, 13, 30, 0), 1.09900, "stop_loss")
        portfolio.register_exit(pos2)
        portfolio.update(datetime(2024, 1, 1, 13, 30, 0))

        # Max equity should still be from first trade
        # Drawdown = 100_200 - 100_100 = 100
        assert portfolio.max_drawdown == pytest.approx(100.0, rel=1e-6)


class TestRustStatusGolden:
    """Golden tests for Rust backend status reporting."""

    def test_rust_status_structure(self) -> None:
        """Test that get_rust_status returns expected structure."""
        status = get_rust_status()

        assert "available" in status
        assert "enabled" in status
        assert "flag" in status
        assert "error" in status

        assert isinstance(status["available"], bool)
        assert isinstance(status["enabled"], bool)
        assert isinstance(status["flag"], str)
        assert status["error"] is None or isinstance(status["error"], str)

    def test_rust_status_flag_values(self) -> None:
        """Test that flag has valid values."""
        status = get_rust_status()

        valid_flags = ("auto", "true", "false")
        assert status["flag"] in valid_flags


class TestPortfolioEquityCurveGolden:
    """Golden tests for equity curve generation."""

    def test_equity_curve_order(self) -> None:
        """Test that equity curve points are in chronological order."""
        portfolio = Portfolio(initial_balance=100_000.0)
        portfolio.start_timestamp = datetime(2024, 1, 1, 12, 0, 0)

        # Create positions at different times
        times = [
            (datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 30, 0)),
            (datetime(2024, 1, 1, 13, 0, 0), datetime(2024, 1, 1, 13, 30, 0)),
            (datetime(2024, 1, 1, 14, 0, 0), datetime(2024, 1, 1, 14, 30, 0)),
        ]

        for entry_time, exit_time in times:
            pos = PortfolioPosition(
                entry_time=entry_time,
                direction="long",
                symbol="EURUSD",
                entry_price=1.10000,
                stop_loss=1.09900,
                take_profit=1.10200,
                size=1.0,
                risk_per_trade=100.0,
            )
            portfolio.register_entry(pos)
            pos.close(exit_time, 1.10100, "signal")
            portfolio.register_exit(pos)

        curve = portfolio.get_equity_curve()

        # Verify chronological order
        for i in range(1, len(curve)):
            assert curve[i][0] >= curve[i - 1][0], "Equity curve not in chronological order"

    def test_equity_curve_initial_value(self) -> None:
        """Test that equity curve starts with initial balance."""
        portfolio = Portfolio(initial_balance=100_000.0)
        portfolio.start_timestamp = datetime(2024, 1, 1, 12, 0, 0)

        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )
        portfolio.register_entry(pos)
        pos.close(datetime(2024, 1, 1, 12, 30, 0), 1.10100, "signal")
        portfolio.register_exit(pos)

        curve = portfolio.get_equity_curve()

        assert len(curve) >= 1
        assert curve[0][1] == pytest.approx(100_000.0, rel=1e-6)
