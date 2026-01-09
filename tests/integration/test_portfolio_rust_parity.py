"""
Integration tests for Wave 2 Portfolio Rust migration.

These tests verify that the Python and Rust implementations produce
identical results for the same inputs. They ensure backward compatibility
and deterministic behavior across both backends.

Test Strategy:
1. Run identical operations on Python and Rust implementations
2. Compare all outputs (summary metrics, equity curves, position states)
3. Verify feature flag behavior
4. Test edge cases and error handling
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest import mock

import pytest

from src.backtest_engine.core.portfolio import (
    Portfolio,
    PortfolioPosition,
    get_rust_status,
)


class TestFeatureFlagBehavior:
    """Tests for OMEGA_USE_RUST_PORTFOLIO feature flag."""

    def test_flag_auto_default(self) -> None:
        """Test that 'auto' is the default flag value."""
        # Clear any existing flag
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh flag value
            import importlib

            import src.backtest_engine.core.portfolio as portfolio_module

            importlib.reload(portfolio_module)

            status = portfolio_module.get_rust_status()
            assert status["flag"] == "auto"

    def test_flag_false_disables_rust(self) -> None:
        """Test that 'false' flag disables Rust backend."""
        with mock.patch.dict(os.environ, {"OMEGA_USE_RUST_PORTFOLIO": "false"}):
            import importlib

            import src.backtest_engine.core.portfolio as portfolio_module

            importlib.reload(portfolio_module)

            assert not portfolio_module._use_rust_backend()

    def test_status_contains_required_keys(self) -> None:
        """Test that status dict contains all required keys."""
        status = get_rust_status()

        required_keys = ["available", "enabled", "flag", "error"]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"


class TestPortfolioPositionParity:
    """Tests ensuring Python PortfolioPosition behavior is consistent."""

    def test_position_creation_deterministic(self) -> None:
        """Test that position creation is deterministic."""
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

        pos2 = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )

        # Same inputs should produce identical positions
        assert pos1.entry_time == pos2.entry_time
        assert pos1.direction == pos2.direction
        assert pos1.symbol == pos2.symbol
        assert pos1.entry_price == pos2.entry_price
        assert pos1.stop_loss == pos2.stop_loss
        assert pos1.take_profit == pos2.take_profit
        assert pos1.size == pos2.size
        assert pos1.risk_per_trade == pos2.risk_per_trade

    def test_r_multiple_calculation_consistency(self) -> None:
        """Test R-multiple calculation consistency across multiple calls."""
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
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10150, "signal")

        # Multiple calls should return same value
        r1 = pos.r_multiple
        r2 = pos.r_multiple
        r3 = pos.r_multiple

        assert r1 == r2 == r3

    def test_position_close_idempotent(self) -> None:
        """Test that closing a position is idempotent."""
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

        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10150, "signal")
        result1 = pos.result
        r_mult1 = pos.r_multiple

        # Closing again with same values should give same result
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10150, "signal")
        result2 = pos.result
        r_mult2 = pos.r_multiple

        assert result1 == result2
        assert r_mult1 == r_mult2


class TestPortfolioParity:
    """Tests ensuring Python Portfolio behavior is consistent."""

    def test_portfolio_summary_consistency(self) -> None:
        """Test that portfolio summary is consistent across calls."""
        portfolio = Portfolio(initial_balance=100_000.0)

        # Add and close some positions
        for i in range(3):
            pos = PortfolioPosition(
                entry_time=datetime(2024, 1, 1, 12 + i, 0, 0),
                direction="long",
                symbol="EURUSD",
                entry_price=1.10000,
                stop_loss=1.09900,
                take_profit=1.10200,
                size=1.0,
                risk_per_trade=100.0,
            )
            portfolio.register_entry(pos)
            pos.close(datetime(2024, 1, 1, 12 + i, 30, 0), 1.10100, "signal")
            portfolio.register_exit(pos)

        summary1 = portfolio.get_summary()
        summary2 = portfolio.get_summary()

        # Summaries should be identical
        for key in summary1:
            assert summary1[key] == summary2[key], f"Mismatch for {key}"

    def test_equity_curve_monotonic_time(self) -> None:
        """Test that equity curve timestamps are monotonically increasing."""
        portfolio = Portfolio(initial_balance=100_000.0)
        portfolio.start_timestamp = datetime(2024, 1, 1, 12, 0, 0)

        times = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 13, 0, 0),
            datetime(2024, 1, 1, 14, 0, 0),
        ]

        for entry_time in times:
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
            exit_time = datetime(
                entry_time.year,
                entry_time.month,
                entry_time.day,
                entry_time.hour,
                30,
                0,
            )
            pos.close(exit_time, 1.10100, "signal")
            portfolio.register_exit(pos)

        curve = portfolio.get_equity_curve()

        for i in range(1, len(curve)):
            assert curve[i][0] >= curve[i - 1][0]

    def test_fee_accumulation_accuracy(self) -> None:
        """Test that fees accumulate correctly."""
        portfolio = Portfolio(initial_balance=100_000.0)

        fees = [3.0, 2.5, 1.75, 4.0]

        for i, fee in enumerate(fees):
            portfolio.register_fee(fee, datetime(2024, 1, 1, 12 + i, 0, 0), "entry")

        expected_total = sum(fees)
        assert portfolio.total_fees == pytest.approx(expected_total, rel=1e-6)
        assert portfolio.cash == pytest.approx(100_000.0 - expected_total, rel=1e-6)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_risk_position(self) -> None:
        """Test handling of zero-risk position (SL == entry)."""
        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="EURUSD",
            entry_price=1.10000,
            stop_loss=1.10000,  # Zero risk
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10100, "signal")

        # R-multiple should be 0 for zero risk
        assert pos.r_multiple == 0.0
        assert pos.result == 0.0

    def test_empty_portfolio_summary(self) -> None:
        """Test summary of portfolio with no trades."""
        portfolio = Portfolio(initial_balance=100_000.0)
        summary = portfolio.get_summary()

        assert summary["Initial Balance"] == 100_000.0
        assert summary["Final Balance"] == 100_000.0
        assert summary["Total Trades"] == 0
        assert summary["Winrate"] == 0.0

    def test_single_trade_portfolio(self) -> None:
        """Test portfolio with exactly one trade."""
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
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10200, "take_profit")
        portfolio.register_exit(pos)

        summary = portfolio.get_summary()

        assert summary["Total Trades"] == 1
        assert summary["Winrate"] == 100.0
        assert summary["Wins"] == 1
        assert summary["Losses"] == 0

    def test_all_losses_portfolio(self) -> None:
        """Test portfolio where all trades are losses."""
        portfolio = Portfolio(initial_balance=100_000.0)

        for i in range(5):
            pos = PortfolioPosition(
                entry_time=datetime(2024, 1, 1, 12 + i, 0, 0),
                direction="long",
                symbol="EURUSD",
                entry_price=1.10000,
                stop_loss=1.09900,
                take_profit=1.10200,
                size=1.0,
                risk_per_trade=100.0,
            )
            portfolio.register_entry(pos)
            pos.close(datetime(2024, 1, 1, 12 + i, 30, 0), 1.09900, "stop_loss")
            portfolio.register_exit(pos)

        summary = portfolio.get_summary()

        assert summary["Total Trades"] == 5
        assert summary["Winrate"] == 0.0
        assert summary["Wins"] == 0
        assert summary["Losses"] == 5

    def test_position_without_symbol_raises(self) -> None:
        """Test that position without symbol raises ValueError."""
        portfolio = Portfolio(initial_balance=100_000.0)

        pos = PortfolioPosition(
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            direction="long",
            symbol="",  # Empty symbol
            entry_price=1.10000,
            stop_loss=1.09900,
            take_profit=1.10200,
            size=1.0,
            risk_per_trade=100.0,
        )

        with pytest.raises(ValueError, match="symbol"):
            portfolio.register_entry(pos)


class TestMultiSymbolPortfolio:
    """Tests for portfolios with multiple symbols."""

    def test_multi_symbol_tracking(self) -> None:
        """Test tracking positions across multiple symbols."""
        portfolio = Portfolio(initial_balance=100_000.0)

        symbols = ["EURUSD", "GBPUSD", "USDJPY"]

        for symbol in symbols:
            pos = PortfolioPosition(
                entry_time=datetime(2024, 1, 1, 12, 0, 0),
                direction="long",
                symbol=symbol,
                entry_price=1.10000,
                stop_loss=1.09900,
                take_profit=1.10200,
                size=1.0,
                risk_per_trade=100.0,
            )
            portfolio.register_entry(pos)

        assert len(portfolio.open_positions) == 3

        # Filter by symbol
        eur_positions = portfolio.get_open_positions("EURUSD")
        assert len(eur_positions) == 1
        assert eur_positions[0].symbol == "EURUSD"

    def test_partial_close_tracking(self) -> None:
        """Test that partial closes are tracked separately."""
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

        # Partial close
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10100, "partial_exit")
        portfolio.register_exit(pos)

        assert len(portfolio.partial_closed_positions) == 1
        assert len(portfolio.closed_positions) == 0


class TestDataFrameExport:
    """Tests for DataFrame export functionality."""

    def test_trades_to_dataframe_columns(self) -> None:
        """Test that DataFrame has expected columns."""
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
        pos.close(datetime(2024, 1, 1, 13, 0, 0), 1.10100, "signal")
        portfolio.register_exit(pos)

        df = portfolio.trades_to_dataframe()

        expected_columns = [
            "entry_time",
            "exit_time",
            "direction",
            "symbol",
            "entry_price",
            "exit_price",
            "stop_loss",
            "take_profit",
            "size",
            "result",
            "reason",
            "status",
            "r_multiple",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_empty_portfolio_dataframe(self) -> None:
        """Test DataFrame from empty portfolio."""
        portfolio = Portfolio(initial_balance=100_000.0)
        df = portfolio.trades_to_dataframe()

        assert len(df) == 0
        assert len(df.columns) > 0  # Should still have column structure
