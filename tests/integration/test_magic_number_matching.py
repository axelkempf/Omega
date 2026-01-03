"""Integration tests for magic number based position filtering."""

from __future__ import annotations

import pytest

from tests.mocks.mock_broker import MockPosition
from tests.utils.trading_test_utils import (
    assert_position_matched,
    assert_positions_filtered_correctly,
    create_mock_position,
)

pytestmark = [pytest.mark.integration, pytest.mark.trading_safety]


class TestMagicNumberFiltering:
    """Tests for magic number based position filtering in the broker mock."""

    def test_get_own_position_filters_by_magic_number(
        self, mock_broker, sample_positions
    ) -> None:
        for pos in sample_positions:
            mock_broker.add_position(pos)

        result = mock_broker.get_own_position("EURUSD", magic_number=12345)

        assert_position_matched(result, 12345)
        assert result.symbol.upper() == "EURUSD"

    def test_get_all_own_positions_returns_only_matching_magic(
        self, mock_broker, sample_positions
    ) -> None:
        for pos in sample_positions:
            mock_broker.add_position(pos)

        positions = mock_broker.get_all_own_positions(magic_number=12345)

        assert len(positions) == 2
        assert_positions_filtered_correctly(positions, 12345)

    def test_magic_number_none_returns_all_positions(
        self, mock_broker, sample_positions
    ) -> None:
        for pos in sample_positions:
            mock_broker.add_position(pos)

        positions = mock_broker.get_all_own_positions(magic_number=None)

        assert len(positions) == len(sample_positions)

    def test_magic_number_zero_behavior(self, mock_broker) -> None:
        mock_broker.add_position(
            create_mock_position(ticket=10, magic=0, symbol="USDJPY")
        )
        mock_broker.add_position(
            create_mock_position(ticket=11, magic=12345, symbol="USDJPY")
        )

        positions = mock_broker.get_all_own_positions(magic_number=0)

        assert len(positions) == 1
        assert_position_matched(positions[0], 0)

    def test_position_magic_type_coercion(self, mock_broker) -> None:
        pos_str = create_mock_position(ticket=20, magic=12345, symbol="EURUSD")
        pos_str.magic = "12345"  # simulate string magic from external source
        mock_broker.add_position(pos_str)
        mock_broker.add_position(
            create_mock_position(ticket=21, magic=99999, symbol="EURUSD")
        )

        positions = mock_broker.get_all_own_positions(magic_number=12345)

        assert len(positions) == 1
        assert_position_matched(positions[0], 12345)

    def test_empty_positions_returns_empty_list(self, mock_broker) -> None:
        positions = mock_broker.get_all_own_positions(magic_number=12345)

        assert positions == []

    def test_symbol_and_magic_combined_filter(self, mock_broker) -> None:
        mock_broker.add_position(
            create_mock_position(ticket=30, magic=12345, symbol="EURUSD")
        )
        mock_broker.add_position(
            create_mock_position(ticket=31, magic=12345, symbol="GBPUSD")
        )

        eur_pos = mock_broker.get_own_position("EURUSD", magic_number=12345)
        gbp_pos = mock_broker.get_own_position("GBPUSD", magic_number=12345)

        assert eur_pos.symbol.upper() == "EURUSD"
        assert gbp_pos.symbol.upper() == "GBPUSD"
        assert_position_matched(eur_pos, 12345)
        assert_position_matched(gbp_pos, 12345)
