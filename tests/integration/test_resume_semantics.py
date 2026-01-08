"""Integration tests for position resume semantics in StrategyRunner."""

from __future__ import annotations

import sys
import threading
import types
from typing import Any, Dict

import pytest

from tests.mocks.mock_broker import MockPosition
from tests.utils.trading_test_utils import create_mock_position

pytestmark = [pytest.mark.integration, pytest.mark.trading_safety]


class _FakeManager:
    """Lightweight position manager stub capturing setup and dependencies."""

    def __init__(self, setup, broker, data_provider):
        self.setup = setup
        self.broker = broker
        self.data_provider = data_provider
        self.id = getattr(setup, "metadata", {}).get("ticket_id") or getattr(
            setup, "magic_number", 0
        )


class _FakeExecutionTracker:
    def get_day_data(self, date=None) -> Dict[str, Any]:  # pragma: no cover - stub
        return {}


class _FakeSessionRunner:
    def __init__(self, strategy):
        self.strategy = strategy

    def start(self):  # pragma: no cover - stub
        return None

    def stop(self):  # pragma: no cover - stub
        return None

    def run_daily_cycle(self):  # pragma: no cover - stub
        return None


@pytest.fixture
def patched_runner(monkeypatch, mock_strategy, mock_broker, mock_data_provider):
    """Factory to create a StrategyRunner with patched dependencies."""

    def _create(pre_existing=None):
        pre_existing = pre_existing or {}

        # Patch MetaTrader5 import before importing StrategyRunner
        # Important: Hypothesis (and other tooling) assumes sys.modules values are hashable.
        # Using ModuleType avoids leaving an unhashable SimpleNamespace in sys.modules.
        mt5_stub = types.ModuleType("MetaTrader5")
        mt5_stub.TIMEFRAME_M1 = 1
        mt5_stub.TIMEFRAME_M5 = 5
        mt5_stub.TIMEFRAME_M15 = 15
        mt5_stub.TIMEFRAME_M30 = 30
        mt5_stub.TIMEFRAME_H1 = 60
        mt5_stub.TIMEFRAME_H4 = 240
        mt5_stub.TIMEFRAME_D1 = 1440
        mt5_stub.TIMEFRAME_W1 = 10080
        mt5_stub.TIMEFRAME_MN1 = 43200

        monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)

        from hf_engine.core.controlling import strategy_runner as sr
        from strategies._base import base_position_manager as bpm

        class _FakePMC:
            def __init__(self):
                self._managers = dict(pre_existing)
                self._lock = threading.Lock()

            def start(self):  # pragma: no cover - no-op
                return None

            def stop_all(self):  # pragma: no cover - no-op
                return None

            def add_manager(self, manager):
                self._managers[getattr(manager, "id", len(self._managers) + 1)] = (
                    manager
                )

        def _factory(setup, broker, data_provider):
            return _FakeManager(setup, broker, data_provider)

        monkeypatch.setattr(sr, "PositionMonitorController", _FakePMC)
        monkeypatch.setattr(sr, "strategy_position_manager_factory", _factory)
        monkeypatch.setattr(sr, "ExecutionTracker", _FakeExecutionTracker)
        monkeypatch.setattr(sr, "SessionRunner", _FakeSessionRunner)
        monkeypatch.setattr(bpm, "strategy_position_manager_factory", _factory)

        return sr.StrategyRunner(
            mock_strategy,
            mock_broker,
            mock_data_provider,
            magic_number=12345,
            controller=None,
            symbol_mapper=None,
        )

    return _create


class TestResumeOpenPositions:
    def test_resume_registers_manager_for_open_positions(
        self, patched_runner, mock_broker
    ) -> None:
        mock_broker.add_position(create_mock_position(ticket=1, magic=12345))

        runner = patched_runner()

        managers = runner.position_monitor_controller._managers
        assert len(managers) == 1
        manager = next(iter(managers.values()))
        assert manager.setup.metadata["ticket_id"] == 1
        assert manager.setup.symbol.upper() == "EURUSD"

    def test_resume_ignores_foreign_magic_positions(
        self, patched_runner, mock_broker
    ) -> None:
        mock_broker.add_position(create_mock_position(ticket=2, magic=99999))

        runner = patched_runner()

        assert runner.position_monitor_controller._managers == {}

    def test_resume_skips_already_registered_positions(
        self, patched_runner, mock_broker
    ) -> None:
        mock_broker.add_position(create_mock_position(ticket=3, magic=12345))

        runner = patched_runner(pre_existing={3: "existing"})

        managers = runner.position_monitor_controller._managers
        # existing remains, no new manager added
        assert list(managers.keys()) == [3]

    def test_resume_handles_missing_broker_connection(
        self, patched_runner, mock_broker
    ) -> None:
        mock_broker.set_connection_status(False)

        runner = patched_runner()

        assert runner.position_monitor_controller._managers == {}

    def test_resume_reconstructs_trade_setup_correctly(
        self, patched_runner, mock_broker
    ) -> None:
        pos = create_mock_position(
            ticket=4,
            magic=12345,
            symbol="GBPUSD",
            direction="sell",
            sl=1.20,
            tp=1.10,
            price_open=1.1500,
        )
        mock_broker.add_position(pos)

        runner = patched_runner()

        manager = next(iter(runner.position_monitor_controller._managers.values()))
        setup = manager.setup
        assert setup.symbol.upper() == "GBPUSD"
        assert setup.direction == "sell"
        assert setup.sl == 1.20
        assert setup.tp == 1.10
        assert setup.entry == 1.1500
        assert setup.metadata.get("ticket_id") == 4


class TestResumeEdgeCases:
    def test_no_positions_leads_to_no_managers(self, patched_runner) -> None:
        runner = patched_runner()
        assert runner.position_monitor_controller._managers == {}

    def test_symbol_filter_excludes_non_configured(
        self, patched_runner, mock_broker, mock_strategy
    ) -> None:
        # strategy symbols are EURUSD/GBPUSD; add USDJPY to ensure skip
        mock_broker.add_position(
            create_mock_position(ticket=5, magic=12345, symbol="USDJPY")
        )

        runner = patched_runner()

        assert runner.position_monitor_controller._managers == {}

    def test_direction_fallback_uses_position_direction(
        self, patched_runner, mock_broker
    ) -> None:
        # direction invalid -> fallback via broker.position_direction
        pos = MockPosition(
            ticket=6,
            symbol="EURUSD",
            magic=12345,
            direction="invalid",
            volume=0.01,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
        )
        mock_broker.add_position(pos)

        runner = patched_runner()

        setup = next(iter(runner.position_monitor_controller._managers.values())).setup
        assert setup.direction in ("buy", "sell")

    def test_ticket_non_numeric_is_skipped(self, patched_runner, mock_broker) -> None:
        pos = create_mock_position(ticket=7, magic=12345)
        mock_broker.add_position(pos)
        # mutate ticket after storing to simulate corrupted ticket string
        pos.ticket = "abc"

        runner = patched_runner()

        assert runner.position_monitor_controller._managers == {}

    def test_broker_exception_during_fetch_is_handled(
        self, monkeypatch, patched_runner, mock_broker
    ) -> None:
        def _fail(*args, **kwargs):
            raise RuntimeError("fail")

        monkeypatch.setattr(mock_broker, "get_all_own_positions", _fail)

        runner = patched_runner()

        assert runner.position_monitor_controller._managers == {}
