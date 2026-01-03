"""Integration tests for the strategy_position_manager_factory."""

from __future__ import annotations

import sys
import types

import pytest

from strategies._base.base_position_manager import (
    BasePositionManager,
    strategy_position_manager_factory,
)
from strategies._base.base_strategy import TradeSetup
from tests.mocks.mock_broker import MockBrokerInterface

pytestmark = [pytest.mark.integration]


class _StubPositionManager(BasePositionManager):
    def __init__(self, setup, broker, data_provider):
        super().__init__(setup, broker, data_provider)
        self.received = (setup, broker, data_provider)

    def monitor_step(self) -> bool:  # pragma: no cover - simple stub
        return True


def _register_fake_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str, cls=None
) -> None:
    package_name = f"strategies.{module_name}"
    module_path = f"{package_name}.position_manager"

    fake_package = types.ModuleType(package_name)
    fake_package.__path__ = []  # mark as package
    fake_module = types.ModuleType(module_path)
    if cls:
        fake_module.StrategyPositionManager = cls

    monkeypatch.setitem(sys.modules, package_name, fake_package)
    monkeypatch.setitem(sys.modules, module_path, fake_module)


def _make_setup(strategy_module: str = "fake_pm") -> TradeSetup:
    return TradeSetup(
        symbol="EURUSD",
        direction="buy",
        entry=1.1,
        sl=1.09,
        tp=1.12,
        magic_number=12345,
        strategy="TestStrategy",
        strategy_module=strategy_module,
        metadata={},
        start_capital=10_000.0,
        risk_pct=0.01,
    )


def test_factory_loads_correct_manager_class(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBrokerInterface()
    data_provider = object()
    _register_fake_module(monkeypatch, "fake_pm", _StubPositionManager)
    setup = _make_setup("fake_pm")

    manager = strategy_position_manager_factory(setup, broker, data_provider)

    assert isinstance(manager, _StubPositionManager)
    assert manager.received[0] is setup
    assert manager.received[1] is broker
    assert manager.received[2] is data_provider


def test_factory_raises_on_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBrokerInterface()
    data_provider = object()
    setup = _make_setup("missing_pm")

    with pytest.raises(ImportError):
        strategy_position_manager_factory(setup, broker, data_provider)


def test_factory_raises_on_missing_class(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBrokerInterface()
    data_provider = object()
    setup = _make_setup("no_class_pm")
    _register_fake_module(monkeypatch, "no_class_pm", cls=None)

    with pytest.raises(ImportError):
        strategy_position_manager_factory(setup, broker, data_provider)


def test_factory_passes_setup_broker_dataprovider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = MockBrokerInterface()
    data_provider = object()

    class _AssertingPM(_StubPositionManager):
        def __init__(self, setup, broker_obj, data_provider_obj):
            super().__init__(setup, broker_obj, data_provider_obj)
            self.setup_symbol = setup.symbol

    _register_fake_module(monkeypatch, "assert_pm", _AssertingPM)
    setup = _make_setup("assert_pm")

    manager = strategy_position_manager_factory(setup, broker, data_provider)

    assert isinstance(manager, _AssertingPM)
    assert manager.setup_symbol == "EURUSD"
    assert manager.broker is broker
    assert manager.data_provider is data_provider
