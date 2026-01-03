from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

# Ensure repository root is on sys.path so imports like `analysis.*` work reliably
# across different pytest/runner configurations.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from strategies._base.base_strategy import Strategy, TradeSetup
from tests.mocks.mock_broker import MockBrokerInterface, MockPosition
from tests.utils.trading_test_utils import create_mock_position


@pytest.fixture
def mock_broker() -> MockBrokerInterface:
    """Provide a configured MockBrokerInterface with cleanup."""

    broker = MockBrokerInterface()
    yield broker
    broker.clear_positions()


@pytest.fixture
def sample_positions() -> list[MockPosition]:
    """Provide mock positions with different magic numbers and symbols."""

    return [
        create_mock_position(ticket=1, magic=12345, symbol="EURUSD"),
        create_mock_position(ticket=2, magic=12345, symbol="GBPUSD"),
        create_mock_position(ticket=3, magic=99999, symbol="EURUSD"),
        create_mock_position(ticket=4, magic=0, symbol="USDJPY"),
    ]


class _DummyStrategy(Strategy):
    def __init__(self, config: dict | None = None) -> None:
        base_config = {
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframe": "M15",
            "runner_max_workers": 1,
            "risk": {"start_capital": 10_000.0, "risk_per_trade_pct": 0.01},
            "session": {},
        }
        merged = {**base_config, **(config or {})}
        super().__init__(merged)
        self.timeframe = self.config.get("timeframe", "M15")

    def name(self) -> str:  # pragma: no cover - trivial
        return "TestStrategy"

    def generate_signal(  # pragma: no cover - not used in these tests
        self,
        symbol: str,
        date: datetime,
        broker=None,
        data_provider=None,
    ) -> list[TradeSetup]:
        return []


@pytest.fixture
def mock_strategy() -> Strategy:
    """Provide a minimal Strategy implementation for tests."""

    return _DummyStrategy()


class _DummyDataProvider:
    def get_ohlc_for_closed_candle(self, symbol: str, timeframe: str, offset: int = 1):
        now = datetime.now(timezone.utc)
        return {"time": now.isoformat().replace("+00:00", "Z"), "symbol": symbol}


@pytest.fixture
def mock_data_provider() -> _DummyDataProvider:
    """Provide a simple MT5DataProvider stub."""

    return _DummyDataProvider()


@pytest.fixture
def temp_heartbeat_dir(tmp_path: Path) -> Path:
    """Provide a temporary var/tmp/ directory for heartbeat tests."""

    heartbeat_dir = tmp_path / "var" / "tmp"
    heartbeat_dir.mkdir(parents=True)
    return heartbeat_dir


@pytest.fixture
def trade_setup_factory() -> Callable[..., TradeSetup]:
    """Factory fixture to create TradeSetup instances for tests."""

    def _create(
        symbol: str = "EURUSD",
        direction: str = "buy",
        magic_number: int = 12345,
        **kwargs,
    ) -> TradeSetup:
        defaults = {
            "entry": 1.1000,
            "sl": 1.0950,
            "tp": 1.1100,
            "strategy": "TestStrategy",
            "strategy_module": "test.live",
            "start_capital": 10_000.0,
            "risk_pct": 0.01,
            "metadata": {},
        }
        defaults.update(kwargs)
        return TradeSetup(
            symbol=symbol,
            direction=direction,
            magic_number=magic_number,
            **defaults,
        )

    return _create
