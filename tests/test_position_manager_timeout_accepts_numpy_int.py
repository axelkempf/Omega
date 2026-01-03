from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np


@dataclass
class _DummyCandle:
    timestamp: datetime
    close: float = 1.0
    high: float = 1.0
    low: float = 1.0


@dataclass
class _DummyPosition:
    status: str
    direction: str
    entry_time: datetime
    trigger_time: datetime
    entry_price: float = 1.0
    stop_loss: float = 0.9
    initial_stop_loss: float = 0.9
    size: float = 1.0
    risk_per_trade: float = 1.0
    order_type: str = "market"

    # pending-order fields used by some managers
    is_closed: bool = False
    reason: str = ""
    exit_time: datetime | None = None
    exit_price: float | None = None

    def close(self, exit_time: datetime, exit_price: float, reason: str = "") -> None:
        self.status = "closed"
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.reason = reason


class _DummyPortfolio:
    def register_exit(self, _pos: object) -> None:
        return


def test_z_score_backtest_position_manager_timeout_accepts_numpy_int64() -> None:
    from strategies.mean_reversion_z_score.backtest.position_manager import (
        BacktestPositionManager,
    )

    mh = np.int64(840)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    pos = _DummyPosition(
        status="open",
        direction="long",
        entry_time=now - timedelta(minutes=int(mh) + 1),
        trigger_time=now - timedelta(minutes=int(mh) + 1),
    )
    bid = _DummyCandle(timestamp=now, close=1.1)
    ask = _DummyCandle(timestamp=now, close=1.1)

    pm = BacktestPositionManager(max_holding_minutes=mh)
    pm.attach_portfolio(_DummyPortfolio())

    pm.manage_positions([pos], symbol_slice=None, bid_candle=bid, ask_candle=ask)

    assert pos.status == "closed"
    assert pos.reason == "timeout"