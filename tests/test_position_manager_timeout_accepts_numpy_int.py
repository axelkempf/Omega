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


def test_other_backtest_position_managers_timeout_accept_numpy_int64() -> None:
    mh = np.int64(120)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    bid = _DummyCandle(timestamp=now, close=1.1, high=1.2, low=1.0)
    ask = _DummyCandle(timestamp=now, close=1.1, high=1.2, low=1.0)

    # mean_reversion_bollinger_bands_plus_macd
    from strategies.mean_reversion_bollinger_bands_plus_macd.backtest.position_manager import (
        BacktestPositionManager as BBMacdPM,
    )

    pos1 = _DummyPosition(
        status="open",
        direction="long",
        entry_time=now - timedelta(minutes=int(mh) + 1),
        trigger_time=now - timedelta(minutes=int(mh) + 1),
    )
    pm1 = BBMacdPM(max_holding_minutes=mh)
    pm1.attach_portfolio(_DummyPortfolio())
    pm1.manage_positions([pos1], symbol_slice=None, bid_candle=bid, ask_candle=ask)
    assert pos1.status == "closed"

    # trading_the_flow
    from strategies.trading_the_flow.backtest.position_manager import (
        BacktestPositionManager as TTFPM,
    )

    pos2 = _DummyPosition(
        status="open",
        direction="long",
        entry_time=now - timedelta(minutes=int(mh) + 1),
        trigger_time=now - timedelta(minutes=int(mh) + 1),
    )
    pm2 = TTFPM(max_holding_minutes=mh)
    pm2.attach_portfolio(_DummyPortfolio())
    pm2.manage_positions([pos2], symbol_slice=None, bid_candle=bid, ask_candle=ask)
    assert pos2.status == "closed"

    # statistical_arbitrage
    from strategies.statistical_arbitrage.backtest.position_manager import (
        BacktestPositionManager as StatArbPM,
    )

    pos3 = _DummyPosition(
        status="open",
        direction="long",
        entry_time=now - timedelta(minutes=int(mh) + 1),
        trigger_time=now - timedelta(minutes=int(mh) + 1),
    )
    pm3 = StatArbPM(max_holding_minutes=mh)
    pm3.attach_portfolio(_DummyPortfolio())
    pm3.manage_positions([pos3], symbol_slice=None, bid_candle=bid, ask_candle=ask)
    assert pos3.status == "closed"


def test_pending_timeout_accepts_numpy_int64() -> None:
    from strategies.ema_rsi_trend_follow.backtest.position_manager import (
        BacktestPositionManager as EmaRsiPM,
    )

    mp = np.int64(30)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bid = _DummyCandle(timestamp=now, close=1.1, high=1.2, low=1.0)
    ask = _DummyCandle(timestamp=now, close=1.1, high=1.2, low=1.0)

    # ensure open-position path doesn't error due to missing portfolio
    open_pos = _DummyPosition(
        status="open",
        direction="long",
        entry_time=now - timedelta(minutes=int(mp) + 1),
        trigger_time=now - timedelta(minutes=int(mp) + 1),
        stop_loss=1.0,
    )
    pending_pos = _DummyPosition(
        status="pending",
        direction="long",
        entry_time=now - timedelta(minutes=int(mp) + 1),
        trigger_time=now - timedelta(minutes=int(mp) + 1),
    )

    pm = EmaRsiPM(max_holding_minutes=mp, max_pending_minutes=mp)
    pm.attach_portfolio(_DummyPortfolio())

    pm.manage_positions(
        [open_pos],
        symbol_slice=None,
        bid_candle=bid,
        ask_candle=ask,
        all_positions=[pending_pos],
    )

    assert pending_pos.is_closed is True
    assert pending_pos.reason == "limit_expired"
