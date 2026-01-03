# hf_engine/strategies/mean_reversion_z_score/backtest/position_manager.py
"""Backtest position manager for mean_reversion_z_score strategy."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from strategies._base.domain_types import CandleProtocol, PositionProtocol

if TYPE_CHECKING:
    from backtest_engine.simulation.portfolio import Portfolio


class BacktestPositionManager:
    """Manages positions during backtesting with timeout functionality."""

    def __init__(
        self,
        max_holding_minutes: int = 30,
    ) -> None:
        """
        Initialize the backtest position manager.

        Args:
            max_holding_minutes: Maximum time to hold a position before forced close.
        """
        self.portfolio: Portfolio | None = None
        # Defensive: Optuna/GridSearch/NumPy können numpy.int64 liefern, was in
        # datetime.timedelta(minutes=...) auf manchen Python-Versionen nicht akzeptiert wird.
        self.max_holding_minutes: int | None = (
            None if max_holding_minutes is None else int(max_holding_minutes)
        )

    def attach_portfolio(self, portfolio: Portfolio) -> None:
        """
        Attach a portfolio instance to the manager.

        Args:
            portfolio: Portfolio to attach.
        """
        self.portfolio = portfolio

    def manage_positions(
        self,
        open_positions: list[PositionProtocol],
        symbol_slice: Any,
        bid_candle: CandleProtocol,
        ask_candle: CandleProtocol,
        all_positions: list[PositionProtocol] | None = None,
    ) -> None:
        """
        Manage open positions, applying timeout rules.

        Args:
            open_positions: List of currently open positions.
            symbol_slice: Symbol data slice (unused in current implementation).
            bid_candle: Current bid candle data.
            ask_candle: Current ask candle data.
            all_positions: All positions including closed (optional).
        """
        for pos in open_positions:
            # 1) Timeout für geöffnete Positionen
            if self.max_holding_minutes is not None and pos.status == "open":
                if bid_candle.timestamp - pos.entry_time >= timedelta(
                    minutes=int(self.max_holding_minutes)
                ):
                    exit_price: float = (
                        bid_candle.close
                        if pos.direction == "long"
                        else ask_candle.close
                    )
                    pos.close(bid_candle.timestamp, exit_price, reason="timeout")
                    if self.portfolio:
                        self.portfolio.register_exit(pos)
                    continue

            # # 2) Break-Even & Teilverkauf
            # if pos.status == "open":
            #     risk = abs(pos.entry_price - pos.initial_stop_loss)

            #     if bid_candle.timestamp == pos.trigger_time and pos.order_type == "limit":
            #         if pos.direction == "long":
            #             current_close = bid_candle.close
            #             profit = current_close - pos.entry_price
            #         else:
            #             current_close = ask_candle.close
            #             profit = pos.entry_price - current_close

            #     else:
            #         if pos.direction == "long":
            #             current_high = bid_candle.high
            #             profit = current_high - pos.entry_price
            #         else:
            #             current_low = ask_candle.low
            #             profit = pos.entry_price - current_low

            #     # Break-Even
            #     executed_flag_break_even = getattr(pos, '_breakeven_done', False)
            #     if not executed_flag_break_even:
            #         if risk > 0 and profit / risk >= self.breakeven_r_multiple:
            #             be_price = pos.entry_price + (self.breakeven_buffer if pos.direction == "long" else -self.breakeven_buffer)
            #             pos.stop_loss = be_price
            #             setattr(pos, '_breakeven_done', True)
            #             if pos.direction == "long" and pos.stop_loss >= bid_candle.close:
            #                 exit_time = bid_candle.timestamp
            #                 pos.exit_price = pos.stop_loss
            #                 pos.close(exit_time, exit_price, reason = "break_even_stop_loss")
            #                 self.portfolio.register_exit(pos)

            #             elif pos.direction == "short" and pos.stop_loss <= ask_candle.close:
            #                 exit_time = bid_candle.timestamp
            #                 pos.exit_price = pos.stop_loss
            #                 pos.close(exit_time, exit_price, reason = "break_even_stop_loss")
            #                 self.portfolio.register_exit(pos)

            #     # Dynamischer Take Profit
            #     if self.dynamic_tp_enabled:
            #         if pos.direction == "long" and c.close > pos.entry_price:
            #             tp_candidates = [level for level in [ema9[-1], ema20[-1], ema50[-1]] if level > c.close]
            #             new_tp_price = min(tp_candidates)
            #             if new_tp_price > pos.take_profit:
            #                 return
            #             else:
            #                 pos.take_profit = new_tp_price
            #         elif pos.direction == "short" and c.close < pos.entry_price:
            #             tp_candidates = [level for level in [ema9[-1], ema20[-1], ema50[-1]] if level < c.close]
            #             new_tp_price = max(tp_candidates)
            #             if new_tp_price < pos.take_profit:
            #                 return
            #             else:
            #                 pos.take_profit = new_tp_price
