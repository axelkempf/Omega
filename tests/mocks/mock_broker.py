"""Mock broker implementation for tests without MT5 dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hf_engine.adapter.broker.broker_interface import (
    BrokerInterface,
    Direction,
    OrderResult,
    OrderType,
    Position,
)


@dataclass
class MockPosition:
    """Mock position for testing broker interactions."""

    ticket: int
    symbol: str
    magic: int
    direction: str  # "buy" or "sell"
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float = 0.0
    comment: str = ""
    open_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def position_id(self) -> int:
        return self.ticket

    @property
    def entry(self) -> float:
        return self.price_open

    def as_position(self) -> Position:
        """Convert to the shared Position dataclass for compatibility."""
        return Position(
            position_id=self.ticket,
            symbol=self.symbol,
            entry=self.price_open,
            direction=Direction(self.direction.lower()),
            sl=self.sl,
            tp=self.tp,
            volume=self.volume,
            comment=self.comment,
            open_time=self.open_time,
        )


class MockBrokerInterface(BrokerInterface):
    """In-memory broker stub fulfilling the BrokerInterface contract."""

    def __init__(self) -> None:
        self._connected: bool = True
        self._magic_number: Optional[int] = None
        self._equity: float = 100_000.0
        self._currency: str = "USD"
        self._tick: Dict[str, float] = {"bid": 1.0, "ask": 1.0002}
        self._positions: Dict[int, MockPosition] = {}

    # --- Helpers -----------------------------------------------------
    def add_position(self, position: MockPosition) -> None:
        self._positions[int(position.ticket)] = position

    def remove_position(self, ticket: int) -> None:
        self._positions.pop(int(ticket), None)

    def clear_positions(self) -> None:
        self._positions.clear()

    def set_connection_status(self, connected: bool) -> None:
        self._connected = connected

    def _matches_magic(
        self, position_magic: Optional[int], magic_number: Optional[int]
    ) -> bool:
        if magic_number is None:
            return True
        try:
            return int(position_magic) == int(magic_number)
        except Exception:
            return False

    def _filtered_positions(
        self, magic_number: Optional[int] = None
    ) -> List[MockPosition]:
        positions = list(self._positions.values())
        return [p for p in positions if self._matches_magic(p.magic, magic_number)]

    def _coerce_direction(self, direction: Any) -> Direction:
        try:
            if isinstance(direction, Direction):
                return direction
            return Direction(str(direction).lower())
        except Exception:
            return Direction.BUY

    # --- Lifecycle / Connection -------------------------------------
    def _connect_mt5(self) -> bool:
        self._connected = True
        return self._connected

    def ensure_connection(self) -> bool:
        return self._connected

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self._connected = False

    def set_magic_number(self, new_magic: int) -> None:
        self._magic_number = int(new_magic)

    def _validate_account(self) -> None:
        return None

    # --- Account / Symbol Info --------------------------------------
    def get_account_equity(self) -> float:
        return self._equity

    def get_account_currency(self) -> str:
        return self._currency

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        return {"digits": 5, "lot_step": 0.01, "symbol": symbol}

    def get_symbol_tick(self, symbol: str) -> Dict[str, float]:
        return self._tick.copy()

    def get_symbol_price(self, symbol: str, direction: Direction) -> Optional[float]:
        tick = self.get_symbol_tick(symbol)
        if direction == Direction.BUY:
            return tick.get("ask")
        return tick.get("bid")

    def get_symbol_spread(self, symbol: str) -> float:
        tick = self.get_symbol_tick(symbol)
        return float(tick.get("ask", 0.0) - tick.get("bid", 0.0))

    def get_min_lot_size(self, symbol: str) -> float:
        return 0.01

    # --- Risk / Analytics --------------------------------------------
    def calculate_risk_amount(
        self, symbol: str, entry_price: float, sl_price: float, volume: float
    ) -> float:
        return abs(entry_price - sl_price) * volume

    def get_current_r_multiple(
        self, symbol: str, initial_sl: float, ticket_id: Optional[int] = None
    ) -> float:
        return 0.0

    def reconstruct_trade_from_deal_ticket(
        self, ticket_id: int, pip_size: float = 0.0001
    ) -> Optional[Dict[str, Any]]:
        return None

    def get_realized_r_multiple(self, magic_number: int) -> float:
        return 0.0

    def get_floating_r_multiple(self, strategy: str) -> float:
        return 0.0

    def get_total_r_multiple(self, magic_number: int, strategy: str) -> float:
        return 0.0

    # --- Orders / Positions -----------------------------------------
    def place_pending_order(
        self,
        symbol: str,
        direction: Direction,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        comment: str = "",
        order_type: OrderType = OrderType.STOP,
        magic_number: Optional[int] = None,
        scenario: Optional[str] = "scenario",
    ) -> OrderResult:
        return OrderResult(success=True, message="pending", magic_number=magic_number)

    def place_market_order(
        self,
        symbol: str,
        direction: Direction,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        volume: float = 1.0,
        comment: str = "Market Order",
        magic_number: Optional[int] = None,
        scenario: Optional[str] = "scenario",
    ) -> OrderResult:
        return OrderResult(success=True, message="market", magic_number=magic_number)

    def cancel_all_pending_orders(self, symbol: str) -> None:
        return None

    def cancel_pending_order_by_ticket(
        self, ticket_id: int, magic_number: Optional[int] = None
    ) -> bool:
        return False

    def get_pending_order_by_ticket(
        self, ticket_id: int, magic_number: Optional[int] = None
    ) -> Optional[object]:
        return None

    def get_own_pending_orders(
        self, symbol: str, magic_number: Optional[int] = None
    ) -> Optional[object]:
        return []

    def get_own_position(
        self, symbol: str, magic_number: Optional[int] = None
    ) -> Optional[MockPosition]:
        symbol_norm = symbol.strip().upper()
        for pos in self._filtered_positions(magic_number):
            if pos.symbol.strip().upper() == symbol_norm:
                return pos
        return None

    def get_all_own_positions(
        self, magic_number: Optional[int] = None
    ) -> List[MockPosition]:
        return list(self._filtered_positions(magic_number))

    def get_position_by_ticket(self, ticket: int) -> Optional[MockPosition]:
        return self._positions.get(int(ticket))

    def position_direction(self, pos: MockPosition) -> Direction | str:
        coerced = self._coerce_direction(getattr(pos, "direction", Direction.BUY))
        return coerced.value

    def pending_order_direction(self, order: object) -> Direction | str:
        direction = getattr(order, "direction", Direction.BUY)
        coerced = self._coerce_direction(direction)
        return coerced.value

    def modify_sl(self, position_id: int, new_sl: float) -> OrderResult:
        pos = self._positions.get(int(position_id))
        if pos:
            pos.sl = new_sl
        return OrderResult(success=True, magic_number=self._magic_number)

    def modify_tp(self, position_id: int, new_tp: float) -> OrderResult:
        pos = self._positions.get(int(position_id))
        if pos:
            pos.tp = new_tp
        return OrderResult(success=True, magic_number=self._magic_number)

    def close_position_partial(self, position_id: int, volume: float) -> OrderResult:
        pos = self._positions.get(int(position_id))
        if pos:
            pos.volume = max(0.0, pos.volume - volume)
        return OrderResult(success=True, magic_number=self._magic_number)

    def close_position_full(self, position_id: int) -> OrderResult:
        self.remove_position(position_id)
        return OrderResult(success=True, magic_number=self._magic_number)
