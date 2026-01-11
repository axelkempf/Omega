"""Execution Simulator: Thin Wrapper für Rust ExecutionSimulatorRust.

Wave 4 Phase 6: Dieser Modul ist ein Thin Wrapper, der ALLE Logik an die
Rust-Implementierung delegiert. Keine Python-Logik-Duplikation.

Feature Flag: OMEGA_USE_RUST_EXECUTION_SIMULATOR=always (required)

Zentrale Komponenten:
- ExecutionSimulator: Python Wrapper für ExecutionSimulatorRust
- Arrow IPC für Zero-Copy Datenübertragung
- Portfolio-Integration via PortfolioPosition Materialisierung
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.ipc as ipc
from omega_rust import ExecutionSimulatorRust

from backtest_engine.core.portfolio import Portfolio, PortfolioPosition
from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from backtest_engine.data.candle import Candle
from backtest_engine.data.tick import Tick
from backtest_engine.sizing.commission import CommissionModel
from backtest_engine.sizing.commission import Side as CommSide
from backtest_engine.sizing.lot_sizer import LotSizer
from backtest_engine.sizing.rate_provider import RateProvider
from backtest_engine.sizing.symbol_specs_registry import SymbolSpec, SymbolSpecsRegistry
from backtest_engine.strategy.strategy_wrapper import TradeSignal
from shared.arrow_schemas import OHLCV_SCHEMA, TRADE_SIGNAL_SCHEMA

if TYPE_CHECKING:
    pass


def _require_rust_always() -> None:
    """Validate OMEGA_USE_RUST_EXECUTION_SIMULATOR=always feature flag.

    Wave 4 requires full Rust delegation. Rollback is deployment-based,
    not runtime fallback.

    Raises:
        ValueError: If env var is not 'always'
    """
    val = os.environ.get("OMEGA_USE_RUST_EXECUTION_SIMULATOR", "always").lower()
    if val != "always":
        raise ValueError(
            "Wave 4 requires OMEGA_USE_RUST_EXECUTION_SIMULATOR=always "
            f"(got {val!r}); rollback is deployment-based, not runtime fallback."
        )


def _datetime_to_utc_micros(t: datetime | int) -> int:
    """Convert datetime to UTC microseconds since epoch."""
    if isinstance(t, int):
        return t
    if t.tzinfo is None:
        utc_dt = t.replace(tzinfo=timezone.utc)
    else:
        utc_dt = t.astimezone(timezone.utc)
    return int(utc_dt.timestamp() * 1_000_000)


def _build_signal_batch(signal: TradeSignal) -> bytes:
    """Build Arrow IPC bytes for a single trade signal.

    Schema: TRADE_SIGNAL_SCHEMA (v2.0.0)
    """
    timestamp_us = _datetime_to_utc_micros(signal.timestamp)
    order_type = getattr(signal, "type", "market")
    reason = getattr(signal, "reason", None)
    scenario = getattr(signal, "scenario", None)

    batch = pa.record_batch(
        [
            pa.array([timestamp_us], type=pa.timestamp("us", tz="UTC")),
            pa.array([signal.direction]).dictionary_encode(),
            pa.array([signal.entry_price], type=pa.float64()),
            pa.array([signal.stop_loss], type=pa.float64()),
            pa.array([signal.take_profit], type=pa.float64()),
            pa.array([0.0], type=pa.float64()),  # size computed in Rust
            pa.array([signal.symbol]).dictionary_encode(),
            pa.array([order_type]).dictionary_encode(),
            pa.array([reason], type=pa.utf8()),
            pa.array([scenario], type=pa.utf8()),
        ],
        schema=TRADE_SIGNAL_SCHEMA,
    )

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, batch.schema) as w:
        w.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _build_candle_batch(candle: Candle) -> bytes:
    """Build Arrow IPC bytes for a single candle.

    Schema: OHLCV_SCHEMA (v2.0.0)
    """
    timestamp_us = _datetime_to_utc_micros(candle.timestamp)

    batch = pa.record_batch(
        [
            pa.array([timestamp_us], type=pa.timestamp("us", tz="UTC")),
            pa.array([candle.open], type=pa.float64()),
            pa.array([candle.high], type=pa.float64()),
            pa.array([candle.low], type=pa.float64()),
            pa.array([candle.close], type=pa.float64()),
            pa.array([getattr(candle, "volume", 0.0)], type=pa.float64()),
            pa.array([True], type=pa.bool_()),  # valid=True
        ],
        schema=OHLCV_SCHEMA,
    )

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, batch.schema) as w:
        w.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _positions_from_ipc(ipc_bytes: bytes, portfolio: Any) -> List[PortfolioPosition]:
    """Materialize PortfolioPosition objects from Arrow IPC bytes.

    This extracts position data from Rust and creates Python PortfolioPosition
    objects for Portfolio integration.
    """
    if not ipc_bytes:
        return []

    reader = ipc.open_stream(pa.BufferReader(ipc_bytes))
    positions: List[PortfolioPosition] = []

    for batch in reader:
        entry_time_col = batch.column("entry_time")
        exit_time_col = batch.column("exit_time")
        direction_col = batch.column("direction")
        symbol_col = batch.column("symbol")
        entry_price_col = batch.column("entry_price")
        exit_price_col = batch.column("exit_price")
        initial_sl_col = batch.column("initial_sl")
        current_sl_col = batch.column("current_sl")
        tp_col = batch.column("tp")
        size_col = batch.column("size")
        result_col = batch.column("result")
        status_col = batch.column("status")

        for i in range(batch.num_rows):
            entry_time_us = entry_time_col[i].as_py()
            if isinstance(entry_time_us, int):
                entry_time = datetime.fromtimestamp(
                    entry_time_us / 1_000_000, tz=timezone.utc
                )
            else:
                entry_time = entry_time_us

            exit_time = None
            if exit_time_col[i].is_valid:
                exit_time_us = exit_time_col[i].as_py()
                if isinstance(exit_time_us, int):
                    exit_time = datetime.fromtimestamp(
                        exit_time_us / 1_000_000, tz=timezone.utc
                    )
                else:
                    exit_time = exit_time_us

            pos = PortfolioPosition(
                entry_time=entry_time,
                direction=direction_col[i].as_py(),
                symbol=symbol_col[i].as_py(),
                entry_price=entry_price_col[i].as_py(),
                stop_loss=current_sl_col[i].as_py(),
                take_profit=tp_col[i].as_py(),
                size=size_col[i].as_py(),
                order_type="market",  # simplified
                status=status_col[i].as_py(),
                risk_per_trade=100.0,  # TODO: pass through metadata
            )
            pos.initial_stop_loss = initial_sl_col[i].as_py()
            pos.initial_take_profit = tp_col[i].as_py()

            if exit_time and exit_price_col[i].is_valid:
                pos._exit_time = exit_time
                pos._exit_price = exit_price_col[i].as_py()
                pos._result = result_col[i].as_py() if result_col[i].is_valid else 0.0

            positions.append(pos)

    return positions


class ExecutionSimulatorRustWrapper:
    """Thin Wrapper für Rust ExecutionSimulatorRust.

    Wave 4 Phase 6: Alle Logik wird an Rust delegiert.
    Python ist nur für:
    - Input validation und Arrow IPC encoding
    - Portfolio integration (register_entry/exit/fee)
    - API Kompatibilität (gleiche Methodennamen)

    Note:
        Diese Klasse ist der Drop-In Replacement für ExecutionSimulator.
        Um sie zu aktivieren, setze OMEGA_USE_RUST_EXECUTION_SIMULATOR=always.

    Attributes:
        portfolio: Portfolio instance for balance/exposure tracking
        risk_per_trade: Risk amount per trade in account currency
        symbol_specs: Symbol specifications for sizing/pip_size
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk_per_trade: float = 100.0,
        slippage_model: Optional[SlippageModel] = None,
        fee_model: Optional[FeeModel] = None,
        symbol_specs: Optional[
            Union[Dict[str, SymbolSpec], SymbolSpecsRegistry]
        ] = None,
        lot_sizer: Optional[LotSizer] = None,
        commission_model: Optional[CommissionModel] = None,
        rate_provider: Optional[RateProvider] = None,
        *,
        base_seed: Optional[int] = None,
        max_slippage_pips: float = 1.0,
        pip_buffer_factor: float = 0.5,
    ) -> None:
        """Initialize ExecutionSimulator with Rust backend.

        Args:
            portfolio: Portfolio instance for tracking
            risk_per_trade: Risk per trade in account currency
            slippage_model: Legacy param (ignored, use base_seed for Rust slippage)
            fee_model: Legacy param (fees computed in Rust)
            symbol_specs: Dict/Registry of symbol -> SymbolSpec for sizing
            lot_sizer: Legacy param (sizing done in Rust)
            commission_model: Legacy param (fees computed in Rust)
            rate_provider: Legacy param (not used in Rust)
            base_seed: Seed for deterministic slippage RNG
            max_slippage_pips: Maximum slippage in pips
            pip_buffer_factor: Buffer factor for SL/TP detection
        """
        _require_rust_always()

        self.portfolio = portfolio
        self.risk_per_trade = risk_per_trade
        self.pip_buffer_factor = pip_buffer_factor

        # Normalize symbol_specs to dict
        self.symbol_specs: Dict[str, SymbolSpec]
        if isinstance(symbol_specs, SymbolSpecsRegistry):
            self.symbol_specs = symbol_specs._specs
        else:
            self.symbol_specs = symbol_specs or {}

        # Legacy attributes for API compatibility
        self.slippage_model = slippage_model
        self.fee_model = fee_model
        self.commission_model = commission_model
        self.lot_sizer = lot_sizer
        self._rate_provider = rate_provider

        # Initialize Rust backend
        self._rust = ExecutionSimulatorRust(
            risk_per_trade=risk_per_trade,
            pip_buffer_factor=pip_buffer_factor,
            base_seed=base_seed,
            max_slippage_pips=max_slippage_pips,
        )

        # Register symbol specs with Rust backend
        for symbol, spec in self.symbol_specs.items():
            self._register_symbol_spec(symbol, spec)

        # Track last known positions for Portfolio sync
        self._last_open_count = 0
        self._last_closed_count = 0

    def _register_symbol_spec(self, symbol: str, spec: SymbolSpec) -> None:
        """Register a symbol specification with the Rust backend."""
        pip_size = getattr(spec, "pip_size", 0.0001) or 0.0001
        contract_size = getattr(spec, "contract_size", 100000.0) or 100000.0
        volume_min = getattr(spec, "volume_min", 0.01) or 0.01
        volume_step = getattr(spec, "volume_step", 0.01) or 0.01
        volume_max = getattr(spec, "volume_max", 100.0) or 100.0
        tick_size = getattr(spec, "tick_size", None)
        tick_value = getattr(spec, "tick_value", None)

        self._rust.add_symbol_spec(
            symbol,
            float(pip_size),
            float(contract_size),
            float(volume_min),
            float(volume_step),
            float(volume_max),
            tick_size,
            tick_value,
        )

    @property
    def active_positions(self) -> List[PortfolioPosition]:
        """Get active positions from Rust backend."""
        ipc_bytes = self._rust.get_active_positions_ipc()
        return _positions_from_ipc(bytes(ipc_bytes), self.portfolio)

    @active_positions.setter
    def active_positions(self, value: List[PortfolioPosition]) -> None:
        """Setter for API compatibility (no-op, state is in Rust)."""
        # State is managed in Rust, this is just for API compatibility
        pass

    def process_signal(self, signal: TradeSignal) -> None:
        """Process a new trade signal (Market, Limit, Stop).

        Delegates to Rust ExecutionSimulatorRust.process_signals_batch().
        After processing, syncs entries with Portfolio.

        Args:
            signal: TradeSignal with direction, entry_price, sl, tp, etc.
        """
        # Track counts before processing
        open_before = self._rust.open_position_count

        # Build Arrow IPC batch and send to Rust
        ipc_bytes = _build_signal_batch(signal)
        self._rust.process_signals_batch(ipc_bytes)

        # Sync new entries with Portfolio
        open_after = self._rust.open_position_count
        if open_after > open_before:
            # New position was opened, register with Portfolio
            positions = self.active_positions
            for pos in positions:
                if pos.status == "open":
                    self.portfolio.register_entry(pos)

    def evaluate_exits(
        self, bid_candle: Candle, ask_candle: Optional[Candle] = None
    ) -> None:
        """Evaluate exits for all open positions.

        Delegates to Rust ExecutionSimulatorRust.evaluate_exits_batch().
        After evaluation, syncs closed positions with Portfolio.

        Args:
            bid_candle: Bid OHLCV candle
            ask_candle: Optional Ask OHLCV candle
        """
        # Track counts before evaluation
        closed_before = self._rust.closed_position_count

        # Build Arrow IPC batches
        bid_ipc = _build_candle_batch(bid_candle)
        ask_ipc = _build_candle_batch(ask_candle) if ask_candle else None

        # Send to Rust
        closed_count = self._rust.evaluate_exits_batch(bid_ipc, ask_ipc)

        # Sync closed positions with Portfolio
        if closed_count > 0:
            closed_positions_ipc = self._rust.get_closed_positions_ipc()
            closed_positions = _positions_from_ipc(
                bytes(closed_positions_ipc), self.portfolio
            )

            # Only process newly closed positions
            for pos in closed_positions[-closed_count:]:
                if pos.is_closed:
                    self.portfolio.register_exit(pos)

    # =========================================================================
    # Tick Mode Methods (API Compatibility)
    # =========================================================================

    def check_if_entry_triggered_tick(self, pos: PortfolioPosition, tick: Tick) -> bool:
        """Check if pending entry is triggered in tick mode.

        Note: In the Rust wrapper, this is handled internally by evaluate_exits.
        This method exists for API compatibility.
        """
        return pos.status == "pending"

    def trigger_entry_tick(self, pos: PortfolioPosition, tick: Tick) -> None:
        """Trigger pending entry in tick mode.

        Note: In the Rust wrapper, entries are handled internally.
        This method exists for API compatibility.
        """
        pass

    def process_signal_tick(self, signal: TradeSignal, tick: Tick) -> None:
        """Process a signal in tick mode.

        Delegates to process_signal (candle mode).
        """
        self.process_signal(signal)

    def evaluate_exits_tick(self, tick: Tick) -> None:
        """Evaluate exits in tick mode.

        Converts tick to candle and delegates to evaluate_exits.
        """
        # Create a synthetic candle from tick
        candle = Candle(
            timestamp=tick.timestamp,
            open=tick.bid,
            high=tick.bid,
            low=tick.bid,
            close=tick.bid,
            volume=0.0,
        )
        ask_candle = Candle(
            timestamp=tick.timestamp,
            open=tick.ask,
            high=tick.ask,
            low=tick.ask,
            close=tick.ask,
            volume=0.0,
        )
        self.evaluate_exits(candle, ask_candle)

    # =========================================================================
    # Legacy Methods (API Compatibility, delegates to Rust)
    # =========================================================================

    def check_if_entry_triggered(
        self, pos: PortfolioPosition, bid_candle: Candle, ask_candle: Optional[Candle]
    ) -> bool:
        """Check if pending entry is triggered.

        Note: In the Rust wrapper, this is handled internally by evaluate_exits.
        This method exists for API compatibility.
        """
        return pos.status == "pending"

    def trigger_entry(self, pos: PortfolioPosition, entry_candle: Candle) -> None:
        """Trigger pending entry.

        Note: In the Rust wrapper, entries are handled internally.
        This method exists for API compatibility.
        """
        pass

    def _pip_size_for_symbol(self, symbol: str) -> float:
        """Get pip size for symbol (for API compatibility)."""
        spec = self.symbol_specs.get(symbol)
        if spec and getattr(spec, "pip_size", None):
            return float(spec.pip_size)
        return 0.0001

    def _get_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """Get SymbolSpec for symbol (for API compatibility)."""
        return self.symbol_specs.get(symbol)

    def _unit_value_per_price(self, symbol: str) -> float:
        """Get unit value per price (for API compatibility)."""
        spec = self.symbol_specs.get(symbol)
        if spec:
            if hasattr(spec, "tick_value") and hasattr(spec, "tick_size"):
                if spec.tick_value and spec.tick_size and spec.tick_size > 0:
                    return spec.tick_value / spec.tick_size
            if hasattr(spec, "contract_size") and spec.contract_size:
                return spec.contract_size
        return 100000.0

    def _quantize_volume(self, symbol: str, size_lots: float) -> float:
        """Quantize volume (for API compatibility)."""
        spec = self.symbol_specs.get(symbol)
        if not spec:
            return max(0.01, round(size_lots, 2))
        vol_min = getattr(spec, "volume_min", 0.01) or 0.01
        vol_max = getattr(spec, "volume_max", 100.0) or 100.0
        vol_step = getattr(spec, "volume_step", 0.01) or 0.01
        if vol_step <= 0:
            vol_step = 0.01
        quantized = max(vol_min, (size_lots // vol_step) * vol_step)
        return min(vol_max, quantized)
