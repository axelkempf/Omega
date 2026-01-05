import math
from typing import Any, Dict, List, Optional, Union

from backtest_engine.core.portfolio import Portfolio, PortfolioPosition
from backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from backtest_engine.data.candle import Candle
from backtest_engine.data.tick import Tick
from backtest_engine.sizing.commission import CommissionModel
from backtest_engine.sizing.commission import Side as CommSide
from backtest_engine.sizing.lot_sizer import LotSizer
from backtest_engine.sizing.rate_provider import RateProvider
from backtest_engine.sizing.symbol_specs_registry import (  # falls du Registry injizierst
    SymbolSpec,
    SymbolSpecsRegistry,
)
from backtest_engine.strategy.strategy_wrapper import TradeSignal

# Für Forex: Standard Lotgröße
LOT_SIZE = 100_000


class ExecutionSimulator:
    """
    Simuliert die Orderausführung (Market, Limit, Stop) im Backtest.

    Args:
        portfolio: Portfolio-Objekt.
        risk_per_trade: Fixes Risiko je Trade (z.B. 100.0).
        slippage_model: Optionales Slippage-Modell.
        fee_model: Optionales Gebührenmodell.
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
    ) -> None:
        self.portfolio = portfolio
        self.active_positions: List[PortfolioPosition] = []
        self.risk_per_trade = risk_per_trade
        self.slippage_model = slippage_model
        self.fee_model = fee_model
        self.performance_mode = True
        self._pip_cache: Dict[str, tuple[float, float]] = {}
        # symbol_specs darf dict oder Registry sein. Intern als dict nutzen.
        self.symbol_specs: Dict[str, SymbolSpec]
        if isinstance(symbol_specs, SymbolSpecsRegistry):
            self.symbol_specs = symbol_specs._specs
        else:
            self.symbol_specs = symbol_specs or {}
        self.lot_sizer = lot_sizer
        self.commission_model = commission_model
        self._rate_provider: Optional[RateProvider] = rate_provider or (
            RateProvider() if RateProvider is not None else None
        )
        self._unit_value_cache: Dict[str, float] = {}

    def _pip_size_for_symbol(self, symbol: str) -> float:
        spec = self._get_spec(symbol)
        if spec and getattr(spec, "pip_size", None):
            return float(spec.pip_size)
        # Harte Warnung: Spezifikation fehlt – letzte Notlösung 0.0001
        # (Für Indizes/Metalle UNZULÄSSIG -> bitte SymbolSpecs pflegen)
        return 0.0001

    def _get_spec(self, symbol: str) -> Optional[SymbolSpec]:
        return self.symbol_specs.get(symbol)

    def _unit_value_per_price(self, symbol: str) -> float:
        """
        Geldwert (Konto-Währung) pro 1.0 Preis-Einheit für 1 Lot.
        = tick_value / tick_size, sonst FX-Fallback via RateProvider.
        """
        # Fast path: cached per symbol
        uv = self._unit_value_cache.get(symbol)
        if uv is not None:
            return uv
        spec = self._get_spec(symbol)
        if spec and spec.tick_size and spec.tick_value:
            uv = spec.tick_value / spec.tick_size
        else:
            # Single provider, no per-call instantiation
            rp = self._rate_provider
            if rp:
                quote = (
                    spec.quote_currency if spec and spec.quote_currency else symbol[3:]
                ).upper()
                acct = getattr(rp, "account_currency", "USD")
                cs = spec.contract_size if spec and spec.contract_size else LOT_SIZE
                # Notional von 1.0 Preis-Einheit * contract_size in Konto-CCY
                notional_acct, _ = rp.fx_convert(amount=cs, from_ccy=quote, to_ccy=acct)
                uv = float(notional_acct)
            else:
                uv = float(LOT_SIZE)
        self._unit_value_cache[symbol] = float(uv)
        return float(uv)

    def _quantize_volume(self, symbol: str, raw_lots: float) -> float:
        """
        Volumen auf broker-konforme Raster bündig nach unten quantisieren,
        damit das Risiko nicht überschritten wird.
        """
        spec = self._get_spec(symbol)
        if not spec:
            return max(0.01, round(raw_lots, 2))
        step = spec.volume_step or 0.01
        vmin = spec.volume_min or step
        vmax = spec.volume_max or float("inf")
        if raw_lots <= vmin:
            lots = vmin
        else:
            n_steps = math.floor((raw_lots - vmin + 1e-12) / step)
            lots = vmin + n_steps * step
        if lots > vmax:
            lots = vmax
        return float(f"{lots:.8f}")

    def check_if_entry_triggered(
        self,
        pos: PortfolioPosition,
        bid_candle: Candle,
        ask_candle: Optional[Candle] = None,
    ) -> bool:
        """Prüft, ob ein Pending-Entry im Candle-Modus ausgelöst wird."""
        if pos.status != "pending":
            return False
        if bid_candle.timestamp <= pos.entry_time:
            return False
        if pos.order_type == "limit":
            if pos.direction == "long":
                return ask_candle is not None and ask_candle.low <= pos.entry_price
            elif pos.direction == "short":
                return bid_candle.high >= pos.entry_price
        elif pos.order_type == "stop":
            if pos.direction == "long":
                return ask_candle is not None and ask_candle.high >= pos.entry_price
            elif pos.direction == "short":
                return bid_candle.low <= pos.entry_price
        return False

    def trigger_entry(self, pos: PortfolioPosition, candle: Candle) -> None:
        """Setzt eine Pending-Position auf OPEN und berechnet Positionsgröße."""
        pip = self._pip_size_for_symbol(pos.symbol)
        stop_pips = abs(pos.entry_price - pos.stop_loss) / pip if pip > 0 else 0.0
        if self.lot_sizer:
            pos.size = self.lot_sizer.size_risk_based(
                symbol=pos.symbol,
                price=pos.entry_price,
                stop_pips=stop_pips,
                risk_amount_acct=self.risk_per_trade,
                t=candle.timestamp,
            )
        else:
            # legacy fallback
            sl_distance = abs(pos.entry_price - pos.stop_loss)
            unit_val = self._unit_value_per_price(pos.symbol)
            risk_per_lot = sl_distance * unit_val
            size_lots = (
                (self.risk_per_trade / risk_per_lot) if risk_per_lot > 0 else 0.0
            )
            pos.size = self._quantize_volume(pos.symbol, size_lots)
        pos.status = "open"
        pos.trigger_time = candle.timestamp
        pos.initial_stop_loss = pos.stop_loss
        pos.initial_take_profit = pos.take_profit
        self.portfolio.register_entry(pos)
        # Fees
        if self.commission_model:
            fee = self.commission_model.fee_for_order(
                pos.symbol,
                pos.size,
                pos.entry_price,
                t=candle.timestamp,
                side=CommSide.ENTRY,
            )
            self.portfolio.register_fee(
                fee, candle.timestamp, kind="entry", position=pos
            )
        elif self.fee_model:
            spec = self._get_spec(pos.symbol)
            fee = self.fee_model.calculate(
                pos.size,
                pos.entry_price,
                contract_size=(spec.contract_size if spec else None),
            )
            self.portfolio.register_fee(
                fee, candle.timestamp, kind="entry", position=pos
            )

    def process_signal(self, signal: TradeSignal) -> None:
        """
        Verarbeitet ein neues Signal (Market, Limit, Stop).
        """
        order_type = getattr(signal, "type", "market")
        # Meta & Tagging vom Signal (robust gegen fehlende Felder)
        _sig_meta = {}
        try:
            if hasattr(signal, "meta") and isinstance(signal.meta, dict):
                _sig_meta.update(signal.meta)
        except Exception:
            pass
        # Falls StrategyWrapper "scenario"/"tags" top-level übergibt, ebenfalls sichern
        for k in ("scenario", "tags", "reason"):
            try:
                if hasattr(signal, k):
                    _sig_meta.setdefault(k, getattr(signal, k))
            except Exception:
                pass
        if order_type in ["limit", "stop"]:
            position = PortfolioPosition(
                entry_time=signal.timestamp,
                direction=signal.direction,
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                size=0,
                order_type=order_type,
                status="pending",
                risk_per_trade=self.risk_per_trade,
                metadata=_sig_meta,
            )
            position.initial_stop_loss = position.stop_loss
            position.initial_take_profit = position.take_profit
            self.active_positions.append(position)
            return

        # Market Orders
        entry_price = signal.entry_price
        if self.slippage_model:
            entry_price = self.slippage_model.apply(
                entry_price,
                signal.direction,
                pip_size=self._pip_size_for_symbol(signal.symbol),
            )
        sl_distance = abs(entry_price - signal.stop_loss)
        if sl_distance < 0.00001:
            print("⚠️ SL-Distanz zu klein – Trade ignoriert")
            return
        pip = self._pip_size_for_symbol(signal.symbol)
        stop_pips = sl_distance / pip if pip > 0 else 0.0
        if self.lot_sizer:
            size_lots = self.lot_sizer.size_risk_based(
                symbol=signal.symbol,
                price=entry_price,
                stop_pips=stop_pips,
                risk_amount_acct=self.risk_per_trade,
                t=signal.timestamp,
            )
        else:
            unit_val = self._unit_value_per_price(signal.symbol)
            risk_per_lot = sl_distance * unit_val
            size_lots = (
                (self.risk_per_trade / risk_per_lot) if risk_per_lot > 0 else 0.0
            )
        position = PortfolioPosition(
            entry_time=signal.timestamp,
            direction=signal.direction,
            symbol=signal.symbol,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size=(
                self._quantize_volume(signal.symbol, size_lots)
                if not self.lot_sizer
                else size_lots
            ),
            order_type="market",
            status="open",
            risk_per_trade=self.risk_per_trade,
            metadata=_sig_meta,
        )
        position.initial_stop_loss = position.stop_loss
        position.initial_take_profit = position.take_profit
        # Audit: Market-Order Triggerzeit anreichern (Execution=Open)
        try:
            position.trigger_time = signal.timestamp
        except Exception:
            pass
        self.active_positions.append(position)
        self.portfolio.register_entry(position)
        if self.commission_model:
            fee = self.commission_model.fee_for_order(
                position.symbol,
                position.size,
                entry_price,
                t=signal.timestamp,
                side=CommSide.ENTRY,
            )
            self.portfolio.register_fee(
                fee, signal.timestamp, kind="entry", position=position
            )
        elif self.fee_model:
            spec = self._get_spec(position.symbol)
            fee = self.fee_model.calculate(
                position.size,
                entry_price,
                contract_size=(spec.contract_size if spec else None),
            )
            self.portfolio.register_fee(
                fee, signal.timestamp, kind="entry", position=position
            )

    def evaluate_exits(
        self, bid_candle: Candle, ask_candle: Optional[Candle] = None
    ) -> None:
        """
        Prüft, ob offene Positionen geschlossen werden müssen.
        """
        pip_size = (
            self._pip_size_for_symbol(self.portfolio.open_positions[0].symbol)
            if self.portfolio.open_positions
            else 0.0001
        )
        cfg_factor = getattr(self, "pip_buffer_factor", 0.5)  # konfigurierbar
        pip_buffer = float(pip_size) * float(cfg_factor)
        closed: List[PortfolioPosition] = []

        for pos in self.active_positions:
            if pos.is_closed or bid_candle.timestamp <= pos.entry_time:
                continue
            # Pending entry
            if pos.status == "pending" and self.check_if_entry_triggered(
                pos, bid_candle, ask_candle
            ):
                entry_candle = (
                    ask_candle if pos.direction == "long" and ask_candle else bid_candle
                )
                self.trigger_entry(pos, entry_candle)

            if pos.status != "open":
                continue

            in_entry_candle = bid_candle.timestamp == pos.trigger_time
            sl_hit = False
            tp_hit = False

            if pos.direction == "long":
                sl_hit = bid_candle.low <= pos.stop_loss + pip_buffer
                tp_hit = bid_candle.high >= pos.take_profit - pip_buffer
            else:
                # Short positions use ask_candle; fallback to bid_candle if unavailable
                ref_candle = ask_candle if ask_candle is not None else bid_candle
                sl_hit = ref_candle.high >= pos.stop_loss - pip_buffer
                tp_hit = ref_candle.low <= pos.take_profit + pip_buffer

            if in_entry_candle:
                if sl_hit:
                    exit_price = pos.stop_loss
                    reason = "stop_loss"
                elif tp_hit:
                    if pos.order_type == "limit":
                        ref_candle = (
                            ask_candle if ask_candle is not None else bid_candle
                        )
                        close_price = (
                            bid_candle.close
                            if pos.direction == "long"
                            else ref_candle.close
                        )
                        if (
                            pos.direction == "long" and close_price > pos.take_profit
                        ) or (
                            pos.direction == "short" and close_price < pos.take_profit
                        ):
                            exit_price = pos.take_profit
                            reason = "take_profit"
                        else:
                            continue
                    else:
                        exit_price = pos.take_profit
                        reason = "take_profit"
                else:
                    continue
            else:
                if sl_hit:
                    exit_price = pos.stop_loss
                    reason = (
                        "stop_loss"
                        if pos.stop_loss == pos.initial_stop_loss
                        else "break_even_stop_loss"
                    )
                elif tp_hit:
                    exit_price = pos.take_profit
                    reason = "take_profit"
                else:
                    continue

            if self.slippage_model:
                exit_price = self.slippage_model.apply(
                    exit_price,
                    pos.direction,
                    pip_size=self._pip_size_for_symbol(pos.symbol),
                )
            pos.close(bid_candle.timestamp, exit_price, reason=reason)
            if self.commission_model:
                fee = self.commission_model.fee_for_order(
                    pos.symbol,
                    pos.size,
                    exit_price,
                    t=bid_candle.timestamp,
                    side=CommSide.EXIT,
                )
                self.portfolio.register_fee(
                    fee, bid_candle.timestamp, kind="exit", position=pos
                )
            elif self.fee_model:
                spec = self._get_spec(pos.symbol)
                fee = self.fee_model.calculate(
                    pos.size,
                    exit_price,
                    contract_size=(spec.contract_size if spec else None),
                )
                self.portfolio.register_fee(
                    fee, bid_candle.timestamp, kind="exit", position=pos
                )
            self.portfolio.register_exit(pos)
            closed.append(pos)

        self.active_positions = [p for p in self.active_positions if not p.is_closed]

    # === Tick-basierter Entry/Exit ===
    def check_if_entry_triggered_tick(self, pos: PortfolioPosition, tick: Tick) -> bool:
        """Prüft Pending-Entry im Tick-Modus."""
        if pos.status != "pending":
            return False
        if tick.timestamp <= pos.entry_time:
            return False
        if pos.order_type == "limit":
            if pos.direction == "long" and tick.ask <= pos.entry_price:
                return True
            if pos.direction == "short" and tick.bid >= pos.entry_price:
                return True
        elif pos.order_type == "stop":
            if pos.direction == "long" and tick.ask >= pos.entry_price:
                return True
            if pos.direction == "short" and tick.bid <= pos.entry_price:
                return True
        return False

    def trigger_entry_tick(self, pos: PortfolioPosition, tick: Tick) -> None:
        """Setzt Pending-Position auf OPEN und berechnet Größe im Tick-Modus."""
        pip = self._pip_size_for_symbol(pos.symbol)
        stop_pips = abs(pos.entry_price - pos.stop_loss) / pip if pip > 0 else 0.0
        if self.lot_sizer:
            pos.size = self.lot_sizer.size_risk_based(
                pos.symbol,
                pos.entry_price,
                stop_pips,
                self.risk_per_trade,
                t=tick.timestamp,
            )
        else:
            sl_distance = abs(pos.entry_price - pos.stop_loss)
            unit_val = self._unit_value_per_price(pos.symbol)
            risk_per_lot = sl_distance * unit_val
            size_lots = (
                (self.risk_per_trade / risk_per_lot) if risk_per_lot > 0 else 0.0
            )
            pos.size = self._quantize_volume(pos.symbol, size_lots)
        pos.status = "open"
        pos.trigger_time = tick.timestamp
        pos.initial_stop_loss = pos.stop_loss
        pos.initial_take_profit = pos.take_profit
        self.portfolio.register_entry(pos)
        if self.commission_model:
            fee = self.commission_model.fee_for_order(
                pos.symbol,
                pos.size,
                pos.entry_price,
                t=tick.timestamp,
                side=CommSide.ENTRY,
            )
            self.portfolio.register_fee(fee, tick.timestamp, kind="entry", position=pos)
        elif self.fee_model:
            spec = self._get_spec(pos.symbol)
            fee = self.fee_model.calculate(
                pos.size,
                pos.entry_price,
                contract_size=(spec.contract_size if spec else None),
            )
            self.portfolio.register_fee(fee, tick.timestamp, kind="entry", position=pos)

    def process_signal_tick(self, signal: TradeSignal, tick: Tick) -> None:
        """
        Verarbeitet ein neues Tick-Signal (Market, Limit, Stop).
        """
        order_type = getattr(signal, "type", "market")
        entry_price = signal.entry_price or (
            tick.bid if signal.direction == "long" else tick.ask
        )
        if self.slippage_model and order_type == "market":
            entry_price = self.slippage_model.apply(
                entry_price,
                signal.direction,
                pip_size=self._pip_size_for_symbol(signal.symbol),
            )

        # Meta aus Tick-Signalen ebenfalls übernehmen
        _sig_meta = {}
        try:
            if hasattr(signal, "meta") and isinstance(signal.meta, dict):
                _sig_meta.update(signal.meta)
        except Exception:
            pass
        for k in ("scenario", "tags", "reason"):
            try:
                if hasattr(signal, k):
                    _sig_meta.setdefault(k, getattr(signal, k))
            except Exception:
                pass

        if order_type in ["limit", "stop"]:
            pos = PortfolioPosition(
                entry_time=tick.timestamp,
                direction=signal.direction,
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                size=0,
                order_type=order_type,
                status="pending",
                risk_per_trade=self.risk_per_trade,
                metadata=_sig_meta,
            )
            pos.initial_stop_loss = pos.stop_loss
            pos.initial_take_profit = pos.take_profit
            self.active_positions.append(pos)
            return

        sl_distance = abs(entry_price - signal.stop_loss)
        if sl_distance < 0.00001:
            print("⚠️ SL-Distanz zu klein – Trade ignoriert")
            return
        pip = self._pip_size_for_symbol(signal.symbol)
        stop_pips = sl_distance / pip if pip > 0 else 0.0
        if self.lot_sizer:
            size_lots = self.lot_sizer.size_risk_based(
                symbol=signal.symbol,
                price=entry_price,
                stop_pips=stop_pips,
                risk_amount_acct=self.risk_per_trade,
                t=signal.timestamp,
            )
        else:
            unit_val = self._unit_value_per_price(signal.symbol)
            risk_per_lot = sl_distance * unit_val
            size_lots = (
                (self.risk_per_trade / risk_per_lot) if risk_per_lot > 0 else 0.0
            )
        pos = PortfolioPosition(
            entry_time=tick.timestamp,
            direction=signal.direction,
            symbol=signal.symbol,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size=self._quantize_volume(signal.symbol, size_lots),
            status="open",
            risk_per_trade=self.risk_per_trade,
            metadata=_sig_meta,
        )
        pos.initial_stop_loss = pos.stop_loss
        pos.initial_take_profit = pos.take_profit
        self.active_positions.append(pos)
        self.portfolio.register_entry(pos)
        if self.commission_model:
            fee = self.commission_model.fee_for_order(
                pos.symbol,
                pos.size,
                pos.entry_price,
                t=tick.timestamp,
                side=CommSide.ENTRY,
            )
            self.portfolio.register_fee(fee, tick.timestamp, kind="entry", position=pos)
        elif self.fee_model:
            spec = self._get_spec(pos.symbol)
            fee = self.fee_model.calculate(
                pos.size,
                pos.entry_price,
                contract_size=(spec.contract_size if spec else None),
            )
            self.portfolio.register_fee(fee, tick.timestamp, kind="entry", position=pos)

    def evaluate_exits_tick(self, tick: Tick) -> None:
        """Prüft Exits im Tick-Modus."""
        closed: List[PortfolioPosition] = []
        for pos in self.active_positions:
            if pos.is_closed or tick.timestamp <= pos.entry_time:
                continue
            if pos.status == "pending" and self.check_if_entry_triggered_tick(
                pos, tick
            ):
                self.trigger_entry_tick(pos, tick)
            if pos.status != "open":
                continue
            exit_price = None
            reason = None
            if pos.direction == "long":
                if tick.bid <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    reason = "stop_loss"
                elif tick.bid >= pos.take_profit:
                    exit_price = pos.take_profit
                    reason = "take_profit"
            else:
                if tick.ask >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    reason = "stop_loss"
                elif tick.ask <= pos.take_profit:
                    exit_price = pos.take_profit
                    reason = "take_profit"
            if not exit_price:
                continue
            if self.slippage_model:
                exit_price = self.slippage_model.apply(
                    exit_price,
                    pos.direction,
                    pip_size=self._pip_size_for_symbol(pos.symbol),
                )
            pos.close(tick.timestamp, exit_price, reason=reason or "exit")
            if self.commission_model:
                fee = self.commission_model.fee_for_order(
                    pos.symbol,
                    pos.size,
                    exit_price,
                    t=tick.timestamp,
                    side=CommSide.EXIT,
                )
                self.portfolio.register_fee(
                    fee, tick.timestamp, kind="exit", position=pos
                )
            elif self.fee_model:
                spec = self._get_spec(pos.symbol)
                fee = self.fee_model.calculate(
                    pos.size,
                    exit_price,
                    contract_size=(spec.contract_size if spec else None),
                )
                self.portfolio.register_fee(
                    fee, tick.timestamp, kind="exit", position=pos
                )
            self.portfolio.register_exit(pos)
            closed.append(pos)
        self.active_positions = [p for p in self.active_positions if not p.is_closed]
