"""
Portfolio module for backtesting.

This module provides Python implementations of portfolio state management.
When the feature flag OMEGA_USE_RUST_PORTFOLIO is enabled, the Rust backend
(PortfolioRust) is used for improved performance.

Feature Flag: OMEGA_USE_RUST_PORTFOLIO
  - "auto": Use Rust if available (default)
  - "true": Force Rust (raises ImportError if unavailable)
  - "false": Always use Python implementation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Feature Flag Configuration for Rust Backend
# =============================================================================

_FEATURE_FLAG = os.environ.get("OMEGA_USE_RUST_PORTFOLIO", "auto").lower()
_RUST_AVAILABLE = False
_RUST_IMPORT_ERROR: Optional[str] = None

if _FEATURE_FLAG in ("auto", "true"):
    try:
        from omega_rust import PositionRust, PortfolioRust  # type: ignore

        _RUST_AVAILABLE = True
    except ImportError as e:
        _RUST_IMPORT_ERROR = str(e)
        if _FEATURE_FLAG == "true":
            raise ImportError(
                f"OMEGA_USE_RUST_PORTFOLIO='true' but omega_rust not available: {e}"
            ) from e


def get_rust_status() -> Dict[str, Any]:
    """
    Return status information about the Rust backend availability.

    Returns:
        Dict with keys: 'available', 'enabled', 'flag', 'error'
    """
    enabled = _RUST_AVAILABLE and _FEATURE_FLAG != "false"
    return {
        "available": _RUST_AVAILABLE,
        "enabled": enabled,
        "flag": _FEATURE_FLAG,
        "error": _RUST_IMPORT_ERROR,
    }


def _use_rust_backend() -> bool:
    """Check if Rust backend should be used based on feature flag."""
    if _FEATURE_FLAG == "false":
        return False
    return _RUST_AVAILABLE


@dataclass
class PortfolioPosition:
    """
    Repräsentiert eine einzelne Handelsposition im Portfolio.
    """

    entry_time: datetime
    direction: str
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    risk_per_trade: float = 100.0
    initial_stop_loss: Optional[float] = None
    initial_take_profit: Optional[float] = None

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    result: Optional[float] = None  # Gewinn/Verlust
    reason: Optional[str] = None  # z.B. "take_profit" oder "stop_loss"
    is_closed: bool = False
    trigger_time: Optional[datetime] = None
    order_type: str = "market"  # "market", "limit", "stop"
    status: str = "open"  # "open", "pending", "closed"

    last_update: Optional[datetime] = None

    entry_fee: float = 0.0
    exit_fee: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Kompatibilitäts-Properties für Metrics (erwartet open_time/close_time) ---
    @property
    def open_time(self) -> Optional[datetime]:
        return self.entry_time

    @property
    def close_time(self) -> Optional[datetime]:
        return self.exit_time

    def close(self, time: datetime, price: float, reason: str) -> None:
        """
        Schliesst die Position und berechnet den Trade-Result auf Basis des initialen SL.
        """
        self.exit_time = time
        self.exit_price = price
        self.reason = reason
        self.is_closed = True

        initial_sl = (
            self.initial_stop_loss
            if self.initial_stop_loss is not None
            else self.stop_loss
        )
        risk = abs(self.entry_price - initial_sl)
        if risk > 0:
            if self.direction == "long":
                reward = price - self.entry_price
            else:
                reward = self.entry_price - price
            r_multiple = reward / risk
            self.result = r_multiple * self.risk_per_trade
        else:
            self.result = 0.0

    @property
    def r_multiple(self) -> float:
        """
        Gibt das R-Multiple (Chance-Risiko-Verhältnis) der Position zurück.
        """
        if self.exit_price is None:
            return 0.0
        initial_sl = (
            self.initial_stop_loss
            if self.initial_stop_loss is not None
            else self.stop_loss
        )
        risk = abs(self.entry_price - initial_sl)
        if risk == 0:
            return 0.0
        if self.direction == "long":
            return (self.exit_price - self.entry_price) / risk
        else:
            return (self.entry_price - self.exit_price) / risk

    def to_dict(self) -> Dict[str, Any]:
        """
        Exportiert die Position als Dictionary (z.B. für Reporting).
        """
        d: Dict[str, Any] = {
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "direction": self.direction,
            "entry_price": (
                round(self.entry_price, 5) if self.entry_price is not None else None
            ),
            "exit_price": (
                round(self.exit_price, 5) if self.exit_price is not None else None
            ),
            "initial_stop_loss": (
                round(self.initial_stop_loss, 5)
                if self.initial_stop_loss is not None
                else None
            ),
            "stop_loss": (
                round(self.stop_loss, 5) if self.stop_loss is not None else None
            ),
            "take_profit": (
                round(self.take_profit, 5) if self.take_profit is not None else None
            ),
            "size": self.size,
            "result": round(self.result, 2) if self.result is not None else None,
            "reason": self.reason,
            "order_type": self.order_type,
            "status": self.status,
            "r_multiple": (
                round(self.r_multiple, 5) if self.r_multiple is not None else None
            ),
        }
        # Meta roh durchreichen (möglichst serialisierbar halten)
        if self.metadata:
            d["meta"] = self.metadata
        return d

    # =========================================================================
    # Rust Conversion Methods
    # =========================================================================

    def _to_rust(self) -> Any:
        """
        Convert this PortfolioPosition to a PositionRust instance.

        Returns:
            PositionRust instance if Rust backend is available

        Raises:
            ImportError: If Rust backend is not available
        """
        if not _RUST_AVAILABLE:
            raise ImportError("Rust backend not available for position conversion")

        direction_int: int = 1 if self.direction == "long" else -1
        entry_time_us = int(self.entry_time.timestamp() * 1_000_000)

        pos = PositionRust(
            entry_time=entry_time_us,
            direction=direction_int,
            symbol=self.symbol,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            size=self.size,
            risk_per_trade=self.risk_per_trade,
        )

        # Set optional fields
        if self.initial_stop_loss is not None:
            pos.initial_stop_loss = self.initial_stop_loss
        if self.initial_take_profit is not None:
            pos.initial_take_profit = self.initial_take_profit
        if self.exit_time is not None:
            pos.exit_time = int(self.exit_time.timestamp() * 1_000_000)
        if self.exit_price is not None:
            pos.exit_price = self.exit_price
        if self.result is not None:
            pos.result = self.result
        if self.reason is not None:
            pos.reason = self.reason

        pos.is_closed = self.is_closed
        pos.order_type = self.order_type
        pos.status = self.status
        pos.entry_fee = self.entry_fee
        pos.exit_fee = self.exit_fee

        return pos

    @classmethod
    def _from_rust(cls, rust_pos: Any) -> "PortfolioPosition":
        """
        Create a PortfolioPosition from a PositionRust instance.

        Args:
            rust_pos: PositionRust instance

        Returns:
            New PortfolioPosition instance
        """
        direction_str = "long" if rust_pos.direction == 1 else "short"
        entry_time = datetime.utcfromtimestamp(rust_pos.entry_time / 1_000_000)

        pos = cls(
            entry_time=entry_time,
            direction=direction_str,
            symbol=rust_pos.symbol,
            entry_price=rust_pos.entry_price,
            stop_loss=rust_pos.stop_loss,
            take_profit=rust_pos.take_profit,
            size=rust_pos.size,
            risk_per_trade=rust_pos.risk_per_trade,
        )

        # Set optional fields
        pos.initial_stop_loss = rust_pos.initial_stop_loss
        pos.initial_take_profit = rust_pos.initial_take_profit

        if rust_pos.exit_time is not None:
            pos.exit_time = datetime.utcfromtimestamp(rust_pos.exit_time / 1_000_000)
        pos.exit_price = rust_pos.exit_price
        pos.result = rust_pos.result
        pos.reason = rust_pos.reason
        pos.is_closed = rust_pos.is_closed
        pos.order_type = rust_pos.order_type
        pos.status = rust_pos.status
        pos.entry_fee = rust_pos.entry_fee
        pos.exit_fee = rust_pos.exit_fee

        return pos


class Portfolio:
    """
    Verwaltet alle offenen und geschlossenen Positionen, berechnet Equity und Statistiken.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.equity = initial_balance
        self.start_timestamp: Optional[datetime] = None
        self.open_positions: List[PortfolioPosition] = []
        self.closed_positions: List[PortfolioPosition] = []
        self.expired_orders: List[PortfolioPosition] = []
        self.partial_closed_positions: List[PortfolioPosition] = []
        self.closed_position_break_even: List[PortfolioPosition] = []
        self.max_equity = initial_balance
        self.max_drawdown = 0.0
        self.initial_max_drawdown = 0.0
        self.equity_curve: List[Tuple[datetime, float]] = [
            (datetime.min.replace(year=2000, month=1, day=1), initial_balance)
        ]
        self.total_fees: float = 0.0
        self.fees_log: List[Dict[str, Any]] = []

    def register_fee(
        self,
        amount: float,
        time: datetime,
        kind: str,
        position: Optional[PortfolioPosition] = None,
    ) -> None:
        """
        Verbucht eine Gebühr (zieht Cash ab), protokolliert sie und ordnet sie optional einer Position zu.
        kind: "entry" | "exit" | "other"
        """
        fee = float(amount or 0.0)
        if fee == 0.0:
            return
        self.cash -= fee
        self.total_fees += fee
        self.equity = self.cash
        self.fees_log.append(
            {
                "time": time,
                "kind": kind,
                "symbol": getattr(position, "symbol", None) if position else None,
                "size": getattr(position, "size", None) if position else None,
                "fee": fee,
            }
        )
        if position is not None:
            if kind == "entry":
                position.entry_fee += fee
            elif kind == "exit":
                position.exit_fee += fee

    def register_entry(self, position: PortfolioPosition) -> None:
        """
        Fügt eine neue Position zum Portfolio hinzu.
        """
        if not position.symbol:
            raise ValueError("Position must have a 'symbol' assigned.")
        self.open_positions.append(position)

    def register_exit(self, position: PortfolioPosition) -> None:
        """
        Behandelt das Schliessen von Positionen nach Exit/Abbruch etc.
        """
        # Pending Expiry
        if position.status == "pending" and position.reason == "limit_expired":
            self.expired_orders.append(position)
            position.status = "closed"
            if position in self.open_positions:
                self.open_positions.remove(position)
            return

        # Partial oder Break Even
        if position.status == "open" and position.reason == "partial_exit":
            self.partial_closed_positions.append(position)
        elif position.status == "open" and position.reason == "break_even_stop_loss":
            self.closed_positions.append(position)
            self.closed_position_break_even.append(position)
        elif position.status == "open":
            self.closed_positions.append(position)

        position.status = "closed"

        # Gutschrift/Belastung
        result = position.result if position.result is not None else 0.0
        self.cash += result
        self.equity = self.cash

        if position in self.open_positions:
            self.open_positions.remove(position)

        # Falls StrategyWrapper existiert: Exit-Zeit dort setzen
        try:
            if hasattr(self, "strategy_wrapper") and self.strategy_wrapper:
                self.strategy_wrapper.last_exit_time = position.exit_time
        except Exception:
            pass

        # Konsistente Cooldowns: wenn StrategyWrapper auf Decision-Time 'close' läuft,
        # speichere last_exit_time als Exit-Close (Open + TF-Dauer).
        try:
            wrapper = getattr(self, "strategy_wrapper", None)
            if wrapper is not None:
                tf = str(wrapper.get_primary_timeframe()).upper()
                if tf.startswith("M"):
                    minutes = int(tf[1:])
                elif tf.startswith("H"):
                    minutes = int(tf[1:]) * 60
                elif tf.startswith("D"):
                    minutes = int(tf[1:]) * 1440
                else:
                    minutes = 0
                if (
                    getattr(wrapper, "entry_timestamp_mode", "open") == "close"
                    and minutes > 0
                    and position.exit_time is not None
                ):
                    exit_close = position.exit_time + timedelta(minutes=minutes)
                    wrapper.last_exit_time = exit_close
                else:
                    wrapper.last_exit_time = position.exit_time
        except Exception:
            # niemals die Verbuchung riskieren
            pass

    def get_open_positions(
        self, symbol: Optional[str] = None
    ) -> List[PortfolioPosition]:
        """
        Gibt alle offenen Positionen (optional gefiltert nach Symbol) zurück.
        """
        if symbol:
            return [pos for pos in self.open_positions if pos.symbol == symbol]
        return self.open_positions

    def update(self, current_time: datetime) -> None:
        """
        Wird in jedem Event aufgerufen – aktualisiert Equity & Drawdown.
        """
        if self.start_timestamp is None and current_time is not None:
            self.start_timestamp = current_time

        self.equity = self.cash

        # Equity-Kurve fortschreiben
        try:
            if self.equity_curve and self.equity_curve[-1][0] == current_time:
                self.equity_curve[-1] = (current_time, float(self.equity))
            else:
                self.equity_curve.append((current_time, float(self.equity)))
        except Exception:
            pass

        if self.equity > self.max_equity:
            self.max_equity = self.equity

        drawdown = self.max_equity - self.equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        drawdown_initial = self.initial_balance - self.equity
        if drawdown_initial > self.initial_max_drawdown:
            self.initial_max_drawdown = drawdown_initial

    def get_summary(self) -> Dict[str, float]:
        """
        Gibt ein Dictionary mit den wichtigsten Metriken und Statistiken zurück.
        """
        summary: Dict[str, float] = {
            "Initial Balance": self.initial_balance,
            "Final Balance": round(self.cash, 2),
            "Equity": round(self.equity, 2),
            "Max Drawdown": round(self.max_drawdown, 2),
            "Drawdown Initial Balance": round(self.initial_max_drawdown, 2),
            "Total Fees": round(self.total_fees, 2),
            "Total Lots": round(
                sum(
                    p.size
                    for p in (self.closed_positions + self.partial_closed_positions)
                ),
                2,
            ),
            "Total Trades": len(self.closed_positions),
            "Expired Orders": len(self.expired_orders),
            "Partial Closed Orders": len(self.partial_closed_positions),
            "Orders closed at Break Even": len(self.closed_position_break_even),
            "Avg R-Multiple": self._avg_r_multiple(),
            "Winrate": self._winrate(),
            "Wins": len(
                [
                    p
                    for p in self.closed_positions
                    if p.result is not None and p.result > 0
                ]
            ),
            "Losses": len(
                [
                    p
                    for p in self.closed_positions
                    if p.result is not None and p.result <= 0
                ]
            ),
        }

        # Optional: Erweiterte Robustheits-/Stabilitätsmetriken nur für Backtests
        # per Config-Flag en-/disablebar (z.B. config["reporting"]["enable_backtest_robust_metrics"]).
        try:
            enabled = bool(getattr(self, "enable_backtest_robust_metrics", False))
        except Exception:
            enabled = False
        if enabled:
            try:
                # Verwende nur vorab im Runner berechnete Werte; keine Berechnung mehr hier
                extra = getattr(self, "backtest_robust_metrics", None)
                if isinstance(extra, dict):
                    summary.update(
                        {
                            "Robustness 1": extra.get("robustness_1", 0.0),
                            "Robustness 1 Num Samples": extra.get(
                                "robustness_1_num_samples", 0
                            ),
                            "Cost Shock Score": extra.get("cost_shock_score", 0.0),
                            "Timing Jitter Score": extra.get(
                                "timing_jitter_score", 0.0
                            ),
                            "Trade Dropout Score": extra.get(
                                "trade_dropout_score", 0.0
                            ),
                            "Ulcer Index": extra.get("ulcer_index", 0.0),
                            "Ulcer Index Score": extra.get("ulcer_index_score", 0.0),
                            "Data Jitter Score": extra.get("data_jitter_score", 0.0),
                            "Data Jitter Num Samples": extra.get(
                                "data_jitter_num_samples", 0
                            ),
                            "p_mean_gt": extra.get("p_mean_gt", 1.0),
                            "Stability Score": extra.get("stability_score", 1.0),
                            "TP/SL Stress Score": extra.get("tp_sl_stress_score", 1.0),
                        }
                    )
                else:
                    # Ausgabe-Form beibehalten, falls aktiviert aber nichts gesetzt
                    summary.update(
                        {
                            "Robustness 1": 0.0,
                            "Robustness 1 Num Samples": 0,
                            "Cost Shock Score": 0.0,
                            "Timing Jitter Score": 0.0,
                            "Trade Dropout Score": 0.0,
                            "Ulcer Index": 0.0,
                            "Ulcer Index Score": 0.0,
                            "Data Jitter Score": 0.0,
                            "Data Jitter Num Samples": 0,
                            "p_mean_gt": 1.0,
                            "Stability Score": 1.0,
                            "TP/SL Stress Score": 1.0,
                        }
                    )
            except Exception:
                # Niemals Zusammenfassung gefährden
                pass

        return summary

    def _winrate(self) -> float:
        """
        Berechnet die Gewinnquote abgeschlossener Trades.
        """
        wins = [
            p for p in self.closed_positions if p.result is not None and p.result > 0
        ]
        return (
            round(len(wins) / len(self.closed_positions) * 100, 2)
            if self.closed_positions
            else 0.0
        )

    def _avg_r_multiple(self) -> float:
        """
        Durchschnittliches (gewichtetes) R-Multiple über alle abgeschlossenen Trades.
        """
        all_positions = self.closed_positions + self.partial_closed_positions
        all_positions = [
            p
            for p in all_positions
            if p.r_multiple is not None and p.risk_per_trade > 0
        ]

        if not all_positions:
            return 0.0

        total_weighted_r = sum(p.r_multiple * p.risk_per_trade for p in all_positions)
        total_risk = sum(p.risk_per_trade for p in all_positions)

        return round(total_weighted_r / total_risk, 3) if total_risk > 0 else 0.0

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """
        Gibt eine Equity-Kurve zurück als Liste von (Zeitpunkt, Equity)-Tupeln.
        Nutzt echte und teilweise abgeschlossene Trades (nach Fees).
        """
        curve: List[Tuple[datetime, float]] = []
        start_ts = getattr(self, "start_timestamp", None)

        # Fallback auf erste Equity-Kurve ohne Platzhalter (falls Startzeitpunkt nicht gesetzt)
        if start_ts is None:
            try:
                placeholder = datetime.min.replace(year=2000, month=1, day=1)
                first_ts = self.equity_curve[0][0] if self.equity_curve else None
                if isinstance(first_ts, datetime) and first_ts != placeholder:
                    start_ts = first_ts
            except Exception:
                start_ts = None

        if start_ts is not None:
            curve.append((start_ts, float(self.initial_balance)))

        equity = float(self.initial_balance)

        all_positions = self.closed_positions + self.partial_closed_positions
        # Filter positions with valid result and exit_time
        valid_positions = [
            p for p in all_positions if p.result is not None and p.exit_time is not None
        ]

        # Sort by exit_time (guaranteed non-None after filter)
        for p in sorted(valid_positions, key=lambda x: x.exit_time or datetime.min):
            entry_fee = float(getattr(p, "entry_fee", 0.0) or 0.0)
            exit_fee = float(getattr(p, "exit_fee", 0.0) or 0.0)
            result_val = p.result if p.result is not None else 0.0
            net_result = result_val - entry_fee - exit_fee
            equity += net_result
            # exit_time is guaranteed non-None here due to filter
            if p.exit_time is not None:
                curve.append((p.exit_time, equity))

        return curve

    def trades_to_dataframe(self) -> pd.DataFrame:
        """
        Exportiert alle geschlossenen/teilgeschlossenen Trades als DataFrame.
        Spalten-Set ist auf Metrics/Signifikanzfunktionen abgestimmt.
        """
        rows: List[Dict[str, Any]] = []

        def _to_builtin(obj: Any) -> Any:
            """Konvertiert beliebige Objekte rekursiv in JSON-serialisierbare Builtins.
            Handhabt numpy/pandas Typen, NaT/NA, verschachtelte Strukturen.
            """
            import numpy as np  # lokale Importe, um Abhängigkeiten schlank zu halten
            import pandas as pd

            if obj is None:
                return None
            if isinstance(obj, (str, bool, int, float)):
                return obj
            # numpy scalar
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            # numpy array / Sequenzen
            if isinstance(obj, (np.ndarray, list, tuple)):
                try:
                    return [_to_builtin(v) for v in list(obj)]
                except Exception:
                    return [str(v) for v in list(obj)]
            # dicts
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    try:
                        out[str(k)] = _to_builtin(v)
                    except Exception:
                        out[str(k)] = str(v)
                return out
            # pandas Timestamp
            if isinstance(obj, pd.Timestamp):
                try:
                    return obj.tz_convert("UTC").isoformat()
                except Exception:
                    try:
                        return obj.tz_localize("UTC").isoformat()
                    except Exception:
                        return obj.isoformat()
            # pandas NA/NaT
            try:
                if pd.isna(obj):
                    return None
            except Exception:
                pass
            # Fallback: erst numerisch, dann string
            try:
                return float(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception:
                    return None

        def _row(p: PortfolioPosition) -> Dict[str, Any]:
            return {
                "entry_time": p.entry_time,
                "exit_time": p.exit_time,
                "direction": p.direction,
                "symbol": p.symbol,
                "entry_price": p.entry_price,
                "exit_price": p.exit_price,
                "initial_stop_loss": (
                    p.initial_stop_loss
                    if p.initial_stop_loss is not None
                    else p.stop_loss
                ),
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "size": p.size,
                "result": p.result,
                "entry_fee": getattr(p, "entry_fee", 0.0),
                "exit_fee": getattr(p, "exit_fee", 0.0),
                "total_fee": float(
                    getattr(p, "entry_fee", 0.0) + getattr(p, "exit_fee", 0.0)
                ),
                "reason": p.reason,
                "order_type": p.order_type,
                "status": p.status,
                "r_multiple": p.r_multiple,
                # Meta auf JSON-serialisierbare Builtins normalisieren
                "meta": _to_builtin(getattr(p, "metadata", None)),
            }

        for p in self.closed_positions:
            rows.append(_row(p))
        for p in self.partial_closed_positions:
            rows.append(_row(p))
        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "direction",
                    "symbol",
                    "entry_price",
                    "exit_price",
                    "initial_stop_loss",
                    "stop_loss",
                    "take_profit",
                    "size",
                    "result",
                    "entry_fee",
                    "exit_fee",
                    "total_fee",
                    "reason",
                    "order_type",
                    "status",
                    "r_multiple",
                    "meta",
                ]
            )
        )
