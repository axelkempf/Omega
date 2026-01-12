# hf_engine/strategies/mean_reversion_z_score/live/strategy.py
"""Mean Reversion Z-Score live trading strategy implementation."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from typing import Any, override

import hf_engine.core.risk.news_filter as news_filter
from hf_engine.adapter.broker.broker_interface import BrokerInterface
from hf_engine.adapter.broker.broker_utils import get_pip_size
from hf_engine.adapter.data.mt5_data_provider import MT5DataProvider
from hf_engine.core.execution.execution_tracker import ExecutionTracker
from hf_engine.infra.config.environment import TIMEZONE
from hf_engine.infra.config.time_utils import now_utc
from strategies._base.base_strategy import Strategy, TradeSetup
from strategies.mean_reversion_z_score.live.scenarios import SzenarioEvaluator
from strategies.mean_reversion_z_score.live.utils import (
    get_next_entry_after_exit,
    get_next_entry_time,
    in_session_utc,
)


def _decimals_from_tick(symbol: str, broker: BrokerInterface) -> int:
    """
    Derive the number of decimal places from tick size or digits.

    Args:
        symbol: Trading symbol.
        broker: Broker interface.

    Returns:
        Number of decimal places for price rounding.
    """
    info = broker.get_symbol_info(symbol) or {}
    ts = float(info.get("tick_size") or 0.0)
    if ts <= 0:
        digits = int(info.get("digits") or 5)
        return digits
    # Anzahl Dezimalstellen aus tick_size ableiten
    s = f"{ts:.10f}".rstrip("0").split(".")
    return len(s[1]) if len(s) == 2 else 0


def _tf_delta(tf: str | None) -> timedelta:
    """
    Map timeframe string to corresponding timedelta.

    Args:
        tf: Timeframe string (M1, M5, H1, etc.).

    Returns:
        Corresponding timedelta, or zero if unknown.
    """
    tf_u = (tf or "").upper()
    mapping: dict[str, timedelta] = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
        "W1": timedelta(weeks=1),
    }
    return mapping.get(tf_u, timedelta(0))


class MeanReversionZScoreStrategy(Strategy):
    """Mean Reversion Z-Score trading strategy using scenario-based evaluation."""

    def __init__(self, config_module_name: str) -> None:
        """
        Initialize the strategy with configuration from a module.

        Args:
            config_module_name: Name of the configuration module to load.
        """
        # Modul dynamisch laden (beibehaltener API-Kontrakt)
        config_module = importlib.import_module(
            f"hf_engine.strategies.mean_reversion_z_score.live.{config_module_name}"
        )
        self.config: dict[str, Any] = config_module.CONFIG

        self.magic_number: int | None = self.config.get("magic_number")

        self.timeframe: str | None = self.config.get("timeframe")

        self.cooldown_minutes: int | None = self.config.get("cooldown_minutes")

        self.szenarien: SzenarioEvaluator | None = None

    @override
    def name(self) -> str:
        """Return the strategy name."""
        return self.config.get("strategy_name", "Mean_Reversion_Z_Score")

    def session_times(self) -> dict[str, Any]:
        """Return session time configuration."""
        return self.config.get("session", {})

    def is_trade_day(self, date: datetime) -> bool:
        """
        Check if the given date is a valid trading day.

        Args:
            date: Date to check.

        Returns:
            True if trading is allowed on this day.
        """
        # Krypto-Setups: immer handeln (7 Tage)
        if self.config.get("asset_class") == "crypto":
            return True
        # Standard: FX/CFD nur Mo–Fr
        return date.astimezone(timezone.utc).weekday() in range(0, 5)

    @override
    def generate_signal(
        self,
        symbol: str,
        date: datetime,
        broker: BrokerInterface,
        data_provider: MT5DataProvider,
    ) -> list[TradeSetup]:
        """
        Generate trading signals for a symbol at the given time.

        Args:
            symbol: Trading symbol to analyze.
            date: Current bar timestamp.
            broker: Broker interface for price and position queries.
            data_provider: Market data provider.

        Returns:
            List of TradeSetup objects, empty if no signal.
        """
        # Sessionfenster (auf Bar‑Close shiften)
        sess = self.session_times() or {}
        start = sess.get("session_start")
        end = sess.get("session_end")
        if start and end:
            gate_time = date + _tf_delta(self.timeframe)
            if not in_session_utc(gate_time, start, end):
                return []
        # Handelstage
        if not self.is_trade_day(date):
            return []

        # News-Filter (auf Bar‑Close shiften)
        gate_time = date + _tf_delta(self.timeframe)
        if news_filter.is_news_nearby(symbol, gate_time):
            print(f"News nearby für: {symbol}")
            return []

        # Positionsgate (nur 1 Position pro Symbol)
        open_positions = broker.get_own_position(symbol, magic_number=self.magic_number)
        if open_positions and len(open_positions) >= 1:
            return []

        tracker = ExecutionTracker()

        # Szenarien-Evaluator einmalig initialisieren
        if self.szenarien is None:
            self.szenarien = SzenarioEvaluator(self.config, data_provider)

        # Signal evaluieren (inkl. defensiver Fehlerbehandlung in Szenario)
        evaluate = self.szenarien.evaluate_all(symbol, self.timeframe)
        if not evaluate:
            return []

        # Szenario-Whitelist respektieren, falls konfiguriert
        try:
            allowed = set(self.config.get("allowed_scenarios") or [])
        except Exception:
            allowed = set()
        if allowed:
            scen = str(evaluate.get("scenario") or "")
            if scen not in allowed:
                return []

        direction = evaluate["direction"]
        # Mappe Overrides: CONFIG nutzt "buy"/"sell", Evaluator/TradeSetup oft "long"/"short"
        _dir_map = {"long": "buy", "short": "sell", "buy": "buy", "sell": "sell"}
        dir_for_overrides = _dir_map.get(str(direction).lower(), str(direction).lower())

        # ---- Composite-Tracker-Key: Strategie | MagicNumber | TF | Richtung | Szenario ----
        # MagicNumber dient als kompakter Setup-Identifier, um unterschiedliche
        # Parameterkombinationen eindeutig zu trennen.
        setup_key_suffix = f"{self.name()}|MN{self.magic_number}|{self.timeframe}|{dir_for_overrides}|{evaluate.get('scenario')}"
        key = f"{symbol}::{setup_key_suffix}"
        debug_prefix = f"[LIVE][DEBUG][ExecutionTracker][{key}]"

        # ---- Cooldown-Berechnung (Exit-/Entry-basiert) für diesen Composite-Key ----
        now = now_utc().astimezone(TIMEZONE)
        next_allowed_dt = None
        # Für Entry-Cooldown-Debug sammeln wir Kandidat & letzte Entry-Zeit
        entry_cand = None
        last_entry_time_tz = None

        # Jüngster geschlossener Trade (unabhängig von Richtung, aber innerhalb des Composite-Keys)
        last_closed_trade = _find_recent_trade_last_exit(
            tracker,
            key,
            date,
            lookback_days=2,
            debug_label=debug_prefix,
        )
        if last_closed_trade:
            exit_time = _parse_to_tz(last_closed_trade.get("exit_time"), TIMEZONE)
            if exit_time:
                enforced_wait = get_next_entry_after_exit(exit_time, self.timeframe)
                next_allowed_dt = (
                    enforced_wait
                    if next_allowed_dt is None
                    else max(next_allowed_dt, enforced_wait)
                )
                print(
                    f"{debug_prefix} last exit at {exit_time.isoformat()} -> cooldown until {enforced_wait.isoformat()}"
                )
                print(f"{debug_prefix} exit cooldown uses now={now.isoformat()})")

        # ---- NEU: cooldown/max_hold/exit-skip je Symbol×TF×Richtung auflösen ----
        cooldown_minutes = _resolve_override(
            self.config,
            symbol,
            self.timeframe,
            dir_for_overrides,
            "cooldown_minutes",
            default=self.config.get("cooldown_minutes", 0),
        )
        max_holding_minutes = _resolve_override(
            self.config,
            symbol,
            self.timeframe,
            dir_for_overrides,
            "max_holding_minutes",
            default=self.config.get("max_holding_minutes", 0),
        )

        # ---- Robustes Gating: letzter Trade gleicher Richtung über mehrere Tage ----
        last_tr = _find_recent_trade_same_dir(
            tracker, key, date, dir_for_overrides, lookback_days=3
        )

        if last_tr:
            # 1) Entry-basierter Cooldown (Minuten)
            entry_time = _parse_to_tz(last_tr.get("entry_time"), TIMEZONE)
            if entry_time and cooldown_minutes and cooldown_minutes > 0:
                cand = get_next_entry_time(entry_time, self.timeframe, cooldown_minutes)
                entry_cand = cand
                last_entry_time_tz = entry_time
                next_allowed_dt = (
                    cand if next_allowed_dt is None else max(next_allowed_dt, cand)
                )

        if next_allowed_dt and now < next_allowed_dt:
            print(
                f"{debug_prefix} cooldown active. now={now.isoformat()} next_allowed={next_allowed_dt.isoformat()}"
            )
            # Minimaler Debug: Wenn Entry-Cooldown der Blocker ist, präzisieren
            if entry_cand and next_allowed_dt == entry_cand and now < entry_cand:
                try:
                    last_entry_iso = (
                        last_entry_time_tz.isoformat()
                        if last_entry_time_tz
                        else "unknown"
                    )
                    print(
                        f"{debug_prefix} entry cooldown blocking: last_entry={last_entry_iso} -> until {entry_cand.isoformat()} (now={now.isoformat()})"
                    )
                except Exception:
                    print(
                        f"{debug_prefix} entry cooldown blocking until {entry_cand} (now={now})"
                    )
            return []

        # Preispräzision aus pip_size
        decimals = _decimals_from_tick(symbol, broker)

        sl = float(evaluate["sl"])
        tp = float(evaluate["tp"])
        entry = broker.get_symbol_price(symbol, direction)
        order_type = evaluate.get("order_type", self.config.get("order_type", "market"))

        setup = TradeSetup(
            symbol=symbol,
            direction=direction,
            entry=round(entry, decimals),
            sl=round(sl, decimals),
            tp=round(tp, decimals),
            strategy=self.name(),
            strategy_module="mean_reversion_z_score.live",
            start_capital=self.config["risk"]["start_capital"],
            risk_pct=self.config["risk"]["risk_per_trade_pct"],
            order_type=order_type,
            session_times=self.config.get("session", {}),
            magic_number=self.magic_number,
            metadata={
                "scenario": evaluate.get("scenario"),
                "timestamp": date,
                "timeframe": self.timeframe,
                # Der Composite-Tracker-Key (Suffix) für den ExecutionTracker.
                # ExecutionEngine nutzt metadata['setup'] als 'strategy'-Teil des Keys.
                "setup": setup_key_suffix,
                # Ausführungs-Parameter (unverändert beibehalten)
                "max_holding_minutes": max_holding_minutes,
                "min_sl_pips": 1.0,
                "max_deviation_pips": 3.0,  # nur Market
                "volatility_buffer_pips": 0.0,  # nur Market (optional)
                "max_spread_pips": 2.0,  # nur Market (falls Broker-Spread verfügbar)
                "min_pending_distance_pips": 2.0,  # nur Pending
                "allow_marketable_pending": False,  # Pending->Market Konvertierung
                # Indicator snapshot (last values) for logging/analysis
                "indicators": evaluate.get("indicators", {}),
            },
        )
        return [setup]


# --------- Helper ---------
def _parse_to_tz(dt_val: datetime | str | None, tz: timezone) -> datetime | None:
    """
    Parse datetime value and convert to specified timezone.

    Args:
        dt_val: Datetime object, ISO string, or None.
        tz: Target timezone.

    Returns:
        Timezone-aware datetime or None if parsing fails.
    """
    if isinstance(dt_val, str):
        try:
            s = dt_val.replace("Z", "+00:00")
            return datetime.fromisoformat(s).astimezone(tz)
        except Exception:
            return None
    if isinstance(dt_val, datetime):
        try:
            return dt_val.astimezone(tz)
        except Exception:
            return None
    return None


def _find_recent_trade_last_exit(
    tracker: ExecutionTracker,
    key: str,
    date: datetime,
    lookback_days: int = 2,
    debug_label: str | None = None,
) -> dict[str, Any] | None:
    """
    Find the most recent closed trade (direction-agnostic) with exit_time.

    Args:
        tracker: ExecutionTracker instance.
        key: Composite tracker key.
        date: Current date for lookback.
        lookback_days: Number of days to look back.
        debug_label: Optional debug prefix for logging.

    Returns:
        Trade record dict or None if not found.
    """
    best: dict[str, Any] | None = None
    best_ts: datetime | None = None

    for d in range(max(1, int(lookback_days))):
        day_dt = date - timedelta(days=d)
        day_iso = day_dt.date().isoformat()
        day_map = tracker.get_day_data(date=day_dt) or {}
        if debug_label:
            print(
                f"{debug_label} read day {day_iso}: available_keys={list(day_map.keys())}"
            )
        tr = day_map.get(key)
        if not tr:
            continue
        status = str(tr.get("status") or "").lower().strip()
        if debug_label:
            print(
                f"{debug_label} record for {day_iso}: status={status} exit_time={tr.get('exit_time')} raw={tr}"
            )
        if status != "closed":
            continue
        exit_ts = _parse_to_tz(tr.get("exit_time"), TIMEZONE)
        if exit_ts and (best_ts is None or exit_ts > best_ts):
            best_ts, best = exit_ts, tr

    return best


def _find_recent_trade_same_dir(
    tracker: ExecutionTracker,
    key: str,
    date: datetime,
    desired_dir: str,
    lookback_days: int = 3,
) -> dict[str, Any] | None:
    """
    Find the most recent trade in the same direction within lookback period.

    Args:
        tracker: ExecutionTracker instance.
        key: Composite tracker key.
        date: Current date for lookback.
        desired_dir: Desired direction ('buy' or 'sell').
        lookback_days: Number of days to look back.

    Returns:
        Trade record dict or None if not found.
    """
    _dir_map: dict[str, str] = {
        "long": "buy",
        "short": "sell",
        "buy": "buy",
        "sell": "sell",
    }
    best: dict[str, Any] | None = None
    best_ts: datetime | None = None

    for d in range(max(1, int(lookback_days))):
        day_map = tracker.get_day_data(date=date - timedelta(days=d)) or {}
        tr = day_map.get(key)
        if not tr:
            continue
        tdir_raw = tr.get("direction")
        if tdir_raw:
            tdir = _dir_map.get(str(tdir_raw).lower(), str(tdir_raw).lower())
            if tdir != desired_dir:
                continue  # andere Richtung -> ignorieren
        cand_ts = _parse_to_tz(tr.get("exit_time"), TIMEZONE) or _parse_to_tz(
            tr.get("entry_time"), TIMEZONE
        )
        if cand_ts and (best_ts is None or cand_ts > best_ts):
            best_ts, best = cand_ts, tr

    return best


def _resolve_override(
    config: dict[str, Any],
    symbol: str,
    timeframe: str,
    direction: str,
    key: str,
    default: Any = None,
) -> Any:
    """
    Get a value from CONFIG['param_overrides'] with wildcard support.

    Checks in order: (symbol, timeframe) > (symbol, *) > (*, timeframe) > (*, *)
    within the direction node ('long'/'short').

    Args:
        config: Strategy configuration dict.
        symbol: Trading symbol.
        timeframe: Timeframe string.
        direction: Direction ('long' or 'short').
        key: Parameter key to look up.
        default: Default value if not found.

    Returns:
        Override value or default.
    """
    po: dict[str, Any] = (config or {}).get("param_overrides", {})
    for sym, tf in ((symbol, timeframe), (symbol, "*"), ("*", timeframe), ("*", "*")):
        node = po.get(sym, {}).get(tf, {})
        val = node.get(direction, {}).get(key, None)
        if val is not None:
            return val
    return default
