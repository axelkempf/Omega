# hf_engine/core/risk/risk_manager.py
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from enum import Enum

import hf_engine.core.risk.news_filter as news_filter
from hf_engine.adapter.broker.broker_interface import BrokerInterface
from hf_engine.core.execution.sl_tp_utils import ensure_abs_levels
from hf_engine.infra.config import TRADE_LOG_CSV
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.error_handler import safe_execute
from hf_engine.infra.logging.log_service import log_service


class RiskStatus(str, Enum):
    OK = "OK"
    EQUITY_MIN_VIOLATION = "EQUITY_MIN_VIOLATION"
    EQUITY_READ_ERROR = "EQUITY_READ_ERROR"
    TRADE_LIMIT_REACHED = "TRADE_LIMIT_REACHED"
    TRADECOUNT_READ_ERROR = "TRADECOUNT_READ_ERROR"
    NEWS_BLOCKED = "NEWS_BLOCKED"
    NEWS_LOOKUP_ERROR = "NEWS_LOOKUP_ERROR"
    DUPLICATE_DIRECTION = "DUPLICATE_DIRECTION"
    DUPLICATE_CHECK_ERROR = "DUPLICATE_CHECK_ERROR"
    SLTP_INVALID = "SLTP_INVALID"


def _parse_log_dt(dt_raw: str) -> datetime | None:
    """Parse ISO datetime that might be naive; assume UTC if no tzinfo."""
    try:
        dt = datetime.fromisoformat(dt_raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _TradeCountCache:
    """
    Lightweight cache for 'trades today' per (symbol, strategy).
    Reloads CSV only if file mtime changed. Thread-safe.
    """

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._lock = threading.Lock()
        self._last_mtime: float | None = None
        self._today = now_utc().date()
        self._counts: dict[tuple[str, str], int] = {}

    def _maybe_reload(self) -> None:
        if not os.path.exists(self._log_path):
            with self._lock:
                self._counts.clear()
                self._last_mtime = None
                self._today = now_utc().date()
            return

        try:
            mtime = os.path.getmtime(self._log_path)
        except Exception:
            # If we cannot stat the file, keep current cache.
            return

        today = now_utc().date()
        # Reload if file changed or date rolled over.
        if self._last_mtime == mtime and self._today == today:
            return

        counts: dict[tuple[str, str], int] = {}
        try:
            with open(self._log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dt_raw = row.get("datetime")
                    if not dt_raw:
                        continue
                    dt = _parse_log_dt(dt_raw)
                    if not dt or dt.date() != today:
                        continue
                    key = (row.get("symbol", ""), row.get("strategy", ""))
                    counts[key] = counts.get(key, 0) + 1
        except Exception:
            # On read/parse issues keep previous cache; caller handles fallback.
            return

        with self._lock:
            self._counts = counts
            self._last_mtime = mtime
            self._today = today

    def count_today(self, symbol: str, strategy: str) -> int:
        self._maybe_reload()
        with self._lock:
            return self._counts.get((symbol, strategy), 0)


class RiskManager:
    def __init__(self, broker: BrokerInterface, log_path: str = TRADE_LOG_CSV):
        self.broker = broker
        self.log_path = log_path
        self._trade_cache = _TradeCountCache(log_path)

    # --- Guards -------------------------------------------------------------

    def _equity_check(self, setup) -> tuple[bool, RiskStatus]:
        current_equity = self.broker.get_account_equity()
        max_dd_pct = float(setup.metadata.get("max_drawdown_pct", 20.0))
        min_equity = setup.start_capital * (1.0 - max_dd_pct / 100.0)
        if current_equity < min_equity:
            return False, RiskStatus.EQUITY_MIN_VIOLATION
        return True, RiskStatus.OK

    def _trade_count_check(self, setup) -> tuple[bool, RiskStatus]:
        max_trades = int(setup.metadata.get("max_trades_per_day", 100))
        count = self._trade_cache.count_today(setup.symbol, setup.strategy)
        if count >= max_trades:
            return False, RiskStatus.TRADE_LIMIT_REACHED
        return True, RiskStatus.OK

    def _news_block_check(self, setup, now_dt: datetime) -> tuple[bool, RiskStatus]:
        min_impact = str(setup.metadata.get("min_news_impact", "high")).lower()
        blocked = news_filter.is_news_nearby(
            setup.symbol, now_dt, min_impact=min_impact
        )
        return (not blocked, RiskStatus.NEWS_BLOCKED if blocked else RiskStatus.OK)

    def _duplicate_direction_check(self, setup) -> tuple[bool, RiskStatus]:
        open_positions = self.broker.get_own_position(
            setup.symbol, magic_number=getattr(setup, "magic_number", None)
        )
        direction = (getattr(setup, "direction", "") or "").lower()
        same_dir = [
            p for p in open_positions if self.broker.position_direction(p) == direction
        ]
        if same_dir:
            return False, RiskStatus.DUPLICATE_DIRECTION
        return True, RiskStatus.OK

    def _sl_tp_validation(self, setup) -> tuple[bool, RiskStatus]:
        # Raises on invalid; we only validate that it does not throw.
        ensure_abs_levels(setup.entry, setup.sl, setup.tp, setup.direction)
        return True, RiskStatus.OK

    # --- Public: zentrale Pre-Trade-Validierung ----------------------------

    def validate_pre_trade_conditions(
        self, setup, now_dt: datetime | None = None
    ) -> tuple[bool, str]:
        """
        Führt alle Risk-Guards aus. Liefert (ok, code_str).
        - Nutzt safe_execute je Check; bei Fehlern wird ein deterministischer Fallback-Code geliefert.
        - Rückgabecodes bleiben Strings für Abwärtskompatibilität.
        """
        now_dt = now_dt or now_utc()

        checks: list[tuple[str, callable, RiskStatus]] = [
            ("EquityCheck", self._equity_check, RiskStatus.EQUITY_READ_ERROR),
            (
                "TradeCountCheck",
                self._trade_count_check,
                RiskStatus.TRADECOUNT_READ_ERROR,
            ),
            (
                "NewsCheck",
                lambda s: self._news_block_check(s, now_dt),
                RiskStatus.NEWS_LOOKUP_ERROR,
            ),
            (
                "DuplicateDirectionCheck",
                self._duplicate_direction_check,
                RiskStatus.DUPLICATE_CHECK_ERROR,
            ),
            ("SLTPValidation", self._sl_tp_validation, RiskStatus.SLTP_INVALID),
        ]

        for task_name, fn, fallback in checks:
            result = safe_execute(task_name, fn, setup)
            if result is None:
                self._log_block(task_name, fallback.value, setup)
                return False, fallback.value

            ok, code = result
            code_str = code.value if isinstance(code, RiskStatus) else str(code)
            if not ok:
                self._log_block(task_name, code_str, setup)
                return False, code_str

        return True, RiskStatus.OK.value

    # --- Internal -----------------------------------------------------------

    def _log_block(self, check_name: str, code: str, setup) -> None:
        """
        Best-Effort Logging ohne harte Abhängigkeit von der Log-API.
        """
        msg = (
            f"[Risk] Blocked by {check_name}: {code} | "
            f"symbol={getattr(setup, 'symbol', '?')}, "
            f"strategy={getattr(setup, 'strategy', '?')}"
        )
        try:
            # Bevorzugte API
            if hasattr(log_service, "warning"):
                log_service.warning(msg)
            elif hasattr(log_service, "warn"):
                log_service.warn(msg)
            elif hasattr(log_service, "info"):
                log_service.info(msg)
        except Exception:
            # Never let logging break trading flow
            pass
