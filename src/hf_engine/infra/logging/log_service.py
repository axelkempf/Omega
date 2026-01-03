# log_service.py
"""
LogService – robuster Logging-Wrapper mit SQLite (WAL), CSV/JSON-Archivierung und
thread-sicheren File/DB-Zugriffen. Pytest- und Mehrprozess-tauglich.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

from concurrent_log_handler import ConcurrentRotatingFileHandler

from hf_engine.infra.config import ARCHIVE_DIR, DB_PATH, SYSTEM_LOGS_DIR, TRADE_LOG_CSV
from hf_engine.infra.config.time_utils import now_utc

# ---------------------------------------------------------------------
# CSV-Schema – konsistent zu den geschriebenen Feldern halten!
# ---------------------------------------------------------------------
CSV_HEADERS: Tuple[str, ...] = (
    "datetime",
    "strategy",
    "symbol",
    "direction",
    "entry_price",
    "sl",
    "tp",
    "exit_price",
    "exit_reason",
    "profit_abs",
    "profit_pct",
    "duration_minutes",
    "duration_sec",
    "sl_hit",
    "tp_hit",
    "was_be_modified",  # bewusst so belassen, konsistent zum Schema
    "confidence_score",
    "entry_time",
    "exit_time",
    "pips",
    "commission",
    "swap",
    "entry_ticket",
    "exit_ticket",
    "position_id",
    "metadata",
)

EventType = Literal["system", "trade", "order", "risk", "error", "signal", "strategy"]


@dataclass(frozen=True)
class LogEvent:
    event_type: EventType
    message: str
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    payload: Optional[dict] = None
    level: str = "INFO"

    def to_dict(self) -> dict:
        return {
            "timestamp": now_utc().isoformat(),
            "type": self.event_type,
            "message": self.message,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "payload": self.payload or {},
            "level": self.level.upper(),
        }


class LogService:
    # DB‑Retry Parameter
    _DB_MAX_RETRIES = 5
    _DB_RETRY_BASE_SLEEP = 0.02  # Sekunden (exponentiell)

    def __init__(
        self,
        account_id: Optional[str] = None,
        use_sqlite: bool = True,
        console: Optional[bool] = None,
    ):
        self.account_id = account_id
        self.use_sqlite = use_sqlite
        self._db_lock = threading.Lock()
        self._csv_lock = threading.Lock()
        self._closed = False
        self.conn: Optional[sqlite3.Connection] = None

        # Verzeichnisse defensiv anlegen
        try:
            SYSTEM_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            TRADE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Logger
        logger_name = f"HFEngine.{account_id}" if account_id else "HFEngine"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Pytest-Kompabilität: propagate in Tests aktivieren
        in_pytest = "PYTEST_CURRENT_TEST" in os.environ
        self.logger.propagate = bool(in_pytest)

        # Console-Handler nur, wenn er explizit gewünscht ist oder wir nicht in Pytest laufen
        use_console = console if console is not None else not in_pytest

        if not self.logger.handlers:
            self._setup_handlers(use_console=use_console)

        if self.use_sqlite:
            self._init_db()

    # ------------------------- Logger Setup -------------------------

    def _setup_handlers(self, use_console: bool) -> None:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        if use_console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(logging.INFO)
            self.logger.addHandler(ch)

        log_filename = f"{self.account_id}.log" if self.account_id else "engine.log"
        file_path = SYSTEM_LOGS_DIR / log_filename
        fh = ConcurrentRotatingFileHandler(
            file_path,
            maxBytes=10_000_000,
            backupCount=7,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    # --------------------------- DB Setup --------------------------

    def _init_db(self) -> None:
        try:
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    strategy TEXT,
                    symbol TEXT,
                    payload TEXT,
                    level TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(timestamp);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_type ON logs(event_type);"
            )
            self.conn.commit()
        except Exception as e:
            # Fallback in Notbetrieb (nur File/Console)
            self.use_sqlite = False
            self.logger.error(f"[LogService] SQLite-Initialisierung deaktiviert: {e}")

    # ----------------------- Öffentliche API -----------------------

    @staticmethod
    def _fmt_num(val: Any, nd: int = 4) -> str:
        try:
            f = float(val)
            return f"{f:.{nd}f}"
        except Exception:
            return str(val)

    def _format_indicator_summary(self, meta: Optional[dict]) -> str:
        """Return concise indicator summary string for file logs."""
        try:
            if not isinstance(meta, dict):
                return ""
            ind = meta.get("indicators") or {}
            if not isinstance(ind, dict) or not ind:
                return ""
            keys = (
                ("ema", "ema"),
                ("atr", "atr"),
                ("zscore", "z"),
                ("kalman_z", "kz"),
                ("kalman_garch_z", "kgz"),
                ("rsi", "rsi"),
                ("macd", "macd"),
                ("signal", "sig"),
                ("bb_upper", "bbU"),
                ("bb_mid", "bbM"),
                ("bb_lower", "bbL"),
            )
            parts = []
            for k, label in keys:
                if k in ind and ind[k] is not None:
                    parts.append(f"{label}={self._fmt_num(ind[k])}")
            if not parts:
                return ""
            return " | ind: " + ", ".join(parts)
        except Exception:
            return ""

    def log_system(self, message: str, level: str = "INFO") -> None:
        self._log(LogEvent("system", message, level=level))

    def log_event(self, event: LogEvent) -> None:
        self._log(event)

    def log_trade(self, data: Dict[str, Any]) -> None:
        self.log_trade_csv(data)
        self.archive_trade_json(data)
        if self.use_sqlite:
            # Build concise message incl. indicator summary
            base_msg = (
                f"Trade abgeschlossen: {data.get('symbol')} @ {data.get('entry_price')}"
            )
            meta = data.get("metadata") if isinstance(data, dict) else None
            base_msg += self._format_indicator_summary(meta)
            self._log(
                LogEvent(
                    event_type="trade",
                    message=base_msg,
                    strategy=data.get("strategy"),
                    symbol=data.get("symbol"),
                    payload=data,
                    level="INFO",
                )
            )

    def log_exception(self, msg: str, exc: Exception) -> None:
        # schreibt Traceback über Logger, zusätzlicher Eintrag in DB via _log
        self.logger.exception(f"{msg}: {exc}")
        self._log(
            LogEvent(
                "error",
                f"{msg}: {exc}",
                level="ERROR",
                payload={"exception": repr(exc)},
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.conn is not None:
                with self._db_lock:
                    try:
                        self.conn.commit()
                    finally:
                        self.conn.close()
        except Exception:
            # Im Shutdown nicht abbrechen
            pass

    # Kontextmanager
    def __enter__(self) -> "LogService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------- Interne Hilfsfunktionen ----------------------

    def _log(self, event: LogEvent) -> None:
        # File/Console
        line = f"[{event.event_type.upper()}] {event.message}"
        if event.strategy:
            line += f" | Strategy: {event.strategy}"
        if event.symbol:
            line += f" | Symbol: {event.symbol}"

        try:
            lvl = getattr(logging, (event.level or "INFO").upper(), logging.INFO)
            self.logger.log(lvl, line)
        except Exception as e:
            # Hard fallback: stdout
            print(f"[LogService] Logger-Fehler: {e} | {line}")

        # DB
        if self.use_sqlite and self.conn is not None:
            payload_str = json.dumps((event.payload or {}), default=str)
            self._db_execute_with_retry(
                """
                INSERT INTO logs (timestamp, event_type, message, strategy, symbol, payload, level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc().isoformat(),
                    event.event_type,
                    event.message,
                    event.strategy,
                    event.symbol,
                    payload_str,
                    (event.level or "INFO").upper(),
                ),
            )

    def _db_execute_with_retry(self, sql: str, params: Iterable[Any]) -> None:
        if not self.conn:
            return
        attempt = 0
        while True:
            try:
                with self._db_lock:
                    self.conn.execute(sql, tuple(params))
                    self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                # typ. "database is locked" – exponentielles Backoff
                attempt += 1
                if attempt > self._DB_MAX_RETRIES:
                    self.logger.error(
                        f"[LogService] SQLite-Insert endgültig fehlgeschlagen nach {attempt} Versuchen: {e}"
                    )
                    return
                sleep_s = self._DB_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                time.sleep(min(sleep_s, 0.5))
            except Exception as e:
                self.logger.error(f"[LogService] SQLite-Fehler: {e}")
                return

    # ------------------------- CSV / JSON Trades -------------------------

    def log_trade_csv(self, data: Dict[str, Any]) -> None:
        """
        Schreibt Tradezeile robust und threadsicher (Lock) in CSV.
        Nutzt CSV_HEADERS als Quelle der Spaltenreihenfolge.
        Unbekannte Keys werden ignoriert, fehlende mit Default befüllt.
        """
        # Pflichtfelder minimal validieren – keine Exception, damit Logging nie blockiert
        symbol = data.get("symbol")
        strategy = data.get("strategy")
        if not symbol:
            self.logger.warning("[LogService] Trade-CSV ohne 'symbol'.")
        if not strategy:
            self.logger.warning("[LogService] Trade-CSV ohne 'strategy'.")

        # Defaults pro Spalte
        defaults = {
            "datetime": now_utc().isoformat(),
            "profit_abs": "",
            "profit_pct": "",
            "duration_minutes": "",
            "duration_sec": "",
            "sl_hit": 0,
            "tp_hit": 0,
            "was_be_modified": False,
            "metadata": {},
        }

        # Reihenfolge strikt aus CSV_HEADERS
        row = []
        for col in CSV_HEADERS:
            val = data.get(col, defaults.get(col, ""))
            if col == "metadata":
                val = json.dumps(val, default=str)
            row.append(val)

        # Schreiben mit Lock
        try:
            is_new = not TRADE_LOG_CSV.exists()
            with self._csv_lock:
                # newline='' für sauberes CSV unter Windows
                with TRADE_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if is_new:
                        writer.writerow(CSV_HEADERS)
                    writer.writerow(row)
        except Exception as e:
            self.logger.error(f"[LogService] Fehler beim Schreiben in CSV: {e}")

    def archive_trade_json(self, data: Dict[str, Any]) -> None:
        try:
            trade_id = uuid.uuid4().hex[:12]
            symbol = (data.get("symbol") or "unknown").replace("/", "_")
            file_path = ARCHIVE_DIR / f"{symbol}_{trade_id}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"[LogService] Fehler beim Archivieren: {e}")


# Optionale globale Instanz (bewusst spät erstellt, damit Module-Importe robust sind)
log_service = LogService()
