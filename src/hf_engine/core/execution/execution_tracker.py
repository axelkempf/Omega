from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from filelock import FileLock

from hf_engine.infra.config.paths import EXECUTION_TRACK_PATH
from hf_engine.infra.config.time_utils import now_utc
from hf_engine.infra.logging.log_service import log_service

logger = log_service.logger

# Pfade/Lock
TRACK_PATH: Path = Path(EXECUTION_TRACK_PATH)
LOCKFILE: Path = TRACK_PATH.with_suffix(TRACK_PATH.suffix + ".lock")

# Retention
MAX_TRACKING_DAYS = 5


def _utc_iso(dt: datetime) -> str:
    """
    Gibt einen RFC3339-kompatiblen UTC-String mit 'Z'-Suffix zurück.
    Erwartet entweder naive UTC (dann wird UTC gesetzt) oder aware.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # Python erzeugt "+00:00"; für Klarheit auf 'Z' normalisieren
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_parent_dir(p: Path) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            f"[ExecutionTracker] Verzeichnisanlage fehlgeschlagen: {p.parent} -> {e}"
        )
        raise


class ExecutionTracker:
    """
    Dateibasierter, gesperrter Tracker für Tages-Trades.
    Struktur im JSON:
    {
      "2025-08-14": {
         "EURUSD::MyStrat": {
             "status": "open"|"closed",
             "entry_time": "...Z",
             "exit_time": "...Z",
             "order_id": 12345,
             "direction": "long"|"short",
             ...
         }
      }
    }
    """

    def __init__(
        self,
        filepath: Union[str, Path] = TRACK_PATH,
        max_days: int = MAX_TRACKING_DAYS,
    ):
        self.filepath: Path = Path(filepath)
        self.lock = FileLock(str(LOCKFILE))
        self.max_days = int(max_days)

    # ---------- Low-level I/O ----------

    def _load_raw(self) -> Dict[str, Dict[str, Any]]:
        """
        Rohes Laden ohne Cutoff-Filter; niemals ohne Lock aufrufen.
        """
        if not self.filepath.exists():
            return {}

        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[ExecutionTracker] JSON korrupt ({self.filepath}): {e}")
            # Defensive: korruptes File nicht crashen lassen, aber melden
            return {}
        except Exception as e:
            logger.error(f"[ExecutionTracker] Fehler beim Laden ({self.filepath}): {e}")
            return {}

    def _apply_cutoff(
        self, data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Entfernt veraltete Tage gemäß max_days.
        """
        today = now_utc().date()
        cutoff = today - timedelta(days=self.max_days)
        cleaned: Dict[str, Dict[str, Any]] = {}
        for day_str, trades in data.items():
            try:
                day = datetime.fromisoformat(day_str).date()
                if day >= cutoff:
                    cleaned[day_str] = trades
            except Exception:
                # Ungültiger Schlüssel -> verwerfen und loggen
                logger.warning(
                    f"[ExecutionTracker] Ungültiges Datum im Key: {day_str!r} -> verworfen"
                )
        return cleaned

    def _load_clean(self) -> Dict[str, Dict[str, Any]]:
        """
        Laden + Cutoff anwenden; niemals ohne Lock aufrufen.
        """
        return self._apply_cutoff(self._load_raw())

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Atomares und möglichst durables Schreiben (tmp + replace + fsync).
        Niemals ohne Lock aufrufen.
        """
        try:
            _ensure_parent_dir(self.filepath)
            tmp_path = self.filepath.with_suffix(self.filepath.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.filepath)  # atomar
        except Exception as e:
            logger.error(
                f"[ExecutionTracker] Fehler beim Speichern ({self.filepath}): {e}"
            )

    def _mutate(self, update_fn: Callable[[Dict[str, Dict[str, Any]]], None]) -> None:
        """
        Einheitliche Mutations-Pipeline: Lock -> LoadClean -> Update -> Save.
        """
        with self.lock:
            data = self._load_clean()
            update_fn(data)
            self._save(data)

    # ---------- Public API ----------

    def get_day_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Liefert die Trade-Map für den gewünschten Tag (ISO-Keys).
        Beim Lesen wird ebenfalls bereinigt (Cutoff).
        """
        with self.lock:
            data = self._load_clean()
            day = (date or now_utc()).date().isoformat()
            return dict(data.get(day, {}))

    def mark_trade_open(
        self,
        symbol: str,
        strategy: Optional[str],
        order_id: int,
        **kwargs: Any,
    ) -> None:
        """
        Markiert/überschreibt den Trade-Key des Tages als 'open'.
        Zulässige optionale Felder: direction, entry_price, volume, sl, tp, risk, scenario
        """
        # Eingabevalidierung (leichtgewichtig, nicht übertreiben)
        if not isinstance(symbol, str) or not symbol.strip():
            logger.error("[ExecutionTracker] symbol ist leer/ungültig.")
            return
        if not isinstance(order_id, int):
            logger.error("[ExecutionTracker] order_id muss int sein.")
            return

        allowed_fields = {
            "direction",
            "entry_price",
            "volume",
            "sl",
            "tp",
            "risk",
            "scenario",
        }

        def update(data: Dict[str, Dict[str, Any]]) -> None:
            today = now_utc().date().isoformat()
            key = f"{symbol}::{strategy}" if strategy else symbol

            if today not in data:
                data[today] = {}

            trade_info: Dict[str, Any] = {
                "status": "open",
                "entry_time": _utc_iso(now_utc()),
                "order_id": order_id,
            }

            for field, value in kwargs.items():
                if field in allowed_fields and value is not None:
                    trade_info[field] = value

            data[today][key] = trade_info

        self._mutate(update)

    def mark_trade_closed(
        self,
        symbol: str,
        strategy: Optional[str],
        **kwargs: Any,
    ) -> None:
        """
        Setzt status auf 'closed', ergänzt exit_time und optionale Felder (exit_price, direction).
        """
        if not isinstance(symbol, str) or not symbol.strip():
            logger.error("[ExecutionTracker] symbol ist leer/ungültig.")
            return

        allowed_fields = {"exit_price", "direction", "exit_time"}

        def update(data: Dict[str, Dict[str, Any]]) -> None:
            today = now_utc().date().isoformat()
            key = f"{symbol}::{strategy}" if strategy else symbol
            target_day = None
            target_rec: Optional[Dict[str, Any]] = None

            if today in data and key in data[today]:
                target_day = today
                target_rec = data[today][key]
            else:
                # Suche rückwärts durch die letzten Tage (bereits durch _apply_cutoff begrenzt)
                for day_str in sorted(data.keys(), reverse=True):
                    if day_str == today:
                        continue
                    trades_for_day = data.get(day_str, {})
                    rec = trades_for_day.get(key)
                    if rec:
                        target_day = day_str
                        target_rec = rec
                        break

            if target_rec is not None:
                if target_day and target_day != today:
                    logger.info(
                        f"[ExecutionTracker] Fallback-Schließen für '{key}' auf Tag {target_day}."
                    )
                target_rec["status"] = "closed"

                # Falls exit_time übergeben wurde, nutze diese (UTC‑normalisiert),
                # sonst aktuelles now_utc().
                raw_exit = kwargs.get("exit_time")
                try:
                    if raw_exit is not None:
                        if isinstance(raw_exit, datetime):
                            target_rec["exit_time"] = _utc_iso(raw_exit)
                        elif isinstance(raw_exit, str):
                            # tolerant: eingehende ISO‑Strings (mit 'Z' oder Offset) normalisieren
                            s = raw_exit.replace("Z", "+00:00")
                            dt = datetime.fromisoformat(s)
                            target_rec["exit_time"] = _utc_iso(dt)
                        else:
                            target_rec["exit_time"] = _utc_iso(now_utc())
                    else:
                        target_rec["exit_time"] = _utc_iso(now_utc())
                except Exception:
                    target_rec["exit_time"] = _utc_iso(now_utc())

                for field, value in kwargs.items():
                    if (
                        field in allowed_fields
                        and field != "exit_time"
                        and value is not None
                    ):
                        target_rec[field] = value
            else:
                # Kein offener Eintrag gefunden – loggen, aber nicht fehlschlagen
                logger.warning(
                    f"[ExecutionTracker] Kein offener Trade für '{key}' am {today} gefunden."
                )

        self._mutate(update)

    def is_trade_open(self, symbol: str, strategy: Optional[str]) -> bool:
        with self.lock:
            data = self._load_clean()
            today = now_utc().date().isoformat()
            key = f"{symbol}::{strategy}" if strategy else symbol
            return (
                today in data
                and key in data[today]
                and data[today][key].get("status") == "open"
            )

    def get_executed_today(self) -> List[str]:
        with self.lock:
            data = self._load_clean()
            today = now_utc().date().isoformat()
            return list(data.get(today, {}).keys())

    def reset_day(self, day: Optional[datetime] = None) -> None:
        """
        Entfernt alle Einträge für den Zieltag.
        """

        def update(data: Dict[str, Dict[str, Any]]) -> None:
            target = (day or now_utc()).date().isoformat()
            if target in data:
                del data[target]

        self._mutate(update)
