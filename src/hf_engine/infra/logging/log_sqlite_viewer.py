# hf_engine/infra/logging/jsonl_logger.py
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _ts_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40}


class JsonlRunLogger:
    """
    Schlanker, robuster JSONL-Logger mit Run-ID.

    Features:
    - Thread-sichere Writes (Lock)
    - Optional: Dateirotation via rotate_max_bytes/rotate_backup_count
    - Resilienz: Fallback auf Temp-Verzeichnis, wenn root nicht beschreibbar
    - Konfigurierbare Mindest-Logstufe (min_level)
    - Context-Manager-Support + explizite close()
    - Flush nach jedem Write für minimierten Datenverlust
    - Append-only Audit-Streams (separat pro Name)

    Pfade:
      <root>/datenspeicher/logging/<run_id>/<component>.jsonl
      <root>/datenspeicher/audit/<run_id>/*.jsonl
    """

    def __init__(
        self,
        run_id: Optional[str],
        component: str,
        root: Optional[Path] = None,
        *,
        min_level: str = "INFO",
        rotate_max_bytes: Optional[int] = None,
        rotate_backup_count: int = 3,
        console: bool = True,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.component = component
        self.console = console

        # Log-Level-Konfiguration
        self._min_level_name = min_level.upper()
        self._min_level = _LEVELS.get(self._min_level_name, 20)

        # Rotation
        self._rotate_max_bytes = rotate_max_bytes
        self._rotate_backup_count = max(0, int(rotate_backup_count))

        # Lock für Thread-Sicherheit
        self._lock = threading.Lock()

        # Root bestimmen & sicherstellen
        env_root = os.environ.get("HF_ENGINE_ROOT")
        root = Path(env_root).resolve() if env_root else (root or Path(".").resolve())
        self.root = self._resolve_writable_root(root)

        self.logs_dir = self._ensure_dir(
            self.root / "datenspeicher" / "logging" / self.run_id
        )
        self.audit_dir = self._ensure_dir(
            self.root / "datenspeicher" / "audit" / self.run_id
        )

        # Hauptlog-Datei öffnen
        self._log_file = self.logs_dir / f"{self.component}.jsonl"
        self._log_fp = self._open_for_append(self._log_file)

    # -------------------------
    # Öffentliche API
    # -------------------------
    def write(self, level: str, message: str, **extra: Any) -> None:
        level = level.upper()
        if not self._enabled(level):
            return

        record = {
            "ts_utc": _ts_iso(),
            "run_id": self.run_id,
            "component": self.component,
            "level": level,
            "msg": message,
            **extra,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        with self._lock:
            # Rotation (nur für Hauptlog)
            self._maybe_rotate_locked()
            self._safe_write_line(self._log_fp, line, self._log_file)

        if self.console:
            # knappe Konsole – robust gegen Encoding-Probleme
            try:
                print(f"[{record['ts_utc']}] {self.component} {level}: {message}")
            except Exception:
                # Fallback auf stderr bei exotischen Encoding-Problemen
                try:
                    sys.stderr.write(
                        f"[{record['ts_utc']}] {self.component} {level}: {message}\n"
                    )
                except Exception:
                    pass  # Als letzte Sicherheit nichts crashen

    def debug(self, message: str, **extra: Any) -> None:
        self.write("DEBUG", message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        self.write("INFO", message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        self.write("WARN", message, **extra)

    warn = warning  # alias

    def error(self, message: str, **extra: Any) -> None:
        self.write("ERROR", message, **extra)

    def audit_append(self, name: str, record: Dict[str, Any]) -> None:
        """
        Append-only Audit-Stream (z. B. orders, fills, state transitions).
        Rotation ist hier bewusst NICHT aktiv, da Audit-Streams
        i. d. R. vollständig und unverändert archiviert werden sollen.
        """
        if "ts_utc" not in record:
            record["ts_utc"] = _ts_iso()
        if "run_id" not in record:
            record["run_id"] = self.run_id

        line = json.dumps(record, ensure_ascii=False) + "\n"
        fp_path = self.audit_dir / f"{name}.jsonl"
        with self._lock:
            # Audit-Stream beim Append jeweils kurz öffnen/schließen
            try:
                with fp_path.open("a", encoding="utf-8") as fp:
                    self._safe_write_line(fp, line, fp_path, do_flush=False)
            except OSError:
                # Als Fallback Audit im Temp ablegen
                fallback = self._fallback_audit_path(name)
                self._ensure_dir(fallback.parent)
                with fallback.open("a", encoding="utf-8") as fp:
                    self._safe_write_line(fp, line, fallback, do_flush=False)

    def set_min_level(self, level: str) -> None:
        level = level.upper()
        self._min_level_name = level
        self._min_level = _LEVELS.get(level, self._min_level)

    def close(self) -> None:
        with self._lock:
            try:
                if getattr(self, "_log_fp", None):
                    self._log_fp.flush()
                    self._log_fp.close()
            except Exception:
                pass
            finally:
                self._log_fp = None

    # Context-Manager
    def __enter__(self) -> "JsonlRunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # defensiv – darf keinen Fehler beim GC werfen
        try:
            self.close()
        except Exception:
            pass

    # -------------------------
    # Interne Helfer
    # -------------------------
    def _enabled(self, level: str) -> bool:
        return _LEVELS.get(level, 0) >= self._min_level

    @staticmethod
    def _ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _open_for_append(self, path: Path):
        # eigenes Open, um an einer Stelle Fehler zu fangen
        try:
            return path.open("a", encoding="utf-8")
        except OSError:
            # Fallback auf Temp-Verzeichnis
            fallback = self._fallback_log_path()
            self._ensure_dir(fallback.parent)
            return fallback.open("a", encoding="utf-8")

    def _safe_write_line(
        self, fp, line: str, path: Path, *, do_flush: bool = True
    ) -> None:
        try:
            fp.write(line)
            if do_flush:
                fp.flush()
        except OSError:
            # Letzter Rettungsanker: in Fallback schreiben
            try:
                fallback = self._fallback_log_path()
                self._ensure_dir(fallback.parent)
                with fallback.open("a", encoding="utf-8") as f2:
                    f2.write(line)
                    if do_flush:
                        f2.flush()
            except Exception:
                # Nichts mehr tun – Logging darf nie die App killen
                pass

    def _maybe_rotate_locked(self) -> None:
        """Nur im Lock-Kontext aufrufen!"""
        if not self._rotate_max_bytes or self._rotate_backup_count <= 0:
            return
        try:
            size = self._log_file.stat().st_size if self._log_file.exists() else 0
            if size < self._rotate_max_bytes:
                return
        except OSError:
            return

        # Rotieren: close -> shift backups -> reopen
        try:
            self._log_fp.flush()
            self._log_fp.close()
        except Exception:
            pass

        try:
            # Alte Backups schieben: .N -> .N+1 (rückwärts, um Überschreiben zu vermeiden)
            for i in range(self._rotate_backup_count - 1, 0, -1):
                src = self._log_file.with_suffix(self._log_file.suffix + f".{i}")
                dst = self._log_file.with_suffix(self._log_file.suffix + f".{i+1}")
                if src.exists():
                    try:
                        if dst.exists():
                            dst.unlink(missing_ok=True)
                    except TypeError:
                        # Python < 3.8 Kompatibilität: missing_ok nicht verfügbar
                        if dst.exists():
                            dst.unlink()
                    try:
                        src.replace(dst)
                    except Exception:
                        # Cross-filesystem Fallback
                        shutil.move(str(src), str(dst))

            # Aktuelle Datei -> .1
            first_backup = self._log_file.with_suffix(self._log_file.suffix + ".1")
            try:
                if first_backup.exists():
                    first_backup.unlink()
            except Exception:
                pass
            try:
                self._log_file.replace(first_backup)
            except Exception:
                shutil.move(str(self._log_file), str(first_backup))
        finally:
            # Neue Datei öffnen
            self._log_fp = self._open_for_append(self._log_file)

    def _resolve_writable_root(self, preferred: Path) -> Path:
        """Prüft, ob preferred beschreibbar ist; sonst auf Temp-Fallback wechseln."""
        try:
            self._ensure_dir(preferred)
            testfile = preferred / ".write_test"
            with testfile.open("w", encoding="utf-8") as f:
                f.write("ok")
            (
                testfile.unlink(missing_ok=True)
                if hasattr(testfile, "unlink")
                else testfile.unlink()
            )
            return preferred
        except Exception:
            fallback = Path(tempfile.gettempdir()).resolve() / "hf_engine_fallback"
            self._ensure_dir(fallback)
            return fallback

    def _fallback_log_path(self) -> Path:
        # Fallback-Pfad für Hauptlog bei I/O-Problemen
        return (
            Path(tempfile.gettempdir()).resolve()
            / "hf_engine_fallback"
            / "logging"
            / self.run_id
            / f"{self.component}.jsonl"
        )

    def _fallback_audit_path(self, name: str) -> Path:
        return (
            Path(tempfile.gettempdir()).resolve()
            / "hf_engine_fallback"
            / "audit"
            / self.run_id
            / f"{name}.jsonl"
        )
