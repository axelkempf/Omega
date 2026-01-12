"""Utilities for recording execution stages and serializing instrumentation metadata."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_jsonable(obj: Any) -> Any:
    """Convert objects into JSON-serialisable structures."""
    import numpy as _np  # Local import to keep dependency optional at module import time.

    if obj is None:
        return None
    if isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        val = 0.0 if abs(obj) < 1e-15 else float(obj)
        return val
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        val = float(obj)
        if abs(val) < 1e-15:
            val = 0.0
        return val
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _memory_fingerprint() -> Dict[str, Any]:
    """Return a snapshot of process memory statistics in megabytes."""
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        snapshot: Dict[str, Any] = {
            "rss_mb": round(info.rss / (1024**2), 3),
            "vms_mb": round(info.vms / (1024**2), 3),
            "timestamp": time.time(),
        }
        try:
            full = proc.memory_full_info()
            if hasattr(full, "uss"):
                snapshot["uss_mb"] = round(full.uss / (1024**2), 3)
            if hasattr(full, "shared"):
                snapshot["shared_mb"] = round(full.shared / (1024**2), 3)
        except Exception:
            pass
        return snapshot
    except Exception:
        return {}


@dataclass
class _StageRecord:
    name: str
    duration_sec: float
    started_at: float
    finished_at: float
    memory_before: Dict[str, Any]
    memory_after: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "duration_sec": round(self.duration_sec, 6),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "details": _to_jsonable(self.details),
        }
        if self.memory_before:
            payload["memory_before"] = _to_jsonable(self.memory_before)
        if self.memory_after:
            payload["memory_after"] = _to_jsonable(self.memory_after)
        try:
            if self.memory_before and self.memory_after:
                before = float(self.memory_before.get("rss_mb", 0.0) or 0.0)
                after = float(self.memory_after.get("rss_mb", 0.0) or 0.0)
                payload["memory_delta_rss_mb"] = round(after - before, 6)
        except Exception:
            pass
        if self.error:
            payload["error"] = str(self.error)
        return payload


class _StageContext:
    def __init__(self, recorder: "StageRecorder", name: str):
        self._recorder = recorder
        self._name = name
        self._details: Dict[str, Any] = {}
        self._manual_error: Optional[str] = None
        self._start_perf: Optional[float] = None
        self._start_wall: Optional[float] = None
        self._memory_before: Dict[str, Any] = {}

    def add_details(
        self, data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> "_StageContext":
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None:
                    continue
                self._details[key] = value
        for key, value in kwargs.items():
            if value is None:
                continue
            self._details[key] = value
        return self

    def mark_error(self, message: str) -> "_StageContext":
        self._manual_error = message
        return self

    def __enter__(self) -> "_StageContext":
        self._start_wall = time.time()
        self._start_perf = time.perf_counter()
        self._memory_before = _memory_fingerprint()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        finished_at = time.time()
        end_perf = time.perf_counter()
        duration = 0.0
        if self._start_perf is not None:
            duration = end_perf - self._start_perf
        record = _StageRecord(
            name=self._name,
            duration_sec=duration,
            started_at=self._start_wall or finished_at,
            finished_at=finished_at,
            memory_before=self._memory_before,
            memory_after=_memory_fingerprint(),
            details=self._details,
            error=self._resolve_error(exc_type, exc_val),
        )
        self._recorder._append(record)

    def _resolve_error(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
    ) -> Optional[str]:
        if self._manual_error:
            return self._manual_error
        if exc_type is None and exc_val is None:
            return None
        if exc_type is None:
            return str(exc_val)
        return f"{exc_type.__name__}: {exc_val}" if exc_val else exc_type.__name__


class StageRecorder:
    """Helper to collect timing, memory usage and additional metrics for stages."""

    def __init__(self, scope: str, *, metadata: Optional[Dict[str, Any]] = None):
        self._scope = scope
        self._metadata: Dict[str, Any] = metadata or {}
        self._records: List[_StageRecord] = []
        self._created_at = time.time()

    def add_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            self._metadata[key] = value

    def stage(self, name: str) -> _StageContext:
        return _StageContext(self, name)

    def _append(self, record: _StageRecord) -> None:
        self._records.append(record)

    def to_dict(self) -> Dict[str, Any]:
        total_duration = sum(r.duration_sec for r in self._records)
        finished_at = (
            self._records[-1].finished_at if self._records else self._created_at
        )
        payload: Dict[str, Any] = {
            "scope": self._scope,
            "created_at": self._created_at,
            "finished_at": finished_at,
            "stage_count": len(self._records),
            "total_duration_sec": round(total_duration, 6),
            "stages": [r.as_dict() for r in self._records],
        }
        if self._metadata:
            payload["metadata"] = _to_jsonable(self._metadata)
        return payload


def _format_stage_summary(stage_payload: Dict[str, Any]) -> str:
    stages = stage_payload.get("stages", []) if isinstance(stage_payload, dict) else []
    parts: List[str] = []
    for stage in stages:
        try:
            name = stage.get("name")
            duration = float(stage.get("duration_sec", 0.0) or 0.0)
            parts.append(f"{name}: {duration:.2f}s")
        except Exception:
            continue
    return "; ".join(parts)


__all__ = [
    "StageRecorder",
    "_StageContext",
    "_StageRecord",
    "_format_stage_summary",
    "_memory_fingerprint",
    "_to_jsonable",
]
