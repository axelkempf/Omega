"""Shared Context Management für Agent Orchestration.

Verwaltet geteilten Zustand zwischen Agents während einer Task-Ausführung.
Unterstützt hierarchische Kontexte, Isolation und Audit-Logging.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContextScope(Enum):
    """Gültigkeitsbereich eines Context-Eintrags."""

    GLOBAL = "global"  # Über alle Tasks hinweg
    TASK = "task"  # Nur für aktuellen Task
    AGENT = "agent"  # Nur für aktuellen Agent


@dataclass
class ContextEntry:
    """Ein einzelner Eintrag im geteilten Context.

    Attributes:
        key: Eindeutiger Schlüssel für den Eintrag
        value: Der gespeicherte Wert (muss JSON-serialisierbar sein)
        scope: Gültigkeitsbereich des Eintrags
        created_by: Name des erstellenden Agents
        created_at: Erstellungszeitpunkt (UTC)
        task_id: Optional - zugehörige Task-ID
        metadata: Zusätzliche Metadaten
    """

    key: str
    value: Any
    scope: ContextScope
    created_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope.value,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "task_id": self.task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextEntry:
        """Erstelle aus Dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            scope=ContextScope(data["scope"]),
            created_by=data["created_by"],
            created_at=datetime.fromisoformat(data["created_at"]),
            task_id=data.get("task_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskContext:
    """Context für einen einzelnen Task.

    Attributes:
        task_id: Eindeutige Task-ID
        task_type: Art des Tasks (z.B. 'implement', 'review')
        description: Beschreibung des Tasks
        is_v2: Ob der Task den V2 Backtest-Core betrifft
        affected_files: Liste betroffener Dateien
        parent_task_id: Optional - übergeordneter Task
        metadata: Zusätzliche Metadaten
    """

    task_id: str
    task_type: str
    description: str
    is_v2: bool = False
    affected_files: list[str] = field(default_factory=list)
    parent_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ContextManager:
    """Verwaltet geteilten Context zwischen Agents.

    Thread-safe Implementation für parallele Agent-Ausführung.

    Features:
    - Hierarchische Kontexte (Global → Task → Agent)
    - Automatische Scope-Isolation
    - Audit-Logging aller Änderungen
    - Persistenz-Optionen
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        enable_audit: bool = True,
    ) -> None:
        """Initialisiere den Context Manager.

        Args:
            persistence_path: Optional - Pfad für Context-Persistenz
            enable_audit: Ob Änderungen geloggt werden sollen
        """
        self._lock = threading.RLock()
        self._entries: dict[str, ContextEntry] = {}
        self._tasks: dict[str, TaskContext] = {}
        self._current_task_id: str | None = None
        self._persistence_path = persistence_path
        self._enable_audit = enable_audit
        self._audit_log: list[dict[str, Any]] = []

        if persistence_path and persistence_path.exists():
            self._load_from_file()

    def set(
        self,
        key: str,
        value: Any,
        scope: ContextScope = ContextScope.TASK,
        agent_name: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Setze einen Wert im Context.

        Args:
            key: Schlüssel für den Eintrag
            value: Zu speichernder Wert
            scope: Gültigkeitsbereich
            agent_name: Name des setzenden Agents
            metadata: Zusätzliche Metadaten
        """
        with self._lock:
            # Scope-spezifischen Key erstellen
            scoped_key = self._make_scoped_key(key, scope)

            entry = ContextEntry(
                key=key,
                value=value,
                scope=scope,
                created_by=agent_name,
                task_id=self._current_task_id,
                metadata=metadata or {},
            )

            self._entries[scoped_key] = entry

            if self._enable_audit:
                self._audit("set", key, value, scope, agent_name)

            logger.debug(
                f"Context set: {key}={value!r} "
                f"(scope={scope.value}, agent={agent_name})"
            )

    def get(
        self,
        key: str,
        default: Any = None,
        scope: ContextScope | None = None,
    ) -> Any:
        """Hole einen Wert aus dem Context.

        Sucht in folgender Reihenfolge (falls kein Scope angegeben):
        1. Agent-Scope (falls Task aktiv)
        2. Task-Scope (falls Task aktiv)
        3. Global-Scope

        Args:
            key: Schlüssel des Eintrags
            default: Fallback-Wert
            scope: Optional - spezifischer Scope

        Returns:
            Der gespeicherte Wert oder default
        """
        with self._lock:
            if scope:
                scoped_key = self._make_scoped_key(key, scope)
                entry = self._entries.get(scoped_key)
                return entry.value if entry else default

            # Hierarchische Suche
            for search_scope in [
                ContextScope.AGENT,
                ContextScope.TASK,
                ContextScope.GLOBAL,
            ]:
                scoped_key = self._make_scoped_key(key, search_scope)
                entry = self._entries.get(scoped_key)
                if entry:
                    return entry.value

            return default

    def delete(
        self,
        key: str,
        scope: ContextScope = ContextScope.TASK,
        agent_name: str = "unknown",
    ) -> bool:
        """Lösche einen Eintrag aus dem Context.

        Args:
            key: Schlüssel des Eintrags
            scope: Gültigkeitsbereich
            agent_name: Name des löschenden Agents

        Returns:
            True wenn gelöscht, False wenn nicht gefunden
        """
        with self._lock:
            scoped_key = self._make_scoped_key(key, scope)

            if scoped_key in self._entries:
                del self._entries[scoped_key]

                if self._enable_audit:
                    self._audit("delete", key, None, scope, agent_name)

                return True
            return False

    def start_task(self, task: TaskContext) -> str:
        """Starte einen neuen Task-Context.

        Args:
            task: Der zu startende Task

        Returns:
            Die Task-ID
        """
        with self._lock:
            self._tasks[task.task_id] = task
            self._current_task_id = task.task_id

            if self._enable_audit:
                self._audit_log.append(
                    {
                        "action": "start_task",
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            logger.info(f"Task gestartet: {task.task_id} ({task.task_type})")
            return task.task_id

    def end_task(self, task_id: str) -> None:
        """Beende einen Task-Context und räume auf.

        Args:
            task_id: Die zu beendende Task-ID
        """
        with self._lock:
            # Task-spezifische Einträge entfernen
            keys_to_remove = [
                k
                for k, v in self._entries.items()
                if v.task_id == task_id and v.scope != ContextScope.GLOBAL
            ]
            for key in keys_to_remove:
                del self._entries[key]

            if self._current_task_id == task_id:
                self._current_task_id = None

            if self._enable_audit:
                self._audit_log.append(
                    {
                        "action": "end_task",
                        "task_id": task_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "entries_cleaned": len(keys_to_remove),
                    }
                )

            logger.info(f"Task beendet: {task_id}")

    def get_current_task(self) -> TaskContext | None:
        """Hole den aktuellen Task-Context."""
        with self._lock:
            if self._current_task_id:
                return self._tasks.get(self._current_task_id)
            return None

    def get_task(self, task_id: str) -> TaskContext | None:
        """Hole einen spezifischen Task-Context."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_entries(
        self,
        scope: ContextScope | None = None,
        task_id: str | None = None,
    ) -> list[ContextEntry]:
        """Liste alle Context-Einträge.

        Args:
            scope: Optional - Filter nach Scope
            task_id: Optional - Filter nach Task

        Returns:
            Liste der Einträge
        """
        with self._lock:
            entries = list(self._entries.values())

            if scope:
                entries = [e for e in entries if e.scope == scope]
            if task_id:
                entries = [e for e in entries if e.task_id == task_id]

            return entries

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Hole das Audit-Log."""
        with self._lock:
            return list(self._audit_log)

    def clear(self, scope: ContextScope | None = None) -> int:
        """Lösche Context-Einträge.

        Args:
            scope: Optional - Nur diesen Scope löschen

        Returns:
            Anzahl gelöschter Einträge
        """
        with self._lock:
            if scope:
                keys_to_remove = [
                    k for k, v in self._entries.items() if v.scope == scope
                ]
            else:
                keys_to_remove = list(self._entries.keys())

            for key in keys_to_remove:
                del self._entries[key]

            if self._enable_audit:
                self._audit_log.append(
                    {
                        "action": "clear",
                        "scope": scope.value if scope else "all",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "entries_cleared": len(keys_to_remove),
                    }
                )

            return len(keys_to_remove)

    def save(self) -> None:
        """Speichere Context zu Datei."""
        if not self._persistence_path:
            logger.warning("Keine Persistence-Path konfiguriert")
            return

        with self._lock:
            data = {
                "entries": {k: v.to_dict() for k, v in self._entries.items()},
                "tasks": {
                    k: {
                        "task_id": v.task_id,
                        "task_type": v.task_type,
                        "description": v.description,
                        "is_v2": v.is_v2,
                        "affected_files": v.affected_files,
                        "metadata": v.metadata,
                    }
                    for k, v in self._tasks.items()
                },
                "current_task_id": self._current_task_id,
                "audit_log": self._audit_log,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._persistence_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False)
            )
            logger.info(f"Context gespeichert: {self._persistence_path}")

    def _load_from_file(self) -> None:
        """Lade Context aus Datei."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            data = json.loads(self._persistence_path.read_text())

            # Entries laden
            for key, entry_data in data.get("entries", {}).items():
                self._entries[key] = ContextEntry.from_dict(entry_data)

            # Audit-Log laden
            self._audit_log = data.get("audit_log", [])

            logger.info(f"Context geladen: {len(self._entries)} Einträge")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Fehler beim Laden des Context: {e}")

    def _make_scoped_key(self, key: str, scope: ContextScope) -> str:
        """Erstelle scope-spezifischen Key."""
        if scope == ContextScope.GLOBAL:
            return f"global:{key}"
        elif scope == ContextScope.TASK:
            task_id = self._current_task_id or "no_task"
            return f"task:{task_id}:{key}"
        else:  # AGENT
            task_id = self._current_task_id or "no_task"
            return f"agent:{task_id}:{key}"

    def _audit(
        self,
        action: str,
        key: str,
        value: Any,
        scope: ContextScope,
        agent: str,
    ) -> None:
        """Füge Audit-Eintrag hinzu."""
        self._audit_log.append(
            {
                "action": action,
                "key": key,
                "value_type": type(value).__name__ if value is not None else None,
                "scope": scope.value,
                "agent": agent,
                "task_id": self._current_task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def create_task_id() -> str:
    """Erstelle eine eindeutige Task-ID."""
    return str(uuid.uuid4())[:8]
