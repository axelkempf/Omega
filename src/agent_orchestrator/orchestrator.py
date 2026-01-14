"""Agent Orchestrator - Zentrale Koordination für Multi-Agent-Workflows.

Dieses Modul enthält die Hauptklasse `AgentOrchestrator`, die alle Komponenten
des Agent-Netzwerks integriert und Tasks an die passenden Agenten routet.

Der Orchestrator ist das Herzstück des Omega Agent Networks und koordiniert:
- Task-Routing über den TaskRouter
- Kontext-Sharing über den ContextManager
- V2-Erkennung für Backtest-Core-spezifische Tasks
- Workflow-Ausführung für komplexe Multi-Step-Prozesse

Example:
    >>> from agent_orchestrator import AgentOrchestrator, OrchestratorConfig
    >>> config = OrchestratorConfig(enable_v2_detection=True)
    >>> orchestrator = AgentOrchestrator(config)
    >>> result = orchestrator.submit_task(
    ...     description="Implementiere Feature X in rust_core/crates/execution/",
    ...     context={"files": ["rust_core/crates/execution/src/lib.rs"]}
    ... )
    >>> print(result.primary_agent, result.status)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .context import ContextManager, ContextScope, TaskContext
from .router import AgentRole, RoutingResult, TaskRouter
from .v2_detector import V2DetectionResult, V2Detector

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status eines Tasks im Orchestrator."""

    PENDING = "pending"
    ROUTING = "routing"
    IN_PROGRESS = "in_progress"
    WAITING_REVIEW = "waiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priorität eines Tasks."""

    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10


@dataclass
class OrchestratorConfig:
    """Konfiguration für den AgentOrchestrator.

    Attributes:
        enable_v2_detection: Aktiviert automatische V2-Kontext-Erkennung.
        context_persistence: Aktiviert persistente Kontext-Speicherung.
        context_dir: Verzeichnis für persistierten Kontext.
        max_concurrent_tasks: Maximale Anzahl paralleler Tasks.
        default_timeout_seconds: Standard-Timeout für Tasks.
        enable_audit_logging: Aktiviert detailliertes Audit-Logging.
        auto_instruction_aggregation: Sammelt automatisch passende Instructions.
        require_approval_for_critical: Erfordert Approval für kritische Pfade.
        critical_paths: Liste kritischer Pfade die Approval erfordern.
        workspace_root: Workspace-Root-Verzeichnis.
    """

    enable_v2_detection: bool = True
    context_persistence: bool = True
    context_dir: Path = field(default_factory=lambda: Path("var/orchestrator/context"))
    max_concurrent_tasks: int = 5
    default_timeout_seconds: int = 3600
    enable_audit_logging: bool = True
    auto_instruction_aggregation: bool = True
    require_approval_for_critical: bool = True
    critical_paths: list[str] = field(
        default_factory=lambda: [
            "src/hf_engine/core/execution/",
            "src/hf_engine/core/risk/",
            "src/hf_engine/adapter/broker/",
            "configs/live/",
            "rust_core/crates/execution/",
            "rust_core/crates/ffi/",
        ]
    )
    workspace_root: Path = field(default_factory=lambda: Path.cwd())


@dataclass
class TaskResult:
    """Ergebnis einer Task-Ausführung.

    Attributes:
        task_id: Eindeutige Task-ID.
        status: Aktueller Status des Tasks.
        primary_agent: Primär zuständiger Agent.
        secondary_agents: Sekundär beteiligte Agenten.
        routing_result: Detailliertes Routing-Ergebnis.
        v2_detection: V2-Erkennungsergebnis falls relevant.
        context: Task-Kontext mit Shared State.
        outputs: Task-Outputs (Code, Dokumente, etc.).
        errors: Aufgetretene Fehler.
        warnings: Warnungen während der Ausführung.
        started_at: Startzeitpunkt.
        completed_at: Endzeitpunkt (falls abgeschlossen).
        duration_seconds: Ausführungsdauer in Sekunden.
        requires_approval: Ob der Task Approval erfordert.
        approval_reason: Grund für erforderliches Approval.
        metadata: Zusätzliche Metadaten.
    """

    task_id: str
    status: TaskStatus
    primary_agent: AgentRole | None = None
    secondary_agents: list[AgentRole] = field(default_factory=list)
    routing_result: RoutingResult | None = None
    v2_detection: V2DetectionResult | None = None
    context: TaskContext | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    requires_approval: bool = False
    approval_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Prüft ob der Task erfolgreich abgeschlossen wurde."""
        return self.status == TaskStatus.COMPLETED and not self.errors

    def add_error(self, error: str) -> None:
        """Fügt einen Fehler hinzu."""
        self.errors.append(error)
        logger.error(f"Task {self.task_id}: {error}")

    def add_warning(self, warning: str) -> None:
        """Fügt eine Warnung hinzu."""
        self.warnings.append(warning)
        logger.warning(f"Task {self.task_id}: {warning}")

    def add_output(self, key: str, value: Any) -> None:
        """Fügt einen Output hinzu."""
        self.outputs[key] = value

    def complete(self, status: TaskStatus = TaskStatus.COMPLETED) -> None:
        """Markiert den Task als abgeschlossen."""
        self.status = status
        self.completed_at = datetime.now()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_seconds = delta.total_seconds()


@dataclass
class TaskRequest:
    """Anfrage zur Task-Ausführung.

    Attributes:
        description: Beschreibung des Tasks.
        files: Betroffene Dateien.
        context: Zusätzlicher Kontext.
        priority: Task-Priorität.
        requested_agent: Optional explizit angeforderter Agent.
        timeout_seconds: Optionaler Timeout.
        skip_approval: Überspringt Approval-Anforderung.
        parent_task_id: Optional Parent-Task für Sub-Tasks.
        tags: Tags für Kategorisierung.
    """

    description: str
    files: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    requested_agent: AgentRole | None = None
    timeout_seconds: int | None = None
    skip_approval: bool = False
    parent_task_id: str | None = None
    tags: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """Zentraler Orchestrator für das Omega Agent Network.

    Der Orchestrator koordiniert alle Agent-Aktivitäten und stellt sicher,
    dass Tasks an die passenden Agenten geroutet werden, der Kontext
    zwischen Agenten geteilt wird und kritische Pfade geschützt sind.

    Attributes:
        config: Orchestrator-Konfiguration.
        router: TaskRouter für intelligentes Routing.
        context_manager: ContextManager für Shared State.
        v2_detector: V2Detector für Backtest-Core-Erkennung.
        active_tasks: Dict der aktiven Tasks.
        task_history: Liste abgeschlossener Tasks.

    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> request = TaskRequest(
        ...     description="Bug Fix in Execution Layer",
        ...     files=["rust_core/crates/execution/src/fill.rs"],
        ...     priority=TaskPriority.HIGH
        ... )
        >>> result = orchestrator.submit_task(request)
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        router: TaskRouter | None = None,
        context_manager: ContextManager | None = None,
        v2_detector: V2Detector | None = None,
    ) -> None:
        """Initialisiert den Orchestrator.

        Args:
            config: Optionale Konfiguration.
            router: Optionaler TaskRouter (Default wird erstellt).
            context_manager: Optionaler ContextManager (Default wird erstellt).
            v2_detector: Optionaler V2Detector (Default wird erstellt).
        """
        self.config = config or OrchestratorConfig()
        self.router = router or TaskRouter()
        self.context_manager = context_manager or ContextManager(
            persistence_enabled=self.config.context_persistence,
            persistence_dir=self.config.context_dir,
        )
        self.v2_detector = v2_detector or V2Detector()

        self._active_tasks: dict[str, TaskResult] = {}
        self._task_history: list[TaskResult] = []
        self._approval_callbacks: list[Callable[[TaskResult], bool]] = []
        self._pre_execution_hooks: list[Callable[[TaskRequest, TaskResult], None]] = []
        self._post_execution_hooks: list[Callable[[TaskResult], None]] = []

        # Kontext-Verzeichnis erstellen
        if self.config.context_persistence:
            self.config.context_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"AgentOrchestrator initialisiert "
            f"(V2-Detection: {self.config.enable_v2_detection}, "
            f"Context-Persistence: {self.config.context_persistence})"
        )

    def submit_task(
        self,
        request: TaskRequest | None = None,
        *,
        description: str = "",
        files: list[str] | None = None,
        context: dict[str, Any] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> TaskResult:
        """Reicht einen Task zur Ausführung ein.

        Kann entweder mit einem TaskRequest-Objekt oder mit einzelnen
        Parametern aufgerufen werden.

        Args:
            request: Optionales TaskRequest-Objekt.
            description: Task-Beschreibung (wenn kein request).
            files: Betroffene Dateien (wenn kein request).
            context: Zusätzlicher Kontext (wenn kein request).
            priority: Task-Priorität (wenn kein request).

        Returns:
            TaskResult mit Routing-Informationen und Status.

        Raises:
            ValueError: Wenn weder request noch description angegeben.
        """
        # TaskRequest erstellen falls nicht übergeben
        if request is None:
            if not description:
                raise ValueError("Entweder request oder description erforderlich")
            request = TaskRequest(
                description=description,
                files=files or [],
                context=context or {},
                priority=priority,
            )

        # Task-ID generieren
        task_id = str(uuid.uuid4())[:8]

        # TaskResult initialisieren
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            started_at=datetime.now(),
            metadata={
                "request_description": request.description,
                "request_files": request.files,
                "priority": request.priority.name,
            },
        )

        logger.info(f"Task {task_id} eingereicht: {request.description[:100]}...")

        try:
            # Phase 1: Routing
            result.status = TaskStatus.ROUTING
            routing_result = self._route_task(request, result)
            result.routing_result = routing_result
            result.primary_agent = routing_result.primary_role
            result.secondary_agents = list(routing_result.secondary_roles)
            result.v2_detection = routing_result.v2_detection

            # Phase 2: Approval-Check für kritische Pfade
            if self._requires_approval(request, result):
                result.requires_approval = True
                result.approval_reason = self._get_approval_reason(request, result)
                result.status = TaskStatus.WAITING_REVIEW
                logger.info(
                    f"Task {task_id} erfordert Approval: {result.approval_reason}"
                )
            else:
                result.status = TaskStatus.IN_PROGRESS

            # Task-Kontext erstellen und speichern
            task_context = self._create_task_context(task_id, request, result)
            result.context = task_context

            # Pre-Execution Hooks
            for hook in self._pre_execution_hooks:
                hook(request, result)

            # Task registrieren
            self._active_tasks[task_id] = result

            logger.info(
                f"Task {task_id} geroutet zu {result.primary_agent.value} "
                f"(V2: {result.v2_detection.is_v2_context if result.v2_detection else False})"
            )

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.add_error(f"Routing fehlgeschlagen: {e}")
            logger.exception(f"Task {task_id} Routing fehlgeschlagen")

        return result

    def _route_task(self, request: TaskRequest, result: TaskResult) -> RoutingResult:
        """Routet einen Task zum passenden Agenten.

        Args:
            request: Task-Anfrage.
            result: TaskResult für Status-Updates.

        Returns:
            RoutingResult mit Agent-Zuweisung.
        """
        # V2-Detection ausführen
        v2_detection = None
        if self.config.enable_v2_detection:
            v2_detection = self.v2_detector.detect(
                files=request.files,
                description=request.description,
            )

        # Kontext für Routing sammeln
        routing_context = {
            "description": request.description,
            "files": request.files,
            "priority": request.priority.value,
            "tags": request.tags,
            **request.context,
        }

        # Routing durchführen
        if request.requested_agent:
            # Explizit angeforderter Agent
            routing_result = RoutingResult(
                primary_role=request.requested_agent,
                secondary_roles=set(),
                instructions=self.router.get_role_instructions(request.requested_agent),
                matched_rule="explicit_request",
                v2_detection=v2_detection,
                routing_context=routing_context,
            )
            result.add_warning(
                f"Explizit angeforderter Agent: {request.requested_agent.value}"
            )
        else:
            # Automatisches Routing
            routing_result = self.router.route(
                task_description=request.description,
                files=request.files,
                context=routing_context,
            )

        return routing_result

    def _requires_approval(self, request: TaskRequest, result: TaskResult) -> bool:
        """Prüft ob ein Task Approval erfordert.

        Args:
            request: Task-Anfrage.
            result: Bisheriges TaskResult.

        Returns:
            True wenn Approval erforderlich.
        """
        if request.skip_approval:
            return False

        if not self.config.require_approval_for_critical:
            return False

        # Kritische Pfade prüfen
        for file in request.files:
            for critical_path in self.config.critical_paths:
                if critical_path in file:
                    return True

        # Safety-kritische Keywords
        safety_keywords = ["live", "execution", "risk", "trading", "broker"]
        desc_lower = request.description.lower()
        if any(kw in desc_lower for kw in safety_keywords):
            # Nur wenn auch Live-Pfade betroffen
            if any("hf_engine" in f or "configs/live" in f for f in request.files):
                return True

        return False

    def _get_approval_reason(self, request: TaskRequest, result: TaskResult) -> str:
        """Ermittelt den Grund für erforderliches Approval.

        Args:
            request: Task-Anfrage.
            result: Bisheriges TaskResult.

        Returns:
            Begründung für das Approval-Erfordernis.
        """
        reasons = []

        # Kritische Pfade
        for file in request.files:
            for critical_path in self.config.critical_paths:
                if critical_path in file:
                    reasons.append(f"Kritischer Pfad: {critical_path}")
                    break

        # V2 FFI-Boundary
        if result.v2_detection and result.v2_detection.is_v2_context:
            if any("ffi" in f.lower() for f in request.files):
                reasons.append("V2 FFI-Boundary Änderung")

        if reasons:
            return "; ".join(reasons)
        return "Sicherheitsrelevante Änderung"

    def _create_task_context(
        self,
        task_id: str,
        request: TaskRequest,
        result: TaskResult,
    ) -> TaskContext:
        """Erstellt den Task-Kontext.

        Args:
            task_id: Task-ID.
            request: Task-Anfrage.
            result: Bisheriges TaskResult.

        Returns:
            Initialisierter TaskContext.
        """
        return self.context_manager.create_task_context(
            task_id=task_id,
            initial_data={
                "description": request.description,
                "files": request.files,
                "priority": request.priority.name,
                "primary_agent": (
                    result.primary_agent.value if result.primary_agent else None
                ),
                "v2_context": (
                    result.v2_detection.is_v2_context if result.v2_detection else False
                ),
                "routing_rule": (
                    result.routing_result.matched_rule
                    if result.routing_result
                    else None
                ),
            },
        )

    def complete_task(
        self,
        task_id: str,
        status: TaskStatus = TaskStatus.COMPLETED,
        outputs: dict[str, Any] | None = None,
    ) -> TaskResult | None:
        """Markiert einen Task als abgeschlossen.

        Args:
            task_id: Task-ID.
            status: Abschluss-Status.
            outputs: Task-Outputs.

        Returns:
            Aktualisiertes TaskResult oder None wenn nicht gefunden.
        """
        result = self._active_tasks.get(task_id)
        if not result:
            logger.warning(f"Task {task_id} nicht gefunden")
            return None

        if outputs:
            result.outputs.update(outputs)

        result.complete(status)

        # Post-Execution Hooks
        for hook in self._post_execution_hooks:
            hook(result)

        # In History verschieben
        self._task_history.append(result)
        del self._active_tasks[task_id]

        logger.info(
            f"Task {task_id} abgeschlossen mit Status {status.value} "
            f"(Dauer: {result.duration_seconds:.1f}s)"
        )

        return result

    def approve_task(self, task_id: str, approver: str = "human") -> bool:
        """Genehmigt einen wartenden Task.

        Args:
            task_id: Task-ID.
            approver: Name des Genehmigenden.

        Returns:
            True wenn erfolgreich genehmigt.
        """
        result = self._active_tasks.get(task_id)
        if not result:
            logger.warning(f"Task {task_id} nicht gefunden")
            return False

        if result.status != TaskStatus.WAITING_REVIEW:
            logger.warning(f"Task {task_id} ist nicht im Review-Status")
            return False

        result.status = TaskStatus.IN_PROGRESS
        result.metadata["approved_by"] = approver
        result.metadata["approved_at"] = datetime.now().isoformat()
        result.requires_approval = False

        logger.info(f"Task {task_id} genehmigt von {approver}")
        return True

    def reject_task(self, task_id: str, reason: str, rejector: str = "human") -> bool:
        """Lehnt einen wartenden Task ab.

        Args:
            task_id: Task-ID.
            reason: Ablehnungsgrund.
            rejector: Name des Ablehnenden.

        Returns:
            True wenn erfolgreich abgelehnt.
        """
        result = self._active_tasks.get(task_id)
        if not result:
            logger.warning(f"Task {task_id} nicht gefunden")
            return False

        result.status = TaskStatus.CANCELLED
        result.add_error(f"Abgelehnt von {rejector}: {reason}")
        result.metadata["rejected_by"] = rejector
        result.metadata["rejected_at"] = datetime.now().isoformat()
        result.metadata["rejection_reason"] = reason

        # In History verschieben
        result.complete(TaskStatus.CANCELLED)
        self._task_history.append(result)
        del self._active_tasks[task_id]

        logger.info(f"Task {task_id} abgelehnt von {rejector}: {reason}")
        return True

    def get_task(self, task_id: str) -> TaskResult | None:
        """Holt ein TaskResult nach ID.

        Args:
            task_id: Task-ID.

        Returns:
            TaskResult oder None wenn nicht gefunden.
        """
        return self._active_tasks.get(task_id)

    def get_active_tasks(self) -> list[TaskResult]:
        """Holt alle aktiven Tasks.

        Returns:
            Liste aktiver TaskResults.
        """
        return list(self._active_tasks.values())

    def get_task_history(self, limit: int = 100) -> list[TaskResult]:
        """Holt die Task-Historie.

        Args:
            limit: Maximale Anzahl zurückgegebener Tasks.

        Returns:
            Liste der letzten Tasks.
        """
        return self._task_history[-limit:]

    def add_pre_execution_hook(
        self,
        hook: Callable[[TaskRequest, TaskResult], None],
    ) -> None:
        """Fügt einen Pre-Execution Hook hinzu.

        Args:
            hook: Callback-Funktion.
        """
        self._pre_execution_hooks.append(hook)

    def add_post_execution_hook(
        self,
        hook: Callable[[TaskResult], None],
    ) -> None:
        """Fügt einen Post-Execution Hook hinzu.

        Args:
            hook: Callback-Funktion.
        """
        self._post_execution_hooks.append(hook)

    def get_statistics(self) -> dict[str, Any]:
        """Holt Orchestrator-Statistiken.

        Returns:
            Dict mit Statistiken.
        """
        completed = [t for t in self._task_history if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self._task_history if t.status == TaskStatus.FAILED]

        agent_distribution: dict[str, int] = {}
        for task in self._task_history:
            if task.primary_agent:
                agent_name = task.primary_agent.value
                agent_distribution[agent_name] = (
                    agent_distribution.get(agent_name, 0) + 1
                )

        avg_duration = 0.0
        if completed:
            durations = [t.duration_seconds for t in completed if t.duration_seconds]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "active_tasks": len(self._active_tasks),
            "total_completed": len(completed),
            "total_failed": len(failed),
            "success_rate": (
                len(completed) / len(self._task_history) if self._task_history else 0.0
            ),
            "average_duration_seconds": avg_duration,
            "agent_distribution": agent_distribution,
            "v2_tasks": sum(
                1
                for t in self._task_history
                if t.v2_detection and t.v2_detection.is_v2_context
            ),
            "approval_required_tasks": sum(
                1 for t in self._task_history if t.requires_approval
            ),
        }
