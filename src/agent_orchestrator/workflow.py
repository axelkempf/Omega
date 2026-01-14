"""Workflow-Engine für YAML-basierte Multi-Agent-Workflows.

Diese Komponente ermöglicht die Definition und Ausführung von
komplexen Multi-Step-Workflows mit parallelen und sequenziellen Steps.

Design-Prinzipien:
- YAML-basierte Workflow-Definitionen (deklarativ)
- Step-Dependencies und parallele Ausführung
- Konditionale Step-Ausführung
- Integrierte Success/Failure Handlers
- V2-Aware für Backtest-Core Workflows
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml

from .router import AgentRole

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class StepStatus(Enum):
    """Status eines Workflow-Steps."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """Status eines Workflows."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StepDefinition:
    """Definition eines Workflow-Steps (aus YAML)."""

    name: str
    agent: AgentRole | str
    description: str = ""

    # Input/Output
    input: list[str] | str | None = None
    output: str | None = None

    # Execution
    commands: list[str] = field(default_factory=list)
    file_filter: str | None = None
    checklist: list[str] = field(default_factory=list)

    # Flow Control
    parallel: bool = False
    condition: str | None = None
    depends_on: list[str] = field(default_factory=list)

    # Timeouts
    timeout_seconds: int = 3600


@dataclass
class StepResult:
    """Ergebnis eines ausgeführten Steps."""

    step_name: str
    status: StepStatus
    agent: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Berechnet die Ausführungsdauer in Sekunden."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class WorkflowDefinition:
    """Definition eines kompletten Workflows (aus YAML)."""

    name: str
    description: str = ""
    version: str = "1.0"

    # Steps
    steps: list[StepDefinition] = field(default_factory=list)

    # Context
    context: dict[str, Any] = field(default_factory=dict)
    v2_mode: bool = False
    requires_approval: bool = False

    # Handlers
    on_success: list[dict[str, Any]] = field(default_factory=list)
    on_failure: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    author: str | None = None


@dataclass
class WorkflowInstance:
    """Laufende Instanz eines Workflows."""

    workflow_id: str
    definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.CREATED
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    step_results: dict[str, StepResult] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # Runtime Context
    context: dict[str, Any] = field(default_factory=dict)
    current_step: str | None = None
    failed_step: str | None = None

    @property
    def is_complete(self) -> bool:
        """Prüft ob der Workflow abgeschlossen ist."""
        return self.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Berechnet die Gesamt-Ausführungsdauer."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# =============================================================================
# Workflow Parser
# =============================================================================


class WorkflowParser:
    """Parser für YAML-Workflow-Definitionen."""

    def parse_file(self, path: Path) -> WorkflowDefinition:
        """Parst eine YAML-Workflow-Datei."""
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self.parse_dict(data)

    def parse_dict(self, data: dict[str, Any]) -> WorkflowDefinition:
        """Parst ein Workflow-Dictionary."""
        steps = []
        for step_data in data.get("steps", []):
            step = self._parse_step(step_data)
            steps.append(step)

        return WorkflowDefinition(
            name=data.get("name", "unnamed_workflow"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            steps=steps,
            context=data.get("context", {}),
            v2_mode=data.get("v2_mode", False),
            requires_approval=data.get("requires_approval", False),
            on_success=data.get("on_success", []),
            on_failure=data.get("on_failure", []),
            tags=data.get("tags", []),
            author=data.get("author"),
        )

    def _parse_step(self, data: dict[str, Any]) -> StepDefinition:
        """Parst einen Step aus YAML-Daten."""
        agent_str = data.get("agent", "implementer")
        agent = self._resolve_agent(agent_str)

        # Input kann string oder liste sein
        input_data = data.get("input")
        if isinstance(input_data, str):
            input_list: list[str] | str | None = input_data
        elif isinstance(input_data, list):
            input_list = input_data
        else:
            input_list = None

        return StepDefinition(
            name=data.get("name", "unnamed_step"),
            agent=agent,
            description=data.get("description", ""),
            input=input_list,
            output=data.get("output"),
            commands=data.get("commands", []),
            file_filter=data.get("file_filter"),
            checklist=data.get("checklist", []),
            parallel=data.get("parallel", False),
            condition=data.get("condition"),
            depends_on=data.get("depends_on", []),
            timeout_seconds=data.get("timeout_seconds", 3600),
        )

    def _resolve_agent(self, agent_str: str) -> AgentRole | str:
        """Löst einen Agent-String zu AgentRole auf."""
        try:
            return AgentRole(agent_str.lower())
        except ValueError:
            logger.warning(f"Unknown agent role: {agent_str}, using as string")
            return agent_str


# =============================================================================
# Workflow Engine
# =============================================================================


class WorkflowEngine:
    """Engine für die Ausführung von Multi-Agent-Workflows.

    Verantwortlichkeiten:
    - Laden von Workflow-Definitionen aus YAML
    - Step-Scheduling und Dependency Resolution
    - Parallele Step-Ausführung
    - Konditionale Step-Evaluierung
    - Success/Failure Handler-Ausführung
    """

    def __init__(
        self,
        workflow_dir: Path | None = None,
        workspace_root: Path | None = None,
    ):
        """Initialisiert die WorkflowEngine.

        Args:
            workflow_dir: Verzeichnis mit Workflow-YAML-Dateien
            workspace_root: Workspace-Root für relative Pfade
        """
        self.workflow_dir = workflow_dir or Path("src/agent_orchestrator/workflows")
        self.workspace_root = workspace_root or Path.cwd()
        self.parser = WorkflowParser()

        # Registrierte Agent-Executor-Funktionen
        self._executors: dict[AgentRole, Callable] = {}

        # Registrierte Condition-Evaluatoren
        self._condition_evaluators: dict[str, Callable[[dict[str, Any]], bool]] = {}

        # Workflow-Cache
        self._workflow_cache: dict[str, WorkflowDefinition] = {}

        # Laufende Instanzen
        self._instances: dict[str, WorkflowInstance] = {}

        logger.info(f"WorkflowEngine initialized with workflow_dir={self.workflow_dir}")

    # -------------------------------------------------------------------------
    # Workflow Loading
    # -------------------------------------------------------------------------

    def load_workflow(self, name: str) -> WorkflowDefinition:
        """Lädt einen Workflow aus YAML.

        Args:
            name: Name des Workflows (ohne .yaml Extension)

        Returns:
            WorkflowDefinition für den angeforderten Workflow

        Raises:
            FileNotFoundError: Wenn die Workflow-Datei nicht existiert
        """
        # Cache prüfen
        if name in self._workflow_cache:
            return self._workflow_cache[name]

        # YAML-Datei laden
        yaml_path = self.workflow_dir / f"{name}.yaml"
        if not yaml_path.exists():
            # Alternative: .yml Extension
            yaml_path = self.workflow_dir / f"{name}.yml"

        workflow = self.parser.parse_file(yaml_path)
        self._workflow_cache[name] = workflow

        logger.debug(f"Loaded workflow '{name}' with {len(workflow.steps)} steps")
        return workflow

    def list_workflows(self) -> list[str]:
        """Listet alle verfügbaren Workflows auf.

        Returns:
            Liste der Workflow-Namen
        """
        workflows = []

        if self.workflow_dir.exists():
            for path in self.workflow_dir.iterdir():
                if path.suffix in (".yaml", ".yml"):
                    workflows.append(path.stem)

        return sorted(workflows)

    def get_workflow_info(self, name: str) -> dict[str, Any]:
        """Holt Informationen über einen Workflow.

        Args:
            name: Name des Workflows

        Returns:
            Dictionary mit Workflow-Informationen
        """
        workflow = self.load_workflow(name)

        return {
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "v2_mode": workflow.v2_mode,
            "requires_approval": workflow.requires_approval,
            "step_count": len(workflow.steps),
            "steps": [
                {
                    "name": s.name,
                    "agent": (
                        s.agent.value if isinstance(s.agent, AgentRole) else s.agent
                    ),
                    "parallel": s.parallel,
                }
                for s in workflow.steps
            ],
            "tags": workflow.tags,
            "author": workflow.author,
        }

    # -------------------------------------------------------------------------
    # Executor Registration
    # -------------------------------------------------------------------------

    def register_executor(
        self,
        role: AgentRole,
        executor: Callable[[StepDefinition, dict[str, Any]], StepResult],
    ) -> None:
        """Registriert einen Executor für eine Agent-Rolle.

        Args:
            role: Agent-Rolle für die der Executor gilt
            executor: Callable das Steps ausführt
        """
        self._executors[role] = executor
        logger.debug(f"Registered executor for role {role.value}")

    def register_condition_evaluator(
        self,
        pattern: str,
        evaluator: Callable[[dict[str, Any]], bool],
    ) -> None:
        """Registriert einen Evaluator für Conditions.

        Args:
            pattern: Pattern das die Condition matcht
            evaluator: Callable das True/False zurückgibt
        """
        self._condition_evaluators[pattern] = evaluator
        logger.debug(f"Registered condition evaluator for pattern '{pattern}'")

    # -------------------------------------------------------------------------
    # Workflow Execution
    # -------------------------------------------------------------------------

    def create_instance(
        self,
        workflow_name: str,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowInstance:
        """Erstellt eine neue Workflow-Instanz.

        Args:
            workflow_name: Name des Workflows
            workflow_id: Eindeutige ID für diese Instanz
            initial_context: Initialer Kontext für den Workflow

        Returns:
            Neue WorkflowInstance
        """
        definition = self.load_workflow(workflow_name)

        instance = WorkflowInstance(
            workflow_id=workflow_id,
            definition=definition,
            context={**definition.context, **(initial_context or {})},
        )

        self._instances[workflow_id] = instance
        logger.info(f"Created workflow instance '{workflow_id}' for '{workflow_name}'")

        return instance

    def get_instance(self, workflow_id: str) -> WorkflowInstance | None:
        """Holt eine Workflow-Instanz.

        Args:
            workflow_id: ID der Instanz

        Returns:
            WorkflowInstance oder None
        """
        return self._instances.get(workflow_id)

    def get_ready_steps(self, instance: WorkflowInstance) -> list[StepDefinition]:
        """Ermittelt alle Steps die bereit zur Ausführung sind.

        Ein Step ist bereit wenn:
        - Er noch nicht ausgeführt wurde
        - Alle Dependencies erfüllt sind
        - Seine Condition (falls vorhanden) True ergibt

        Args:
            instance: Die Workflow-Instanz

        Returns:
            Liste der ausführbereiten Steps
        """
        ready = []

        for step in instance.definition.steps:
            # Bereits ausgeführt?
            if step.name in instance.step_results:
                continue

            # Dependencies prüfen
            if not self._check_dependencies(step, instance):
                continue

            # Condition prüfen
            if not self._evaluate_condition(step, instance):
                # Step wird übersprungen
                instance.step_results[step.name] = StepResult(
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                    agent=(
                        step.agent.value
                        if isinstance(step.agent, AgentRole)
                        else step.agent
                    ),
                    metadata={"reason": "Condition not met"},
                )
                continue

            ready.append(step)

        return ready

    def execute_step(
        self,
        instance: WorkflowInstance,
        step: StepDefinition,
    ) -> StepResult:
        """Führt einen einzelnen Step aus.

        Args:
            instance: Die Workflow-Instanz
            step: Der auszuführende Step

        Returns:
            StepResult mit dem Ergebnis
        """
        agent_role = (
            step.agent if isinstance(step.agent, AgentRole) else AgentRole.IMPLEMENTER
        )
        agent_str = (
            agent_role.value if isinstance(step.agent, AgentRole) else step.agent
        )

        result = StepResult(
            step_name=step.name,
            status=StepStatus.RUNNING,
            agent=agent_str,
            started_at=datetime.now(),
        )

        logger.info(f"Executing step '{step.name}' with agent '{agent_str}'")

        try:
            # Executor suchen
            executor = self._executors.get(agent_role)

            if executor:
                # Mit registriertem Executor ausführen
                result = executor(step, instance.context)
            else:
                # Placeholder-Execution (für Tests/Demo)
                logger.warning(
                    f"No executor for role {agent_role.value}, "
                    f"using placeholder execution"
                )
                result.status = StepStatus.COMPLETED
                result.outputs = {"placeholder": True}

            # Output in Context speichern
            if step.output and result.outputs:
                instance.outputs[step.output] = result.outputs
                instance.context[step.output] = result.outputs

        except Exception as e:
            logger.exception(f"Step '{step.name}' failed with error: {e}")
            result.status = StepStatus.FAILED
            result.errors.append(str(e))

        result.completed_at = datetime.now()
        instance.step_results[step.name] = result

        return result

    def run_workflow(
        self,
        instance: WorkflowInstance,
    ) -> WorkflowInstance:
        """Führt einen kompletten Workflow aus.

        Args:
            instance: Die Workflow-Instanz

        Returns:
            Die aktualisierte WorkflowInstance
        """
        instance.status = WorkflowStatus.RUNNING
        instance.started_at = datetime.now()

        logger.info(f"Starting workflow '{instance.definition.name}'")

        try:
            while True:
                ready_steps = self.get_ready_steps(instance)

                if not ready_steps:
                    # Keine Steps mehr ausführbar
                    break

                # Parallele Steps gruppieren
                parallel_steps = [s for s in ready_steps if s.parallel]
                sequential_steps = [s for s in ready_steps if not s.parallel]

                # TODO: Echte parallele Ausführung mit asyncio/threading
                for step in parallel_steps + sequential_steps:
                    instance.current_step = step.name
                    result = self.execute_step(instance, step)

                    if result.status == StepStatus.FAILED:
                        instance.failed_step = step.name
                        instance.status = WorkflowStatus.FAILED
                        instance.errors.append(
                            f"Step '{step.name}' failed: {result.errors}"
                        )
                        break

                if instance.status == WorkflowStatus.FAILED:
                    break

            # Workflow erfolgreich abgeschlossen?
            if instance.status != WorkflowStatus.FAILED:
                # Prüfen ob alle nicht-optionalen Steps ausgeführt wurden
                all_completed = all(
                    step.name in instance.step_results
                    for step in instance.definition.steps
                )
                if all_completed:
                    instance.status = WorkflowStatus.COMPLETED
                else:
                    # Einige Steps wurden übersprungen (Conditions)
                    instance.status = WorkflowStatus.COMPLETED

            # Success/Failure Handlers ausführen
            self._run_handlers(instance)

        except Exception as e:
            logger.exception(f"Workflow failed with unexpected error: {e}")
            instance.status = WorkflowStatus.FAILED
            instance.errors.append(str(e))

        instance.completed_at = datetime.now()
        instance.current_step = None

        logger.info(
            f"Workflow '{instance.definition.name}' completed with status "
            f"{instance.status.value}"
        )

        return instance

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _check_dependencies(
        self,
        step: StepDefinition,
        instance: WorkflowInstance,
    ) -> bool:
        """Prüft ob alle Dependencies eines Steps erfüllt sind."""
        for dep_name in step.depends_on:
            dep_result = instance.step_results.get(dep_name)
            if not dep_result:
                return False
            if dep_result.status not in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                return False
        return True

    def _evaluate_condition(
        self,
        step: StepDefinition,
        instance: WorkflowInstance,
    ) -> bool:
        """Evaluiert die Condition eines Steps."""
        if not step.condition:
            return True

        # Registrierte Evaluatoren prüfen
        for pattern, evaluator in self._condition_evaluators.items():
            if re.search(pattern, step.condition):
                return evaluator(instance.context)

        # Simple Expression Evaluation (für demo/tests)
        # Format: "variable.property" oder "variable"
        try:
            parts = step.condition.split(".")
            value = instance.context.get(parts[0])

            if value is None:
                return False

            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)

            return bool(value)

        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{step.condition}': {e}")
            return True  # Default: Step ausführen

    def _run_handlers(self, instance: WorkflowInstance) -> None:
        """Führt Success/Failure Handler aus."""
        handlers = (
            instance.definition.on_success
            if instance.status == WorkflowStatus.COMPLETED
            else instance.definition.on_failure
        )

        for handler in handlers:
            for action, value in handler.items():
                self._execute_handler_action(action, value, instance)

    def _execute_handler_action(
        self,
        action: str,
        value: Any,
        instance: WorkflowInstance,
    ) -> None:
        """Führt eine einzelne Handler-Action aus."""
        # Template-Variablen ersetzen
        if isinstance(value, str):
            value = value.format(
                failed_step=instance.failed_step or "unknown",
                workflow_name=instance.definition.name,
                workflow_id=instance.workflow_id,
            )

        if action == "notify":
            logger.info(f"[NOTIFY] {value}")
        elif action == "run":
            logger.info(f"[RUN] Would execute: {value}")
            # TODO: Tatsächliche Command-Ausführung
        elif action == "create_pr":
            logger.info(f"[CREATE_PR] Would create PR: {value}")
        elif action == "rollback":
            logger.info(f"[ROLLBACK] Would rollback: {value}")
        else:
            logger.warning(f"Unknown handler action: {action}")

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Statistiken über die WorkflowEngine zurück."""
        instances = list(self._instances.values())

        return {
            "loaded_workflows": len(self._workflow_cache),
            "registered_executors": len(self._executors),
            "registered_conditions": len(self._condition_evaluators),
            "total_instances": len(instances),
            "instances_by_status": {
                status.value: len([i for i in instances if i.status == status])
                for status in WorkflowStatus
            },
        }

    def clear_cache(self) -> None:
        """Leert den Workflow-Cache."""
        self._workflow_cache.clear()
        logger.debug("Workflow cache cleared")


# =============================================================================
# Predefined Workflows (Fallback wenn keine YAML-Dateien)
# =============================================================================


def create_default_workflows() -> dict[str, WorkflowDefinition]:
    """Erstellt Standard-Workflows für häufige Aufgaben.

    Diese werden verwendet wenn keine YAML-Dateien vorhanden sind.
    """
    return {
        "architecture_design": WorkflowDefinition(
            name="architecture_design",
            description="Workflow für Architektur-Design-Aufgaben",
            steps=[
                StepDefinition(
                    name="analyze",
                    agent=AgentRole.ARCHITECT,
                    description="Analyse der Anforderungen",
                ),
                StepDefinition(
                    name="design",
                    agent=AgentRole.ARCHITECT,
                    description="Entwurf der Architektur",
                    depends_on=["analyze"],
                ),
                StepDefinition(
                    name="document",
                    agent=AgentRole.ARCHITECT,
                    description="Dokumentation als ADR",
                    depends_on=["design"],
                ),
            ],
        ),
        "implementation": WorkflowDefinition(
            name="implementation",
            description="Standard-Implementierungs-Workflow",
            steps=[
                StepDefinition(
                    name="implement",
                    agent=AgentRole.IMPLEMENTER,
                    description="Code implementieren",
                ),
                StepDefinition(
                    name="test",
                    agent=AgentRole.TESTER,
                    description="Tests schreiben",
                    depends_on=["implement"],
                ),
                StepDefinition(
                    name="review",
                    agent=AgentRole.REVIEWER,
                    description="Code Review",
                    depends_on=["test"],
                ),
            ],
        ),
        "code_review": WorkflowDefinition(
            name="code_review",
            description="Code Review Workflow",
            steps=[
                StepDefinition(
                    name="review",
                    agent=AgentRole.REVIEWER,
                    description="Code Review durchführen",
                ),
            ],
        ),
        "test_generation": WorkflowDefinition(
            name="test_generation",
            description="Test-Generierungs-Workflow",
            steps=[
                StepDefinition(
                    name="analyze",
                    agent=AgentRole.TESTER,
                    description="Code analysieren",
                ),
                StepDefinition(
                    name="generate_tests",
                    agent=AgentRole.TESTER,
                    description="Tests generieren",
                    depends_on=["analyze"],
                ),
            ],
        ),
        "v2_backtest_implementation": WorkflowDefinition(
            name="v2_backtest_implementation",
            description="V2 Backtest-Core Implementierung",
            v2_mode=True,
            steps=[
                StepDefinition(
                    name="rust_implementation",
                    agent=AgentRole.IMPLEMENTER,
                    description="Rust-Code implementieren",
                    file_filter="rust_core/**/*.rs",
                ),
                StepDefinition(
                    name="python_wrapper",
                    agent=AgentRole.IMPLEMENTER,
                    description="Python-Wrapper implementieren",
                    file_filter="python/bt/**/*.py",
                    depends_on=["rust_implementation"],
                ),
                StepDefinition(
                    name="rust_tests",
                    agent=AgentRole.TESTER,
                    description="Rust-Tests",
                    commands=["cargo test --all", "cargo clippy -- -D warnings"],
                    depends_on=["rust_implementation"],
                    parallel=True,
                ),
                StepDefinition(
                    name="golden_tests",
                    agent=AgentRole.TESTER,
                    description="Golden-File-Tests",
                    commands=["pytest python/bt/tests/test_golden.py -k smoke"],
                    depends_on=["python_wrapper"],
                    parallel=True,
                ),
                StepDefinition(
                    name="review",
                    agent=AgentRole.REVIEWER,
                    description="V2 Code Review",
                    checklist=[
                        "Single FFI Boundary eingehalten",
                        "Determinismus (DEV-Mode)",
                        "Error Contract (Setup→Exception, Runtime→JSON)",
                    ],
                    depends_on=["rust_tests", "golden_tests"],
                ),
            ],
            on_success=[
                {"notify": "V2 implementation complete"},
                {"run": "cargo fmt && pre-commit run -a"},
            ],
            on_failure=[
                {"notify": "V2 implementation failed at step: {failed_step}"},
                {"rollback": True},
            ],
        ),
    }
