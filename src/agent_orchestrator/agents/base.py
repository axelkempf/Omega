"""Base Agent für den Omega Agent Orchestrator.

Dieses Modul definiert die abstrakte Basisklasse für alle Agenten im Orchestrator.
Jeder spezialisierte Agent (Architect, Implementer, Reviewer, etc.) erbt von BaseAgent.

Struktur:
- BaseAgent: Abstrakte Basisklasse mit execute() Methode
- AgentContext: Kontextinformationen für Agent-Ausführung
- AgentResult: Strukturiertes Ergebnis einer Agent-Ausführung
- AgentCapability: Enum für Agent-Fähigkeiten
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Fähigkeiten die ein Agent haben kann.
    
    Wird verwendet für Capability-basiertes Routing und
    zur Validierung ob ein Agent einen Task ausführen kann.
    """
    
    # Code-bezogene Fähigkeiten
    CODE_READ = auto()
    CODE_WRITE = auto()
    CODE_REVIEW = auto()
    CODE_REFACTOR = auto()
    
    # Test-bezogene Fähigkeiten
    TEST_GENERATE = auto()
    TEST_EXECUTE = auto()
    TEST_ANALYZE = auto()
    
    # Dokumentations-Fähigkeiten
    DOC_READ = auto()
    DOC_WRITE = auto()
    DOC_REVIEW = auto()
    
    # Architektur-Fähigkeiten
    ARCHITECTURE_DESIGN = auto()
    ARCHITECTURE_REVIEW = auto()
    ADR_CREATE = auto()
    
    # Recherche-Fähigkeiten
    WEB_SEARCH = auto()
    DEPENDENCY_ANALYZE = auto()
    CONTEXT7_QUERY = auto()
    
    # DevOps-Fähigkeiten
    CI_CD_CONFIGURE = auto()
    DOCKER_BUILD = auto()
    DEPLOYMENT_MANAGE = auto()
    
    # Sicherheits-Fähigkeiten
    SECURITY_AUDIT = auto()
    PROMPT_REVIEW = auto()
    
    # V2-spezifische Fähigkeiten
    V2_RUST_CODE = auto()
    V2_FFI_DESIGN = auto()
    V2_GOLDEN_TEST = auto()


class AgentStatus(Enum):
    """Status eines Agenten."""
    
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()
    DISABLED = auto()


@dataclass
class AgentContext:
    """Kontextinformationen für eine Agent-Ausführung.
    
    Enthält alle Informationen die ein Agent benötigt um einen
    Task auszuführen, inkl. Workspace-Info, V2-Status, und
    shared Context vom Orchestrator.
    """
    
    # Task-Informationen
    task_id: str
    task_description: str
    
    # Workspace-Informationen
    workspace_root: Path
    current_file: Path | None = None
    selected_files: list[Path] = field(default_factory=list)
    
    # V2-Kontext
    is_v2_context: bool = False
    v2_crate: str | None = None
    v2_instructions: list[str] = field(default_factory=list)
    
    # Shared Context vom Orchestrator
    shared_context: dict[str, Any] = field(default_factory=dict)
    
    # Vorherige Ergebnisse in Workflow
    previous_results: list[AgentResult] = field(default_factory=list)
    
    # Konfiguration
    config: dict[str, Any] = field(default_factory=dict)
    
    # Zeitlimits
    timeout_seconds: int = 300
    
    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_relevant_files(self, patterns: list[str] | None = None) -> list[Path]:
        """Hole relevante Dateien basierend auf Patterns.
        
        Args:
            patterns: Glob-Patterns zum Filtern (None = alle)
            
        Returns:
            Liste der relevanten Dateien
        """
        if not patterns:
            if self.current_file:
                return [self.current_file] + self.selected_files
            return self.selected_files
        
        result = []
        for pattern in patterns:
            result.extend(self.workspace_root.glob(pattern))
        return result
    
    def get_v2_paths(self) -> dict[str, Path]:
        """Hole V2-spezifische Pfade.
        
        Returns:
            Dict mit rust_core und python/bt Pfaden
        """
        return {
            "rust_core": self.workspace_root / "rust_core",
            "python_bt": self.workspace_root / "python" / "bt",
            "crates": self.workspace_root / "rust_core" / "crates",
            "golden_tests": self.workspace_root / "python" / "bt" / "tests" / "golden",
        }


@dataclass
class AgentResult:
    """Ergebnis einer Agent-Ausführung.
    
    Strukturiertes Ergebnis das vom Orchestrator verarbeitet
    und an nachfolgende Agenten weitergegeben werden kann.
    """
    
    # Identifikation
    agent_name: str
    task_id: str
    
    # Status
    success: bool
    
    # Zeitstempel
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    
    # Outputs
    output: str = ""
    structured_output: dict[str, Any] = field(default_factory=dict)
    
    # Erzeugte/geänderte Dateien
    files_created: list[Path] = field(default_factory=list)
    files_modified: list[Path] = field(default_factory=list)
    files_deleted: list[Path] = field(default_factory=list)
    
    # Fehlerinformationen
    error: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    
    # Für nachfolgende Agenten
    next_steps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float | None:
        """Berechne Ausführungsdauer in Sekunden."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_context_entry(self) -> dict[str, Any]:
        """Konvertiere zu Context-Entry für nachfolgende Agenten.
        
        Returns:
            Dict das als shared_context verwendet werden kann
        """
        return {
            "agent": self.agent_name,
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "structured_output": self.structured_output,
            "files_created": [str(f) for f in self.files_created],
            "files_modified": [str(f) for f in self.files_modified],
            "next_steps": self.next_steps,
            "recommendations": self.recommendations,
            "duration": self.duration_seconds,
        }


class BaseAgent(ABC):
    """Abstrakte Basisklasse für alle Agenten.
    
    Jeder spezialisierte Agent muss von dieser Klasse erben
    und die execute() Methode implementieren.
    
    Beispiel:
        class ImplementerAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "implementer"
            
            @property
            def capabilities(self) -> set[AgentCapability]:
                return {
                    AgentCapability.CODE_WRITE,
                    AgentCapability.CODE_REFACTOR,
                }
            
            async def execute(self, context: AgentContext) -> AgentResult:
                # Implementierung...
                pass
    """
    
    def __init__(self) -> None:
        """Initialisiere den Agent."""
        self._status = AgentStatus.IDLE
        self._current_task: str | None = None
        self._execution_count = 0
        self._error_count = 0
        self._hooks: dict[str, list[Callable]] = {
            "pre_execute": [],
            "post_execute": [],
            "on_error": [],
        }
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Name des Agenten.
        
        Returns:
            Name wie "architect", "implementer", etc.
        """
        ...
    
    @property
    @abstractmethod
    def capabilities(self) -> set[AgentCapability]:
        """Fähigkeiten dieses Agenten.
        
        Returns:
            Set der AgentCapability Werte
        """
        ...
    
    @property
    def description(self) -> str:
        """Beschreibung des Agenten.
        
        Returns:
            Kurze Beschreibung der Verantwortlichkeiten
        """
        return f"Agent: {self.name}"
    
    @property
    def status(self) -> AgentStatus:
        """Aktueller Status des Agenten."""
        return self._status
    
    @property
    def is_available(self) -> bool:
        """Prüfe ob Agent verfügbar ist."""
        return self._status == AgentStatus.IDLE
    
    @property
    def stats(self) -> dict[str, Any]:
        """Statistiken über den Agenten.
        
        Returns:
            Dict mit execution_count, error_count, etc.
        """
        return {
            "name": self.name,
            "status": self._status.name,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "current_task": self._current_task,
            "capabilities": [c.name for c in self.capabilities],
        }
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Führe den Agent-Task aus.
        
        Dies ist die Hauptmethode die jeder Agent implementieren muss.
        
        Args:
            context: AgentContext mit allen notwendigen Informationen
            
        Returns:
            AgentResult mit Ergebnis der Ausführung
        """
        ...
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Prüfe ob Agent eine bestimmte Fähigkeit hat.
        
        Args:
            capability: Die zu prüfende Fähigkeit
            
        Returns:
            True wenn Agent die Fähigkeit hat
        """
        return capability in self.capabilities
    
    def can_handle_task(self, required_capabilities: set[AgentCapability]) -> bool:
        """Prüfe ob Agent alle erforderlichen Fähigkeiten hat.
        
        Args:
            required_capabilities: Set der benötigten Fähigkeiten
            
        Returns:
            True wenn alle Fähigkeiten vorhanden
        """
        return required_capabilities.issubset(self.capabilities)
    
    def register_hook(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Registriere einen Hook für ein Event.
        
        Args:
            event: Event-Name (pre_execute, post_execute, on_error)
            callback: Callback-Funktion
        """
        if event in self._hooks:
            self._hooks[event].append(callback)
        else:
            logger.warning(f"Unknown hook event: {event}")
    
    async def run(self, context: AgentContext) -> AgentResult:
        """Führe Agent mit Lifecycle-Management aus.
        
        Diese Methode wrapped execute() mit:
        - Status-Management
        - Hook-Ausführung
        - Error-Handling
        - Statistik-Tracking
        
        Args:
            context: AgentContext für die Ausführung
            
        Returns:
            AgentResult mit Ergebnis
        """
        self._status = AgentStatus.BUSY
        self._current_task = context.task_id
        
        # Pre-execute Hooks
        for hook in self._hooks["pre_execute"]:
            try:
                hook(self, context)
            except Exception as e:
                logger.warning(f"Pre-execute hook error: {e}")
        
        result: AgentResult
        try:
            logger.info(
                f"Agent '{self.name}' starting task '{context.task_id}'"
            )
            
            result = await self.execute(context)
            result.completed_at = datetime.now()
            
            self._execution_count += 1
            
            if not result.success:
                self._error_count += 1
                # On-error Hooks
                for hook in self._hooks["on_error"]:
                    try:
                        hook(self, context, result)
                    except Exception as e:
                        logger.warning(f"On-error hook error: {e}")
            
            logger.info(
                f"Agent '{self.name}' completed task '{context.task_id}' "
                f"(success={result.success})"
            )
            
        except Exception as e:
            logger.exception(f"Agent '{self.name}' failed with exception")
            self._error_count += 1
            
            result = AgentResult(
                agent_name=self.name,
                task_id=context.task_id,
                success=False,
                error=str(e),
                error_details={"exception_type": type(e).__name__},
                completed_at=datetime.now(),
            )
            
            # On-error Hooks
            for hook in self._hooks["on_error"]:
                try:
                    hook(self, context, result)
                except Exception as hook_error:
                    logger.warning(f"On-error hook error: {hook_error}")
        
        finally:
            self._status = AgentStatus.IDLE
            self._current_task = None
        
        # Post-execute Hooks
        for hook in self._hooks["post_execute"]:
            try:
                hook(self, context, result)
            except Exception as e:
                logger.warning(f"Post-execute hook error: {e}")
        
        return result
    
    def validate_context(self, context: AgentContext) -> list[str]:
        """Validiere den Context vor Ausführung.
        
        Kann von Subklassen überschrieben werden für
        spezifische Validierungen.
        
        Args:
            context: Zu validierender Context
            
        Returns:
            Liste von Validierungsfehlern (leer wenn OK)
        """
        errors = []
        
        if not context.task_id:
            errors.append("task_id is required")
        
        if not context.workspace_root.exists():
            errors.append(f"workspace_root does not exist: {context.workspace_root}")
        
        return errors
    
    def __repr__(self) -> str:
        """String-Repräsentation des Agenten."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"status={self._status.name}, "
            f"capabilities={len(self.capabilities)})"
        )


class AgentRegistry:
    """Registry für alle verfügbaren Agenten.
    
    Singleton-Pattern für zentrale Agent-Verwaltung.
    """
    
    _instance: AgentRegistry | None = None
    
    def __new__(cls) -> AgentRegistry:
        """Singleton-Pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: dict[str, BaseAgent] = {}
        return cls._instance
    
    def register(self, agent: BaseAgent) -> None:
        """Registriere einen Agenten.
        
        Args:
            agent: Der zu registrierende Agent
        """
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def unregister(self, name: str) -> None:
        """Entferne einen Agenten.
        
        Args:
            name: Name des zu entfernenden Agenten
        """
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Unregistered agent: {name}")
    
    def get(self, name: str) -> BaseAgent | None:
        """Hole Agent nach Name.
        
        Args:
            name: Name des Agenten
            
        Returns:
            Agent oder None wenn nicht gefunden
        """
        return self._agents.get(name)
    
    def get_by_capability(
        self,
        capability: AgentCapability,
    ) -> list[BaseAgent]:
        """Finde Agenten mit bestimmter Fähigkeit.
        
        Args:
            capability: Die gesuchte Fähigkeit
            
        Returns:
            Liste der Agenten mit dieser Fähigkeit
        """
        return [
            agent for agent in self._agents.values()
            if agent.has_capability(capability)
        ]
    
    def get_available(self) -> list[BaseAgent]:
        """Hole alle verfügbaren Agenten.
        
        Returns:
            Liste der Agenten mit Status IDLE
        """
        return [
            agent for agent in self._agents.values()
            if agent.is_available
        ]
    
    def all(self) -> dict[str, BaseAgent]:
        """Hole alle registrierten Agenten.
        
        Returns:
            Dict aller Agenten
        """
        return dict(self._agents)
    
    @classmethod
    def reset(cls) -> None:
        """Reset der Registry (für Tests)."""
        cls._instance = None


def get_agent_registry() -> AgentRegistry:
    """Hole die globale Agent-Registry.
    
    Returns:
        AgentRegistry Singleton-Instanz
    """
    return AgentRegistry()


# Convenience-Decorator für Agent-Registrierung
def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Decorator zum automatischen Registrieren eines Agenten.
    
    Beispiel:
        @register_agent
        class MyAgent(BaseAgent):
            ...
    
    Args:
        cls: Die Agent-Klasse
        
    Returns:
        Unveränderte Klasse (nach Registrierung)
    """
    registry = get_agent_registry()
    registry.register(cls())
    return cls
