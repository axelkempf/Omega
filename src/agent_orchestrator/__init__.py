"""Agent Orchestrator - Zentraler Koordinator für Multi-Agent Workflows.

Der Agent Orchestrator koordiniert mehrere KI-Agenten für komplexe Tasks:
- Automatisches Task-Routing an die richtige Agent-Rolle
- Kontext-Sharing zwischen Agent-Aufrufen
- Workflow-Definitionen für Multi-Step Tasks
- V1 (Live-Engine) und V2 (Backtest-Core) Awareness

Beispiel:
    >>> from src.agent_orchestrator import AgentOrchestrator
    >>> orchestrator = AgentOrchestrator()
    >>> result = orchestrator.execute("Implementiere Feature X")
    >>> print(result.success)
    True
"""

from __future__ import annotations

from .orchestrator import AgentOrchestrator, OrchestratorConfig, TaskResult
from .router import AgentRole, RoutingRule, TaskRouter
from .context import ContextManager, ContextEntry
from .v2_detector import V2Detector
from .workflow import WorkflowDefinition, WorkflowEngine, WorkflowInstance

__all__ = [
    "AgentOrchestrator",
    "OrchestratorConfig",
    "TaskResult",
    "TaskRouter",
    "AgentRole",
    "RoutingRule",
    "WorkflowDefinition",
    "WorkflowInstance",
    "WorkflowEngine",
    "ContextManager",
    "ContextEntry",
    "V2Detector",
]
