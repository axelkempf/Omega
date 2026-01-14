"""Agent adapters package.

Enthält Adapter für die verschiedenen Agent-Rollen:
- BaseAgent: Abstrakte Basis für alle Agents
- ArchitectAgent: System-Design und ADRs
- ImplementerAgent: Code-Implementierung
- ReviewerAgent: Code-Review
- TesterAgent: Test-Generierung
- ResearcherAgent: Bibliotheks-Recherche
- DevOpsAgent: CI/CD und Deployment
- SafetyAuditorAgent: Sicherheits-Reviews
"""

from __future__ import annotations

from .base import AgentContext, AgentResult, BaseAgent

__all__ = [
    "BaseAgent",
    "AgentContext",
    "AgentResult",
]
