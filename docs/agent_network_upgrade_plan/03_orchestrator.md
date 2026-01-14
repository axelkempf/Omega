# 03 - Agent Orchestrator

> Zentraler Koordinator für Multi-Agent Workflows

**Status:** 🔴 Offen
**Priorität:** Mittel
**Komplexität:** Mittel
**Geschätzter Aufwand:** 1-2 Tage

---

## Objective

Implementiere einen **Agent Orchestrator** der:
- Tasks automatisch an die richtige Agent-Rolle zuweist
- Ergebnisse mehrerer Agents kombiniert
- Kontext zwischen Agent-Aufrufen teilt
- Workflow-Definitionen unterstützt

---

## Current State

### Problem

Aktuell ist die Agent-Koordination **manuell**:

```
[User] ──► "Implementiere Feature X"
           │
           ▼
      [Manuell entscheiden welcher Agent]
           │
           ▼
      [Agent aufrufen]
           │
           ▼
      [Ergebnis manuell prüfen]
           │
           ▼
      [Nächsten Agent manuell aufrufen]
```

### Probleme

1. **Kein automatisches Routing** - User muss wissen welcher Agent
2. **Kein Kontext-Sharing** - Jeder Agent startet bei Null
3. **Keine Workflow-Automation** - Multi-Step Tasks sind manuell
4. **Keine Parallelisierung** - Sequenzielle Ausführung

---

## Target State

### Orchestrator Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Orchestrator                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Task Router                            │   │
│  │  - Analysiert Task-Beschreibung                          │   │
│  │  - Wählt Agent-Rolle basierend auf Keywords              │   │
│  │  - Erstellt Execution Plan                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────┼───────────────────────────┐     │
│  │                           ▼                            │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐│     │
│  │  │Architect │  │Implementer│  │ Reviewer │  │ Tester ││     │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘│     │
│  │       │              │              │            │     │     │
│  │       └──────────────┴──────────────┴────────────┘     │     │
│  │                           │                            │     │
│  │                    Agent Pool                          │     │
│  └───────────────────────────┼───────────────────────────┘     │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Context Manager                          │   │
│  │  - Shared Memory zwischen Agents                         │   │
│  │  - Task History                                          │   │
│  │  - Artifact Storage                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Workflow Definition (YAML)

```yaml
# workflows/feature_implementation.yaml
name: feature_implementation
description: "End-to-end feature implementation workflow"

triggers:
  - keyword: "implement"
  - keyword: "add feature"
  - keyword: "create"

steps:
  - name: architecture
    agent: architect
    input: task_description
    output: implementation_plan
    condition: "complexity >= medium"

  - name: implementation
    agent: implementer
    input:
      - task_description
      - implementation_plan  # from previous step
    output: source_code

  - name: testing
    agent: tester
    input: source_code
    output: test_files
    parallel: true  # can run alongside review

  - name: review
    agent: reviewer
    input: source_code
    output: review_comments
    parallel: true

  - name: fix_issues
    agent: implementer
    input:
      - source_code
      - review_comments
    output: fixed_code
    condition: "review_comments.has_critical"

on_success:
  - notify: "Feature implementation complete"
  - create_pr: true

on_failure:
  - notify: "Feature implementation failed at step: {failed_step}"
  - rollback: true
```

---

## Implementation Plan

### Schritt 1: Orchestrator Core

Erstelle `src/agent_orchestrator/`:

```
src/agent_orchestrator/
├── __init__.py
├── orchestrator.py      # Main orchestrator class
├── router.py            # Task routing logic
├── context.py           # Shared context management
├── workflow.py          # Workflow execution
└── agents/
    ├── __init__.py
    ├── base.py          # Base agent interface
    ├── architect.py
    ├── implementer.py
    ├── reviewer.py
    └── tester.py
```

#### `orchestrator.py`

```python
"""Central agent orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .router import TaskRouter
from .context import ContextManager
from .workflow import WorkflowEngine

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from an agent execution."""

    success: bool
    output: Any
    agent_role: str
    execution_time_ms: float
    artifacts: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    workflow_dir: Path = Path("workflows")
    max_parallel_agents: int = 3
    timeout_seconds: int = 300
    enable_context_sharing: bool = True


class AgentOrchestrator:
    """Coordinates multiple AI agents for complex tasks."""

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()
        self.router = TaskRouter()
        self.context = ContextManager()
        self.workflow_engine = WorkflowEngine(self.config.workflow_dir)

    def execute(self, task: str, context: dict[str, Any] | None = None) -> TaskResult:
        """Execute a task, automatically routing to appropriate agent(s)."""

        # 1. Analyze task and determine workflow
        workflow = self.router.route(task)
        logger.info(f"Routed task to workflow: {workflow.name}")

        # 2. Initialize context
        if context:
            self.context.set_initial(context)
        self.context.set("task_description", task)

        # 3. Execute workflow
        try:
            result = self.workflow_engine.execute(
                workflow=workflow,
                context=self.context,
                timeout=self.config.timeout_seconds
            )
            return TaskResult(
                success=True,
                output=result.final_output,
                agent_role=workflow.primary_agent,
                execution_time_ms=result.execution_time_ms,
                artifacts=result.artifacts
            )
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                agent_role=workflow.primary_agent,
                execution_time_ms=0,
                errors=[str(e)]
            )

    def execute_workflow(
        self,
        workflow_name: str,
        inputs: dict[str, Any]
    ) -> TaskResult:
        """Execute a specific named workflow."""

        workflow = self.workflow_engine.load(workflow_name)
        self.context.set_initial(inputs)

        result = self.workflow_engine.execute(
            workflow=workflow,
            context=self.context,
            timeout=self.config.timeout_seconds
        )

        return TaskResult(
            success=result.success,
            output=result.final_output,
            agent_role=workflow.primary_agent,
            execution_time_ms=result.execution_time_ms,
            artifacts=result.artifacts,
            errors=result.errors
        )
```

#### `router.py`

```python
"""Task routing based on keywords and patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class AgentRole(Enum):
    """Available agent roles."""

    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    RESEARCHER = "researcher"
    DEVOPS = "devops"
    SAFETY_AUDITOR = "safety_auditor"


@dataclass
class RoutingRule:
    """Rule for routing tasks to agents."""

    patterns: list[str]
    agent: AgentRole
    priority: int = 0
    requires_confirmation: bool = False


@dataclass
class Workflow:
    """Workflow definition."""

    name: str
    primary_agent: str
    steps: list[str]


class TaskRouter:
    """Routes tasks to appropriate agent workflows."""

    # Default routing rules
    RULES: list[RoutingRule] = [
        # Architect
        RoutingRule(
            patterns=[r"design", r"architect", r"plan", r"adr", r"structure"],
            agent=AgentRole.ARCHITECT,
            priority=10
        ),
        # Reviewer
        RoutingRule(
            patterns=[r"review", r"check", r"audit", r"verify"],
            agent=AgentRole.REVIEWER,
            priority=9
        ),
        # Tester
        RoutingRule(
            patterns=[r"test", r"coverage", r"pytest", r"unit test"],
            agent=AgentRole.TESTER,
            priority=8
        ),
        # Researcher
        RoutingRule(
            patterns=[r"research", r"evaluate", r"compare", r"benchmark"],
            agent=AgentRole.RESEARCHER,
            priority=7
        ),
        # DevOps
        RoutingRule(
            patterns=[r"deploy", r"ci/cd", r"docker", r"pipeline", r"github action"],
            agent=AgentRole.DEVOPS,
            priority=6
        ),
        # Safety
        RoutingRule(
            patterns=[r"security", r"vulnerability", r"safety", r"owasp"],
            agent=AgentRole.SAFETY_AUDITOR,
            priority=5
        ),
        # Implementer (default)
        RoutingRule(
            patterns=[r"implement", r"fix", r"add", r"create", r"update", r"refactor"],
            agent=AgentRole.IMPLEMENTER,
            priority=0
        ),
    ]

    def route(self, task: str) -> Workflow:
        """Determine the best workflow for a task."""

        task_lower = task.lower()
        matched_rules: list[tuple[RoutingRule, int]] = []

        for rule in self.RULES:
            for pattern in rule.patterns:
                if re.search(pattern, task_lower):
                    matched_rules.append((rule, rule.priority))
                    break

        if not matched_rules:
            # Default to implementer
            return Workflow(
                name="default_implementation",
                primary_agent=AgentRole.IMPLEMENTER.value,
                steps=["implement"]
            )

        # Sort by priority (highest first)
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        best_rule = matched_rules[0][0]

        return self._create_workflow(best_rule.agent, task)

    def _create_workflow(self, agent: AgentRole, task: str) -> Workflow:
        """Create a workflow for the given agent role."""

        workflows = {
            AgentRole.ARCHITECT: Workflow(
                name="architecture_design",
                primary_agent="architect",
                steps=["analyze", "design", "document"]
            ),
            AgentRole.IMPLEMENTER: Workflow(
                name="implementation",
                primary_agent="implementer",
                steps=["implement", "test", "review"]
            ),
            AgentRole.REVIEWER: Workflow(
                name="code_review",
                primary_agent="reviewer",
                steps=["review"]
            ),
            AgentRole.TESTER: Workflow(
                name="test_generation",
                primary_agent="tester",
                steps=["analyze", "generate_tests"]
            ),
            AgentRole.RESEARCHER: Workflow(
                name="research",
                primary_agent="researcher",
                steps=["search", "analyze", "summarize"]
            ),
            AgentRole.DEVOPS: Workflow(
                name="devops",
                primary_agent="devops",
                steps=["plan", "implement", "verify"]
            ),
            AgentRole.SAFETY_AUDITOR: Workflow(
                name="security_audit",
                primary_agent="safety_auditor",
                steps=["scan", "analyze", "report"]
            ),
        }

        return workflows.get(agent, workflows[AgentRole.IMPLEMENTER])
```

#### `context.py`

```python
"""Shared context management between agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ContextEntry:
    """Single entry in the context."""

    key: str
    value: Any
    timestamp: datetime
    source_agent: str | None = None


class ContextManager:
    """Manages shared context between agent executions."""

    def __init__(self, persist_path: Path | None = None):
        self._context: dict[str, ContextEntry] = {}
        self._history: list[ContextEntry] = []
        self._persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load()

    def set(
        self,
        key: str,
        value: Any,
        source_agent: str | None = None
    ) -> None:
        """Set a context value."""

        entry = ContextEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            source_agent=source_agent
        )
        self._context[key] = entry
        self._history.append(entry)

        if self._persist_path:
            self._save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""

        entry = self._context.get(key)
        return entry.value if entry else default

    def set_initial(self, context: dict[str, Any]) -> None:
        """Set initial context from a dictionary."""

        for key, value in context.items():
            self.set(key, value, source_agent="initial")

    def get_all(self) -> dict[str, Any]:
        """Get all context values as a dictionary."""

        return {k: v.value for k, v in self._context.items()}

    def get_history(self, agent: str | None = None) -> list[ContextEntry]:
        """Get context history, optionally filtered by agent."""

        if agent:
            return [e for e in self._history if e.source_agent == agent]
        return self._history.copy()

    def clear(self) -> None:
        """Clear all context."""

        self._context.clear()
        self._history.clear()

        if self._persist_path:
            self._save()

    def _save(self) -> None:
        """Persist context to disk."""

        data = {
            "context": {
                k: {
                    "value": v.value,
                    "timestamp": v.timestamp.isoformat(),
                    "source_agent": v.source_agent
                }
                for k, v in self._context.items()
            }
        }
        self._persist_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        """Load context from disk."""

        data = json.loads(self._persist_path.read_text())
        for key, entry_data in data.get("context", {}).items():
            self._context[key] = ContextEntry(
                key=key,
                value=entry_data["value"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                source_agent=entry_data.get("source_agent")
            )
```

### Schritt 2: Workflow Definitions

Erstelle `workflows/` Ordner mit YAML-Definitionen.

### Schritt 3: CLI Integration

```python
# src/agent_orchestrator/cli.py
"""CLI for agent orchestrator."""

import argparse
from .orchestrator import AgentOrchestrator, OrchestratorConfig


def main():
    parser = argparse.ArgumentParser(description="Agent Orchestrator")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--workflow", help="Specific workflow to execute")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    orchestrator = AgentOrchestrator()

    if args.workflow:
        result = orchestrator.execute_workflow(args.workflow, {"task": args.task})
    else:
        result = orchestrator.execute(args.task)

    if result.success:
        print(f"✅ Task completed by {result.agent_role}")
        print(f"   Time: {result.execution_time_ms:.0f}ms")
    else:
        print(f"❌ Task failed: {result.errors}")


if __name__ == "__main__":
    main()
```

---

## Acceptance Criteria

- [ ] `src/agent_orchestrator/` Modul existiert
- [ ] Task-Routing funktioniert für alle Rollen
- [ ] Context wird zwischen Agents geteilt
- [ ] Mindestens 3 Workflow-Definitionen existieren
- [ ] CLI kann Tasks ausführen
- [ ] Unit Tests für Router und Context

---

## Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Overhead zu hoch | Mittel | Mittel | Lazy Loading, Caching |
| Falsche Routing-Entscheidungen | Hoch | Niedrig | Confirmation für kritische Tasks |
| Context zu groß | Niedrig | Mittel | Size Limits, Cleanup |

---

## Dependencies

- `AGENT_ROLES.md` muss existieren (01_agent_roles.md)
- Rollen-spezifische Instruktionen müssen vorhanden sein

---

## Future Enhancements

1. **Parallel Execution** - Mehrere Agents gleichzeitig
2. **Retry Logic** - Automatische Wiederholung bei Fehlern
3. **Cost Tracking** - Token-Verbrauch pro Workflow
4. **A/B Testing** - Verschiedene Workflows vergleichen
