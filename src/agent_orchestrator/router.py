"""Task Router für intelligente Agent-Zuweisung.

Routet eingehende Tasks an die passenden Agents basierend auf:
- Task-Typ und Kontext
- V1 vs. V2 Erkennung
- Agent-Verfügbarkeit und Spezialisierung
- Routing-Regeln und Prioritäten
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .v2_detector import V2DetectionResult, V2Detector

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Definierte Agent-Rollen im Omega-Projekt.

    Jede Rolle hat spezifische Verantwortlichkeiten und Instruktionen.
    Referenz: AGENT_ROLES.md
    """

    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    RESEARCHER = "researcher"
    DEVOPS = "devops"
    SAFETY_AUDITOR = "safety_auditor"


# Mapping von Rollen zu ihren primären Instruktions-Dateien
ROLE_INSTRUCTIONS: dict[AgentRole, list[str]] = {
    AgentRole.ARCHITECT: [
        ".github/instructions/architect.instructions.md",
        ".github/instructions/ffi-boundaries.instructions.md",
        ".github/instructions/performance-optimization.instructions.md",
    ],
    AgentRole.IMPLEMENTER: [
        ".github/copilot-instructions.md",
        ".github/instructions/_core/python-standards.instructions.md",
        ".github/instructions/_core/rust-standards.instructions.md",
    ],
    AgentRole.REVIEWER: [
        ".github/instructions/code-review-generic.instructions.md",
        ".github/instructions/_core/security-standards.instructions.md",
    ],
    AgentRole.TESTER: [
        ".github/instructions/tester.instructions.md",
        ".github/instructions/_core/testing-standards.instructions.md",
    ],
    AgentRole.RESEARCHER: [
        ".github/instructions/codexer.instructions.md",
    ],
    AgentRole.DEVOPS: [
        ".github/instructions/devops-core-principles.instructions.md",
        ".github/instructions/github-actions-ci-cd-best-practices.instructions.md",
        ".github/instructions/containerization-docker-best-practices.instructions.md",
    ],
    AgentRole.SAFETY_AUDITOR: [
        ".github/instructions/security-and-owasp.instructions.md",
        ".github/instructions/_domain/trading-safety.instructions.md",
    ],
}

# V2-spezifische Zusatz-Instruktionen
V2_INSTRUCTIONS: list[str] = [
    ".github/instructions/omega-v2-backtest.instructions.md",
    ".github/instructions/ffi-boundaries.instructions.md",
]


@dataclass
class RoutingRule:
    """Eine Regel für Task-Routing.

    Attributes:
        name: Name der Regel (für Logging)
        patterns: Regex-Patterns die matchen müssen
        target_roles: Ziel-Agents für diese Regel
        priority: Priorität (höher = wichtiger)
        requires_v2: Nur für V2-Tasks anwenden
        condition: Optional - zusätzliche Bedingung
    """

    name: str
    patterns: list[str]
    target_roles: list[AgentRole]
    priority: int = 0
    requires_v2: bool = False
    condition: Callable[[dict[str, Any]], bool] | None = None

    def matches(
        self,
        text: str,
        is_v2: bool = False,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Prüfe ob die Regel auf den Text zutrifft."""
        # V2-Anforderung prüfen
        if self.requires_v2 and not is_v2:
            return False

        # Pattern-Matching
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Zusätzliche Bedingung prüfen
                if self.condition:
                    return self.condition(context or {})
                return True

        return False


@dataclass
class RoutingResult:
    """Ergebnis einer Routing-Entscheidung.

    Attributes:
        primary_role: Primär zuständiger Agent
        secondary_roles: Zusätzlich involvierte Agents
        instructions: Liste relevanter Instruktions-Dateien
        matched_rule: Name der angewandten Regel
        v2_detection: V2-Erkennungsergebnis
        routing_context: Zusätzlicher Kontext für den Agent
    """

    primary_role: AgentRole
    secondary_roles: list[AgentRole] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    matched_rule: str = ""
    v2_detection: V2DetectionResult | None = None
    routing_context: dict[str, Any] = field(default_factory=dict)


class TaskRouter:
    """Router für intelligente Task-Zuweisung an Agents.

    Features:
    - Automatische V2-Erkennung
    - Regel-basiertes Routing
    - Fallback-Strategien
    - Instruktions-Aggregation
    """

    def __init__(self) -> None:
        """Initialisiere den Router."""
        self._v2_detector = V2Detector()
        self._rules: list[RoutingRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setze Standard-Routing-Regeln auf."""
        self._rules = [
            # Architektur-Entscheidungen
            RoutingRule(
                name="architecture_decision",
                patterns=[
                    r"ADR",
                    r"architektur",
                    r"architecture",
                    r"design\s*decision",
                    r"refactor.*struct",
                    r"modul.*struktur",
                ],
                target_roles=[AgentRole.ARCHITECT],
                priority=100,
            ),
            # V2 Rust-Implementierung
            RoutingRule(
                name="v2_rust_implementation",
                patterns=[
                    r"\.rs$",
                    r"rust.*crate",
                    r"cargo",
                    r"pyo3",
                    r"maturin",
                ],
                target_roles=[AgentRole.IMPLEMENTER, AgentRole.REVIEWER],
                priority=90,
                requires_v2=True,
            ),
            # V2 Golden-File Tests
            RoutingRule(
                name="v2_golden_tests",
                patterns=[
                    r"golden.*file",
                    r"golden.*test",
                    r"parität",
                    r"parity.*test",
                    r"v1.*v2.*vergleich",
                ],
                target_roles=[AgentRole.TESTER, AgentRole.IMPLEMENTER],
                priority=85,
                requires_v2=True,
            ),
            # FFI-Boundary Änderungen
            RoutingRule(
                name="ffi_changes",
                patterns=[
                    r"ffi",
                    r"python.*rust",
                    r"rust.*python",
                    r"pyfunction",
                    r"pymodule",
                ],
                target_roles=[AgentRole.ARCHITECT, AgentRole.IMPLEMENTER],
                priority=80,
            ),
            # Security-kritische Änderungen
            RoutingRule(
                name="security_critical",
                patterns=[
                    r"security",
                    r"sicherheit",
                    r"authentif",
                    r"authoriz",
                    r"secret",
                    r"credential",
                    r"owasp",
                ],
                target_roles=[AgentRole.SAFETY_AUDITOR, AgentRole.REVIEWER],
                priority=95,
            ),
            # Trading-kritische Änderungen
            RoutingRule(
                name="trading_critical",
                patterns=[
                    r"hf_engine.*core",
                    r"execution",
                    r"risk.*manage",
                    r"position.*siz",
                    r"stop.*loss",
                    r"take.*profit",
                    r"order.*send",
                ],
                target_roles=[AgentRole.SAFETY_AUDITOR, AgentRole.IMPLEMENTER],
                priority=90,
            ),
            # CI/CD und DevOps
            RoutingRule(
                name="cicd_changes",
                patterns=[
                    r"workflow",
                    r"\.github/",
                    r"ci.*cd",
                    r"docker",
                    r"deploy",
                ],
                target_roles=[AgentRole.DEVOPS],
                priority=70,
            ),
            # Test-Generierung
            RoutingRule(
                name="test_generation",
                patterns=[
                    r"test.*schreib",
                    r"write.*test",
                    r"add.*test",
                    r"test.*hinzufüg",
                    r"coverage",
                ],
                target_roles=[AgentRole.TESTER],
                priority=60,
            ),
            # Code Review
            RoutingRule(
                name="code_review",
                patterns=[
                    r"review",
                    r"prüf",
                    r"check.*code",
                    r"qualit",
                ],
                target_roles=[AgentRole.REVIEWER],
                priority=50,
            ),
            # Recherche
            RoutingRule(
                name="research",
                patterns=[
                    r"research",
                    r"evaluat",
                    r"vergleich.*lib",
                    r"best.*practice",
                    r"wie.*funktioniert",
                ],
                target_roles=[AgentRole.RESEARCHER],
                priority=40,
            ),
            # Standard-Implementierung (Fallback)
            RoutingRule(
                name="standard_implementation",
                patterns=[
                    r"implement",
                    r"erstell",
                    r"create",
                    r"hinzufüg",
                    r"add",
                    r"änder",
                    r"change",
                    r"fix",
                    r"bug",
                ],
                target_roles=[AgentRole.IMPLEMENTER],
                priority=10,
            ),
        ]

    def route(
        self,
        task_description: str,
        affected_files: list[str] | None = None,
        additional_context: str = "",
    ) -> RoutingResult:
        """Route einen Task an den passenden Agent.

        Args:
            task_description: Beschreibung des Tasks
            affected_files: Optional - betroffene Dateien
            additional_context: Optional - zusätzlicher Kontext

        Returns:
            RoutingResult mit Ziel-Agent und Instruktionen
        """
        # V2-Erkennung
        v2_result = self._v2_detector.detect(
            file_paths=affected_files or [],
            text=f"{task_description}\n{additional_context}",
        )
        is_v2 = v2_result.is_v2

        # Kombinierter Text für Pattern-Matching
        combined_text = "\n".join(
            [
                task_description,
                additional_context,
                *[str(f) for f in (affected_files or [])],
            ]
        )

        # Regeln nach Priorität sortieren und erste passende finden
        sorted_rules = sorted(self._rules, key=lambda r: -r.priority)

        matched_rule: RoutingRule | None = None
        for rule in sorted_rules:
            if rule.matches(combined_text, is_v2=is_v2):
                matched_rule = rule
                break

        # Fallback auf Implementer
        if not matched_rule:
            logger.warning(
                f"Keine Routing-Regel gefunden für: {task_description[:50]}..."
            )
            matched_rule = RoutingRule(
                name="fallback",
                patterns=[],
                target_roles=[AgentRole.IMPLEMENTER],
            )

        # Instruktionen zusammenstellen
        instructions = self._collect_instructions(
            matched_rule.target_roles[0],
            is_v2=is_v2,
            v2_result=v2_result,
        )

        # Routing-Context erstellen
        routing_context: dict[str, Any] = {
            "is_v2": is_v2,
            "v2_confidence": v2_result.confidence,
            "affected_crates": v2_result.affected_crates,
        }

        if v2_result.involves_ffi:
            routing_context["ffi_warning"] = (
                "FFI-Grenze betroffen - Single FFI Boundary beachten!"
            )
        if v2_result.involves_golden:
            routing_context["golden_warning"] = (
                "Golden-Files betroffen - Breaking Change Review erforderlich!"
            )

        return RoutingResult(
            primary_role=matched_rule.target_roles[0],
            secondary_roles=matched_rule.target_roles[1:],
            instructions=instructions,
            matched_rule=matched_rule.name,
            v2_detection=v2_result,
            routing_context=routing_context,
        )

    def _collect_instructions(
        self,
        primary_role: AgentRole,
        is_v2: bool,
        v2_result: V2DetectionResult,
    ) -> list[str]:
        """Sammle relevante Instruktions-Dateien."""
        instructions: list[str] = []

        # Primäre Rollen-Instruktionen
        role_instructions = ROLE_INSTRUCTIONS.get(primary_role, [])
        instructions.extend(role_instructions)

        # V2-spezifische Instruktionen
        if is_v2:
            instructions.extend(V2_INSTRUCTIONS)

            # Crate-spezifische Instruktionen
            if "execution" in v2_result.affected_crates:
                instructions.append("docs/OMEGA_V2_EXECUTION_MODEL_PLAN.md")
            if "strategy" in v2_result.affected_crates:
                instructions.append("docs/OMEGA_V2_STRATEGIES_PLAN.md")
            if "metrics" in v2_result.affected_crates:
                instructions.append("docs/OMEGA_V2_METRICS_DEFINITION_PLAN.md")

        # Deduplizieren und Reihenfolge beibehalten
        seen: set[str] = set()
        unique_instructions: list[str] = []
        for instr in instructions:
            if instr not in seen:
                seen.add(instr)
                unique_instructions.append(instr)

        return unique_instructions

    def add_rule(self, rule: RoutingRule) -> None:
        """Füge eine benutzerdefinierte Routing-Regel hinzu."""
        self._rules.append(rule)
        logger.debug(f"Routing-Regel hinzugefügt: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """Entferne eine Routing-Regel nach Name."""
        initial_count = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < initial_count

    def list_rules(self) -> list[RoutingRule]:
        """Liste alle Routing-Regeln."""
        return sorted(self._rules, key=lambda r: -r.priority)

    def get_role_instructions(
        self,
        role: AgentRole,
        include_v2: bool = False,
    ) -> list[str]:
        """Hole Instruktionen für eine spezifische Rolle."""
        instructions = list(ROLE_INSTRUCTIONS.get(role, []))
        if include_v2:
            instructions.extend(V2_INSTRUCTIONS)
        return instructions
