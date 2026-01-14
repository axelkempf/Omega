"""Unit tests for TaskRouter component.

Tests rule matching, agent assignment, and routing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent_orchestrator.router import (
    ROLE_INSTRUCTIONS,
    AgentRole,
    RoutingResult,
    RoutingRule,
    TaskRouter,
)
from src.agent_orchestrator.v2_detector import V2DetectionResult


class TestAgentRole:
    """Test AgentRole enum."""

    def test_all_roles_defined(self) -> None:
        """Verify all expected roles are defined."""
        expected_roles = [
            "ARCHITECT",
            "IMPLEMENTER",
            "REVIEWER",
            "TESTER",
            "RESEARCHER",
            "DEVOPS",
            "SAFETY_AUDITOR",
        ]
        for role_name in expected_roles:
            assert hasattr(AgentRole, role_name), f"Missing role: {role_name}"

    def test_role_values_are_unique(self) -> None:
        """Verify all role values are unique."""
        values = [role.value for role in AgentRole]
        assert len(values) == len(set(values))


class TestRoleInstructions:
    """Test ROLE_INSTRUCTIONS mapping."""

    def test_all_roles_have_instructions(self) -> None:
        """Verify all roles have instruction file mappings."""
        for role in AgentRole:
            assert role in ROLE_INSTRUCTIONS, f"No instructions for {role}"

    def test_instructions_are_valid_paths(self) -> None:
        """Verify instruction paths are .md files."""
        for role, instructions in ROLE_INSTRUCTIONS.items():
            assert len(instructions) > 0, f"No instructions for {role}"
            for instr in instructions:
                assert instr.endswith(".md"), f"Invalid instruction file: {instr}"


class TestRoutingRule:
    """Test RoutingRule dataclass."""

    def test_creation(self) -> None:
        """Test RoutingRule creation."""
        rule = RoutingRule(
            name="test_rule",
            pattern=r"implement.*feature",
            agent_role=AgentRole.IMPLEMENTER,
            priority=10,
        )
        assert rule.name == "test_rule"
        assert rule.pattern == r"implement.*feature"
        assert rule.agent_role == AgentRole.IMPLEMENTER
        assert rule.priority == 10
        assert rule.v2_only is False
        assert rule.conditions == {}

    def test_creation_with_all_fields(self) -> None:
        """Test RoutingRule creation with all fields."""
        rule = RoutingRule(
            name="v2_rust_rule",
            pattern=r"rust_core",
            agent_role=AgentRole.IMPLEMENTER,
            priority=100,
            v2_only=True,
            conditions={"file_extension": ".rs"},
        )
        assert rule.v2_only is True
        assert rule.conditions["file_extension"] == ".rs"

    def test_matches_basic_pattern(self) -> None:
        """Test basic pattern matching."""
        rule = RoutingRule(
            name="impl_rule",
            pattern=r"implement",
            agent_role=AgentRole.IMPLEMENTER,
            priority=10,
        )
        assert rule.matches("Please implement this feature") is True
        assert rule.matches("Please design this feature") is False

    def test_matches_regex_pattern(self) -> None:
        """Test regex pattern matching."""
        rule = RoutingRule(
            name="review_rule",
            pattern=r"(review|check|audit)",
            agent_role=AgentRole.REVIEWER,
            priority=10,
        )
        assert rule.matches("Please review this code") is True
        assert rule.matches("Please check this code") is True
        assert rule.matches("Please audit this code") is True
        assert rule.matches("Please implement this") is False

    def test_matches_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        rule = RoutingRule(
            name="test_rule",
            pattern=r"test",
            agent_role=AgentRole.TESTER,
            priority=10,
        )
        assert rule.matches("Please TEST this") is True
        assert rule.matches("Please Test this") is True
        assert rule.matches("please test this") is True


class TestRoutingResult:
    """Test RoutingResult dataclass."""

    def test_creation(self) -> None:
        """Test RoutingResult creation."""
        result = RoutingResult(
            agent_role=AgentRole.IMPLEMENTER,
            instructions=["impl.md"],
            matched_rule="impl_rule",
            confidence=0.9,
        )
        assert result.agent_role == AgentRole.IMPLEMENTER
        assert result.instructions == ["impl.md"]
        assert result.matched_rule == "impl_rule"
        assert result.confidence == 0.9
        assert result.is_v2_context is False

    def test_default_values(self) -> None:
        """Test default values."""
        result = RoutingResult(
            agent_role=AgentRole.RESEARCHER,
            instructions=[],
            matched_rule=None,
            confidence=0.5,
        )
        assert result.is_v2_context is False
        assert result.additional_context == {}


class TestTaskRouter:
    """Test TaskRouter class."""

    @pytest.fixture
    def router(self) -> TaskRouter:
        """Create a fresh TaskRouter instance."""
        return TaskRouter()

    def test_has_default_rules(self, router: TaskRouter) -> None:
        """Test that default rules are loaded."""
        assert len(router._rules) > 0

    def test_route_implementation_task(self, router: TaskRouter) -> None:
        """Test routing implementation task."""
        result = router.route("Please implement a new feature for data loading")
        assert result.agent_role == AgentRole.IMPLEMENTER
        assert result.confidence > 0.5

    def test_route_review_task(self, router: TaskRouter) -> None:
        """Test routing review task."""
        result = router.route("Please review this pull request")
        assert result.agent_role == AgentRole.REVIEWER

    def test_route_test_task(self, router: TaskRouter) -> None:
        """Test routing test task."""
        result = router.route("Write unit tests for the calculator module")
        assert result.agent_role == AgentRole.TESTER

    def test_route_architecture_task(self, router: TaskRouter) -> None:
        """Test routing architecture task."""
        result = router.route("Design the architecture for the new payment system")
        assert result.agent_role == AgentRole.ARCHITECT

    def test_route_research_task(self, router: TaskRouter) -> None:
        """Test routing research task."""
        result = router.route("Research best practices for caching strategies")
        assert result.agent_role == AgentRole.RESEARCHER

    def test_route_devops_task(self, router: TaskRouter) -> None:
        """Test routing DevOps task."""
        result = router.route("Set up the CI/CD pipeline for the new service")
        assert result.agent_role == AgentRole.DEVOPS

    def test_route_security_task(self, router: TaskRouter) -> None:
        """Test routing security task."""
        result = router.route("Perform a security audit on the authentication module")
        assert result.agent_role == AgentRole.SAFETY_AUDITOR

    def test_route_with_v2_detection(self, router: TaskRouter) -> None:
        """Test routing with V2 context detection."""
        v2_result = V2DetectionResult(
            is_v2_context=True,
            confidence=0.9,
            detected_patterns=["rust_core/"],
            affected_crates=["types", "data"],
            recommended_instructions=["omega-v2-backtest.instructions.md"],
        )
        
        result = router.route(
            "Implement the data loader in rust_core",
            v2_detection=v2_result,
        )
        
        assert result.is_v2_context is True
        assert any("v2" in instr.lower() for instr in result.instructions)

    def test_route_with_files(self, router: TaskRouter) -> None:
        """Test routing with file context."""
        files = [Path("rust_core/crates/types/src/lib.rs")]
        result = router.route("Update the type definitions", files=files)
        
        # Should detect V2 context from files
        assert result.is_v2_context is True

    def test_route_unknown_task_defaults(self, router: TaskRouter) -> None:
        """Test routing unknown task type."""
        result = router.route("Do something vague and undefined")
        
        # Should still return a result with default agent
        assert result.agent_role is not None
        assert result.confidence < 0.8  # Lower confidence for unclear tasks

    def test_add_custom_rule(self, router: TaskRouter) -> None:
        """Test adding custom routing rule."""
        custom_rule = RoutingRule(
            name="custom_ml_rule",
            pattern=r"machine learning|neural network|model training",
            agent_role=AgentRole.RESEARCHER,
            priority=100,  # High priority
        )
        
        router.add_rule(custom_rule)
        
        result = router.route("Train a neural network for prediction")
        assert result.agent_role == AgentRole.RESEARCHER
        assert result.matched_rule == "custom_ml_rule"

    def test_rule_priority_ordering(self, router: TaskRouter) -> None:
        """Test that higher priority rules match first."""
        # Add two rules that could both match
        low_priority = RoutingRule(
            name="low_priority",
            pattern=r"code",
            agent_role=AgentRole.IMPLEMENTER,
            priority=1,
        )
        high_priority = RoutingRule(
            name="high_priority",
            pattern=r"code",
            agent_role=AgentRole.REVIEWER,
            priority=100,
        )
        
        router.add_rule(low_priority)
        router.add_rule(high_priority)
        
        result = router.route("Review this code")
        assert result.matched_rule == "high_priority"

    def test_get_instructions_for_role(self, router: TaskRouter) -> None:
        """Test getting instructions for a role."""
        instructions = router.get_instructions_for_role(AgentRole.IMPLEMENTER)
        assert len(instructions) > 0
        assert all(instr.endswith(".md") for instr in instructions)

    def test_get_instructions_with_v2_context(self, router: TaskRouter) -> None:
        """Test getting instructions with V2 context adds V2 instructions."""
        instructions_v1 = router.get_instructions_for_role(
            AgentRole.IMPLEMENTER,
            is_v2=False,
        )
        instructions_v2 = router.get_instructions_for_role(
            AgentRole.IMPLEMENTER,
            is_v2=True,
        )
        
        # V2 should have more or different instructions
        assert len(instructions_v2) >= len(instructions_v1)


class TestTaskRouterEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def router(self) -> TaskRouter:
        """Create a fresh TaskRouter instance."""
        return TaskRouter()

    def test_route_empty_query(self, router: TaskRouter) -> None:
        """Test routing empty query."""
        result = router.route("")
        assert result.agent_role is not None
        assert result.confidence < 0.5

    def test_route_whitespace_query(self, router: TaskRouter) -> None:
        """Test routing whitespace-only query."""
        result = router.route("   \n\t  ")
        assert result.agent_role is not None

    def test_route_very_long_query(self, router: TaskRouter) -> None:
        """Test routing very long query."""
        long_query = "implement " * 1000
        result = router.route(long_query)
        assert result.agent_role is not None

    def test_route_special_characters(self, router: TaskRouter) -> None:
        """Test routing with special characters."""
        result = router.route("Implement feature: @#$%^&*() [brackets] {braces}")
        assert result.agent_role is not None

    def test_route_unicode_query(self, router: TaskRouter) -> None:
        """Test routing with unicode characters."""
        result = router.route("Implementiere diese Funktion für Umlaute: äöü")
        assert result.agent_role is not None

    def test_multiple_role_keywords(self, router: TaskRouter) -> None:
        """Test query with multiple role keywords."""
        # Contains both "implement" and "review"
        result = router.route("Implement this feature and then review the code")
        assert result.agent_role is not None
        # Should pick one based on priority/order

    def test_v2_only_rule_not_matched_in_v1(self, router: TaskRouter) -> None:
        """Test that V2-only rules don't match in V1 context."""
        v2_rule = RoutingRule(
            name="v2_only_rule",
            pattern=r"crate",
            agent_role=AgentRole.IMPLEMENTER,
            priority=100,
            v2_only=True,
        )
        router.add_rule(v2_rule)
        
        # Without V2 context, should not match this rule
        result = router.route("Update the crate")
        # Either doesn't match or matches a different rule
        if result.matched_rule == "v2_only_rule":
            assert result.is_v2_context is True

    def test_remove_rule(self, router: TaskRouter) -> None:
        """Test removing a rule."""
        initial_count = len(router._rules)
        
        custom_rule = RoutingRule(
            name="temp_rule",
            pattern=r"temporary",
            agent_role=AgentRole.TESTER,
            priority=50,
        )
        router.add_rule(custom_rule)
        assert len(router._rules) == initial_count + 1
        
        router.remove_rule("temp_rule")
        assert len(router._rules) == initial_count

    def test_clear_rules(self, router: TaskRouter) -> None:
        """Test clearing all rules."""
        router.clear_rules()
        assert len(router._rules) == 0
        
        # Routing should still work with fallback
        result = router.route("Do something")
        assert result.agent_role is not None
