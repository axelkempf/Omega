"""Tests for BaseAgent and agent registration system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent_orchestrator.agents.base import (
    AgentCapability,
    AgentContext,
    AgentRegistry,
    AgentResult,
    BaseAgent,
    register_agent,
)

# ============================================================================
# Test AgentCapability Enum
# ============================================================================


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_all_capabilities_exist(self):
        """Verify all expected capabilities are defined."""
        expected = [
            "READ_FILES",
            "WRITE_FILES",
            "EXECUTE_COMMANDS",
            "SEARCH_CODE",
            "ANALYZE_CODE",
            "GENERATE_CODE",
            "RUN_TESTS",
            "ACCESS_NETWORK",
        ]

        for cap_name in expected:
            assert hasattr(AgentCapability, cap_name)

    def test_capability_values_are_strings(self):
        """Test that capability values are usable as strings."""
        assert AgentCapability.READ_FILES.value == "read_files"
        assert AgentCapability.WRITE_FILES.value == "write_files"

    def test_capability_comparison(self):
        """Test capability enum comparison."""
        cap1 = AgentCapability.READ_FILES
        cap2 = AgentCapability.READ_FILES
        cap3 = AgentCapability.WRITE_FILES

        assert cap1 == cap2
        assert cap1 != cap3


# ============================================================================
# Test AgentContext
# ============================================================================


class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_agent_context_creation(self):
        """Test creating an agent context."""
        context = AgentContext(
            task_id="task-123",
            workspace_path=Path("/workspace"),
            v2_scope=True,
            relevant_files=["src/main.py", "tests/test_main.py"],
            metadata={"priority": "high"},
        )

        assert context.task_id == "task-123"
        assert context.workspace_path == Path("/workspace")
        assert context.v2_scope is True
        assert len(context.relevant_files) == 2

    def test_agent_context_minimal(self):
        """Test creating minimal agent context."""
        context = AgentContext(
            task_id="task-456",
            workspace_path=Path("."),
        )

        assert context.v2_scope is False
        assert context.relevant_files == []
        assert context.metadata == {}

    def test_agent_context_immutable_defaults(self):
        """Test that default mutable fields don't share state."""
        context1 = AgentContext(task_id="1", workspace_path=Path("."))
        context2 = AgentContext(task_id="2", workspace_path=Path("."))

        context1.relevant_files.append("file.py")

        assert "file.py" not in context2.relevant_files


# ============================================================================
# Test AgentResult
# ============================================================================


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_agent_result_success(self):
        """Test creating a success result."""
        result = AgentResult(
            success=True,
            output={"files_modified": ["src/module.py"]},
            artifacts=["var/results/output.json"],
            messages=["Successfully updated module"],
        )

        assert result.success is True
        assert "files_modified" in result.output
        assert len(result.artifacts) == 1

    def test_agent_result_failure(self):
        """Test creating a failure result."""
        result = AgentResult(
            success=False,
            error="Failed to compile module",
            output={},
        )

        assert result.success is False
        assert result.error == "Failed to compile module"

    def test_agent_result_minimal(self):
        """Test creating minimal result."""
        result = AgentResult(success=True, output={"status": "ok"})

        assert result.artifacts == []
        assert result.messages == []
        assert result.error is None


# ============================================================================
# Test BaseAgent Abstract Class
# ============================================================================


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore

    def test_subclass_must_implement_execute(self):
        """Test that subclasses must implement execute method."""

        class IncompleteAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

        with pytest.raises(TypeError):
            IncompleteAgent()  # type: ignore

    def test_valid_subclass_implementation(self):
        """Test a valid BaseAgent subclass."""

        class ValidAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "valid_agent"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return {AgentCapability.READ_FILES, AgentCapability.ANALYZE_CODE}

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={"action": action})

        agent = ValidAgent()
        assert agent.name == "valid_agent"
        assert AgentCapability.READ_FILES in agent.capabilities

    def test_can_perform_checks_capabilities(self):
        """Test can_perform method checks capabilities."""

        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return {AgentCapability.READ_FILES}

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        agent = TestAgent()

        assert agent.can_perform(AgentCapability.READ_FILES) is True
        assert agent.can_perform(AgentCapability.WRITE_FILES) is False


# ============================================================================
# Test AgentRegistry
# ============================================================================


class TestAgentRegistry:
    """Tests for AgentRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        AgentRegistry._instance = None
        AgentRegistry._agents = {}
        yield
        AgentRegistry._instance = None
        AgentRegistry._agents = {}

    def test_singleton_pattern(self):
        """Test that AgentRegistry is a singleton."""
        registry1 = AgentRegistry()
        registry2 = AgentRegistry()

        assert registry1 is registry2

    def test_register_agent(self):
        """Test registering an agent."""

        class MockAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "mock_agent"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        registry = AgentRegistry()
        registry.register(MockAgent)

        assert "mock_agent" in registry._agents

    def test_get_agent_by_name(self):
        """Test retrieving agent by name."""

        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        registry = AgentRegistry()
        registry.register(TestAgent)

        agent = registry.get("test_agent")
        assert agent is not None
        assert agent.name == "test_agent"

    def test_get_unknown_agent_returns_none(self):
        """Test that getting unknown agent returns None."""
        registry = AgentRegistry()
        agent = registry.get("nonexistent")
        assert agent is None

    def test_list_agents(self):
        """Test listing all registered agents."""

        class Agent1(BaseAgent):
            @property
            def name(self) -> str:
                return "agent1"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        class Agent2(BaseAgent):
            @property
            def name(self) -> str:
                return "agent2"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        registry = AgentRegistry()
        registry.register(Agent1)
        registry.register(Agent2)

        agents = registry.list_agents()
        assert "agent1" in agents
        assert "agent2" in agents

    def test_find_by_capability(self):
        """Test finding agents by capability."""

        class ReadAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "reader"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return {AgentCapability.READ_FILES}

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        class WriteAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "writer"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return {AgentCapability.WRITE_FILES}

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        registry = AgentRegistry()
        registry.register(ReadAgent)
        registry.register(WriteAgent)

        readers = registry.find_by_capability(AgentCapability.READ_FILES)
        assert len(readers) == 1
        assert readers[0].name == "reader"


# ============================================================================
# Test @register_agent Decorator
# ============================================================================


class TestRegisterAgentDecorator:
    """Tests for @register_agent decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the registry before each test."""
        AgentRegistry._instance = None
        AgentRegistry._agents = {}
        yield
        AgentRegistry._instance = None
        AgentRegistry._agents = {}

    def test_decorator_registers_agent(self):
        """Test that decorator registers the agent class."""

        @register_agent
        class DecoratedAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "decorated"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        registry = AgentRegistry()
        agent = registry.get("decorated")
        assert agent is not None

    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""

        @register_agent
        class MyAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "my_agent"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return set()

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                return AgentResult(success=True, output={})

        assert MyAgent is not None
        instance = MyAgent()
        assert instance.name == "my_agent"


# ============================================================================
# Test Agent Execution
# ============================================================================


class TestAgentExecution:
    """Tests for agent execution flow."""

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing."""

        class SampleAgent(BaseAgent):
            def __init__(self):
                self.execution_count = 0

            @property
            def name(self) -> str:
                return "sample"

            @property
            def capabilities(self) -> set[AgentCapability]:
                return {
                    AgentCapability.READ_FILES,
                    AgentCapability.WRITE_FILES,
                }

            async def execute(
                self, action: str, context: AgentContext, inputs: dict[str, Any]
            ) -> AgentResult:
                self.execution_count += 1

                if action == "fail":
                    return AgentResult(
                        success=False,
                        output={},
                        error="Intentional failure",
                    )

                return AgentResult(
                    success=True,
                    output={
                        "action": action,
                        "inputs": inputs,
                        "task_id": context.task_id,
                    },
                )

        return SampleAgent()

    @pytest.mark.asyncio
    async def test_successful_execution(self, sample_agent):
        """Test successful agent execution."""
        context = AgentContext(
            task_id="test-123",
            workspace_path=Path("/workspace"),
        )

        result = await sample_agent.execute(
            action="process",
            context=context,
            inputs={"key": "value"},
        )

        assert result.success is True
        assert result.output["action"] == "process"
        assert result.output["task_id"] == "test-123"
        assert sample_agent.execution_count == 1

    @pytest.mark.asyncio
    async def test_failed_execution(self, sample_agent):
        """Test failed agent execution."""
        context = AgentContext(
            task_id="test-456",
            workspace_path=Path("/workspace"),
        )

        result = await sample_agent.execute(
            action="fail",
            context=context,
            inputs={},
        )

        assert result.success is False
        assert result.error == "Intentional failure"

    @pytest.mark.asyncio
    async def test_multiple_executions(self, sample_agent):
        """Test multiple sequential executions."""
        context = AgentContext(
            task_id="multi-test",
            workspace_path=Path("/workspace"),
        )

        for i in range(3):
            result = await sample_agent.execute(
                action=f"action_{i}",
                context=context,
                inputs={"iteration": i},
            )
            assert result.success is True

        assert sample_agent.execution_count == 3
