"""Tests for WorkflowEngine and YAML workflow execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_orchestrator.workflow import (
    StepDefinition,
    StepResult,
    StepStatus,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowInstance,
    WorkflowStatus,
)


# ============================================================================
# Test StepDefinition
# ============================================================================


class TestStepDefinition:
    """Tests for StepDefinition dataclass."""

    def test_step_definition_creation(self):
        """Test basic step definition creation."""
        step = StepDefinition(
            name="test_step",
            agent="architect",
            action="design",
            inputs={"feature": "{{ task.description }}"},
            depends_on=[],
            condition=None,
            timeout=300,
        )

        assert step.name == "test_step"
        assert step.agent == "architect"
        assert step.action == "design"
        assert step.timeout == 300
        assert step.depends_on == []

    def test_step_definition_with_dependencies(self):
        """Test step with dependencies."""
        step = StepDefinition(
            name="implementation",
            agent="implementer",
            action="implement",
            inputs={},
            depends_on=["design", "review"],
            condition="steps.design.status == 'completed'",
            timeout=600,
        )

        assert step.depends_on == ["design", "review"]
        assert step.condition is not None

    def test_step_definition_default_timeout(self):
        """Test default timeout value."""
        step = StepDefinition(
            name="quick_step",
            agent="tester",
            action="test",
            inputs={},
        )

        assert step.timeout == 300  # Default


# ============================================================================
# Test StepResult
# ============================================================================


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_pending(self):
        """Test pending step result."""
        result = StepResult(
            step_name="test_step",
            status=StepStatus.PENDING,
        )

        assert result.status == StepStatus.PENDING
        assert result.output is None
        assert result.error is None
        assert result.duration_ms is None

    def test_step_result_completed(self):
        """Test completed step result."""
        result = StepResult(
            step_name="test_step",
            status=StepStatus.COMPLETED,
            output={"files_created": ["src/module.py"]},
            duration_ms=1500,
        )

        assert result.status == StepStatus.COMPLETED
        assert result.output["files_created"] == ["src/module.py"]
        assert result.duration_ms == 1500

    def test_step_result_failed(self):
        """Test failed step result."""
        result = StepResult(
            step_name="test_step",
            status=StepStatus.FAILED,
            error="Compilation error in module.py",
            duration_ms=500,
        )

        assert result.status == StepStatus.FAILED
        assert "Compilation error" in result.error


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_statuses(self):
        """Test all step status values exist."""
        assert StepStatus.PENDING
        assert StepStatus.RUNNING
        assert StepStatus.COMPLETED
        assert StepStatus.FAILED
        assert StepStatus.SKIPPED


# ============================================================================
# Test WorkflowDefinition
# ============================================================================


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition dataclass."""

    def test_workflow_definition_creation(self):
        """Test workflow definition creation."""
        step1 = StepDefinition(
            name="step1",
            agent="architect",
            action="design",
            inputs={},
        )
        step2 = StepDefinition(
            name="step2",
            agent="implementer",
            action="implement",
            inputs={},
            depends_on=["step1"],
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow for testing",
            version="1.0.0",
            steps=[step1, step2],
            triggers=["manual"],
            context_requirements=["v2_backtest"],
        )

        assert workflow.name == "test_workflow"
        assert len(workflow.steps) == 2
        assert workflow.version == "1.0.0"

    def test_workflow_definition_minimal(self):
        """Test minimal workflow definition."""
        workflow = WorkflowDefinition(
            name="minimal",
            description="Minimal workflow",
            version="0.1.0",
            steps=[],
        )

        assert workflow.triggers == []
        assert workflow.context_requirements == []


# ============================================================================
# Test WorkflowInstance
# ============================================================================


class TestWorkflowInstance:
    """Tests for WorkflowInstance dataclass."""

    def test_workflow_instance_creation(self):
        """Test workflow instance creation."""
        workflow_def = WorkflowDefinition(
            name="test",
            description="Test",
            version="1.0.0",
            steps=[],
        )

        instance = WorkflowInstance(
            workflow_id="wf-123",
            definition=workflow_def,
            status=WorkflowStatus.PENDING,
            step_results={},
            context={"task_id": "task-456"},
        )

        assert instance.workflow_id == "wf-123"
        assert instance.status == WorkflowStatus.PENDING
        assert instance.context["task_id"] == "task-456"


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_all_workflow_statuses(self):
        """Test all workflow status values exist."""
        assert WorkflowStatus.PENDING
        assert WorkflowStatus.RUNNING
        assert WorkflowStatus.COMPLETED
        assert WorkflowStatus.FAILED
        assert WorkflowStatus.CANCELLED


# ============================================================================
# Test WorkflowEngine
# ============================================================================


class TestWorkflowEngine:
    """Tests for WorkflowEngine class."""

    @pytest.fixture
    def engine(self) -> WorkflowEngine:
        """Create a WorkflowEngine instance."""
        return WorkflowEngine()

    @pytest.fixture
    def sample_workflow_yaml(self, tmp_path: Path) -> Path:
        """Create a sample workflow YAML file."""
        yaml_content = """
name: test_workflow
description: Test workflow for unit tests
version: 1.0.0
triggers:
  - manual
context_requirements:
  - v2_backtest
steps:
  - name: design
    agent: architect
    action: design_component
    inputs:
      component: "{{ task.component }}"
    timeout: 300
  - name: implement
    agent: implementer
    action: implement_design
    inputs:
      design: "{{ steps.design.output }}"
    depends_on:
      - design
    timeout: 600
"""
        yaml_file = tmp_path / "test_workflow.yaml"
        yaml_file.write_text(yaml_content)
        return yaml_file

    def test_engine_initialization(self, engine: WorkflowEngine):
        """Test engine initializes with empty state."""
        assert engine._workflows == {}
        assert engine._instances == {}

    def test_load_workflow_from_yaml(
        self, engine: WorkflowEngine, sample_workflow_yaml: Path
    ):
        """Test loading workflow from YAML file."""
        workflow = engine.load_workflow(sample_workflow_yaml)

        assert workflow.name == "test_workflow"
        assert workflow.version == "1.0.0"
        assert len(workflow.steps) == 2
        assert workflow.steps[0].name == "design"
        assert workflow.steps[1].name == "implement"
        assert "design" in workflow.steps[1].depends_on

    def test_load_workflow_registers_it(
        self, engine: WorkflowEngine, sample_workflow_yaml: Path
    ):
        """Test that loading a workflow registers it."""
        workflow = engine.load_workflow(sample_workflow_yaml)

        assert workflow.name in engine._workflows
        assert engine._workflows[workflow.name] == workflow

    def test_load_workflow_invalid_file(self, engine: WorkflowEngine, tmp_path: Path):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            engine.load_workflow(tmp_path / "nonexistent.yaml")

    def test_load_workflow_invalid_yaml(
        self, engine: WorkflowEngine, tmp_path: Path
    ):
        """Test loading invalid YAML raises error."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # Could be YAMLError
            engine.load_workflow(bad_yaml)

    def test_get_workflow_by_name(
        self, engine: WorkflowEngine, sample_workflow_yaml: Path
    ):
        """Test retrieving workflow by name."""
        engine.load_workflow(sample_workflow_yaml)

        workflow = engine.get_workflow("test_workflow")
        assert workflow is not None
        assert workflow.name == "test_workflow"

    def test_get_workflow_not_found(self, engine: WorkflowEngine):
        """Test retrieving non-existent workflow returns None."""
        workflow = engine.get_workflow("nonexistent")
        assert workflow is None

    def test_list_workflows(
        self, engine: WorkflowEngine, sample_workflow_yaml: Path
    ):
        """Test listing all registered workflows."""
        engine.load_workflow(sample_workflow_yaml)

        workflows = engine.list_workflows()
        assert "test_workflow" in workflows


class TestWorkflowEngineExecution:
    """Tests for WorkflowEngine execution functionality."""

    @pytest.fixture
    def engine(self) -> WorkflowEngine:
        """Create a WorkflowEngine instance."""
        return WorkflowEngine()

    @pytest.fixture
    def simple_workflow(self) -> WorkflowDefinition:
        """Create a simple workflow for testing."""
        return WorkflowDefinition(
            name="simple_test",
            description="Simple test workflow",
            version="1.0.0",
            steps=[
                StepDefinition(
                    name="only_step",
                    agent="tester",
                    action="run_tests",
                    inputs={"path": "tests/"},
                )
            ],
        )

    def test_create_instance(
        self, engine: WorkflowEngine, simple_workflow: WorkflowDefinition
    ):
        """Test creating a workflow instance."""
        engine._workflows["simple_test"] = simple_workflow

        instance = engine.create_instance(
            workflow_name="simple_test",
            context={"task_id": "test-123"},
        )

        assert instance is not None
        assert instance.status == WorkflowStatus.PENDING
        assert instance.context["task_id"] == "test-123"
        assert instance.workflow_id in engine._instances

    def test_create_instance_unknown_workflow(self, engine: WorkflowEngine):
        """Test creating instance for unknown workflow."""
        with pytest.raises(ValueError, match="Unknown workflow"):
            engine.create_instance("nonexistent", {})

    def test_get_instance(
        self, engine: WorkflowEngine, simple_workflow: WorkflowDefinition
    ):
        """Test retrieving a workflow instance."""
        engine._workflows["simple_test"] = simple_workflow
        instance = engine.create_instance("simple_test", {})

        retrieved = engine.get_instance(instance.workflow_id)
        assert retrieved == instance

    def test_get_instance_not_found(self, engine: WorkflowEngine):
        """Test retrieving non-existent instance."""
        instance = engine.get_instance("nonexistent-id")
        assert instance is None


class TestWorkflowEngineDependencyResolution:
    """Tests for workflow dependency resolution."""

    @pytest.fixture
    def engine(self) -> WorkflowEngine:
        """Create a WorkflowEngine instance."""
        return WorkflowEngine()

    def test_resolve_step_order_simple(self, engine: WorkflowEngine):
        """Test simple step ordering with dependencies."""
        steps = [
            StepDefinition(name="step1", agent="a", action="x", inputs={}),
            StepDefinition(
                name="step2", agent="b", action="y", inputs={}, depends_on=["step1"]
            ),
            StepDefinition(
                name="step3", agent="c", action="z", inputs={}, depends_on=["step2"]
            ),
        ]

        order = engine._resolve_step_order(steps)

        assert order.index("step1") < order.index("step2")
        assert order.index("step2") < order.index("step3")

    def test_resolve_step_order_parallel(self, engine: WorkflowEngine):
        """Test steps that can run in parallel."""
        steps = [
            StepDefinition(name="step1", agent="a", action="x", inputs={}),
            StepDefinition(name="step2", agent="b", action="y", inputs={}),
            StepDefinition(
                name="step3",
                agent="c",
                action="z",
                inputs={},
                depends_on=["step1", "step2"],
            ),
        ]

        order = engine._resolve_step_order(steps)

        # step3 must come after both step1 and step2
        assert order.index("step1") < order.index("step3")
        assert order.index("step2") < order.index("step3")

    def test_detect_circular_dependency(self, engine: WorkflowEngine):
        """Test detection of circular dependencies."""
        steps = [
            StepDefinition(
                name="step1", agent="a", action="x", inputs={}, depends_on=["step2"]
            ),
            StepDefinition(
                name="step2", agent="b", action="y", inputs={}, depends_on=["step1"]
            ),
        ]

        with pytest.raises(ValueError, match="[Cc]ircular"):
            engine._resolve_step_order(steps)


class TestWorkflowEngineEdgeCases:
    """Edge case tests for WorkflowEngine."""

    @pytest.fixture
    def engine(self) -> WorkflowEngine:
        """Create a WorkflowEngine instance."""
        return WorkflowEngine()

    def test_empty_workflow(self, engine: WorkflowEngine):
        """Test workflow with no steps."""
        workflow = WorkflowDefinition(
            name="empty",
            description="Empty workflow",
            version="1.0.0",
            steps=[],
        )
        engine._workflows["empty"] = workflow

        instance = engine.create_instance("empty", {})
        assert instance is not None

    def test_workflow_with_complex_inputs(
        self, engine: WorkflowEngine, tmp_path: Path
    ):
        """Test workflow with Jinja2 template inputs."""
        yaml_content = """
name: complex_inputs
description: Workflow with complex inputs
version: 1.0.0
steps:
  - name: process
    agent: implementer
    action: process_data
    inputs:
      file_path: "{{ task.base_path }}/{{ task.filename }}"
      options:
        verbose: true
        format: "{{ task.output_format | default('json') }}"
"""
        yaml_file = tmp_path / "complex.yaml"
        yaml_file.write_text(yaml_content)

        workflow = engine.load_workflow(yaml_file)
        assert workflow.steps[0].inputs["file_path"] == (
            "{{ task.base_path }}/{{ task.filename }}"
        )

    def test_multiple_workflow_instances(
        self, engine: WorkflowEngine
    ):
        """Test creating multiple instances of same workflow."""
        workflow = WorkflowDefinition(
            name="multi_instance",
            description="Multi-instance workflow",
            version="1.0.0",
            steps=[
                StepDefinition(name="step", agent="a", action="x", inputs={})
            ],
        )
        engine._workflows["multi_instance"] = workflow

        instance1 = engine.create_instance("multi_instance", {"run": 1})
        instance2 = engine.create_instance("multi_instance", {"run": 2})

        assert instance1.workflow_id != instance2.workflow_id
        assert instance1.context["run"] == 1
        assert instance2.context["run"] == 2
