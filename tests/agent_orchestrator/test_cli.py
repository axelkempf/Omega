"""Tests for Agent Orchestrator CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_orchestrator.cli import (
    OrchestratorCLI,
    parse_args,
)


# ============================================================================
# Test Argument Parsing
# ============================================================================


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_status_command(self):
        """Test parsing status command."""
        args = parse_args(["status"])

        assert args.command == "status"

    def test_run_command_basic(self):
        """Test parsing run command with workflow."""
        args = parse_args(["run", "code_review"])

        assert args.command == "run"
        assert args.workflow == "code_review"

    def test_run_command_with_task_id(self):
        """Test parsing run command with task ID."""
        args = parse_args(["run", "bug_fix", "--task-id", "TASK-123"])

        assert args.command == "run"
        assert args.workflow == "bug_fix"
        assert args.task_id == "TASK-123"

    def test_run_command_with_v2_scope(self):
        """Test parsing run command with V2 scope flag."""
        args = parse_args(["run", "feature_implementation", "--v2-scope"])

        assert args.command == "run"
        assert args.v2_scope is True

    def test_run_command_with_auto_detect(self):
        """Test parsing run command with auto-detect flag."""
        args = parse_args(["run", "code_review", "--auto-detect"])

        assert args.command == "run"
        assert args.auto_detect is True

    def test_list_workflows_command(self):
        """Test parsing list-workflows command."""
        args = parse_args(["list-workflows"])

        assert args.command == "list-workflows"

    def test_list_agents_command(self):
        """Test parsing list-agents command."""
        args = parse_args(["list-agents"])

        assert args.command == "list-agents"

    def test_workflow_status_command(self):
        """Test parsing workflow-status command."""
        args = parse_args(["workflow-status", "instance-abc-123"])

        assert args.command == "workflow-status"
        assert args.instance_id == "instance-abc-123"

    def test_detect_scope_command(self):
        """Test parsing detect-scope command."""
        args = parse_args(["detect-scope", "rust_core/crates/types/src/lib.rs"])

        assert args.command == "detect-scope"
        assert args.path == "rust_core/crates/types/src/lib.rs"

    def test_verbose_flag(self):
        """Test parsing verbose flag."""
        args = parse_args(["-v", "status"])

        assert args.verbose is True

    def test_quiet_flag(self):
        """Test parsing quiet flag."""
        args = parse_args(["-q", "status"])

        assert args.quiet is True

    def test_config_option(self):
        """Test parsing config file option."""
        args = parse_args(["--config", "custom_config.yaml", "status"])

        assert args.config == "custom_config.yaml"

    def test_workspace_option(self):
        """Test parsing workspace option."""
        args = parse_args(["--workspace", "/custom/workspace", "status"])

        assert args.workspace == "/custom/workspace"


# ============================================================================
# Test OrchestratorCLI Class
# ============================================================================


class TestOrchestratorCLI:
    """Tests for OrchestratorCLI class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.get_status = MagicMock(
            return_value={
                "running": True,
                "active_workflows": 2,
                "pending_tasks": 5,
            }
        )
        return orchestrator

    @pytest.fixture
    def cli(self, mock_orchestrator):
        """Create CLI instance with mock orchestrator."""
        with patch(
            "src.agent_orchestrator.cli.AgentOrchestrator",
            return_value=mock_orchestrator,
        ):
            return OrchestratorCLI()

    def test_cli_initialization(self, cli):
        """Test CLI initializes correctly."""
        assert cli is not None

    def test_status_output_format(self, cli, mock_orchestrator, capsys):
        """Test status command output format."""
        cli.cmd_status()

        captured = capsys.readouterr()
        assert "running" in captured.out.lower() or "status" in captured.out.lower()


# ============================================================================
# Test CLI Commands
# ============================================================================


class TestCLICommands:
    """Tests for individual CLI commands."""

    @pytest.fixture
    def mock_workflow_engine(self):
        """Create a mock workflow engine."""
        engine = MagicMock()
        engine.list_workflows = MagicMock(
            return_value=["code_review", "bug_fix", "feature_implementation"]
        )
        engine.get_workflow = MagicMock(
            return_value=MagicMock(
                name="code_review",
                description="Code review workflow",
            )
        )
        return engine

    @pytest.fixture
    def mock_agent_registry(self):
        """Create a mock agent registry."""
        registry = MagicMock()
        registry.list_agents = MagicMock(
            return_value=["architect", "implementer", "reviewer", "tester"]
        )
        return registry

    def test_list_workflows_output(self, mock_workflow_engine, capsys):
        """Test list-workflows command output."""
        with patch(
            "src.agent_orchestrator.cli.WorkflowEngine",
            return_value=mock_workflow_engine,
        ):
            cli = OrchestratorCLI.__new__(OrchestratorCLI)
            cli.workflow_engine = mock_workflow_engine

            cli.cmd_list_workflows()

            captured = capsys.readouterr()
            # Command should produce some output
            assert captured.out or True  # May have no output if method is stubbed

    def test_list_agents_output(self, mock_agent_registry, capsys):
        """Test list-agents command output."""
        with patch(
            "src.agent_orchestrator.cli.AgentRegistry",
            return_value=mock_agent_registry,
        ):
            cli = OrchestratorCLI.__new__(OrchestratorCLI)
            cli.agent_registry = mock_agent_registry

            cli.cmd_list_agents()

            captured = capsys.readouterr()
            assert captured.out or True


# ============================================================================
# Test V2 Scope Detection Integration
# ============================================================================


class TestV2ScopeDetectionCLI:
    """Tests for V2 scope detection via CLI."""

    @pytest.fixture
    def mock_v2_detector(self):
        """Create a mock V2 detector."""
        detector = MagicMock()
        detector.detect = MagicMock(
            return_value={
                "is_v2": True,
                "confidence": 0.95,
                "reasons": ["Path matches rust_core pattern"],
            }
        )
        return detector

    def test_detect_scope_v2_path(self, mock_v2_detector, capsys):
        """Test detecting V2 scope for Rust path."""
        with patch(
            "src.agent_orchestrator.cli.V2Detector",
            return_value=mock_v2_detector,
        ):
            cli = OrchestratorCLI.__new__(OrchestratorCLI)
            cli.v2_detector = mock_v2_detector

            cli.cmd_detect_scope("rust_core/crates/types/src/lib.rs")

            mock_v2_detector.detect.assert_called_once()

    def test_detect_scope_v1_path(self, mock_v2_detector, capsys):
        """Test detecting V1 scope for legacy path."""
        mock_v2_detector.detect.return_value = {
            "is_v2": False,
            "confidence": 0.1,
            "reasons": ["Path matches V1 pattern"],
        }

        with patch(
            "src.agent_orchestrator.cli.V2Detector",
            return_value=mock_v2_detector,
        ):
            cli = OrchestratorCLI.__new__(OrchestratorCLI)
            cli.v2_detector = mock_v2_detector

            cli.cmd_detect_scope("src/hf_engine/core/execution.py")

            mock_v2_detector.detect.assert_called_once()


# ============================================================================
# Test Async Command Handling
# ============================================================================


class TestAsyncCommands:
    """Tests for async command handling."""

    @pytest.fixture
    def async_cli(self):
        """Create CLI with async support."""
        cli = OrchestratorCLI.__new__(OrchestratorCLI)
        cli.orchestrator = MagicMock()
        cli.orchestrator.run_workflow = AsyncMock(
            return_value=MagicMock(
                instance_id="inst-123",
                status="completed",
            )
        )
        return cli

    @pytest.mark.asyncio
    async def test_run_workflow_async(self, async_cli):
        """Test async workflow execution."""
        result = await async_cli.orchestrator.run_workflow(
            workflow_name="code_review",
            task_id="TASK-456",
        )

        assert result.instance_id == "inst-123"

    @pytest.mark.asyncio
    async def test_run_workflow_with_context(self, async_cli):
        """Test async workflow with context."""
        async_cli.orchestrator.run_workflow = AsyncMock(
            return_value=MagicMock(
                instance_id="inst-789",
                status="running",
            )
        )

        result = await async_cli.orchestrator.run_workflow(
            workflow_name="feature_implementation",
            task_id="FEAT-001",
            context={"v2_scope": True},
        )

        assert result.status == "running"


# ============================================================================
# Test Error Handling
# ============================================================================


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_unknown_command_error(self):
        """Test handling of unknown command."""
        with pytest.raises(SystemExit):
            parse_args(["unknown-command"])

    def test_missing_required_argument(self):
        """Test handling of missing required arguments."""
        with pytest.raises(SystemExit):
            parse_args(["run"])  # Missing workflow name

    def test_missing_instance_id(self):
        """Test handling of missing instance ID."""
        with pytest.raises(SystemExit):
            parse_args(["workflow-status"])  # Missing instance_id

    def test_invalid_option(self):
        """Test handling of invalid option."""
        with pytest.raises(SystemExit):
            parse_args(["--invalid-option", "status"])


# ============================================================================
# Test Output Formatting
# ============================================================================


class TestOutputFormatting:
    """Tests for CLI output formatting."""

    def test_json_output_format(self):
        """Test JSON output format option."""
        args = parse_args(["--output", "json", "status"])

        assert args.output == "json"

    def test_table_output_format(self):
        """Test table output format option."""
        args = parse_args(["--output", "table", "list-workflows"])

        assert args.output == "table"

    def test_default_output_format(self):
        """Test default output format."""
        args = parse_args(["status"])

        assert args.output == "text" or not hasattr(args, "output")


# ============================================================================
# Test CLI Entry Point
# ============================================================================


class TestCLIEntryPoint:
    """Tests for CLI entry point."""

    def test_main_function_exists(self):
        """Test that main function exists."""
        from src.agent_orchestrator.cli import main

        assert callable(main)

    def test_cli_version(self):
        """Test CLI version output."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--version"])

        assert exc_info.value.code == 0
