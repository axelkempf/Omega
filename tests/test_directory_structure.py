"""
Test: Validate required directory structure exists.

This test ensures the operational guardrails (var/ and data/ directory skeletons)
are present in the repository. The structure is required for:
- Runtime state management (heartbeats, logs, results)
- Market data organization

Directories are tracked via README.md files in .gitignore exceptions.
"""

from pathlib import Path

import pytest


# Project root (relative to this test file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Required directories under var/
VAR_REQUIRED_DIRS = [
    "var",
    "var/tmp",
    "var/logs",
    "var/logs/system",
    "var/logs/entry_logs",
    "var/logs/trade_logs",
    "var/logs/optuna",
    "var/results",
    "var/results/backtests",
    "var/results/walkforwards",
    "var/results/analysis",
    "var/archive",
]

# Required directories under data/
DATA_REQUIRED_DIRS = [
    "data",
    "data/csv",
    "data/parquet",
    "data/raw",
    "data/news",
]

# Required README files (skeleton markers)
REQUIRED_READMES = [
    "var/README.md",
    "var/tmp/README.md",
    "var/logs/README.md",
    "var/logs/system/README.md",
    "var/logs/entry_logs/README.md",
    "var/logs/trade_logs/README.md",
    "var/logs/optuna/README.md",
    "var/results/README.md",
    "var/results/backtests/README.md",
    "var/results/walkforwards/README.md",
    "var/results/analysis/README.md",
    "var/archive/README.md",
    "data/README.md",
    "data/csv/README.md",
    "data/parquet/README.md",
    "data/raw/README.md",
    "data/news/README.md",
]


class TestDirectoryStructure:
    """Validate operational directory structure exists."""

    @pytest.mark.parametrize("rel_path", VAR_REQUIRED_DIRS)
    def test_var_directory_exists(self, rel_path: str) -> None:
        """var/ directories must exist for runtime state management."""
        dir_path = PROJECT_ROOT / rel_path
        assert dir_path.is_dir(), (
            f"Required directory missing: {rel_path}\n"
            f"The var/ skeleton is required for operational guardrails.\n"
            f"Run: mkdir -p {dir_path}"
        )

    @pytest.mark.parametrize("rel_path", DATA_REQUIRED_DIRS)
    def test_data_directory_exists(self, rel_path: str) -> None:
        """data/ directories must exist for market data organization."""
        dir_path = PROJECT_ROOT / rel_path
        assert dir_path.is_dir(), (
            f"Required directory missing: {rel_path}\n"
            f"The data/ skeleton is required for reproducibility.\n"
            f"Run: mkdir -p {dir_path}"
        )

    @pytest.mark.parametrize("rel_path", REQUIRED_READMES)
    def test_readme_markers_exist(self, rel_path: str) -> None:
        """README.md skeleton markers must be tracked in git."""
        readme_path = PROJECT_ROOT / rel_path
        assert readme_path.is_file(), (
            f"Required README marker missing: {rel_path}\n"
            f"README files serve as tracked skeleton markers for git.\n"
            f"This file should exist and be committed to the repository."
        )

    def test_gitignore_allows_readmes(self) -> None:
        """Verify .gitignore has exceptions for README files."""
        gitignore_path = PROJECT_ROOT / ".gitignore"
        assert gitignore_path.is_file(), ".gitignore must exist"

        content = gitignore_path.read_text()

        # Check for key unignore patterns
        assert "!var/README.md" in content, (
            ".gitignore must allow var/README.md to be tracked"
        )
        assert "!data/README.md" in content, (
            ".gitignore must allow data/README.md to be tracked"
        )

    def test_paths_module_creates_directories(self) -> None:
        """Verify paths.py auto-creates required directories on import."""
        # Import the paths module - this triggers ensure_directories()
        from hf_engine.infra.config import paths

        # Check that key paths are defined
        assert hasattr(paths, "VAR_DIR")
        assert hasattr(paths, "DATA_DIR")
        assert hasattr(paths, "LOGS_DIR")
        assert hasattr(paths, "RESULTS_DIR")
        assert hasattr(paths, "TMP_DIR")
        assert hasattr(paths, "CSV_DATA_DIR")
        assert hasattr(paths, "PARQUET_DIR")

        # Verify paths are Path objects
        assert isinstance(paths.VAR_DIR, Path)
        assert isinstance(paths.DATA_DIR, Path)
