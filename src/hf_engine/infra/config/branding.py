"""
Central branding and application metadata for the Omega Trading Stack.

This module provides a single source of truth for display names, version info,
and repository URLs used across UI, logging, documentation, and artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

# Application Branding
APP_NAME = os.getenv("APP_NAME", "Omega")
APP_DISPLAY_NAME = "Omega Trading Stack"
APP_DESCRIPTION = (
    "Python-based trading stack with MetaTrader 5 live adapter, "
    "event-driven backtest/optimizer, and FastAPI UI"
)

# Repository Information
REPO_OWNER = "axelkempf"
REPO_NAME = "Omega"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

# Version (can be enhanced to read from pyproject.toml dynamically)
APP_VERSION = "1.2.0"


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Returns the parent of the src directory where this module is located.
    """
    return Path(__file__).parent.parent.parent.parent.parent
