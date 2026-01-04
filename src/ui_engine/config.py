from __future__ import annotations

import os

from hf_engine.infra.config.paths import (
    BACKTEST_CONFIG_DIR,
    CONFIG_DIR,
    LIVE_CONFIG_DIR,
    PROJECT_ROOT,
    SYSTEM_LOGS_DIR,
)

LOG_DIR = SYSTEM_LOGS_DIR
IS_WINDOWS = os.name == "nt"

__all__ = [
    "BACKTEST_CONFIG_DIR",
    "CONFIG_DIR",
    "LIVE_CONFIG_DIR",
    "PROJECT_ROOT",
    "SYSTEM_LOGS_DIR",
    "LOG_DIR",
    "IS_WINDOWS",
]
