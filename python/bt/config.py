"""Config loading and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SCHEMA_VERSION = "2.0"
DEFAULT_RUN_MODE = "dev"
DEFAULT_DATA_MODE = "candle"
DEFAULT_EXECUTION_VARIANT = "v2"
DEFAULT_WARMUP_BARS = 500
DEFAULT_ADDITIONAL_SOURCE = "separate_parquet"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config from a JSON file.

    Args:
        path: Path to config JSON.

    Returns:
        Normalized config dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    return _normalize_config(config)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate config structure.

    Args:
        config: Config dictionary to validate.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    required = ["strategy_name", "symbol", "start_date", "end_date"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    if config["start_date"] >= config["end_date"]:
        raise ValueError("start_date must be before end_date")


def _normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply defaults and compatibility shims to a config dict."""
    normalized = copy.deepcopy(dict(config))
    normalized.setdefault("schema_version", DEFAULT_SCHEMA_VERSION)
    normalized.setdefault("run_mode", DEFAULT_RUN_MODE)
    normalized.setdefault("data_mode", DEFAULT_DATA_MODE)
    normalized.setdefault("execution_variant", DEFAULT_EXECUTION_VARIANT)
    normalized.setdefault("warmup_bars", DEFAULT_WARMUP_BARS)

    data_mode = normalized.get("data_mode")
    if isinstance(data_mode, str) and data_mode.lower() == "parquet":
        normalized["data_mode"] = DEFAULT_DATA_MODE

    if "strategy_parameters" not in normalized and "strategy_params" in normalized:
        normalized["strategy_parameters"] = normalized.get("strategy_params")

    timeframes = normalized.get("timeframes")
    if isinstance(timeframes, dict):
        if "additional" not in timeframes and "htf" in timeframes:
            timeframes["additional"] = list(timeframes.get("htf") or [])
        timeframes.setdefault("additional", [])
        timeframes.setdefault("additional_source", DEFAULT_ADDITIONAL_SOURCE)

    return normalized
