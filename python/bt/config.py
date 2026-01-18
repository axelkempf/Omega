"""Config loading and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

DEFAULT_SCHEMA_VERSION = "2.0"
DEFAULT_RUN_MODE = "dev"
DEFAULT_DATA_MODE = "candle"
DEFAULT_EXECUTION_VARIANT = "v2"
DEFAULT_WARMUP_BARS = 500
DEFAULT_ADDITIONAL_SOURCE = "separate_parquet"

_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_schema() -> dict[str, Any]:
    """Load and cache the V2 config JSON schema.

    Returns:
        JSON schema dictionary.
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_path = Path(__file__).parent / "schema" / "v2_config.json"
        with schema_path.open("r", encoding="utf-8") as handle:
            _SCHEMA_CACHE = json.load(handle)
    return _SCHEMA_CACHE


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


def validate_config(config: Mapping[str, Any], *, strict: bool = False) -> None:
    """Validate config structure.

    Args:
        config: Config dictionary to validate.
        strict: If True and jsonschema is available, perform full schema validation.
            Default is False (basic required field checks only).

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    # Basic required field checks (always run)
    required = ["strategy_name", "symbol", "start_date", "end_date"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    if config["start_date"] >= config["end_date"]:
        raise ValueError("start_date must be before end_date")

    # execution_variant validation (CONFIG_SCHEMA_PLAN 5.1)
    if config.get("execution_variant") == "v1_parity":
        if config.get("run_mode", "dev") != "dev":
            raise ValueError("execution_variant 'v1_parity' requires run_mode='dev'")

    # Full schema validation if available and requested
    if strict and HAS_JSONSCHEMA:
        schema = _load_schema()
        try:
            jsonschema.validate(instance=dict(config), schema=schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Schema validation failed: {exc.message}") from exc


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

    profiling = normalized.get("profiling")
    if isinstance(profiling, bool):
        normalized["profiling"] = {"enabled": profiling}
    elif profiling is None:
        normalized.setdefault("profiling", {"enabled": False})

    return normalized
