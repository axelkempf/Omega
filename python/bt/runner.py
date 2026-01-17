"""High-level backtest runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import load_config, validate_config, _normalize_config
from .output import write_artifacts


def run_backtest(
    config_path: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a V2 backtest.

    Args:
        config_path: Path to config JSON file.
        config_dict: Config as dictionary (alternative to config_path).
        output_dir: Directory for output artifacts.

    Returns:
        Backtest result as a dictionary.
    """
    if config_dict is not None:
        config = _normalize_config(config_dict)
    elif config_path is not None:
        config = load_config(config_path)
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    validate_config(config)

    config_json = json.dumps(config)
    omega_module = _load_ffi()
    result_json = omega_module.run_backtest(config_json)
    result = json.loads(result_json)

    if output_dir is not None or result.get("ok", False):
        run_id = _extract_run_id(result)
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("var/results/backtests") / str(run_id)
        )
        write_artifacts(result, resolved_output_dir)

    return result


def _load_ffi() -> Any:
    """Import the Rust FFI module with a compatible fallback."""
    try:
        import omega_bt as omega_module
        if not hasattr(omega_module, "run_backtest"):
            try:
                from omega_bt import _native as omega_module
            except Exception:
                pass
    except ImportError:
        try:
            import omega_v2_core as omega_module
        except ImportError as exc:
            raise ImportError(
                "omega_bt FFI module not available; build with maturin develop"
            ) from exc
    return omega_module


def _extract_run_id(result: Mapping[str, Any]) -> str:
    meta = result.get("meta") or {}
    if isinstance(meta, Mapping):
        run_id = meta.get("run_id")
        if run_id:
            return str(run_id)
        extra = meta.get("extra") or {}
        if isinstance(extra, Mapping):
            run_id = extra.get("run_id")
            if run_id:
                return str(run_id)
    return "unknown"
