from __future__ import annotations

from typing import Any


def _legacy_validate_config(config: dict[str, Any]) -> list[str]:
    """Legacy validation kept as a fallback if Pydantic models are unavailable."""

    errors: list[str] = []

    if "start_date" not in config:
        errors.append("🟥 'start_date' fehlt")
    if "end_date" not in config:
        errors.append("🟥 'end_date' fehlt")

    mode = config.get("mode", "candle")
    if mode not in ["candle", "tick"]:
        errors.append("🟥 'mode' muss 'tick' oder 'candle' sein")

    if "strategy" not in config and "strategies" not in config:
        errors.append("🟥 Weder 'strategy' noch 'strategies' definiert")

    # Optional: fees/slippage sanity checks
    fees = config.get("fees")
    if (
        isinstance(fees, dict)
        and "per_million" in fees
        and not isinstance(fees.get("per_million"), (int, float))
    ):
        errors.append("🟥 'fees.per_million' muss eine Zahl sein")

    slippage = config.get("slippage")
    if isinstance(slippage, dict):
        for key in ["fixed_pips", "random_pips"]:
            if key in slippage and not isinstance(slippage.get(key), (int, float)):
                errors.append(f"🟥 'slippage.{key}' muss eine Zahl sein")

    # Historical field: allow but do not require it (data path is handled by the data handler).
    return errors


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a backtest config dict.

    New-style validation is implemented via Pydantic models (Phase 1 / P1-04).
    Unknown keys remain supported to keep configs forward-compatible.
    """

    try:
        from pydantic import ValidationError

        from backtest_engine.config.models import BacktestConfig

        BacktestConfig.model_validate(config)
        return []
    except Exception as e:
        # If the config fails Pydantic validation, surface structured errors.
        try:
            from pydantic import ValidationError

            if isinstance(e, ValidationError):
                errors: list[str] = []
                for err in e.errors():
                    loc = ".".join(str(p) for p in (err.get("loc") or []))
                    msg = str(err.get("msg") or "Invalid value")
                    errors.append(f"🟥 {loc}: {msg}")
                return errors
        except Exception:
            pass

        # Fallback: minimal legacy validation (never hard-fail because of missing optional fields).
        errors = _legacy_validate_config(config)
        if errors:
            return errors

        return [f"🟥 Konfiguration konnte nicht validiert werden: {e}"]
