"""Optional reporting helpers for backtest results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# Key metrics for reporting pipelines (aligned with OMEGA_V2_OUTPUT_CONTRACT_PLAN)
REPORT_METRICS: list[str] = [
    "total_trades",
    "win_rate",
    "profit_net",
    "max_drawdown",
    "profit_factor",
    "sharpe_equity_daily",
    "calmar_ratio",
]


def summarize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of metrics for reporting pipelines.

    Args:
        metrics: Metrics mapping from backtest result.

    Returns:
        Dictionary copy of metrics.
    """
    return dict(metrics)


def extract_key_metrics(
    metrics: Mapping[str, Any],
    keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Extract subset of metrics for summary reports.

    Args:
        metrics: Full metrics mapping from backtest result.
        keys: Metric keys to extract. Defaults to REPORT_METRICS.

    Returns:
        Dictionary with requested metrics only. Missing keys are omitted.
    """
    if keys is None:
        keys = REPORT_METRICS
    return {k: metrics[k] for k in keys if k in metrics}


def format_metric(
    key: str,
    value: Any,
    definitions: Mapping[str, Any] | None = None,
) -> str:
    """Format a single metric for display.

    Args:
        key: Metric key name.
        value: Metric value.
        definitions: Optional metric definitions for unit information.

    Returns:
        Formatted string representation of the metric.
    """
    if value is None or value == "n/a":
        return f"{key}: n/a"

    unit = ""
    if definitions and key in definitions:
        unit_str = definitions[key].get("unit", "")
        if unit_str and unit_str not in ("ratio", "count"):
            unit = f" {unit_str}"

    if isinstance(value, float):
        if "rate" in key or "ratio" in key.lower():
            return f"{key}: {value:.2%}{unit}"
        return f"{key}: {value:.4f}{unit}"
    return f"{key}: {value}{unit}"
