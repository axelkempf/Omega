"""Optional reporting helpers."""

from __future__ import annotations

from typing import Any, Mapping


def summarize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of metrics for reporting pipelines.

    Args:
        metrics: Metrics mapping.

    Returns:
        Dictionary copy of metrics.
    """
    return dict(metrics)
