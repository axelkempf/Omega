from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def _to_finite(x: object, *, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _pct_drop(base: float, x: float, *, invert: bool = False) -> float:
    base = max(float(base), 1e-9)
    x = max(float(x), 0.0)
    if invert:
        return max(0.0, (x - base) / base)  # drawdown increase
    return max(0.0, (base - x) / base)  # relative drop


def compute_robustness_score_1(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Robustness-1 score (parameter jitter):
      - penalty = mean over repeats of mean(%drops in profit, avg_r, winrate, drawdown-increase)
      - penalty clipped to [0, penalty_cap]
      - score = 1 - penalty

    Expected keys in metrics:
      - profit, avg_r, winrate, drawdown

    Guardrail / interpretation:
      - The score is designed for strategies with *positive* base performance.
        When a base metric is <= 0, the relative-drop denominator is guarded via 1e-9
        and the resulting penalty contribution may become uninformative.
        Upstream selection should typically filter out unprofitable candidates before
        relying on this robustness score.
    """
    cap = float(max(0.0, penalty_cap))
    if cap == 0.0:
        return 1.0

    base_profit = _to_finite(base_metrics.get("profit", 0.0), default=0.0)
    base_avg_r = _to_finite(base_metrics.get("avg_r", 0.0), default=0.0)
    base_winrate = _to_finite(base_metrics.get("winrate", 0.0), default=0.0)
    base_drawdown = _to_finite(base_metrics.get("drawdown", 0.0), default=0.0)

    if not jitter_metrics:
        return float(max(0.0, 1.0 - cap))

    drops: list[float] = []
    for m in jitter_metrics:
        profit = _to_finite(m.get("profit", 0.0), default=0.0)
        avg_r = _to_finite(m.get("avg_r", 0.0), default=0.0)
        winrate = _to_finite(m.get("winrate", 0.0), default=0.0)
        drawdown = _to_finite(m.get("drawdown", 0.0), default=0.0)
        d = (
            _pct_drop(base_profit, profit, invert=False)
            + _pct_drop(base_avg_r, avg_r, invert=False)
            + _pct_drop(base_winrate, winrate, invert=False)
            + _pct_drop(base_drawdown, drawdown, invert=True)
        ) / 4.0
        if math.isfinite(d):
            drops.append(float(d))

    if not drops:
        penalty = cap
    else:
        penalty = float(np.nanmean(np.asarray(drops, dtype=float)))
        if not math.isfinite(penalty):
            penalty = cap
        penalty = float(max(0.0, min(cap, penalty)))

    return float(max(0.0, 1.0 - penalty))
