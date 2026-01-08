from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from backtest_engine.rating._rust_bridge import (
    is_rust_enabled,
    rust_compute_penalty_profit_drawdown_sharpe,
    rust_score_from_penalty,
)


def _to_finite(x: object, *, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def compute_penalty_profit_drawdown_sharpe(
    base_metrics: Mapping[str, float],
    stress_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """
    Stress penalty used by the new independent robustness/stress scores.

    Uses only:
      - profit  (drop is bad)
      - drawdown (increase is bad)
      - sharpe  (drop is bad)

    penalty is clipped to [0, penalty_cap].

    Expected keys:
      base_metrics: profit, drawdown, sharpe
      stress_metrics[i]: profit, drawdown, sharpe

    Implementation:
      - Uses Rust implementation when OMEGA_USE_RUST_RATING is enabled.
      - Falls back to Python for compatibility.
    """
    cap = float(max(0.0, penalty_cap))
    if cap == 0.0:
        return 0.0

    base_profit = _to_finite(base_metrics.get("profit", 0.0), default=0.0)
    base_drawdown = _to_finite(base_metrics.get("drawdown", 0.0), default=0.0)
    base_sharpe = _to_finite(base_metrics.get("sharpe", 0.0), default=0.0)

    # Rust dispatch
    if is_rust_enabled() and stress_metrics:
        stress_profits = [
            _to_finite(m.get("profit", 0.0), default=0.0) for m in stress_metrics if m
        ]
        stress_drawdowns = [
            _to_finite(m.get("drawdown", 0.0), default=0.0) for m in stress_metrics if m
        ]
        stress_sharpes = [
            _to_finite(m.get("sharpe", 0.0), default=0.0) for m in stress_metrics if m
        ]
        if stress_profits:
            return rust_compute_penalty_profit_drawdown_sharpe(
                base_profit,
                base_drawdown,
                base_sharpe,
                stress_profits,
                stress_drawdowns,
                stress_sharpes,
                cap,
            )

    # Python fallback
    def rel_drop(base: float, stress: float) -> float:
        if base <= 0.0:
            return 0.0
        return max(0.0, (base - stress) / base)

    def rel_increase(base: float, stress: float) -> float:
        if base <= 0.0:
            base = 1e-9
        return max(0.0, (stress - base) / base)

    penalties: list[float] = []
    for metrics in stress_metrics:
        if not metrics:
            continue
        profit = _to_finite(metrics.get("profit", 0.0), default=0.0)
        drawdown = _to_finite(metrics.get("drawdown", 0.0), default=0.0)
        sharpe = _to_finite(metrics.get("sharpe", 0.0), default=0.0)

        p = rel_drop(base_profit, profit)
        d = rel_increase(max(base_drawdown, 1e-9), drawdown)
        s = rel_drop(max(base_sharpe, 1e-9), sharpe)
        pen = (p + d + s) / 3.0
        if math.isfinite(pen):
            penalties.append(float(pen))

    if not penalties:
        return 0.0

    penalty = float(np.nanmean(np.asarray(penalties, dtype=float)))
    if not math.isfinite(penalty):
        penalty = cap
    return float(max(0.0, min(cap, penalty)))


def score_from_penalty(penalty: float, *, penalty_cap: float = 0.5) -> float:
    """Convert penalty to score.

    Implementation:
      - Uses Rust implementation when OMEGA_USE_RUST_RATING is enabled.
      - Falls back to Python for compatibility.
    """
    if is_rust_enabled():
        return rust_score_from_penalty(penalty, penalty_cap)

    cap = float(max(0.0, penalty_cap))
    pen = _to_finite(penalty, default=cap)
    pen = float(max(0.0, min(cap, pen)))
    return float(max(0.0, 1.0 - pen))
