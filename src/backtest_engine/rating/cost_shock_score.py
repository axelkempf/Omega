from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from backtest_engine.rating._rust_bridge import (
    is_rust_enabled,
    rust_compute_cost_shock_score,
    rust_compute_multi_factor_cost_shock_score,
)
from backtest_engine.rating.stress_penalty import (
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
)

# Deterministic cost shock multipliers (+25%, +50%, +100%).
COST_SHOCK_FACTORS: Tuple[float, ...] = (1.25, 1.50, 2.00)


def apply_cost_shock_inplace(cfg: Dict[str, Any], *, factor: float) -> None:
    """
    Apply a multiplicative "cost shock" to the effective execution costs.

    Why this exists:
    - Most backtest configs do **not** contain explicit `slippage/fees/commission` sections.
      In that case, execution costs come from `configs/execution_costs.yaml` and are
      only modified via `config["execution"]` multipliers in `runner.py`.
    - Therefore we apply the shock primarily by scaling these multipliers so the
      shock cannot become a silent no-op.

    Behavior:
    - Multiplies `config["execution"]["slippage_multiplier"]` and
      `config["execution"]["fee_multiplier"]` by `factor` (creating them if missing).

    Notes:
    - We deliberately do **not** scale explicit `slippage/fees/...` sections here,
      because the runner already applies the multipliers to those fields. Scaling both
      would accidentally apply the shock twice.

    Targets (if present): slippage, fees, commission
    """
    if factor <= 0:
        return

    # Primary path: multipliers used by run_backtest_and_return_portfolio (and optimizers).
    exec_cfg = cfg.get("execution")
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}
        cfg["execution"] = exec_cfg
    # Slippage multiplier scales fixed/random pips.
    exec_cfg["slippage_multiplier"] = float(
        exec_cfg.get("slippage_multiplier", 1.0)
    ) * float(factor)
    # Fee multiplier scales CommissionModel and legacy FeeModel.
    exec_cfg["fee_multiplier"] = float(exec_cfg.get("fee_multiplier", 1.0)) * float(
        factor
    )

    # No additional in-place scaling of `slippage/fees/...` sections to avoid double shock.


def _to_finite(x: object, *, default: float = 0.0) -> float:
    import math

    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def compute_cost_shock_score(
    base_metrics: Mapping[str, float],
    shocked_metrics: Mapping[str, float],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """Compute cost shock robustness score.

    Implementation:
      - Uses Rust implementation when OMEGA_USE_RUST_RATING is enabled.
      - Falls back to Python for compatibility.
    """
    if is_rust_enabled():
        base_profit = _to_finite(base_metrics.get("profit", 0.0), default=0.0)
        base_drawdown = _to_finite(base_metrics.get("drawdown", 0.0), default=0.0)
        base_sharpe = _to_finite(base_metrics.get("sharpe", 0.0), default=0.0)
        shocked_profit = _to_finite(shocked_metrics.get("profit", 0.0), default=0.0)
        shocked_drawdown = _to_finite(shocked_metrics.get("drawdown", 0.0), default=0.0)
        shocked_sharpe = _to_finite(shocked_metrics.get("sharpe", 0.0), default=0.0)
        return rust_compute_cost_shock_score(
            base_profit,
            base_drawdown,
            base_sharpe,
            shocked_profit,
            shocked_drawdown,
            shocked_sharpe,
            penalty_cap,
        )

    penalty = compute_penalty_profit_drawdown_sharpe(
        base_metrics, [shocked_metrics], penalty_cap=penalty_cap
    )
    return float(score_from_penalty(penalty, penalty_cap=penalty_cap))


def compute_multi_factor_cost_shock_score(
    base_metrics: Mapping[str, float],
    shocked_metrics_list: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """Aggregate cost shock robustness across multiple deterministic factors.

    Args:
        base_metrics: Baseline performance metrics (profit, drawdown, sharpe).
        shocked_metrics_list: Metrics from multiple shocked backtests.
        penalty_cap: Maximum penalty per shocked run.

    Returns:
        Mean score across shocks. Returns 1.0 when no shocks are provided.

    Implementation:
      - Uses Rust implementation when OMEGA_USE_RUST_RATING is enabled.
      - Falls back to Python for compatibility.
    """
    if not shocked_metrics_list:
        return 1.0

    if is_rust_enabled():
        base_profit = _to_finite(base_metrics.get("profit", 0.0), default=0.0)
        base_drawdown = _to_finite(base_metrics.get("drawdown", 0.0), default=0.0)
        base_sharpe = _to_finite(base_metrics.get("sharpe", 0.0), default=0.0)
        shocked_profits = [
            _to_finite(m.get("profit", 0.0), default=0.0) for m in shocked_metrics_list
        ]
        shocked_drawdowns = [
            _to_finite(m.get("drawdown", 0.0), default=0.0) for m in shocked_metrics_list
        ]
        shocked_sharpes = [
            _to_finite(m.get("sharpe", 0.0), default=0.0) for m in shocked_metrics_list
        ]
        return rust_compute_multi_factor_cost_shock_score(
            base_profit,
            base_drawdown,
            base_sharpe,
            shocked_profits,
            shocked_drawdowns,
            shocked_sharpes,
            penalty_cap,
        )

    scores = [
        compute_cost_shock_score(base_metrics, sm, penalty_cap=penalty_cap)
        for sm in shocked_metrics_list
    ]
    return float(sum(scores) / len(scores))
