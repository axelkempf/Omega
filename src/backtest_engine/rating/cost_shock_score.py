from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

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


def compute_cost_shock_score(
    base_metrics: Mapping[str, float],
    shocked_metrics: Mapping[str, float],
    *,
    penalty_cap: float = 0.5,
) -> float:
    penalty = compute_penalty_profit_drawdown_sharpe(
        base_metrics, [shocked_metrics], penalty_cap=penalty_cap
    )
    return score_from_penalty(penalty, penalty_cap=penalty_cap)


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
    """
    if not shocked_metrics_list:
        return 1.0

    scores = [
        compute_cost_shock_score(base_metrics, sm, penalty_cap=penalty_cap)
        for sm in shocked_metrics_list
    ]
    return float(sum(scores) / len(scores))
