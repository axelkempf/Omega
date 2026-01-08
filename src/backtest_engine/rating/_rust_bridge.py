"""
Rust FFI bridge for rating score modules.

Provides feature-flagged access to high-performance Rust implementations of
rating score calculations. Falls back to Python implementations when:
- Rust module not available
- OMEGA_USE_RUST_RATING=false environment variable set

Wave 1 Migration: This module provides the FFI bridge for rating calculations,
following the pattern established in Wave 0 (slippage/fee).

Reference: docs/WAVE_1_RATING_MODULE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import List, Mapping, Optional, Sequence, Tuple

# =============================================================================
# Rust FFI Feature Flag
# =============================================================================
# Auto-detect Rust module availability with override via environment variable
# OMEGA_USE_RUST_RATING=true|false|auto (default: auto)

_RUST_AVAILABLE: bool = False
_RUST_MODULE: Any = None

# List of required Rust functions for rating module
_REQUIRED_RUST_FUNCTIONS = [
    "compute_robustness_score_1",
    "compute_stability_score",
    "compute_stability_score_and_wmape",
    "compute_cost_shock_score",
    "compute_multi_factor_cost_shock_score",
    "compute_trade_dropout_score",
    "compute_multi_run_trade_dropout_score",
    "simulate_trade_dropout_metrics",
    "compute_ulcer_index",
    "compute_ulcer_index_and_score",
    "compute_penalty_profit_drawdown_sharpe",
    "score_from_penalty",
]


def _check_rust_available() -> bool:
    """Check if Rust module is available and has all required rating functions."""
    global _RUST_MODULE
    try:
        import omega_rust

        # Verify all required functions exist
        for func_name in _REQUIRED_RUST_FUNCTIONS:
            if not hasattr(omega_rust, func_name):
                return False
        _RUST_MODULE = omega_rust
        return True
    except ImportError:
        pass
    return False


def _should_use_rust() -> bool:
    """Determine if Rust implementation should be used."""
    env_val = os.environ.get("OMEGA_USE_RUST_RATING", "auto").lower()
    if env_val == "false":
        return False
    if env_val == "true":
        return _RUST_AVAILABLE
    # auto: use Rust if available
    return _RUST_AVAILABLE


# Initialize on module load
_RUST_AVAILABLE = _check_rust_available()


def get_rust_status() -> dict[str, bool | str]:
    """Get current Rust FFI status for debugging.

    Returns:
        Dict with 'available', 'enabled', and 'reason' keys.
    """
    enabled = _should_use_rust()
    env_val = os.environ.get("OMEGA_USE_RUST_RATING", "auto").lower()
    if not _RUST_AVAILABLE:
        reason = "Rust module not installed or missing required functions"
    elif env_val == "false":
        reason = "Disabled via OMEGA_USE_RUST_RATING=false"
    elif enabled:
        reason = "Rust module active"
    else:
        reason = "Unknown"
    return {
        "available": _RUST_AVAILABLE,
        "enabled": enabled,
        "reason": reason,
    }


def is_rust_enabled() -> bool:
    """Check if Rust implementation is currently enabled."""
    return _should_use_rust()


# =============================================================================
# Rust Bridge Functions
# =============================================================================


def rust_compute_robustness_score_1(
    base_profit: float,
    base_avg_r: float,
    base_winrate: float,
    base_drawdown: float,
    jitter_profits: List[float],
    jitter_avg_rs: List[float],
    jitter_winrates: List[float],
    jitter_drawdowns: List[float],
    penalty_cap: float = 0.5,
) -> float:
    """Call Rust compute_robustness_score_1."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_robustness_score_1(
        base_profit,
        base_avg_r,
        base_winrate,
        base_drawdown,
        jitter_profits,
        jitter_avg_rs,
        jitter_winrates,
        jitter_drawdowns,
        penalty_cap,
    )


def rust_compute_stability_score(
    years: List[int],
    profits: List[float],
    durations: Optional[List[float]] = None,
) -> float:
    """Call Rust compute_stability_score."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_stability_score(years, profits, durations)


def rust_compute_stability_score_and_wmape(
    years: List[int],
    profits: List[float],
    durations: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Call Rust compute_stability_score_and_wmape."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_stability_score_and_wmape(years, profits, durations)


def rust_compute_cost_shock_score(
    base_profit: float,
    base_drawdown: float,
    base_sharpe: float,
    shocked_profit: float,
    shocked_drawdown: float,
    shocked_sharpe: float,
    penalty_cap: float = 0.5,
) -> float:
    """Call Rust compute_cost_shock_score."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_cost_shock_score(
        base_profit,
        base_drawdown,
        base_sharpe,
        shocked_profit,
        shocked_drawdown,
        shocked_sharpe,
        penalty_cap,
    )


def rust_compute_multi_factor_cost_shock_score(
    base_profit: float,
    base_drawdown: float,
    base_sharpe: float,
    shocked_profits: List[float],
    shocked_drawdowns: List[float],
    shocked_sharpes: List[float],
    penalty_cap: float = 0.5,
) -> float:
    """Call Rust compute_multi_factor_cost_shock_score."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_multi_factor_cost_shock_score(
        base_profit,
        base_drawdown,
        base_sharpe,
        shocked_profits,
        shocked_drawdowns,
        shocked_sharpes,
        penalty_cap,
    )


def rust_compute_trade_dropout_score(
    base_profit: float,
    base_drawdown: float,
    base_sharpe: float,
    dropout_profit: float,
    dropout_drawdown: float,
    dropout_sharpe: float,
    penalty_cap: float = 0.5,
) -> float:
    """Call Rust compute_trade_dropout_score."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_trade_dropout_score(
        base_profit,
        base_drawdown,
        base_sharpe,
        dropout_profit,
        dropout_drawdown,
        dropout_sharpe,
        penalty_cap,
    )


def rust_compute_multi_run_trade_dropout_score(
    base_profit: float,
    base_drawdown: float,
    base_sharpe: float,
    dropout_profits: List[float],
    dropout_drawdowns: List[float],
    dropout_sharpes: List[float],
    penalty_cap: float = 0.5,
) -> float:
    """Call Rust compute_multi_run_trade_dropout_score."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_multi_run_trade_dropout_score(
        base_profit,
        base_drawdown,
        base_sharpe,
        dropout_profits,
        dropout_drawdowns,
        dropout_sharpes,
        penalty_cap,
    )


def rust_simulate_trade_dropout_metrics(
    pnls: List[float],
    r_multiples: List[float],
    dropout_frac: float,
    seed: int = 987_654_321,
) -> Tuple[float, float, float]:
    """Call Rust simulate_trade_dropout_metrics."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.simulate_trade_dropout_metrics(
        pnls, r_multiples, dropout_frac, seed
    )


def rust_compute_ulcer_index(equity_values: List[float]) -> float:
    """Call Rust compute_ulcer_index."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_ulcer_index(equity_values)


def rust_compute_ulcer_index_and_score(
    equity_values: List[float],
    ulcer_cap: float = 10.0,
) -> Tuple[float, float]:
    """Call Rust compute_ulcer_index_and_score."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_ulcer_index_and_score(equity_values, ulcer_cap)


def rust_compute_penalty_profit_drawdown_sharpe(
    base_profit: float,
    base_drawdown: float,
    base_sharpe: float,
    stress_profits: List[float],
    stress_drawdowns: List[float],
    stress_sharpes: List[float],
    penalty_cap: float = 0.5,
) -> float:
    """Call Rust compute_penalty_profit_drawdown_sharpe."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.compute_penalty_profit_drawdown_sharpe(
        base_profit,
        base_drawdown,
        base_sharpe,
        stress_profits,
        stress_drawdowns,
        stress_sharpes,
        penalty_cap,
    )


def rust_score_from_penalty(penalty: float, penalty_cap: float = 0.5) -> float:
    """Call Rust score_from_penalty."""
    if _RUST_MODULE is None:
        raise RuntimeError("Rust module not available")
    return _RUST_MODULE.score_from_penalty(penalty, penalty_cap)
