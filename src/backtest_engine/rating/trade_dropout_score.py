from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from backtest_engine.rating._rust_bridge import (
    is_rust_enabled,
    rust_compute_multi_run_trade_dropout_score,
    rust_compute_trade_dropout_score,
    rust_simulate_trade_dropout_metrics,
)
from backtest_engine.rating.stress_penalty import (
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
)


def _drawdown_from_results(results: np.ndarray | None) -> float:
    if results is None:
        return 0.0
    res = np.asarray(results, dtype=float)
    if res.size == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(res)])
    peaks = np.maximum.accumulate(cum)
    return float(np.max(peaks - cum)) if cum.size > 0 else 0.0


def _sharpe_from_r_multiples(r: np.ndarray) -> float:
    arr = np.asarray(r, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    return float(mu / sigma) if sigma > 0.0 else 0.0


def simulate_trade_dropout_metrics(
    trades_df: Optional[pd.DataFrame],
    *,
    dropout_frac: float,
    base_metrics: Optional[Mapping[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    pnl_col: str = "result",
    r_col: str = "r_multiple",
    debug: bool | None = None,
) -> Dict[str, float]:
    """
    Simulate trade dropout by removing a fraction of trades and recomputing
    (profit, drawdown, sharpe) from the remaining trade series.

    Important semantics:
    - Profit/Drawdown are computed **net of fees** whenever fee columns exist.
      This aligns the dropout stress metrics with the base metrics in the pipeline
      (which use `net_profit_after_fees_eur` and drawdown from `Portfolio.cash`).
    - Trades are sorted chronologically (by exit/close time if available) before
      applying dropout, so results are deterministic and invariant to input row order.
    """

    debug_enabled = bool(debug) if debug is not None else False
    if debug is None:
        try:
            env_flag = str(os.getenv("TRADE_DROPOUT_DEBUG", "") or "").strip().lower()
            if env_flag in {"1", "true", "yes", "y", "on"}:
                debug_enabled = True
        except Exception:
            debug_enabled = False

    if trades_df is None or trades_df.empty or float(dropout_frac) <= 0:
        if debug_enabled:
            try:
                n_dbg = 0
                try:
                    n_dbg = int(len(trades_df)) if trades_df is not None else 0
                except Exception:
                    n_dbg = 0
                print(
                    "[trade-dropout][debug] skipped (empty/frac<=0) n={} dropout_frac={}".format(
                        n_dbg,
                        float(dropout_frac),
                    )
                )
            except Exception:
                pass
        if base_metrics is None:
            return {"profit": 0.0, "drawdown": 0.0, "sharpe": 0.0}
        return {
            "profit": float(base_metrics.get("profit", 0.0) or 0.0),
            "drawdown": float(base_metrics.get("drawdown", 0.0) or 0.0),
            "sharpe": float(base_metrics.get("sharpe", 0.0) or 0.0),
        }

    t = trades_df
    if pnl_col not in t.columns:
        if debug_enabled:
            try:
                print(
                    "[trade-dropout][debug] skipped (missing pnl_col) n={} dropout_frac={} pnl_col={}".format(
                        int(len(t)) if hasattr(t, "__len__") else 0,
                        float(dropout_frac),
                        str(pnl_col),
                    )
                )
            except Exception:
                pass
        if base_metrics is None:
            return {"profit": 0.0, "drawdown": 0.0, "sharpe": 0.0}
        return {
            "profit": float(base_metrics.get("profit", 0.0) or 0.0),
            "drawdown": float(base_metrics.get("drawdown", 0.0) or 0.0),
            "sharpe": float(base_metrics.get("sharpe", 0.0) or 0.0),
        }

    # Ensure deterministic ordering independent of caller DataFrame row order.
    try:
        time_col = None
        for c in (
            "exit_time",
            "close_time",
            "exit_timestamp",
            "close_timestamp",
            "exit",
        ):
            if c in t.columns:
                time_col = c
                break
        if time_col is None:
            for c in (
                "entry_time",
                "open_time",
                "entry_timestamp",
                "open_timestamp",
                "entry",
            ):
                if c in t.columns:
                    time_col = c
                    break
        if time_col is not None:
            times = pd.to_datetime(t[time_col], errors="coerce")
            t = (
                t.assign(_t_sort=times)
                .sort_values("_t_sort", kind="mergesort")
                .drop(columns=["_t_sort"])
            )
    except Exception:
        t = trades_df

    # Profit/Drawdown: net-of-fee if available
    pnl = pd.to_numeric(t[pnl_col], errors="coerce").to_numpy(dtype=float, copy=False)
    pnl = np.where(np.isfinite(pnl), pnl, 0.0)
    fees = np.zeros_like(pnl, dtype=float)
    try:
        if "total_fee" in t.columns:
            fees = pd.to_numeric(t["total_fee"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
        elif "entry_fee" in t.columns and "exit_fee" in t.columns:
            f1 = pd.to_numeric(t["entry_fee"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
            f2 = pd.to_numeric(t["exit_fee"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
            fees = f1 + f2
    except Exception:
        fees = np.zeros_like(pnl, dtype=float)
    fees = np.where(np.isfinite(fees), fees, 0.0)

    results_all = pnl - fees
    n = int(results_all.size)
    if n <= 0:
        if debug_enabled:
            try:
                print(
                    "[trade-dropout][debug] skipped (n<=0) n={} dropout_frac={}".format(
                        int(n),
                        float(dropout_frac),
                    )
                )
            except Exception:
                pass
        if base_metrics is None:
            return {"profit": 0.0, "drawdown": 0.0, "sharpe": 0.0}
        return {
            "profit": float(base_metrics.get("profit", 0.0) or 0.0),
            "drawdown": float(base_metrics.get("drawdown", 0.0) or 0.0),
            "sharpe": float(base_metrics.get("sharpe", 0.0) or 0.0),
        }

    n_drop = int(math.ceil(n * float(dropout_frac)))
    n_drop = min(max(n_drop, 1), n)

    if debug_enabled:
        try:
            print(
                "[trade-dropout][debug] runs=1 n={} dropout_frac={} n_drop={} n_keep={}".format(
                    int(n),
                    float(dropout_frac),
                    int(n_drop),
                    int(n - n_drop),
                )
            )
        except Exception:
            pass

    if rng is None:
        rng = np.random.default_rng(int(seed) if seed is not None else 987654321)

    idx = np.arange(n, dtype=int)
    drop_idx = rng.choice(idx, size=n_drop, replace=False)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[drop_idx] = False

    results = results_all[keep_mask]
    profit = float(results.sum()) if results.size else 0.0
    dd = float(_drawdown_from_results(results)) if results.size else 0.0

    sharpe = 0.0
    if r_col in t.columns:
        r_all = pd.to_numeric(t[r_col], errors="coerce").to_numpy(dtype=float)
        r_all = np.where(np.isfinite(r_all), r_all, np.nan)
        r_kept = r_all[keep_mask]
        sharpe = float(_sharpe_from_r_multiples(r_kept))

    return {"profit": profit, "drawdown": dd, "sharpe": sharpe}


def compute_trade_dropout_score(
    base_metrics: Mapping[str, float],
    dropout_metrics: Mapping[str, float],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """Compute trade dropout robustness score.

    Implementation:
      - Uses Rust implementation when OMEGA_USE_RUST_RATING is enabled.
      - Falls back to Python for compatibility.
    """
    if is_rust_enabled():

        def _to_finite(x: object, default: float = 0.0) -> float:
            try:
                v = float(x)  # type: ignore[arg-type]
            except Exception:
                return float(default)
            return v if math.isfinite(v) else float(default)

        base_profit = _to_finite(base_metrics.get("profit", 0.0), default=0.0)
        base_drawdown = _to_finite(base_metrics.get("drawdown", 0.0), default=0.0)
        base_sharpe = _to_finite(base_metrics.get("sharpe", 0.0), default=0.0)
        dropout_profit = _to_finite(dropout_metrics.get("profit", 0.0), default=0.0)
        dropout_drawdown = _to_finite(dropout_metrics.get("drawdown", 0.0), default=0.0)
        dropout_sharpe = _to_finite(dropout_metrics.get("sharpe", 0.0), default=0.0)
        return rust_compute_trade_dropout_score(
            base_profit,
            base_drawdown,
            base_sharpe,
            dropout_profit,
            dropout_drawdown,
            dropout_sharpe,
            penalty_cap,
        )

    penalty = compute_penalty_profit_drawdown_sharpe(
        base_metrics, [dropout_metrics], penalty_cap=penalty_cap
    )
    return float(score_from_penalty(penalty, penalty_cap=penalty_cap))


def simulate_trade_dropout_metrics_multi(
    trades_df: Optional[pd.DataFrame],
    *,
    dropout_frac: float,
    base_metrics: Optional[Mapping[str, float]] = None,
    n_runs: int = 1,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    pnl_col: str = "result",
    r_col: str = "r_multiple",
    debug: bool | None = None,
) -> List[Dict[str, float]]:
    """Run trade dropout simulation multiple times and return metrics per run."""

    runs = int(n_runs)
    if runs <= 0:
        return []

    # Preserve legacy single-run behaviour exactly.
    if runs == 1:
        return [
            simulate_trade_dropout_metrics(
                trades_df,
                dropout_frac=dropout_frac,
                base_metrics=base_metrics,
                rng=rng,
                seed=seed,
                pnl_col=pnl_col,
                r_col=r_col,
                debug=debug,
            )
        ]

    debug_enabled = bool(debug) if debug is not None else False
    if debug is None:
        try:
            env_flag = str(os.getenv("TRADE_DROPOUT_DEBUG", "") or "").strip().lower()
            if env_flag in {"1", "true", "yes", "y", "on"}:
                debug_enabled = True
        except Exception:
            debug_enabled = False

    base_seed = int(seed) if seed is not None else None
    results: List[Dict[str, float]] = []

    # Use deterministic sub-seeds to keep runs reproducible and independent of run count.
    default_seed_base = 987654321

    for i_run in range(runs):
        run_seed = (
            base_seed + i_run if base_seed is not None else default_seed_base + i_run
        )
        run_rng = np.random.default_rng(run_seed) if rng is None else rng

        if debug_enabled:
            try:
                print(
                    "[trade-dropout][debug] run {}/{} seed={}".format(
                        int(i_run + 1), int(runs), int(run_seed)
                    )
                )
            except Exception:
                pass

        metrics = simulate_trade_dropout_metrics(
            trades_df,
            dropout_frac=dropout_frac,
            base_metrics=base_metrics,
            rng=run_rng,
            seed=None,  # rng already provided
            pnl_col=pnl_col,
            r_col=r_col,
            debug=debug,
        )
        results.append(metrics)

    return results


def compute_multi_run_trade_dropout_score(
    base_metrics: Mapping[str, float],
    dropout_metrics_list: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """Aggregate dropout scores across multiple runs by averaging.

    Implementation:
      - Uses Rust implementation when OMEGA_USE_RUST_RATING is enabled.
      - Falls back to Python for compatibility.
    """

    if not dropout_metrics_list:
        return 1.0

    if is_rust_enabled():

        def _to_finite(x: object, default: float = 0.0) -> float:
            try:
                v = float(x)  # type: ignore[arg-type]
            except Exception:
                return float(default)
            return v if math.isfinite(v) else float(default)

        base_profit = _to_finite(base_metrics.get("profit", 0.0), default=0.0)
        base_drawdown = _to_finite(base_metrics.get("drawdown", 0.0), default=0.0)
        base_sharpe = _to_finite(base_metrics.get("sharpe", 0.0), default=0.0)
        dropout_profits = [
            _to_finite(m.get("profit", 0.0), default=0.0) for m in dropout_metrics_list
        ]
        dropout_drawdowns = [
            _to_finite(m.get("drawdown", 0.0), default=0.0) for m in dropout_metrics_list
        ]
        dropout_sharpes = [
            _to_finite(m.get("sharpe", 0.0), default=0.0) for m in dropout_metrics_list
        ]
        return rust_compute_multi_run_trade_dropout_score(
            base_profit,
            base_drawdown,
            base_sharpe,
            dropout_profits,
            dropout_drawdowns,
            dropout_sharpes,
            penalty_cap,
        )

    scores = [
        compute_trade_dropout_score(base_metrics, dm, penalty_cap=penalty_cap)
        for dm in dropout_metrics_list
    ]
    return float(sum(scores) / len(scores))
