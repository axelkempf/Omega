from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


def _clean_numeric(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=np.float64)
    if isinstance(x, np.ndarray):
        arr: np.ndarray = np.asarray(x, dtype=np.float64)
    elif isinstance(x, pd.Series):
        arr = np.asarray(
            pd.to_numeric(x, errors="coerce").to_numpy(copy=False), dtype=np.float64
        )
    else:
        try:
            arr = np.asarray(list(x), dtype=np.float64)
        except Exception:
            return np.array([], dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr


def _net_of_fees_series(
    trades_df: pd.DataFrame,
    *,
    pnl_col: str,
    fees_col: str = "total_fee",
) -> pd.Series:
    """
    Return per-trade PnL series net of fees where possible.

    Expected (from Portfolio.trades_to_dataframe):
      - pnl_col: "result" (gross PnL)
      - fees_col: "total_fee" (entry_fee + exit_fee)

    Fallbacks:
      - If fees_col is missing but entry_fee/exit_fee exist, subtract their sum.
      - If no fee columns exist, return pnl_col unchanged.
    """
    pnl = pd.to_numeric(trades_df[pnl_col], errors="coerce")
    if fees_col in trades_df.columns:
        fees = pd.to_numeric(trades_df[fees_col], errors="coerce")
        return pnl - fees
    if "entry_fee" in trades_df.columns and "exit_fee" in trades_df.columns:
        f1 = pd.to_numeric(trades_df["entry_fee"], errors="coerce")
        f2 = pd.to_numeric(trades_df["exit_fee"], errors="coerce")
        return pnl - (f1 + f2)
    return pnl


def bootstrap_p_value_mean_gt_zero(
    x: Any,
    *,
    n_boot: int = 2000,
    seed: int = 123,
) -> float:
    """
    IID bootstrap p-value for H0: mean(x) <= 0, i.e. p = P(mean_boot <= 0).

    Note: This is a bootstrap tail probability under the *empirical* distribution.
    It is not a multiple-testing corrected measure and may be optimistic when used
    after extensive parameter search / selection.

    Returns 1.0 for empty / too-small / non-finite inputs.
    """
    arr = _clean_numeric(x)
    if arr.size < 2:
        return 1.0
    rng = np.random.default_rng(int(seed))
    means = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    p = float((means <= 0.0).mean())
    return p if np.isfinite(p) else 1.0


def compute_p_mean_r_gt_0(
    trades_df: Optional[pd.DataFrame],
    *,
    r_col: str = "r_multiple",
    n_boot: int = 2000,
    seed: int = 123,
) -> float:
    if trades_df is None or trades_df.empty or r_col not in trades_df.columns:
        return 1.0
    return bootstrap_p_value_mean_gt_zero(trades_df[r_col], n_boot=n_boot, seed=seed)


def compute_p_net_profit_gt_0(
    trades_df: Optional[pd.DataFrame],
    *,
    pnl_col: str = "result",
    fees_col: str = "total_fee",
    net_of_fees: bool = True,
    n_boot: int = 2000,
    seed: int = 456,
) -> float:
    if trades_df is None or trades_df.empty or pnl_col not in trades_df.columns:
        return 1.0
    x = (
        _net_of_fees_series(trades_df, pnl_col=pnl_col, fees_col=fees_col)
        if net_of_fees
        else trades_df[pnl_col]
    )
    return bootstrap_p_value_mean_gt_zero(x, n_boot=n_boot, seed=seed)


def compute_p_values(
    trades_df: Optional[pd.DataFrame],
    *,
    r_col: str = "r_multiple",
    pnl_col: str = "result",
    fees_col: str = "total_fee",
    net_of_fees_pnl: bool = True,
    n_boot: int = 2000,
    seed_r: int = 123,
    seed_pnl: int = 456,
) -> Mapping[str, float]:
    """
    Convenience wrapper returning the project-standard p-value fields.
    """
    if trades_df is None or trades_df.empty:
        return {"p_mean_r_gt_0": 1.0, "p_net_profit_gt_0": 1.0}
    return {
        "p_mean_r_gt_0": float(
            compute_p_mean_r_gt_0(trades_df, r_col=r_col, n_boot=n_boot, seed=seed_r)
        ),
        "p_net_profit_gt_0": float(
            compute_p_net_profit_gt_0(
                trades_df,
                pnl_col=pnl_col,
                fees_col=fees_col,
                net_of_fees=net_of_fees_pnl,
                n_boot=n_boot,
                seed=seed_pnl,
            )
        ),
    }
