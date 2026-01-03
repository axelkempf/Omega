from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np


def compute_ulcer_index_and_score(
    equity_curve: Sequence[object] | Iterable[object], *, ulcer_cap: float = 10.0
) -> Tuple[float, float]:
    """
    Compute Ulcer Index (weekly, drawdown in percent) and mapped score.

    This implementation is intentionally aligned with the logic used in
    ``analysis/combined_walkforward_matrix_analyzer.py`` (weekly resampling and
    percent drawdowns):

    - weekly = equity.resample("W").last()
    - dd_pct = ((weekly - weekly.cummax()) / weekly.cummax() * 100).fillna(0)
    - ulcer = sqrt(mean(dd_pct^2))

    Args:
        equity_curve: Sequence of equity values or (timestamp, equity) tuples.
            If timestamps are provided, values are resampled to weekly closes
            ("W", week ending Sunday) before the Ulcer Index is computed.
            If timestamps are not provided, the input is treated as already
            representing weekly closes.
        ulcer_cap: Cap for linear score mapping in the same unit as the Ulcer
            Index (i.e., percent). Higher Ulcer → lower score.

    Returns:
        Tuple of (ulcer_index, ulcer_score). ``ulcer_index`` is NaN when no
        usable data is available; ``ulcer_score`` is clamped to [0, 1].
    """

    # Parse input points. Prefer timestamped points so we can resample weekly.
    # Fallback: treat the sequence as already being weekly closes.
    weekly_like: list[float] = []
    ts_points: list[tuple[object, float]] = []

    for point in equity_curve:
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            ts, val = point[0], point[1]
            try:
                num = float(val)  # type: ignore[arg-type]
            except Exception:
                continue
            if not math.isfinite(num):
                continue
            ts_points.append((ts, num))
            weekly_like.append(num)
            continue

        try:
            num = float(point)  # type: ignore[arg-type]
        except Exception:
            continue
        if math.isfinite(num):
            weekly_like.append(num)

    ulcer_index = math.nan

    # Timestamped path: weekly resampling (W-SUN) + percent drawdowns.
    if len(ts_points) >= 2:
        try:
            import pandas as pd

            ts = pd.to_datetime([p[0] for p in ts_points], utc=True, errors="coerce")
            vals = np.asarray([p[1] for p in ts_points], dtype=float)
            mask = ts.notna() & np.isfinite(vals)
            if int(mask.sum()) >= 2:
                s = pd.Series(vals[mask], index=ts[mask]).sort_index()
                # Keep last value for duplicate timestamps.
                s = s[~s.index.duplicated(keep="last")]
                weekly = s.resample("W").last()
                if len(weekly) >= 2:
                    roll_max = weekly.cummax()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dd_pct = (weekly - roll_max) / roll_max * 100.0
                    dd_pct = dd_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    arr = dd_pct.to_numpy(dtype=float, copy=False)
                    ulcer_index = float(np.sqrt(np.mean(arr**2)))
        except Exception:
            # Any parsing/resampling issue -> fall back to weekly_like path.
            pass

    # Fallback path: treat values as already-weekly closes.
    if not math.isfinite(ulcer_index):
        if len(weekly_like) < 2:
            return math.nan, 0.0
        equities = np.asarray(weekly_like, dtype=float)
        equities = equities[np.isfinite(equities)]
        if equities.size < 2:
            return math.nan, 0.0

        roll_max = np.maximum.accumulate(equities)
        dd_pct = np.zeros_like(equities, dtype=float)
        valid = roll_max > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct[valid] = (equities[valid] - roll_max[valid]) / roll_max[valid] * 100.0
        dd_pct[~np.isfinite(dd_pct)] = 0.0
        ulcer_index = float(np.sqrt(np.mean(dd_pct**2)))

    if not math.isfinite(ulcer_index):
        ulcer_index = math.nan

    cap = float(ulcer_cap)
    if not math.isfinite(cap) or cap <= 0.0 or not math.isfinite(ulcer_index):
        return ulcer_index, 0.0

    score = 1.0 - ulcer_index / cap
    score = max(0.0, min(1.0, float(score)))
    return ulcer_index, score
