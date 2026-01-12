from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np


def _days_in_year(y: int) -> int:
    return 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365


def compute_stability_score_and_wmape_from_yearly_profits(
    profits_by_year: Mapping[int, float],
    *,
    durations_by_year: Optional[Mapping[int, float]] = None,
) -> Tuple[float, float]:
    """
    Stability score used in Step-5 scoring:

      wmape = sum_y w_y * |P_y - E_y| / max(|E_y|, S_min)
      score = 1 / (1 + wmape)

    where:
      - w_y = duration_y / sum(duration_y)
      - E_y is the expected profit for that year under a constant daily rate
      - S_min = max(100, 0.02 * |P_total|)

    Returns (score, wmape). Uses safe defaults (1.0, 0.0) for empty/invalid inputs.

    If `durations_by_year` is provided, it must contain the **actual number of days**
    in each segment used to produce `profits_by_year` (e.g. partial-year segments from
    start/end windows). Missing/invalid durations fall back to calendar year day counts.
    """
    if not profits_by_year:
        return 1.0, 0.0

    years = sorted({int(y) for y in profits_by_year.keys()})
    durations: dict[int, float] = {}
    for y in years:
        d = None
        if durations_by_year is not None:
            try:
                d = float(durations_by_year.get(y, 0.0))
            except Exception:
                d = None
        if d is None or not np.isfinite(d) or d <= 0.0:
            d = float(_days_in_year(y))
        durations[y] = float(d)
    d_total = float(sum(durations.values()))
    if d_total <= 0.0 or not np.isfinite(d_total):
        return 1.0, 0.0

    profits = {y: float(profits_by_year.get(y, 0.0) or 0.0) for y in years}
    p_total = float(sum(profits.values()))
    mu = p_total / d_total if d_total > 0 else 0.0
    s_min = float(max(100.0, 0.02 * abs(p_total)))

    wmape = 0.0
    for y in years:
        d_y = float(durations.get(y, 0.0))
        if d_y <= 0:
            continue
        p_y = float(profits.get(y, 0.0))
        e_y = mu * d_y
        denom = max(abs(e_y), s_min)
        r_y = abs(p_y - e_y) / denom if denom > 0 else 0.0
        w_y = d_y / d_total
        wmape += w_y * r_y

    if not np.isfinite(wmape):
        return 1.0, 0.0
    score = float(1.0 / (1.0 + float(wmape)))
    if not np.isfinite(score):
        return 1.0, 0.0
    return score, float(wmape)


def compute_stability_score_from_yearly_profits(
    profits_by_year: Mapping[int, float],
    *,
    durations_by_year: Optional[Mapping[int, float]] = None,
) -> float:
    return float(
        compute_stability_score_and_wmape_from_yearly_profits(
            profits_by_year, durations_by_year=durations_by_year
        )[0]
    )
