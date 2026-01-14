from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from backtest_engine.rating.stress_penalty import (
    compute_penalty_profit_drawdown_sharpe,
    score_from_penalty,
)
from dateutil.relativedelta import relativedelta


def _parse_date_string(s: str) -> datetime:
    """Parse date-only (YYYY-MM-DD) or ISO datetime strings."""
    if "T" in s or ":" in s:
        return datetime.fromisoformat(s)
    return datetime.strptime(s, "%Y-%m-%d")


def _is_date_only(s: str) -> bool:
    """Check if string is date-only (YYYY-MM-DD) vs ISO datetime."""
    # Defensive: callers might pass unvalidated data.
    return isinstance(s, str) and "T" not in s and ":" not in s


def _window_months(start: datetime, end: datetime) -> int:
    """Return an approximate window length in whole months.

    Uses dateutil.relativedelta when available for accurate month arithmetic.
    The result is "inclusive-ish": leftover days beyond full months count as +1.

    Examples:
      - 2020-01-01 -> 2020-02-01 => 1 month
      - 2020-01-01 -> 2024-12-31 => 60 months (4y11m + leftover days => +1)
    """
    try:
        rd = relativedelta(end, start)
        months = int(rd.years) * 12 + int(rd.months)
        has_remainder = bool(
            getattr(rd, "days", 0)
            or getattr(rd, "hours", 0)
            or getattr(rd, "minutes", 0)
            or getattr(rd, "seconds", 0)
            or getattr(rd, "microseconds", 0)
        )
        if has_remainder:
            months += 1
        return max(0, months)
    except (TypeError, ValueError, AttributeError):
        pass

    # Fallback: approximate months by days.
    try:
        days = max(0.0, (end - start).total_seconds() / 86400.0)
        approx = int(days / 30.4375)
        if days % 30.4375:
            approx += 1
        return max(0, approx)
    except (TypeError, ValueError):
        return 0


def get_timing_jitter_backward_shift_months(
    *,
    start_date: str,
    end_date: str,
    divisors: Sequence[int] = (10, 5, 20),
    min_months: int = 1,
) -> list[int]:
    """Compute backward timing-jitter shift sizes in months.

    This implements the new approach:
      - Determine total window length in months.
      - Compute shifts for each divisor (total_months // divisor).
      - Enforce a minimum bound (>= min_months).

    The caller typically runs 3 additional backtests, each with start/end shifted
    BACKWARD by one of the returned month values.
    """
    try:
        start = _parse_date_string(start_date)
        end = _parse_date_string(end_date)
    except (ValueError, TypeError):
        return []

    total_months = _window_months(start, end)
    if total_months <= 0:
        return []

    out: list[int] = []
    for d in divisors:
        try:
            div = int(d)
        except (TypeError, ValueError):
            continue
        if div <= 0:
            continue
        out.append(max(int(min_months), total_months // div))
    return out


def apply_timing_jitter_month_shift_inplace(
    cfg: Any,
    *,
    shift_months_backward: int,
) -> None:
    """Shift start/end dates BACKWARD by a given number of months.

    Backward-only shifts ensure no future data leakage in robustness testing.

    - Both `start_date` and `end_date` are shifted by the same interval.
    - Works with date-only strings (YYYY-MM-DD) and ISO datetime strings.
    - Uses dateutil.relativedelta when available; otherwise falls back to 30-day months.
    """
    if not isinstance(cfg, dict):
        return

    m = shift_months_backward
    if m <= 0:
        return

    start_s = cfg.get("start_date")
    end_s = cfg.get("end_date")
    if (
        not start_s
        or not end_s
        or not isinstance(start_s, str)
        or not isinstance(end_s, str)
    ):
        return

    date_only = _is_date_only(start_s) and _is_date_only(end_s)
    try:
        start = _parse_date_string(start_s)
        end = _parse_date_string(end_s)

        try:
            rd_offset = relativedelta(months=-m)
            new_start = start + rd_offset
            new_end = end + rd_offset
        except Exception:
            td_offset = timedelta(days=30 * m)
            new_start = start - td_offset
            new_end = end - td_offset

        if date_only:
            cfg["start_date"] = new_start.strftime("%Y-%m-%d")
            cfg["end_date"] = new_end.strftime("%Y-%m-%d")
        else:
            cfg["start_date"] = new_start.isoformat(timespec="seconds")
            cfg["end_date"] = new_end.isoformat(timespec="seconds")
    except (ValueError, TypeError):
        return


def compute_timing_jitter_score(
    base_metrics: Mapping[str, float],
    jitter_metrics: Sequence[Mapping[str, float]],
    *,
    penalty_cap: float = 0.5,
) -> float:
    """Compute timing jitter robustness score from base and jittered metrics."""
    penalty = compute_penalty_profit_drawdown_sharpe(
        base_metrics, jitter_metrics, penalty_cap=penalty_cap
    )
    return float(score_from_penalty(penalty, penalty_cap=penalty_cap))
