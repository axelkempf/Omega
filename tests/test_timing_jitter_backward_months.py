from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from unittest.mock import patch

import pytest

from backtest_engine.rating.timing_jitter_score import (
    _is_date_only,
    _parse_date_string,
    _window_months,
    apply_timing_jitter_month_shift_inplace,
    get_timing_jitter_backward_shift_months,
)


def test_get_timing_jitter_backward_shift_months_5y_window_is_scaled_and_clamped():
    """A ~5-year window should produce meaningful month shifts.

    2020-01-01 -> 2024-12-31 is treated as ~60 months (inclusive-ish) and thus:
      /10 -> 6
      /5  -> 12
      /20 -> 3
    """
    shifts = get_timing_jitter_backward_shift_months(
        start_date="2020-01-01",
        end_date="2024-12-31",
        divisors=(10, 5, 20),
        min_months=1,
    )
    assert shifts == [6, 12, 3]


def test_get_timing_jitter_backward_shift_months_minimum_bound_is_enforced():
    """Very short windows must still produce >= 1 month shifts."""
    shifts = get_timing_jitter_backward_shift_months(
        start_date="2024-01-01",
        end_date="2024-01-15",
        divisors=(10, 5, 20),
        min_months=1,
    )
    assert shifts == [1, 1, 1]


def test_apply_timing_jitter_month_shift_inplace_shifts_both_start_and_end_backward():
    """Start and end must be shifted by the same amount (window length preserved)."""
    cfg = {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "timeframes": {"primary": "D1"},
    }
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=6)

    # dateutil.relativedelta handles month length (Dec 31 -> Jun 30)
    assert cfg["start_date"] == "2023-07-01"
    assert cfg["end_date"] == "2024-06-30"


def test_apply_timing_jitter_month_shift_inplace_non_positive_is_noop():
    cfg = {"start_date": "2024-01-01", "end_date": "2024-12-31"}
    before = deepcopy(cfg)
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=0)
    assert cfg == before


def test_apply_timing_jitter_month_shift_inplace_invalid_dates_are_ignored():
    cfg = {"start_date": "not-a-date", "end_date": "2024-12-31"}
    before = deepcopy(cfg)
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=6)
    assert cfg == before


# =============================================================================
# Tests für _window_months() Helper
# =============================================================================


def test_window_months_exact_one_year():
    """Exactly 12 months from Jan 1 to Jan 1 next year."""
    start = datetime(2020, 1, 1)
    end = datetime(2021, 1, 1)
    assert _window_months(start, end) == 12


def test_window_months_exact_one_month():
    """Exactly 1 month from Jan 1 to Feb 1."""
    start = datetime(2020, 1, 1)
    end = datetime(2020, 2, 1)
    assert _window_months(start, end) == 1


def test_window_months_with_remainder_days():
    """12 months + leftover days should round up to 13."""
    start = datetime(2020, 1, 1)
    end = datetime(2021, 1, 15)  # 12 months + 14 days
    assert _window_months(start, end) == 13


def test_window_months_five_years_with_remainder():
    """~60 months (5 years) - tests docstring example."""
    start = datetime(2020, 1, 1)
    end = datetime(2024, 12, 31)  # 4y11m + 30 days => 60
    result = _window_months(start, end)
    assert result == 60


def test_window_months_zero_for_same_date():
    """Same start and end should return 0."""
    start = datetime(2024, 6, 15)
    end = datetime(2024, 6, 15)
    assert _window_months(start, end) == 0


def test_window_months_negative_interval_returns_zero():
    """End before start should return 0 (not negative)."""
    start = datetime(2024, 6, 15)
    end = datetime(2024, 1, 1)
    assert _window_months(start, end) == 0


def test_window_months_short_interval_rounds_up():
    """A few days should still return 1 month (ceiling behavior)."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 10)  # 9 days
    assert _window_months(start, end) == 1


# =============================================================================
# Tests für _parse_date_string() Helper
# =============================================================================


def test_parse_date_string_date_only():
    """Parse YYYY-MM-DD format."""
    result = _parse_date_string("2024-06-15")
    assert result == datetime(2024, 6, 15)


def test_parse_date_string_iso_datetime():
    """Parse ISO datetime with T separator."""
    result = _parse_date_string("2024-06-15T10:30:00")
    assert result == datetime(2024, 6, 15, 10, 30, 0)


def test_parse_date_string_iso_datetime_with_microseconds():
    """Parse ISO datetime with microseconds."""
    result = _parse_date_string("2024-06-15T10:30:00.123456")
    assert result == datetime(2024, 6, 15, 10, 30, 0, 123456)


def test_parse_date_string_invalid_raises():
    """Invalid date string should raise ValueError."""
    with pytest.raises(ValueError):
        _parse_date_string("not-a-date")


# =============================================================================
# Tests für _is_date_only() Helper
# =============================================================================


def test_is_date_only_returns_true_for_date():
    """YYYY-MM-DD should be recognized as date-only."""
    assert _is_date_only("2024-06-15") is True


def test_is_date_only_returns_false_for_iso_datetime():
    """ISO datetime with T should not be date-only."""
    assert _is_date_only("2024-06-15T10:30:00") is False


def test_is_date_only_returns_false_for_time_with_colon():
    """String with colon should not be date-only."""
    assert _is_date_only("10:30:00") is False


def test_is_date_only_returns_false_for_non_string():
    """Non-string input should return False (defensive)."""
    assert _is_date_only(None) is False  # type: ignore[arg-type]
    assert _is_date_only(12345) is False  # type: ignore[arg-type]


# =============================================================================
# Tests für ISO Datetime Shift
# =============================================================================


def test_apply_timing_jitter_shift_with_iso_datetime():
    """Shift should work with full ISO datetime strings."""
    cfg = {
        "start_date": "2024-01-15T10:30:00",
        "end_date": "2024-12-15T18:00:00",
    }
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=3)

    # Should preserve ISO format with T separator
    assert "T" in cfg["start_date"]
    assert "T" in cfg["end_date"]
    # Dates shifted back by 3 months
    assert cfg["start_date"] == "2023-10-15T10:30:00"
    assert cfg["end_date"] == "2024-09-15T18:00:00"


def test_apply_timing_jitter_shift_mixed_formats_uses_iso():
    """If either date has time component, output should be ISO."""
    cfg = {
        "start_date": "2024-01-15T10:30:00",
        "end_date": "2024-12-15",  # date-only
    }
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=1)

    # Mixed input: date_only = False because start has T
    assert "T" in cfg["start_date"]


# =============================================================================
# Tests für Fallback ohne relativedelta
# =============================================================================


def test_window_months_fallback_without_relativedelta():
    """Test fallback path when relativedelta is None."""
    with patch("backtest_engine.rating.timing_jitter_score.relativedelta", None):
        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)
        result = _window_months(start, end)
        # Fallback uses 30.4375 days/month: 365 / 30.4375 ≈ 11.99 + remainder -> 12
        # But 366 days (leap year 2020) / 30.4375 ≈ 12.02 + remainder -> 13
        assert result == 13  # Leap year 2020 has 366 days


def test_window_months_fallback_with_remainder():
    """Test fallback rounds up partial months."""
    with patch("backtest_engine.rating.timing_jitter_score.relativedelta", None):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 20)  # ~19 days
        result = _window_months(start, end)
        # 19 / 30.4375 ≈ 0.62 -> rounds up to 1
        assert result == 1


def test_apply_timing_jitter_shift_fallback_without_relativedelta():
    """Shift should work using timedelta fallback."""
    with patch("backtest_engine.rating.timing_jitter_score.relativedelta", None):
        cfg = {
            "start_date": "2024-06-15",
            "end_date": "2024-12-15",
        }
        apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=2)

        # Fallback uses 30-day months: 2 months = 60 days back
        # 2024-06-15 - 60 days = 2024-04-16
        # 2024-12-15 - 60 days = 2024-10-16
        assert cfg["start_date"] == "2024-04-16"
        assert cfg["end_date"] == "2024-10-16"


# =============================================================================
# Tests für Edge Cases und Defensive Checks
# =============================================================================


def test_apply_timing_jitter_shift_non_dict_is_noop():
    """Non-dict config should be ignored (defensive)."""
    cfg = ["not", "a", "dict"]
    original = cfg.copy()
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=1)  # type: ignore[arg-type]
    assert cfg == original


def test_apply_timing_jitter_shift_missing_dates_is_noop():
    """Config without start_date or end_date should be ignored."""
    cfg = {"timeframes": {"primary": "D1"}}
    before = deepcopy(cfg)
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=1)
    assert cfg == before


def test_apply_timing_jitter_shift_non_string_dates_is_noop():
    """Non-string date values should be ignored."""
    cfg = {"start_date": 20240101, "end_date": 20241231}
    before = deepcopy(cfg)
    apply_timing_jitter_month_shift_inplace(cfg, shift_months_backward=1)
    assert cfg == before


def test_get_timing_jitter_backward_shift_invalid_divisor_skipped():
    """Invalid divisors (zero, negative) should be skipped."""
    shifts = get_timing_jitter_backward_shift_months(
        start_date="2020-01-01",
        end_date="2024-12-31",
        divisors=(10, 0, -5, 5),  # 0 and -5 should be skipped
        min_months=1,
    )
    assert shifts == [6, 12]  # Only valid divisors: 10 and 5


def test_get_timing_jitter_backward_shift_empty_divisors():
    """Empty divisors list should return empty result."""
    shifts = get_timing_jitter_backward_shift_months(
        start_date="2020-01-01",
        end_date="2024-12-31",
        divisors=(),
        min_months=1,
    )
    assert shifts == []


def test_get_timing_jitter_backward_shift_non_string_dates():
    """Non-string date inputs should return empty list."""
    shifts = get_timing_jitter_backward_shift_months(
        start_date=20240101,  # type: ignore[arg-type]
        end_date="2024-12-31",
        divisors=(10,),
        min_months=1,
    )
    assert shifts == []
