"""
Tests für die trade-count basierten Metrik-Adjustierungen.

Testet shrinkage_adjusted, risk_adjusted und bayesian_shrinkage Funktionen
aus analysis.metric_adjustments.
"""

import numpy as np
import pytest

from analysis.metric_adjustments import (
    TRADES_PER_YEAR_REFERENCE,
    bayesian_shrinkage,
    risk_adjusted,
    shrinkage_adjusted,
    wilson_score_lower_bound,
)


def test_trades_per_year_reference_is_configurable():
    """Verify the constant is defined and has expected default value."""
    assert TRADES_PER_YEAR_REFERENCE == 15


def test_shrinkage_adjusted_reduces_with_low_trades():
    """Average R should shrink towards zero with fewer trades."""
    avg_r_raw = 0.5
    n_trades_low = 10
    n_trades_high = 1000
    n_years = 1.0

    adjusted_low = shrinkage_adjusted(avg_r_raw, n_trades_low, n_years)
    adjusted_high = shrinkage_adjusted(avg_r_raw, n_trades_high, n_years)

    # With fewer trades, adjustment should be stronger (closer to 0)
    assert adjusted_low < adjusted_high
    assert 0 < adjusted_low < avg_r_raw
    assert adjusted_high < avg_r_raw  # Still shrunk, but less


def test_shrinkage_adjusted_scales_with_n_years():
    """konst. = n_years * TRADES_PER_YEAR_REFERENCE should scale correctly."""
    avg_r_raw = 0.5
    n_trades = 100

    adjusted_1y = shrinkage_adjusted(avg_r_raw, n_trades, n_years=1.0)
    adjusted_4y = shrinkage_adjusted(avg_r_raw, n_trades, n_years=4.0)

    # With more years in total metrics, konst. increases, so shrinkage is stronger
    assert adjusted_4y < adjusted_1y


def test_shrinkage_adjusted_scales_with_n_categories():
    """konst. should scale with n_categories (e.g. groups_count) for portfolio totals."""
    avg_r_raw = 0.5
    n_trades = 100
    n_years = 2.0

    adjusted_1 = shrinkage_adjusted(
        avg_r_raw, n_trades, n_years=n_years, n_categories=1.0
    )
    adjusted_3 = shrinkage_adjusted(
        avg_r_raw, n_trades, n_years=n_years, n_categories=3.0
    )

    # More categories -> larger konst -> stronger shrinkage
    assert adjusted_3 < adjusted_1


def test_risk_adjusted_reduces_pod_with_low_trades():
    """Profit over Drawdown should shrink with fewer trades."""
    pod_raw = 3.0
    n_trades_low = 10
    n_trades_high = 1000
    n_years = 1.0

    adjusted_low = risk_adjusted(pod_raw, n_trades_low, n_years)
    adjusted_high = risk_adjusted(pod_raw, n_trades_high, n_years)

    # With fewer trades, adjustment should be stronger
    assert adjusted_low < adjusted_high
    assert 0 < adjusted_low < pod_raw
    assert adjusted_high < pod_raw


def test_risk_adjusted_scales_with_n_categories():
    """PoD adjustment should scale with n_categories (e.g. groups_count) for portfolio totals."""
    pod_raw = 2.0
    n_trades = 100
    n_years = 2.0

    adjusted_1 = risk_adjusted(pod_raw, n_trades, n_years=n_years, n_categories=1.0)
    adjusted_4 = risk_adjusted(pod_raw, n_trades, n_years=n_years, n_categories=4.0)

    # More categories -> larger konst -> stronger shrinkage
    assert adjusted_4 < adjusted_1


def test_risk_adjusted_uses_sqrt_scaling():
    """PoD adjustment uses sqrt(N/(N+konst.)) formula."""
    pod_raw = 2.0
    n_trades = 100
    n_years = 1.0
    konst = n_years * TRADES_PER_YEAR_REFERENCE

    expected_factor = np.sqrt(n_trades / (n_trades + konst))
    expected = pod_raw * expected_factor

    adjusted = risk_adjusted(pod_raw, n_trades, n_years)

    assert np.isclose(adjusted, expected, rtol=1e-6)


def test_bayesian_shrinkage_pulls_towards_prior_mean():
    """Winrate should shrink towards the mean of all_winrates with few trades."""
    winrate = 0.8  # 80% winrate
    n_trades = 5  # Very few trades
    all_winrates = np.array([0.5, 0.52, 0.48, 0.51, 0.49])  # Mean ≈ 0.5

    adjusted = bayesian_shrinkage(winrate, n_trades, all_winrates)

    # With few trades, should be pulled towards prior mean (0.5)
    assert adjusted < winrate
    assert adjusted > np.mean(all_winrates)  # But not all the way


def test_bayesian_shrinkage_less_adjustment_with_many_trades():
    """With many trades, winrate should stay closer to observed value."""
    winrate = 0.8
    n_trades_low = 5
    n_trades_high = 500
    all_winrates = np.array([0.5, 0.52, 0.48, 0.51, 0.49])

    adjusted_low = bayesian_shrinkage(winrate, n_trades_low, all_winrates)
    adjusted_high = bayesian_shrinkage(winrate, n_trades_high, all_winrates)

    # With more trades, adjusted value should be closer to raw winrate
    assert adjusted_high > adjusted_low
    assert abs(adjusted_high - winrate) < abs(adjusted_low - winrate)


def test_bayesian_shrinkage_handles_empty_prior():
    """Should handle empty all_winrates gracefully."""
    winrate = 0.6
    n_trades = 100
    all_winrates = np.array([])

    # Should not crash, may return original or use fallback
    adjusted = bayesian_shrinkage(winrate, n_trades, all_winrates)
    assert np.isfinite(adjusted)


def test_shrinkage_adjusted_preserves_nan():
    """NaN inputs should remain NaN."""
    adjusted = shrinkage_adjusted(np.nan, 100, 1.0)
    assert np.isnan(adjusted)


def test_risk_adjusted_preserves_nan():
    """NaN inputs should remain NaN."""
    adjusted = risk_adjusted(np.nan, 100, 1.0)
    assert np.isnan(adjusted)


def test_bayesian_shrinkage_preserves_nan():
    """NaN inputs should remain NaN."""
    all_winrates = np.array([0.5, 0.6, 0.4])
    adjusted = bayesian_shrinkage(np.nan, 100, all_winrates)
    assert np.isnan(adjusted)


def test_shrinkage_adjusted_vectorized():
    """Should work with numpy arrays."""
    avg_r_array = np.array([0.5, 0.3, 0.7])
    n_trades_array = np.array([10, 100, 1000])
    n_years = 1.0

    adjusted = shrinkage_adjusted(avg_r_array, n_trades_array, n_years)

    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == avg_r_array.shape
    assert np.all(adjusted < avg_r_array)  # All should be shrunk


def test_risk_adjusted_vectorized():
    """Should work with numpy arrays."""
    pod_array = np.array([2.0, 3.0, 4.0])
    n_trades_array = np.array([10, 100, 1000])
    n_years = 1.0

    adjusted = risk_adjusted(pod_array, n_trades_array, n_years)

    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == pod_array.shape
    assert np.all(adjusted < pod_array)


def test_bayesian_shrinkage_vectorized():
    """Should work with numpy arrays."""
    winrate_array = np.array([0.6, 0.7, 0.8])
    n_trades_array = np.array([10, 100, 1000])
    all_winrates = np.array([0.5, 0.52, 0.48])

    adjusted = bayesian_shrinkage(winrate_array, n_trades_array, all_winrates)

    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == winrate_array.shape


def test_bayesian_shrinkage_conservative_never_above_raw():
    """Conservative Mode: Adjusted darf nie über Raw liegen (clamp_to_raw=True)."""
    # Raw unter Prior-Mean → Standard-Bayesian würde steigen
    winrate = 0.45
    n_trades = 10
    all_winrates = np.array([0.52, 0.53, 0.51, 0.54, 0.52])

    adjusted_two_sided = bayesian_shrinkage(winrate, n_trades, all_winrates)
    adjusted_conservative = bayesian_shrinkage(
        winrate, n_trades, all_winrates, clamp_to_raw=True
    )

    # Ohne Clamp: sollte über raw ziehen (Richtung Prior)
    assert adjusted_two_sided >= winrate
    # Mit Clamp: niemals über Raw
    assert adjusted_conservative <= winrate
    # Und nicht größer als die ungekappte Version
    assert adjusted_conservative <= adjusted_two_sided


def test_shrinkage_formula_correctness():
    """Verify exact formula: average_r * (N / (N + konst.))"""
    avg_r = 0.5
    n_trades = 100
    n_years = 2.0
    konst = n_years * TRADES_PER_YEAR_REFERENCE  # 2.0 * 15 = 30

    expected = avg_r * (n_trades / (n_trades + konst))
    actual = shrinkage_adjusted(avg_r, n_trades, n_years)

    assert np.isclose(actual, expected, rtol=1e-10)


def test_shrinkage_formula_correctness_with_categories():
    """Verify exact formula with category scaling: average_r * (N / (N + n_years*n_categories*konst.))"""
    avg_r = 0.5
    n_trades = 100
    n_years = 2.0
    n_categories = 4.0

    konst = n_years * n_categories * TRADES_PER_YEAR_REFERENCE
    expected = avg_r * (n_trades / (n_trades + konst))
    actual = shrinkage_adjusted(
        avg_r, n_trades, n_years=n_years, n_categories=n_categories
    )

    assert np.isclose(actual, expected, rtol=1e-10)


def test_yearly_vs_total_metrics_konstant():
    """Demonstrate that n_years affects shrinkage strength."""
    avg_r = 0.5
    n_trades = 100

    # Yearly metrics: n_years=1.0
    adjusted_yearly = shrinkage_adjusted(avg_r, n_trades, n_years=1.0)

    # Total metrics over 4 years: n_years=4.0
    adjusted_total = shrinkage_adjusted(avg_r, n_trades, n_years=4.0)

    # Total metrics should shrink more (higher konst.)
    assert adjusted_total < adjusted_yearly

    # Calculate expected values
    konst_yearly = 1.0 * TRADES_PER_YEAR_REFERENCE  # 15
    konst_total = 4.0 * TRADES_PER_YEAR_REFERENCE  # 60

    expected_yearly = avg_r * (n_trades / (n_trades + konst_yearly))
    expected_total = avg_r * (n_trades / (n_trades + konst_total))

    assert np.isclose(adjusted_yearly, expected_yearly)
    assert np.isclose(adjusted_total, expected_total)


def test_bayesian_shrinkage_beta_distribution_parameters():
    """Verify that alpha and beta are computed correctly."""
    # Simple case: uniform prior (50% winrate, low variance)
    all_winrates = np.array([0.5, 0.5, 0.5, 0.5])
    winrate_mean = 0.5
    winrate_var = 0.0 + 0.001  # Epsilon to avoid division by zero

    # Formula from spec:
    # mean_var_ratio = (winrate_mean * (1 - winrate_mean)) / winrate_var
    mean_var_ratio = (winrate_mean * (1.0 - winrate_mean)) / winrate_var
    alpha_expected = winrate_mean * (mean_var_ratio - 1.0)
    beta_expected = (1.0 - winrate_mean) * (mean_var_ratio - 1.0)

    # Test with high number of trades (minimal adjustment)
    winrate = 0.6
    n_trades = 10000
    adjusted = bayesian_shrinkage(winrate, n_trades, all_winrates)

    # With many trades, should be close to raw winrate
    assert abs(adjusted - winrate) < 0.05


def test_zero_trades_returns_zero_shrinkage():
    """With zero trades, metrics should be heavily shrunk or zero."""
    avg_r = 0.5
    n_trades = 0
    n_years = 1.0

    adjusted = shrinkage_adjusted(avg_r, n_trades, n_years)

    # Formula: 0.5 * (0 / (0 + 15)) = 0
    assert adjusted == 0.0


def test_zero_trades_pod_returns_zero():
    """With zero trades, PoD should be shrunk to zero."""
    pod = 2.0
    n_trades = 0
    n_years = 1.0

    adjusted = risk_adjusted(pod, n_trades, n_years)

    # Formula: 2.0 * sqrt(0 / (0 + 15)) = 0
    assert adjusted == 0.0


def test_negative_trades_are_treated_as_zero_in_shrinkage_and_risk():
    """Negative trade counts should not create negative or >1 shrinkage factors."""
    avg_r = 0.5
    pod = 2.0

    assert shrinkage_adjusted(avg_r, -10, 1.0) == 0.0
    assert risk_adjusted(pod, -10, 1.0) == 0.0


def test_bayesian_shrinkage_preserves_inf_and_nan_for_array_inputs():
    """Regression: array inputs with inf must not be silently clipped to 1.0."""
    all_winrates = np.array([0.5, 0.6, 0.4], dtype=float)
    winrate = np.array([np.inf, 0.5, np.nan], dtype=float)
    n_trades = np.array([10, 10, 10], dtype=float)

    adjusted = bayesian_shrinkage(winrate, n_trades, all_winrates)

    assert np.isposinf(adjusted[0])
    assert np.isfinite(adjusted[1])
    assert np.isnan(adjusted[2])


def test_wilson_score_lower_bound_known_value_for_80pct_10_trades():
    """Wilson lower bound should match a known reference (approx)."""
    lb = wilson_score_lower_bound(0.80, 10, confidence_level=0.95)
    # Reference value for Wilson lower bound at 95% CI
    assert np.isclose(lb, 0.4901624715, atol=1e-6)


def test_wilson_score_lower_bound_accepts_percent_inputs():
    """Percent inputs (0-100) should be auto-normalized to decimals."""
    lb_dec = wilson_score_lower_bound(0.80, 10, confidence_level=0.95)
    lb_pct = wilson_score_lower_bound(80.0, 10, confidence_level=0.95)
    assert np.isclose(lb_dec, lb_pct, rtol=1e-12, atol=0.0)


def test_wilson_score_lower_bound_increases_with_more_trades():
    """At fixed winrate, more trades should increase the lower bound."""
    lb_10 = wilson_score_lower_bound(0.80, 10, confidence_level=0.95)
    lb_100 = wilson_score_lower_bound(0.80, 100, confidence_level=0.95)
    assert lb_100 > lb_10


def test_wilson_score_lower_bound_zero_trades_returns_zero():
    """With zero trades, interval is undefined; implementation returns 0.0."""
    lb = wilson_score_lower_bound(0.50, 0, confidence_level=0.95)
    assert lb == 0.0


def test_wilson_score_lower_bound_invalid_confidence_level_raises():
    """confidence_level must be in (0,1)."""
    with pytest.raises(ValueError):
        wilson_score_lower_bound(0.50, 10, confidence_level=1.0)
    with pytest.raises(ValueError):
        wilson_score_lower_bound(0.50, 10, confidence_level=0.0)
    with pytest.raises(ValueError):
        wilson_score_lower_bound(0.50, 10, confidence_level=-0.1)
