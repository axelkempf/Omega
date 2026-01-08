//! Common helper functions for rating module calculations.
//!
//! Provides shared utilities used across multiple rating score modules.
//! These functions handle numerical safety guards (NaN, Inf) and relative
//! change calculations.

/// Convert any value to finite f64, returning default if NaN or Inf.
///
/// Mirrors Python's `_to_finite()` helper used in rating modules.
///
/// # Arguments
/// * `x` - Value to convert
/// * `default` - Default value to return if x is not finite
///
/// # Returns
/// `x` if finite, otherwise `default`
#[inline]
pub fn to_finite(x: f64, default: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        default
    }
}

/// Calculate percentage drop from base to value.
///
/// Mirrors Python's `_pct_drop()` helper.
///
/// # Arguments
/// * `base` - Base value (denominator, guarded to >= 1e-9)
/// * `value` - Compared value
/// * `invert` - If true, calculate increase instead of drop
///
/// # Returns
/// Relative change as fraction [0.0, ∞), clamped to max(0.0, result)
#[inline]
pub fn pct_drop(base: f64, value: f64, invert: bool) -> f64 {
    let base_safe = base.max(1e-9);
    let value_safe = value.max(0.0);

    if invert {
        // For drawdown: increase is bad
        ((value_safe - base_safe) / base_safe).max(0.0)
    } else {
        // For profit/winrate/sharpe: drop is bad
        ((base_safe - value_safe) / base_safe).max(0.0)
    }
}

/// Calculate relative drop between base and stress value.
///
/// Used by `stress_penalty` calculations.
///
/// # Arguments
/// * `base` - Base value
/// * `stress` - Stress-tested value
///
/// # Returns
/// Relative drop as fraction [0.0, ∞)
#[inline]
pub fn rel_drop(base: f64, stress: f64) -> f64 {
    if base <= 0.0 {
        0.0
    } else {
        ((base - stress) / base).max(0.0)
    }
}

/// Calculate relative increase from base to stress value.
///
/// Used for metrics where increase is bad (e.g., drawdown).
///
/// # Arguments
/// * `base` - Base value (guarded to >= 1e-9)
/// * `stress` - Stress-tested value
///
/// # Returns
/// Relative increase as fraction [0.0, ∞)
#[inline]
pub fn rel_increase(base: f64, stress: f64) -> f64 {
    let base_safe = if base <= 0.0 { 1e-9 } else { base };
    ((stress - base_safe) / base_safe).max(0.0)
}

/// Calculate mean of a slice, handling empty slices.
///
/// # Arguments
/// * `values` - Slice of values to average
///
/// # Returns
/// Mean value, or 0.0 if slice is empty
#[inline]
pub fn safe_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    let count = values.len() as f64;
    let mean = sum / count;
    if mean.is_finite() {
        mean
    } else {
        0.0
    }
}

/// Calculate nanmean of a slice, filtering out NaN values.
///
/// # Arguments
/// * `values` - Slice of values to average
///
/// # Returns
/// Mean of finite values, or NaN if no finite values exist
#[inline]
pub fn nan_mean(values: &[f64]) -> f64 {
    let finite_values: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite_values.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = finite_values.iter().sum();
    sum / finite_values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_to_finite() {
        assert_relative_eq!(to_finite(1.5, 0.0), 1.5, epsilon = 1e-10);
        assert_relative_eq!(to_finite(f64::NAN, 0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(to_finite(f64::INFINITY, -1.0), -1.0, epsilon = 1e-10);
        assert_relative_eq!(to_finite(f64::NEG_INFINITY, 2.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pct_drop_normal() {
        // 100 -> 80 = 20% drop
        assert_relative_eq!(pct_drop(100.0, 80.0, false), 0.2, epsilon = 1e-10);
        // 100 -> 120 = no drop (we return 0.0 for increases when not inverted)
        assert_relative_eq!(pct_drop(100.0, 120.0, false), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pct_drop_inverted() {
        // Inverted: 100 -> 120 = 20% increase
        assert_relative_eq!(pct_drop(100.0, 120.0, true), 0.2, epsilon = 1e-10);
        // Inverted: 100 -> 80 = no increase
        assert_relative_eq!(pct_drop(100.0, 80.0, true), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pct_drop_edge_cases() {
        // Base <= 0 should be guarded to 1e-9
        let result = pct_drop(0.0, 50.0, false);
        assert!(result >= 0.0);

        // Negative value is clamped to 0 first (value_safe = max(0, value))
        // pct_drop(100, 0, false) = (100-0)/100 = 1.0
        assert_relative_eq!(pct_drop(100.0, -50.0, false), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rel_drop() {
        assert_relative_eq!(rel_drop(100.0, 80.0), 0.2, epsilon = 1e-10);
        assert_relative_eq!(rel_drop(100.0, 100.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rel_drop(0.0, 50.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rel_increase() {
        assert_relative_eq!(rel_increase(100.0, 120.0), 0.2, epsilon = 1e-10);
        assert_relative_eq!(rel_increase(100.0, 100.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rel_increase(100.0, 80.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_safe_mean() {
        assert_relative_eq!(safe_mean(&[1.0, 2.0, 3.0]), 2.0, epsilon = 1e-10);
        assert_relative_eq!(safe_mean(&[]), 0.0, epsilon = 1e-10);
        assert_relative_eq!(safe_mean(&[5.0]), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_mean() {
        assert_relative_eq!(nan_mean(&[1.0, 2.0, 3.0]), 2.0, epsilon = 1e-10);
        assert_relative_eq!(nan_mean(&[1.0, f64::NAN, 3.0]), 2.0, epsilon = 1e-10);
        assert!(nan_mean(&[f64::NAN, f64::NAN]).is_nan());
        assert!(nan_mean(&[]).is_nan());
    }
}
