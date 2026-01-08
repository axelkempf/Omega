//! Stability Score calculation module.
//!
//! Computes yearly profit stability score based on weighted mean absolute
//! percentage error (WMAPE) against expected yearly profits.
//!
//! ## Formula
//!
//! ```text
//! mu = total_profit / total_days
//! s_min = max(100, 0.02 * |total_profit|)
//!
//! For each year y:
//!     e_y = mu * days_y          # expected profit
//!     r_y = |P_y - e_y| / max(|e_y|, s_min)   # relative error
//!     w_y = days_y / total_days  # weight
//!
//! wmape = sum(w_y * r_y)
//! score = 1 / (1 + wmape)
//! ```
//!
//! ## Reference
//!
//! Python implementation: `src/backtest_engine/rating/stability_score.py`

use pyo3::prelude::*;
use std::collections::HashMap;

/// Check if year is a leap year.
#[inline]
const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0))
}

/// Get days in year (365 or 366).
#[inline]
const fn days_in_year(year: i32) -> f64 {
    if is_leap_year(year) {
        366.0
    } else {
        365.0
    }
}

/// Compute stability score and WMAPE from yearly profits.
///
/// # Arguments
/// * `profits_by_year` - Map of year -> profit
/// * `durations_by_year` - Optional map of year -> actual duration in days
///
/// # Returns
/// Tuple of (score, wmape)
#[allow(clippy::implicit_hasher)]
pub fn compute_stability_score_and_wmape_impl(
    profits_by_year: &HashMap<i32, f64>,
    durations_by_year: Option<&HashMap<i32, f64>>,
) -> (f64, f64) {
    if profits_by_year.is_empty() {
        return (1.0, 0.0);
    }

    // Sort years for deterministic processing
    let mut years: Vec<i32> = profits_by_year.keys().copied().collect();
    years.sort_unstable();

    // Calculate durations
    let mut durations: HashMap<i32, f64> = HashMap::new();
    for &year in &years {
        #[allow(clippy::option_if_let_else)]
        let d = if let Some(dur_map) = durations_by_year {
            dur_map.get(&year).copied().filter(|&v| v.is_finite() && v > 0.0)
        } else {
            None
        };
        let duration = d.unwrap_or_else(|| days_in_year(year));
        durations.insert(year, duration);
    }

    let d_total: f64 = durations.values().sum();
    if d_total <= 0.0 || !d_total.is_finite() {
        return (1.0, 0.0);
    }

    // Calculate profits and total
    let profits: HashMap<i32, f64> = years
        .iter()
        .map(|&y| {
            let p = profits_by_year.get(&y).copied().unwrap_or(0.0);
            (y, if p.is_finite() { p } else { 0.0 })
        })
        .collect();

    let p_total: f64 = profits.values().sum();
    let mu = if d_total > 0.0 { p_total / d_total } else { 0.0 };
    let s_min = 100.0_f64.max(0.02 * p_total.abs());

    // Calculate WMAPE
    let mut wmape = 0.0;
    for &year in &years {
        let d_y = durations.get(&year).copied().unwrap_or(0.0);
        if d_y <= 0.0 {
            continue;
        }

        let p_y = profits.get(&year).copied().unwrap_or(0.0);
        let e_y = mu * d_y;
        let denom = e_y.abs().max(s_min);
        let r_y = if denom > 0.0 {
            (p_y - e_y).abs() / denom
        } else {
            0.0
        };
        let w_y = d_y / d_total;
        wmape += w_y * r_y;
    }

    if !wmape.is_finite() {
        return (1.0, 0.0);
    }

    let score = 1.0 / (1.0 + wmape);
    if !score.is_finite() {
        return (1.0, 0.0);
    }

    (score, wmape)
}

/// Compute stability score only (convenience wrapper).
#[allow(clippy::implicit_hasher)]
pub fn compute_stability_score_impl(
    profits_by_year: &HashMap<i32, f64>,
    durations_by_year: Option<&HashMap<i32, f64>>,
) -> f64 {
    compute_stability_score_and_wmape_impl(profits_by_year, durations_by_year).0
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Compute stability score and WMAPE from yearly profits.
///
/// # Arguments
/// * `years` - List of years
/// * `profits` - List of profits (parallel to years)
/// * `durations` - Optional list of durations (parallel to years)
///
/// # Returns
/// Tuple of (score, wmape)
#[pyfunction]
#[pyo3(signature = (years, profits, durations = None))]
#[allow(clippy::needless_pass_by_value, clippy::missing_errors_doc)]
pub fn compute_stability_score_and_wmape(
    years: Vec<i32>,
    profits: Vec<f64>,
    durations: Option<Vec<f64>>,
) -> PyResult<(f64, f64)> {
    let len = years.len().min(profits.len());

    let profits_by_year: HashMap<i32, f64> = years
        .iter()
        .take(len)
        .zip(profits.iter().take(len))
        .map(|(&y, &p)| (y, p))
        .collect();

    let durations_by_year: Option<HashMap<i32, f64>> = durations.map(|dur| {
        let dur_len = len.min(dur.len());
        years
            .iter()
            .take(dur_len)
            .zip(dur.iter().take(dur_len))
            .map(|(&y, &d)| (y, d))
            .collect()
    });

    Ok(compute_stability_score_and_wmape_impl(
        &profits_by_year,
        durations_by_year.as_ref(),
    ))
}

/// Compute stability score from yearly profits.
///
/// # Arguments
/// * `years` - List of years
/// * `profits` - List of profits (parallel to years)
/// * `durations` - Optional list of durations (parallel to years)
///
/// # Returns
/// Score value in [0.0, 1.0]
#[pyfunction]
#[pyo3(signature = (years, profits, durations = None))]
#[allow(clippy::missing_errors_doc)]
pub fn compute_stability_score(
    years: Vec<i32>,
    profits: Vec<f64>,
    durations: Option<Vec<f64>>,
) -> PyResult<f64> {
    let (score, _wmape) = compute_stability_score_and_wmape(years, profits, durations)?;
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(is_leap_year(2004)); // Divisible by 4, not 100
        assert!(!is_leap_year(1900)); // Divisible by 100, not 400
        assert!(!is_leap_year(2001)); // Not divisible by 4
    }

    #[test]
    fn test_days_in_year() {
        assert_relative_eq!(days_in_year(2000), 366.0, epsilon = 1e-10);
        assert_relative_eq!(days_in_year(2001), 365.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stability_empty() {
        let profits = HashMap::new();
        let (score, wmape) = compute_stability_score_and_wmape_impl(&profits, None);
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
        assert_relative_eq!(wmape, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stability_single_year() {
        let mut profits = HashMap::new();
        profits.insert(2020, 10000.0);
        let (score, wmape) = compute_stability_score_and_wmape_impl(&profits, None);
        // Single year: e_y = P_y, so deviation = 0, wmape = 0, score = 1
        assert_relative_eq!(score, 1.0, epsilon = 1e-10);
        assert_relative_eq!(wmape, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stability_uniform_profits() {
        // Equal profits per year should give high score
        let mut profits = HashMap::new();
        profits.insert(2020, 10000.0);
        profits.insert(2021, 10000.0);
        profits.insert(2022, 10000.0);

        let (score, wmape) = compute_stability_score_and_wmape_impl(&profits, None);
        // Not exactly 1.0 because days differ slightly between years
        assert!(score > 0.99);
        assert!(wmape < 0.01);
    }

    #[test]
    fn test_stability_variable_profits() {
        // Variable profits should give lower score
        let mut profits = HashMap::new();
        profits.insert(2020, 5000.0);
        profits.insert(2021, 15000.0);
        profits.insert(2022, 10000.0);

        let (score, _wmape) = compute_stability_score_and_wmape_impl(&profits, None);
        assert!(score > 0.0);
        assert!(score < 1.0);
    }

    #[test]
    fn test_stability_with_durations() {
        let mut profits = HashMap::new();
        profits.insert(2020, 1000.0);
        profits.insert(2021, 1000.0);

        let mut durations = HashMap::new();
        durations.insert(2020, 100.0); // Partial year
        durations.insert(2021, 365.0); // Full year

        let (score, _wmape) = compute_stability_score_and_wmape_impl(&profits, Some(&durations));
        // Different durations but same profit means different rates
        assert!(score > 0.0);
        assert!(score < 1.0);
    }
}
