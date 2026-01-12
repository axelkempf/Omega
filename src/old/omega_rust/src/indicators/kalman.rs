//! Kalman Filter implementation for price smoothing.
//!
//! The Kalman Filter is an optimal recursive estimation algorithm
//! that estimates the state of a dynamic system from noisy measurements.
//!
//! ## Components
//!
//! - **State (x)**: Estimated true price
//! - **Estimate Uncertainty (P)**: Covariance of the state estimate
//! - **Process Noise (Q)**: Expected variance of the process
//! - **Measurement Noise (R)**: Expected variance of measurements
//!
//! ## Kalman Variants
//!
//! - `kalman_mean`: Smooth mean estimation
//! - `kalman_zscore`: Z-score based on Kalman filtered mean and variance

use crate::error::{OmegaError, Result};

/// Kalman Filter result.
pub struct KalmanResult {
    /// Kalman-filtered mean (state estimate)
    pub mean: Vec<f64>,
    /// Kalman variance estimate
    pub variance: Vec<f64>,
}

/// Calculate Kalman-filtered mean.
///
/// Uses a simple 1D Kalman Filter with constant dynamics.
///
/// # Arguments
///
/// * `values` - Input values (typically close prices)
/// * `process_variance` - Process noise variance (Q)
/// * `measurement_variance` - Measurement noise variance (R)
///
/// # Returns
///
/// `KalmanResult` containing filtered mean and variance estimates.
///
/// # Errors
///
/// Returns an error if:
/// - `values` is empty
/// - `measurement_variance` is 0 or negative
pub fn kalman_impl(
    values: &[f64],
    process_variance: f64,
    measurement_variance: f64,
) -> Result<KalmanResult> {
    let n = values.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    if measurement_variance <= 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: "measurement_variance must be positive".to_string(),
        });
    }

    let mut mean = vec![f64::NAN; n];
    let mut variance = vec![f64::NAN; n];

    // Find first valid value
    let first_valid = values.iter().position(|&v| v.is_finite());
    let first_valid = match first_valid {
        Some(idx) => idx,
        None => return Ok(KalmanResult { mean, variance }),
    };

    // Initialize state
    let mut x = values[first_valid]; // Initial state estimate
    let mut p = measurement_variance; // Initial estimate uncertainty

    mean[first_valid] = x;
    variance[first_valid] = p;

    // Kalman Filter loop
    for i in (first_valid + 1)..n {
        let z = values[i]; // Measurement

        // Predict step
        // x_pred = x (assuming constant dynamics)
        let p_pred = p + process_variance; // Predict uncertainty

        if z.is_nan() {
            // No measurement, just propagate uncertainty
            p = p_pred;
            mean[i] = x;
            variance[i] = p;
            continue;
        }

        // Update step
        let k = p_pred / (p_pred + measurement_variance); // Kalman gain
        x = x + k * (z - x); // Updated state estimate
        p = (1.0 - k) * p_pred; // Updated estimate uncertainty

        mean[i] = x;
        variance[i] = p;
    }

    Ok(KalmanResult { mean, variance })
}

/// Calculate Kalman-based Z-Score.
///
/// Uses Kalman filter to estimate mean and variance, then calculates
/// z-score relative to the Kalman-filtered state.
///
/// # Arguments
///
/// * `values` - Input values
/// * `process_variance` - Process noise variance
/// * `measurement_variance` - Measurement noise variance
///
/// # Returns
///
/// Vector of z-scores: (current_value - kalman_mean) / sqrt(kalman_variance)
pub fn kalman_zscore_impl(
    values: &[f64],
    process_variance: f64,
    measurement_variance: f64,
) -> Result<Vec<f64>> {
    let kalman = kalman_impl(values, process_variance, measurement_variance)?;

    let n = values.len();
    let mut zscore = vec![f64::NAN; n];

    for i in 0..n {
        let v = values[i];
        let m = kalman.mean[i];
        let var = kalman.variance[i];

        if v.is_finite() && m.is_finite() && var.is_finite() && var > 0.0 {
            zscore[i] = (v - m) / var.sqrt();
        }
    }

    Ok(zscore)
}

/// Adaptive Kalman Filter with dynamic noise estimation.
///
/// Automatically adjusts process and measurement variance based on
/// recent prediction errors.
///
/// # Arguments
///
/// * `values` - Input values
/// * `window` - Window size for adaptive noise estimation
/// * `alpha` - Smoothing factor for noise updates (0-1)
pub fn kalman_adaptive_impl(values: &[f64], window: usize, alpha: f64) -> Result<KalmanResult> {
    if window == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "window must be greater than 0".to_string(),
        });
    }

    if alpha <= 0.0 || alpha > 1.0 {
        return Err(OmegaError::InvalidParameter {
            reason: "alpha must be in (0, 1]".to_string(),
        });
    }

    let n = values.len();
    if n == 0 {
        return Err(OmegaError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let mut mean = vec![f64::NAN; n];
    let mut variance = vec![f64::NAN; n];

    // Find first valid
    let first_valid = values.iter().position(|&v| v.is_finite());
    let first_valid = match first_valid {
        Some(idx) => idx,
        None => return Ok(KalmanResult { mean, variance }),
    };

    // Initialize
    let mut x = values[first_valid];
    let mut p = 1.0; // Initial uncertainty
    let mut q = 0.01; // Process variance (adaptive)
    let mut r = 1.0; // Measurement variance (adaptive)

    let mut errors: Vec<f64> = Vec::with_capacity(window);

    mean[first_valid] = x;
    variance[first_valid] = p;

    for i in (first_valid + 1)..n {
        let z = values[i];

        // Predict
        let p_pred = p + q;

        if z.is_nan() {
            p = p_pred;
            mean[i] = x;
            variance[i] = p;
            continue;
        }

        // Innovation (prediction error)
        let innovation = z - x;

        // Update errors buffer for adaptive variance
        if errors.len() >= window {
            errors.remove(0);
        }
        errors.push(innovation);

        // Update noise estimates (after sufficient data)
        if errors.len() >= window / 2 {
            let error_variance =
                errors.iter().map(|e| e.powi(2)).sum::<f64>() / errors.len() as f64;
            r = alpha * error_variance + (1.0 - alpha) * r;
            q = alpha * r * 0.1 + (1.0 - alpha) * q; // Q proportional to R
        }

        // Kalman gain
        let k = p_pred / (p_pred + r);

        // Update
        x = x + k * innovation;
        p = (1.0 - k) * p_pred;

        mean[i] = x;
        variance[i] = p;
    }

    Ok(KalmanResult { mean, variance })
}

/// Kalman Z-Score with stepwise (HTF bar) semantics.
///
/// Calculates Kalman Z-Score only at specified bar indices, then forward-fills
/// the values. This is useful for higher-timeframe (HTF) analysis where
/// signals should only update when a new HTF bar completes.
///
/// # Arguments
///
/// * `values` - Full input series (typically close prices)
/// * `new_bar_indices` - Indices where new HTF bars occur (sorted, ascending)
/// * `window` - Rolling window for residual std calculation
/// * `process_variance` - Kalman process noise (Q)
/// * `measurement_variance` - Kalman measurement noise (R)
///
/// # Returns
///
/// Z-Score vector with forward-filled values.
pub fn kalman_zscore_stepwise_impl(
    values: &[f64],
    new_bar_indices: &[usize],
    window: usize,
    process_variance: f64,
    measurement_variance: f64,
) -> Result<Vec<f64>> {
    let n = values.len();
    if n == 0 {
        return Ok(vec![]);
    }

    if new_bar_indices.is_empty() {
        return Ok(vec![f64::NAN; n]);
    }

    if window == 0 {
        return Err(OmegaError::InvalidParameter {
            reason: "window must be greater than 0".to_string(),
        });
    }

    // Extract values at new_bar_indices only
    let reduced_values: Vec<f64> = new_bar_indices
        .iter()
        .filter_map(|&idx| if idx < n { Some(values[idx]) } else { None })
        .collect();

    if reduced_values.is_empty() {
        return Ok(vec![f64::NAN; n]);
    }

    // Apply Kalman filter on reduced series
    let kalman_result = kalman_impl(&reduced_values, process_variance, measurement_variance)?;

    // Calculate residuals
    let residuals: Vec<f64> = reduced_values
        .iter()
        .zip(kalman_result.mean.iter())
        .map(|(&v, &m)| {
            if v.is_finite() && m.is_finite() {
                v - m
            } else {
                f64::NAN
            }
        })
        .collect();

    // Calculate rolling std of residuals
    let mut rolling_std = vec![f64::NAN; reduced_values.len()];
    for i in window.saturating_sub(1)..reduced_values.len() {
        let start = i.saturating_sub(window - 1);
        let slice = &residuals[start..=i];
        let valid: Vec<f64> = slice.iter().copied().filter(|x| x.is_finite()).collect();

        if valid.len() >= window {
            let mean_val = valid.iter().sum::<f64>() / valid.len() as f64;
            let variance =
                valid.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / valid.len() as f64;
            rolling_std[i] = variance.sqrt();
        }
    }

    // Calculate Z-score on reduced series
    let z_reduced: Vec<f64> = residuals
        .iter()
        .zip(rolling_std.iter())
        .map(|(&r, &s)| {
            if r.is_finite() && s.is_finite() && s > 0.0 {
                r / s
            } else {
                f64::NAN
            }
        })
        .collect();

    // Forward-fill to full series
    let mut z_full = vec![f64::NAN; n];

    // Place Z-scores at their original indices
    for (reduced_idx, &orig_idx) in new_bar_indices.iter().enumerate() {
        if orig_idx < n && reduced_idx < z_reduced.len() {
            z_full[orig_idx] = z_reduced[reduced_idx];
        }
    }

    // Forward-fill
    let mut last_valid = f64::NAN;
    for i in 0..n {
        if z_full[i].is_finite() {
            last_valid = z_full[i];
        } else {
            z_full[i] = last_valid;
        }
    }

    Ok(z_full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kalman_basic() {
        let values = vec![100.0, 102.0, 101.0, 103.0, 104.0];
        let result = kalman_impl(&values, 0.01, 1.0).unwrap();

        assert_eq!(result.mean.len(), 5);
        assert_eq!(result.variance.len(), 5);

        // Mean should be smoothed
        assert!(result.mean[0].is_finite());
        // Variance should decrease as more data is seen
        assert!(result.variance[4] < result.variance[1]);
    }

    #[test]
    fn test_kalman_constant() {
        let values = vec![100.0; 10];
        let result = kalman_impl(&values, 0.01, 1.0).unwrap();

        // For constant values, Kalman mean should converge to that value
        assert_relative_eq!(result.mean[9], 100.0, epsilon = 0.01);
    }

    #[test]
    fn test_kalman_with_nan() {
        let values = vec![100.0, 102.0, f64::NAN, 101.0, 103.0];
        let result = kalman_impl(&values, 0.01, 1.0).unwrap();

        // Should handle NaN by propagating state
        assert!(result.mean[2].is_finite());
    }

    #[test]
    fn test_kalman_zscore() {
        let values = vec![100.0, 102.0, 101.0, 103.0, 110.0]; // Jump at end
        let zscore = kalman_zscore_impl(&values, 0.01, 1.0).unwrap();

        // Last value should have high z-score (outlier)
        let last_z = zscore[4];
        assert!(last_z.is_finite());
        assert!(last_z > 1.0); // Significantly above Kalman estimate
    }

    #[test]
    fn test_kalman_zscore_stepwise() {
        // 20 data points, HTF bar every 4 points
        let values: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let new_bar_indices: Vec<usize> = vec![0, 4, 8, 12, 16];

        let result = kalman_zscore_stepwise_impl(&values, &new_bar_indices, 3, 0.01, 1.0).unwrap();

        assert_eq!(result.len(), 20);
        // Values should be forward-filled between HTF bars
        // Points 1, 2, 3 should have same value as point 0 (if finite)
        // Points 5, 6, 7 should have same value as point 4
        if result[4].is_finite() {
            assert_eq!(result[5], result[4]);
            assert_eq!(result[6], result[4]);
            assert_eq!(result[7], result[4]);
        }
    }

    #[test]
    fn test_kalman_zscore_stepwise_empty() {
        let values = vec![100.0; 10];
        let new_bar_indices: Vec<usize> = vec![];

        let result = kalman_zscore_stepwise_impl(&values, &new_bar_indices, 3, 0.01, 1.0).unwrap();
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&x| x.is_nan()));
    }

    #[test]
    fn test_kalman_adaptive() {
        let values: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() + rand_like(i))
            .collect();

        let result = kalman_adaptive_impl(&values, 10, 0.3).unwrap();

        assert_eq!(result.mean.len(), 50);
        // Should produce finite results
        assert!(result.mean[49].is_finite());
    }

    #[test]
    fn test_kalman_invalid_params() {
        let values = vec![1.0, 2.0, 3.0];

        let result = kalman_impl(&values, 0.01, 0.0);
        assert!(result.is_err()); // measurement_variance = 0

        let result = kalman_adaptive_impl(&values, 0, 0.5);
        assert!(result.is_err()); // window = 0
    }

    // Simple deterministic pseudo-random for testing
    fn rand_like(seed: usize) -> f64 {
        let x = (seed * 1103515245 + 12345) % 2147483648;
        (x as f64 / 2147483648.0 - 0.5) * 0.5
    }
}
