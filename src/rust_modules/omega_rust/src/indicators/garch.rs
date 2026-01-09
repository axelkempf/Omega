//! GARCH(1,1) Volatility Model Implementation
//!
//! Provides lightweight recursive GARCH(1,1) on returns for volatility estimation.
//!
//! ## Algorithm
//!
//! GARCH(1,1): var[t] = ω + α × ε²[t-1] + β × var[t-1]
//!
//! Where:
//! - ω (omega): Long-run variance component
//! - α (alpha): Weight for previous squared shock
//! - β (beta): Weight for previous variance (persistence)
//! - ε: Centered returns (r[t] - μ)
//!
//! ## Performance Target
//!
//! | Operation       | Python Baseline | Rust Target | Speedup |
//! |-----------------|-----------------|-------------|---------|
//! | GARCH(1,1) 200k | ~120ms         | ≤6ms        | 20x     |
//!
//! ## References
//!
//! Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"

use crate::error::{OmegaError, Result};

/// Result of GARCH(1,1) calculation.
#[derive(Debug, Clone)]
pub struct GarchResult {
    /// Variance series (σ²)
    pub variance: Vec<f64>,
    /// Volatility series (σ) - square root of variance, scaled back
    pub sigma: Vec<f64>,
}

/// GARCH(1,1) parameters with validation.
#[derive(Debug, Clone, Copy)]
pub struct GarchParams {
    /// Weight for squared shock (typically 0.05)
    pub alpha: f64,
    /// Persistence weight (typically 0.90)
    pub beta: f64,
    /// Long-run variance component (auto-computed if None)
    pub omega: Option<f64>,
    /// Use log returns vs simple returns
    pub use_log_returns: bool,
    /// Scaling factor for returns (e.g., 100 for percentage)
    pub scale: f64,
    /// Minimum periods before trusting estimate
    pub min_periods: usize,
    /// Floor for sigma to prevent numerical issues
    pub sigma_floor: f64,
}

impl Default for GarchParams {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            beta: 0.90,
            omega: None,
            use_log_returns: true,
            scale: 100.0,
            min_periods: 50,
            sigma_floor: 1e-6,
        }
    }
}

impl GarchParams {
    /// Validate GARCH parameters.
    pub fn validate(&self) -> Result<()> {
        if self.alpha < 0.0 || self.alpha >= 1.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!("alpha must be in [0, 1), got {}", self.alpha),
            });
        }
        if self.beta < 0.0 || self.beta >= 1.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!("beta must be in [0, 1), got {}", self.beta),
            });
        }
        // Stationarity condition: α + β < 1
        if self.alpha + self.beta >= 1.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!(
                    "GARCH stationarity requires alpha + beta < 1, got {} + {} = {}",
                    self.alpha,
                    self.beta,
                    self.alpha + self.beta
                ),
            });
        }
        if self.scale <= 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!("scale must be positive, got {}", self.scale),
            });
        }
        if self.sigma_floor < 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!("sigma_floor must be non-negative, got {}", self.sigma_floor),
            });
        }
        Ok(())
    }
}

/// Calculate returns from prices.
///
/// # Arguments
///
/// * `prices` - Price series
/// * `use_log` - Use log returns if true, simple returns otherwise
/// * `scale` - Scaling factor (e.g., 100 for percentage)
///
/// # Returns
///
/// Returns vector with first element as NaN.
fn calculate_returns(prices: &[f64], use_log: bool, scale: f64) -> Vec<f64> {
    let n = prices.len();
    let mut returns = vec![f64::NAN; n];

    for i in 1..n {
        let prev = prices[i - 1];
        let curr = prices[i];

        if !prev.is_finite() || !curr.is_finite() || prev == 0.0 {
            returns[i] = f64::NAN;
            continue;
        }

        if use_log {
            returns[i] = (curr / prev).ln() * scale;
        } else {
            returns[i] = (curr / prev - 1.0) * scale;
        }
    }

    returns
}

/// Calculate long-run variance from returns (robust estimator).
///
/// Uses sample variance from a lookback window.
fn estimate_long_run_variance(returns: &[f64], end_idx: usize, lookback: usize) -> f64 {
    let start = end_idx.saturating_sub(lookback);
    let slice = &returns[start..=end_idx];

    let valid: Vec<f64> = slice.iter().copied().filter(|x| x.is_finite()).collect();

    if valid.is_empty() {
        return 1e-6;
    }

    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let var = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;

    if var.is_finite() && var > 0.0 {
        var
    } else {
        1e-6
    }
}

/// Calculate mean from returns for centering.
fn estimate_mean(returns: &[f64], end_idx: usize, lookback: usize) -> f64 {
    let start = end_idx.saturating_sub(lookback);
    let slice = &returns[start..=end_idx];

    let valid: Vec<f64> = slice.iter().copied().filter(|x| x.is_finite()).collect();

    if valid.is_empty() {
        return 0.0;
    }

    valid.iter().sum::<f64>() / valid.len() as f64
}

/// GARCH(1,1) volatility estimation on full price series.
///
/// Returns scaled volatility (σ) of returns, not prices.
///
/// # Arguments
///
/// * `prices` - Close prices
/// * `params` - GARCH parameters
///
/// # Returns
///
/// GarchResult with variance and sigma series.
pub fn garch_volatility_impl(prices: &[f64], params: &GarchParams) -> Result<GarchResult> {
    params.validate()?;

    let n = prices.len();
    if n == 0 {
        return Ok(GarchResult {
            variance: vec![],
            sigma: vec![],
        });
    }

    // Calculate returns
    let returns = calculate_returns(prices, params.use_log_returns, params.scale);

    // Find first valid return
    let first_valid = returns.iter().position(|&r| r.is_finite());
    let first_idx = match first_valid {
        Some(idx) => idx,
        None => {
            return Ok(GarchResult {
                variance: vec![f64::NAN; n],
                sigma: vec![f64::NAN; n],
            });
        }
    };

    // Initialize variance array
    let mut out_var = vec![f64::NAN; n];

    // Estimate long-run variance for initialization
    let lr_var = estimate_long_run_variance(&returns, first_idx, 1000);

    // Compute omega if not provided
    let omega = params.omega.unwrap_or_else(|| {
        let unconditional = lr_var * (1.0 - params.alpha - params.beta).max(1e-6);
        unconditional.max(0.0)
    });

    // Estimate mean for centering
    let mu = estimate_mean(&returns, first_idx, 2000);

    // Initialize recursion
    let sigma_floor_sq = params.sigma_floor.powi(2);
    let eps_init = (returns[first_idx] - mu).powi(2);
    let mut eps_prev2 = eps_init;
    let mut var_prev = lr_var.max(eps_init).max(sigma_floor_sq);
    out_var[first_idx] = var_prev;

    // GARCH recursion
    for i in (first_idx + 1)..n {
        let r_i = returns[i];

        if !r_i.is_finite() {
            // Carry forward on NaN
            out_var[i] = out_var[i - 1];
            continue;
        }

        // var[t] = ω + α × ε²[t-1] + β × var[t-1]
        let var_t = omega + params.alpha * eps_prev2 + params.beta * var_prev;
        let var_t = var_t.max(sigma_floor_sq);
        out_var[i] = var_t;

        // Update state
        eps_prev2 = (r_i - mu).powi(2);
        var_prev = var_t;
    }

    // Convert variance to sigma (scaled back)
    let mut sigma = vec![f64::NAN; n];
    for i in 0..n {
        if out_var[i].is_finite() {
            // Apply min_periods mask
            let valid = i >= first_idx + params.min_periods;
            if valid {
                sigma[i] = out_var[i].sqrt() / params.scale;
            }
        }
    }

    Ok(GarchResult {
        variance: out_var,
        sigma,
    })
}

/// GARCH(1,1) volatility on a local window slice.
///
/// # Arguments
///
/// * `prices` - Full price series
/// * `idx` - Current index (0-based, inclusive)
/// * `lookback` - Window size for local estimation
/// * `params` - GARCH parameters
///
/// # Returns
///
/// GarchResult for the local window.
pub fn garch_volatility_local_impl(
    prices: &[f64],
    idx: usize,
    lookback: usize,
    params: &GarchParams,
) -> Result<GarchResult> {
    params.validate()?;

    let n = prices.len();
    if n == 0 || idx >= n {
        return Ok(GarchResult {
            variance: vec![],
            sigma: vec![],
        });
    }

    // Extract local window: [max(0, idx - lookback + 1) .. idx + 1]
    let end_pos = idx + 1;
    let start_pos = end_pos.saturating_sub(lookback);
    let window_prices = &prices[start_pos..end_pos];

    // Run GARCH on local window
    garch_volatility_impl(window_prices, params)
}

/// Get the final sigma value from a local GARCH calculation.
///
/// Convenience function that returns just the last sigma value.
pub fn garch_volatility_local_last(
    prices: &[f64],
    idx: usize,
    lookback: usize,
    params: &GarchParams,
) -> Result<f64> {
    let result = garch_volatility_local_impl(prices, idx, lookback, params)?;

    if result.sigma.is_empty() {
        return Ok(f64::NAN);
    }

    Ok(*result.sigma.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_prices() -> Vec<f64> {
        // Simulate price series with some volatility clustering
        let mut prices = vec![100.0];
        let mut rng_state: u64 = 42;

        for _ in 0..199 {
            // Simple LCG for deterministic "random"
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345) % (1 << 31);
            let rand_val = (rng_state as f64 / (1u64 << 31) as f64 - 0.5) * 0.02;
            let last = *prices.last().unwrap();
            prices.push(last * (1.0 + rand_val));
        }
        prices
    }

    #[test]
    fn test_garch_basic() {
        let prices = create_test_prices();
        let params = GarchParams::default();
        let result = garch_volatility_impl(&prices, &params).unwrap();

        assert_eq!(result.sigma.len(), 200);
        // First min_periods values should be NaN
        assert!(result.sigma[0].is_nan());
        // After warmup, should have finite values
        assert!(result.sigma[100].is_finite());
    }

    #[test]
    fn test_garch_constant_prices() {
        let prices = vec![100.0; 100];
        let params = GarchParams::default();
        let result = garch_volatility_impl(&prices, &params).unwrap();

        // Constant prices → very low volatility
        let last_sigma = result.sigma.last().unwrap();
        if last_sigma.is_finite() {
            assert!(*last_sigma < 0.01);
        }
    }

    #[test]
    fn test_garch_stationarity_check() {
        let prices = create_test_prices();
        let params = GarchParams {
            alpha: 0.5,
            beta: 0.6, // α + β = 1.1 > 1 → unstable
            ..Default::default()
        };

        let result = garch_volatility_impl(&prices, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_garch_local() {
        let prices = create_test_prices();
        let params = GarchParams {
            min_periods: 20,
            ..Default::default()
        };

        let result = garch_volatility_local_impl(&prices, 150, 100, &params).unwrap();
        assert_eq!(result.sigma.len(), 100);

        let last = garch_volatility_local_last(&prices, 150, 100, &params).unwrap();
        assert!(last.is_finite());
    }

    #[test]
    fn test_garch_with_nans() {
        let mut prices = create_test_prices();
        prices[50] = f64::NAN;
        prices[51] = f64::NAN;

        let params = GarchParams::default();
        let result = garch_volatility_impl(&prices, &params).unwrap();

        // Should handle NaNs gracefully
        assert!(result.sigma[100].is_finite());
    }

    #[test]
    fn test_returns_calculation() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];

        // Log returns
        let log_ret = calculate_returns(&prices, true, 100.0);
        assert!(log_ret[0].is_nan());
        assert_relative_eq!(log_ret[1], (101.0 / 100.0_f64).ln() * 100.0, epsilon = 1e-10);

        // Simple returns
        let simple_ret = calculate_returns(&prices, false, 100.0);
        assert_relative_eq!(simple_ret[1], 1.0, epsilon = 1e-10); // 1% return
    }

    #[test]
    fn test_empty_prices() {
        let prices: Vec<f64> = vec![];
        let params = GarchParams::default();
        let result = garch_volatility_impl(&prices, &params).unwrap();

        assert!(result.sigma.is_empty());
        assert!(result.variance.is_empty());
    }
}
