//! Local GARCH(1,1) volatility over a capped window.

/// Parameters for local GARCH volatility estimation.
#[derive(Debug, Clone, Copy)]
pub struct GarchLocalParams {
    /// GARCH alpha
    pub alpha: f64,
    /// GARCH beta
    pub beta: f64,
    /// Optional omega; if None, derived from long-run variance
    pub omega: Option<f64>,
    /// Use log returns when true
    pub use_log_returns: bool,
    /// Scale factor applied to returns (e.g., 100.0 for percent)
    pub scale: f64,
    /// Minimum periods required after first valid return
    pub min_periods: usize,
    /// Volatility floor
    pub sigma_floor: f64,
}

impl GarchLocalParams {
    /// Creates a new parameter set with common defaults.
    #[must_use]
    pub fn new(alpha: f64, beta: f64, omega: Option<f64>) -> Self {
        Self {
            alpha,
            beta,
            omega,
            use_log_returns: true,
            scale: 100.0,
            min_periods: 50,
            sigma_floor: 1e-6,
        }
    }
}

/// Computes local GARCH volatility for a window ending at idx (inclusive).
///
/// Returns a vector of sigma values for the window, with NaNs for warmup/invalid values.
#[must_use]
pub fn garch_volatility_local(
    prices: &[f64],
    idx: usize,
    lookback: usize,
    params: GarchLocalParams,
) -> Vec<f64> {
    if params.alpha + params.beta >= 1.0 {
        return Vec::new();
    }
    if prices.is_empty() {
        return Vec::new();
    }

    let idx = idx.min(prices.len().saturating_sub(1));
    let end_pos = idx + 1;
    let window_len = lookback.max(1);
    let start_pos = end_pos.saturating_sub(window_len);
    let window = &prices[start_pos..end_pos];
    if window.is_empty() {
        return Vec::new();
    }

    let n = window.len();
    let mut returns = vec![f64::NAN; n];

    if n > 1 {
        for i in 1..n {
            let prev = window[i - 1];
            let curr = window[i];
            if prev.is_finite() && curr.is_finite() && prev != 0.0 {
                returns[i] = if params.use_log_returns {
                    (curr / prev).ln() * params.scale
                } else {
                    ((curr / prev) - 1.0) * params.scale
                };
            }
        }
    }

    let first_idx = returns.iter().position(|v| v.is_finite());
    let Some(first_idx) = first_idx else {
        return vec![f64::NAN; n];
    };

    let mut out_var = vec![f64::NAN; n];

    let lr_slice_start = first_idx.saturating_sub(1000);
    let lr_slice = &returns[lr_slice_start..=first_idx];
    let mut lr_var = nan_var(lr_slice).unwrap_or(1e-6);
    if !lr_var.is_finite() || lr_var <= 0.0 {
        lr_var = 1e-6;
    }

    let omega = params
        .omega
        .unwrap_or_else(|| lr_var * (1.0 - params.alpha - params.beta).max(1e-6));
    let omega = omega.max(0.0);

    let mu_slice_start = first_idx.saturating_sub(2000);
    let mu_slice = &returns[mu_slice_start..=first_idx];
    let mu = nan_mean(mu_slice).unwrap_or(0.0);

    let mut eps_prev2 = (returns[first_idx] - mu).powi(2);
    let sigma_floor_sq = params.sigma_floor.powi(2);
    let mut var_prev = lr_var.max(eps_prev2).max(sigma_floor_sq);
    out_var[first_idx] = var_prev;

    for i in (first_idx + 1)..n {
        let ri = returns[i];
        if !ri.is_finite() {
            out_var[i] = out_var[i - 1];
            continue;
        }
        let mut var_t = omega + params.alpha * eps_prev2 + params.beta * var_prev;
        if var_t < sigma_floor_sq {
            var_t = sigma_floor_sq;
        }
        out_var[i] = var_t;
        let eps = (ri - mu).powi(2);
        eps_prev2 = eps;
        var_prev = var_t;
    }

    let mut sigma: Vec<f64> = out_var
        .iter()
        .map(|v| {
            if v.is_finite() {
                v.sqrt() / params.scale
            } else {
                f64::NAN
            }
        })
        .collect();

    let valid_after = first_idx + params.min_periods;
    for (i, v) in sigma.iter_mut().enumerate() {
        if i < valid_after {
            *v = f64::NAN;
        }
    }

    sigma
}

fn nan_mean(values: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        if v.is_finite() {
            sum += *v;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        #[allow(clippy::cast_precision_loss)]
        let count_f = count as f64;
        Some(sum / count_f)
    }
}

fn nan_var(values: &[f64]) -> Option<f64> {
    let mean = nan_mean(values)?;
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        if v.is_finite() {
            let diff = *v - mean;
            sum += diff * diff;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        #[allow(clippy::cast_precision_loss)]
        let count_f = count as f64;
        Some(sum / count_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_garch_invalid_params() {
        let prices = vec![1.0, 1.1, 1.2];
        let params = GarchLocalParams {
            alpha: 0.6,
            beta: 0.5,
            omega: None,
            use_log_returns: true,
            scale: 100.0,
            min_periods: 1,
            sigma_floor: 1e-6,
        };
        let out = garch_volatility_local(&prices, 2, 10, params);
        assert!(out.is_empty());
    }

    #[test]
    fn test_local_garch_window_length() {
        let prices = (0..10_u32)
            .map(|i| 1.0 + f64::from(i) * 0.01)
            .collect::<Vec<_>>();
        let params = GarchLocalParams::new(0.1, 0.8, None);
        let out = garch_volatility_local(&prices, 5, 4, params);
        assert_eq!(out.len(), 4);
    }
}
