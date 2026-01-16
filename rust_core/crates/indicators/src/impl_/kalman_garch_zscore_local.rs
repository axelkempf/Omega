//! Local Kalman+GARCH Z-Score for a capped window.

use crate::impl_::garch_volatility_local::{GarchLocalParams, garch_volatility_local};

/// Parameters for local Kalman+GARCH Z-Score computation.
#[derive(Debug, Clone, Copy)]
pub struct KalmanGarchLocalParams {
    /// Kalman measurement noise (R)
    pub r: f64,
    /// Kalman process noise (Q)
    pub q: f64,
    /// GARCH alpha
    pub alpha: f64,
    /// GARCH beta
    pub beta: f64,
    /// Optional omega; if None, derived from long-run variance
    pub omega: Option<f64>,
    /// Use log returns when true
    pub use_log_returns: bool,
    /// Scale factor applied to returns
    pub scale: f64,
    /// Minimum periods required after first valid return
    pub min_periods: usize,
    /// Volatility floor
    pub sigma_floor: f64,
}

impl KalmanGarchLocalParams {
    /// Creates params with defaults aligned to V1.
    #[must_use]
    pub fn new(r: f64, q: f64, alpha: f64, beta: f64, omega: Option<f64>) -> Self {
        Self {
            r,
            q,
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

/// Computes the local Kalman+GARCH Z-Score at idx using a capped lookback window.
#[must_use]
pub fn kalman_garch_zscore_local(
    prices: &[f64],
    idx: usize,
    lookback: usize,
    params: KalmanGarchLocalParams,
) -> Option<f64> {
    if params.alpha + params.beta >= 1.0 {
        return None;
    }
    if prices.is_empty() {
        return None;
    }
    if idx >= prices.len() {
        return None;
    }

    let end_pos = idx + 1;
    let window_len = lookback.max(1);
    let start_pos = end_pos.saturating_sub(window_len);
    let window = &prices[start_pos..end_pos];
    if window.len() < 2 {
        return None;
    }

    let kalman = kalman_mean_segment(window, params.r, params.q);
    let km_last = *kalman.last()?;
    if !km_last.is_finite() {
        return None;
    }
    let close_last = *window.last()?;
    if !close_last.is_finite() {
        return None;
    }

    let resid_last = close_last - km_last;

    let garch_params = GarchLocalParams {
        alpha: params.alpha,
        beta: params.beta,
        omega: params.omega,
        use_log_returns: params.use_log_returns,
        scale: params.scale,
        min_periods: params.min_periods,
        sigma_floor: params.sigma_floor,
    };
    let sigma_series = garch_volatility_local(window, window.len() - 1, window.len(), garch_params);
    if sigma_series.is_empty() {
        return None;
    }

    let sigma_ret_last = *sigma_series.last()?;
    if !sigma_ret_last.is_finite() || sigma_ret_last <= 0.0 {
        return None;
    }

    let sigma_price_last = close_last.abs() * sigma_ret_last;
    if !sigma_price_last.is_finite() || sigma_price_last == 0.0 {
        return None;
    }

    let z = resid_last / sigma_price_last;
    if z.is_finite() { Some(z) } else { None }
}

fn kalman_mean_segment(prices: &[f64], r: f64, q: f64) -> Vec<f64> {
    let len = prices.len();
    let mut xhat = vec![f64::NAN; len];
    let mut p = vec![f64::NAN; len];

    if len == 0 {
        return xhat;
    }

    let first_idx = prices.iter().position(|v| v.is_finite());
    let Some(first_idx) = first_idx else {
        return xhat;
    };

    xhat[first_idx] = prices[first_idx];
    p[first_idx] = r;

    for k in (first_idx + 1)..len {
        let meas = prices[k];
        let xhat_minus = xhat[k - 1];
        let p_minus = if p[k - 1].is_finite() { p[k - 1] } else { r } + q;

        if meas.is_finite() && xhat_minus.is_finite() {
            let k_gain = p_minus / (p_minus + r);
            xhat[k] = xhat_minus + k_gain * (meas - xhat_minus);
            p[k] = (1.0 - k_gain) * p_minus;
        } else {
            xhat[k] = f64::NAN;
            p[k] = f64::NAN;
        }
    }

    xhat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_kalman_garch_invalid_params() {
        let prices = vec![1.0, 1.1, 1.2];
        let params = KalmanGarchLocalParams {
            r: 0.01,
            q: 1.0,
            alpha: 0.6,
            beta: 0.5,
            omega: None,
            use_log_returns: true,
            scale: 100.0,
            min_periods: 1,
            sigma_floor: 1e-6,
        };
        assert!(kalman_garch_zscore_local(&prices, 2, 10, params).is_none());
    }

    #[test]
    fn test_local_kalman_garch_basic() {
        let prices = (0..30_u32)
            .map(|i| 1.0 + f64::from(i) * 0.01)
            .collect::<Vec<_>>();
        let mut params = KalmanGarchLocalParams::new(0.01, 1.0, 0.1, 0.8, None);
        params.min_periods = 5;
        let result = kalman_garch_zscore_local(&prices, 20, 20, params);
        assert!(result.is_some());
    }
}
