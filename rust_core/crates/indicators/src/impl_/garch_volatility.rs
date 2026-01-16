//! GARCH(1,1) Volatility indicator

use crate::traits::Indicator;
use omega_types::Candle;

/// GARCH(1,1) Volatility Estimator
///
/// Estimates conditional volatility using the GARCH(1,1) model:
/// σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
///
/// Where:
/// - ω (omega): Long-run variance weight
/// - α (alpha): Weight for squared returns (shock impact)
/// - β (beta): Weight for lagged variance (persistence)
/// - Constraint: α + β < 1 for stationarity
#[derive(Debug, Clone)]
pub struct GarchVolatility {
    /// Alpha: squared return coefficient
    pub alpha: f64,
    /// Beta: lagged variance coefficient
    pub beta: f64,
    /// Omega: constant term
    pub omega: f64,
    /// Use log returns instead of simple returns
    pub use_log_returns: bool,
    /// Scale factor for output (e.g., 100 for percentage)
    pub scale: f64,
    /// Minimum periods for initial variance estimate
    pub min_periods: usize,
    /// Minimum volatility floor (prevents near-zero variance)
    pub sigma_floor: f64,
}

impl GarchVolatility {
    /// Creates a new GARCH volatility indicator with default settings.
    pub fn new(alpha: f64, beta: f64, omega: f64) -> Self {
        Self {
            alpha,
            beta,
            omega,
            use_log_returns: true,
            scale: 100.0,
            min_periods: 20,
            sigma_floor: 0.0001,
        }
    }

    /// Creates from x1000/x1000000 encoded parameters.
    pub fn from_encoded(alpha_x1000: u32, beta_x1000: u32, omega_x1000000: u32) -> Self {
        Self::new(
            alpha_x1000 as f64 / 1000.0,
            beta_x1000 as f64 / 1000.0,
            omega_x1000000 as f64 / 1_000_000.0,
        )
    }

    /// Sets whether to use log returns.
    pub fn with_log_returns(mut self, use_log: bool) -> Self {
        self.use_log_returns = use_log;
        self
    }

    /// Sets the output scale factor.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the minimum periods for initialization.
    pub fn with_min_periods(mut self, periods: usize) -> Self {
        self.min_periods = periods;
        self
    }

    /// Sets the volatility floor.
    pub fn with_sigma_floor(mut self, floor: f64) -> Self {
        self.sigma_floor = floor;
        self
    }

    /// Calculates return from two consecutive prices.
    #[inline]
    fn calc_return(&self, prev_close: f64, close: f64) -> f64 {
        if self.use_log_returns {
            (close / prev_close).ln()
        } else {
            (close - prev_close) / prev_close
        }
    }
}

impl Indicator for GarchVolatility {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let result = vec![f64::NAN; len];

        if self.alpha + self.beta >= 1.0 {
            return result;
        }

        if len == 0 {
            return result;
        }

        let mut returns = vec![f64::NAN; len];
        for i in 1..len {
            let prev = candles[i - 1].close;
            let curr = candles[i].close;
            if prev.is_finite() && curr.is_finite() && prev != 0.0 {
                returns[i] = self.calc_return(prev, curr) * self.scale;
            }
        }

        let first_idx = returns.iter().position(|v| v.is_finite());
        let Some(first_idx) = first_idx else {
            return result;
        };

        let mut out_var = vec![f64::NAN; len];

        let lr_slice_start = first_idx.saturating_sub(1000);
        let lr_slice = &returns[lr_slice_start..=first_idx];
        let mut lr_var = nan_var(lr_slice).unwrap_or(1e-6);
        if !lr_var.is_finite() || lr_var <= 0.0 {
            lr_var = 1e-6;
        }

        let omega = if self.omega.is_finite() && self.omega > 0.0 {
            self.omega
        } else {
            lr_var * (1.0 - self.alpha - self.beta).max(1e-6)
        }
        .max(0.0);

        let mu_slice_start = first_idx.saturating_sub(2000);
        let mu_slice = &returns[mu_slice_start..=first_idx];
        let mu = nan_mean(mu_slice).unwrap_or(0.0);

        let mut eps_prev2 = (returns[first_idx] - mu).powi(2);
        let sigma_floor_sq = self.sigma_floor.powi(2);
        let mut var_prev = lr_var.max(eps_prev2).max(sigma_floor_sq);
        out_var[first_idx] = var_prev;

        for i in (first_idx + 1)..len {
            let r_i = returns[i];
            if !r_i.is_finite() {
                out_var[i] = out_var[i - 1];
                continue;
            }
            let mut var_t = omega + self.alpha * eps_prev2 + self.beta * var_prev;
            if var_t < sigma_floor_sq {
                var_t = sigma_floor_sq;
            }
            out_var[i] = var_t;
            eps_prev2 = (r_i - mu).powi(2);
            var_prev = var_t;
        }

        let mut sigma: Vec<f64> = out_var
            .iter()
            .map(|v| if v.is_finite() { v.sqrt() / self.scale } else { f64::NAN })
            .collect();

        let valid_after = first_idx + self.min_periods;
        for (i, v) in sigma.iter_mut().enumerate() {
            if i < valid_after {
                *v = f64::NAN;
            }
        }

        sigma
    }

    fn name(&self) -> &str {
        "GARCH_VOL"
    }

    fn warmup_periods(&self) -> usize {
        self.min_periods
    }
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
        Some(sum / count as f64)
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
        Some(sum / count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            open: close,
            high: close + 0.01,
            low: close - 0.01,
            close,
            volume: 0.0,
        }
    }

    #[test]
    fn test_garch_basic() {
        // Generate some prices with varying volatility
        let mut prices = vec![100.0];
        for i in 1..50 {
            let change = if i % 2 == 0 { 0.5 } else { -0.5 };
            prices.push(prices[i - 1] + change);
        }
        let candles: Vec<Candle> = prices.into_iter().map(make_candle).collect();

        let garch = GarchVolatility::new(0.1, 0.85, 0.00001).with_min_periods(10);
        let result = garch.compute(&candles);

        // First finite value should appear after warmup
        let first_finite = result.iter().position(|v| v.is_finite()).unwrap();
        assert!(first_finite >= 10);

        // Subsequent values should be positive finite
        for (i, value) in result.iter().enumerate().skip(first_finite) {
            assert!(
                value.is_finite() && *value > 0.0,
                "Expected positive finite at {}, got {}",
                i,
                value
            );
        }
    }

    #[test]
    fn test_garch_persistence() {
        // After a shock, volatility should decay gradually
        let mut prices = vec![100.0];
        for i in 1..30 {
            // Normal fluctuation
            let change = if i % 2 == 0 { 0.1 } else { -0.1 };
            prices.push(prices[i - 1] + change);
        }
        // Add a shock
        prices.push(prices.last().unwrap() + 5.0);
        // Back to normal
        for _ in 0..20 {
            let last = *prices.last().unwrap();
            prices.push(last + 0.1);
            prices.push(last);
        }

        let candles: Vec<Candle> = prices.into_iter().map(make_candle).collect();
        let garch = GarchVolatility::new(0.1, 0.85, 0.00001).with_min_periods(10);
        let result = garch.compute(&candles);

        // Find the shock point (index 30)
        let shock_idx = 30;

        // Volatility after shock should be higher
        let pre_shock_vol = result[shock_idx - 1];
        let post_shock_vol = result[shock_idx + 1];

        assert!(
            post_shock_vol > pre_shock_vol,
            "Volatility should increase after shock: {} vs {}",
            post_shock_vol,
            pre_shock_vol
        );

        // Should eventually decay (much later)
        let late_vol = result[result.len() - 1];
        assert!(
            late_vol < post_shock_vol,
            "Volatility should decay: {} vs {}",
            late_vol,
            post_shock_vol
        );
    }

    #[test]
    fn test_garch_floor() {
        // With constant prices, volatility should hit floor
        let candles: Vec<Candle> = vec![100.0; 50].into_iter().map(make_candle).collect();

        let floor = 0.001;
        let garch = GarchVolatility::new(0.1, 0.85, 0.00001)
            .with_min_periods(10)
            .with_sigma_floor(floor);
        let result = garch.compute(&candles);

        // After convergence, should be at floor
        let expected_min = floor / garch.scale;
        for (i, value) in result.iter().enumerate().take(50).skip(20) {
            assert!(
                *value >= expected_min * 0.99,
                "Expected >= {} at {}, got {}",
                expected_min,
                i,
                value
            );
        }
    }

    #[test]
    fn test_garch_insufficient_data() {
        let candles: Vec<Candle> = vec![100.0; 5].into_iter().map(make_candle).collect();

        let garch = GarchVolatility::new(0.1, 0.85, 0.00001).with_min_periods(20);
        let result = garch.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_garch_from_encoded() {
        let garch = GarchVolatility::from_encoded(100, 850, 10);
        assert!((garch.alpha - 0.1).abs() < 1e-10);
        assert!((garch.beta - 0.85).abs() < 1e-10);
        assert!((garch.omega - 0.00001).abs() < 1e-10);
    }

    #[test]
    fn test_garch_simple_returns_non_log() {
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.2).collect();
        let candles: Vec<Candle> = prices.into_iter().map(make_candle).collect();

        let garch = GarchVolatility::new(0.1, 0.85, 0.00001)
            .with_min_periods(5)
            .with_log_returns(false)
            .with_scale(1.0);
        let result = garch.compute(&candles);

        let value = result[6];
        assert!(value.is_finite() && value > 0.0);
    }
}
