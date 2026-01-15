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
        let mut result = vec![f64::NAN; len];

        if len < self.min_periods + 1 || self.min_periods == 0 {
            return result;
        }

        // Calculate returns series
        let mut returns = vec![0.0; len];
        for i in 1..len {
            returns[i] = self.calc_return(candles[i - 1].close, candles[i].close);
        }

        // Initial variance from first min_periods returns
        let init_returns = &returns[1..=self.min_periods];
        let mean = init_returns.iter().sum::<f64>() / self.min_periods as f64;
        let init_var = init_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / self.min_periods as f64;

        // Ensure minimum variance
        let mut sigma2 = init_var.max(self.sigma_floor.powi(2));
        result[self.min_periods] = sigma2.sqrt() * self.scale;

        // GARCH recursion: σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
        for i in self.min_periods..returns.len() - 1 {
            let r_prev = returns[i];
            sigma2 = self.omega + self.alpha * r_prev.powi(2) + self.beta * sigma2;
            sigma2 = sigma2.max(self.sigma_floor.powi(2));
            result[i + 1] = sigma2.sqrt() * self.scale;
        }

        result
    }

    fn name(&self) -> &str {
        "GARCH_VOL"
    }

    fn warmup_periods(&self) -> usize {
        self.min_periods + 1
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

        // First min_periods values should be NaN
        for i in 0..10 {
            assert!(result[i].is_nan(), "Expected NaN at {}", i);
        }

        // Rest should be positive finite values
        for i in 10..50 {
            assert!(
                result[i].is_finite() && result[i] > 0.0,
                "Expected positive finite at {}, got {}",
                i,
                result[i]
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

        // After convergence, should be at floor * scale
        let expected_min = floor * 100.0;
        for i in 20..50 {
            assert!(
                result[i] >= expected_min * 0.99,
                "Expected >= {} at {}, got {}",
                expected_min,
                i,
                result[i]
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
}
