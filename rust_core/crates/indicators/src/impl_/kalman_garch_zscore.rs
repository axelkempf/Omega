//! Kalman+GARCH Z-Score indicator

use crate::impl_::garch_volatility::GarchVolatility;
use crate::impl_::kalman_mean::KalmanFilter;
use crate::traits::Indicator;
use omega_types::Candle;

/// Kalman+GARCH Z-Score
///
/// Combines Kalman filtering with GARCH volatility estimation:
/// 1. Compute Kalman-smoothed price level
/// 2. Calculate residuals (price - kalman_level)
/// 3. Estimate volatility using GARCH on returns
/// 4. Normalize residuals by GARCH volatility
///
/// This provides a more adaptive Z-Score that accounts for
/// time-varying volatility.
#[derive(Debug, Clone)]
pub struct KalmanGarchZScore {
    /// Kalman measurement noise (R)
    pub kalman_r: f64,
    /// Kalman process noise (Q)
    pub kalman_q: f64,
    /// GARCH alpha
    pub garch_alpha: f64,
    /// GARCH beta
    pub garch_beta: f64,
    /// GARCH omega
    pub garch_omega: f64,
    /// Minimum periods for GARCH initialization
    pub min_periods: usize,
}

impl KalmanGarchZScore {
    /// Creates a new Kalman+GARCH Z-Score indicator.
    pub fn new(
        kalman_r: f64,
        kalman_q: f64,
        garch_alpha: f64,
        garch_beta: f64,
        garch_omega: f64,
    ) -> Self {
        Self {
            kalman_r,
            kalman_q,
            garch_alpha,
            garch_beta,
            garch_omega,
            min_periods: 20,
        }
    }

    /// Creates from encoded parameters.
    pub fn from_encoded(
        r_x1000: u32,
        q_x1000: u32,
        alpha_x1000: u32,
        beta_x1000: u32,
        omega_x1000000: u32,
    ) -> Self {
        Self::new(
            r_x1000 as f64 / 1000.0,
            q_x1000 as f64 / 1000.0,
            alpha_x1000 as f64 / 1000.0,
            beta_x1000 as f64 / 1000.0,
            omega_x1000000 as f64 / 1_000_000.0,
        )
    }

    /// Sets minimum periods for initialization.
    pub fn with_min_periods(mut self, periods: usize) -> Self {
        self.min_periods = periods;
        self
    }
}

impl Indicator for KalmanGarchZScore {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if len < self.min_periods + 1 {
            return result;
        }

        // Extract prices
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

        // Compute Kalman level
        let kalman = KalmanFilter::new(self.kalman_r, self.kalman_q);
        let kalman_level = kalman.compute_level(&prices);

        // Compute GARCH volatility (scaled to 1, not 100)
        let garch = GarchVolatility::new(self.garch_alpha, self.garch_beta, self.garch_omega)
            .with_scale(1.0)
            .with_min_periods(self.min_periods)
            .with_sigma_floor(1e-8);
        let garch_vol = garch.compute(candles);

        // Compute Z-Score: (price - kalman) / garch_vol
        for i in self.min_periods..len {
            let residual = prices[i] - kalman_level[i];
            let vol = garch_vol[i];

            if vol.is_finite() && vol > 1e-10 {
                result[i] = residual / vol;
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    fn name(&self) -> &str {
        "KALMAN_GARCH_Z"
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
    fn test_kalman_garch_zscore_basic() {
        let mut prices = vec![100.0];
        for i in 1..50 {
            let change = if i % 2 == 0 { 0.5 } else { -0.5 };
            prices.push(prices[i - 1] + change);
        }
        let candles: Vec<Candle> = prices.into_iter().map(make_candle).collect();

        let kgz = KalmanGarchZScore::new(0.5, 0.1, 0.1, 0.85, 0.00001).with_min_periods(10);
        let result = kgz.compute(&candles);

        // First min_periods values should be NaN
        for (i, value) in result.iter().enumerate().take(10) {
            assert!(value.is_nan(), "Expected NaN at {}", i);
        }

        // Rest should be finite
        for (i, value) in result.iter().enumerate().take(50).skip(10) {
            assert!(
                value.is_finite(),
                "Expected finite at {}, got {}",
                i,
                value
            );
        }
    }

    #[test]
    fn test_kalman_garch_zscore_insufficient_data() {
        let candles: Vec<Candle> = vec![100.0; 5].into_iter().map(make_candle).collect();

        let kgz = KalmanGarchZScore::new(0.5, 0.1, 0.1, 0.85, 0.00001).with_min_periods(20);
        let result = kgz.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_kalman_garch_zscore_from_encoded() {
        let kgz = KalmanGarchZScore::from_encoded(500, 100, 100, 850, 10);
        assert!((kgz.kalman_r - 0.5).abs() < 1e-10);
        assert!((kgz.kalman_q - 0.1).abs() < 1e-10);
        assert!((kgz.garch_alpha - 0.1).abs() < 1e-10);
        assert!((kgz.garch_beta - 0.85).abs() < 1e-10);
        assert!((kgz.garch_omega - 0.00001).abs() < 1e-10);
    }
}
