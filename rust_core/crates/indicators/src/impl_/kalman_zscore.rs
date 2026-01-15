//! Kalman Z-Score indicator

use crate::impl_::kalman_mean::KalmanFilter;
use crate::traits::Indicator;
use omega_types::Candle;

/// Kalman Z-Score
///
/// Calculates Z-Score based on residuals from Kalman-filtered price level.
/// Steps:
/// 1. Compute Kalman-smoothed price level
/// 2. Calculate residuals (price - kalman_level)
/// 3. Compute Z-Score of current residual over rolling window
#[derive(Debug, Clone)]
pub struct KalmanZScore {
    /// Window size for Z-Score calculation on residuals
    pub window: usize,
    /// Kalman measurement noise (R)
    pub r: f64,
    /// Kalman process noise (Q)
    pub q: f64,
}

impl KalmanZScore {
    /// Creates a new Kalman Z-Score indicator.
    pub fn new(window: usize, r: f64, q: f64) -> Self {
        Self { window, r, q }
    }

    /// Creates from x1000 encoded Kalman parameters.
    pub fn from_x1000(window: usize, r_x1000: u32, q_x1000: u32) -> Self {
        Self {
            window,
            r: r_x1000 as f64 / 1000.0,
            q: q_x1000 as f64 / 1000.0,
        }
    }
}

impl Indicator for KalmanZScore {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if len < self.window || self.window == 0 {
            return result;
        }

        // Extract prices
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

        // Compute Kalman level
        let kalman = KalmanFilter::new(self.r, self.q);
        let kalman_level = kalman.compute_level(&prices);

        // Compute Z-Score of residuals over rolling window
        for i in (self.window - 1)..len {
            let start = i + 1 - self.window;

            // Collect residuals in window
            let residuals: Vec<f64> = (start..=i)
                .map(|j| prices[j] - kalman_level[j])
                .collect();

            // Mean and std of residuals
            let mean = residuals.iter().sum::<f64>() / self.window as f64;
            let variance =
                residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / self.window as f64;
            let std = variance.sqrt();

            // Current residual
            let current_residual = prices[i] - kalman_level[i];

            if std > 1e-10 {
                result[i] = (current_residual - mean) / std;
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    fn name(&self) -> &str {
        "KALMAN_Z"
    }

    fn warmup_periods(&self) -> usize {
        self.window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            open: close,
            high: close,
            low: close,
            close,
            volume: 0.0,
        }
    }

    #[test]
    fn test_kalman_zscore_basic() {
        let candles: Vec<Candle> = vec![
            100.0, 101.0, 99.0, 100.0, 102.0, 98.0, 100.0, 101.0, 99.0, 100.0,
        ]
        .into_iter()
        .map(make_candle)
        .collect();

        let kz = KalmanZScore::new(5, 0.5, 0.1);
        let result = kz.compute(&candles);

        // First 4 values should be NaN (window = 5)
        for value in result.iter().take(4) {
            assert!(value.is_nan());
        }

        // Rest should be finite
        for (i, value) in result.iter().enumerate().take(10).skip(4) {
            assert!(
                value.is_finite(),
                "Expected finite at {}, got {}",
                i,
                value
            );
        }
    }

    #[test]
    fn test_kalman_zscore_constant_input() {
        // With constant input, residuals are ~0, so Z-Score should be 0
        let candles: Vec<Candle> = vec![100.0; 20].into_iter().map(make_candle).collect();

        let kz = KalmanZScore::new(5, 0.5, 0.1);
        let result = kz.compute(&candles);

        for (i, value) in result.iter().enumerate().take(20).skip(4) {
            assert!(
                (*value - 0.0).abs() < 1e-6,
                "Expected ~0 at {}, got {}",
                i,
                value
            );
        }
    }

    #[test]
    fn test_kalman_zscore_insufficient_data() {
        let candles: Vec<Candle> = vec![100.0, 101.0, 99.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let kz = KalmanZScore::new(5, 0.5, 0.1);
        let result = kz.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_kalman_zscore_from_x1000() {
        let kz = KalmanZScore::from_x1000(20, 500, 100);
        assert_eq!(kz.window, 20);
        assert!((kz.r - 0.5).abs() < 1e-10);
        assert!((kz.q - 0.1).abs() < 1e-10);
    }
}
