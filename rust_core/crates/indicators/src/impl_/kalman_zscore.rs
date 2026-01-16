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
/// 3. Compute Z-Score of current residual over rolling window (sample std, ddof=1)
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

        let residuals: Vec<f64> = prices
            .iter()
            .zip(kalman_level.iter())
            .map(|(p, k)| p - k)
            .collect();

        for i in (self.window - 1)..len {
            let start = i + 1 - self.window;
            let window_residuals = &residuals[start..=i];
            let std = sample_std(window_residuals);
            let current_residual = residuals[i];

            if std.is_finite() && std > 0.0 {
                result[i] = current_residual / std;
            } else {
                result[i] = f64::NAN;
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

fn sample_std(values: &[f64]) -> f64 {
    if values.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let denom = (values.len() as f64) - 1.0;
    let variance = values
        .iter()
        .map(|v| (*v - mean).powi(2))
        .sum::<f64>()
        / denom;
    variance.sqrt()
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
        // With constant input, std is 0, so Z-Score is NaN
        let candles: Vec<Candle> = vec![100.0; 20].into_iter().map(make_candle).collect();

        let kz = KalmanZScore::new(5, 0.5, 0.1);
        let result = kz.compute(&candles);

        for value in result.iter().enumerate().take(20).skip(4).map(|(_, v)| v) {
            assert!(value.is_nan());
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
