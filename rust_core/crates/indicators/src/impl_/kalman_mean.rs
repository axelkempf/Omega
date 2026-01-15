//! Kalman Filter for price level estimation

use omega_types::Candle;

/// Kalman Filter for 1D state estimation
///
/// Uses a simple constant velocity model to estimate the underlying price level.
/// - R: Measurement noise covariance (higher = trust measurements less)
/// - Q: Process noise covariance (higher = expect more change between observations)
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// Measurement noise variance
    pub r: f64,
    /// Process noise variance
    pub q: f64,
}

impl KalmanFilter {
    /// Creates a new Kalman Filter with the given noise parameters.
    pub fn new(r: f64, q: f64) -> Self {
        Self { r, q }
    }

    /// Creates a Kalman Filter from x1000 encoded parameters.
    pub fn from_x1000(r_x1000: u32, q_x1000: u32) -> Self {
        Self {
            r: r_x1000 as f64 / 1000.0,
            q: q_x1000 as f64 / 1000.0,
        }
    }

    /// Computes Kalman-smoothed price level series from raw prices.
    ///
    /// Returns Vec<f64> with the same length as prices.
    /// First value is initialized to the first price.
    pub fn compute_level(&self, prices: &[f64]) -> Vec<f64> {
        let len = prices.len();
        let mut result = vec![f64::NAN; len];

        if prices.is_empty() {
            return result;
        }

        // Initialize state
        let mut x = prices[0]; // State estimate
        let mut p = 1.0; // Error covariance (start with unit variance)

        result[0] = x;

        for i in 1..len {
            // Predict step
            let x_pred = x;
            let p_pred = p + self.q;

            // Update step
            let k = p_pred / (p_pred + self.r);
            x = x_pred + k * (prices[i] - x_pred);
            p = (1.0 - k) * p_pred;

            result[i] = x;
        }

        result
    }

    /// Computes Kalman-smoothed level from candles (using close prices).
    pub fn compute_level_from_candles(&self, candles: &[Candle]) -> Vec<f64> {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        self.compute_level(&prices)
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
    fn test_kalman_constant_input() {
        // With constant input, output should converge to that value
        let prices: Vec<f64> = vec![100.0; 20];
        let kalman = KalmanFilter::new(0.5, 0.01);
        let result = kalman.compute_level(&prices);

        // First value is initialized to input
        assert!((result[0] - 100.0).abs() < 1e-10);

        // Should stay near 100
        for level in result.iter() {
            assert!((level - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kalman_smoothing() {
        // Noisy signal around 100
        let prices = vec![
            100.0, 102.0, 98.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.0, 99.0,
        ];
        let kalman = KalmanFilter::new(1.0, 0.1);
        let result = kalman.compute_level(&prices);

        // Kalman should be smoother - check variance is lower
        let price_var: f64 = prices.iter().map(|x| (x - 100.0).powi(2)).sum::<f64>() / 10.0;
        let kalman_var: f64 = result.iter().map(|x| (x - 100.0).powi(2)).sum::<f64>() / 10.0;

        assert!(
            kalman_var < price_var,
            "Kalman should smooth: {} vs {}",
            kalman_var,
            price_var
        );
    }

    #[test]
    fn test_kalman_tracks_trend() {
        // Linear trend
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let kalman = KalmanFilter::new(0.5, 0.1);
        let result = kalman.compute_level(&prices);

        // Should track the trend (with some lag)
        // Last value should be close to last price
        let last_diff = (result[19] - prices[19]).abs();
        assert!(last_diff < 2.0, "Should track trend: diff = {}", last_diff);
    }

    #[test]
    fn test_kalman_empty_input() {
        let prices: Vec<f64> = vec![];
        let kalman = KalmanFilter::new(0.5, 0.01);
        let result = kalman.compute_level(&prices);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kalman_from_candles() {
        let candles: Vec<Candle> = vec![100.0, 101.0, 99.0, 100.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let kalman = KalmanFilter::new(0.5, 0.1);
        let result = kalman.compute_level_from_candles(&candles);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_kalman_from_x1000() {
        let kalman = KalmanFilter::from_x1000(500, 10);
        assert!((kalman.r - 0.5).abs() < 1e-10);
        assert!((kalman.q - 0.01).abs() < 1e-10);
    }
}
