//! Z-Score indicator

use crate::traits::Indicator;
use omega_types::Candle;

/// Z-Score indicator
///
/// Calculates the standardized score: (close - mean) / std
/// Uses a rolling window of the specified size.
/// When std is near zero (< 1e-10), returns 0.0 instead of NaN/Inf.
#[derive(Debug, Clone)]
pub struct ZScore {
    /// Window size for mean and standard deviation
    pub window: usize,
}

impl ZScore {
    /// Creates a new Z-Score indicator with the given window.
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

impl Indicator for ZScore {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if len < self.window || self.window == 0 {
            return result;
        }

        for i in (self.window - 1)..len {
            let start = i + 1 - self.window;
            let window: Vec<f64> = candles[start..=i].iter().map(|c| c.close).collect();

            let mean = window.iter().sum::<f64>() / self.window as f64;

            // Population variance (n, not n-1)
            let variance =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.window as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                result[i] = (candles[i].close - mean) / std;
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    fn name(&self) -> &str {
        "Z_SCORE"
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
    fn test_zscore_basic() {
        // window = [1, 2, 3], current = 3
        // mean = 2, variance = 2/3, std = 0.8165
        // z = (3 - 2) / 0.8165 = 1.2247
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(3);
        let result = zscore.compute(&candles);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        let expected_std = (2.0_f64 / 3.0).sqrt();
        let expected_z = (3.0 - 2.0) / expected_std;
        assert!((result[2] - expected_z).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_zero_std() {
        // All same values -> std = 0 -> should return 0
        let candles: Vec<Candle> = vec![5.0; 10].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(5);
        let result = zscore.compute(&candles);

        for i in 4..10 {
            assert!(
                (result[i] - 0.0).abs() < 1e-10,
                "Expected 0.0 at {}, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_zscore_symmetric_distribution() {
        // At the mean, z-score should be 0
        // Below mean should be negative, above should be positive
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let zscore = ZScore::new(3);
        let result = zscore.compute(&candles);

        // Index 3: window = [2, 3, 2], mean = 7/3, close = 2
        // close is at (2 - 7/3) which is negative
        assert!(result[3] < 0.0);
    }

    #[test]
    fn test_zscore_insufficient_data() {
        let candles: Vec<Candle> = vec![1.0, 2.0].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(5);
        let result = zscore.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }
}
