//! Z-Score indicator

use crate::traits::Indicator;
use omega_types::Candle;

/// Z-Score indicator
///
/// Calculates the standardized score: (close - mean) / std
/// Uses a rolling window of the specified size with sample std (ddof=1).
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

            let denom = (self.window as f64) - 1.0;
            let std = if denom > 0.0 {
                let variance =
                    window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / denom;
                variance.sqrt()
            } else {
                f64::NAN
            };

            if std.is_finite() && std > 0.0 {
                result[i] = (candles[i].close - mean) / std;
            } else {
                result[i] = f64::NAN;
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
        // mean = 2, variance = 2/2 = 1, std = 1
        // z = (3 - 2) / 1 = 1
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(3);
        let result = zscore.compute(&candles);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        let expected_z = 1.0;
        assert!((result[2] - expected_z).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_zero_std() {
        // All same values -> std = 0 -> should return NaN
        let candles: Vec<Candle> = vec![5.0; 10].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(5);
        let result = zscore.compute(&candles);

        for value in result.iter().enumerate().take(10).skip(4).map(|(_, v)| v) {
            assert!(value.is_nan());
        }
    }

    #[test]
    fn test_zscore_symmetric_distribution() {
        // At the mean, z-score should be 0 or NaN depending on std
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let zscore = ZScore::new(3);
        let result = zscore.compute(&candles);

        // Index 3: window = [2, 3, 2], mean = 7/3, close = 2
        // close is at (2 - 7/3) which is negative
        assert!(result[3].is_nan() || result[3] < 0.0);
    }

    #[test]
    fn test_zscore_insufficient_data() {
        let candles: Vec<Candle> = vec![1.0, 2.0].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(5);
        let result = zscore.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_zscore_window_one_returns_nan() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let zscore = ZScore::new(1);
        let result = zscore.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }
}
