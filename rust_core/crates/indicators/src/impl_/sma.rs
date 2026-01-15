//! Simple Moving Average (SMA) indicator

use crate::traits::Indicator;
use omega_types::Candle;

/// Simple Moving Average
///
/// Calculates the arithmetic mean of the last N close prices.
#[derive(Debug, Clone)]
pub struct SMA {
    /// Number of periods for the moving average
    pub period: usize,
}

impl SMA {
    /// Creates a new SMA indicator with the given period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Indicator for SMA {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if len < self.period || self.period == 0 {
            return result;
        }

        // Calculate initial sum
        let mut sum: f64 = candles[..self.period].iter().map(|c| c.close).sum();
        result[self.period - 1] = sum / self.period as f64;

        // Rolling calculation
        for i in self.period..len {
            sum += candles[i].close - candles[i - self.period].close;
            result[i] = sum / self.period as f64;
        }

        result
    }

    fn name(&self) -> &str {
        "SMA"
    }

    fn warmup_periods(&self) -> usize {
        self.period
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
    fn test_sma_basic() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let sma = SMA::new(3);
        let result = sma.compute(&candles);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10); // (1+2+3)/3 = 2.0
        assert!((result[3] - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3.0
        assert!((result[4] - 4.0).abs() < 1e-10); // (3+4+5)/3 = 4.0
    }

    #[test]
    fn test_sma_constant_input() {
        let candles: Vec<Candle> = vec![5.0; 10].into_iter().map(make_candle).collect();

        let sma = SMA::new(3);
        let result = sma.compute(&candles);

        // After warmup, all values should be 5.0
        for value in result.iter().take(10).skip(2) {
            assert!((*value - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sma_insufficient_data() {
        let candles: Vec<Candle> = vec![1.0, 2.0].into_iter().map(make_candle).collect();

        let sma = SMA::new(5);
        let result = sma.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_sma_period_one_matches_close() {
        let candles: Vec<Candle> = vec![1.5, 2.5, 3.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let sma = SMA::new(1);
        let result = sma.compute(&candles);

        for (candle, value) in candles.iter().zip(result.iter()) {
            assert!((*value - candle.close).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sma_period_zero_returns_nan() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let sma = SMA::new(0);
        let result = sma.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }
}
