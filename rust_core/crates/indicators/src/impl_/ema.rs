//! Exponential Moving Average (EMA) indicator

use crate::traits::Indicator;
use omega_types::Candle;

/// Exponential Moving Average
///
/// Matches pandas `ewm(span=period, adjust=False).mean()` semantics.
/// Multiplier = 2 / (period + 1)
#[derive(Debug, Clone)]
pub struct EMA {
    /// Number of periods for the EMA
    pub period: usize,
}

impl EMA {
    /// Creates a new EMA indicator with the given period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculates the EMA multiplier (smoothing factor).
    fn multiplier(&self) -> f64 {
        2.0 / (self.period as f64 + 1.0)
    }
}

impl Indicator for EMA {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if self.period == 0 || len == 0 {
            return result;
        }

        let alpha = self.multiplier();
        let mut prev = f64::NAN;

        for (i, candle) in candles.iter().enumerate() {
            let value = candle.close;
            if !value.is_finite() {
                if prev.is_finite() {
                    result[i] = prev;
                }
                continue;
            }

            if !prev.is_finite() {
                prev = value;
            } else {
                prev = alpha * value + (1.0 - alpha) * prev;
            }
            result[i] = prev;
        }

        result
    }

    fn name(&self) -> &str {
        "EMA"
    }

    fn warmup_periods(&self) -> usize {
        1
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
    fn test_ema_basic() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let ema = EMA::new(3);
        let result = ema.compute(&candles);

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.5).abs() < 1e-10);
        assert!((result[2] - 2.25).abs() < 1e-10);
        assert!((result[3] - 3.125).abs() < 1e-10);
        assert!((result[4] - 4.0625).abs() < 1e-10);
    }

    #[test]
    fn test_ema_converges_to_constant() {
        // When input is constant, EMA should converge to that constant
        let candles: Vec<Candle> = vec![5.0; 20].into_iter().map(make_candle).collect();

        let ema = EMA::new(5);
        let result = ema.compute(&candles);

        // EMA should stay at 5.0 for constant input
        for (i, value) in result.iter().enumerate().take(20) {
            assert!(
                (*value - 5.0).abs() < 1e-10,
                "EMA[{}] = {} != 5.0",
                i,
                value
            );
        }
    }

    #[test]
    fn test_ema_insufficient_data() {
        let candles: Vec<Candle> = vec![1.0, 2.0].into_iter().map(make_candle).collect();

        let ema = EMA::new(5);
        let result = ema.compute(&candles);

        assert_eq!(result.len(), candles.len());
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ema_multiplier() {
        let ema = EMA::new(10);
        let expected = 2.0 / 11.0;
        assert!((ema.multiplier() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ema_period_one_matches_close() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.5, 2.5]
            .into_iter()
            .map(make_candle)
            .collect();

        let ema = EMA::new(1);
        let result = ema.compute(&candles);

        for (candle, value) in candles.iter().zip(result.iter()) {
            assert!((*value - candle.close).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ema_period_zero_returns_nan() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let ema = EMA::new(0);
        let result = ema.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }
}
