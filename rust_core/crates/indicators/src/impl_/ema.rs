//! Exponential Moving Average (EMA) indicator

use crate::traits::Indicator;
use omega_types::Candle;

/// Exponential Moving Average
///
/// Uses SMA for initialization and then applies exponential smoothing.
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

        if len < self.period || self.period == 0 {
            return result;
        }

        let multiplier = self.multiplier();

        // Initial SMA for seed value
        let initial_sma: f64 =
            candles[..self.period].iter().map(|c| c.close).sum::<f64>() / self.period as f64;

        result[self.period - 1] = initial_sma;

        // EMA calculation: EMA_t = (close - EMA_{t-1}) * multiplier + EMA_{t-1}
        for i in self.period..len {
            let prev_ema = result[i - 1];
            result[i] = (candles[i].close - prev_ema) * multiplier + prev_ema;
        }

        result
    }

    fn name(&self) -> &str {
        "EMA"
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
    fn test_ema_basic() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let ema = EMA::new(3);
        let result = ema.compute(&candles);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Initial SMA = (1+2+3)/3 = 2.0
        assert!((result[2] - 2.0).abs() < 1e-10);

        // EMA[3] = (4 - 2) * 0.5 + 2 = 3.0 (multiplier = 2/(3+1) = 0.5)
        assert!((result[3] - 3.0).abs() < 1e-10);

        // EMA[4] = (5 - 3) * 0.5 + 3 = 4.0
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_converges_to_constant() {
        // When input is constant, EMA should converge to that constant
        let candles: Vec<Candle> = vec![5.0; 20].into_iter().map(make_candle).collect();

        let ema = EMA::new(5);
        let result = ema.compute(&candles);

        // After warmup, all values should be 5.0
        for i in 4..20 {
            assert!(
                (result[i] - 5.0).abs() < 1e-10,
                "EMA[{}] = {} != 5.0",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_ema_insufficient_data() {
        let candles: Vec<Candle> = vec![1.0, 2.0].into_iter().map(make_candle).collect();

        let ema = EMA::new(5);
        let result = ema.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_ema_multiplier() {
        let ema = EMA::new(10);
        let expected = 2.0 / 11.0;
        assert!((ema.multiplier() - expected).abs() < 1e-10);
    }
}
