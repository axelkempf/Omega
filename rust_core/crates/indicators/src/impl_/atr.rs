//! Average True Range (ATR) indicator with Wilder smoothing

use crate::traits::Indicator;
use omega_types::Candle;

/// Average True Range (Wilder)
///
/// Uses Wilder's smoothing method: ATR = (prev_ATR * (n-1) + TR) / n
/// This is different from a simple moving average and provides a more
/// responsive measure of volatility.
#[derive(Debug, Clone)]
pub struct ATR {
    /// Number of periods for ATR calculation
    pub period: usize,
}

impl ATR {
    /// Creates a new ATR indicator with the given period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculates True Range for a candle given the previous close.
    ///
    /// TR = max(High - Low, |High - Prev_Close|, |Low - Prev_Close|)
    #[inline]
    fn true_range(candle: &Candle, prev_close: f64) -> f64 {
        let hl = candle.high - candle.low;
        let hc = (candle.high - prev_close).abs();
        let lc = (candle.low - prev_close).abs();
        hl.max(hc).max(lc)
    }
}

impl Indicator for ATR {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if self.period == 0 || len == 0 {
            return result;
        }

        // Calculate True Range series
        let mut tr = vec![0.0; len];
        tr[0] = candles[0].high - candles[0].low; // First TR is just H-L

        for i in 1..len {
            tr[i] = Self::true_range(&candles[i], candles[i - 1].close);
        }

        let first_valid = tr.iter().position(|v| v.is_finite());
        let Some(first_valid) = first_valid else {
            return result;
        };

        if first_valid + self.period <= len
            && tr[first_valid..first_valid + self.period]
                .iter()
                .all(|v| v.is_finite())
        {
            let initial: f64 = tr[first_valid..first_valid + self.period]
                .iter()
                .sum::<f64>()
                / self.period as f64;
            let start_idx = first_valid + self.period - 1;
            result[start_idx] = initial;

            for i in (start_idx + 1)..len {
                if !tr[i].is_finite() {
                    result[i] = result[i - 1];
                    continue;
                }
                result[i] = (result[i - 1] * (self.period - 1) as f64 + tr[i])
                    / self.period as f64;
            }
        }

        result
    }

    fn name(&self) -> &str {
        "ATR"
    }

    fn warmup_periods(&self) -> usize {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle_ohlc(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            open,
            high,
            low,
            close,
            volume: 0.0,
        }
    }

    #[test]
    fn test_true_range() {
        // Normal range
        let candle = make_candle_ohlc(100.0, 105.0, 95.0, 102.0);
        let tr = ATR::true_range(&candle, 100.0);
        assert!((tr - 10.0).abs() < 1e-10); // H-L = 10

        // Gap up
        let candle = make_candle_ohlc(110.0, 115.0, 108.0, 112.0);
        let tr = ATR::true_range(&candle, 100.0);
        assert!((tr - 15.0).abs() < 1e-10); // H - prev_close = 15

        // Gap down
        let candle = make_candle_ohlc(90.0, 92.0, 85.0, 88.0);
        let tr = ATR::true_range(&candle, 100.0);
        assert!((tr - 15.0).abs() < 1e-10); // prev_close - L = 15
    }

    #[test]
    fn test_atr_basic() {
        let candles = vec![
            make_candle_ohlc(100.0, 102.0, 98.0, 101.0),  // TR = 4 (H-L)
            make_candle_ohlc(101.0, 104.0, 99.0, 103.0),  // TR = 5 (H-L)
            make_candle_ohlc(103.0, 106.0, 101.0, 105.0), // TR = 5 (H-L)
            make_candle_ohlc(105.0, 108.0, 103.0, 107.0), // TR = 5 (H-L)
            make_candle_ohlc(107.0, 110.0, 105.0, 109.0), // TR = 5 (H-L)
        ];

        let atr = ATR::new(3);
        let result = atr.compute(&candles);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Initial ATR = (4 + 5 + 5) / 3 = 4.6666...
        assert!((result[2] - 4.6666666667).abs() < 1e-8);

        // ATR[3] = (4.6666... * 2 + 5) / 3
        assert!((result[3] - 4.7777777778).abs() < 1e-8);
    }

    #[test]
    fn test_atr_insufficient_data() {
        let candles = vec![
            make_candle_ohlc(100.0, 102.0, 98.0, 101.0),
            make_candle_ohlc(101.0, 104.0, 99.0, 103.0),
        ];

        let atr = ATR::new(5);
        let result = atr.compute(&candles);

        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_atr_warmup() {
        let atr = ATR::new(14);
        assert_eq!(atr.warmup_periods(), 14);
    }
}
