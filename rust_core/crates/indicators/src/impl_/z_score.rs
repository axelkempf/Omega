//! Z-Score indicator.

use crate::traits::{Indicator, ZScoreMeanSource};
use omega_types::Candle;

/// Z-Score indicator.
///
/// Calculates the standardized score: (close - mean) / std.
///
/// Mean source can be rolling or EMA-based. Standard deviation uses
/// sample std (ddof=1) with NaN propagation on invalid inputs.
#[derive(Debug, Clone)]
pub struct ZScore {
    /// Window size for mean and standard deviation.
    pub window: usize,
    /// Mean source selection.
    pub mean_source: ZScoreMeanSource,
    /// EMA period when `mean_source == ZScoreMeanSource::Ema`.
    pub ema_period: Option<usize>,
}

impl ZScore {
    /// Creates a new Z-Score indicator with the given window.
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self::with_mean_source(window, ZScoreMeanSource::Rolling, None)
    }

    /// Creates a new Z-Score indicator with explicit mean source.
    #[must_use]
    pub fn with_mean_source(
        window: usize,
        mean_source: ZScoreMeanSource,
        ema_period: Option<usize>,
    ) -> Self {
        Self {
            window,
            mean_source,
            ema_period,
        }
    }

    /// Creates a new Z-Score indicator using EMA mean source.
    #[must_use]
    pub fn with_ema_mean(window: usize, ema_period: usize) -> Self {
        Self::with_mean_source(window, ZScoreMeanSource::Ema, Some(ema_period))
    }
}

impl Indicator for ZScore {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let len = candles.len();
        let mut result = vec![f64::NAN; len];

        if len < self.window || self.window == 0 {
            return result;
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

        match self.mean_source {
            ZScoreMeanSource::Rolling => {
                let (mean, std) = rolling_sample_mean_std(&closes, self.window);
                for i in (self.window - 1)..len {
                    let close = closes[i];
                    let mean = mean[i];
                    let std = std[i];
                    if !close.is_finite() || !mean.is_finite() {
                        continue;
                    }
                    if std.is_finite() && std > 0.0 {
                        result[i] = (close - mean) / std;
                    } else if std.is_finite() && std == 0.0 {
                        result[i] = 0.0;
                    }
                }
            }
            ZScoreMeanSource::Ema => {
                let ema_period = match self.ema_period {
                    Some(period) if period > 0 => period,
                    _ => return result,
                };

                let ema = ema_mean_series(&closes, ema_period);
                let residuals: Vec<f64> = closes
                    .iter()
                    .zip(ema.iter())
                    .map(|(close, mean)| {
                        if close.is_finite() && mean.is_finite() {
                            close - mean
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                let std = rolling_sample_std(&residuals, self.window);

                for i in (self.window - 1)..len {
                    let resid = residuals[i];
                    let std = std[i];
                    if resid.is_finite() && std.is_finite() && std > 0.0 {
                        result[i] = resid / std;
                    }
                }
            }
        }

        result
    }

    fn name(&self) -> &'static str {
        "Z_SCORE"
    }

    fn warmup_periods(&self) -> usize {
        self.window
    }
}

#[must_use]
fn rolling_sample_mean_std(values: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let len = values.len();
    let mut mean = vec![f64::NAN; len];
    let mut std = vec![f64::NAN; len];

    if window == 0 || len < window {
        return (mean, std);
    }

    let mut sum = vec![0.0; len + 1];
    let mut sum_sq = vec![0.0; len + 1];
    let mut nonfinite = vec![0usize; len + 1];

    for (i, value) in values.iter().enumerate() {
        sum[i + 1] = sum[i];
        sum_sq[i + 1] = sum_sq[i];
        nonfinite[i + 1] = nonfinite[i];

        if value.is_finite() {
            sum[i + 1] += *value;
            sum_sq[i + 1] += *value * *value;
        } else {
            nonfinite[i + 1] += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let window_f = window as f64;
    let denom = window_f - 1.0;

    for i in (window - 1)..len {
        let start = i + 1 - window;
        let end = i + 1;
        if nonfinite[end] - nonfinite[start] > 0 {
            continue;
        }

        let window_sum = sum[end] - sum[start];
        let window_sum_sq = sum_sq[end] - sum_sq[start];
        let mean_val = window_sum / window_f;
        mean[i] = mean_val;

        if denom <= 0.0 {
            continue;
        }

        let variance = (window_sum_sq - (window_sum * window_sum) / window_f) / denom;
        if variance.is_finite() {
            let variance = if variance < 0.0 { 0.0 } else { variance };
            std[i] = variance.sqrt();
        }
    }

    (mean, std)
}

#[must_use]
fn rolling_sample_std(values: &[f64], window: usize) -> Vec<f64> {
    let (_mean, std) = rolling_sample_mean_std(values, window);
    std
}

#[must_use]
fn ema_mean_series(values: &[f64], period: usize) -> Vec<f64> {
    let len = values.len();
    let mut result = vec![f64::NAN; len];
    if period == 0 || len == 0 {
        return result;
    }

    #[allow(clippy::cast_precision_loss)]
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut prev = f64::NAN;

    for (i, value) in values.iter().enumerate() {
        if !value.is_finite() {
            if prev.is_finite() {
                result[i] = prev;
            }
            continue;
        }

        if prev.is_finite() {
            prev = alpha.mul_add(*value, (1.0 - alpha) * prev);
        } else {
            prev = *value;
        }
        result[i] = prev;
    }

    result
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
        // All same values -> std = 0 -> z-score should be 0
        let candles: Vec<Candle> = vec![5.0; 10].into_iter().map(make_candle).collect();

        let zscore = ZScore::new(5);
        let result = zscore.compute(&candles);

        for value in result.iter().enumerate().take(10).skip(4).map(|(_, v)| v) {
            assert!((*value - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zscore_with_ema_mean() {
        let candles: Vec<Candle> = vec![1.0, 1.2, 0.9, 1.1, 1.3, 1.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let zscore = ZScore::with_ema_mean(3, 2);
        let result = zscore.compute(&candles);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_finite());
        assert!(result[3].is_finite());
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
