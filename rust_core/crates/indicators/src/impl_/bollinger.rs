//! Bollinger Bands indicator

use crate::traits::{IntoMultiVecs, MultiOutputIndicator};
use omega_types::Candle;

/// Bollinger Bands result containing upper, middle, and lower bands.
#[derive(Debug, Clone)]
pub struct BollingerResult {
    /// Upper band = SMA + std_factor * std
    pub upper: Vec<f64>,
    /// Middle band = SMA
    pub middle: Vec<f64>,
    /// Lower band = SMA - std_factor * std
    pub lower: Vec<f64>,
}

impl IntoMultiVecs for BollingerResult {
    fn into_vecs(self) -> Vec<Vec<f64>> {
        vec![self.upper, self.middle, self.lower]
    }
}

/// Bollinger Bands
///
/// Calculates three bands based on standard deviation around a simple moving average:
/// - Upper Band = SMA + (std_factor * StdDev)
/// - Middle Band = SMA
/// - Lower Band = SMA - (std_factor * StdDev)
///
/// Uses population standard deviation (n), not sample (n-1), for V1 parity.
#[derive(Debug, Clone)]
pub struct BollingerBands {
    /// Period for the SMA and standard deviation
    pub period: usize,
    /// Multiplier for standard deviation (typically 2.0)
    pub std_factor: f64,
}

impl BollingerBands {
    /// Creates new Bollinger Bands with the given parameters.
    pub fn new(period: usize, std_factor: f64) -> Self {
        Self { period, std_factor }
    }

    /// Creates Bollinger Bands from x100 encoded std_factor.
    pub fn from_x100(period: usize, std_factor_x100: u32) -> Self {
        Self {
            period,
            std_factor: std_factor_x100 as f64 / 100.0,
        }
    }
}

impl MultiOutputIndicator for BollingerBands {
    type Output = BollingerResult;

    fn compute_all(&self, candles: &[Candle]) -> Self::Output {
        let len = candles.len();
        let mut upper = vec![f64::NAN; len];
        let mut middle = vec![f64::NAN; len];
        let mut lower = vec![f64::NAN; len];

        if len < self.period || self.period == 0 {
            return BollingerResult {
                upper,
                middle,
                lower,
            };
        }

        for i in (self.period - 1)..len {
            let start = i + 1 - self.period;
            let window: Vec<f64> = candles[start..=i].iter().map(|c| c.close).collect();

            // SMA
            let sma = window.iter().sum::<f64>() / self.period as f64;

            // Population variance (n, not n-1) for V1 parity
            let variance =
                window.iter().map(|x| (x - sma).powi(2)).sum::<f64>() / self.period as f64;
            let std = variance.sqrt();

            middle[i] = sma;
            upper[i] = sma + self.std_factor * std;
            lower[i] = sma - self.std_factor * std;
        }

        BollingerResult {
            upper,
            middle,
            lower,
        }
    }

    fn name(&self) -> &str {
        "BOLLINGER"
    }

    fn warmup_periods(&self) -> usize {
        self.period
    }

    fn output_names(&self) -> &'static [&'static str] {
        &["upper", "middle", "lower"]
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
    fn test_bollinger_basic() {
        // Use simple sequence where math is easy
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let bb = BollingerBands::new(3, 2.0);
        let result = bb.compute_all(&candles);

        assert!(result.middle[0].is_nan());
        assert!(result.middle[1].is_nan());

        // At index 2: window = [1, 2, 3]
        // SMA = 2.0
        // Variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        // Std = sqrt(2/3) = 0.8165...
        let expected_sma = 2.0;
        let expected_std = (2.0_f64 / 3.0).sqrt();

        assert!((result.middle[2] - expected_sma).abs() < 1e-10);
        assert!((result.upper[2] - (expected_sma + 2.0 * expected_std)).abs() < 1e-10);
        assert!((result.lower[2] - (expected_sma - 2.0 * expected_std)).abs() < 1e-10);
    }

    #[test]
    fn test_bollinger_constant_input() {
        // When all values are the same, std should be 0
        let candles: Vec<Candle> = vec![100.0; 10].into_iter().map(make_candle).collect();

        let bb = BollingerBands::new(5, 2.0);
        let result = bb.compute_all(&candles);

        for i in 4..10 {
            assert!((result.middle[i] - 100.0).abs() < 1e-10);
            assert!((result.upper[i] - 100.0).abs() < 1e-10); // std = 0
            assert!((result.lower[i] - 100.0).abs() < 1e-10); // std = 0
        }
    }

    #[test]
    fn test_bollinger_symmetry() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let bb = BollingerBands::new(3, 2.0);
        let result = bb.compute_all(&candles);

        // Check that upper and lower are symmetric around middle
        for i in 2..candles.len() {
            let mid = result.middle[i];
            let upper_dist = result.upper[i] - mid;
            let lower_dist = mid - result.lower[i];
            assert!(
                (upper_dist - lower_dist).abs() < 1e-10,
                "Bands not symmetric at index {}",
                i
            );
        }
    }

    #[test]
    fn test_bollinger_insufficient_data() {
        let candles: Vec<Candle> = vec![1.0, 2.0].into_iter().map(make_candle).collect();

        let bb = BollingerBands::new(5, 2.0);
        let result = bb.compute_all(&candles);

        assert!(result.upper.iter().all(|v| v.is_nan()));
        assert!(result.middle.iter().all(|v| v.is_nan()));
        assert!(result.lower.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_bollinger_from_x100() {
        let bb = BollingerBands::from_x100(20, 200);
        assert_eq!(bb.period, 20);
        assert!((bb.std_factor - 2.0).abs() < 1e-10);
    }
}
