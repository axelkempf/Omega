//! Volatility Cluster State indicator.

use crate::impl_::atr::ATR;
use crate::traits::Indicator;
use omega_types::Candle;

/// Volatility regime states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolClusterState {
    /// Low volatility regime
    Low = 0,
    /// Normal volatility regime
    Normal = 1,
    /// High volatility regime
    High = 2,
}

impl VolClusterState {
    /// Converts from numeric representation.
    #[must_use]
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(VolClusterState::Low),
            1 => Some(VolClusterState::Normal),
            2 => Some(VolClusterState::High),
            _ => None,
        }
    }
}

/// Volatility Cluster indicator.
///
/// Classifies the current volatility regime based on ATR percentile
/// within a lookback window:
/// - Low: ATR below low threshold percentile
/// - Normal: ATR between thresholds
/// - High: ATR above high threshold percentile
#[derive(Debug, Clone)]
pub struct VolCluster {
    /// ATR period
    pub vol_period: usize,
    /// High volatility threshold (as percentile, e.g., 0.8 = 80th percentile)
    pub high_vol_threshold: f64,
    /// Low volatility threshold (as percentile, e.g., 0.2 = 20th percentile)
    pub low_vol_threshold: f64,
    /// Lookback period for percentile calculation
    pub lookback: usize,
}

impl VolCluster {
    /// Creates a new `VolCluster` indicator.
    #[must_use]
    pub fn new(
        vol_period: usize,
        high_vol_threshold: f64,
        low_vol_threshold: f64,
        lookback: usize,
    ) -> Self {
        Self {
            vol_period,
            high_vol_threshold,
            low_vol_threshold,
            lookback,
        }
    }

    /// Creates from x100 encoded thresholds.
    #[must_use]
    pub fn from_x100(
        vol_period: usize,
        high_vol_threshold_x100: u32,
        low_vol_threshold_x100: u32,
        lookback: usize,
    ) -> Self {
        Self {
            vol_period,
            high_vol_threshold: f64::from(high_vol_threshold_x100) / 100.0,
            low_vol_threshold: f64::from(low_vol_threshold_x100) / 100.0,
            lookback,
        }
    }

    /// Computes the state at each bar.
    ///
    #[must_use]
    pub fn compute_states(&self, candles: &[Candle]) -> Vec<Option<VolClusterState>> {
        let len = candles.len();
        let mut result = vec![None; len];

        if len < self.vol_period + self.lookback {
            return result;
        }

        // Compute ATR
        let atr = ATR::new(self.vol_period);
        let atr_values = atr.compute(candles);

        // For each bar after warmup, compute percentile rank
        let warmup = self.vol_period + self.lookback;
        for i in warmup..len {
            let current_atr = atr_values[i];
            if !current_atr.is_finite() {
                continue;
            }

            // Get lookback window of ATR values
            let start = i - self.lookback;
            let mut window = Vec::with_capacity(self.lookback);
            let mut has_invalid = false;
            for &v in atr_values.iter().take(i).skip(start) {
                if !v.is_finite() {
                    has_invalid = true;
                    break;
                }
                window.push(v);
            }

            if has_invalid {
                continue;
            }

            if window.is_empty() {
                continue;
            }

            // Sort for percentile calculation
            window.sort_by(f64::total_cmp);

            // Compute percentile rank of current ATR
            let count_below = window.iter().filter(|&&v| v < current_atr).count();
            #[allow(clippy::cast_precision_loss)]
            let percentile = count_below as f64 / window.len() as f64;

            // Classify
            let state = if percentile >= self.high_vol_threshold {
                VolClusterState::High
            } else if percentile <= self.low_vol_threshold {
                VolClusterState::Low
            } else {
                VolClusterState::Normal
            };

            result[i] = Some(state);
        }

        result
    }
}

impl Indicator for VolCluster {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        // Returns numeric state: 0=Low, 1=Normal, 2=High, NaN=undefined
        self.compute_states(candles)
            .into_iter()
            .map(|opt| opt.map_or(f64::NAN, |s| f64::from(s as i32)))
            .collect()
    }

    fn name(&self) -> &'static str {
        "VOL_CLUSTER"
    }

    fn warmup_periods(&self) -> usize {
        self.vol_period + self.lookback + 1
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
    fn test_vol_cluster_state_conversion() {
        assert_eq!(VolClusterState::from_i32(0), Some(VolClusterState::Low));
        assert_eq!(VolClusterState::from_i32(1), Some(VolClusterState::Normal));
        assert_eq!(VolClusterState::from_i32(2), Some(VolClusterState::High));
        assert_eq!(VolClusterState::from_i32(3), None);
    }

    #[test]
    fn test_vol_cluster_basic() {
        // Create candles with varying volatility
        let mut candles = Vec::new();

        // Low vol period (small ranges)
        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.1;
            candles.push(make_candle_ohlc(base, base + 0.1, base - 0.1, base));
        }

        // High vol period (large ranges)
        for i in 0..20 {
            let base = 103.0 + i as f64 * 0.1;
            candles.push(make_candle_ohlc(base, base + 2.0, base - 2.0, base));
        }

        let vc = VolCluster::new(5, 0.8, 0.2, 20);
        let states = vc.compute_states(&candles);

        // Find a bar in high vol period (should be High)
        // High vol starts at index 30, warmup is 5 + 20 = 25
        // So from index 30+ we should start seeing High
        let high_vol_idx = 45;
        assert!(
            states[high_vol_idx].is_some(),
            "Should have state at {}",
            high_vol_idx
        );

        // The last few bars should be High (high volatility)
        if let Some(state) = states[high_vol_idx] {
            assert_eq!(
                state,
                VolClusterState::High,
                "Expected High state in high vol period"
            );
        }
    }

    #[test]
    fn test_vol_cluster_insufficient_data() {
        let candles: Vec<Candle> = (0..10)
            .map(|i| make_candle_ohlc(100.0 + i as f64, 101.0, 99.0, 100.0))
            .collect();

        let vc = VolCluster::new(5, 0.8, 0.2, 20);
        let states = vc.compute_states(&candles);

        assert!(states.iter().all(|s| s.is_none()));
    }

    #[test]
    fn test_vol_cluster_nan_in_window_yields_none() {
        let mut candles = Vec::new();

        for _ in 0..3 {
            candles.push(make_candle_ohlc(f64::NAN, f64::NAN, f64::NAN, f64::NAN));
        }

        for i in 0..6 {
            let base = 100.0 + i as f64;
            candles.push(make_candle_ohlc(base, base + 1.0, base - 1.0, base));
        }

        let vc = VolCluster::new(2, 0.8, 0.2, 2);
        let states = vc.compute_states(&candles);

        let idx = vc.vol_period + vc.lookback;
        assert!(states[idx].is_none());
    }

    #[test]
    fn test_vol_cluster_from_x100() {
        let vc = VolCluster::from_x100(14, 80, 20, 50);
        assert_eq!(vc.vol_period, 14);
        assert!((vc.high_vol_threshold - 0.8).abs() < 1e-10);
        assert!((vc.low_vol_threshold - 0.2).abs() < 1e-10);
        assert_eq!(vc.lookback, 50);
    }
}
