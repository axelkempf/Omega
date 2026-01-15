//! Indicator traits and specifications
//!
//! Defines the core traits and types for indicators.

use omega_types::Candle;
use std::collections::HashMap;

/// Specification for an indicator including name and parameters.
/// Used as cache keys to identify computed indicator series.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndicatorSpec {
    /// Indicator name (e.g., "EMA", "ATR", "BOLLINGER_upper")
    pub name: String,
    /// Parameters for the indicator
    pub params: IndicatorParams,
}

impl IndicatorSpec {
    /// Creates a new indicator specification.
    pub fn new(name: impl Into<String>, params: IndicatorParams) -> Self {
        Self {
            name: name.into(),
            params,
        }
    }

    /// Creates a composite key for multi-output indicators.
    pub fn with_output_suffix(&self, output_name: &str) -> Self {
        Self {
            name: format!("{}_{}", self.name, output_name),
            params: self.params.clone(),
        }
    }
}

/// Parameters for indicator configuration.
/// Uses integer representations (x100, x1000, etc.) for hashability.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndicatorParams {
    /// Simple period-based parameter (EMA, SMA, ATR, Z-Score)
    Period(usize),

    /// Bollinger Bands parameters
    Bollinger {
        period: usize,
        /// Standard deviation factor * 100 (e.g., 200 = 2.0)
        std_factor_x100: u32,
    },

    /// Kalman filter parameters
    Kalman {
        window: usize,
        /// R (measurement noise) * 1000
        r_x1000: u32,
        /// Q (process noise) * 1000
        q_x1000: u32,
    },

    /// GARCH(1,1) parameters
    Garch {
        /// Alpha * 1000
        alpha_x1000: u32,
        /// Beta * 1000
        beta_x1000: u32,
        /// Omega * 1,000,000
        omega_x1000000: u32,
    },

    /// Kalman+GARCH combination
    KalmanGarch {
        window: usize,
        r_x1000: u32,
        q_x1000: u32,
        alpha_x1000: u32,
        beta_x1000: u32,
        omega_x1000000: u32,
    },

    /// Vol-Cluster parameters
    VolCluster {
        vol_period: usize,
        /// High volatility threshold * 100
        high_vol_threshold_x100: u32,
        /// Low volatility threshold * 100
        low_vol_threshold_x100: u32,
    },

    /// Custom parameters as key-value pairs (for extensibility)
    Custom(Vec<(String, i64)>),
}

impl Default for IndicatorParams {
    fn default() -> Self {
        IndicatorParams::Period(14)
    }
}

/// Trait for single-output indicators.
///
/// All indicators compute over the full candle series and return a Vec<f64>
/// of the same length. Values before the warmup period are NaN.
pub trait Indicator: Send + Sync {
    /// Computes the indicator for all candles.
    ///
    /// Returns Vec<f64> with the same length as candles.
    /// Values at indices < warmup_periods() are f64::NAN.
    fn compute(&self, candles: &[Candle]) -> Vec<f64>;

    /// Name of the indicator (e.g., "EMA", "ATR").
    fn name(&self) -> &str;

    /// Minimum number of bars required for valid output.
    fn warmup_periods(&self) -> usize;
}

/// Trait for multi-output indicators like Bollinger Bands.
///
/// These indicators produce multiple series (e.g., upper, middle, lower bands)
/// that are computed together for efficiency.
pub trait MultiOutputIndicator: Send + Sync {
    /// Type of the output structure
    type Output: IntoMultiVecs;

    /// Computes all outputs at once.
    fn compute_all(&self, candles: &[Candle]) -> Self::Output;

    /// Name of the indicator.
    fn name(&self) -> &str;

    /// Minimum number of bars for valid output.
    fn warmup_periods(&self) -> usize;

    /// List of output names (used for cache keys).
    fn output_names(&self) -> &'static [&'static str];
}

/// Trait for converting multi-output results into a vector of vectors.
pub trait IntoMultiVecs {
    /// Converts the output structure into a vector of value vectors.
    fn into_vecs(self) -> Vec<Vec<f64>>;
}

/// Result container for multi-output indicator access.
#[derive(Debug, Clone)]
pub struct MultiOutputResult {
    /// Map from output name to computed values
    pub outputs: HashMap<String, Vec<f64>>,
}

impl MultiOutputResult {
    /// Creates a new empty result.
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
        }
    }

    /// Gets a specific output by name.
    pub fn get(&self, name: &str) -> Option<&Vec<f64>> {
        self.outputs.get(name)
    }

    /// Gets a value at a specific index from an output.
    pub fn get_at(&self, name: &str, idx: usize) -> Option<f64> {
        self.outputs.get(name).and_then(|v| v.get(idx).copied())
    }
}

impl Default for MultiOutputResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator_spec_with_output_suffix() {
        let spec = IndicatorSpec::new("BOLLINGER", IndicatorParams::Bollinger {
            period: 20,
            std_factor_x100: 200,
        });

        let upper_spec = spec.with_output_suffix("upper");
        assert_eq!(upper_spec.name, "BOLLINGER_upper");
        assert_eq!(upper_spec.params, spec.params);
    }

    #[test]
    fn test_indicator_params_hash_equality() {
        let p1 = IndicatorParams::Period(14);
        let p2 = IndicatorParams::Period(14);
        let p3 = IndicatorParams::Period(20);

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);

        // Test as HashMap key
        let mut map = HashMap::new();
        map.insert(p1.clone(), "value1");
        assert!(map.contains_key(&p2));
        assert!(!map.contains_key(&p3));
    }
}
