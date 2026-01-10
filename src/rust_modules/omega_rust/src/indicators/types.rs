//! Type definitions for the IndicatorCache module.
//!
//! This module provides the core data structures used by the Rust-based
//! indicator cache implementation.
//!
//! ## Reference
//!
//! - FFI Specification: `docs/ffi/indicator_cache.md`
//! - Migration Plan: `docs/WAVE_1_INDICATOR_CACHE_IMPLEMENTATION_PLAN.md`

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// OHLCV data as columnar arrays.
///
/// Represents a single timeframe/price_type combination of candle data.
/// All arrays have the same length (n_bars).
#[derive(Debug, Clone)]
pub struct OhlcvData {
    /// Opening prices
    pub open: Vec<f64>,
    /// High prices
    pub high: Vec<f64>,
    /// Low prices
    pub low: Vec<f64>,
    /// Closing prices
    pub close: Vec<f64>,
    /// Volume
    pub volume: Vec<f64>,
    /// Validity mask (true = valid candle, false = None/missing)
    pub valid: Vec<bool>,
    /// Number of bars
    pub n_bars: usize,
}

impl OhlcvData {
    /// Create a new OhlcvData instance.
    ///
    /// # Arguments
    ///
    /// * `open` - Opening prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Closing prices
    /// * `volume` - Volume values
    /// * `valid` - Validity mask
    ///
    /// # Panics
    ///
    /// Panics if arrays have different lengths.
    pub fn new(
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
        valid: Vec<bool>,
    ) -> Self {
        let n_bars = open.len();
        assert_eq!(high.len(), n_bars, "high length mismatch");
        assert_eq!(low.len(), n_bars, "low length mismatch");
        assert_eq!(close.len(), n_bars, "close length mismatch");
        assert_eq!(volume.len(), n_bars, "volume length mismatch");
        assert_eq!(valid.len(), n_bars, "valid length mismatch");

        Self {
            open,
            high,
            low,
            close,
            volume,
            valid,
            n_bars,
        }
    }

    /// Create empty OHLCV data.
    #[inline]
    pub fn empty() -> Self {
        Self {
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
            valid: Vec::new(),
            n_bars: 0,
        }
    }

    /// Check if the data is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_bars == 0
    }

    /// Get the number of valid (non-NaN) bars.
    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|&&v| v).count()
    }
}

/// Cache key for indicator lookup.
///
/// This struct is hashable and can be used as a key in HashMap.
/// It uniquely identifies an indicator computation result.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CacheKey {
    /// Indicator name (e.g., "ema", "rsi", "atr")
    pub indicator: String,
    /// Symbol (e.g., "EUR/USD", "GBPUSD")
    pub symbol: String,
    /// Timeframe (e.g., "M1", "H1", "D1")
    pub timeframe: String,
    /// Price type ("BID" or "ASK")
    pub price_type: String,
    /// Serialized parameters (JSON-like string for multi-param indicators)
    pub params: String,
}

impl CacheKey {
    /// Create a new cache key.
    pub fn new(
        indicator: &str,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        params: &str,
    ) -> Self {
        Self {
            indicator: indicator.to_string(),
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            price_type: price_type.to_string(),
            params: params.to_string(),
        }
    }

    /// Create a cache key from a parameter slice.
    ///
    /// This is useful when parameters are computed as f64 values.
    pub fn from_params(
        indicator: &str,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        params: &[f64],
    ) -> Self {
        let params_str = params
            .iter()
            .map(|p| format!("{:.10}", p))
            .collect::<Vec<_>>()
            .join(",");
        Self::new(indicator, symbol, timeframe, price_type, &params_str)
    }

    /// Create cache key for single-period indicators (EMA, SMA, RSI, etc.).
    pub fn single_period(
        indicator: &str,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Self {
        Self::new(
            indicator,
            symbol,
            timeframe,
            price_type,
            &period.to_string(),
        )
    }

    /// Create cache key for MACD.
    pub fn macd(
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Self {
        Self::new(
            "macd",
            symbol,
            timeframe,
            price_type,
            &format!("{},{},{}", fast_period, slow_period, signal_period),
        )
    }

    /// Create cache key for Bollinger Bands.
    pub fn bollinger(
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        std_factor: f64,
        stepwise: bool,
    ) -> Self {
        let name = if stepwise { "bb_stepwise" } else { "bb" };
        Self::new(
            name,
            symbol,
            timeframe,
            price_type,
            &format!("{},{:.6}", period, std_factor),
        )
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.indicator.hash(state);
        self.symbol.hash(state);
        self.timeframe.hash(state);
        self.price_type.hash(state);
        self.params.hash(state);
    }
}

/// Result variants for indicator computations.
///
/// Different indicators return different shapes of data.
#[derive(Debug, Clone)]
pub enum IndicatorResult {
    /// Single series output (EMA, SMA, RSI, ATR, etc.)
    Single(Vec<f64>),
    /// Double series output (MACD: line + signal)
    Double(Vec<f64>, Vec<f64>),
    /// Triple series output (Bollinger: upper, mid, lower; DMI: +DI, -DI, ADX)
    Triple(Vec<f64>, Vec<f64>, Vec<f64>),
}

impl IndicatorResult {
    /// Get the length of the result arrays.
    pub fn len(&self) -> usize {
        match self {
            Self::Single(v) => v.len(),
            Self::Double(v, _) => v.len(),
            Self::Triple(v, _, _) => v.len(),
        }
    }

    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Extract as single series (panics if not Single variant).
    pub fn as_single(&self) -> &Vec<f64> {
        match self {
            Self::Single(v) => v,
            _ => panic!("Expected Single variant"),
        }
    }

    /// Extract as double series (panics if not Double variant).
    pub fn as_double(&self) -> (&Vec<f64>, &Vec<f64>) {
        match self {
            Self::Double(a, b) => (a, b),
            _ => panic!("Expected Double variant"),
        }
    }

    /// Extract as triple series (panics if not Triple variant).
    pub fn as_triple(&self) -> (&Vec<f64>, &Vec<f64>, &Vec<f64>) {
        match self {
            Self::Triple(a, b, c) => (a, b, c),
            _ => panic!("Expected Triple variant"),
        }
    }
}

/// Data key for OHLCV data lookup.
pub type OhlcvKey = (String, String, String); // (symbol, timeframe, price_type)

/// Type alias for the OHLCV data storage.
pub type OhlcvStorage = HashMap<OhlcvKey, OhlcvData>;

/// Type alias for the indicator cache storage.
pub type IndicatorStorage = HashMap<CacheKey, IndicatorResult>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_data_new() {
        let data = OhlcvData::new(
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![0.5, 1.5, 2.5],
            vec![1.2, 2.2, 3.2],
            vec![100.0, 200.0, 300.0],
            vec![true, true, false],
        );
        assert_eq!(data.n_bars, 3);
        assert_eq!(data.valid_count(), 2);
    }

    #[test]
    fn test_ohlcv_data_empty() {
        let data = OhlcvData::empty();
        assert!(data.is_empty());
        assert_eq!(data.n_bars, 0);
    }

    #[test]
    fn test_cache_key_hash() {
        use std::collections::hash_map::DefaultHasher;

        let key1 = CacheKey::single_period("ema", "EURUSD", "H1", "BID", 14);
        let key2 = CacheKey::single_period("ema", "EURUSD", "H1", "BID", 14);
        let key3 = CacheKey::single_period("ema", "EURUSD", "H1", "BID", 20);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        key1.hash(&mut hasher1);
        key2.hash(&mut hasher2);
        key3.hash(&mut hasher3);

        assert_eq!(hasher1.finish(), hasher2.finish());
        assert_ne!(hasher1.finish(), hasher3.finish());
    }

    #[test]
    fn test_indicator_result_variants() {
        let single = IndicatorResult::Single(vec![1.0, 2.0, 3.0]);
        assert_eq!(single.len(), 3);
        assert_eq!(single.as_single(), &vec![1.0, 2.0, 3.0]);

        let double = IndicatorResult::Double(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert_eq!(double.len(), 2);
        let (a, b) = double.as_double();
        assert_eq!(a, &vec![1.0, 2.0]);
        assert_eq!(b, &vec![3.0, 4.0]);

        let triple = IndicatorResult::Triple(vec![1.0], vec![2.0], vec![3.0]);
        assert_eq!(triple.len(), 1);
    }
}
