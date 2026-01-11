//! Symbol specification for position sizing and cost calculations.
//!
//! Provides the [`SymbolSpec`] struct containing broker-specific parameters
//! needed for accurate position sizing, volume quantization, and value calculations.

use crate::error::{OmegaError, Result};
use serde::{Deserialize, Serialize};

/// Default contract size for Forex (100,000 units).
pub const DEFAULT_LOT_SIZE: f64 = 100_000.0;

/// Default volume step (0.01 lots).
pub const DEFAULT_VOLUME_STEP: f64 = 0.01;

/// Default minimum volume (0.01 lots).
pub const DEFAULT_VOLUME_MIN: f64 = 0.01;

/// Default pip buffer factor for SL/TP detection.
pub const DEFAULT_PIP_BUFFER_FACTOR: f64 = 0.5;

/// Symbol specifications for trading.
///
/// Contains all broker-specific parameters needed for accurate position
/// sizing, volume quantization, and pip value calculations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolSpec {
    /// Symbol name (e.g., "EURUSD")
    pub symbol: String,

    /// Pip size (e.g., 0.0001 for most Forex pairs, 0.01 for JPY pairs)
    pub pip_size: f64,

    /// Contract size (units per lot, e.g., 100000 for Forex)
    pub contract_size: f64,

    /// Tick size (minimum price increment)
    pub tick_size: Option<f64>,

    /// Tick value (value of one tick movement in account currency)
    pub tick_value: Option<f64>,

    /// Minimum volume (lots)
    pub volume_min: f64,

    /// Maximum volume (lots)
    pub volume_max: f64,

    /// Volume step (lot increment)
    pub volume_step: f64,

    /// Quote currency (e.g., "USD" for EURUSD)
    pub quote_currency: Option<String>,

    /// Base currency (e.g., "EUR" for EURUSD)
    pub base_currency: Option<String>,
}

impl Default for SymbolSpec {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            pip_size: 0.0001,
            contract_size: DEFAULT_LOT_SIZE,
            tick_size: None,
            tick_value: None,
            volume_min: DEFAULT_VOLUME_MIN,
            volume_max: 100.0,
            volume_step: DEFAULT_VOLUME_STEP,
            quote_currency: None,
            base_currency: None,
        }
    }
}

impl SymbolSpec {
    /// Create a new symbol spec with the given symbol name.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            ..Default::default()
        }
    }

    /// Create a Forex symbol spec with standard parameters.
    pub fn forex(symbol: impl Into<String>, pip_size: f64) -> Self {
        Self {
            symbol: symbol.into(),
            pip_size,
            contract_size: DEFAULT_LOT_SIZE,
            tick_size: Some(pip_size / 10.0), // 5-digit pricing
            tick_value: None,
            volume_min: DEFAULT_VOLUME_MIN,
            volume_max: 100.0,
            volume_step: DEFAULT_VOLUME_STEP,
            quote_currency: None,
            base_currency: None,
        }
    }

    /// Calculate unit value per price movement.
    ///
    /// This is the monetary value (in account currency) of a 1.0 price
    /// movement for 1 lot.
    ///
    /// If `tick_value` and `tick_size` are available, uses:
    /// `tick_value / tick_size`
    ///
    /// Otherwise, falls back to `contract_size`.
    pub fn unit_value_per_price(&self) -> f64 {
        if let (Some(tick_val), Some(tick_sz)) = (self.tick_value, self.tick_size) {
            if tick_sz > 0.0 {
                return tick_val / tick_sz;
            }
        }
        // Fallback: assume account currency = quote currency
        self.contract_size
    }

    /// Convert price distance to pip distance.
    #[inline]
    pub fn price_to_pips(&self, price_distance: f64) -> f64 {
        if self.pip_size > 0.0 {
            price_distance / self.pip_size
        } else {
            0.0
        }
    }

    /// Convert pip distance to price distance.
    #[inline]
    pub fn pips_to_price(&self, pips: f64) -> f64 {
        pips * self.pip_size
    }

    /// Quantize volume to broker-compliant step size.
    ///
    /// Uses conservative floor rounding to ensure risk is not exceeded.
    /// Applies the formula: `floor((raw - min) / step) * step + min`
    pub fn quantize_volume(&self, raw_lots: f64) -> f64 {
        if raw_lots <= self.volume_min {
            return self.volume_min;
        }

        // Conservative floor rounding with epsilon guard
        let n_steps = ((raw_lots - self.volume_min + 1e-12) / self.volume_step).floor();
        let quantized = self.volume_min + n_steps * self.volume_step;

        // Clamp to max
        let clamped = quantized.min(self.volume_max);

        // Format to 8 decimal places for numerical stability
        // (matching Python's f"{lots:.8f}" behavior)
        (clamped * 1e8).round() / 1e8
    }

    /// Calculate position size based on risk.
    ///
    /// # Arguments
    /// * `risk_amount` - Risk amount in account currency
    /// * `sl_distance` - Stop-loss distance in price units
    ///
    /// # Returns
    /// Quantized position size in lots
    pub fn size_for_risk(&self, risk_amount: f64, sl_distance: f64) -> Result<f64> {
        if sl_distance <= 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!("SL distance must be positive, got {sl_distance}"),
            });
        }

        let unit_value = self.unit_value_per_price();
        let risk_per_lot = sl_distance * unit_value;

        if risk_per_lot <= 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: "Risk per lot is zero or negative".to_string(),
            });
        }

        let raw_lots = risk_amount / risk_per_lot;
        Ok(self.quantize_volume(raw_lots))
    }
}

/// Cache for symbol specifications, keyed by symbol name.
#[derive(Clone, Debug, Default)]
pub struct SymbolSpecCache {
    specs: std::collections::HashMap<String, SymbolSpec>,
    /// Cached unit values per price for quick lookup
    unit_values: std::collections::HashMap<String, f64>,
}

impl SymbolSpecCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol spec to the cache.
    pub fn insert(&mut self, spec: SymbolSpec) {
        let unit_value = spec.unit_value_per_price();
        self.unit_values.insert(spec.symbol.clone(), unit_value);
        self.specs.insert(spec.symbol.clone(), spec);
    }

    /// Get a symbol spec by name.
    pub fn get(&self, symbol: &str) -> Option<&SymbolSpec> {
        self.specs.get(symbol)
    }

    /// Get cached unit value for a symbol.
    pub fn get_unit_value(&self, symbol: &str) -> Option<f64> {
        self.unit_values.get(symbol).copied()
    }

    /// Get pip size for a symbol, with default fallback.
    pub fn pip_size(&self, symbol: &str) -> f64 {
        self.specs
            .get(symbol)
            .map(|s| s.pip_size)
            .unwrap_or(0.0001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_symbol_spec() {
        let spec = SymbolSpec::default();
        assert!((spec.pip_size - 0.0001).abs() < 1e-10);
        assert!((spec.contract_size - 100_000.0).abs() < 1e-10);
        assert!((spec.volume_min - 0.01).abs() < 1e-10);
        assert!((spec.volume_step - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_forex_spec() {
        let spec = SymbolSpec::forex("EURUSD", 0.0001);
        assert_eq!(spec.symbol, "EURUSD");
        assert!((spec.pip_size - 0.0001).abs() < 1e-10);
        assert!((spec.tick_size.unwrap() - 0.00001).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_volume() {
        let spec = SymbolSpec {
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
            ..Default::default()
        };

        // Exact value
        assert!((spec.quantize_volume(0.10) - 0.10).abs() < 1e-10);

        // Round down
        assert!((spec.quantize_volume(0.105) - 0.10).abs() < 1e-10);
        assert!((spec.quantize_volume(0.109) - 0.10).abs() < 1e-10);

        // Minimum
        assert!((spec.quantize_volume(0.005) - 0.01).abs() < 1e-10);

        // Maximum
        assert!((spec.quantize_volume(150.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_size_for_risk() {
        let spec = SymbolSpec {
            pip_size: 0.0001,
            contract_size: 100_000.0,
            tick_size: Some(0.00001),
            tick_value: Some(1.0), // $1 per tick
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
            ..Default::default()
        };

        // $100 risk with 50 pip SL (0.0050 price distance)
        // unit_value = 1.0 / 0.00001 = 100,000 per price unit
        // risk_per_lot = 0.0050 * 100,000 = 500
        // raw_lots = 100 / 500 = 0.2
        let size = spec.size_for_risk(100.0, 0.0050).unwrap();
        assert!((size - 0.20).abs() < 1e-10);
    }

    #[test]
    fn test_price_to_pips() {
        let spec = SymbolSpec::forex("EURUSD", 0.0001);
        assert!((spec.price_to_pips(0.0050) - 50.0).abs() < 1e-10);
        assert!((spec.price_to_pips(0.0001) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symbol_spec_cache() {
        let mut cache = SymbolSpecCache::new();
        cache.insert(SymbolSpec::forex("EURUSD", 0.0001));
        cache.insert(SymbolSpec::forex("USDJPY", 0.01));

        assert!(cache.get("EURUSD").is_some());
        assert!((cache.pip_size("EURUSD") - 0.0001).abs() < 1e-10);
        assert!((cache.pip_size("USDJPY") - 0.01).abs() < 1e-10);
        assert!((cache.pip_size("UNKNOWN") - 0.0001).abs() < 1e-10); // default
    }
}
