//! Symbol specification loading and validation.
//!
//! Provides volume constraints, pip sizing, and minimum SL distance
//! sourced from `configs/symbol_specs.yaml`.

use crate::error::ExecutionError;
use serde::Deserialize;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::path::Path;

/// Default minimum SL distance in pips when not specified per symbol.
pub const DEFAULT_MIN_SL_DISTANCE_PIPS: f64 = 0.1;

/// Default minimum volume when symbol specs are missing.
const DEFAULT_VOLUME_MIN: f64 = 0.01;
/// Default volume step when symbol specs are missing.
const DEFAULT_VOLUME_STEP: f64 = 0.01;
/// Default maximum volume when symbol specs are missing.
const DEFAULT_VOLUME_MAX: f64 = 100.0;

/// Symbol specification loaded from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct SymbolSpec {
    /// Contract size (optional)
    #[serde(default)]
    pub contract_size: Option<f64>,
    /// Tick size (optional)
    #[serde(default)]
    pub tick_size: Option<f64>,
    /// Tick value (optional)
    #[serde(default)]
    pub tick_value: Option<f64>,
    /// Pip size for the symbol
    #[serde(default)]
    pub pip_size: f64,
    /// Minimum volume
    #[serde(default = "default_volume_min")]
    pub volume_min: f64,
    /// Volume step
    #[serde(default = "default_volume_step")]
    pub volume_step: f64,
    /// Maximum volume
    #[serde(default = "default_volume_max")]
    pub volume_max: f64,
    /// Optional per-symbol minimum SL distance in pips
    #[serde(default)]
    pub min_sl_distance_pips: Option<f64>,
    /// Base currency (optional)
    #[serde(default)]
    pub base_currency: Option<String>,
    /// Quote currency (optional)
    #[serde(default)]
    pub quote_currency: Option<String>,
    /// Profit currency (optional)
    #[serde(default)]
    pub profit_currency: Option<String>,
    /// Explicit symbol name (optional)
    #[serde(default)]
    pub symbol: Option<String>,
}

impl SymbolSpec {
    /// Returns the minimum SL distance in pips, with fallback default.
    #[must_use]
    pub fn min_sl_distance_pips(&self) -> f64 {
        self.min_sl_distance_pips
            .unwrap_or(DEFAULT_MIN_SL_DISTANCE_PIPS)
    }

    /// Returns volume bounds and step, applying safe defaults.
    #[must_use]
    pub fn volume_limits(&self) -> (f64, f64, f64) {
        (self.volume_min, self.volume_step, self.volume_max)
    }

    /// Returns pip size, falling back to default if invalid.
    #[must_use]
    pub fn resolved_pip_size(&self) -> f64 {
        if self.pip_size > 0.0 {
            self.pip_size
        } else {
            0.0001
        }
    }
}

fn default_volume_min() -> f64 {
    DEFAULT_VOLUME_MIN
}

fn default_volume_step() -> f64 {
    DEFAULT_VOLUME_STEP
}

fn default_volume_max() -> f64 {
    DEFAULT_VOLUME_MAX
}

/// Loads symbol specifications from a YAML file.
///
/// The YAML is expected to be a map keyed by symbol name.
///
/// # Errors
/// Returns an error if the file cannot be read or YAML parsing fails.
pub fn load_symbol_specs(path: &Path) -> Result<HashMap<String, SymbolSpec>, ExecutionError> {
    let content = std::fs::read_to_string(path)?;
    let specs: HashMap<String, SymbolSpec> = serde_yaml::from_str(&content)?;
    Ok(specs)
}

/// Returns the symbol spec for a given symbol or a defaulted spec.
///
/// This is a safe fallback when a symbol is missing from the YAML.
#[must_use]
pub fn get_symbol_spec_or_default<S: BuildHasher>(
    specs: &HashMap<String, SymbolSpec, S>,
    symbol: &str,
) -> SymbolSpec {
    specs.get(symbol).cloned().unwrap_or(SymbolSpec {
        contract_size: None,
        tick_size: None,
        tick_value: None,
        pip_size: 0.0001,
        volume_min: DEFAULT_VOLUME_MIN,
        volume_step: DEFAULT_VOLUME_STEP,
        volume_max: DEFAULT_VOLUME_MAX,
        min_sl_distance_pips: Some(DEFAULT_MIN_SL_DISTANCE_PIPS),
        base_currency: None,
        quote_currency: None,
        profit_currency: None,
        symbol: Some(symbol.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::path::PathBuf;

    #[test]
    fn test_load_symbol_specs() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let specs_path = manifest_dir.join("../../../configs/symbol_specs.yaml");
        let specs = load_symbol_specs(&specs_path).unwrap();

        let eurusd = specs.get("EURUSD").unwrap();
        assert!(eurusd.pip_size > 0.0);
        assert!(eurusd.volume_min > 0.0);
        assert!(eurusd.volume_step > 0.0);
        assert!(eurusd.volume_max >= eurusd.volume_min);
    }

    #[test]
    fn test_min_sl_distance_default() {
        let spec = SymbolSpec {
            contract_size: None,
            tick_size: None,
            tick_value: None,
            pip_size: 0.0001,
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
            min_sl_distance_pips: None,
            base_currency: None,
            quote_currency: None,
            profit_currency: None,
            symbol: None,
        };

        assert_relative_eq!(
            spec.min_sl_distance_pips(),
            DEFAULT_MIN_SL_DISTANCE_PIPS,
            epsilon = 1e-12
        );
    }
}
