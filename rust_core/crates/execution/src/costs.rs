//! Execution costs loading from YAML configuration.
//!
//! Loads slippage and fee configurations from YAML files compatible
//! with the existing Omega Python stack.

use crate::error::ExecutionError;
use crate::fees::{FeeModel, FixedFee, NoFee, PercentageFee};
use crate::slippage::{FixedSlippage, NoSlippage, SlippageModel, VolatilitySlippage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Slippage configuration from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    /// Fixed slippage in pips
    #[serde(default)]
    pub fixed_pips: f64,
    /// Random/jitter slippage in pips (uniform distribution)
    #[serde(default)]
    pub random_pips: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            fixed_pips: 0.2,
            random_pips: 0.3,
        }
    }
}

/// Legacy fee configuration from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyFeeConfig {
    /// Fee per million notional
    #[serde(default)]
    pub per_million: f64,
    /// Standard lot size
    #[serde(default = "default_lot_size")]
    pub lot_size: f64,
    /// Minimum fee
    #[serde(default)]
    pub min_fee: f64,
}

fn default_lot_size() -> f64 {
    100_000.0
}

impl Default for LegacyFeeConfig {
    fn default() -> Self {
        Self {
            per_million: 25.0,
            lot_size: default_lot_size(),
            min_fee: 0.0,
        }
    }
}

/// Commission schema type.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommissionSchema {
    #[default]
    PerLot,
    PerMillionNotional,
    PercentOfNotional,
}

/// Commission side (when fee is applied).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommissionSide {
    Entry,
    Exit,
    #[default]
    Both,
}

/// Commission model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommissionConfig {
    /// Commission schema type
    #[serde(default)]
    pub schema: CommissionSchema,
    /// When fee is applied
    #[serde(default)]
    pub side: CommissionSide,
    /// Fee currency (optional)
    pub fee_ccy: Option<String>,
    /// Rate per million notional (for per_million_notional schema)
    #[serde(default)]
    pub rate_per_million: f64,
    /// Fee per lot (for per_lot schema)
    #[serde(default)]
    pub per_lot: f64,
    /// Percentage of notional (for percent_of_notional schema)
    #[serde(default)]
    pub pct: f64,
    /// Minimum fee per order/side
    #[serde(default)]
    pub min_fee: f64,
}

impl Default for CommissionConfig {
    fn default() -> Self {
        Self {
            schema: CommissionSchema::default(),
            side: CommissionSide::default(),
            fee_ccy: None,
            rate_per_million: 5.0,
            per_lot: 2.5,
            pct: 0.0,
            min_fee: 0.0,
        }
    }
}

/// Default costs configuration section.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefaultsConfig {
    /// Slippage configuration
    #[serde(default)]
    pub slippage: SlippageConfig,
    /// Legacy fees configuration
    #[serde(default)]
    pub fees: LegacyFeeConfig,
    /// Default commission schema
    #[serde(default)]
    pub schema: CommissionSchema,
    /// Default commission side
    #[serde(default)]
    pub side: CommissionSide,
    /// Default rate per million
    #[serde(default)]
    pub rate_per_million: f64,
    /// Default per lot fee
    #[serde(default)]
    pub per_lot: f64,
    /// Default percentage fee
    #[serde(default)]
    pub pct: f64,
    /// Default minimum fee
    #[serde(default)]
    pub min_fee: f64,
}

/// Complete execution costs configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionCostsConfig {
    /// Default configuration
    #[serde(default)]
    pub defaults: DefaultsConfig,
    /// Per-symbol overrides
    #[serde(default)]
    pub per_symbol: HashMap<String, CommissionConfig>,
}

impl ExecutionCostsConfig {
    /// Loads execution costs from a YAML file.
    pub fn load(path: &Path) -> Result<Self, ExecutionError> {
        let content = std::fs::read_to_string(path)?;
        let config: ExecutionCostsConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Loads execution costs from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, ExecutionError> {
        let config: ExecutionCostsConfig = serde_yaml::from_str(yaml)?;
        Ok(config)
    }

    /// Gets the commission config for a symbol, falling back to defaults.
    pub fn get_commission_config(&self, symbol: &str) -> CommissionConfig {
        if let Some(config) = self.per_symbol.get(symbol) {
            config.clone()
        } else {
            CommissionConfig {
                schema: self.defaults.schema.clone(),
                side: self.defaults.side.clone(),
                fee_ccy: None,
                rate_per_million: self.defaults.rate_per_million,
                per_lot: self.defaults.per_lot,
                pct: self.defaults.pct,
                min_fee: self.defaults.min_fee,
            }
        }
    }

    /// Creates a slippage model from the configuration.
    ///
    /// If both fixed_pips and random_pips are set, creates a VolatilitySlippage.
    /// If only fixed_pips is set, creates a FixedSlippage.
    /// Otherwise, creates a NoSlippage.
    pub fn create_slippage_model(&self, pip_size: f64) -> Box<dyn SlippageModel> {
        let config = &self.defaults.slippage;

        if config.random_pips > 0.0 {
            Box::new(VolatilitySlippage::new(
                config.fixed_pips,
                pip_size,
                config.random_pips / config.fixed_pips.max(0.1), // jitter factor
            ))
        } else if config.fixed_pips > 0.0 {
            Box::new(FixedSlippage::new(config.fixed_pips, pip_size))
        } else {
            Box::new(NoSlippage)
        }
    }

    /// Creates a fee model for a symbol from the configuration.
    pub fn create_fee_model(&self, symbol: &str, _lot_size: f64) -> Box<dyn FeeModel> {
        let config = self.get_commission_config(symbol);

        match config.schema {
            CommissionSchema::PerLot => Box::new(FixedFee::new(config.per_lot)),
            CommissionSchema::PerMillionNotional => {
                // Convert per-million rate to percentage
                let pct = config.rate_per_million / 1_000_000.0;
                Box::new(PercentageFee::new(pct))
            }
            CommissionSchema::PercentOfNotional => Box::new(PercentageFee::new(config.pct)),
        }
    }

    /// Checks if fees should be applied on entry for a symbol.
    pub fn apply_on_entry(&self, symbol: &str) -> bool {
        let config = self.get_commission_config(symbol);
        matches!(config.side, CommissionSide::Entry | CommissionSide::Both)
    }

    /// Checks if fees should be applied on exit for a symbol.
    pub fn apply_on_exit(&self, symbol: &str) -> bool {
        let config = self.get_commission_config(symbol);
        matches!(config.side, CommissionSide::Exit | CommissionSide::Both)
    }
}

/// Resolved costs for a specific symbol with concrete models.
pub struct SymbolCosts {
    /// Slippage model
    pub slippage: Box<dyn SlippageModel>,
    /// Fee model
    pub fee: Box<dyn FeeModel>,
    /// Whether to apply fee on entry
    pub apply_entry_fee: bool,
    /// Whether to apply fee on exit
    pub apply_exit_fee: bool,
    /// Pip size for the symbol
    pub pip_size: f64,
    /// Pip buffer factor for SL/TP checks
    pub pip_buffer_factor: f64,
}

impl SymbolCosts {
    /// Creates symbol costs from config.
    pub fn from_config(config: &ExecutionCostsConfig, symbol: &str, pip_size: f64) -> Self {
        Self {
            slippage: config.create_slippage_model(pip_size),
            fee: config.create_fee_model(symbol, 100_000.0),
            apply_entry_fee: config.apply_on_entry(symbol),
            apply_exit_fee: config.apply_on_exit(symbol),
            pip_size,
            pip_buffer_factor: 0.5,
        }
    }

    /// Creates symbol costs with no fees or slippage (for testing).
    pub fn zero_cost(pip_size: f64) -> Self {
        Self {
            slippage: Box::new(NoSlippage),
            fee: Box::new(NoFee),
            apply_entry_fee: false,
            apply_exit_fee: false,
            pip_size,
            pip_buffer_factor: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_YAML: &str = r#"
defaults:
  slippage:
    fixed_pips: 0.20
    random_pips: 0.30
  fees:
    per_million: 25.0
    lot_size: 100000.0
    min_fee: 0.0
  schema: per_lot
  side: both
  rate_per_million: 5.0
  per_lot: 2.5
  pct: 0.0
  min_fee: 0.0

per_symbol:
  XAUUSD:
    schema: per_lot
    per_lot: 3.50
    fee_ccy: USD
    side: both
    min_fee: 0.00
  USDJPY:
    schema: per_million_notional
    fee_ccy: USD
    rate_per_million: 6.0
    side: both
    min_fee: 1.0
"#;

    #[test]
    fn test_load_yaml() {
        let config = ExecutionCostsConfig::from_yaml(TEST_YAML).unwrap();

        // Check defaults
        assert!((config.defaults.slippage.fixed_pips - 0.20).abs() < 0.001);
        assert!((config.defaults.slippage.random_pips - 0.30).abs() < 0.001);
        assert_eq!(config.defaults.schema, CommissionSchema::PerLot);
        assert!((config.defaults.per_lot - 2.5).abs() < 0.001);

        // Check per-symbol overrides
        assert!(config.per_symbol.contains_key("XAUUSD"));
        assert!(config.per_symbol.contains_key("USDJPY"));
    }

    #[test]
    fn test_get_commission_config_default() {
        let config = ExecutionCostsConfig::from_yaml(TEST_YAML).unwrap();
        let comm = config.get_commission_config("EURUSD");

        assert_eq!(comm.schema, CommissionSchema::PerLot);
        assert!((comm.per_lot - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_get_commission_config_override() {
        let config = ExecutionCostsConfig::from_yaml(TEST_YAML).unwrap();
        let comm = config.get_commission_config("XAUUSD");

        assert_eq!(comm.schema, CommissionSchema::PerLot);
        assert!((comm.per_lot - 3.50).abs() < 0.001);
    }

    #[test]
    fn test_create_slippage_model() {
        let config = ExecutionCostsConfig::from_yaml(TEST_YAML).unwrap();
        let model = config.create_slippage_model(0.0001);

        assert_eq!(model.name(), "VolatilitySlippage");
    }

    #[test]
    fn test_fee_side() {
        let config = ExecutionCostsConfig::from_yaml(TEST_YAML).unwrap();

        assert!(config.apply_on_entry("EURUSD"));
        assert!(config.apply_on_exit("EURUSD"));
    }

    #[test]
    fn test_symbol_costs() {
        let config = ExecutionCostsConfig::from_yaml(TEST_YAML).unwrap();
        let costs = SymbolCosts::from_config(&config, "EURUSD", 0.0001);

        assert_eq!(costs.pip_size, 0.0001);
        assert!(costs.apply_entry_fee);
        assert!(costs.apply_exit_fee);
    }

    #[test]
    fn test_zero_cost() {
        let costs = SymbolCosts::zero_cost(0.0001);

        assert_eq!(costs.slippage.name(), "NoSlippage");
        assert_eq!(costs.fee.name(), "NoFee");
        assert!(!costs.apply_entry_fee);
        assert!(!costs.apply_exit_fee);
    }
}
