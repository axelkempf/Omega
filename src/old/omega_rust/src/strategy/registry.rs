//! Strategy Registry
//!
//! Zentrale Registry für alle verfügbaren Rust-Strategien.
//! Ermöglicht das dynamische Laden von Strategien nach Name.

use std::collections::HashMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use super::traits::{RustStrategy, StrategyError, StrategyFactory};
use super::types::StrategyConfig;
use super::mean_reversion_zscore::MeanReversionZScoreFactory;

/// Globale Strategie-Registry
static STRATEGY_REGISTRY: Lazy<RwLock<StrategyRegistry>> = Lazy::new(|| {
    let mut registry = StrategyRegistry::new();
    
    // Registriere Built-in Strategien
    registry.register_factory(Box::new(MeanReversionZScoreFactory));
    
    RwLock::new(registry)
});

/// Registry für Strategie-Factories
pub struct StrategyRegistry {
    factories: HashMap<String, Box<dyn StrategyFactory>>,
}

impl StrategyRegistry {
    /// Erstellt eine neue, leere Registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Registriert eine neue Strategie-Factory.
    pub fn register_factory(&mut self, factory: Box<dyn StrategyFactory>) {
        let name = factory.name().to_string();
        self.factories.insert(name, factory);
    }

    /// Erstellt eine Strategie-Instanz nach Name und Config.
    pub fn create(
        &self,
        name: &str,
        config: &StrategyConfig,
    ) -> Result<Box<dyn RustStrategy>, StrategyError> {
        let factory = self.factories.get(name).ok_or_else(|| {
            StrategyError::ConfigError(format!("Strategy '{}' not found in registry", name))
        })?;

        factory.create(config)
    }

    /// Prüft ob eine Strategie registriert ist.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Gibt alle registrierten Strategie-Namen zurück.
    pub fn list_strategies(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Registriert eine Strategie-Factory in der globalen Registry.
///
/// # Example
///
/// ```rust,ignore
/// register_strategy(Box::new(MyStrategyFactory));
/// ```
pub fn register_strategy(factory: Box<dyn StrategyFactory>) {
    let mut registry = STRATEGY_REGISTRY
        .write()
        .expect("Failed to acquire write lock on strategy registry");
    registry.register_factory(factory);
}

/// Erstellt eine Strategie-Instanz aus der globalen Registry.
///
/// # Example
///
/// ```rust,ignore
/// let config = StrategyConfig::new("EURUSD", Timeframe::M5, 10000.0, 0.01);
/// let strategy = create_strategy("mean_reversion_z_score", &config)?;
/// ```
pub fn create_strategy(
    name: &str,
    config: &StrategyConfig,
) -> Result<Box<dyn RustStrategy>, StrategyError> {
    let registry = STRATEGY_REGISTRY
        .read()
        .expect("Failed to acquire read lock on strategy registry");
    registry.create(name, config)
}

/// Prüft ob eine Strategie in der globalen Registry existiert.
pub fn strategy_exists(name: &str) -> bool {
    let registry = STRATEGY_REGISTRY
        .read()
        .expect("Failed to acquire read lock on strategy registry");
    registry.contains(name)
}

/// Listet alle registrierten Strategien.
pub fn list_strategies() -> Vec<String> {
    let registry = STRATEGY_REGISTRY
        .read()
        .expect("Failed to acquire read lock on strategy registry");
    registry.list_strategies().iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::IndicatorCache;
    use super::super::types::{DataSlice, Position, PositionAction, Timeframe, TradeSignal};
    use super::super::traits::RustStrategy;

    struct TestStrategy {
        name: String,
    }

    impl RustStrategy for TestStrategy {
        fn evaluate(&self, _slice: &DataSlice, _cache: &mut IndicatorCache) -> Option<TradeSignal> {
            None
        }

        fn manage_position(&self, _position: &Position, _slice: &DataSlice) -> PositionAction {
            PositionAction::Hold()
        }

        fn primary_timeframe(&self) -> Timeframe {
            Timeframe::M5
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    struct TestStrategyFactory;

    impl StrategyFactory for TestStrategyFactory {
        fn create(&self, config: &StrategyConfig) -> Result<Box<dyn RustStrategy>, StrategyError> {
            Ok(Box::new(TestStrategy {
                name: config.symbol.clone(),
            }))
        }

        fn name(&self) -> &str {
            "test_strategy"
        }
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = StrategyRegistry::new();
        
        assert!(!registry.contains("test_strategy"));
        assert!(registry.list_strategies().is_empty());
        
        registry.register_factory(Box::new(TestStrategyFactory));
        
        assert!(registry.contains("test_strategy"));
        assert_eq!(registry.list_strategies().len(), 1);
    }

    #[test]
    fn test_registry_create() {
        let mut registry = StrategyRegistry::new();
        registry.register_factory(Box::new(TestStrategyFactory));
        
        let config = StrategyConfig {
            symbol: "EURUSD".to_string(),
            primary_timeframe: "M5".to_string(),
            initial_capital: 10000.0,
            risk_per_trade: 0.01,
            params: HashMap::new(),
        };
        
        let strategy = registry.create("test_strategy", &config).unwrap();
        assert_eq!(strategy.name(), "EURUSD");
        assert_eq!(strategy.primary_timeframe(), Timeframe::M5);
    }

    #[test]
    fn test_registry_not_found() {
        let registry = StrategyRegistry::new();
        let config = StrategyConfig {
            symbol: "EURUSD".to_string(),
            primary_timeframe: "M5".to_string(),
            initial_capital: 10000.0,
            risk_per_trade: 0.01,
            params: HashMap::new(),
        };
        
        let result = registry.create("nonexistent", &config);
        assert!(result.is_err());
    }
}
