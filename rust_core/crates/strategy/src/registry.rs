//! Strategy registry for dynamic strategy creation
//!
//! Allows strategies to be registered and created by name at runtime.

use crate::error::StrategyError;
use crate::traits::Strategy;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory function type for creating strategies from JSON params.
pub type StrategyFactory =
    Box<dyn Fn(&serde_json::Value) -> Result<Box<dyn Strategy>, StrategyError> + Send + Sync>;

/// Registry for strategy factories.
///
/// Allows strategies to be registered by name and instantiated
/// dynamically from configuration.
///
/// # Example
/// ```ignore
/// let mut registry = StrategyRegistry::new();
/// registry.register("mean_reversion_z_score", |params| {
///     let strategy = MeanReversionZScore::from_params(params)?;
///     Ok(Box::new(strategy))
/// });
///
/// let strategy = registry.create("mean_reversion_z_score", &params)?;
/// ```
pub struct StrategyRegistry {
    factories: HashMap<String, Arc<StrategyFactory>>,
}

impl StrategyRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Creates a registry with default strategies pre-registered.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register_defaults();
        registry
    }

    /// Registers a strategy factory.
    ///
    /// # Arguments
    /// * `name` - Strategy name (case-insensitive)
    /// * `factory` - Factory function that creates the strategy
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn(&serde_json::Value) -> Result<Box<dyn Strategy>, StrategyError>
            + Send
            + Sync
            + 'static,
    {
        self.factories
            .insert(name.into().to_lowercase(), Arc::new(Box::new(factory)));
    }

    /// Creates a strategy by name.
    ///
    /// # Arguments
    /// * `name` - Strategy name (case-insensitive)
    /// * `params` - JSON parameters for the strategy
    ///
    /// # Errors
    /// Returns `StrategyError::UnknownStrategy` if the name is not registered.
    pub fn create(
        &self,
        name: &str,
        params: &serde_json::Value,
    ) -> Result<Box<dyn Strategy>, StrategyError> {
        let factory = self
            .factories
            .get(&name.to_lowercase())
            .ok_or_else(|| StrategyError::UnknownStrategy(name.to_string()))?;

        factory(params)
    }

    /// Checks if a strategy is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(&name.to_lowercase())
    }

    /// Returns all registered strategy names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.factories.keys()
    }

    /// Returns the number of registered strategies.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Checks if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Registers default strategies.
    fn register_defaults(&mut self) {
        use crate::impl_::mean_reversion_z_score::MeanReversionZScore;

        self.register("mean_reversion_z_score", |params| {
            let strategy = MeanReversionZScore::from_params(params)?;
            Ok(Box::new(strategy))
        });
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::BarContext;
    use crate::traits::IndicatorRequirement;
    use omega_types::Signal;

    /// Dummy strategy for testing
    struct DummyStrategy {
        name: String,
    }

    impl Strategy for DummyStrategy {
        fn on_bar(&mut self, _ctx: &BarContext) -> Option<Signal> {
            None
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn required_indicators(&self) -> Vec<IndicatorRequirement> {
            Vec::new()
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = StrategyRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_register_and_create() {
        let mut registry = StrategyRegistry::new();

        registry.register("dummy", |params| {
            let name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();
            Ok(Box::new(DummyStrategy { name }))
        });

        assert!(registry.contains("dummy"));
        assert!(registry.contains("DUMMY")); // Case insensitive

        let params = serde_json::json!({"name": "test"});
        let strategy = registry.create("dummy", &params).unwrap();
        assert_eq!(strategy.name(), "test");
    }

    #[test]
    fn test_registry_unknown_strategy() {
        let registry = StrategyRegistry::new();
        let params = serde_json::json!({});

        let result = registry.create("nonexistent", &params);
        assert!(matches!(result, Err(StrategyError::UnknownStrategy(_))));
    }

    #[test]
    fn test_registry_names() {
        let mut registry = StrategyRegistry::new();
        registry.register("strategy_a", |_| {
            Ok(Box::new(DummyStrategy {
                name: "a".to_string(),
            }))
        });
        registry.register("strategy_b", |_| {
            Ok(Box::new(DummyStrategy {
                name: "b".to_string(),
            }))
        });

        let names: Vec<_> = registry.names().collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&&"strategy_a".to_string()));
        assert!(names.contains(&&"strategy_b".to_string()));
    }
}
