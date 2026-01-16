//! Strategy trait and related types
//!
//! Defines the core Strategy trait and IndicatorRequirement for
//! declaring indicator dependencies.

use crate::context::BarContext;
use omega_types::Signal;
use serde::{Deserialize, Serialize};

/// Trait for trading strategies.
///
/// Strategies process bar data and generate trading signals.
/// They must declare their indicator requirements upfront for
/// pre-computation and caching.
///
/// # Thread Safety
/// Strategies must be `Send + Sync` to allow parallel backtest execution.
///
/// # Example
/// ```ignore
/// impl Strategy for MyStrategy {
///     fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal> {
///         let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": 20}))?;
///         if zscore <= -2.0 {
///             Some(Signal { /* ... */ })
///         } else {
///             None
///         }
///     }
///
///     fn name(&self) -> &str { "my_strategy" }
///
///     fn required_indicators(&self) -> Vec<IndicatorRequirement> {
///         vec![IndicatorRequirement {
///             name: "Z_SCORE".to_string(),
///             timeframe: None,
///             params: json!({"window": 20}),
///         }]
///     }
/// }
/// ```
pub trait Strategy: Send + Sync {
    /// Processes a bar and optionally returns a signal.
    ///
    /// Called for each bar in the backtest sequence (after warmup).
    /// Returns `None` if no trade signal should be generated.
    fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal>;

    /// Name of the strategy for registry lookup.
    fn name(&self) -> &str;

    /// Indicators required by this strategy.
    ///
    /// Used for pre-computation before the backtest loop.
    fn required_indicators(&self) -> Vec<IndicatorRequirement>;

    /// HTF timeframes required by this strategy.
    ///
    /// Returns an empty vector if no HTF data is needed.
    fn required_htf_timeframes(&self) -> Vec<String> {
        Vec::new()
    }

    /// Resets the strategy state for a new backtest run.
    ///
    /// Called before starting a new backtest to clear any internal state.
    fn reset(&mut self) {}
}

/// Requirement specification for an indicator.
///
/// Declares what indicators a strategy needs, allowing the backtest
/// engine to pre-compute them before the main loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorRequirement {
    /// Indicator name (e.g., "EMA", "ATR", "Z_SCORE")
    pub name: String,
    /// Timeframe for this indicator (None = primary timeframe)
    #[serde(default)]
    pub timeframe: Option<String>,
    /// Parameters as JSON value
    #[serde(default)]
    pub params: serde_json::Value,
}

impl IndicatorRequirement {
    /// Creates a new indicator requirement.
    pub fn new(name: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            timeframe: None,
            params,
        }
    }

    /// Creates a new indicator requirement with a specific timeframe.
    pub fn with_timeframe(
        name: impl Into<String>,
        timeframe: impl Into<String>,
        params: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            timeframe: Some(timeframe.into()),
            params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator_requirement_new() {
        let req = IndicatorRequirement::new("EMA", serde_json::json!({"period": 20}));
        assert_eq!(req.name, "EMA");
        assert!(req.timeframe.is_none());
        assert_eq!(req.params["period"], 20);
    }

    #[test]
    fn test_indicator_requirement_with_timeframe() {
        let req =
            IndicatorRequirement::with_timeframe("EMA", "H4", serde_json::json!({"period": 200}));
        assert_eq!(req.name, "EMA");
        assert_eq!(req.timeframe, Some("H4".to_string()));
        assert_eq!(req.params["period"], 200);
    }

    #[test]
    fn test_indicator_requirement_serde() {
        let req = IndicatorRequirement::new("ATR", serde_json::json!({"period": 14}));
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: IndicatorRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(req.name, deserialized.name);
    }
}
