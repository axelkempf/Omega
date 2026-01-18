//! Omega Strategy
//!
//! Strategy layer for the Omega V2 trading system.
//! Provides the Strategy trait, `BarContext`, and strategy implementations.
//!
//! # Features
//! - Strategy trait for implementing trading strategies
//! - `BarContext` for accessing market data and indicators
//! - `StrategyRegistry` for dynamic strategy creation
//! - Mean Reversion Z-Score strategy with 6 scenarios
//!
//! # Example
//! ```ignore
//! use omega_strategy::{Strategy, BarContext, StrategyRegistry};
//!
//! let mut registry = StrategyRegistry::with_defaults();
//! let params = serde_json::json!({
//!     "ema_length": 20,
//!     "z_score_long": -2.0,
//!     "z_score_short": 2.0,
//!     "enabled_scenarios": [1, 2]
//! });
//!
//! let mut strategy = registry.create("mean_reversion_z_score", &params)?;
//!
//! // In backtest loop:
//! if let Some(signal) = strategy.on_bar(&ctx) {
//!     // Process signal...
//! }
//! ```

#![deny(clippy::all)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]

pub mod context;
pub mod error;
pub mod impl_;
pub mod registry;
pub mod runner;
pub mod traits;

// Re-export main types
pub use context::{BarContext, HtfContext};
pub use error::StrategyError;
pub use registry::StrategyRegistry;
pub use runner::{BarContextBuilder, StrategyRunner};
pub use traits::{IndicatorRequirement, Strategy};

// Re-export strategy implementations
pub use impl_::mean_reversion_z_score::{
    DirectionFilter, HtfFilter, MeanReversionZScore, MrzParams, Scenario6Mode,
};
