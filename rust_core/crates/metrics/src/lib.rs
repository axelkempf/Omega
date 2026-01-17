//! Omega Metrics
//!
//! Computes backtest metrics and outputs metric definitions for downstream consumers.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

use serde_json as _;

/// Metric computation entrypoints.
pub mod compute;
/// Metric definition catalog for output contract.
pub mod definitions;
/// Equity-curve metric helpers.
pub mod equity_metrics;
/// Output formatting helpers.
pub mod output;
/// Trade-based metric helpers.
pub mod trade_metrics;

pub use compute::compute_metrics;
pub use definitions::{MetricDefinition, MetricDefinitions};
pub use output::MetricsOutput;
