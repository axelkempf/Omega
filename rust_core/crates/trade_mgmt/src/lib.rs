//! Omega Trade Management
//!
//! Trade management layer for the Omega V2 trading system.
//! Provides rules for managing open positions (stop loss updates, timeouts, etc.).
//!
//! # Features
//! - Rule trait for implementing trade management rules
//! - TradeManager for evaluating rules against positions
//! - Built-in rules: MaxHoldingTime (MVP), BreakEven/Trailing (Post-MVP)
//! - Read-only context types (PositionView, MarketView, TradeContext)
//!
//! # Example
//! ```ignore
//! use omega_trade_mgmt::{
//!     TradeManager, TradeManagerBuilder, MaxHoldingTimeRule,
//!     TradeContext, MarketView, PositionView,
//! };
//!
//! let manager = TradeManagerBuilder::new()
//!     .with_rule(MaxHoldingTimeRule::new(100, bar_duration_ns))
//!     .build();
//!
//! // In backtest loop:
//! let market = MarketView::from_close(timestamp_ns, bid_close, ask_close);
//! let ctx = TradeContext::new(bar_idx, market, bar_duration_ns);
//! let actions = manager.evaluate(&ctx, &positions);
//!
//! for action in actions {
//!     match action {
//!         Action::ClosePosition { position_id, reason, exit_price_hint, .. } => {
//!             // Close the position at exit_price_hint...
//!         }
//!         Action::ModifyStopLoss { position_id, new_stop_loss, effective_from_idx, .. } => {
//!             // Update stop loss (applies from effective_from_idx)...
//!         }
//!         _ => {}
//!     }
//! }
//! ```

#![deny(clippy::all)]

pub mod actions;
pub mod context;
pub mod engine;
pub mod error;
pub mod rules;

// Re-export main types
pub use actions::{Action, StopModifyReason};
pub use context::{MarketView, PositionView, TradeContext, TradeMgmtMode};
pub use engine::{TradeManager, TradeManagerBuilder};
pub use error::TradeManagementError;
pub use rules::{BreakEvenRule, MaxHoldingTimeRule, Rule, RuleSet, TrailingStopRule};
