//! # Omega Portfolio
//!
//! Portfolio management, position tracking, and equity calculation for the Omega V2 trading system.
//!
//! ## Overview
//!
//! This crate provides:
//!
//! - **Portfolio**: Main portfolio management with cash, positions, and trade history
//! - **Position Manager**: Open position tracking with SL/TP modification
//! - **Equity Tracker**: Equity curve and drawdown calculation
//! - **Stop Checks**: SL/TP hit detection with priority rules and entry candle logic
//!
//! ## Key Features
//!
//! - **SL Priority**: When both SL and TP are hit in the same candle, SL is executed
//! - **Entry Candle Rule**: TP is only valid in entry candle if close is beyond TP
//! - **R-Multiple Tracking**: Automatic calculation of risk-adjusted returns
//! - **Equity Curve**: Full equity history with drawdown tracking

#![deny(clippy::all)]

//! ## Example
//!
//! ```rust
//! use omega_portfolio::{Portfolio, check_stops, DEFAULT_PIP_BUFFER_FACTOR};
//! use omega_types::{Direction, OrderType, Signal, ExitReason, Candle};
//! use serde_json::json;
//!
//! // Create portfolio
//! let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");
//!
//! // Open a position
//! let signal = Signal {
//!     direction: Direction::Long,
//!     order_type: OrderType::Market,
//!     entry_price: 1.2000,
//!     stop_loss: 1.1950,
//!     take_profit: 1.2100,
//!     size: Some(1.0),
//!     scenario_id: 0,
//!     tags: vec![],
//!     meta: json!({}),
//! };
//!
//! let id = portfolio.open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0).unwrap();
//!
//! // Check stops on a new bar
//! let bid = Candle {
//!     timestamp_ns: 2_000_000,
//!     open: 1.2050,
//!     high: 1.2110,
//!     low: 1.2040,
//!     close: 1.2100,
//!     volume: 1000.0,
//! };
//! let ask = bid.clone(); // Simplified for example
//!
//! let position = &portfolio.positions()[0];
//! if let Some(result) = check_stops(position, &bid, &ask, 0.0001, 0.5, false) {
//!     let trade = portfolio.close_position(
//!         id,
//!         result.exit_price,
//!         2_000_000,
//!         result.reason,
//!         0.0,
//!     );
//! }
//! ```

pub mod equity;
pub mod error;
pub mod portfolio;
pub mod position_manager;
pub mod stops;

// Re-exports for convenience
pub use equity::EquityTracker;
pub use error::PortfolioError;
pub use portfolio::{Portfolio, TradeStats};
pub use position_manager::PositionManager;
pub use stops::{
    calculate_gap_exit_price, check_stop_loss_only, check_stops, check_take_profit_only,
    StopCheckResult, DEFAULT_PIP_BUFFER_FACTOR,
};
