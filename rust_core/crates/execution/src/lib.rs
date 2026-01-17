//! # Omega Execution
//!
//! Order execution, fill simulation, and cost modeling for the Omega V2 trading system.
//!
//! ## Overview
//!
//! This crate provides:
//!
//! - **Slippage Models**: Simulate realistic fill prices with fixed or volatility-based slippage
//! - **Fee Models**: Calculate trading costs (percentage, fixed, or combined)
//! - **Fill Logic**: Market and pending order (limit/stop) fill simulation with gap-aware pricing
//! - **State Machines**: Deterministic order and position state transitions
//! - **Pending Book**: Management of limit and stop orders with FIFO trigger ordering
//! - **Execution Engine**: Coordinated order processing with deterministic RNG
//! - **Costs Loading**: YAML-based execution costs configuration
//!
//! ## Determinism
//!
//! All execution is deterministic:
//!
//! - Uses `ChaCha8Rng` with configurable seed for reproducible slippage
//! - Pending orders are processed in FIFO order (by creation time, then ID)
//! - State transitions follow defined rules with no rollback

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

//! ## Example
//!
//! ```rust
//! use omega_execution::{ExecutionEngine, ExecutionEngineConfig, SymbolCosts};
//! use omega_types::{Direction, OrderType, Signal};
//! use serde_json::json;
//!
//! // Create engine with seed for determinism
//! let config = ExecutionEngineConfig {
//!     rng_seed: 42,
//!     ..Default::default()
//! };
//! let mut engine = ExecutionEngine::new(config);
//!
//! // Zero-cost model for testing
//! let costs = SymbolCosts::zero_cost(0.0001);
//!
//! // Execute a market order
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
//! let result = engine.execute_market_order(&signal, &costs).unwrap();
//! assert!(result.filled);
//! ```

pub mod costs;
pub mod engine;
pub mod error;
pub mod fees;
pub mod fill;
pub mod pending_book;
pub mod slippage;
pub mod state;
pub mod symbol_specs;

// Re-exports for convenience
/// Execution cost configuration and symbol cost types.
pub use costs::{
    CommissionConfig, CommissionSchema, CommissionSide, ExecutionCostsConfig, SymbolCosts,
};
/// Execution engine types.
pub use engine::{
    ExecutionEngine, ExecutionEngineConfig, ExitFillResult, MarketOrderResult, PendingFillResult,
    PendingOrderRejection, PendingProcessResult,
};
/// Execution error type.
pub use error::ExecutionError;
/// Fee model implementations.
pub use fees::{
    CombinedFee, FeeModel, FixedFee, MinFee, NoFee, PerMillionNotionalFee, PercentageFee, TieredFee,
};
/// Fill helpers and results.
pub use fill::{FillResult, market_fill, pending_fill};
/// Pending order book types.
pub use pending_book::{PendingBook, PendingOrder, TriggerEvent};
/// Slippage model implementations.
pub use slippage::{FixedSlippage, NoSlippage, SlippageModel, VolatilitySlippage};
/// Order and position state types.
pub use state::{OrderState, PositionState, StateTransition, TriggerOrderPolicy};
/// Symbol specifications loader.
pub use symbol_specs::{SymbolSpec, get_symbol_spec_or_default, load_symbol_specs};
