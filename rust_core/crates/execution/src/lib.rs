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
//! - Uses ChaCha8Rng with configurable seed for reproducible slippage
//! - Pending orders are processed in FIFO order (by creation time, then ID)
//! - State transitions follow defined rules with no rollback

#![deny(clippy::all)]

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
//! let result = engine.execute_market_order(&signal, &costs);
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

// Re-exports for convenience
pub use costs::{
    CommissionConfig, CommissionSchema, CommissionSide, ExecutionCostsConfig, SymbolCosts,
};
pub use engine::{
    ExecutionEngine, ExecutionEngineConfig, MarketOrderResult, PendingFillResult,
    PendingProcessResult,
};
pub use error::ExecutionError;
pub use fees::{CombinedFee, FeeModel, FixedFee, NoFee, PercentageFee, TieredFee};
pub use fill::{check_exit_triggered, market_fill, pending_fill, FillResult};
pub use pending_book::{PendingBook, PendingOrder, TriggerEvent};
pub use slippage::{FixedSlippage, NoSlippage, SlippageModel, VolatilitySlippage};
pub use state::{OrderState, PositionState, StateTransition, TriggerOrderPolicy};
