//! Cost calculation modules for trade execution simulation.
//!
//! Provides slippage and fee calculations optimized for backtest scenarios.
//! This is the **Wave 0 Pilot Module** for the Rust migration.
//!
//! ## Available Functions
//!
//! - [`calculate_slippage`] - Single trade slippage calculation
//! - [`calculate_slippage_batch`] - Batch slippage calculation (optimized)
//! - [`calculate_fee`] - Single trade fee calculation
//! - [`calculate_fee_batch`] - Batch fee calculation (optimized)
//!
//! ## Determinism
//!
//! All random components use `ChaCha8` RNG with optional seeding for
//! reproducible backtest results across platforms.
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_0_SLIPPAGE_FEE_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/slippage_fee.md`
//! - Golden Reference: `tests/golden/reference/slippage_fee/slippage_fee_v1.json`

mod fee;
mod slippage;

pub use fee::{calculate_fee, calculate_fee_batch};
pub use slippage::{calculate_slippage, calculate_slippage_batch, DIRECTION_LONG, DIRECTION_SHORT};
