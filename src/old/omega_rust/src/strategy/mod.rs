//! Pure Rust Strategy Module
//!
//! Dieses Modul implementiert das RustStrategy-Pattern für maximale Performance
//! bei Backtests durch Eliminierung des FFI-Overheads.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Python API Layer                             │
//! │  run_backtest_rust(strategy_name, config, candle_data)          │
//! └──────────────────────────────┬──────────────────────────────────┘
//!                                │ FFI (1x Init)
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Rust Strategy Layer                          │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
//! │  │   Registry  │  │  Executor   │  │  MeanReversionZScore    │ │
//! │  │  (Lookup)   │──│  (Loop)     │──│  (Strategy Impl)        │ │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
//! │                                │                                │
//! │                                ▼                                │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │              IndicatorCache (Wave 1)                        ││
//! │  │  ema(), rsi(), zscore(), atr(), bollinger()                 ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! └──────────────────────────────┬──────────────────────────────────┘
//!                                │ FFI (1x Result)
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Python Result Layer                          │
//! │  BacktestResult { trades, equity_curve, metrics }               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Goals
//!
//! - FFI Calls: Reduziert von ~150.000 auf 2
//! - Speedup: ≥10x vs Python+Rust-Hybrid (Wave 3)
//! - Memory: ≥30% Reduktion vs Python
//!
//! # Usage
//!
//! ```python
//! from omega_rust import run_backtest_rust, StrategyConfig
//!
//! config = StrategyConfig("EURUSD", "M5")
//! result = run_backtest_rust("mean_reversion_z_score", config, candle_data)
//! print(f"Trades: {len(result.trades)}, Time: {result.execution_time_ms}ms")
//! ```

mod types;
mod traits;
mod registry;
mod executor;
pub mod mean_reversion_zscore;

// Re-exports für externe Verwendung
pub use types::{
    CandleData, DataSlice, Direction, Position, PositionAction, StrategyConfig, Timeframe, TradeSignal,
};
pub use traits::RustStrategy;
pub use registry::{create_strategy, list_strategies, register_strategy, strategy_exists};
pub use executor::{run_backtest_rust, BacktestResult, RustExecutor, TradeResult};
