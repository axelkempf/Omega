//! Portfolio implementation for high-performance backtesting.
//!
//! Provides the `PortfolioRust` class which manages portfolio state,
//! position tracking, equity curve, and performance metrics.

use pyo3::prelude::*;
use std::collections::HashMap;

use super::position::PositionRust;
use super::state::{EquityPoint, FeeLogEntry, PortfolioState};
use crate::error::{OmegaError, Result};

/// Rust implementation of Portfolio for high-performance backtesting.
///
/// This class provides the same interface as the Python `Portfolio` class
/// but with significantly improved performance for state management
/// and metric calculation.
///
/// # Example
///
/// ```python
/// from omega_rust import PortfolioRust, PositionRust
///
/// portfolio = PortfolioRust(initial_balance=100000.0)
///
/// # Create and register a position
/// pos = PositionRust(
///     entry_time=1704067200000000,  # Microseconds
///     direction=1,  # Long
///     symbol="EURUSD",
///     entry_price=1.10000,
///     stop_loss=1.09900,
///     take_profit=1.10200,
///     size=1.0,
///     risk_per_trade=100.0
/// )
/// portfolio.register_entry(pos)
///
/// # Close the position
/// pos.close(1704067260000000, 1.10200, "take_profit")
/// portfolio.register_exit(pos)
///
/// # Get summary
/// summary = portfolio.get_summary()
/// ```
#[pyclass]
pub struct PortfolioRust {
    state: PortfolioState,
    open_positions: Vec<PositionRust>,
    closed_positions: Vec<PositionRust>,
    expired_orders: Vec<PositionRust>,
    partial_closed_positions: Vec<PositionRust>,
    closed_position_break_even: Vec<PositionRust>,
    equity_curve: Vec<EquityPoint>,
    fees_log: Vec<FeeLogEntry>,
    /// Enable extra robust metrics in summary
    #[pyo3(get, set)]
    pub enable_backtest_robust_metrics: bool,
    /// Store for extra robust metrics (set externally)
    #[pyo3(get, set)]
    pub backtest_robust_metrics: Option<HashMap<String, f64>>,
}

#[pymethods]
impl PortfolioRust {
    /// Create a new portfolio with the given initial balance.
    ///
    /// # Arguments
    /// * `initial_balance` - Starting balance in account currency (default: 10000.0)
    #[new]
    #[pyo3(signature = (initial_balance=10000.0))]
    #[must_use]
    pub fn new(initial_balance: f64) -> Self {
        // Sentinel value for placeholder timestamp (will be replaced on first update)
        // Using i64::MIN would cause issues with Python datetime conversion,
        // so we use 0 (Unix epoch) as a recognizable placeholder
        const PLACEHOLDER_TIMESTAMP: i64 = 0;
        Self {
            state: PortfolioState::new(initial_balance),
            open_positions: Vec::new(),
            closed_positions: Vec::new(),
            expired_orders: Vec::new(),
            partial_closed_positions: Vec::new(),
            closed_position_break_even: Vec::new(),
            equity_curve: vec![EquityPoint {
                timestamp: PLACEHOLDER_TIMESTAMP,
                equity: initial_balance,
            }],
            fees_log: Vec::new(),
            enable_backtest_robust_metrics: false,
            backtest_robust_metrics: None,
        }
    }

    /// Register a fee (deducts from cash).
    ///
    /// # Arguments
    /// * `amount` - Fee amount
    /// * `time` - Timestamp in microseconds
    /// * `kind` - Fee type (e.g., "entry", "exit", "swap")
    /// * `position` - Optional position associated with the fee
    #[pyo3(signature = (amount, time, kind, position=None))]
    pub fn register_fee(
        &mut self,
        amount: f64,
        time: i64,
        kind: &str,
        position: Option<&mut PositionRust>,
    ) {
        if amount == 0.0 {
            return;
        }

        self.state.cash -= amount;
        self.state.total_fees += amount;
        self.state.equity = self.state.cash;

        let (symbol, size) = match position {
            Some(pos) => {
                if kind == "entry" {
                    pos.entry_fee += amount;
                } else if kind == "exit" {
                    pos.exit_fee += amount;
                }
                (Some(pos.symbol.clone()), Some(pos.size))
            }
            None => (None, None),
        };

        self.fees_log.push(FeeLogEntry {
            time,
            kind: kind.to_string(),
            symbol,
            size,
            fee: amount,
        });
    }

    /// Register a new position entry.
    ///
    /// # Arguments
    /// * `position` - The position to register
    ///
    /// # Errors
    /// Returns error if position has no symbol assigned
    pub fn register_entry(&mut self, position: PositionRust) -> PyResult<()> {
        self.register_entry_impl(position).map_err(Into::into)
    }

    /// Register a position exit.
    ///
    /// # Arguments
    /// * `position` - The position to close (will be mutated)
    ///
    /// # Errors
    /// Returns error if position state is invalid
    pub fn register_exit(&mut self, position: &mut PositionRust) -> PyResult<()> {
        self.register_exit_impl(position);
        Ok(())
    }

    /// Get all open positions, optionally filtered by symbol.
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol filter
    #[pyo3(signature = (symbol=None))]
    #[must_use]
    pub fn get_open_positions(&self, symbol: Option<&str>) -> Vec<PositionRust> {
        symbol.map_or_else(
            || self.open_positions.clone(),
            |s| {
                self.open_positions
                    .iter()
                    .filter(|p| p.symbol == s)
                    .cloned()
                    .collect()
            },
        )
    }

    /// Get all closed positions.
    #[must_use]
    pub fn get_closed_positions(&self) -> Vec<PositionRust> {
        self.closed_positions.clone()
    }

    /// Update equity and drawdown tracking.
    ///
    /// # Arguments
    /// * `current_time` - Current timestamp in microseconds
    pub fn update(&mut self, current_time: i64) {
        if self.state.start_timestamp.is_none() {
            self.state.start_timestamp = Some(current_time);
        }

        self.state.equity = self.state.cash;

        // Update equity curve
        if let Some(last) = self.equity_curve.last_mut() {
            if last.timestamp == current_time {
                last.equity = self.state.equity;
            } else {
                self.equity_curve.push(EquityPoint {
                    timestamp: current_time,
                    equity: self.state.equity,
                });
            }
        }

        // Update max equity
        if self.state.equity > self.state.max_equity {
            self.state.max_equity = self.state.equity;
        }

        // Update drawdowns
        let drawdown = self.state.max_equity - self.state.equity;
        if drawdown > self.state.max_drawdown {
            self.state.max_drawdown = drawdown;
        }

        let drawdown_initial = self.state.initial_balance - self.state.equity;
        if drawdown_initial > self.state.initial_max_drawdown {
            self.state.initial_max_drawdown = drawdown_initial;
        }
    }

    /// Get portfolio summary metrics.
    ///
    /// Returns a dictionary with all performance metrics.
    #[must_use]
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        // Basic metrics
        summary.insert("Initial Balance".to_string(), self.state.initial_balance);
        summary.insert("Final Balance".to_string(), round_2(self.state.cash));
        summary.insert("Equity".to_string(), round_2(self.state.equity));
        summary.insert(
            "Max Drawdown".to_string(),
            round_2(self.state.max_drawdown),
        );
        summary.insert(
            "Drawdown Initial Balance".to_string(),
            round_2(self.state.initial_max_drawdown),
        );
        summary.insert("Total Fees".to_string(), round_2(self.state.total_fees));

        // Total Lots
        let total_lots: f64 = self
            .closed_positions
            .iter()
            .chain(self.partial_closed_positions.iter())
            .map(|p| p.size)
            .sum();
        summary.insert("Total Lots".to_string(), round_2(total_lots));

        // Trade counts
        summary.insert(
            "Total Trades".to_string(),
            self.closed_positions.len() as f64,
        );
        summary.insert(
            "Expired Orders".to_string(),
            self.expired_orders.len() as f64,
        );
        summary.insert(
            "Partial Closed Orders".to_string(),
            self.partial_closed_positions.len() as f64,
        );
        summary.insert(
            "Orders closed at Break Even".to_string(),
            self.closed_position_break_even.len() as f64,
        );

        // R-Multiple
        summary.insert("Avg R-Multiple".to_string(), self.avg_r_multiple());

        // Winrate
        let wins = self
            .closed_positions
            .iter()
            .filter(|p| p.result.is_some_and(|r| r > 0.0))
            .count();
        let losses = self.closed_positions.len() - wins;

        let winrate = if self.closed_positions.is_empty() {
            0.0
        } else {
            round_2((wins as f64 / self.closed_positions.len() as f64) * 100.0)
        };

        summary.insert("Winrate".to_string(), winrate);
        summary.insert("Wins".to_string(), wins as f64);
        summary.insert("Losses".to_string(), losses as f64);

        // Add robust metrics if enabled
        if self.enable_backtest_robust_metrics {
            self.add_robust_metrics(&mut summary);
        }

        summary
    }

    /// Get equity curve as list of (timestamp, equity) tuples.
    ///
    /// Returns the equity curve built from closed positions.
    #[must_use]
    pub fn get_equity_curve(&self) -> Vec<(i64, f64)> {
        let mut curve: Vec<(i64, f64)> = Vec::new();

        if let Some(start_ts) = self.state.start_timestamp {
            curve.push((start_ts, self.state.initial_balance));
        }

        let mut equity = self.state.initial_balance;

        // Collect all positions with results
        let mut all_positions: Vec<&PositionRust> = self
            .closed_positions
            .iter()
            .chain(self.partial_closed_positions.iter())
            .filter(|p| p.result.is_some() && p.exit_time.is_some())
            .collect();

        // Sort by exit_time
        all_positions.sort_by_key(|p| p.exit_time.unwrap_or(0));

        for pos in all_positions {
            let entry_fee = pos.entry_fee;
            let exit_fee = pos.exit_fee;
            let result = pos.result.unwrap_or(0.0);
            let net_result = result - entry_fee - exit_fee;
            equity += net_result;

            if let Some(exit_time) = pos.exit_time {
                curve.push((exit_time, equity));
            }
        }

        curve
    }

    // =========================================================================
    // Getters for state values
    // =========================================================================

    #[getter]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn initial_balance(&self) -> f64 {
        self.state.initial_balance
    }

    #[getter]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn cash(&self) -> f64 {
        self.state.cash
    }

    #[getter]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn equity(&self) -> f64 {
        self.state.equity
    }

    #[getter]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn max_equity(&self) -> f64 {
        self.state.max_equity
    }

    #[getter]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn max_drawdown(&self) -> f64 {
        self.state.max_drawdown
    }

    #[getter]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn total_fees(&self) -> f64 {
        self.state.total_fees
    }

    #[getter]
    #[must_use]
    pub fn num_open_positions(&self) -> usize {
        self.open_positions.len()
    }

    #[getter]
    #[must_use]
    pub fn num_closed_positions(&self) -> usize {
        self.closed_positions.len()
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "PortfolioRust(balance={:.2}, equity={:.2}, open={}, closed={})",
            self.state.cash,
            self.state.equity,
            self.open_positions.len(),
            self.closed_positions.len()
        )
    }
}

// =============================================================================
// Internal Implementation
// =============================================================================

impl PortfolioRust {
    /// Internal implementation of `register_entry`
    fn register_entry_impl(&mut self, position: PositionRust) -> Result<()> {
        if position.symbol.is_empty() {
            return Err(OmegaError::InvalidParameter {
                reason: "Position must have a 'symbol' assigned.".to_string(),
            });
        }
        self.open_positions.push(position);
        Ok(())
    }

    /// Internal implementation of `register_exit`
    fn register_exit_impl(&mut self, position: &mut PositionRust) {
        // Handle pending expiry
        if position.status == "pending" && position.reason.as_deref() == Some("limit_expired") {
            self.expired_orders.push(position.clone());
            position.status = "closed".to_string();
            self.remove_from_open(position.entry_time);
            return;
        }

        // Handle different exit types
        if position.status == "open" {
            match position.reason.as_deref() {
                Some("partial_exit") => {
                    self.partial_closed_positions.push(position.clone());
                }
                Some("break_even_stop_loss") => {
                    self.closed_positions.push(position.clone());
                    self.closed_position_break_even.push(position.clone());
                }
                _ => {
                    self.closed_positions.push(position.clone());
                }
            }
        }

        position.status = "closed".to_string();

        // Credit/debit the result
        let result = position.result.unwrap_or(0.0);
        self.state.cash += result;
        self.state.equity = self.state.cash;

        self.remove_from_open(position.entry_time);
    }

    /// Remove position from `open_positions` by `entry_time`
    fn remove_from_open(&mut self, entry_time: i64) {
        self.open_positions.retain(|p| p.entry_time != entry_time);
    }

    /// Calculate average R-multiple weighted by `risk_per_trade`
    fn avg_r_multiple(&self) -> f64 {
        let all_positions: Vec<&PositionRust> = self
            .closed_positions
            .iter()
            .chain(self.partial_closed_positions.iter())
            .filter(|p| p.risk_per_trade > 0.0)
            .collect();

        if all_positions.is_empty() {
            return 0.0;
        }

        let total_weighted_r: f64 = all_positions
            .iter()
            .map(|p| p.r_multiple() * p.risk_per_trade)
            .sum();
        let total_risk: f64 = all_positions.iter().map(|p| p.risk_per_trade).sum();

        if total_risk > 0.0 {
            round_3(total_weighted_r / total_risk)
        } else {
            0.0
        }
    }

    /// Add robust metrics to summary if enabled
    fn add_robust_metrics(&self, summary: &mut HashMap<String, f64>) {
        let Some(metrics) = &self.backtest_robust_metrics else {
            // Add defaults
            summary.insert("Robustness 1".to_string(), 0.0);
            summary.insert("Robustness 1 Num Samples".to_string(), 0.0);
            summary.insert("Cost Shock Score".to_string(), 0.0);
            summary.insert("Timing Jitter Score".to_string(), 0.0);
            summary.insert("Trade Dropout Score".to_string(), 0.0);
            summary.insert("Ulcer Index".to_string(), 0.0);
            summary.insert("Ulcer Index Score".to_string(), 0.0);
            summary.insert("Data Jitter Score".to_string(), 0.0);
            summary.insert("Data Jitter Num Samples".to_string(), 0.0);
            summary.insert("p_mean_gt".to_string(), 1.0);
            summary.insert("Stability Score".to_string(), 1.0);
            summary.insert("TP/SL Stress Score".to_string(), 1.0);
            return;
        };

        // Map metric names from snake_case to human-readable
        summary.insert(
            "Robustness 1".to_string(),
            *metrics.get("robustness_1").unwrap_or(&0.0),
        );
        summary.insert(
            "Robustness 1 Num Samples".to_string(),
            *metrics.get("robustness_1_num_samples").unwrap_or(&0.0),
        );
        summary.insert(
            "Cost Shock Score".to_string(),
            *metrics.get("cost_shock_score").unwrap_or(&0.0),
        );
        summary.insert(
            "Timing Jitter Score".to_string(),
            *metrics.get("timing_jitter_score").unwrap_or(&0.0),
        );
        summary.insert(
            "Trade Dropout Score".to_string(),
            *metrics.get("trade_dropout_score").unwrap_or(&0.0),
        );
        summary.insert(
            "Ulcer Index".to_string(),
            *metrics.get("ulcer_index").unwrap_or(&0.0),
        );
        summary.insert(
            "Ulcer Index Score".to_string(),
            *metrics.get("ulcer_index_score").unwrap_or(&0.0),
        );
        summary.insert(
            "Data Jitter Score".to_string(),
            *metrics.get("data_jitter_score").unwrap_or(&0.0),
        );
        summary.insert(
            "Data Jitter Num Samples".to_string(),
            *metrics.get("data_jitter_num_samples").unwrap_or(&0.0),
        );
        summary.insert(
            "p_mean_gt".to_string(),
            *metrics.get("p_mean_gt").unwrap_or(&1.0),
        );
        summary.insert(
            "Stability Score".to_string(),
            *metrics.get("stability_score").unwrap_or(&1.0),
        );
        summary.insert(
            "TP/SL Stress Score".to_string(),
            *metrics.get("tp_sl_stress_score").unwrap_or(&1.0),
        );
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Round to 2 decimal places
#[inline]
fn round_2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Round to 3 decimal places
#[inline]
fn round_3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::position::DIRECTION_LONG;
    use approx::assert_relative_eq;

    fn create_test_position(
        entry_time: i64,
        direction: i8,
        symbol: &str,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
    ) -> PositionRust {
        PositionRust::new(
            entry_time,
            direction,
            symbol.to_string(),
            entry_price,
            stop_loss,
            take_profit,
            1.0,
            100.0,
        )
    }

    #[test]
    fn test_portfolio_new() {
        let portfolio = PortfolioRust::new(100_000.0);
        assert_relative_eq!(portfolio.initial_balance(), 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.cash(), 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.equity(), 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.max_equity(), 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.max_drawdown(), 0.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.total_fees(), 0.0, epsilon = 1e-8);
        assert_eq!(portfolio.num_open_positions(), 0);
        assert_eq!(portfolio.num_closed_positions(), 0);
    }

    #[test]
    fn test_register_entry() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_entry(pos).unwrap();
        assert_eq!(portfolio.num_open_positions(), 1);
    }

    #[test]
    fn test_register_entry_empty_symbol() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "",
            1.10000,
            1.09900,
            1.10200,
        );

        let result = portfolio.register_entry_impl(pos);
        assert!(result.is_err());
    }

    #[test]
    fn test_register_exit_win() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let mut pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_entry(pos.clone()).unwrap();
        pos.close(
            1_704_067_260_000_000,
            1.10200,
            "take_profit".to_string(),
        );
        portfolio.register_exit(&mut pos).unwrap();

        assert_eq!(portfolio.num_open_positions(), 0);
        assert_eq!(portfolio.num_closed_positions(), 1);
        // 2R * $100 = $200 profit
        assert_relative_eq!(portfolio.cash(), 100_200.0, epsilon = 1e-8);
    }

    #[test]
    fn test_register_exit_loss() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let mut pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_entry(pos.clone()).unwrap();
        pos.close(1_704_067_260_000_000, 1.09900, "stop_loss".to_string());
        portfolio.register_exit(&mut pos).unwrap();

        // -1R * $100 = -$100 loss
        assert_relative_eq!(portfolio.cash(), 99_900.0, epsilon = 1e-8);
    }

    #[test]
    fn test_register_fee() {
        let mut portfolio = PortfolioRust::new(100_000.0);

        portfolio.register_fee(3.0, 1_704_067_200_000_000, "entry", None);

        assert_relative_eq!(portfolio.cash(), 99_997.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.total_fees(), 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_register_fee_with_position() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let mut pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_fee(3.0, 1_704_067_200_000_000, "entry", Some(&mut pos));

        assert_relative_eq!(pos.entry_fee, 3.0, epsilon = 1e-8);
        assert_relative_eq!(portfolio.total_fees(), 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_update_drawdown() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let mut pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_entry(pos.clone()).unwrap();
        portfolio.update(1_704_067_200_000_000);

        // Close with loss
        pos.close(1_704_067_260_000_000, 1.09900, "stop_loss".to_string());
        portfolio.register_exit(&mut pos).unwrap();
        portfolio.update(1_704_067_260_000_000);

        assert!(portfolio.max_drawdown() > 0.0);
    }

    #[test]
    fn test_get_summary() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        let mut pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_entry(pos.clone()).unwrap();
        pos.close(
            1_704_067_260_000_000,
            1.10200,
            "take_profit".to_string(),
        );
        portfolio.register_exit(&mut pos).unwrap();

        let summary = portfolio.get_summary();

        assert_eq!(summary.get("Initial Balance"), Some(&100_000.0));
        assert_relative_eq!(
            *summary.get("Final Balance").unwrap(),
            100_200.0,
            epsilon = 1e-2
        );
        assert_eq!(summary.get("Total Trades"), Some(&1.0));
        assert_eq!(summary.get("Wins"), Some(&1.0));
        assert_eq!(summary.get("Losses"), Some(&0.0));
        assert_eq!(summary.get("Winrate"), Some(&100.0));
    }

    #[test]
    fn test_get_summary_empty() {
        let portfolio = PortfolioRust::new(100_000.0);
        let summary = portfolio.get_summary();

        assert_eq!(summary.get("Total Trades"), Some(&0.0));
        assert_eq!(summary.get("Winrate"), Some(&0.0));
    }

    #[test]
    fn test_get_open_positions_filter() {
        let mut portfolio = PortfolioRust::new(100_000.0);

        let pos1 = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );
        let pos2 = create_test_position(
            1_704_067_201_000_000,
            DIRECTION_LONG,
            "GBPUSD",
            1.27000,
            1.26900,
            1.27200,
        );

        portfolio.register_entry(pos1).unwrap();
        portfolio.register_entry(pos2).unwrap();

        let all = portfolio.get_open_positions(None);
        assert_eq!(all.len(), 2);

        let eurusd = portfolio.get_open_positions(Some("EURUSD"));
        assert_eq!(eurusd.len(), 1);
        assert_eq!(eurusd[0].symbol, "EURUSD");

        let gbpusd = portfolio.get_open_positions(Some("GBPUSD"));
        assert_eq!(gbpusd.len(), 1);

        let usdjpy = portfolio.get_open_positions(Some("USDJPY"));
        assert_eq!(usdjpy.len(), 0);
    }

    #[test]
    fn test_avg_r_multiple() {
        let mut portfolio = PortfolioRust::new(100_000.0);

        // Position 1: 2R win
        let mut pos1 = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );
        portfolio.register_entry(pos1.clone()).unwrap();
        pos1.close(
            1_704_067_260_000_000,
            1.10200,
            "take_profit".to_string(),
        );
        portfolio.register_exit(&mut pos1).unwrap();

        // Position 2: -1R loss
        let mut pos2 = create_test_position(
            1_704_067_300_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );
        portfolio.register_entry(pos2.clone()).unwrap();
        pos2.close(1_704_067_360_000_000, 1.09900, "stop_loss".to_string());
        portfolio.register_exit(&mut pos2).unwrap();

        let summary = portfolio.get_summary();
        // (2R + -1R) / 2 = 0.5R
        assert_relative_eq!(
            *summary.get("Avg R-Multiple").unwrap(),
            0.5,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_equity_curve() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        portfolio.update(1_704_067_100_000_000);

        let mut pos = create_test_position(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD",
            1.10000,
            1.09900,
            1.10200,
        );

        portfolio.register_entry(pos.clone()).unwrap();
        pos.close(
            1_704_067_260_000_000,
            1.10200,
            "take_profit".to_string(),
        );
        portfolio.register_exit(&mut pos).unwrap();

        let curve = portfolio.get_equity_curve();
        assert!(!curve.is_empty());
        // First point should be initial balance
        assert_relative_eq!(curve[0].1, 100_000.0, epsilon = 1e-8);
    }

    #[test]
    fn test_portfolio_repr() {
        let portfolio = PortfolioRust::new(100_000.0);
        let repr = portfolio.__repr__();
        assert!(repr.contains("100000.00"));
        assert!(repr.contains("PortfolioRust"));
    }

    #[test]
    fn test_robust_metrics_disabled() {
        let portfolio = PortfolioRust::new(100_000.0);
        let summary = portfolio.get_summary();

        // Should NOT contain robust metrics when disabled
        assert!(!summary.contains_key("Cost Shock Score"));
        assert!(!summary.contains_key("Robustness 1"));
    }

    #[test]
    fn test_robust_metrics_enabled_with_none() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        portfolio.enable_backtest_robust_metrics = true;

        let summary = portfolio.get_summary();

        // Should contain defaults
        assert_eq!(summary.get("Robustness 1"), Some(&0.0));
        assert_eq!(summary.get("Cost Shock Score"), Some(&0.0));
        assert_eq!(summary.get("p_mean_gt"), Some(&1.0));
        assert_eq!(summary.get("Stability Score"), Some(&1.0));
    }

    #[test]
    fn test_robust_metrics_enabled_with_values() {
        let mut portfolio = PortfolioRust::new(100_000.0);
        portfolio.enable_backtest_robust_metrics = true;

        let mut metrics = HashMap::new();
        metrics.insert("robustness_1".to_string(), 0.85);
        metrics.insert("cost_shock_score".to_string(), 0.92);
        metrics.insert("stability_score".to_string(), 0.78);
        portfolio.backtest_robust_metrics = Some(metrics);

        let summary = portfolio.get_summary();

        assert_relative_eq!(
            *summary.get("Robustness 1").unwrap(),
            0.85,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            *summary.get("Cost Shock Score").unwrap(),
            0.92,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            *summary.get("Stability Score").unwrap(),
            0.78,
            epsilon = 1e-8
        );
    }
}
