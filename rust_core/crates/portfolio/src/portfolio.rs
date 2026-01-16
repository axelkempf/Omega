//! Portfolio management for backtesting.
//!
//! The Portfolio struct combines position management, equity tracking,
//! and trade recording into a unified interface.

use crate::equity::EquityTracker;
use crate::error::PortfolioError;
use crate::position_manager::PositionManager;
use omega_types::{Direction, ExitReason, Position, Signal, Trade};
use serde_json::{Map as JsonMap, Value as JsonValue};

const CONSISTENCY_EPS: f64 = 1e-8;

/// Portfolio state for backtesting.
///
/// Manages:
/// - Cash balance
/// - Open positions
/// - Closed trades
/// - Equity curve tracking
#[derive(Debug)]
pub struct Portfolio {
    /// Current cash balance (realized funds)
    cash: f64,
    /// Position manager
    position_manager: PositionManager,
    /// Closed trades history
    closed_trades: Vec<Trade>,
    /// Equity tracker
    equity_tracker: EquityTracker,
    /// Symbol being traded
    symbol: String,
}

impl Portfolio {
    /// Creates a new portfolio with the given initial balance.
    ///
    /// # Arguments
    /// * `initial_balance` - Starting cash balance
    /// * `max_positions` - Maximum concurrent positions allowed
    /// * `symbol` - Trading symbol
    pub fn new(initial_balance: f64, max_positions: usize, symbol: impl Into<String>) -> Self {
        Self {
            cash: initial_balance,
            position_manager: PositionManager::new(max_positions),
            closed_trades: Vec::new(),
            equity_tracker: EquityTracker::new(initial_balance),
            symbol: symbol.into(),
        }
    }

    /// Checks if a new position can be opened.
    #[must_use]
    pub fn can_open_position(&self) -> bool {
        self.position_manager.can_open()
    }

    /// Returns the number of available position slots.
    #[must_use]
    pub fn available_slots(&self) -> usize {
        self.position_manager.available_slots()
    }

    /// Opens a new position.
    ///
    /// # Arguments
    /// * `signal` - The trading signal
    /// * `fill_price` - Actual fill price (after slippage)
    /// * `size` - Position size in lots
    /// * `entry_time_ns` - Entry timestamp
    /// * `entry_fee` - Entry fee to deduct from cash
    ///
    /// # Returns
    /// The assigned position ID.
    ///
    /// # Errors
    /// Returns an error if position limits are exceeded or size is invalid.
    pub fn open_position(
        &mut self,
        signal: &Signal,
        fill_price: f64,
        size: f64,
        entry_time_ns: i64,
        entry_fee: f64,
    ) -> Result<u64, PortfolioError> {
        let id = self
            .position_manager
            .open_position(signal, fill_price, size, entry_time_ns)?;

        // Deduct entry fee from cash
        self.cash -= entry_fee;

        Ok(id)
    }

    /// Closes a position.
    ///
    /// # Arguments
    /// * `position_id` - ID of the position to close
    /// * `exit_price` - Exit fill price
    /// * `exit_time_ns` - Exit timestamp
    /// * `reason` - Exit reason (SL, TP, Manual, etc.)
    /// * `exit_fee` - Exit fee to deduct from cash
    ///
    /// # Returns
    /// The completed Trade if the position was found.
    pub fn close_position(
        &mut self,
        position_id: u64,
        exit_price: f64,
        exit_time_ns: i64,
        reason: ExitReason,
        exit_fee: f64,
    ) -> Option<Trade> {
        let position = self.position_manager.close_position(position_id)?;

        // Calculate PnL
        let pnl = match position.direction {
            Direction::Long => (exit_price - position.entry_price) * position.size,
            Direction::Short => (position.entry_price - exit_price) * position.size,
        };

        // Calculate risk and R-multiple
        let risk = (position.entry_price - position.stop_loss).abs() * position.size;
        let r_multiple = if risk > 0.0 { pnl / risk } else { 0.0 };

        let mut meta_map = match position.meta {
            JsonValue::Object(map) => map,
            other => {
                let mut map = JsonMap::new();
                if !other.is_null() {
                    map.insert("raw_meta".to_string(), other);
                }
                map
            }
        };

        if !meta_map.contains_key("stop_loss_kind") {
            let kind = match reason {
                ExitReason::BreakEvenStopLoss => "break_even",
                ExitReason::TrailingStopLoss => "trailing",
                _ => "initial",
            };
            meta_map.insert("stop_loss_kind".to_string(), JsonValue::String(kind.to_string()));
        }

        if !meta_map.contains_key("in_entry_candle") {
            let in_entry_candle = exit_time_ns == position.entry_time_ns;
            meta_map.insert(
                "in_entry_candle".to_string(),
                JsonValue::Bool(in_entry_candle),
            );
        }

        // Create trade record
        let trade = Trade {
            entry_time_ns: position.entry_time_ns,
            exit_time_ns,
            direction: position.direction,
            symbol: self.symbol.clone(),
            entry_price: position.entry_price,
            exit_price,
            stop_loss: position.stop_loss,
            take_profit: position.take_profit,
            size: position.size,
            result: pnl,
            r_multiple,
            reason,
            scenario_id: position.scenario_id,
            meta: JsonValue::Object(meta_map),
        };

        // Update cash: add PnL, deduct exit fee
        self.cash += pnl - exit_fee;

        // Record trade
        self.closed_trades.push(trade.clone());

        Some(trade)
    }

    /// Updates the equity tracker with current market price.
    ///
    /// Should be called at the end of each bar to track equity curve.
    ///
    /// # Arguments
    /// * `timestamp_ns` - Current timestamp
    /// * `current_price` - Current market price for unrealized `PnL` calculation
    pub fn update_equity(&mut self, timestamp_ns: i64, current_price: f64) {
        let unrealized_pnl = self.position_manager.total_unrealized_pnl(current_price);
        let equity = self.cash + unrealized_pnl;
        self.equity_tracker.update(timestamp_ns, equity, self.cash);
    }

    /// Returns the current cash balance.
    #[must_use]
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Returns the current equity (cash + unrealized `PnL`).
    #[must_use]
    pub fn equity(&self) -> f64 {
        self.equity_tracker.equity()
    }

    /// Returns a reference to the position manager.
    #[must_use]
    pub fn position_manager(&self) -> &PositionManager {
        &self.position_manager
    }

    /// Returns a mutable reference to the position manager.
    pub fn position_manager_mut(&mut self) -> &mut PositionManager {
        &mut self.position_manager
    }

    /// Returns all open positions.
    #[must_use]
    pub fn positions(&self) -> &[Position] {
        self.position_manager.positions()
    }

    /// Returns the number of open positions.
    #[must_use]
    pub fn position_count(&self) -> usize {
        self.position_manager.len()
    }

    /// Returns all closed trades.
    #[must_use]
    pub fn closed_trades(&self) -> &[Trade] {
        &self.closed_trades
    }

    /// Returns the number of closed trades.
    #[must_use]
    pub fn trade_count(&self) -> usize {
        self.closed_trades.len()
    }

    /// Returns a reference to the equity tracker.
    #[must_use]
    pub fn equity_tracker(&self) -> &EquityTracker {
        &self.equity_tracker
    }

    /// Consumes the portfolio and returns the equity tracker.
    #[must_use]
    pub fn into_equity_tracker(self) -> EquityTracker {
        self.equity_tracker
    }

    /// Consumes the portfolio and returns the closed trades.
    #[must_use]
    pub fn into_trades(self) -> Vec<Trade> {
        self.closed_trades
    }

    /// Returns the maximum drawdown percentage.
    #[must_use]
    pub fn max_drawdown(&self) -> f64 {
        self.equity_tracker.max_drawdown()
    }

    /// Returns the maximum drawdown in absolute terms.
    #[must_use]
    pub fn max_drawdown_abs(&self) -> f64 {
        self.equity_tracker.max_drawdown_abs()
    }

    /// Returns the total return percentage.
    #[must_use]
    pub fn total_return(&self) -> f64 {
        self.equity_tracker.total_return()
    }

    /// Returns the total return in absolute terms.
    #[must_use]
    pub fn total_return_abs(&self) -> f64 {
        self.equity_tracker.total_return_abs()
    }

    /// Returns the initial balance.
    #[must_use]
    pub fn initial_balance(&self) -> f64 {
        self.equity_tracker.initial_balance()
    }

    /// Checks if a position was entered in the current bar.
    #[must_use]
    pub fn is_entry_candle(&self, position_id: u64, current_bar_ns: i64) -> bool {
        self.position_manager
            .is_entry_candle(position_id, current_bar_ns)
    }

    /// Calculates total fees paid (sum of all trade fees).
    ///
    /// Note: This is an approximation based on cash changes.
    #[must_use]
    pub fn total_fees(&self) -> f64 {
        // Total PnL from trades
        let gross_pnl: f64 = self.closed_trades.iter().map(|t| t.result).sum();

        // Cash change from initial
        let cash_change = self.cash - self.equity_tracker.initial_balance();

        // Fees = gross_pnl - cash_change (approximately)
        // This doesn't account for unrealized PnL
        0.0_f64.max(gross_pnl - cash_change)
    }

    /// Returns the symbol being traded.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Validates portfolio consistency after a backtest run.
    ///
    /// Ensures the recorded equity equals cash plus unrealized PnL within
    /// a floating-point tolerance and that all values are finite.
    ///
    /// # Arguments
    /// * `current_price` - Market price used to compute unrealized PnL
    ///
    /// # Errors
    /// Returns [`PortfolioError::ConsistencyViolation`] when the invariant
    /// is violated, or [`PortfolioError::NonFiniteValue`] if any input or
    /// computed value is not finite.
    pub fn validate_consistency(&self, current_price: f64) -> Result<(), PortfolioError> {
        let ensure_finite = |field: &str, value: f64| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(PortfolioError::NonFiniteValue {
                    field: field.to_string(),
                    value,
                })
            }
        };

        ensure_finite("current_price", current_price)?;

        let cash = self.cash;
        let equity = self.equity_tracker.equity();
        let unrealized = self.position_manager.total_unrealized_pnl(current_price);
        let expected = cash + unrealized;

        ensure_finite("cash", cash)?;
        ensure_finite("equity", equity)?;
        ensure_finite("unrealized_pnl", unrealized)?;
        ensure_finite("expected_equity", expected)?;

        let diff = (equity - expected).abs();
        if diff > CONSISTENCY_EPS {
            return Err(PortfolioError::ConsistencyViolation {
                equity,
                expected,
                diff,
                tolerance: CONSISTENCY_EPS,
            });
        }

        Ok(())
    }

    /// Calculates summary statistics for closed trades.
    pub fn calculate_stats(&self) -> TradeStats {
        let trades = &self.closed_trades;

        if trades.is_empty() {
            return TradeStats::default();
        }

        let wins: Vec<_> = trades.iter().filter(|t| t.result > 0.0).collect();
        let losses: Vec<_> = trades.iter().filter(|t| t.result <= 0.0).collect();

        let total_trades = trades.len();
        let win_count = wins.len();
        let loss_count = losses.len();

        let to_f64 = |value: usize| f64::from(u32::try_from(value).unwrap_or(u32::MAX));
        let total_trades_f64 = to_f64(total_trades);
        let win_count_f64 = to_f64(win_count);
        let loss_count_f64 = to_f64(loss_count);

        let win_rate = win_count_f64 / total_trades_f64;

        let gross_profit: f64 = wins.iter().map(|t| t.result).sum();
        let gross_loss: f64 = losses.iter().map(|t| t.result).sum::<f64>().abs();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_win = if win_count > 0 {
            gross_profit / win_count_f64
        } else {
            0.0
        };

        let avg_loss = if loss_count > 0 {
            gross_loss / loss_count_f64
        } else {
            0.0
        };

        let avg_r_multiple = trades.iter().map(|t| t.r_multiple).sum::<f64>() / total_trades_f64;

        let largest_win = wins.iter().map(|t| t.result).fold(0.0, f64::max);
        let largest_loss = losses.iter().map(|t| t.result.abs()).fold(0.0, f64::max);

        TradeStats {
            total_trades,
            wins: win_count,
            losses: loss_count,
            win_rate,
            gross_profit,
            gross_loss,
            net_profit: gross_profit - gross_loss,
            profit_factor,
            avg_win,
            avg_loss,
            avg_r_multiple,
            largest_win,
            largest_loss,
        }
    }
}

/// Summary statistics for trades.
#[derive(Debug, Clone, Default)]
pub struct TradeStats {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub wins: usize,
    /// Number of losing trades
    pub losses: usize,
    /// Win rate (0-1)
    pub win_rate: f64,
    /// Gross profit from winners
    pub gross_profit: f64,
    /// Gross loss from losers (positive value)
    pub gross_loss: f64,
    /// Net profit (`gross_profit` - `gross_loss`)
    pub net_profit: f64,
    /// Profit factor (`gross_profit` / `gross_loss`)
    pub profit_factor: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade (positive value)
    pub avg_loss: f64,
    /// Average R-multiple
    pub avg_r_multiple: f64,
    /// Largest winning trade
    pub largest_win: f64,
    /// Largest losing trade (positive value)
    pub largest_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use omega_types::OrderType;
    use serde_json::json;

    fn make_signal(direction: Direction, entry: f64, sl: f64, tp: f64) -> Signal {
        Signal {
            direction,
            order_type: OrderType::Market,
            entry_price: entry,
            stop_loss: sl,
            take_profit: tp,
            size: Some(1.0),
            scenario_id: 0,
            tags: vec![],
            meta: json!({}),
        }
    }

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        assert_relative_eq!(portfolio.cash(), 10_000.0, epsilon = 1e-10);
        assert_relative_eq!(portfolio.initial_balance(), 10_000.0, epsilon = 1e-10);
        assert_eq!(portfolio.position_count(), 0);
        assert!(portfolio.can_open_position());
    }

    #[test]
    fn test_open_and_close_position() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);

        // Open position with 5 EUR entry fee
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 5.0)
            .unwrap();

        assert_eq!(portfolio.position_count(), 1);
        assert_relative_eq!(portfolio.cash(), 9_995.0, epsilon = 1e-10);

        // Close at TP with 5 EUR exit fee
        let trade = portfolio
            .close_position(id, 1.2100, 2_000_000, ExitReason::TakeProfit, 5.0)
            .unwrap();

        assert_eq!(portfolio.position_count(), 0);
        // PnL = 0.0100 * 1.0 = 0.01
        // Cash = 9995 + 0.01 - 5 = 9990.01
        assert_relative_eq!(portfolio.cash(), 9_990.01, epsilon = 1e-10);
        assert_relative_eq!(trade.result, 0.0100, epsilon = 1e-10);
    }

    #[test]
    fn test_r_multiple_calculation() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        // Entry at 1.2000, SL at 1.1950 (50 pips risk), TP at 1.2100 (100 pips reward)
        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        // Close at TP
        let trade = portfolio
            .close_position(id, 1.2100, 2_000_000, ExitReason::TakeProfit, 0.0)
            .unwrap();

        // Risk = 0.0050 * 1.0 = 0.0050
        // PnL = 0.0100 * 1.0 = 0.0100
        // R = 0.0100 / 0.0050 = 2.0
        assert_relative_eq!(trade.r_multiple, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_losing_trade() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        // Close at SL
        let trade = portfolio
            .close_position(id, 1.1950, 2_000_000, ExitReason::StopLoss, 0.0)
            .unwrap();

        // PnL = -0.0050
        // R = -0.0050 / 0.0050 = -1.0
        assert_relative_eq!(trade.result, -0.0050, epsilon = 1e-10);
        assert_relative_eq!(trade.r_multiple, -1.0, epsilon = 1e-10);
        assert_relative_eq!(portfolio.cash(), 9_999.995_0, epsilon = 1e-10);
    }

    #[test]
    fn test_equity_update() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        // Update equity with price at 1.2050 (profit)
        portfolio.update_equity(1_500_000, 1.2050);

        // Unrealized PnL = 0.0050
        assert_relative_eq!(portfolio.equity(), 10_000.005_0, epsilon = 1e-10);
    }

    #[test]
    fn test_trade_stats() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        // Win 1: 100 pips
        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();
        portfolio.close_position(id, 1.2100, 2_000_000, ExitReason::TakeProfit, 0.0);

        // Win 2: 50 pips
        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2050);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 3_000_000, 0.0)
            .unwrap();
        portfolio.close_position(id, 1.2050, 4_000_000, ExitReason::TakeProfit, 0.0);

        // Loss: 50 pips
        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 5_000_000, 0.0)
            .unwrap();
        portfolio.close_position(id, 1.1950, 6_000_000, ExitReason::StopLoss, 0.0);

        let stats = portfolio.calculate_stats();

        assert_eq!(stats.total_trades, 3);
        assert_eq!(stats.wins, 2);
        assert_eq!(stats.losses, 1);
        assert_relative_eq!(stats.win_rate, 2.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.gross_profit, 0.0150, epsilon = 1e-10);
        assert_relative_eq!(stats.gross_loss, 0.0050, epsilon = 1e-10);
        assert_relative_eq!(stats.profit_factor, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_balance_consistency() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        // Open and close multiple positions
        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);

        for i in 0..5 {
            let id = portfolio
                .open_position(&signal, 1.2000, 1.0, i * 1_000_000, 2.5)
                .unwrap();
            portfolio.close_position(
                id,
                if i % 2 == 0 { 1.2100 } else { 1.1950 },
                (i + 1) * 1_000_000,
                if i % 2 == 0 {
                    ExitReason::TakeProfit
                } else {
                    ExitReason::StopLoss
                },
                2.5,
            );
        }

        // Verify no positions open
        assert_eq!(portfolio.position_count(), 0);

        // Verify equity matches cash (no unrealized PnL)
        portfolio.update_equity(10_000_000, 1.2000);
        assert_relative_eq!(portfolio.equity(), portfolio.cash(), epsilon = 1e-10);
    }

    #[test]
    fn test_trade_meta_defaults() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        let trade = portfolio
            .close_position(id, 1.1950, 2_000_000, ExitReason::StopLoss, 0.0)
            .unwrap();

        let stop_loss_kind = trade
            .meta
            .get("stop_loss_kind")
            .and_then(|value| value.as_str());
        assert_eq!(stop_loss_kind, Some("initial"));

        let in_entry_candle = trade
            .meta
            .get("in_entry_candle")
            .and_then(serde_json::Value::as_bool);
        assert_eq!(in_entry_candle, Some(false));
    }

    #[test]
    fn test_trade_meta_break_even_kind() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        let trade = portfolio
            .close_position(id, 1.2000, 2_000_000, ExitReason::BreakEvenStopLoss, 0.0)
            .unwrap();

        let stop_loss_kind = trade
            .meta
            .get("stop_loss_kind")
            .and_then(|value| value.as_str());
        assert_eq!(stop_loss_kind, Some("break_even"));
    }

    #[test]
    fn test_consistency_validation_passes() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        portfolio.update_equity(1_500_000, 1.2050);

        let result = portfolio.validate_consistency(1.2050);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consistency_validation_fails_on_mismatch() {
        let mut portfolio = Portfolio::new(10_000.0, 5, "EURUSD");

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        portfolio
            .open_position(&signal, 1.2000, 1.0, 1_000_000, 0.0)
            .unwrap();

        portfolio.update_equity(1_500_000, 1.2050);

        let bad_equity = portfolio.cash + 0.0100;
        portfolio
            .equity_tracker
            .update(2_000_000, bad_equity, portfolio.cash);

        let result = portfolio.validate_consistency(1.2050);
        assert!(matches!(
            result,
            Err(PortfolioError::ConsistencyViolation { .. })
        ));
    }
}
