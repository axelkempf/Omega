//! Equity tracking and drawdown calculation.
//!
//! Tracks portfolio equity over time and calculates drawdown metrics.

use omega_types::EquityPoint;
use serde::{Deserialize, Serialize};

/// Tracks equity changes over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityTracker {
    /// Initial balance
    initial_balance: f64,
    /// High-water mark (peak equity)
    high_water_mark: f64,
    /// Maximum drawdown percentage (0-1)
    max_drawdown: f64,
    /// Maximum drawdown in absolute terms
    max_drawdown_abs: f64,
    /// Current equity
    current_equity: f64,
    /// Current balance (realized)
    current_balance: f64,
    /// Equity curve history
    equity_curve: Vec<EquityPoint>,
}

impl EquityTracker {
    /// Creates a new equity tracker with the given initial balance.
    pub fn new(initial_balance: f64) -> Self {
        Self {
            initial_balance,
            high_water_mark: initial_balance,
            max_drawdown: 0.0,
            max_drawdown_abs: 0.0,
            current_equity: initial_balance,
            current_balance: initial_balance,
            equity_curve: Vec::new(),
        }
    }

    /// Updates the equity tracker with new values.
    ///
    /// # Arguments
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    /// * `equity` - Current equity (balance + unrealized PnL)
    /// * `balance` - Current balance (realized funds)
    pub fn update(&mut self, timestamp_ns: i64, equity: f64, balance: f64) {
        self.current_equity = equity;
        self.current_balance = balance;

        // Update high-water mark
        if equity > self.high_water_mark {
            self.high_water_mark = equity;
        }

        // Calculate drawdown
        let drawdown_abs = self.high_water_mark - equity;
        let drawdown = if self.high_water_mark > 0.0 {
            drawdown_abs / self.high_water_mark
        } else {
            0.0
        };

        // Update max drawdown
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
            self.max_drawdown_abs = drawdown_abs;
        }

        // Record equity point
        self.equity_curve.push(EquityPoint {
            timestamp_ns,
            equity,
            balance,
            drawdown,
            high_water: self.high_water_mark,
        });
    }

    /// Returns the initial balance.
    pub fn initial_balance(&self) -> f64 {
        self.initial_balance
    }

    /// Returns the current equity.
    pub fn equity(&self) -> f64 {
        self.current_equity
    }

    /// Returns the current balance.
    pub fn balance(&self) -> f64 {
        self.current_balance
    }

    /// Returns the high-water mark.
    pub fn high_water_mark(&self) -> f64 {
        self.high_water_mark
    }

    /// Returns the maximum drawdown as a percentage (0-1).
    pub fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    /// Returns the maximum drawdown in absolute terms.
    pub fn max_drawdown_abs(&self) -> f64 {
        self.max_drawdown_abs
    }

    /// Returns the current drawdown as a percentage (0-1).
    pub fn current_drawdown(&self) -> f64 {
        if self.high_water_mark > 0.0 {
            (self.high_water_mark - self.current_equity) / self.high_water_mark
        } else {
            0.0
        }
    }

    /// Returns the current drawdown in absolute terms.
    pub fn current_drawdown_abs(&self) -> f64 {
        self.high_water_mark - self.current_equity
    }

    /// Returns the total return as a percentage.
    pub fn total_return(&self) -> f64 {
        if self.initial_balance > 0.0 {
            (self.current_equity - self.initial_balance) / self.initial_balance
        } else {
            0.0
        }
    }

    /// Returns the total return in absolute terms.
    pub fn total_return_abs(&self) -> f64 {
        self.current_equity - self.initial_balance
    }

    /// Returns the equity curve history.
    pub fn equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    /// Consumes the tracker and returns the equity curve.
    pub fn into_equity_curve(self) -> Vec<EquityPoint> {
        self.equity_curve
    }

    /// Returns the number of equity points recorded.
    pub fn len(&self) -> usize {
        self.equity_curve.len()
    }

    /// Checks if the equity curve is empty.
    pub fn is_empty(&self) -> bool {
        self.equity_curve.is_empty()
    }

    /// Clears the equity curve history (keeps current state).
    pub fn clear_history(&mut self) {
        self.equity_curve.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_initial_state() {
        let tracker = EquityTracker::new(10_000.0);

        assert_relative_eq!(tracker.initial_balance(), 10_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.equity(), 10_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.balance(), 10_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.high_water_mark(), 10_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.max_drawdown(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_equity_increase() {
        let mut tracker = EquityTracker::new(10_000.0);

        tracker.update(1_000_000, 11_000.0, 11_000.0);

        assert_relative_eq!(tracker.equity(), 11_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.high_water_mark(), 11_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.max_drawdown(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.total_return(), 0.10, epsilon = 1e-10);
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut tracker = EquityTracker::new(10_000.0);

        // Equity rises to 12,000
        tracker.update(1_000_000, 12_000.0, 12_000.0);
        assert_relative_eq!(tracker.high_water_mark(), 12_000.0, epsilon = 1e-10);

        // Equity drops to 10,800 (10% drawdown)
        tracker.update(2_000_000, 10_800.0, 10_800.0);
        assert_relative_eq!(tracker.current_drawdown(), 0.10, epsilon = 1e-10);
        assert_relative_eq!(tracker.max_drawdown(), 0.10, epsilon = 1e-10);
        assert_relative_eq!(tracker.max_drawdown_abs(), 1_200.0, epsilon = 1e-10);

        // Equity recovers to 11,500 (still in drawdown)
        tracker.update(3_000_000, 11_500.0, 11_500.0);
        assert_relative_eq!(tracker.high_water_mark(), 12_000.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.current_drawdown_abs(), 500.0, epsilon = 1e-10);

        // Max drawdown unchanged
        assert_relative_eq!(tracker.max_drawdown(), 0.10, epsilon = 1e-10);
    }

    #[test]
    fn test_equity_curve() {
        let mut tracker = EquityTracker::new(10_000.0);

        tracker.update(1_000_000, 10_500.0, 10_500.0);
        tracker.update(2_000_000, 10_200.0, 10_200.0);
        tracker.update(3_000_000, 11_000.0, 11_000.0);

        assert_eq!(tracker.len(), 3);

        let curve = tracker.equity_curve();
        assert_eq!(curve[0].timestamp_ns, 1_000_000);
        assert_relative_eq!(curve[0].equity, 10_500.0, epsilon = 1e-10);
        assert_eq!(curve[2].timestamp_ns, 3_000_000);
        assert_relative_eq!(curve[2].equity, 11_000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut tracker = EquityTracker::new(10_000.0);

        // Balance 10,000 but unrealized profit of 500
        tracker.update(1_000_000, 10_500.0, 10_000.0);

        assert_relative_eq!(tracker.equity(), 10_500.0, epsilon = 1e-10);
        assert_relative_eq!(tracker.balance(), 10_000.0, epsilon = 1e-10);
    }
}
