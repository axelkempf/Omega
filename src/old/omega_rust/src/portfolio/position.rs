//! Position representation for trading positions.
//!
//! Provides the `PositionRust` struct which represents a single trading position
//! in the portfolio, with all necessary fields for tracking entry, exit,
//! stop-loss, take-profit, and result calculation.

use pyo3::prelude::*;

/// Direction indicator for trade.
/// 1 = Long (buy), -1 = Short (sell)
pub type Direction = i8;

/// Long direction constant (buy)
pub const DIRECTION_LONG: Direction = 1;
/// Short direction constant (sell)
pub const DIRECTION_SHORT: Direction = -1;

/// Represents a single trading position in the portfolio.
///
/// This struct mirrors the Python `PortfolioPosition` dataclass and provides
/// all necessary fields for position tracking and R-multiple calculation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PositionRust {
    /// Entry timestamp in microseconds (Unix epoch)
    #[pyo3(get, set)]
    pub entry_time: i64,
    /// Exit timestamp in microseconds (None if still open)
    #[pyo3(get, set)]
    pub exit_time: Option<i64>,
    /// Trade direction: 1 = long, -1 = short
    #[pyo3(get, set)]
    pub direction: i8,
    /// Trading symbol (e.g., "EURUSD")
    #[pyo3(get, set)]
    pub symbol: String,
    /// Entry price
    #[pyo3(get, set)]
    pub entry_price: f64,
    /// Exit price (None if still open)
    #[pyo3(get, set)]
    pub exit_price: Option<f64>,
    /// Current stop-loss price
    #[pyo3(get, set)]
    pub stop_loss: f64,
    /// Current take-profit price
    #[pyo3(get, set)]
    pub take_profit: f64,
    /// Position size in lots
    #[pyo3(get, set)]
    pub size: f64,
    /// Risk amount per trade in account currency
    #[pyo3(get, set)]
    pub risk_per_trade: f64,
    /// Initial stop-loss price (for R-multiple calculation)
    #[pyo3(get, set)]
    pub initial_stop_loss: Option<f64>,
    /// Initial take-profit price
    #[pyo3(get, set)]
    pub initial_take_profit: Option<f64>,
    /// Trade result in account currency (None if still open)
    #[pyo3(get, set)]
    pub result: Option<f64>,
    /// Exit reason (e.g., `stop_loss`, `take_profit`, `signal`)
    #[pyo3(get, set)]
    pub reason: Option<String>,
    /// Whether the position is closed
    #[pyo3(get, set)]
    pub is_closed: bool,
    /// Order type (e.g., "market", "limit")
    #[pyo3(get, set)]
    pub order_type: String,
    /// Position status (e.g., "open", "closed", "pending")
    #[pyo3(get, set)]
    pub status: String,
    /// Entry fee amount
    #[pyo3(get, set)]
    pub entry_fee: f64,
    /// Exit fee amount
    #[pyo3(get, set)]
    pub exit_fee: f64,
}

#[pymethods]
impl PositionRust {
    /// Create a new position.
    ///
    /// # Arguments
    /// * `entry_time` - Entry timestamp in microseconds
    /// * `direction` - Trade direction: 1 = long, -1 = short
    /// * `symbol` - Trading symbol
    /// * `entry_price` - Entry price
    /// * `stop_loss` - Stop-loss price
    /// * `take_profit` - Take-profit price
    /// * `size` - Position size in lots
    /// * `risk_per_trade` - Risk amount per trade (default: 100.0)
    #[new]
    #[pyo3(signature = (entry_time, direction, symbol, entry_price, stop_loss, take_profit, size, risk_per_trade=100.0))]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        entry_time: i64,
        direction: i8,
        symbol: String,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        size: f64,
        risk_per_trade: f64,
    ) -> Self {
        Self {
            entry_time,
            exit_time: None,
            direction,
            symbol,
            entry_price,
            exit_price: None,
            stop_loss,
            take_profit,
            size,
            risk_per_trade,
            initial_stop_loss: Some(stop_loss),
            initial_take_profit: Some(take_profit),
            result: None,
            reason: None,
            is_closed: false,
            order_type: "market".to_string(),
            status: "open".to_string(),
            entry_fee: 0.0,
            exit_fee: 0.0,
        }
    }

    /// Close the position and calculate the result based on R-multiple.
    ///
    /// # Arguments
    /// * `time` - Exit timestamp in microseconds
    /// * `price` - Exit price
    /// * `reason` - Exit reason
    pub fn close(&mut self, time: i64, price: f64, reason: String) {
        self.exit_time = Some(time);
        self.exit_price = Some(price);
        self.reason = Some(reason);
        self.is_closed = true;
        self.status = "closed".to_string();

        let initial_sl = self.initial_stop_loss.unwrap_or(self.stop_loss);
        let risk = (self.entry_price - initial_sl).abs();

        if risk > 0.0 {
            let reward = if self.direction == DIRECTION_LONG {
                price - self.entry_price
            } else {
                self.entry_price - price
            };
            let r_multiple = reward / risk;
            self.result = Some(r_multiple * self.risk_per_trade);
        } else {
            self.result = Some(0.0);
        }
    }

    /// Calculate R-multiple for the position.
    ///
    /// Returns 0.0 if the position is still open or if risk is zero.
    #[getter]
    #[must_use]
    pub fn r_multiple(&self) -> f64 {
        let Some(exit_price) = self.exit_price else {
            return 0.0;
        };

        let initial_sl = self.initial_stop_loss.unwrap_or(self.stop_loss);
        let risk = (self.entry_price - initial_sl).abs();

        if risk == 0.0 {
            return 0.0;
        }

        if self.direction == DIRECTION_LONG {
            (exit_price - self.entry_price) / risk
        } else {
            (self.entry_price - exit_price) / risk
        }
    }

    /// Get total fees for this position (entry + exit).
    #[getter]
    #[must_use]
    pub fn total_fees(&self) -> f64 {
        self.entry_fee + self.exit_fee
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        let direction_str = if self.direction == DIRECTION_LONG {
            "long"
        } else {
            "short"
        };
        let status = if self.is_closed { "closed" } else { "open" };
        format!(
            "PositionRust({} {} {} @ {} [{}])",
            self.symbol, direction_str, self.size, self.entry_price, status
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_position_new() {
        let pos = PositionRust::new(
            1_704_067_200_000_000, // 2024-01-01 00:00:00 UTC
            DIRECTION_LONG,
            "EURUSD".to_string(),
            1.10000,
            1.09900, // SL 10 pips below
            1.10200, // TP 20 pips above
            1.0,
            100.0,
        );

        assert_eq!(pos.entry_time, 1_704_067_200_000_000);
        assert_eq!(pos.direction, DIRECTION_LONG);
        assert_eq!(pos.symbol, "EURUSD");
        assert_relative_eq!(pos.entry_price, 1.10000, epsilon = 1e-8);
        assert_relative_eq!(pos.stop_loss, 1.09900, epsilon = 1e-8);
        assert_relative_eq!(pos.take_profit, 1.10200, epsilon = 1e-8);
        assert_relative_eq!(pos.size, 1.0, epsilon = 1e-8);
        assert_relative_eq!(pos.risk_per_trade, 100.0, epsilon = 1e-8);
        assert!(!pos.is_closed);
        assert_eq!(pos.status, "open");
        assert!(pos.exit_time.is_none());
        assert!(pos.exit_price.is_none());
        assert!(pos.result.is_none());
    }

    #[test]
    fn test_position_close_long_win() {
        let mut pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD".to_string(),
            1.10000,
            1.09900, // SL 10 pips below (risk = 0.001)
            1.10200,
            1.0,
            100.0, // Risk $100
        );

        // Close at TP (20 pips profit = 2R)
        pos.close(1_704_067_260_000_000, 1.10200, "take_profit".to_string());

        assert!(pos.is_closed);
        assert_eq!(pos.status, "closed");
        assert_eq!(pos.exit_time, Some(1_704_067_260_000_000));
        assert_eq!(pos.exit_price, Some(1.10200));
        assert_eq!(pos.reason, Some("take_profit".to_string()));

        // R-multiple should be 2.0 (20 pips profit / 10 pips risk)
        assert_relative_eq!(pos.r_multiple(), 2.0, epsilon = 1e-8);
        // Result should be 2.0 * 100 = 200
        assert_relative_eq!(pos.result.unwrap(), 200.0, epsilon = 1e-8);
    }

    #[test]
    fn test_position_close_long_loss() {
        let mut pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD".to_string(),
            1.10000,
            1.09900, // SL 10 pips below
            1.10200,
            1.0,
            100.0,
        );

        // Close at SL (10 pips loss = -1R)
        pos.close(1_704_067_260_000_000, 1.09900, "stop_loss".to_string());

        assert_relative_eq!(pos.r_multiple(), -1.0, epsilon = 1e-8);
        assert_relative_eq!(pos.result.unwrap(), -100.0, epsilon = 1e-8);
    }

    #[test]
    fn test_position_close_short_win() {
        let mut pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_SHORT,
            "EURUSD".to_string(),
            1.10000,
            1.10100, // SL 10 pips above (risk = 0.001)
            1.09800, // TP 20 pips below
            1.0,
            100.0,
        );

        // Close at TP (20 pips profit = 2R)
        pos.close(1_704_067_260_000_000, 1.09800, "take_profit".to_string());

        assert_relative_eq!(pos.r_multiple(), 2.0, epsilon = 1e-8);
        assert_relative_eq!(pos.result.unwrap(), 200.0, epsilon = 1e-8);
    }

    #[test]
    fn test_position_close_short_loss() {
        let mut pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_SHORT,
            "EURUSD".to_string(),
            1.10000,
            1.10100, // SL 10 pips above
            1.09800,
            1.0,
            100.0,
        );

        // Close at SL (-1R)
        pos.close(1_704_067_260_000_000, 1.10100, "stop_loss".to_string());

        assert_relative_eq!(pos.r_multiple(), -1.0, epsilon = 1e-8);
        assert_relative_eq!(pos.result.unwrap(), -100.0, epsilon = 1e-8);
    }

    #[test]
    fn test_r_multiple_open_position() {
        let pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD".to_string(),
            1.10000,
            1.09900,
            1.10200,
            1.0,
            100.0,
        );

        // R-multiple should be 0.0 for open position
        assert_relative_eq!(pos.r_multiple(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_total_fees() {
        let mut pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD".to_string(),
            1.10000,
            1.09900,
            1.10200,
            1.0,
            100.0,
        );

        pos.entry_fee = 3.0;
        pos.exit_fee = 3.0;

        assert_relative_eq!(pos.total_fees(), 6.0, epsilon = 1e-8);
    }

    #[test]
    fn test_position_repr() {
        let pos = PositionRust::new(
            1_704_067_200_000_000,
            DIRECTION_LONG,
            "EURUSD".to_string(),
            1.10000,
            1.09900,
            1.10200,
            1.0,
            100.0,
        );

        let repr = pos.__repr__();
        assert!(repr.contains("EURUSD"));
        assert!(repr.contains("long"));
        assert!(repr.contains("open"));
    }
}
