//! Position representation and state machine.
//!
//! Provides the [`Position`] struct for tracking trading positions
//! through their lifecycle: pending → open → closed.

use crate::error::{OmegaError, Result};
use serde::{Deserialize, Serialize};

/// Trade direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
}

impl Direction {
    /// Convert from string representation.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "long" | "buy" | "1" => Ok(Self::Long),
            "short" | "sell" | "-1" => Ok(Self::Short),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!("Invalid direction: '{s}', expected 'long' or 'short'"),
            }),
        }
    }

    /// Convert from numeric representation (1 = long, -1 = short).
    pub fn from_i8(v: i8) -> Result<Self> {
        match v {
            1 => Ok(Self::Long),
            -1 => Ok(Self::Short),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!("Invalid direction value: {v}, expected 1 or -1"),
            }),
        }
    }

    /// Convert to numeric representation (1 = long, -1 = short).
    #[inline]
    pub const fn as_i8(self) -> i8 {
        match self {
            Self::Long => 1,
            Self::Short => -1,
        }
    }

    /// Convert to string representation.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Long => "long",
            Self::Short => "short",
        }
    }
}

/// Order type for trade entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order - immediate execution at current price
    Market,
    /// Limit order - execute when price reaches limit level
    Limit,
    /// Stop order - execute when price breaks through stop level
    Stop,
}

impl OrderType {
    /// Convert from string representation.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "market" => Ok(Self::Market),
            "limit" => Ok(Self::Limit),
            "stop" => Ok(Self::Stop),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!("Invalid order type: '{s}', expected 'market', 'limit', or 'stop'"),
            }),
        }
    }

    /// Convert to string representation.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Market => "market",
            Self::Limit => "limit",
            Self::Stop => "stop",
        }
    }
}

/// Position status in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionStatus {
    /// Pending - waiting for entry trigger (limit/stop orders)
    Pending,
    /// Open - active position in the market
    Open,
    /// Closed - position has been closed
    Closed,
}

impl PositionStatus {
    /// Convert from string representation.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "open" => Ok(Self::Open),
            "closed" => Ok(Self::Closed),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!(
                    "Invalid position status: '{s}', expected 'pending', 'open', or 'closed'"
                ),
            }),
        }
    }

    /// Convert to string representation.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Open => "open",
            Self::Closed => "closed",
        }
    }
}

/// Exit reason for closed positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitReason {
    /// Stop-loss hit
    StopLoss,
    /// Take-profit hit
    TakeProfit,
    /// Break-even stop-loss (SL moved to entry)
    BreakEvenStopLoss,
    /// Manual signal exit
    Signal,
    /// Timeout/expiry
    Timeout,
}

impl ExitReason {
    /// Convert to string representation (parity with Python).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StopLoss => "stop_loss",
            Self::TakeProfit => "take_profit",
            Self::BreakEvenStopLoss => "break_even_stop_loss",
            Self::Signal => "signal",
            Self::Timeout => "timeout",
        }
    }

    /// Convert from string representation.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "stop_loss" | "stoploss" | "sl" => Ok(Self::StopLoss),
            "take_profit" | "takeprofit" | "tp" => Ok(Self::TakeProfit),
            "break_even_stop_loss" | "breakeven" | "be" => Ok(Self::BreakEvenStopLoss),
            "signal" => Ok(Self::Signal),
            "timeout" | "expiry" => Ok(Self::Timeout),
            _ => Err(OmegaError::InvalidParameter {
                reason: format!("Invalid exit reason: '{s}'"),
            }),
        }
    }
}

/// A trading position in the execution simulator.
///
/// Tracks the full lifecycle of a position from signal to close,
/// including entry/exit prices, stop-loss, take-profit, and fees.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    /// Unique position ID (monotonically increasing)
    pub id: u64,

    /// Entry time (signal timestamp) in microseconds
    pub entry_time_us: i64,

    /// Trigger time (when pending order was activated) in microseconds
    pub trigger_time_us: Option<i64>,

    /// Exit time in microseconds (None if still open/pending)
    pub exit_time_us: Option<i64>,

    /// Trade direction
    pub direction: Direction,

    /// Trading symbol
    pub symbol: String,

    /// Order type
    pub order_type: OrderType,

    /// Position status
    pub status: PositionStatus,

    /// Entry price (requested for pending, actual for open)
    pub entry_price: f64,

    /// Exit price (None if not closed)
    pub exit_price: Option<f64>,

    /// Current stop-loss price
    pub stop_loss: f64,

    /// Current take-profit price
    pub take_profit: f64,

    /// Initial stop-loss price (for R-multiple calculation)
    pub initial_stop_loss: f64,

    /// Initial take-profit price
    pub initial_take_profit: f64,

    /// Position size in lots
    pub size: f64,

    /// Risk amount per trade in account currency
    pub risk_per_trade: f64,

    /// Trade result in account currency (None if open)
    pub result: Option<f64>,

    /// Exit reason (None if still open)
    pub exit_reason: Option<ExitReason>,

    /// Entry fee amount
    pub entry_fee: f64,

    /// Exit fee amount
    pub exit_fee: f64,

    /// Additional metadata as JSON string
    pub metadata_json: Option<String>,
}

impl Position {
    /// Create a new pending position from a signal.
    #[allow(clippy::too_many_arguments)]
    pub fn new_pending(
        id: u64,
        entry_time_us: i64,
        symbol: String,
        direction: Direction,
        order_type: OrderType,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        risk_per_trade: f64,
    ) -> Self {
        Self {
            id,
            entry_time_us,
            trigger_time_us: None,
            exit_time_us: None,
            direction,
            symbol,
            order_type,
            status: if order_type == OrderType::Market {
                PositionStatus::Open
            } else {
                PositionStatus::Pending
            },
            entry_price,
            exit_price: None,
            stop_loss,
            take_profit,
            initial_stop_loss: stop_loss,
            initial_take_profit: take_profit,
            size: 0.0, // Set after sizing calculation
            risk_per_trade,
            result: None,
            exit_reason: None,
            entry_fee: 0.0,
            exit_fee: 0.0,
            metadata_json: None,
        }
    }

    /// Activate a pending position (set to open).
    pub fn activate(&mut self, trigger_time_us: i64, size: f64) {
        debug_assert_eq!(
            self.status,
            PositionStatus::Pending,
            "Can only activate pending positions"
        );
        self.status = PositionStatus::Open;
        self.trigger_time_us = Some(trigger_time_us);
        self.size = size;
    }

    /// Close the position.
    pub fn close(&mut self, exit_time_us: i64, exit_price: f64, reason: ExitReason) {
        debug_assert_eq!(
            self.status,
            PositionStatus::Open,
            "Can only close open positions"
        );
        self.status = PositionStatus::Closed;
        self.exit_time_us = Some(exit_time_us);
        self.exit_price = Some(exit_price);
        self.exit_reason = Some(reason);
    }

    /// Check if the position is closed.
    #[inline]
    pub const fn is_closed(&self) -> bool {
        matches!(self.status, PositionStatus::Closed)
    }

    /// Check if the position is open (active in market).
    #[inline]
    pub const fn is_open(&self) -> bool {
        matches!(self.status, PositionStatus::Open)
    }

    /// Check if the position is pending (waiting for trigger).
    #[inline]
    pub const fn is_pending(&self) -> bool {
        matches!(self.status, PositionStatus::Pending)
    }

    /// Calculate unrealized P&L at a given price.
    pub fn unrealized_pnl(&self, current_price: f64, unit_value: f64) -> f64 {
        let price_diff = match self.direction {
            Direction::Long => current_price - self.entry_price,
            Direction::Short => self.entry_price - current_price,
        };
        price_diff * self.size * unit_value
    }

    /// Calculate realized P&L (only valid for closed positions).
    pub fn realized_pnl(&self, unit_value: f64) -> Option<f64> {
        self.exit_price.map(|exit| {
            let price_diff = match self.direction {
                Direction::Long => exit - self.entry_price,
                Direction::Short => self.entry_price - exit,
            };
            price_diff * self.size * unit_value - self.entry_fee - self.exit_fee
        })
    }

    /// Calculate R-multiple (result / initial risk).
    pub fn r_multiple(&self, unit_value: f64) -> Option<f64> {
        let pnl = self.realized_pnl(unit_value)?;
        let initial_risk = (self.entry_price - self.initial_stop_loss).abs() * self.size * unit_value;
        if initial_risk < 1e-10 {
            return None;
        }
        Some(pnl / initial_risk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_from_str() {
        assert_eq!(Direction::from_str("long").unwrap(), Direction::Long);
        assert_eq!(Direction::from_str("LONG").unwrap(), Direction::Long);
        assert_eq!(Direction::from_str("short").unwrap(), Direction::Short);
        assert_eq!(Direction::from_str("1").unwrap(), Direction::Long);
        assert_eq!(Direction::from_str("-1").unwrap(), Direction::Short);
        assert!(Direction::from_str("invalid").is_err());
    }

    #[test]
    fn test_order_type_from_str() {
        assert_eq!(OrderType::from_str("market").unwrap(), OrderType::Market);
        assert_eq!(OrderType::from_str("limit").unwrap(), OrderType::Limit);
        assert_eq!(OrderType::from_str("stop").unwrap(), OrderType::Stop);
        assert!(OrderType::from_str("invalid").is_err());
    }

    #[test]
    fn test_position_lifecycle() {
        let mut pos = Position::new_pending(
            1,
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Limit,
            1.1000,
            1.0950,
            1.1100,
            100.0,
        );

        assert!(pos.is_pending());
        assert!(!pos.is_open());
        assert!(!pos.is_closed());

        pos.activate(1704067260_000_000, 0.1);
        assert!(!pos.is_pending());
        assert!(pos.is_open());
        assert!(!pos.is_closed());

        pos.close(1704067320_000_000, 1.1100, ExitReason::TakeProfit);
        assert!(!pos.is_pending());
        assert!(!pos.is_open());
        assert!(pos.is_closed());
        assert_eq!(pos.exit_reason, Some(ExitReason::TakeProfit));
    }

    #[test]
    fn test_market_order_opens_immediately() {
        let pos = Position::new_pending(
            1,
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
            100.0,
        );

        // Market orders start as open, not pending
        assert!(pos.is_open());
    }
}
