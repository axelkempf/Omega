//! Trade signal representation and parsing.
//!
//! Provides the [`TradeSignal`] struct for representing trade signals
//! from strategy evaluation, including entry price, stop-loss, take-profit,
//! direction, and order type.

use crate::error::{OmegaError, Result};
use serde::{Deserialize, Serialize};

use super::position::{Direction, OrderType};

/// A trade signal from strategy evaluation.
///
/// Represents a request to enter a trade with specified parameters.
/// Signals are processed by the execution simulator to create positions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeSignal {
    /// Signal timestamp in microseconds (Unix epoch UTC)
    pub timestamp_us: i64,

    /// Trading symbol (e.g., "EURUSD")
    pub symbol: String,

    /// Trade direction (Long or Short)
    pub direction: Direction,

    /// Order type (Market, Limit, or Stop)
    pub order_type: OrderType,

    /// Entry price
    pub entry_price: f64,

    /// Stop-loss price
    pub stop_loss: f64,

    /// Take-profit price
    pub take_profit: f64,

    /// Position size in lots (0.0 for pending orders)
    pub size: f64,

    /// Signal reason/description (optional)
    pub reason: Option<String>,

    /// Scenario tag for analysis (optional)
    pub scenario: Option<String>,

    /// Additional metadata as JSON string (optional)
    pub metadata_json: Option<String>,
}

impl TradeSignal {
    /// Create a new trade signal.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp_us: i64,
        symbol: String,
        direction: Direction,
        order_type: OrderType,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
    ) -> Self {
        Self {
            timestamp_us,
            symbol,
            direction,
            order_type,
            entry_price,
            stop_loss,
            take_profit,
            size: 0.0,
            reason: None,
            scenario: None,
            metadata_json: None,
        }
    }

    /// Validate the trade signal for basic constraints.
    ///
    /// Checks:
    /// - Entry price is finite and positive
    /// - Stop-loss is finite and positive
    /// - Take-profit is finite and positive
    /// - SL distance is meaningful (not zero)
    /// - Direction matches SL/TP placement
    pub fn validate(&self) -> Result<()> {
        // Entry price validation
        if !self.entry_price.is_finite() || self.entry_price <= 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!(
                    "Entry price must be finite and positive, got {}",
                    self.entry_price
                ),
            });
        }

        // Stop-loss validation
        if !self.stop_loss.is_finite() || self.stop_loss <= 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!(
                    "Stop-loss must be finite and positive, got {}",
                    self.stop_loss
                ),
            });
        }

        // Take-profit validation
        if !self.take_profit.is_finite() || self.take_profit <= 0.0 {
            return Err(OmegaError::InvalidParameter {
                reason: format!(
                    "Take-profit must be finite and positive, got {}",
                    self.take_profit
                ),
            });
        }

        // SL distance validation (minimum threshold)
        let sl_distance = (self.entry_price - self.stop_loss).abs();
        if sl_distance < 1e-10 {
            return Err(OmegaError::InvalidParameter {
                reason: format!("SL distance too small: {sl_distance:.2e}"),
            });
        }

        // Direction/SL/TP placement validation
        match self.direction {
            Direction::Long => {
                if self.stop_loss >= self.entry_price {
                    return Err(OmegaError::InvalidParameter {
                        reason: format!(
                            "Long: SL ({}) must be below entry ({})",
                            self.stop_loss, self.entry_price
                        ),
                    });
                }
                if self.take_profit <= self.entry_price {
                    return Err(OmegaError::InvalidParameter {
                        reason: format!(
                            "Long: TP ({}) must be above entry ({})",
                            self.take_profit, self.entry_price
                        ),
                    });
                }
            }
            Direction::Short => {
                if self.stop_loss <= self.entry_price {
                    return Err(OmegaError::InvalidParameter {
                        reason: format!(
                            "Short: SL ({}) must be above entry ({})",
                            self.stop_loss, self.entry_price
                        ),
                    });
                }
                if self.take_profit >= self.entry_price {
                    return Err(OmegaError::InvalidParameter {
                        reason: format!(
                            "Short: TP ({}) must be below entry ({})",
                            self.take_profit, self.entry_price
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Calculate the SL distance in price units.
    #[inline]
    pub fn sl_distance(&self) -> f64 {
        (self.entry_price - self.stop_loss).abs()
    }

    /// Calculate the TP distance in price units.
    #[inline]
    pub fn tp_distance(&self) -> f64 {
        (self.take_profit - self.entry_price).abs()
    }

    /// Calculate the risk/reward ratio.
    pub fn risk_reward_ratio(&self) -> f64 {
        let sl_dist = self.sl_distance();
        if sl_dist < 1e-10 {
            return 0.0;
        }
        self.tp_distance() / sl_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );

        assert_eq!(signal.symbol, "EURUSD");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.order_type, OrderType::Market);
        assert!((signal.entry_price - 1.1000).abs() < 1e-10);
    }

    #[test]
    fn test_signal_validation_valid_long() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );

        assert!(signal.validate().is_ok());
    }

    #[test]
    fn test_signal_validation_valid_short() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Short,
            OrderType::Market,
            1.1000,
            1.1050,
            1.0900,
        );

        assert!(signal.validate().is_ok());
    }

    #[test]
    fn test_signal_validation_invalid_sl_long() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.1050, // SL above entry for long = invalid
            1.1100,
        );

        assert!(signal.validate().is_err());
    }

    #[test]
    fn test_signal_validation_zero_sl_distance() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.1000, // Same as entry = zero distance
            1.1100,
        );

        assert!(signal.validate().is_err());
    }

    #[test]
    fn test_sl_distance() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );

        assert!((signal.sl_distance() - 0.0050).abs() < 1e-10);
    }

    #[test]
    fn test_risk_reward_ratio() {
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950, // 50 pips SL
            1.1100, // 100 pips TP
        );

        assert!((signal.risk_reward_ratio() - 2.0).abs() < 1e-10);
    }
}
