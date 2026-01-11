//! Entry trigger and exit detection logic.
//!
//! Implements the core logic for:
//! - Checking if pending orders (limit/stop) should be triggered
//! - Detecting stop-loss and take-profit hits
//! - Handling the entry candle special case

use super::position::{Direction, ExitReason, OrderType, Position, PositionStatus};

/// OHLCV candle data for trigger/exit evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Candle {
    /// Candle timestamp in microseconds
    pub timestamp_us: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

impl Candle {
    /// Create a new candle.
    pub const fn new(
        timestamp_us: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp_us,
            open,
            high,
            low,
            close,
            volume,
        }
    }
}

/// Result of entry trigger check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerResult {
    /// Entry was not triggered
    NotTriggered,
    /// Entry was triggered
    Triggered,
}

/// Result of exit evaluation.
#[derive(Clone, Copy, Debug)]
pub enum ExitResult {
    /// Position should remain open
    NoExit,
    /// Position should be closed
    Exit {
        /// Exit price
        price: f64,
        /// Exit reason
        reason: ExitReason,
    },
}

/// Check if a pending entry order should be triggered.
///
/// For limit orders:
/// - Long: triggered when ask.low <= entry_price
/// - Short: triggered when bid.high >= entry_price
///
/// For stop orders:
/// - Long: triggered when ask.high >= entry_price
/// - Short: triggered when bid.low <= entry_price
///
/// # Arguments
/// * `position` - The pending position to check
/// * `bid_candle` - Current bid candle
/// * `ask_candle` - Current ask candle (optional, uses bid if None)
///
/// # Returns
/// `TriggerResult::Triggered` if entry should occur, `TriggerResult::NotTriggered` otherwise
pub fn check_entry_trigger(
    position: &Position,
    bid_candle: &Candle,
    ask_candle: Option<&Candle>,
) -> TriggerResult {
    // Only check pending positions
    if position.status != PositionStatus::Pending {
        return TriggerResult::NotTriggered;
    }

    // Don't trigger on the same candle as the signal
    if bid_candle.timestamp_us <= position.entry_time_us {
        return TriggerResult::NotTriggered;
    }

    match position.order_type {
        OrderType::Limit => {
            match position.direction {
                Direction::Long => {
                    // Long limit: buy when price drops to entry level
                    // Use ask candle if available
                    if let Some(ask) = ask_candle {
                        if ask.low <= position.entry_price {
                            return TriggerResult::Triggered;
                        }
                    }
                }
                Direction::Short => {
                    // Short limit: sell when price rises to entry level
                    // Use bid candle
                    if bid_candle.high >= position.entry_price {
                        return TriggerResult::Triggered;
                    }
                }
            }
        }
        OrderType::Stop => {
            match position.direction {
                Direction::Long => {
                    // Long stop: buy when price breaks above entry level
                    // Use ask candle if available
                    if let Some(ask) = ask_candle {
                        if ask.high >= position.entry_price {
                            return TriggerResult::Triggered;
                        }
                    }
                }
                Direction::Short => {
                    // Short stop: sell when price breaks below entry level
                    // Use bid candle
                    if bid_candle.low <= position.entry_price {
                        return TriggerResult::Triggered;
                    }
                }
            }
        }
        OrderType::Market => {
            // Market orders are triggered immediately during signal processing
            // This should not be reached for pending positions
        }
    }

    TriggerResult::NotTriggered
}

/// Evaluate if an open position should be exited.
///
/// Checks for stop-loss and take-profit hits with the following logic:
///
/// For long positions (using bid candle for exit):
/// - SL hit if bid.low <= stop_loss + pip_buffer
/// - TP hit if bid.high >= take_profit - pip_buffer
///
/// For short positions (using ask candle for exit):
/// - SL hit if ask.high >= stop_loss - pip_buffer
/// - TP hit if ask.low <= take_profit + pip_buffer
///
/// # Entry Candle Special Case
///
/// On the entry candle (same timestamp as trigger), extra validation is applied:
/// - For limit orders with TP hit, verify close price confirms TP level
///
/// # Arguments
/// * `position` - The open position to evaluate
/// * `bid_candle` - Current bid candle
/// * `ask_candle` - Current ask candle (optional, uses bid if None)
/// * `pip_buffer` - Buffer around SL/TP levels (pip_size * buffer_factor)
///
/// # Returns
/// `ExitResult::Exit` with price and reason if position should close,
/// `ExitResult::NoExit` otherwise
pub fn evaluate_exit(
    position: &Position,
    bid_candle: &Candle,
    ask_candle: Option<&Candle>,
    pip_buffer: f64,
) -> ExitResult {
    // Only evaluate open positions
    if position.status != PositionStatus::Open {
        return ExitResult::NoExit;
    }

    // Skip if candle is before entry
    if bid_candle.timestamp_us <= position.entry_time_us {
        return ExitResult::NoExit;
    }

    // Determine if this is the entry candle
    let in_entry_candle = position
        .trigger_time_us
        .map(|t| bid_candle.timestamp_us == t)
        .unwrap_or(false);

    // Check SL/TP based on direction
    let (sl_hit, tp_hit, ref_candle) = match position.direction {
        Direction::Long => {
            // Long positions exit on bid prices
            let sl_hit = bid_candle.low <= position.stop_loss + pip_buffer;
            let tp_hit = bid_candle.high >= position.take_profit - pip_buffer;
            (sl_hit, tp_hit, bid_candle)
        }
        Direction::Short => {
            // Short positions exit on ask prices (fallback to bid)
            let ask = ask_candle.unwrap_or(bid_candle);
            let sl_hit = ask.high >= position.stop_loss - pip_buffer;
            let tp_hit = ask.low <= position.take_profit + pip_buffer;
            (sl_hit, tp_hit, ask)
        }
    };

    if in_entry_candle {
        // Entry candle special handling
        if sl_hit {
            return ExitResult::Exit {
                price: position.stop_loss,
                reason: ExitReason::StopLoss,
            };
        } else if tp_hit {
            // For limit orders on entry candle, validate close price
            if position.order_type == OrderType::Limit {
                let close_price = match position.direction {
                    Direction::Long => bid_candle.close,
                    Direction::Short => ref_candle.close,
                };

                let tp_confirmed = match position.direction {
                    Direction::Long => close_price > position.take_profit,
                    Direction::Short => close_price < position.take_profit,
                };

                if tp_confirmed {
                    return ExitResult::Exit {
                        price: position.take_profit,
                        reason: ExitReason::TakeProfit,
                    };
                } else {
                    return ExitResult::NoExit;
                }
            } else {
                return ExitResult::Exit {
                    price: position.take_profit,
                    reason: ExitReason::TakeProfit,
                };
            }
        }
    } else {
        // Normal candle (not entry candle)
        if sl_hit {
            // Check if SL was moved (break-even)
            let reason = if (position.stop_loss - position.initial_stop_loss).abs() < 1e-10 {
                ExitReason::StopLoss
            } else {
                ExitReason::BreakEvenStopLoss
            };
            return ExitResult::Exit {
                price: position.stop_loss,
                reason,
            };
        } else if tp_hit {
            return ExitResult::Exit {
                price: position.take_profit,
                reason: ExitReason::TakeProfit,
            };
        }
    }

    ExitResult::NoExit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pending_long_limit() -> Position {
        Position::new_pending(
            1,
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Limit,
            1.1000,
            1.0950,
            1.1100,
            100.0,
        )
    }

    fn make_pending_short_stop() -> Position {
        Position::new_pending(
            2,
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Short,
            OrderType::Stop,
            1.1000,
            1.1050,
            1.0900,
            100.0,
        )
    }

    #[test]
    fn test_trigger_long_limit_not_triggered() {
        let pos = make_pending_long_limit();
        let bid = Candle::new(1704067260_000_000, 1.1010, 1.1020, 1.1005, 1.1015, 100.0);
        let ask = Candle::new(1704067260_000_000, 1.1011, 1.1021, 1.1006, 1.1016, 100.0);

        let result = check_entry_trigger(&pos, &bid, Some(&ask));
        assert_eq!(result, TriggerResult::NotTriggered);
    }

    #[test]
    fn test_trigger_long_limit_triggered() {
        let pos = make_pending_long_limit();
        let bid = Candle::new(1704067260_000_000, 1.1010, 1.1020, 1.0995, 1.1015, 100.0);
        let ask = Candle::new(1704067260_000_000, 1.1011, 1.1021, 1.0996, 1.1016, 100.0);

        // Ask low (1.0996) <= entry (1.1000)
        let result = check_entry_trigger(&pos, &bid, Some(&ask));
        assert_eq!(result, TriggerResult::Triggered);
    }

    #[test]
    fn test_trigger_short_stop_triggered() {
        let pos = make_pending_short_stop();
        let bid = Candle::new(1704067260_000_000, 1.1010, 1.1020, 1.0995, 1.1000, 100.0);

        // Bid low (1.0995) <= entry (1.1000)
        let result = check_entry_trigger(&pos, &bid, None);
        assert_eq!(result, TriggerResult::Triggered);
    }

    #[test]
    fn test_trigger_same_candle_as_signal() {
        let pos = make_pending_long_limit();
        // Same timestamp as entry_time
        let bid = Candle::new(1704067200_000_000, 1.0990, 1.1020, 1.0980, 1.1015, 100.0);

        let result = check_entry_trigger(&pos, &bid, None);
        assert_eq!(result, TriggerResult::NotTriggered);
    }

    #[test]
    fn test_exit_sl_hit_long() {
        let mut pos = make_pending_long_limit();
        pos.activate(1704067260_000_000, 0.1);

        // SL = 1.0950, bid low reaches below SL
        let bid = Candle::new(1704067320_000_000, 1.1000, 1.1010, 1.0940, 1.0960, 100.0);

        let result = evaluate_exit(&pos, &bid, None, 0.0);
        match result {
            ExitResult::Exit { price, reason } => {
                assert!((price - 1.0950).abs() < 1e-10);
                assert_eq!(reason, ExitReason::StopLoss);
            }
            ExitResult::NoExit => panic!("Expected exit"),
        }
    }

    #[test]
    fn test_exit_tp_hit_long() {
        let mut pos = make_pending_long_limit();
        pos.activate(1704067260_000_000, 0.1);

        // TP = 1.1100, bid high reaches above TP
        let bid = Candle::new(1704067320_000_000, 1.1050, 1.1110, 1.1040, 1.1090, 100.0);

        let result = evaluate_exit(&pos, &bid, None, 0.0);
        match result {
            ExitResult::Exit { price, reason } => {
                assert!((price - 1.1100).abs() < 1e-10);
                assert_eq!(reason, ExitReason::TakeProfit);
            }
            ExitResult::NoExit => panic!("Expected exit"),
        }
    }

    #[test]
    fn test_exit_break_even_sl() {
        let mut pos = make_pending_long_limit();
        pos.activate(1704067260_000_000, 0.1);
        // Move SL to entry (break-even)
        pos.stop_loss = 1.1000;

        let bid = Candle::new(1704067320_000_000, 1.1010, 1.1020, 1.0995, 1.1000, 100.0);

        let result = evaluate_exit(&pos, &bid, None, 0.0);
        match result {
            ExitResult::Exit { price, reason } => {
                assert!((price - 1.1000).abs() < 1e-10);
                assert_eq!(reason, ExitReason::BreakEvenStopLoss);
            }
            ExitResult::NoExit => panic!("Expected exit"),
        }
    }

    #[test]
    fn test_no_exit_when_candle_before_entry() {
        let mut pos = make_pending_long_limit();
        pos.activate(1704067260_000_000, 0.1);

        // Candle timestamp before entry time
        let bid = Candle::new(1704067200_000_000, 1.0900, 1.1200, 1.0800, 1.1000, 100.0);

        let result = evaluate_exit(&pos, &bid, None, 0.0);
        assert!(matches!(result, ExitResult::NoExit));
    }
}
