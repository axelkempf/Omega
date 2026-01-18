//! Stop-loss and take-profit checking.
//!
//! Provides functions for checking if SL/TP levels have been hit,
//! with proper handling of edge cases like entry candle and priority rules.

use omega_types::{Candle, Direction, ExitReason, Position};

/// Default pip buffer factor for SL/TP checks.
///
/// This factor creates a small buffer zone around SL/TP levels
/// to account for spread and slippage.
pub const DEFAULT_PIP_BUFFER_FACTOR: f64 = 0.5;

/// Result of a stop check.
#[derive(Debug, Clone)]
pub struct StopCheckResult {
    /// Whether a stop was triggered
    pub triggered: bool,
    /// Exit reason (`StopLoss` or `TakeProfit`)
    pub reason: ExitReason,
    /// Exit price (at the SL or TP level)
    pub exit_price: f64,
}

/// Checks if a position's SL or TP has been hit.
///
/// # Priority Rules
///
/// **SL has priority over TP in the same candle.**
///
/// When both SL and TP are hit in the same candle, the SL is executed
/// because risk management takes precedence over profit-taking.
///
/// # Order Type Differentiation (The 4 Rules)
///
/// 1. **Market Orders**: SL/TP can only trigger in the NEXT candle after entry.
///    In the entry candle (`in_entry_candle = true`), no exit is allowed.
///
/// 2. **Pending Orders (Limit/Stop)**: Can trigger SL/TP in the SAME candle
///    where the pending order was triggered (not where it was placed).
///    This is because the pending order already waited for trigger.
///
/// # Entry Candle Rule (for Pending Orders only)
///
/// When checking in the entry candle for pending orders, TP is only
/// valid if the candle's close price is beyond the TP level. This prevents
/// false TP triggers from intra-candle price spikes.
///
/// # Arguments
/// * `position` - The position to check (includes `order_type`)
/// * `bid` - Current bid candle (used for long position exits)
/// * `ask` - Current ask candle (used for short position exits)
/// * `pip_size` - Pip size for the instrument
/// * `pip_buffer_factor` - Buffer factor (typically 0.5)
/// * `in_entry_candle` - Whether this is the entry candle (affects behavior)
///
/// # Returns
/// `Some(StopCheckResult)` if a stop was triggered, `None` otherwise.
#[must_use]
pub fn check_stops(
    position: &Position,
    bid: &Candle,
    ask: &Candle,
    pip_size: f64,
    pip_buffer_factor: f64,
    in_entry_candle: bool,
) -> Option<StopCheckResult> {
    // Rule 1+2: Market Orders cannot exit in entry candle
    // Pending Orders (Limit/Stop) CAN exit in entry candle (where triggered)
    if in_entry_candle && position.order_type.is_market() {
        return None;
    }

    let pip_buffer = pip_size * pip_buffer_factor;

    // Check SL and TP hit conditions based on direction
    let (sl_hit, tp_hit) = match position.direction {
        Direction::Long => {
            // Long position exits on bid side
            // SL: bid drops to/below SL level (plus buffer for slippage)
            let sl = bid.low <= position.stop_loss + pip_buffer;
            // TP: bid rises to/above TP level (minus buffer)
            let tp = bid.high >= position.take_profit - pip_buffer;
            (sl, tp)
        }
        Direction::Short => {
            // Short position exits on ask side
            // SL: ask rises to/above SL level (minus buffer for slippage)
            let sl = ask.high >= position.stop_loss - pip_buffer;
            // TP: ask drops to/below TP level (plus buffer)
            let tp = ask.low <= position.take_profit + pip_buffer;
            (sl, tp)
        }
    };

    // SL has priority over TP
    if sl_hit {
        return Some(StopCheckResult {
            triggered: true,
            reason: ExitReason::StopLoss,
            exit_price: position.stop_loss,
        });
    }

    // TP check with entry candle special logic (for pending orders only)
    if tp_hit {
        // In entry candle (pending order): TP only valid if close is "beyond" TP
        // Note: Market orders never reach here in entry candle (returned early)
        if in_entry_candle {
            let tp_valid = match position.direction {
                // Long: close must be above TP
                Direction::Long => bid.close > position.take_profit,
                // Short: close must be below TP
                Direction::Short => ask.close < position.take_profit,
            };
            if !tp_valid {
                return None;
            }
        }

        return Some(StopCheckResult {
            triggered: true,
            reason: ExitReason::TakeProfit,
            exit_price: position.take_profit,
        });
    }

    None
}

/// Checks if a position's stop-loss has been hit (SL only, ignores TP).
///
/// This is useful for the first pass of stop checks where SL takes priority.
#[must_use]
pub fn check_stop_loss_only(
    position: &Position,
    bid: &Candle,
    ask: &Candle,
    pip_size: f64,
    pip_buffer_factor: f64,
) -> Option<StopCheckResult> {
    let pip_buffer = pip_size * pip_buffer_factor;

    let sl_hit = match position.direction {
        Direction::Long => bid.low <= position.stop_loss + pip_buffer,
        Direction::Short => ask.high >= position.stop_loss - pip_buffer,
    };

    if sl_hit {
        Some(StopCheckResult {
            triggered: true,
            reason: ExitReason::StopLoss,
            exit_price: position.stop_loss,
        })
    } else {
        None
    }
}

/// Checks if a position's take-profit has been hit (TP only, ignores SL).
///
/// This is useful for the second pass of stop checks after SL has been processed.
#[must_use]
pub fn check_take_profit_only(
    position: &Position,
    bid: &Candle,
    ask: &Candle,
    pip_size: f64,
    pip_buffer_factor: f64,
    in_entry_candle: bool,
) -> Option<StopCheckResult> {
    let pip_buffer = pip_size * pip_buffer_factor;

    let tp_hit = match position.direction {
        Direction::Long => bid.high >= position.take_profit - pip_buffer,
        Direction::Short => ask.low <= position.take_profit + pip_buffer,
    };

    if !tp_hit {
        return None;
    }

    // Entry candle special logic
    if in_entry_candle {
        let tp_valid = match position.direction {
            Direction::Long => bid.close > position.take_profit,
            Direction::Short => ask.close < position.take_profit,
        };
        if !tp_valid {
            return None;
        }
    }

    Some(StopCheckResult {
        triggered: true,
        reason: ExitReason::TakeProfit,
        exit_price: position.take_profit,
    })
}

/// Calculates the exit price for a stop with potential gap handling.
///
/// If the price gaps through the stop level, the exit price may be
/// worse than the stop level (gap fill).
///
/// # Arguments
/// * `stop_level` - The SL or TP level
/// * `direction` - Position direction
/// * `bid` - Current bid candle
/// * `ask` - Current ask candle
/// * `is_stop_loss` - Whether this is an SL (affects gap direction)
#[must_use]
pub fn calculate_gap_exit_price(
    stop_level: f64,
    direction: &Direction,
    bid: &Candle,
    ask: &Candle,
    is_stop_loss: bool,
) -> f64 {
    match (direction, is_stop_loss) {
        // Long SL: exit at worse of SL level or bar open (if gapped down)
        (Direction::Long, true) => stop_level.min(bid.open),
        // Long TP: exit at better of TP level or bar open (if gapped up)
        (Direction::Long, false) => stop_level.max(bid.open),
        // Short SL: exit at worse of SL level or bar open (if gapped up)
        (Direction::Short, true) => stop_level.max(ask.open),
        // Short TP: exit at better of TP level or bar open (if gapped down)
        (Direction::Short, false) => stop_level.min(ask.open),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use omega_types::OrderType;
    use serde_json::json;

    /// Creates a Position with specified order type for testing.
    fn make_position_with_order_type(
        direction: Direction,
        order_type: OrderType,
        entry: f64,
        sl: f64,
        tp: f64,
    ) -> Position {
        Position {
            id: 1,
            direction,
            order_type,
            entry_time_ns: 0,
            entry_price: entry,
            size: 1.0,
            stop_loss: sl,
            take_profit: tp,
            scenario_id: 0,
            meta: json!({}),
        }
    }

    /// Creates a Market Order Position (default for most tests).
    fn make_position(direction: Direction, entry: f64, sl: f64, tp: f64) -> Position {
        make_position_with_order_type(direction, OrderType::Market, entry, sl, tp)
    }

    /// Creates a Pending (Limit) Order Position.
    fn make_pending_position(direction: Direction, entry: f64, sl: f64, tp: f64) -> Position {
        make_position_with_order_type(direction, OrderType::Limit, entry, sl, tp)
    }

    fn make_candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            close_time_ns: 60_000_000_000 - 1,
            open,
            high,
            low,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_long_sl_hit() {
        // Long position with SL at 1.1950
        let position = make_position(Direction::Long, 1.2000, 1.1950, 1.2100);

        // Bid drops to 1.1940 (below SL)
        let bid = make_candle(1.2000, 1.2020, 1.1940, 1.1960);
        let ask = make_candle(1.2002, 1.2022, 1.1942, 1.1962);

        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, false);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.triggered);
        assert_eq!(result.reason, ExitReason::StopLoss);
        assert_relative_eq!(result.exit_price, 1.1950, epsilon = 1e-10);
    }

    #[test]
    fn test_long_tp_hit() {
        // Long position with TP at 1.2100
        let position = make_position(Direction::Long, 1.2000, 1.1950, 1.2100);

        // Bid rises to 1.2110 (above TP)
        let bid = make_candle(1.2000, 1.2110, 1.1990, 1.2090);
        let ask = make_candle(1.2002, 1.2112, 1.1992, 1.2092);

        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, false);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.triggered);
        assert_eq!(result.reason, ExitReason::TakeProfit);
        assert_relative_eq!(result.exit_price, 1.2100, epsilon = 1e-10);
    }

    #[test]
    fn test_short_sl_hit() {
        // Short position with SL at 1.2050
        let position = make_position(Direction::Short, 1.2000, 1.2050, 1.1900);

        // Ask rises to 1.2060 (above SL)
        let bid = make_candle(1.2000, 1.2058, 1.1990, 1.2040);
        let ask = make_candle(1.2002, 1.2060, 1.1992, 1.2042);

        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, false);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.triggered);
        assert_eq!(result.reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_sl_priority_over_tp() {
        // Long position where both SL and TP would be hit
        let position = make_position(Direction::Long, 1.2000, 1.1950, 1.2100);

        // Wide range candle that hits both
        let bid = make_candle(1.2000, 1.2150, 1.1900, 1.2050);
        let ask = make_candle(1.2002, 1.2152, 1.1902, 1.2052);

        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, false);
        assert!(result.is_some());

        // SL should win
        let result = result.unwrap();
        assert_eq!(result.reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_entry_candle_tp_rule() {
        // Long position with TP at 1.2100
        let position = make_position(Direction::Long, 1.2000, 1.1950, 1.2100);

        // High reaches TP but close is below TP
        let bid = make_candle(1.2000, 1.2110, 1.1990, 1.2080);
        let ask = make_candle(1.2002, 1.2112, 1.1992, 1.2082);

        // In entry candle: TP should NOT trigger
        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, true);
        assert!(result.is_none());

        // Not in entry candle: TP should trigger
        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, false);
        assert!(result.is_some());
        assert_eq!(result.unwrap().reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_entry_candle_tp_valid_when_close_beyond() {
        // Pending (Limit) Order position with TP at 1.2100.
        // This tests Rule 4: A triggered Pending Order CAN exit in the same candle
        // where it was triggered (the "entry candle").
        let position = make_pending_position(Direction::Long, 1.2000, 1.1950, 1.2100);

        // Close is above TP
        let bid = make_candle(1.2000, 1.2150, 1.1990, 1.2120);
        let ask = make_candle(1.2002, 1.2152, 1.1992, 1.2122);

        // In entry candle and close is beyond TP: should trigger for Pending Orders
        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, true);
        assert!(result.is_some());
        assert_eq!(result.unwrap().reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_no_stop_hit() {
        let position = make_position(Direction::Long, 1.2000, 1.1950, 1.2100);

        // Price stays within range
        let bid = make_candle(1.2000, 1.2050, 1.1970, 1.2030);
        let ask = make_candle(1.2002, 1.2052, 1.1972, 1.2032);

        let result = check_stops(&position, &bid, &ask, 0.0001, 0.5, false);
        assert!(result.is_none());
    }

    #[test]
    fn test_gap_exit_price_sl() {
        let bid = make_candle(1.1920, 1.1960, 1.1910, 1.1950); // Gapped down

        // SL at 1.1950 but opened at 1.1920 - exit at worse price
        let exit = calculate_gap_exit_price(1.1950, &Direction::Long, &bid, &bid, true);
        assert_relative_eq!(exit, 1.1920, epsilon = 1e-10);
    }

    #[test]
    fn test_gap_exit_price_tp() {
        let bid = make_candle(1.2120, 1.2150, 1.2110, 1.2140); // Gapped up

        // TP at 1.2100 but opened at 1.2120 - exit at better price
        let exit = calculate_gap_exit_price(1.2100, &Direction::Long, &bid, &bid, false);
        assert_relative_eq!(exit, 1.2120, epsilon = 1e-10);
    }

    // =========================================================================
    // Explicit tests for the 4 Order Execution Rules
    // =========================================================================
    //
    // Rule 1: Two order types exist - Market Order and Pending Order (Limit/Stop)
    // Rule 2: Market Order -> SL/TP can only trigger in NEXT candle
    // Rule 3: Pending Order -> Can only trigger (be filled) in NEXT candle
    // Rule 4: Pending Order -> Once triggered, CAN exit in SAME candle

    /// Rule 2: Market Order cannot exit via SL in entry candle
    #[test]
    fn test_rule2_market_order_no_sl_in_entry_candle() {
        let bid = make_candle(1.2000, 1.2050, 1.1900, 1.2020); // Low at 1.1900
        let ask = make_candle(1.2002, 1.2052, 1.1902, 1.2022);

        // Market Order Long with SL at 1.1950 - hit by candle low
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Market,
            1.2000,
            1.1950, // SL would be hit
            1.2100,
        );

        // Entry candle (in_entry_candle = true) -> NO exit for Market Order
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, true);
        assert!(
            exit.is_none(),
            "Market Order should NOT exit in entry candle"
        );
    }

    /// Rule 2: Market Order cannot exit via TP in entry candle
    #[test]
    fn test_rule2_market_order_no_tp_in_entry_candle() {
        let bid = make_candle(1.2000, 1.2200, 1.1980, 1.2150); // High at 1.2200
        let ask = make_candle(1.2002, 1.2202, 1.1982, 1.2152);

        // Market Order Long with TP at 1.2100 - hit by candle high
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Market,
            1.2000,
            1.1950,
            1.2100, // TP would be hit
        );

        // Entry candle (in_entry_candle = true) -> NO exit for Market Order
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, true);
        assert!(
            exit.is_none(),
            "Market Order should NOT exit in entry candle"
        );
    }

    /// Rule 2: Market Order CAN exit in next candle (`in_entry_candle` = false)
    #[test]
    fn test_rule2_market_order_exits_in_next_candle() {
        let bid = make_candle(1.2000, 1.2050, 1.1900, 1.2020); // Low hits SL
        let ask = make_candle(1.2002, 1.2052, 1.1902, 1.2022);

        // Market Order Long with SL at 1.1950
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Market,
            1.2000,
            1.1950, // SL hit
            1.2100,
        );

        // Next candle (in_entry_candle = false) -> Market Order CAN exit
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, false);
        assert!(exit.is_some(), "Market Order SHOULD exit in next candle");
        assert_eq!(exit.unwrap().reason, ExitReason::StopLoss);
    }

    /// Rule 4: Pending Order (Limit) CAN exit via SL in entry candle
    #[test]
    fn test_rule4_pending_order_sl_in_entry_candle() {
        let bid = make_candle(1.2000, 1.2050, 1.1900, 1.2020); // Low at 1.1900
        let ask = make_candle(1.2002, 1.2052, 1.1902, 1.2022);

        // Pending Order (Limit) Long with SL at 1.1950 - hit by low
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Limit,
            1.2000,
            1.1950, // SL hit
            1.2100,
        );

        // Entry candle (in_entry_candle = true) -> Pending Order CAN exit
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, true);
        assert!(
            exit.is_some(),
            "Pending Order SHOULD exit via SL in entry candle"
        );
        assert_eq!(exit.unwrap().reason, ExitReason::StopLoss);
    }

    /// Rule 4: Pending Order (Limit) CAN exit via TP in entry candle
    #[test]
    fn test_rule4_pending_order_tp_in_entry_candle() {
        let bid = make_candle(1.2000, 1.2200, 1.1980, 1.2150); // High at 1.2200
        let ask = make_candle(1.2002, 1.2202, 1.1982, 1.2152);

        // Pending Order (Limit) Long with TP at 1.2100 - hit by high
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Limit,
            1.2000,
            1.1950,
            1.2100, // TP hit
        );

        // Entry candle (in_entry_candle = true) -> Pending Order CAN exit
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, true);
        assert!(
            exit.is_some(),
            "Pending Order SHOULD exit via TP in entry candle"
        );
        assert_eq!(exit.unwrap().reason, ExitReason::TakeProfit);
    }

    /// Rule 4: Pending Order (Stop) also CAN exit in entry candle
    #[test]
    fn test_rule4_stop_order_exits_in_entry_candle() {
        let bid = make_candle(1.2000, 1.2050, 1.1900, 1.2020); // Low at 1.1900
        let ask = make_candle(1.2002, 1.2052, 1.1902, 1.2022);

        // Stop Order Long with SL at 1.1950
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Stop,
            1.2000,
            1.1950, // SL hit
            1.2100,
        );

        // Entry candle (in_entry_candle = true) -> Stop Order (is_pending) CAN exit
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, true);
        assert!(exit.is_some(), "Stop Order SHOULD exit in entry candle");
        assert_eq!(exit.unwrap().reason, ExitReason::StopLoss);
    }

    /// Rule 4: SL has priority over TP when both hit in same candle
    #[test]
    fn test_rule4_sl_priority_over_tp_in_entry_candle() {
        // Candle where both SL and TP would be hit
        let bid = make_candle(1.2000, 1.2200, 1.1800, 1.2100);
        let ask = make_candle(1.2002, 1.2202, 1.1802, 1.2102);

        // Pending Order Long - both SL (1.1900) and TP (1.2100) hit
        let pos = make_position_with_order_type(
            Direction::Long,
            OrderType::Limit,
            1.2000,
            1.1900, // SL hit by low (1.1800)
            1.2100, // TP hit by high (1.2200)
        );

        // SL takes priority per documented behavior
        let exit = check_stops(&pos, &bid, &ask, 0.0001, 0.5, true);
        assert!(exit.is_some());
        assert_eq!(
            exit.unwrap().reason,
            ExitReason::StopLoss,
            "SL should have priority over TP"
        );
    }
}
