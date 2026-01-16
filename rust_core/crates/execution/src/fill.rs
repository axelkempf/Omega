//! Fill logic for order execution.
//!
//! Handles market fills and pending order (limit/stop) fills with gap-aware pricing.

use omega_types::{Candle, Direction, OrderType};

/// Result of a fill attempt.
#[derive(Debug, Clone)]
pub struct FillResult {
    /// Whether the order was filled
    pub filled: bool,
    /// The actual fill price (after slippage)
    pub fill_price: f64,
    /// Slippage that was applied
    pub slippage_applied: f64,
}

/// Calculates fill price for a market order.
///
/// Market orders are always filled at the signal price plus slippage.
///
/// # Arguments
/// * `signal_price` - The price at which the signal was generated
/// * `direction` - Trade direction
/// * `slippage` - Pre-calculated slippage amount
#[must_use]
pub fn market_fill(signal_price: f64, direction: Direction, slippage: f64) -> FillResult {
    let fill_price = match direction {
        Direction::Long => signal_price + slippage,
        Direction::Short => signal_price - slippage,
    };
    FillResult {
        filled: true,
        fill_price,
        slippage_applied: slippage,
    }
}

/// Checks if a pending order (limit/stop) is triggered and calculates gap-aware fill price.
///
/// # Gap-Aware Fill Logic
///
/// When a pending order triggers, the fill price accounts for potential gaps:
/// - For long entries: fill price is the worse of `entry_price` and bar open (plus slippage)
/// - For short entries: fill price is the worse of `entry_price` and bar open (minus slippage)
///
/// This simulates realistic execution where gaps can cause adverse fills.
///
/// # Trigger Conditions
///
/// | Order Type | Direction | Trigger Condition |
/// |------------|-----------|-------------------|
/// | Limit      | Long      | Ask Low <= Entry Price |
/// | Limit      | Short     | Bid High >= Entry Price |
/// | Stop       | Long      | Ask High >= Entry Price |
/// | Stop       | Short     | Bid Low <= Entry Price |
///
/// # Arguments
/// * `order_type` - Type of pending order (Limit or Stop)
/// * `entry_price` - The order's entry price
/// * `direction` - Trade direction
/// * `bid` - Current bid candle (for short exits/entries)
/// * `ask` - Current ask candle (for long exits/entries)
/// * `slippage` - Pre-calculated slippage amount
#[must_use]
pub fn pending_fill(
    order_type: OrderType,
    entry_price: f64,
    direction: Direction,
    bid: &Candle,
    ask: &Candle,
    slippage: f64,
) -> Option<FillResult> {
    // Check if order is triggered based on type and direction
    let triggered = match (order_type, &direction) {
        // Limit Long: Buy when price drops to entry level (ask low touches entry)
        (OrderType::Limit, Direction::Long) => ask.low <= entry_price,
        // Limit Short: Sell when price rises to entry level (bid high touches entry)
        (OrderType::Limit, Direction::Short) => bid.high >= entry_price,
        // Stop Long: Buy when price rises above entry (breakout)
        (OrderType::Stop, Direction::Long) => ask.high >= entry_price,
        // Stop Short: Sell when price drops below entry (breakdown)
        (OrderType::Stop, Direction::Short) => bid.low <= entry_price,
        // Market orders should not be in pending book
        (OrderType::Market, _) => return None,
    };

    if !triggered {
        return None;
    }

    Some(pending_fill_triggered(
        entry_price,
        direction,
        bid,
        ask,
        slippage,
    ))
}

/// Calculates a fill for an already-triggered pending order (next-bar fill).
///
/// This uses gap-aware pricing against the current bar open and applies
/// slippage in the entry direction.
pub(crate) fn pending_fill_triggered(
    entry_price: f64,
    direction: Direction,
    bid: &Candle,
    ask: &Candle,
    slippage: f64,
) -> FillResult {
    let base_fill = match direction {
        // For longs: worse fill is the higher price
        Direction::Long => entry_price.max(ask.open),
        // For shorts: worse fill is the lower price
        Direction::Short => entry_price.min(bid.open),
    };

    let fill_price = match direction {
        Direction::Long => base_fill + slippage,
        Direction::Short => base_fill - slippage,
    };

    FillResult {
        filled: true,
        fill_price,
        slippage_applied: slippage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            open,
            high,
            low,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_market_fill_long() {
        let result = market_fill(1.2000, Direction::Long, 0.0001);
        assert!(result.filled);
        assert_relative_eq!(result.fill_price, 1.2001, epsilon = 1e-10);
    }

    #[test]
    fn test_market_fill_short() {
        let result = market_fill(1.2000, Direction::Short, 0.0001);
        assert!(result.filled);
        assert_relative_eq!(result.fill_price, 1.1999, epsilon = 1e-10);
    }

    #[test]
    fn test_limit_long_triggered() {
        let bid = make_candle(1.2000, 1.2050, 1.1980, 1.2030);
        let ask = make_candle(1.2002, 1.2052, 1.1982, 1.2032);

        // Limit buy at 1.1990 - ask.low (1.1982) is below entry
        let result = pending_fill(
            OrderType::Limit,
            1.1990,
            Direction::Long,
            &bid,
            &ask,
            0.0001,
        );
        assert!(result.is_some());

        let fill = result.unwrap();
        // Gap-aware: entry (1.1990) > ask.open (1.2002), so use ask.open
        // Actually: max(1.1990, 1.2002) = 1.2002, + slippage = 1.2003
        assert_relative_eq!(fill.fill_price, 1.2003, epsilon = 1e-10);
    }

    #[test]
    fn test_limit_long_not_triggered() {
        let bid = make_candle(1.2000, 1.2050, 1.1990, 1.2030);
        let ask = make_candle(1.2002, 1.2052, 1.1992, 1.2032);

        // Limit buy at 1.1980 - ask.low (1.1992) is above entry
        let result = pending_fill(
            OrderType::Limit,
            1.1980,
            Direction::Long,
            &bid,
            &ask,
            0.0001,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_stop_long_triggered() {
        let bid = make_candle(1.2000, 1.2050, 1.1980, 1.2030);
        let ask = make_candle(1.2002, 1.2052, 1.1982, 1.2032);

        // Stop buy at 1.2040 - ask.high (1.2052) is above entry
        let result = pending_fill(OrderType::Stop, 1.2040, Direction::Long, &bid, &ask, 0.0001);
        assert!(result.is_some());

        let fill = result.unwrap();
        // max(1.2040, 1.2002) = 1.2040 + slippage = 1.2041
        assert_relative_eq!(fill.fill_price, 1.2041, epsilon = 1e-10);
    }

    #[test]
    fn test_limit_short_triggered() {
        let bid = make_candle(1.2000, 1.2050, 1.1980, 1.2030);
        let ask = make_candle(1.2002, 1.2052, 1.1982, 1.2032);

        // Limit sell at 1.2040 - bid.high (1.2050) is above entry
        let result = pending_fill(
            OrderType::Limit,
            1.2040,
            Direction::Short,
            &bid,
            &ask,
            0.0001,
        );
        assert!(result.is_some());

        let fill = result.unwrap();
        // min(1.2040, 1.2000) = 1.2000 - slippage = 1.1999
        assert_relative_eq!(fill.fill_price, 1.1999, epsilon = 1e-10);
    }

    #[test]
    fn test_gap_adverse_fill_long() {
        // Simulate a gap up scenario
        let bid = make_candle(1.2050, 1.2100, 1.2040, 1.2080); // Opens higher
        let ask = make_candle(1.2052, 1.2102, 1.2042, 1.2082);

        // Stop buy at 1.2030 - triggered because ask.high > entry
        // But opens at 1.2052, so fill should be at the worse (higher) open price
        let result = pending_fill(OrderType::Stop, 1.2030, Direction::Long, &bid, &ask, 0.0001);
        assert!(result.is_some());

        let fill = result.unwrap();
        // max(1.2030, 1.2052) = 1.2052 + slippage = 1.2053
        assert_relative_eq!(fill.fill_price, 1.2053, epsilon = 1e-10);
    }

    #[test]
    fn test_gap_adverse_fill_short() {
        // Simulate a gap down scenario
        let bid = make_candle(1.1950, 1.1980, 1.1940, 1.1960); // Opens lower
        let ask = make_candle(1.1952, 1.1982, 1.1942, 1.1962);

        // Stop sell at 1.1970 - triggered because bid.low < entry
        // But opens at 1.1950, so fill should be at the worse (lower) open price
        let result = pending_fill(
            OrderType::Stop,
            1.1970,
            Direction::Short,
            &bid,
            &ask,
            0.0001,
        );
        assert!(result.is_some());

        let fill = result.unwrap();
        // min(1.1970, 1.1950) = 1.1950 - slippage = 1.1949
        assert_relative_eq!(fill.fill_price, 1.1949, epsilon = 1e-10);
    }

    #[test]
    fn test_pending_fill_triggered_long() {
        let bid = make_candle(1.2000, 1.2050, 1.1980, 1.2030);
        let ask = make_candle(1.2002, 1.2052, 1.1982, 1.2032);

        let fill = pending_fill_triggered(1.1990, Direction::Long, &bid, &ask, 0.0001);
        assert_relative_eq!(fill.fill_price, 1.2003, epsilon = 1e-10);
    }
}
