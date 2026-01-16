//! Pending order book management.
//!
//! Manages all pending orders (Limit/Stop) with deterministic trigger ordering.

use crate::error::ExecutionError;
use crate::state::OrderState;
use omega_types::{Candle, Direction, OrderType};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

/// A pending order in the book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOrder {
    /// Unique order ID
    pub id: u64,
    /// Order type (Limit or Stop)
    pub order_type: OrderType,
    /// Trade direction
    pub direction: Direction,
    /// Entry price level
    pub entry_price: f64,
    /// Position size in lots
    pub size: f64,
    /// Stop-loss price
    pub stop_loss: f64,
    /// Take-profit price
    pub take_profit: f64,
    /// Current order state
    pub state: OrderState,
    /// Creation timestamp in nanoseconds (Unix epoch UTC)
    pub created_at_ns: i64,
    /// Expiration timestamp (None = GTC - Good Till Cancelled)
    pub good_till_ns: Option<i64>,
    /// Scenario ID for multi-scenario backtesting
    pub scenario_id: u8,
    /// Custom metadata
    pub meta: JsonValue,
    /// Timestamp when order was triggered (if triggered)
    pub triggered_at_ns: Option<i64>,
}

impl PendingOrder {
    /// Creates a new pending order.
    ///
    /// # Errors
    /// Returns [`ExecutionError::InvalidOrder`] if `order_type` is Market.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        order_type: OrderType,
        direction: Direction,
        entry_price: f64,
        size: f64,
        stop_loss: f64,
        take_profit: f64,
        created_at_ns: i64,
        scenario_id: u8,
        meta: JsonValue,
    ) -> Result<Self, ExecutionError> {
        if matches!(order_type, OrderType::Market) {
            return Err(ExecutionError::InvalidOrder(
                "market orders cannot be pending".to_string(),
            ));
        }

        Ok(Self {
            id: 0, // Will be assigned by PendingBook
            order_type,
            direction,
            entry_price,
            size,
            stop_loss,
            take_profit,
            state: OrderState::Pending,
            created_at_ns,
            good_till_ns: None,
            scenario_id,
            meta,
            triggered_at_ns: None,
        })
    }

    /// Sets the expiration time.
    #[must_use]
    pub fn with_good_till(mut self, good_till_ns: i64) -> Self {
        self.good_till_ns = Some(good_till_ns);
        self
    }
}

/// Event generated when an order is triggered.
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    /// ID of the triggered order
    pub order_id: u64,
    /// Timestamp when triggered (bar timestamp)
    pub triggered_at_ns: i64,
    /// Price level that caused the trigger
    pub trigger_price: f64,
}

/// Manages all pending orders with deterministic trigger ordering.
///
/// # Determinism Rules (Normative)
///
/// 1. Orders are sorted by `(created_at_ns, id)` for FIFO + tie-break
/// 2. Trigger checks occur in this deterministic order
/// 3. Multiple orders triggered in the same bar: all trigger, fill can occur in that bar
#[derive(Debug, Default)]
pub struct PendingBook {
    /// Orders sorted by (`created_at_ns`, `order_id`) for deterministic iteration
    orders: BTreeMap<(i64, u64), PendingOrder>,
    /// Next order ID to assign
    next_id: u64,
}

impl PendingBook {
    /// Creates a new empty pending book.
    #[must_use]
    pub fn new() -> Self {
        Self {
            orders: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Adds a new pending order to the book.
    ///
    /// Returns the assigned order ID.
    ///
    /// # Errors
    /// Returns [`ExecutionError::InvalidOrder`] if the order type is Market.
    pub fn add_order(&mut self, mut order: PendingOrder) -> Result<u64, ExecutionError> {
        if matches!(order.order_type, OrderType::Market) {
            return Err(ExecutionError::InvalidOrder(
                "market orders cannot be pending".to_string(),
            ));
        }

        let id = self.next_id;
        self.next_id += 1;
        order.id = id;
        order.state = OrderState::Pending;

        let key = (order.created_at_ns, id);
        self.orders.insert(key, order);
        Ok(id)
    }

    /// Checks all orders for trigger conditions and returns triggered events.
    ///
    /// # Next-Bar Rule
    ///
    /// Pending orders are created at candle close. They do NOT trigger in the
    /// same candle as placement, but only starting from the next bar.
    ///
    /// # Arguments
    /// * `bid` - Current bid candle
    /// * `ask` - Current ask candle
    /// * `current_bar_ns` - Timestamp of the current bar
    pub fn check_triggers(
        &mut self,
        bid: &Candle,
        ask: &Candle,
        current_bar_ns: i64,
    ) -> Vec<TriggerEvent> {
        let mut events = Vec::new();

        for ((created_ns, order_id), order) in &mut self.orders {
            // Skip: already triggered or terminal
            if order.state != OrderState::Pending {
                continue;
            }

            // Next-bar rule: order must be created BEFORE this bar
            if *created_ns >= current_bar_ns {
                continue;
            }

            // Expiration check
            if let Some(gtd) = order.good_till_ns
                && current_bar_ns > gtd
            {
                order.state = OrderState::Expired;
                continue;
            }

            // Check trigger condition
            let triggered = match (order.order_type, &order.direction) {
                // Limit Long: Ask falls to/below entry (buy low)
                (OrderType::Limit, Direction::Long) => ask.low <= order.entry_price,
                // Limit Short: Bid rises to/above entry (sell high)
                (OrderType::Limit, Direction::Short) => bid.high >= order.entry_price,
                // Stop Long: Ask rises to/above entry (breakout buy)
                (OrderType::Stop, Direction::Long) => ask.high >= order.entry_price,
                // Stop Short: Bid falls to/below entry (breakdown sell)
                (OrderType::Stop, Direction::Short) => bid.low <= order.entry_price,
                // Market orders should not be in PendingBook
                (OrderType::Market, _) => {
                    order.state = OrderState::Rejected;
                    false
                }
            };

            if triggered {
                order.state = OrderState::Triggered;
                order.triggered_at_ns = Some(current_bar_ns);
                events.push(TriggerEvent {
                    order_id: *order_id,
                    triggered_at_ns: current_bar_ns,
                    trigger_price: order.entry_price,
                });
            }
        }

        events
    }

    /// Marks an order as filled and removes it from the book.
    ///
    /// Returns the filled order if found.
    pub fn mark_filled(&mut self, order_id: u64) -> Option<PendingOrder> {
        let key = self
            .orders
            .iter()
            .find(|((_, id), _)| *id == order_id)
            .map(|(k, _)| *k)?;

        let mut order = self.orders.remove(&key)?;
        order.state = OrderState::Filled;
        Some(order)
    }

    /// Marks an order as rejected.
    ///
    /// Returns true if the order was found and rejected.
    pub fn mark_rejected(&mut self, order_id: u64) -> bool {
        for ((_, id), order) in &mut self.orders {
            if *id == order_id && order.state == OrderState::Triggered {
                order.state = OrderState::Rejected;
                return true;
            }
        }
        false
    }

    /// Cancels a pending order.
    ///
    /// Returns true if the order was found and cancelled.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        for ((_, id), order) in &mut self.orders {
            if *id == order_id && order.state == OrderState::Pending {
                order.state = OrderState::Cancelled;
                return true;
            }
        }
        false
    }

    /// Returns all triggered orders (for fill processing).
    #[must_use]
    pub fn get_triggered(&self) -> Vec<&PendingOrder> {
        self.orders
            .values()
            .filter(|o| o.state == OrderState::Triggered)
            .collect()
    }

    /// Returns orders that are eligible to fill on the current bar.
    ///
    /// Triggered orders are eligible in the same bar as the trigger. This still
    /// cannot happen in the placement candle because triggers require
    /// `created_at_ns < current_bar_ns`.
    #[must_use]
    pub fn eligible_for_fill(&self, current_bar_ns: i64) -> Vec<&PendingOrder> {
        self.orders
            .values()
            .filter(|order| {
                order.state == OrderState::Triggered
                    && order
                        .triggered_at_ns
                        .is_some_and(|triggered_at| triggered_at <= current_bar_ns)
            })
            .collect()
    }

    /// Returns a specific order by ID.
    #[must_use]
    pub fn get_order(&self, order_id: u64) -> Option<&PendingOrder> {
        self.orders.values().find(|o| o.id == order_id)
    }

    /// Returns a mutable reference to a specific order by ID.
    pub fn get_order_mut(&mut self, order_id: u64) -> Option<&mut PendingOrder> {
        self.orders.values_mut().find(|o| o.id == order_id)
    }

    /// Removes all terminal orders from the book.
    pub fn cleanup_terminal(&mut self) {
        self.orders.retain(|_, o| !o.state.is_terminal());
    }

    /// Returns the count of pending (not yet triggered) orders.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.orders
            .values()
            .filter(|o| o.state == OrderState::Pending)
            .count()
    }

    /// Returns the count of triggered (awaiting fill) orders.
    #[must_use]
    pub fn triggered_count(&self) -> usize {
        self.orders
            .values()
            .filter(|o| o.state == OrderState::Triggered)
            .count()
    }

    /// Returns the total count of orders in the book.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.orders.len()
    }

    /// Checks if the book is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Returns all orders in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = &PendingOrder> {
        self.orders.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde_json::json;

    fn make_candle(open: f64, high: f64, low: f64, close: f64, ts: i64) -> Candle {
        Candle {
            timestamp_ns: ts,
            close_time_ns: ts + 60_000_000_000 - 1,
            open,
            high,
            low,
            close,
            volume: 1000.0,
        }
    }

    fn make_limit_long(entry: f64, created_at: i64) -> PendingOrder {
        PendingOrder::new(
            OrderType::Limit,
            Direction::Long,
            entry,
            1.0,
            entry - 0.0050,
            entry + 0.0100,
            created_at,
            0,
            json!({}),
        )
        .unwrap()
    }

    fn make_stop_long(entry: f64, created_at: i64) -> PendingOrder {
        PendingOrder::new(
            OrderType::Stop,
            Direction::Long,
            entry,
            1.0,
            entry - 0.0050,
            entry + 0.0100,
            created_at,
            0,
            json!({}),
        )
        .unwrap()
    }

    #[test]
    fn test_add_order_assigns_id() {
        let mut book = PendingBook::new();

        let order1 = make_limit_long(1.2000, 1_000_000);
        let order2 = make_limit_long(1.2010, 2_000_000);

        let id1 = book.add_order(order1).unwrap();
        let id2 = book.add_order(order2).unwrap();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(book.total_count(), 2);
    }

    #[test]
    fn test_next_bar_rule() {
        let mut book = PendingBook::new();

        // Order created at bar 1000
        let order = make_limit_long(1.1980, 1000);
        book.add_order(order).unwrap();

        // Check in same bar - should NOT trigger
        let bid = make_candle(1.2000, 1.2050, 1.1960, 1.2030, 1000);
        let ask = make_candle(1.2002, 1.2052, 1.1962, 1.2032, 1000);
        let events = book.check_triggers(&bid, &ask, 1000);
        assert!(events.is_empty());
        assert_eq!(book.pending_count(), 1);

        // Check in next bar - should trigger
        let events = book.check_triggers(&bid, &ask, 2000);
        assert_eq!(events.len(), 1);
        assert_eq!(book.triggered_count(), 1);
    }

    #[test]
    fn test_limit_long_trigger() {
        let mut book = PendingBook::new();

        // Limit buy at 1.1980
        let order = make_limit_long(1.1980, 0);
        book.add_order(order).unwrap();

        // Ask low reaches entry price
        let bid = make_candle(1.2000, 1.2050, 1.1970, 1.2030, 1000);
        let ask = make_candle(1.2002, 1.2052, 1.1972, 1.2032, 1000);

        let events = book.check_triggers(&bid, &ask, 1000);
        assert_eq!(events.len(), 1);
        assert_relative_eq!(events[0].trigger_price, 1.1980, epsilon = 1e-10);
    }

    #[test]
    fn test_stop_long_trigger() {
        let mut book = PendingBook::new();

        // Stop buy at 1.2050
        let order = make_stop_long(1.2050, 0);
        book.add_order(order).unwrap();

        // Ask high reaches entry price
        let bid = make_candle(1.2000, 1.2060, 1.1980, 1.2040, 1000);
        let ask = make_candle(1.2002, 1.2062, 1.1982, 1.2042, 1000);

        let events = book.check_triggers(&bid, &ask, 1000);
        assert_eq!(events.len(), 1);
        assert_relative_eq!(events[0].trigger_price, 1.2050, epsilon = 1e-10);
    }

    #[test]
    fn test_order_expiration() {
        let mut book = PendingBook::new();

        // Order with expiration at bar 5000
        let order = make_limit_long(1.1980, 0).with_good_till(5000);
        book.add_order(order).unwrap();

        // Check after expiration
        let bid = make_candle(1.2000, 1.2050, 1.1960, 1.2030, 6000);
        let ask = make_candle(1.2002, 1.2052, 1.1962, 1.2032, 6000);

        let events = book.check_triggers(&bid, &ask, 6000);
        assert!(events.is_empty());

        // Order should be expired
        let order = book.get_order(1).unwrap();
        assert_eq!(order.state, OrderState::Expired);
    }

    #[test]
    fn test_mark_filled() {
        let mut book = PendingBook::new();

        let order = make_limit_long(1.1980, 0);
        let id = book.add_order(order).unwrap();

        // Trigger the order
        let bid = make_candle(1.2000, 1.2050, 1.1960, 1.2030, 1000);
        let ask = make_candle(1.2002, 1.2052, 1.1962, 1.2032, 1000);
        book.check_triggers(&bid, &ask, 1000);

        // Mark as filled
        let filled = book.mark_filled(id);
        assert!(filled.is_some());
        assert_eq!(filled.unwrap().state, OrderState::Filled);
        assert_eq!(book.total_count(), 0);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = PendingBook::new();

        let order = make_limit_long(1.1980, 0);
        let id = book.add_order(order).unwrap();

        assert!(book.cancel_order(id));
        let order = book.get_order(id).unwrap();
        assert_eq!(order.state, OrderState::Cancelled);
    }

    #[test]
    fn test_deterministic_order() {
        let mut book = PendingBook::new();

        // Add orders in non-chronological order of creation time
        let order3 = make_limit_long(1.1970, 3000);
        let order1 = make_limit_long(1.1970, 1000);
        let order2 = make_limit_long(1.1970, 2000);

        book.add_order(order3).unwrap();
        book.add_order(order1).unwrap();
        book.add_order(order2).unwrap();

        // Iteration should be in chronological order by (created_at_ns, id)
        let ordered: Vec<_> = book.iter().collect();
        assert_eq!(ordered[0].created_at_ns, 1000);
        assert_eq!(ordered[1].created_at_ns, 2000);
        assert_eq!(ordered[2].created_at_ns, 3000);
    }

    #[test]
    fn test_cleanup_terminal() {
        let mut book = PendingBook::new();

        let order1 = make_limit_long(1.1980, 0);
        let order2 = make_limit_long(1.1990, 0);

        let id1 = book.add_order(order1).unwrap();
        book.add_order(order2).unwrap();

        // Cancel first order
        book.cancel_order(id1);
        assert_eq!(book.total_count(), 2);

        // Cleanup terminal
        book.cleanup_terminal();
        assert_eq!(book.total_count(), 1);
    }

    #[test]
    fn test_market_order_rejected() {
        let order = PendingOrder::new(
            OrderType::Market,
            Direction::Long,
            1.2000,
            1.0,
            1.1900,
            1.2100,
            0,
            0,
            json!({}),
        );

        assert!(matches!(order, Err(ExecutionError::InvalidOrder(_))));

        let mut book = PendingBook::new();
        let order = PendingOrder {
            id: 0,
            order_type: OrderType::Market,
            direction: Direction::Long,
            entry_price: 1.2000,
            size: 1.0,
            stop_loss: 1.1900,
            take_profit: 1.2100,
            state: OrderState::Pending,
            created_at_ns: 0,
            good_till_ns: None,
            scenario_id: 0,
            meta: json!({}),
            triggered_at_ns: None,
        };

        assert!(matches!(
            book.add_order(order),
            Err(ExecutionError::InvalidOrder(_))
        ));
    }
}
