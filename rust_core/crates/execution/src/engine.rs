//! Execution engine for order processing and fill simulation.
//!
//! The ExecutionEngine coordinates order execution, slippage/fee application,
//! and pending order management with deterministic behavior.

use crate::costs::SymbolCosts;
use crate::fill::{market_fill, pending_fill};
use crate::pending_book::{PendingBook, PendingOrder, TriggerEvent};
use omega_types::{Candle, Signal};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Result of executing a market order.
#[derive(Debug, Clone)]
pub struct MarketOrderResult {
    /// Whether the order was filled
    pub filled: bool,
    /// Fill price after slippage
    pub fill_price: f64,
    /// Slippage that was applied
    pub slippage: f64,
    /// Entry fee that was charged
    pub entry_fee: f64,
}

/// Result of processing pending orders.
#[derive(Debug, Clone)]
pub struct PendingProcessResult {
    /// Orders that were triggered this bar
    pub triggered: Vec<TriggerEvent>,
    /// Fills for triggered orders
    pub fills: Vec<PendingFillResult>,
}

/// Fill result for a pending order.
#[derive(Debug, Clone)]
pub struct PendingFillResult {
    /// Order ID
    pub order_id: u64,
    /// Fill price after slippage
    pub fill_price: f64,
    /// Slippage that was applied
    pub slippage: f64,
    /// Entry fee that was charged
    pub entry_fee: f64,
    /// The filled order details
    pub order: PendingOrder,
}

/// Execution engine configuration.
#[derive(Debug, Clone)]
pub struct ExecutionEngineConfig {
    /// RNG seed for deterministic slippage
    pub rng_seed: u64,
    /// Whether to apply entry fees
    pub apply_entry_fees: bool,
    /// Whether to apply exit fees
    pub apply_exit_fees: bool,
}

impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            rng_seed: 42,
            apply_entry_fees: true,
            apply_exit_fees: true,
        }
    }
}

/// Execution engine for order processing.
///
/// Provides deterministic order execution with slippage and fee simulation.
/// All randomness uses ChaCha8Rng with configurable seed for reproducibility.
pub struct ExecutionEngine {
    /// Random number generator for slippage
    rng: ChaCha8Rng,
    /// Pending order book
    pending_book: PendingBook,
    /// Configuration
    config: ExecutionEngineConfig,
}

impl ExecutionEngine {
    /// Creates a new execution engine with the given configuration.
    pub fn new(config: ExecutionEngineConfig) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(config.rng_seed),
            pending_book: PendingBook::new(),
            config,
        }
    }

    /// Creates a new execution engine with default configuration.
    pub fn with_seed(seed: u64) -> Self {
        Self::new(ExecutionEngineConfig {
            rng_seed: seed,
            ..Default::default()
        })
    }

    /// Executes a market order immediately.
    ///
    /// # Arguments
    /// * `signal` - The trading signal
    /// * `costs` - Symbol-specific costs configuration
    ///
    /// # Returns
    /// Result containing fill price and fees
    pub fn execute_market_order(
        &mut self,
        signal: &Signal,
        costs: &SymbolCosts,
    ) -> MarketOrderResult {
        // Calculate slippage
        let slippage = costs
            .slippage
            .calculate(signal.entry_price, signal.direction, &mut self.rng);

        // Execute fill
        let fill = market_fill(signal.entry_price, signal.direction, slippage);

        // Calculate entry fee
        let entry_fee = if self.config.apply_entry_fees && costs.apply_entry_fee {
            let size = signal.size.unwrap_or(1.0);
            costs.fee.calculate(size, fill.fill_price)
        } else {
            0.0
        };

        MarketOrderResult {
            filled: fill.filled,
            fill_price: fill.fill_price,
            slippage: fill.slippage_applied,
            entry_fee,
        }
    }

    /// Adds a pending order (limit/stop) to the book.
    ///
    /// # Arguments
    /// * `signal` - The trading signal
    /// * `created_at_ns` - Creation timestamp
    ///
    /// # Returns
    /// The assigned order ID
    pub fn add_pending_order(&mut self, signal: &Signal, created_at_ns: i64) -> u64 {
        let order = PendingOrder::new(
            signal.order_type,
            signal.direction,
            signal.entry_price,
            signal.size.unwrap_or(1.0),
            signal.stop_loss,
            signal.take_profit,
            created_at_ns,
            signal.scenario_id,
            signal.meta.clone(),
        );
        self.pending_book.add_order(order)
    }

    /// Processes pending orders for the current bar.
    ///
    /// This includes:
    /// 1. Checking trigger conditions
    /// 2. Calculating fills for triggered orders
    /// 3. Removing filled orders from the book
    ///
    /// # Arguments
    /// * `bid` - Current bid candle
    /// * `ask` - Current ask candle
    /// * `current_bar_ns` - Current bar timestamp
    /// * `costs` - Symbol-specific costs configuration
    pub fn process_pending_orders(
        &mut self,
        bid: &Candle,
        ask: &Candle,
        current_bar_ns: i64,
        costs: &SymbolCosts,
    ) -> PendingProcessResult {
        // Check for triggers
        let triggered = self.pending_book.check_triggers(bid, ask, current_bar_ns);

        // Process fills for triggered orders
        let mut fills = Vec::new();

        for event in &triggered {
            if let Some(order) = self.pending_book.get_order(event.order_id) {
                // Calculate slippage for this fill
                let slippage = costs.slippage.calculate(
                    order.entry_price,
                    order.direction,
                    &mut self.rng,
                );

                // Attempt fill
                if let Some(fill_result) = pending_fill(
                    order.order_type,
                    order.entry_price,
                    order.direction,
                    bid,
                    ask,
                    slippage,
                ) {
                    // Calculate entry fee
                    let entry_fee = if self.config.apply_entry_fees && costs.apply_entry_fee {
                        costs.fee.calculate(order.size, fill_result.fill_price)
                    } else {
                        0.0
                    };

                    // Clone order before marking filled
                    let order_clone = order.clone();

                    fills.push(PendingFillResult {
                        order_id: event.order_id,
                        fill_price: fill_result.fill_price,
                        slippage: fill_result.slippage_applied,
                        entry_fee,
                        order: order_clone,
                    });
                }
            }
        }

        // Mark filled orders
        for fill in &fills {
            self.pending_book.mark_filled(fill.order_id);
        }

        PendingProcessResult { triggered, fills }
    }

    /// Calculates exit fee for closing a position.
    ///
    /// # Arguments
    /// * `size` - Position size
    /// * `exit_price` - Exit price
    /// * `costs` - Symbol-specific costs configuration
    pub fn calculate_exit_fee(&self, size: f64, exit_price: f64, costs: &SymbolCosts) -> f64 {
        if self.config.apply_exit_fees && costs.apply_exit_fee {
            costs.fee.calculate(size, exit_price)
        } else {
            0.0
        }
    }

    /// Cancels a pending order.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        self.pending_book.cancel_order(order_id)
    }

    /// Gets a pending order by ID.
    pub fn get_pending_order(&self, order_id: u64) -> Option<&PendingOrder> {
        self.pending_book.get_order(order_id)
    }

    /// Returns the number of pending orders.
    pub fn pending_count(&self) -> usize {
        self.pending_book.pending_count()
    }

    /// Returns the number of triggered (awaiting fill) orders.
    pub fn triggered_count(&self) -> usize {
        self.pending_book.triggered_count()
    }

    /// Cleans up terminal orders from the book.
    pub fn cleanup(&mut self) {
        self.pending_book.cleanup_terminal();
    }

    /// Resets the RNG to its initial state (for reproducibility testing).
    pub fn reset_rng(&mut self) {
        self.rng = ChaCha8Rng::seed_from_u64(self.config.rng_seed);
    }

    /// Gets a reference to the pending book.
    pub fn pending_book(&self) -> &PendingBook {
        &self.pending_book
    }

    /// Gets a mutable reference to the pending book.
    pub fn pending_book_mut(&mut self) -> &mut PendingBook {
        &mut self.pending_book
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::costs::SymbolCosts;
    use approx::assert_relative_eq;
    use omega_types::{Direction, OrderType};
    use serde_json::json;

    fn make_signal(direction: Direction, order_type: OrderType, entry: f64) -> Signal {
        Signal {
            direction,
            order_type,
            entry_price: entry,
            stop_loss: entry - 0.0050,
            take_profit: entry + 0.0100,
            size: Some(1.0),
            scenario_id: 0,
            tags: vec![],
            meta: json!({}),
        }
    }

    fn make_candle(open: f64, high: f64, low: f64, close: f64, ts: i64) -> Candle {
        Candle {
            timestamp_ns: ts,
            open,
            high,
            low,
            close,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_market_order_execution() {
        let mut engine = ExecutionEngine::with_seed(42);
        let costs = SymbolCosts::zero_cost(0.0001);

        let signal = make_signal(Direction::Long, OrderType::Market, 1.2000);
        let result = engine.execute_market_order(&signal, &costs);

        assert!(result.filled);
        assert_relative_eq!(result.fill_price, 1.2000, epsilon = 1e-6);
        assert_relative_eq!(result.slippage, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.entry_fee, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pending_order_lifecycle() {
        let mut engine = ExecutionEngine::with_seed(42);
        let costs = SymbolCosts::zero_cost(0.0001);

        // Add a limit buy order
        let signal = make_signal(Direction::Long, OrderType::Limit, 1.1980);
        let order_id = engine.add_pending_order(&signal, 1000);
        assert_eq!(order_id, 1);
        assert_eq!(engine.pending_count(), 1);

        // Process in same bar - should not trigger (next-bar rule)
        let bid = make_candle(1.2000, 1.2050, 1.1960, 1.2030, 1000);
        let ask = make_candle(1.2002, 1.2052, 1.1962, 1.2032, 1000);
        let result = engine.process_pending_orders(&bid, &ask, 1000, &costs);
        assert!(result.triggered.is_empty());

        // Process in next bar - should trigger and fill
        let result = engine.process_pending_orders(&bid, &ask, 2000, &costs);
        assert_eq!(result.triggered.len(), 1);
        assert_eq!(result.fills.len(), 1);
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn test_deterministic_execution() {
        let costs = SymbolCosts::zero_cost(0.0001);

        // Run same sequence twice
        let mut engine1 = ExecutionEngine::with_seed(42);
        let mut engine2 = ExecutionEngine::with_seed(42);

        let signal = make_signal(Direction::Long, OrderType::Market, 1.2000);

        let result1 = engine1.execute_market_order(&signal, &costs);
        let result2 = engine2.execute_market_order(&signal, &costs);

        assert_relative_eq!(result1.fill_price, result2.fill_price, epsilon = 1e-10);
        assert_relative_eq!(result1.slippage, result2.slippage, epsilon = 1e-10);
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = ExecutionEngine::with_seed(42);

        let signal = make_signal(Direction::Long, OrderType::Limit, 1.1980);
        let order_id = engine.add_pending_order(&signal, 1000);

        assert!(engine.cancel_order(order_id));
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn test_exit_fee_calculation() {
        let engine = ExecutionEngine::with_seed(42);
        let costs = SymbolCosts::zero_cost(0.0001);

        let fee = engine.calculate_exit_fee(1.0, 1.2000, &costs);
        assert_relative_eq!(fee, 0.0, epsilon = 1e-10);
    }
}
