//! Execution engine for order processing and fill simulation.
//!
//! The `ExecutionEngine` coordinates order execution, slippage/fee application,
//! and pending order management with deterministic behavior.

use crate::costs::SymbolCosts;
use crate::error::ExecutionError;
use crate::fill::{market_fill, pending_fill_triggered};
use crate::pending_book::{PendingBook, PendingOrder, TriggerEvent};
use omega_types::{Candle, Direction, ExecutionVariant, Signal};
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
    /// Quantized fill size
    pub size: f64,
}

/// Result of processing pending orders.
#[derive(Debug, Clone)]
pub struct PendingProcessResult {
    /// Orders that were triggered this bar
    pub triggered: Vec<TriggerEvent>,
    /// Fills for triggered orders
    pub fills: Vec<PendingFillResult>,
    /// Orders that were rejected during fill validation
    pub rejected: Vec<PendingOrderRejection>,
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

/// Rejection information for a pending order.
#[derive(Debug, Clone)]
pub struct PendingOrderRejection {
    /// Order ID
    pub order_id: u64,
    /// Rejection reason
    pub reason: String,
}

/// Result of applying exit slippage and fees.
#[derive(Debug, Clone)]
pub struct ExitFillResult {
    /// Exit fill price after slippage
    pub fill_price: f64,
    /// Slippage that was applied
    pub slippage: f64,
    /// Exit fee that was charged
    pub exit_fee: f64,
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
    /// Execution variant (v2 vs v1 parity)
    pub execution_variant: ExecutionVariant,
}

impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            rng_seed: 42,
            apply_entry_fees: true,
            apply_exit_fees: true,
            execution_variant: ExecutionVariant::V2,
        }
    }
}

/// Execution engine for order processing.
///
/// Provides deterministic order execution with slippage and fee simulation.
/// All randomness uses `ChaCha8Rng` with configurable seed for reproducibility.
pub struct ExecutionEngine {
    /// Pending order book
    pending_book: PendingBook,
    /// Configuration
    config: ExecutionEngineConfig,
    /// Trade index for deterministic RNG seeding
    trade_index: u64,
}

impl ExecutionEngine {
    /// Creates a new execution engine with the given configuration.
    #[must_use]
    pub fn new(config: ExecutionEngineConfig) -> Self {
        Self {
            pending_book: PendingBook::new(),
            config,
            trade_index: 0,
        }
    }

    /// Creates a new execution engine with default configuration.
    #[must_use]
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
    /// Result containing fill price, fees, and quantized size
    ///
    /// # Errors
    /// Returns an error if size is invalid or stop-loss distance is too small.
    pub fn execute_market_order(
        &mut self,
        signal: &Signal,
        costs: &SymbolCosts,
    ) -> Result<MarketOrderResult, ExecutionError> {
        let raw_size = signal.size.unwrap_or(1.0);
        let size = Self::quantize_volume(raw_size, costs)?;

        // Calculate slippage (deterministic per trade_index)
        let slippage = self.preview_slippage(signal.entry_price, signal.direction, costs);

        // Execute fill
        let fill = market_fill(signal.entry_price, signal.direction, slippage);

        // Validate minimum SL distance using fill price
        Self::validate_min_sl_distance(fill.fill_price, signal.stop_loss, costs)?;

        // Consume trade index only after validation
        self.trade_index = self.trade_index.wrapping_add(1);

        // Calculate entry fee
        let entry_fee = if self.config.apply_entry_fees && costs.apply_entry_fee {
            costs.fee.calculate(size, fill.fill_price)
        } else {
            0.0
        };

        Ok(MarketOrderResult {
            filled: fill.filled,
            fill_price: fill.fill_price,
            slippage: fill.slippage_applied,
            entry_fee,
            size,
        })
    }

    /// Adds a pending order (limit/stop) to the book.
    ///
    /// # Arguments
    /// * `signal` - The trading signal
    /// * `created_at_ns` - Creation timestamp
    /// * `costs` - Symbol-specific costs configuration
    ///
    /// # Returns
    /// The assigned order ID, or an error if the order is invalid.
    ///
    /// # Errors
    /// Returns an error if order type is Market or validation fails.
    pub fn add_pending_order(
        &mut self,
        signal: &Signal,
        created_at_ns: i64,
        costs: &SymbolCosts,
    ) -> Result<u64, ExecutionError> {
        if matches!(signal.order_type, omega_types::OrderType::Market) {
            return Err(ExecutionError::InvalidOrder(
                "market orders cannot be pending".to_string(),
            ));
        }

        let raw_size = signal.size.unwrap_or(1.0);
        let size = Self::quantize_volume(raw_size, costs)?;
        Self::validate_min_sl_distance(signal.entry_price, signal.stop_loss, costs)?;

        let order = PendingOrder::new(
            signal.order_type,
            signal.direction,
            signal.entry_price,
            size,
            signal.stop_loss,
            signal.take_profit,
            created_at_ns,
            signal.scenario_id,
            signal.meta.clone(),
        )?;
        self.pending_book.add_order(order)
    }

    /// Processes pending orders for the current bar.
    ///
    /// This includes:
    /// 1. Checking trigger conditions
    /// 2. Calculating fills for eligible triggered orders (same-bar fills allowed)
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

        let eligible_orders: Vec<PendingOrder> = self
            .pending_book
            .eligible_for_fill(current_bar_ns)
            .into_iter()
            .cloned()
            .collect();

        // Process fills for eligible triggered orders
        let mut fills = Vec::new();
        let mut rejected = Vec::new();

        for order in eligible_orders {
            let fill_result = match self.config.execution_variant {
                ExecutionVariant::V2 => {
                    let slippage =
                        self.preview_slippage(order.entry_price, order.direction, costs);
                    pending_fill_triggered(
                        order.entry_price,
                        order.direction,
                        bid,
                        ask,
                        slippage,
                    )
                }
                ExecutionVariant::V1Parity => market_fill(order.entry_price, order.direction, 0.0),
            };

            if let Err(err) =
                Self::validate_min_sl_distance(fill_result.fill_price, order.stop_loss, costs)
            {
                self.pending_book.mark_rejected(order.id);
                rejected.push(PendingOrderRejection {
                    order_id: order.id,
                    reason: err.to_string(),
                });
                continue;
            }

            // Consume trade index only after validation
            self.trade_index = self.trade_index.wrapping_add(1);

            // Calculate entry fee
            let entry_fee = if self.config.apply_entry_fees && costs.apply_entry_fee {
                costs.fee.calculate(order.size, fill_result.fill_price)
            } else {
                0.0
            };

            fills.push(PendingFillResult {
                order_id: order.id,
                fill_price: fill_result.fill_price,
                slippage: fill_result.slippage_applied,
                entry_fee,
                order: order.clone(),
            });
        }

        // Mark filled orders
        for fill in &fills {
            self.pending_book.mark_filled(fill.order_id);
        }

        PendingProcessResult {
            triggered,
            fills,
            rejected,
        }
    }

    /// Calculates exit fee for closing a position.
    ///
    /// # Arguments
    /// * `size` - Position size
    /// * `exit_price` - Exit price
    /// * `costs` - Symbol-specific costs configuration
    #[must_use]
    pub fn calculate_exit_fee(&self, size: f64, exit_price: f64, costs: &SymbolCosts) -> f64 {
        if self.config.apply_exit_fees && costs.apply_exit_fee {
            costs.fee.calculate(size, exit_price)
        } else {
            0.0
        }
    }

    /// Applies exit slippage and fees for a position close.
    ///
    /// Exit slippage is applied with inverted direction (adverse exit).
    /// The `exit_price` should already reflect pip-buffer and entry-candle
    /// logic from the stop-check phase.
    pub fn apply_exit_slippage(
        &mut self,
        exit_price: f64,
        position_direction: Direction,
        size: f64,
        costs: &SymbolCosts,
    ) -> ExitFillResult {
        let slippage = self
            .preview_slippage(exit_price, position_direction, costs)
            .abs();
        let exit_direction = match self.config.execution_variant {
            ExecutionVariant::V2 => Self::invert_direction(position_direction),
            ExecutionVariant::V1Parity => position_direction,
        };
        let fill = market_fill(exit_price, exit_direction, slippage);

        self.trade_index = self.trade_index.wrapping_add(1);

        let exit_fee = if self.config.apply_exit_fees && costs.apply_exit_fee {
            costs.fee.calculate(size, fill.fill_price)
        } else {
            0.0
        };

        ExitFillResult {
            fill_price: fill.fill_price,
            slippage: fill.slippage_applied,
            exit_fee,
        }
    }

    /// Cancels a pending order.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        self.pending_book.cancel_order(order_id)
    }

    /// Gets a pending order by ID.
    #[must_use]
    pub fn get_pending_order(&self, order_id: u64) -> Option<&PendingOrder> {
        self.pending_book.get_order(order_id)
    }

    /// Returns the number of pending orders.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending_book.pending_count()
    }

    /// Returns the number of triggered (awaiting fill) orders.
    #[must_use]
    pub fn triggered_count(&self) -> usize {
        self.pending_book.triggered_count()
    }

    /// Cleans up terminal orders from the book.
    pub fn cleanup(&mut self) {
        self.pending_book.cleanup_terminal();
    }

    /// Resets the deterministic trade index (for reproducibility testing).
    pub fn reset_rng(&mut self) {
        self.trade_index = 0;
    }

    /// Gets a reference to the pending book.
    #[must_use]
    pub fn pending_book(&self) -> &PendingBook {
        &self.pending_book
    }

    /// Gets a mutable reference to the pending book.
    pub fn pending_book_mut(&mut self) -> &mut PendingBook {
        &mut self.pending_book
    }

    fn invert_direction(direction: Direction) -> Direction {
        match direction {
            Direction::Long => Direction::Short,
            Direction::Short => Direction::Long,
        }
    }

    fn preview_slippage(&self, price: f64, direction: Direction, costs: &SymbolCosts) -> f64 {
        let seed = self.config.rng_seed.wrapping_add(self.trade_index);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        costs.slippage.calculate(price, direction, &mut rng)
    }

    fn validate_min_sl_distance(
        entry_price: f64,
        stop_loss: f64,
        costs: &SymbolCosts,
    ) -> Result<(), ExecutionError> {
        let min_distance = costs.min_sl_distance_pips * costs.pip_size;
        let distance = (entry_price - stop_loss).abs();
        if distance < min_distance {
            return Err(ExecutionError::InvalidOrder(format!(
                "stop-loss distance {distance:.6} below minimum {min_distance:.6}"
            )));
        }
        Ok(())
    }

    fn quantize_volume(size: f64, costs: &SymbolCosts) -> Result<f64, ExecutionError> {
        if !size.is_finite() || size <= 0.0 {
            return Err(ExecutionError::InvalidOrder(
                "volume must be positive and finite".to_string(),
            ));
        }

        let step = costs.volume_step;
        let min = costs.volume_min;
        let max = costs.volume_max;

        if step <= 0.0 {
            return Err(ExecutionError::InvalidOrder(
                "volume_step must be positive".to_string(),
            ));
        }

        let steps = (size / step).floor();
        let mut quantized = steps * step;

        if quantized > max {
            quantized = max;
        }

        if quantized < min {
            return Err(ExecutionError::InvalidOrder(format!(
                "volume {quantized:.6} below min {min:.6}"
            )));
        }

        Ok(quantized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::costs::SymbolCosts;
    use crate::slippage::SlippageModel;
    use approx::assert_relative_eq;
    use omega_types::{Direction, ExecutionVariant, OrderType};
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
            close_time_ns: ts + 60_000_000_000 - 1,
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
        let result = engine.execute_market_order(&signal, &costs).unwrap();

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
        let order_id = engine.add_pending_order(&signal, 1000, &costs).unwrap();
        assert_eq!(order_id, 1);
        assert_eq!(engine.pending_count(), 1);

        // Process in same bar - should not trigger (next-bar rule)
        let bid = make_candle(1.2000, 1.2050, 1.1960, 1.2030, 1000);
        let ask = make_candle(1.2002, 1.2052, 1.1962, 1.2032, 1000);
        let result = engine.process_pending_orders(&bid, &ask, 1000, &costs);
        assert!(result.triggered.is_empty());
        assert!(result.fills.is_empty());

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

        let result1 = engine1.execute_market_order(&signal, &costs).unwrap();
        let result2 = engine2.execute_market_order(&signal, &costs).unwrap();

        assert_relative_eq!(result1.fill_price, result2.fill_price, epsilon = 1e-10);
        assert_relative_eq!(result1.slippage, result2.slippage, epsilon = 1e-10);
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = ExecutionEngine::with_seed(42);
        let costs = SymbolCosts::zero_cost(0.0001);

        let signal = make_signal(Direction::Long, OrderType::Limit, 1.1980);
        let order_id = engine.add_pending_order(&signal, 1000, &costs).unwrap();

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

    #[test]
    fn test_trade_index_deterministic_slippage() {
        let mut engine = ExecutionEngine::with_seed(10);
        let model = crate::slippage::VolatilitySlippage::new(1.0, 0.0001, 0.5);
        let costs = SymbolCosts {
            slippage: Box::new(model.clone()),
            fee: Box::new(crate::fees::NoFee),
            apply_entry_fee: false,
            apply_exit_fee: false,
            pip_size: 0.0001,
            pip_buffer_factor: 0.5,
            min_sl_distance_pips: 0.1,
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
        };

        let signal = make_signal(Direction::Long, OrderType::Market, 1.2000);

        let result1 = engine.execute_market_order(&signal, &costs).unwrap();
        let result2 = engine.execute_market_order(&signal, &costs).unwrap();

        let mut rng1 = ChaCha8Rng::seed_from_u64(10);
        let expected1 = model.calculate(signal.entry_price, signal.direction, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(11);
        let expected2 = model.calculate(signal.entry_price, signal.direction, &mut rng2);

        assert_relative_eq!(result1.slippage, expected1, epsilon = 1e-10);
        assert_relative_eq!(result2.slippage, expected2, epsilon = 1e-10);
    }

    #[test]
    fn test_volume_quantization_floor_and_min() {
        let mut engine = ExecutionEngine::with_seed(42);
        let costs = SymbolCosts {
            slippage: Box::new(crate::slippage::NoSlippage),
            fee: Box::new(crate::fees::NoFee),
            apply_entry_fee: false,
            apply_exit_fee: false,
            pip_size: 0.0001,
            pip_buffer_factor: 0.5,
            min_sl_distance_pips: 0.1,
            volume_min: 0.2,
            volume_step: 0.1,
            volume_max: 1.0,
        };

        let mut signal = make_signal(Direction::Long, OrderType::Market, 1.2000);
        signal.size = Some(0.26);

        let result = engine.execute_market_order(&signal, &costs).unwrap();
        assert_relative_eq!(result.size, 0.2, epsilon = 1e-12);

        signal.size = Some(0.05);
        let result = engine.execute_market_order(&signal, &costs);
        assert!(matches!(result, Err(ExecutionError::InvalidOrder(_))));
    }

    #[test]
    fn test_exit_slippage_inverted_direction() {
        let mut engine = ExecutionEngine::with_seed(42);
        let costs = SymbolCosts {
            slippage: Box::new(crate::slippage::FixedSlippage::new(1.0, 0.0001)),
            fee: Box::new(crate::fees::NoFee),
            apply_entry_fee: false,
            apply_exit_fee: false,
            pip_size: 0.0001,
            pip_buffer_factor: 0.5,
            min_sl_distance_pips: 0.1,
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
        };

        let long_exit = engine.apply_exit_slippage(1.2000, Direction::Long, 1.0, &costs);
        assert_relative_eq!(long_exit.fill_price, 1.1999, epsilon = 1e-10);

        let short_exit = engine.apply_exit_slippage(1.2000, Direction::Short, 1.0, &costs);
        assert_relative_eq!(short_exit.fill_price, 1.2001, epsilon = 1e-10);
    }

    #[test]
    fn test_pending_fill_v1_parity_ideal_fill() {
        let mut engine = ExecutionEngine::new(ExecutionEngineConfig {
            rng_seed: 42,
            apply_entry_fees: true,
            apply_exit_fees: true,
            execution_variant: ExecutionVariant::V1Parity,
        });
        let costs = SymbolCosts {
            slippage: Box::new(crate::slippage::FixedSlippage::new(1.0, 0.0001)),
            fee: Box::new(crate::fees::NoFee),
            apply_entry_fee: false,
            apply_exit_fee: false,
            pip_size: 0.0001,
            pip_buffer_factor: 0.5,
            min_sl_distance_pips: 0.1,
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
        };

        let signal = make_signal(Direction::Long, OrderType::Limit, 1.1980);
        engine.add_pending_order(&signal, 1000, &costs).unwrap();

        let bid = make_candle(1.2050, 1.2100, 1.1970, 1.2060, 2000);
        let ask = make_candle(1.2052, 1.2102, 1.1972, 1.2062, 2000);

        let result = engine.process_pending_orders(&bid, &ask, 2000, &costs);
        assert_eq!(result.fills.len(), 1);
        assert_relative_eq!(result.fills[0].fill_price, 1.1980, epsilon = 1e-10);
        assert_relative_eq!(result.fills[0].slippage, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exit_slippage_v1_parity_direction() {
        let mut engine = ExecutionEngine::new(ExecutionEngineConfig {
            rng_seed: 42,
            apply_entry_fees: true,
            apply_exit_fees: true,
            execution_variant: ExecutionVariant::V1Parity,
        });
        let costs = SymbolCosts {
            slippage: Box::new(crate::slippage::FixedSlippage::new(1.0, 0.0001)),
            fee: Box::new(crate::fees::NoFee),
            apply_entry_fee: false,
            apply_exit_fee: false,
            pip_size: 0.0001,
            pip_buffer_factor: 0.5,
            min_sl_distance_pips: 0.1,
            volume_min: 0.01,
            volume_step: 0.01,
            volume_max: 100.0,
        };

        let long_exit = engine.apply_exit_slippage(1.2000, Direction::Long, 1.0, &costs);
        assert_relative_eq!(long_exit.fill_price, 1.2001, epsilon = 1e-10);

        let short_exit = engine.apply_exit_slippage(1.2000, Direction::Short, 1.0, &costs);
        assert_relative_eq!(short_exit.fill_price, 1.1999, epsilon = 1e-10);
    }
}
