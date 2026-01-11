//! Execution Simulator - main PyO3 binding.
//!
//! Provides the [`ExecutionSimulatorRust`] class which is exposed to Python
//! and handles all execution logic: signal processing, entry triggers, and
//! exit evaluation.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::position::{Direction, OrderType, Position};
use super::signal::TradeSignal;
use super::sizing::{SymbolSpec, SymbolSpecCache};
use super::slippage::SlippageCalculator;
use super::trigger::{check_entry_trigger, evaluate_exit, Candle, ExitResult, TriggerResult};
use crate::error::Result;

/// Rust implementation of the Execution Simulator.
///
/// This class manages the full lifecycle of trading positions:
/// - Processing trade signals (market, limit, stop orders)
/// - Tracking pending and open positions
/// - Evaluating entry triggers for pending orders
/// - Detecting stop-loss and take-profit hits
/// - Position sizing and volume quantization
///
/// ## Python Usage
///
/// ```python
/// from omega_rust import ExecutionSimulatorRust
///
/// simulator = ExecutionSimulatorRust(
///     risk_per_trade=100.0,
///     pip_buffer_factor=0.5,
/// )
///
/// # Add symbol specifications
/// simulator.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0)
///
/// # Process signals (via Arrow IPC in production)
/// # simulator.process_signals_batch(signals_ipc)
///
/// # Evaluate exits
/// # simulator.evaluate_exits_batch(bid_candle_ipc, ask_candle_ipc)
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct ExecutionSimulatorRust {
    /// Active positions (pending and open)
    positions: Vec<Position>,

    /// Symbol specifications cache
    symbol_specs: SymbolSpecCache,

    /// Risk amount per trade in account currency
    risk_per_trade: f64,

    /// Pip buffer factor for SL/TP detection (default: 0.5)
    pip_buffer_factor: f64,

    /// Next position ID (monotonically increasing)
    next_position_id: u64,

    /// Slippage calculator (uses base_seed for deterministic RNG)
    slippage_calculator: Option<SlippageCalculator>,

    /// Closed positions (for export)
    closed_positions: Vec<Position>,

    /// Total entry fees accumulated
    total_entry_fees: f64,

    /// Total exit fees accumulated
    total_exit_fees: f64,
}

#[pymethods]
impl ExecutionSimulatorRust {
    /// Create a new execution simulator.
    ///
    /// # Arguments
    /// * `risk_per_trade` - Risk amount per trade in account currency (default: 100.0)
    /// * `pip_buffer_factor` - Buffer factor for SL/TP detection (default: 0.5)
    /// * `base_seed` - Optional seed for deterministic slippage RNG
    /// * `max_slippage_pips` - Maximum slippage in pips (default: 1.0)
    #[new]
    #[pyo3(signature = (risk_per_trade=100.0, pip_buffer_factor=0.5, base_seed=None, max_slippage_pips=1.0))]
    pub fn new(
        risk_per_trade: f64,
        pip_buffer_factor: f64,
        base_seed: Option<u64>,
        max_slippage_pips: f64,
    ) -> Self {
        let slippage_calculator = base_seed.map(|seed| SlippageCalculator::new(seed, max_slippage_pips));

        Self {
            positions: Vec::new(),
            symbol_specs: SymbolSpecCache::new(),
            risk_per_trade,
            pip_buffer_factor,
            next_position_id: 1,
            slippage_calculator,
            closed_positions: Vec::new(),
            total_entry_fees: 0.0,
            total_exit_fees: 0.0,
        }
    }

    /// Add a symbol specification.
    ///
    /// # Arguments
    /// * `symbol` - Symbol name (e.g., "EURUSD")
    /// * `pip_size` - Pip size (e.g., 0.0001)
    /// * `contract_size` - Contract size (e.g., 100000)
    /// * `volume_min` - Minimum volume in lots
    /// * `volume_step` - Volume step in lots
    /// * `volume_max` - Maximum volume in lots
    /// * `tick_size` - Optional tick size
    /// * `tick_value` - Optional tick value
    #[pyo3(signature = (symbol, pip_size, contract_size, volume_min, volume_step, volume_max, tick_size=None, tick_value=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_symbol_spec(
        &mut self,
        symbol: &str,
        pip_size: f64,
        contract_size: f64,
        volume_min: f64,
        volume_step: f64,
        volume_max: f64,
        tick_size: Option<f64>,
        tick_value: Option<f64>,
    ) {
        let spec = SymbolSpec {
            symbol: symbol.to_string(),
            pip_size,
            contract_size,
            tick_size,
            tick_value,
            volume_min,
            volume_max,
            volume_step,
            quote_currency: None,
            base_currency: None,
        };
        self.symbol_specs.insert(spec);
    }

    /// Get the number of active positions (pending + open).
    #[getter]
    pub fn active_position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get the number of open positions.
    #[getter]
    pub fn open_position_count(&self) -> usize {
        self.positions.iter().filter(|p| p.is_open()).count()
    }

    /// Get the number of pending positions.
    #[getter]
    pub fn pending_position_count(&self) -> usize {
        self.positions.iter().filter(|p| p.is_pending()).count()
    }

    /// Get the number of closed positions.
    #[getter]
    pub fn closed_position_count(&self) -> usize {
        self.closed_positions.len()
    }

    /// Get total accumulated entry fees.
    #[getter]
    pub fn total_entry_fees(&self) -> f64 {
        self.total_entry_fees
    }

    /// Get total accumulated exit fees.
    #[getter]
    pub fn total_exit_fees(&self) -> f64 {
        self.total_exit_fees
    }

    /// Clear all positions (for reset/restart).
    pub fn clear(&mut self) {
        self.positions.clear();
        self.closed_positions.clear();
        self.next_position_id = 1;
        self.total_entry_fees = 0.0;
        self.total_exit_fees = 0.0;
    }

    /// Process a single signal (for testing/debugging).
    ///
    /// In production, use `process_signals_batch` with Arrow IPC.
    #[pyo3(signature = (timestamp_us, symbol, direction, order_type, entry_price, stop_loss, take_profit))]
    #[allow(clippy::too_many_arguments)]
    pub fn process_signal_single(
        &mut self,
        timestamp_us: i64,
        symbol: &str,
        direction: &str,
        order_type: &str,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
    ) -> PyResult<u64> {
        let dir = Direction::from_str(direction)?;
        let ot = OrderType::from_str(order_type)?;

        let signal = TradeSignal::new(
            timestamp_us,
            symbol.to_string(),
            dir,
            ot,
            entry_price,
            stop_loss,
            take_profit,
        );

        self.process_signal_internal(signal)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Evaluate exits for a single candle (for testing/debugging).
    ///
    /// In production, use `evaluate_exits_batch` with Arrow IPC.
    #[pyo3(signature = (timestamp_us, bid_open, bid_high, bid_low, bid_close, ask_open=None, ask_high=None, ask_low=None, ask_close=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_exits_single(
        &mut self,
        timestamp_us: i64,
        bid_open: f64,
        bid_high: f64,
        bid_low: f64,
        bid_close: f64,
        ask_open: Option<f64>,
        ask_high: Option<f64>,
        ask_low: Option<f64>,
        ask_close: Option<f64>,
    ) -> usize {
        let bid_candle = Candle::new(timestamp_us, bid_open, bid_high, bid_low, bid_close, 0.0);

        let ask_candle = if let (Some(o), Some(h), Some(l), Some(c)) =
            (ask_open, ask_high, ask_low, ask_close)
        {
            Some(Candle::new(timestamp_us, o, h, l, c, 0.0))
        } else {
            None
        };

        self.evaluate_exits_internal(&bid_candle, ask_candle.as_ref())
    }

    // =========================================================================
    // Arrow IPC Batch APIs
    // =========================================================================

    /// Process signals from Arrow IPC bytes.
    ///
    /// # Arguments
    /// * `signals_ipc` - Arrow IPC stream containing trade signals
    ///
    /// # Returns
    /// Number of signals processed
    pub fn process_signals_batch(&mut self, signals_ipc: &Bound<'_, PyBytes>) -> PyResult<usize> {
        let ipc_bytes = signals_ipc.as_bytes();

        // Decode Arrow IPC to trade signals
        let signals = crate::execution::arrow::decode_trade_signals(ipc_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let count = signals.len();

        // Process each signal
        for signal in signals {
            self.process_signal_internal(signal)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        Ok(count)
    }

    /// Evaluate exits from Arrow IPC candle data.
    ///
    /// # Arguments
    /// * `bid_candles_ipc` - Arrow IPC stream containing bid candles
    /// * `ask_candles_ipc` - Arrow IPC stream containing ask candles (optional)
    ///
    /// # Returns
    /// Number of positions closed
    pub fn evaluate_exits_batch(
        &mut self,
        bid_candles_ipc: &Bound<'_, PyBytes>,
        ask_candles_ipc: Option<&Bound<'_, PyBytes>>,
    ) -> PyResult<usize> {
        // Decode bid candles from Arrow IPC
        let bid_candles = crate::execution::arrow::decode_candles(bid_candles_ipc.as_bytes())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Decode ask candles if provided
        let ask_candles = if let Some(ask_ipc) = ask_candles_ipc {
            Some(
                crate::execution::arrow::decode_candles(ask_ipc.as_bytes())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let mut total_closed = 0;

        // Process each candle pair
        for (idx, bid_candle) in bid_candles.iter().enumerate() {
            // Get corresponding ask candle if available
            let ask_candle = ask_candles.as_ref().and_then(|v| v.get(idx));

            let closed = self.evaluate_exits_internal(bid_candle, ask_candle);
            total_closed += closed;
        }

        Ok(total_closed)
    }

    /// Export active positions as Arrow IPC bytes.
    pub fn get_active_positions_ipc<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let ipc_bytes = crate::execution::arrow::encode_positions(&self.positions)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyBytes::new(py, &ipc_bytes))
    }

    /// Export closed positions as Arrow IPC bytes.
    pub fn get_closed_positions_ipc<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let ipc_bytes = crate::execution::arrow::encode_positions(&self.closed_positions)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyBytes::new(py, &ipc_bytes))
    }
}

impl ExecutionSimulatorRust {
    /// Internal signal processing logic.
    fn process_signal_internal(&mut self, signal: TradeSignal) -> Result<u64> {
        // Validate signal
        signal.validate()?;

        // Get symbol spec for sizing and pip_size
        let spec = self.symbol_specs.get(&signal.symbol);
        let pip_size = spec.map(|s| s.pip_size).unwrap_or(0.0001);

        // Create position
        let position_id = self.next_position_id;
        self.next_position_id += 1;

        // Apply entry slippage for market orders
        let entry_price = if signal.order_type == OrderType::Market {
            if let Some(ref slippage_calc) = self.slippage_calculator {
                slippage_calc.apply_entry_slippage(
                    signal.entry_price,
                    pip_size,
                    signal.direction,
                    position_id,
                    signal.timestamp_us,
                )
            } else {
                signal.entry_price
            }
        } else {
            signal.entry_price
        };

        let mut position = Position::new_pending(
            position_id,
            signal.timestamp_us,
            signal.symbol.clone(),
            signal.direction,
            signal.order_type,
            entry_price,
            signal.stop_loss,
            signal.take_profit,
            self.risk_per_trade,
        );

        // Copy metadata (clone to avoid move)
        position.metadata_json = signal.metadata_json.clone();

        // For market orders, calculate size and activate immediately
        if signal.order_type == OrderType::Market {
            let sl_distance = (entry_price - signal.stop_loss).abs();
            let size = if let Some(s) = spec {
                s.size_for_risk(self.risk_per_trade, sl_distance)
                    .unwrap_or(0.01)
            } else {
                // Fallback sizing
                0.01
            };
            position.size = size;
            position.trigger_time_us = Some(signal.timestamp_us);
        }

        self.positions.push(position);
        Ok(position_id)
    }

    /// Internal exit evaluation logic.
    fn evaluate_exits_internal(
        &mut self,
        bid_candle: &Candle,
        ask_candle: Option<&Candle>,
    ) -> usize {
        let mut closed_count = 0;

        // First pass: check pending triggers
        for pos in &mut self.positions {
            if pos.is_pending() {
                let result = check_entry_trigger(pos, bid_candle, ask_candle);
                if result == TriggerResult::Triggered {
                    // Get symbol spec for sizing and pip_size
                    let spec = self.symbol_specs.get(&pos.symbol);
                    let pip_size = spec.map(|s| s.pip_size).unwrap_or(0.0001);

                    // Apply entry slippage for pending orders upon trigger
                    let slipped_entry = if let Some(ref slippage_calc) = self.slippage_calculator {
                        slippage_calc.apply_entry_slippage(
                            pos.entry_price,
                            pip_size,
                            pos.direction,
                            pos.id,
                            bid_candle.timestamp_us,
                        )
                    } else {
                        pos.entry_price
                    };
                    pos.entry_price = slipped_entry;

                    // Calculate size and activate
                    let sl_distance = (slipped_entry - pos.stop_loss).abs();
                    let size = if let Some(s) = spec {
                        s.size_for_risk(self.risk_per_trade, sl_distance)
                            .unwrap_or(0.01)
                    } else {
                        0.01
                    };
                    pos.activate(bid_candle.timestamp_us, size);
                }
            }
        }

        // Second pass: check exits for open positions
        let pip_buffer = self.calculate_pip_buffer(&self.positions);

        let mut to_close = Vec::new();
        for (idx, pos) in self.positions.iter().enumerate() {
            if pos.is_open() {
                let result = evaluate_exit(pos, bid_candle, ask_candle, pip_buffer);
                if let ExitResult::Exit { price, reason } = result {
                    // Apply exit slippage
                    let pip_size = self.symbol_specs.pip_size(&pos.symbol);
                    let slipped_exit = if let Some(ref slippage_calc) = self.slippage_calculator {
                        slippage_calc.apply_exit_slippage(
                            price,
                            pip_size,
                            pos.direction,
                            pos.id,
                            bid_candle.timestamp_us,
                        )
                    } else {
                        price
                    };
                    to_close.push((idx, slipped_exit, reason));
                }
            }
        }

        // Apply closes (reverse order to maintain indices)
        for (idx, exit_price, reason) in to_close.into_iter().rev() {
            self.positions[idx].close(bid_candle.timestamp_us, exit_price, reason);
            closed_count += 1;
        }

        // Move closed positions
        let (closed, active): (Vec<_>, Vec<_>) =
            self.positions.drain(..).partition(|p| p.is_closed());

        self.positions = active;
        self.closed_positions.extend(closed);

        closed_count
    }

    /// Calculate pip buffer based on first position's symbol.
    fn calculate_pip_buffer(&self, positions: &[Position]) -> f64 {
        if let Some(first) = positions.first() {
            let pip_size = self.symbol_specs.pip_size(&first.symbol);
            pip_size * self.pip_buffer_factor
        } else {
            0.0001 * self.pip_buffer_factor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simulator without slippage for deterministic tests
    fn create_simulator(risk_per_trade: f64, pip_buffer_factor: f64) -> ExecutionSimulatorRust {
        ExecutionSimulatorRust::new(risk_per_trade, pip_buffer_factor, None, 1.0)
    }

    #[test]
    fn test_simulator_creation() {
        let sim = create_simulator(100.0, 0.5);
        assert_eq!(sim.active_position_count(), 0);
        assert!((sim.risk_per_trade - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_symbol_spec() {
        let mut sim = create_simulator(100.0, 0.5);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        assert!(sim.symbol_specs.get("EURUSD").is_some());
    }

    #[test]
    fn test_process_market_signal() {
        let mut sim = create_simulator(100.0, 0.5);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );

        let id = sim.process_signal_internal(signal).unwrap();
        assert_eq!(id, 1);
        assert_eq!(sim.active_position_count(), 1);
        assert_eq!(sim.open_position_count(), 1);
    }

    #[test]
    fn test_process_limit_signal() {
        let mut sim = create_simulator(100.0, 0.5);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Limit,
            1.1000,
            1.0950,
            1.1100,
        );

        let id = sim.process_signal_internal(signal).unwrap();
        assert_eq!(id, 1);
        assert_eq!(sim.pending_position_count(), 1);
        assert_eq!(sim.open_position_count(), 0);
    }

    #[test]
    fn test_evaluate_exits() {
        let mut sim = create_simulator(100.0, 0.0); // No pip buffer for test
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        // Create and process a market order
        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950, // SL
            1.1100, // TP
        );
        sim.process_signal_internal(signal).unwrap();

        // Candle that hits SL
        let bid = Candle::new(1704067260_000_000, 1.1000, 1.1010, 1.0940, 1.0960, 100.0);

        let closed = sim.evaluate_exits_internal(&bid, None);
        assert_eq!(closed, 1);
        assert_eq!(sim.closed_position_count(), 1);
        assert_eq!(sim.open_position_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut sim = create_simulator(100.0, 0.5);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );
        sim.process_signal_internal(signal).unwrap();

        sim.clear();
        assert_eq!(sim.active_position_count(), 0);
        assert_eq!(sim.closed_position_count(), 0);
    }

    #[test]
    fn test_slippage_applied_to_market_order() {
        // Create simulator with slippage enabled
        let mut sim = ExecutionSimulatorRust::new(100.0, 0.0, Some(12345), 1.0);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );

        sim.process_signal_internal(signal).unwrap();

        // Entry price should be slipped (higher for long)
        let pos = &sim.positions[0];
        assert!(pos.entry_price > 1.1000, "Long entry should be slipped higher");
        assert!(pos.entry_price < 1.1002, "Slippage should be bounded by max_pips");
    }

    #[test]
    fn test_slippage_deterministic() {
        // Two simulators with same seed should produce same slippage
        let mut sim1 = ExecutionSimulatorRust::new(100.0, 0.0, Some(42), 1.0);
        let mut sim2 = ExecutionSimulatorRust::new(100.0, 0.0, Some(42), 1.0);

        sim1.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);
        sim2.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal1 = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );
        let signal2 = signal1.clone();

        sim1.process_signal_internal(signal1).unwrap();
        sim2.process_signal_internal(signal2).unwrap();

        let entry1 = sim1.positions[0].entry_price;
        let entry2 = sim2.positions[0].entry_price;

        assert!((entry1 - entry2).abs() < 1e-15, "Same seed should produce same slippage");
    }

    #[test]
    fn test_no_slippage_when_seed_none() {
        let mut sim = create_simulator(100.0, 0.0);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
        );

        sim.process_signal_internal(signal).unwrap();

        // Without slippage, entry price should be exact
        let pos = &sim.positions[0];
        assert!((pos.entry_price - 1.1000).abs() < 1e-10, "No slippage should mean exact price");
    }

    #[test]
    fn test_short_entry_slippage_decreases_price() {
        let mut sim = ExecutionSimulatorRust::new(100.0, 0.0, Some(12345), 1.0);
        sim.add_symbol_spec("EURUSD", 0.0001, 100000.0, 0.01, 0.01, 100.0, None, None);

        let signal = TradeSignal::new(
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Short,
            OrderType::Market,
            1.1000,
            1.1050, // SL above for short
            1.0900, // TP below for short
        );

        sim.process_signal_internal(signal).unwrap();

        // Entry price should be slipped lower for short (unfavorable)
        let pos = &sim.positions[0];
        assert!(pos.entry_price < 1.1000, "Short entry should be slipped lower");
    }
}
