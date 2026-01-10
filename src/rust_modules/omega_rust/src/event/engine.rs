//! Event Engine Rust implementation for high-performance backtesting.
//!
//! Provides the main event loop that coordinates candle data, strategy evaluation,
//! execution, and portfolio updates. This is **Wave 3** of the Rust migration.
//!
//! ## Architecture (Hybrid: Rust Loop + Python Callbacks)
//!
//! The Event Engine runs the main backtest loop in Rust for performance,
//! while calling back to Python for strategy evaluation (which remains in Python
//! for compatibility with existing strategies).
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │  Rust Event Loop                                               │
//! │                                                                │
//! │  for i in start_index..total_bars:                             │
//! │      1. PREPARE: slice.set_index(i)                            │
//! │      2. STRATEGY: callback to Python strategy.evaluate()       │
//! │      3. PROCESS: handle signals (Rust/Python Executor)         │
//! │      4. EXITS: evaluate exit conditions                        │
//! │      5. PORTFOLIO: update portfolio state                      │
//! │      6. PROGRESS: optional callback for UI                     │
//! │                                                                │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/event_engine.md`

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use std::time::Instant;

use super::types::{CandleData, EventEngineStats};

// =============================================================================
// EventEngineRust - Main Event Loop Implementation
// =============================================================================

/// Rust implementation of EventEngine for high-performance backtesting.
///
/// This struct holds the candle data and coordinates the main event loop.
/// Strategy evaluation is delegated to Python callbacks for compatibility.
///
/// # Example (from Python)
///
/// ```python
/// from omega_rust import EventEngineRust
///
/// engine = EventEngineRust(
///     bid_candles=bid_list,
///     ask_candles=ask_list,
///     start_index=100,
///     symbol="EURUSD"
/// )
///
/// stats = engine.run(
///     strategy_callback=strategy.evaluate,
///     executor=execution_simulator,
///     portfolio=portfolio,
///     slice_obj=symbol_slice,
///     progress_callback=on_progress
/// )
/// ```
#[pyclass]
pub struct EventEngineRust {
    /// Bid candle data
    bid_candles: Vec<CandleData>,
    /// Ask candle data
    ask_candles: Vec<CandleData>,
    /// Index to start processing (after warmup)
    start_index: usize,
    /// Total number of bars
    total_bars: usize,
    /// Symbol being processed
    symbol: String,
    /// Current bar index during processing
    current_index: usize,
    /// Statistics for performance monitoring
    stats: EventEngineStats,
}

#[pymethods]
impl EventEngineRust {
    /// Creates a new EventEngineRust instance.
    ///
    /// # Arguments
    ///
    /// * `bid_candles` - List of bid candles as Python objects
    /// * `ask_candles` - List of ask candles as Python objects
    /// * `start_index` - Index to start processing (after warmup period)
    /// * `symbol` - Trading symbol (e.g., "EURUSD")
    ///
    /// # Returns
    ///
    /// A new EventEngineRust instance ready to run.
    #[new]
    #[pyo3(signature = (bid_candles, ask_candles, start_index, symbol))]
    pub fn new(
        bid_candles: &Bound<'_, PyList>,
        ask_candles: &Bound<'_, PyList>,
        start_index: usize,
        symbol: String,
    ) -> PyResult<Self> {
        let total_bars = bid_candles.len();

        if total_bars != ask_candles.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Bid and ask candle counts must match: {} vs {}",
                total_bars,
                ask_candles.len()
            )));
        }

        if start_index >= total_bars {
            return Err(PyRuntimeError::new_err(format!(
                "start_index ({}) must be less than total_bars ({})",
                start_index, total_bars
            )));
        }

        // Convert Python candles to Rust representation
        let mut bid_vec = Vec::with_capacity(total_bars);
        let mut ask_vec = Vec::with_capacity(total_bars);

        for i in 0..total_bars {
            let bid_obj = bid_candles.get_item(i)?;
            let ask_obj = ask_candles.get_item(i)?;

            bid_vec.push(CandleData::from_pyobject(&bid_obj)?);
            ask_vec.push(CandleData::from_pyobject(&ask_obj)?);
        }

        Ok(Self {
            bid_candles: bid_vec,
            ask_candles: ask_vec,
            start_index,
            total_bars,
            symbol,
            current_index: start_index,
            stats: EventEngineStats::default(),
        })
    }

    /// Returns the total number of bars.
    #[getter]
    pub fn total_bars(&self) -> usize {
        self.total_bars
    }

    /// Returns the start index (after warmup).
    #[getter]
    pub fn start_index(&self) -> usize {
        self.start_index
    }

    /// Returns the current processing index.
    #[getter]
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Returns the symbol being processed.
    #[getter]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Returns the current statistics.
    #[getter]
    pub fn stats(&self) -> EventEngineStats {
        self.stats.clone()
    }

    /// Runs the main event loop.
    ///
    /// This is the core method that executes the backtest. It iterates through
    /// all candles (from start_index to total_bars), calling the strategy
    /// for signals and processing executions.
    ///
    /// # Arguments
    ///
    /// * `strategy_callback` - Python callable: `fn(index, slice_map) -> Optional[Signal|List[Signal]]`
    /// * `executor` - Python ExecutionSimulator object with `process_signal` and `evaluate_exits` methods
    /// * `portfolio` - Python Portfolio object with `update` method
    /// * `slice_obj` - Python SymbolDataSlice object with `set_index` method
    /// * `progress_callback` - Optional Python callable: `fn(current, total) -> None`
    /// * `position_mgmt_callback` - Optional Python callable for position management
    ///
    /// # Returns
    ///
    /// EventEngineStats with performance metrics.
    #[pyo3(signature = (strategy_callback, executor, portfolio, slice_obj, progress_callback=None, position_mgmt_callback=None))]
    pub fn run(
        &mut self,
        py: Python<'_>,
        strategy_callback: Py<PyAny>,
        executor: Py<PyAny>,
        portfolio: Py<PyAny>,
        slice_obj: Py<PyAny>,
        progress_callback: Option<Py<PyAny>>,
        position_mgmt_callback: Option<Py<PyAny>>,
    ) -> PyResult<EventEngineStats> {
        let loop_start = Instant::now();

        // Reset stats
        self.stats = EventEngineStats::default();

        // Create slice_map with symbol -> slice_obj
        // We need to create it inside the loop to avoid move issues
        // But we can create the key once
        let symbol_key = self.symbol.clone();

        // Main event loop
        for i in self.start_index..self.total_bars {
            self.current_index = i;

            // 1. PREPARE: Update slice index
            slice_obj.call_method1(py, "set_index", (i,))?;

            // Get current candles for this bar
            let bid_candle = &self.bid_candles[i];
            let ask_candle = &self.ask_candles[i];
            let timestamp = bid_candle.timestamp;

            // Create slice_map for this iteration
            let slice_map = PyDict::new(py);
            slice_map.set_item(&symbol_key, &slice_obj)?;

            // 2. STRATEGY: Call Python strategy.evaluate(index, slice_map)
            let callback_start = Instant::now();
            let signals_result = strategy_callback.call1(py, (i, slice_map))?;
            self.stats.callback_time_ms += callback_start.elapsed().as_secs_f64() * 1000.0;

            // 3. PROCESS: Handle signals
            if !signals_result.is_none(py) {
                let exec_start = Instant::now();

                // Check if it's a list or single signal
                let signals_list = if signals_result.bind(py).is_instance_of::<PyList>() {
                    signals_result.bind(py).downcast::<PyList>()?.to_owned()
                } else {
                    // Single signal - wrap in list
                    let list = PyList::empty(py);
                    list.append(&signals_result)?;
                    list
                };

                for signal in signals_list.iter() {
                    if !signal.is_none() {
                        executor.call_method1(py, "process_signal", (signal,))?;
                        self.stats.signals_generated += 1;
                        self.stats.trades_executed += 1;
                    }
                }

                self.stats.execution_time_ms += exec_start.elapsed().as_secs_f64() * 1000.0;
            }

            // 4. EXITS: Check for exits on active positions
            // Get active_positions from executor
            let active_positions = executor.getattr(py, "active_positions")?;
            if !active_positions.is_none(py) {
                // Check if list is not empty
                let positions_list = active_positions.bind(py);
                if let Ok(len) = positions_list.len() {
                    if len > 0 {
                        let exec_start = Instant::now();

                        // Convert Rust candles to Python objects for evaluate_exits
                        let bid_dict = bid_candle.to_pydict(py)?;
                        let ask_dict = ask_candle.to_pydict(py)?;

                        // Create simple Python objects from dicts for evaluate_exits
                        // The executor expects Candle objects, so we pass the raw candle data
                        executor.call_method1(
                            py,
                            "evaluate_exits_from_dict",
                            (bid_dict, ask_dict),
                        ).or_else(|_| {
                            // Fallback: If evaluate_exits_from_dict doesn't exist,
                            // we need to convert back to Python Candle objects.
                            // This is less efficient but maintains compatibility.
                            let bid_py = self.candle_to_pyobject(py, bid_candle)?;
                            let ask_py = self.candle_to_pyobject(py, ask_candle)?;
                            executor.call_method1(py, "evaluate_exits", (bid_py, ask_py))
                        })?;

                        self.stats.exits_processed += 1;
                        self.stats.execution_time_ms += exec_start.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }

            // 5. POSITION MANAGEMENT: Call Python callback if provided
            if let Some(ref pm_callback) = position_mgmt_callback {
                // Check if executor has active positions
                let active_positions = executor.getattr(py, "active_positions")?;
                if !active_positions.is_none(py) {
                    if let Ok(len) = active_positions.bind(py).len() {
                        if len > 0 {
                            // Call position management callback with (slice_obj, bid_candle, ask_candle)
                            let bid_py = self.candle_to_pyobject(py, bid_candle)?;
                            let ask_py = self.candle_to_pyobject(py, ask_candle)?;
                            pm_callback.call1(py, (&slice_obj, bid_py, ask_py))?;
                        }
                    }
                }
            }

            // 6. PORTFOLIO: Update portfolio state
            let portfolio_start = Instant::now();
            portfolio.call_method1(py, "update", (timestamp,))?;
            self.stats.portfolio_time_ms += portfolio_start.elapsed().as_secs_f64() * 1000.0;

            // 7. PROGRESS: Report progress if callback provided
            if let Some(ref callback) = progress_callback {
                let current = (i - self.start_index) + 1;
                let total = self.total_bars - self.start_index;
                callback.call1(py, (current, total))?;
            }

            self.stats.bars_processed += 1;
        }

        self.stats.loop_time_ms = loop_start.elapsed().as_secs_f64() * 1000.0;

        Ok(self.stats.clone())
    }

    /// Gets the bid candle at the specified index.
    ///
    /// # Arguments
    /// * `index` - Bar index
    ///
    /// # Returns
    /// Python dict with candle data
    pub fn get_bid_candle(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        if index >= self.total_bars {
            return Err(PyRuntimeError::new_err(format!(
                "Index {} out of bounds (total: {})",
                index, self.total_bars
            )));
        }
        self.bid_candles[index].to_pydict(py)
    }

    /// Gets the ask candle at the specified index.
    ///
    /// # Arguments
    /// * `index` - Bar index
    ///
    /// # Returns
    /// Python dict with candle data
    pub fn get_ask_candle(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        if index >= self.total_bars {
            return Err(PyRuntimeError::new_err(format!(
                "Index {} out of bounds (total: {})",
                index, self.total_bars
            )));
        }
        self.ask_candles[index].to_pydict(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "EventEngineRust(symbol={}, total_bars={}, start_index={}, current_index={})",
            self.symbol, self.total_bars, self.start_index, self.current_index
        )
    }
}

// Private helper methods
impl EventEngineRust {
    /// Converts a CandleData to a Python Candle object.
    ///
    /// This is used when we need to pass candles to Python methods that
    /// expect Candle objects rather than dicts.
    fn candle_to_pyobject(&self, py: Python<'_>, candle: &CandleData) -> PyResult<Py<PyAny>> {
        // Import the Candle class
        let candle_module = py.import("backtest_engine.data.candle")?;
        let candle_class = candle_module.getattr("Candle")?;

        // Convert timestamp from microseconds back to timezone-aware UTC datetime
        // This is critical for comparison with other timestamps in the backtest
        let datetime_module = py.import("datetime")?;
        let timezone_module = datetime_module.getattr("timezone")?;
        let utc = timezone_module.getattr("utc")?;

        // Create UTC-aware datetime using fromtimestamp with tz parameter
        let dt = datetime_module.getattr("datetime")?.call_method(
            "fromtimestamp",
            (candle.timestamp as f64 / 1_000_000.0,),
            Some(&[("tz", utc)].into_py_dict(py)?),
        )?;

        // Create Candle instance
        let candle_obj = candle_class.call1((
            dt,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        ))?;

        Ok(candle_obj.into())
    }
}

// =============================================================================
// CrossSymbolEventEngineRust - Multi-Symbol Event Loop (Placeholder)
// =============================================================================

/// Rust implementation of CrossSymbolEventEngine for multi-symbol backtesting.
///
/// **Note:** This is a placeholder for future implementation. Multi-symbol
/// backtesting has additional complexity around timestamp synchronization
/// and will be implemented after single-symbol is validated.
#[pyclass]
pub struct CrossSymbolEventEngineRust {
    // Placeholder fields
    symbols: Vec<String>,
    total_timestamps: usize,
}

#[pymethods]
impl CrossSymbolEventEngineRust {
    #[new]
    pub fn new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "CrossSymbolEventEngineRust is not yet implemented. Use Python CrossSymbolEventEngine.",
        ))
    }
}

// =============================================================================
// Helper function for active backend verification
// =============================================================================

/// Returns "rust" to indicate the Rust backend is active.
///
/// This function is used by CI to verify that the Rust backend is actually
/// being used when the feature flag is enabled (Wave 1 Learning).
#[pyfunction]
pub fn get_event_engine_backend() -> &'static str {
    "rust"
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_engine_stats_summary() {
        let mut stats = EventEngineStats::default();
        stats.bars_processed = 100;
        stats.signals_generated = 10;
        stats.loop_time_ms = 50.0;

        let summary = stats.summary();
        assert!(summary.contains("bars=100"));
        assert!(summary.contains("signals=10"));
    }
}
