//! Type definitions for the Event Engine.
//!
//! Provides Rust representations of candle data, signals, and event loop statistics.
//! This is **Wave 3** of the Rust migration.
//!
//! ## Available Types
//!
//! - [`CandleData`] - Single candle representation with OHLCV data
//! - [`SignalDirection`] - Trade signal direction (Long/Short/None)
//! - [`TradeSignalRust`] - Complete trade signal with metadata
//! - [`EventEngineStats`] - Performance statistics for the event loop
//!
//! ## Reference
//!
//! - Implementation Plan: `docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md`
//! - FFI Specification: `docs/ffi/event_engine.md`

use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;

// =============================================================================
// CandleData - Represents a single OHLCV candle
// =============================================================================

/// Represents a single OHLCV candle with timestamp.
///
/// This is the Rust counterpart to Python's `Candle` class from
/// `backtest_engine.data.candle`.
#[derive(Clone, Debug)]
pub struct CandleData {
    /// Unix timestamp in microseconds (matches Python datetime precision)
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
}

impl CandleData {
    /// Creates a new CandleData instance.
    #[inline]
    pub const fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Extracts CandleData from a Python Candle object.
    ///
    /// # Arguments
    /// * `obj` - Python object with `timestamp`, `open`, `high`, `low`, `close`, `volume` attributes
    ///
    /// # Returns
    /// * `PyResult<Self>` - CandleData or conversion error
    pub fn from_pyobject(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Get timestamp - could be datetime or int
        let timestamp: i64 = if let Ok(ts) = obj.getattr("timestamp")?.extract::<i64>() {
            ts
        } else {
            // Try to get as datetime and convert to microseconds
            let dt = obj.getattr("timestamp")?;
            let ts_method = dt.call_method0("timestamp")?;
            let ts_float: f64 = ts_method.extract()?;
            (ts_float * 1_000_000.0) as i64
        };

        Ok(Self {
            timestamp,
            open: obj.getattr("open")?.extract()?,
            high: obj.getattr("high")?.extract()?,
            low: obj.getattr("low")?.extract()?,
            close: obj.getattr("close")?.extract()?,
            volume: obj.getattr("volume")?.extract().unwrap_or(0.0),
        })
    }

    /// Converts to a Python dictionary for FFI return values.
    pub fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("open", self.open)?;
        dict.set_item("high", self.high)?;
        dict.set_item("low", self.low)?;
        dict.set_item("close", self.close)?;
        dict.set_item("volume", self.volume)?;
        Ok(dict.into())
    }
}

// =============================================================================
// SignalDirection - Trade signal direction enum
// =============================================================================

/// Direction of a trade signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalDirection {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// No signal / neutral
    None,
}

impl SignalDirection {
    /// Converts from Python signal direction string/int.
    pub fn from_pyobject(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try string first
        if let Ok(s) = obj.extract::<String>() {
            return Ok(match s.to_lowercase().as_str() {
                "long" | "buy" => Self::Long,
                "short" | "sell" => Self::Short,
                _ => Self::None,
            });
        }

        // Try integer (1 = long, -1 = short, 0 = none)
        if let Ok(i) = obj.extract::<i32>() {
            return Ok(match i {
                1 => Self::Long,
                -1 => Self::Short,
                _ => Self::None,
            });
        }

        // Default to None
        Ok(Self::None)
    }

    /// Returns the numeric representation (1 = long, -1 = short, 0 = none).
    #[inline]
    pub const fn as_i32(self) -> i32 {
        match self {
            Self::Long => 1,
            Self::Short => -1,
            Self::None => 0,
        }
    }
}

// =============================================================================
// TradeSignalRust - Complete trade signal with metadata
// =============================================================================

/// Complete trade signal with all metadata.
///
/// Corresponds to Python's `TradeSignal` dataclass.
#[derive(Clone, Debug)]
pub struct TradeSignalRust {
    /// Signal direction (Long/Short/None)
    pub direction: SignalDirection,
    /// Target symbol (e.g., "EURUSD")
    pub symbol: String,
    /// Position size / lot size
    pub size: f64,
    /// Entry price (optional, may be None for market orders)
    pub entry_price: Option<f64>,
    /// Stop loss price (optional)
    pub stop_loss: Option<f64>,
    /// Take profit price (optional)
    pub take_profit: Option<f64>,
    /// Signal timestamp in microseconds
    pub timestamp: i64,
    /// Human-readable reason for the signal
    pub reason: Option<String>,
    /// Additional metadata as key-value pairs
    pub metadata: HashMap<String, String>,
}

impl TradeSignalRust {
    /// Extracts TradeSignalRust from a Python TradeSignal object.
    ///
    /// Returns `None` if the signal is None or has no direction.
    pub fn from_pyobject(obj: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        // Check if None
        if obj.is_none() {
            return Ok(None);
        }

        // Get direction
        let direction = if let Ok(dir_attr) = obj.getattr("direction") {
            SignalDirection::from_pyobject(&dir_attr)?
        } else {
            SignalDirection::None
        };

        // Skip if no direction
        if direction == SignalDirection::None {
            return Ok(None);
        }

        // Extract required fields
        let symbol: String = obj
            .getattr("symbol")
            .and_then(|s| s.extract())
            .unwrap_or_else(|_| "UNKNOWN".to_string());

        let size: f64 = obj
            .getattr("size")
            .and_then(|s| s.extract())
            .unwrap_or(1.0);

        // Extract optional fields
        let entry_price: Option<f64> = obj
            .getattr("entry_price")
            .ok()
            .and_then(|p| if p.is_none() { None } else { p.extract().ok() });

        let stop_loss: Option<f64> = obj
            .getattr("stop_loss")
            .ok()
            .and_then(|p| if p.is_none() { None } else { p.extract().ok() });

        let take_profit: Option<f64> = obj
            .getattr("take_profit")
            .ok()
            .and_then(|p| if p.is_none() { None } else { p.extract().ok() });

        let timestamp: i64 = obj
            .getattr("timestamp")
            .and_then(|t| t.extract())
            .unwrap_or(0);

        let reason: Option<String> = obj
            .getattr("reason")
            .ok()
            .and_then(|r| if r.is_none() { None } else { r.extract().ok() });

        Ok(Some(Self {
            direction,
            symbol,
            size,
            entry_price,
            stop_loss,
            take_profit,
            timestamp,
            reason,
            metadata: HashMap::new(),
        }))
    }
}

// =============================================================================
// EventEngineStats - Performance statistics for the event loop
// =============================================================================

/// Performance statistics for the event engine.
///
/// Tracks timing information and counts for profiling and optimization.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct EventEngineStats {
    /// Total number of bars processed
    #[pyo3(get)]
    pub bars_processed: usize,

    /// Number of signals generated
    #[pyo3(get)]
    pub signals_generated: usize,

    /// Number of trades executed
    #[pyo3(get)]
    pub trades_executed: usize,

    /// Number of exits processed
    #[pyo3(get)]
    pub exits_processed: usize,

    /// Total loop time in milliseconds
    #[pyo3(get)]
    pub loop_time_ms: f64,

    /// Total strategy callback time in milliseconds
    #[pyo3(get)]
    pub callback_time_ms: f64,

    /// Total execution processing time in milliseconds
    #[pyo3(get)]
    pub execution_time_ms: f64,

    /// Total portfolio update time in milliseconds
    #[pyo3(get)]
    pub portfolio_time_ms: f64,
}

#[pymethods]
impl EventEngineStats {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a summary string of the statistics.
    pub fn summary(&self) -> String {
        format!(
            "EventEngineStats(bars={}, signals={}, trades={}, exits={}, loop_ms={:.2}, callback_ms={:.2})",
            self.bars_processed,
            self.signals_generated,
            self.trades_executed,
            self.exits_processed,
            self.loop_time_ms,
            self.callback_time_ms
        )
    }

    fn __repr__(&self) -> String {
        self.summary()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_data_new() {
        let candle = CandleData::new(1704067200000000, 1.1, 1.15, 1.05, 1.12, 1000.0);
        assert_eq!(candle.timestamp, 1704067200000000);
        assert!((candle.open - 1.1).abs() < f64::EPSILON);
        assert!((candle.close - 1.12).abs() < f64::EPSILON);
    }

    #[test]
    fn test_signal_direction() {
        assert_eq!(SignalDirection::Long.as_i32(), 1);
        assert_eq!(SignalDirection::Short.as_i32(), -1);
        assert_eq!(SignalDirection::None.as_i32(), 0);
    }

    #[test]
    fn test_event_engine_stats_default() {
        let stats = EventEngineStats::default();
        assert_eq!(stats.bars_processed, 0);
        assert_eq!(stats.signals_generated, 0);
    }
}
