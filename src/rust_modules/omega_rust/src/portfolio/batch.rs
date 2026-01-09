//! Batch processing for portfolio operations.
//!
//! Provides high-performance batch processing of multiple portfolio operations
//! in a single FFI call, reducing Python↔Rust overhead significantly.
//!
//! ## Motivation
//!
//! With 20K trade events, sequential FFI calls create ~80K individual calls
//! with ~1-2µs overhead each. Batch processing reduces this to 1-2 calls total.
//!
//! ## Example
//!
//! ```python
//! from omega_rust import PortfolioRust, BatchOperation, BatchResult
//!
//! portfolio = PortfolioRust(initial_balance=100000.0)
//! operations = [
//!     {"type": "entry", "position": pos1_dict},
//!     {"type": "fee", "amount": 3.0, "time": 1704067200, "kind": "entry"},
//!     {"type": "exit", "position_idx": 0, "price": 1.102, "time": 1704067260},
//!     {"type": "update", "time": 1704067260},
//! ]
//! result = portfolio.process_batch(operations)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use super::position::PositionRust;

/// Represents a single batch operation.
///
/// Operations are processed sequentially in the order provided.
#[derive(Clone, Debug)]
pub enum BatchOperation {
    /// Register a new position entry
    RegisterEntry {
        position: PositionRust,
        fee: Option<f64>,
        fee_kind: Option<String>,
    },
    /// Register a position exit by index
    RegisterExit {
        position_idx: usize,
        price: f64,
        time: i64,
        reason: String,
        fee: Option<f64>,
    },
    /// Update portfolio state (equity, drawdown)
    Update { time: i64 },
    /// Register a standalone fee
    RegisterFee {
        amount: f64,
        time: i64,
        kind: String,
    },
}

/// Result of batch processing.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct BatchResult {
    /// Number of operations processed
    #[pyo3(get)]
    pub operations_processed: usize,
    /// Number of entries registered
    #[pyo3(get)]
    pub entries_registered: usize,
    /// Number of exits registered
    #[pyo3(get)]
    pub exits_registered: usize,
    /// Number of updates performed
    #[pyo3(get)]
    pub updates_performed: usize,
    /// Number of fees registered
    #[pyo3(get)]
    pub fees_registered: usize,
    /// Total fees collected
    #[pyo3(get)]
    pub total_fees: f64,
    /// Final equity after batch
    #[pyo3(get)]
    pub final_equity: f64,
    /// Final cash after batch
    #[pyo3(get)]
    pub final_cash: f64,
    /// Errors encountered (index, message)
    #[pyo3(get)]
    pub errors: Vec<(usize, String)>,
}

#[pymethods]
impl BatchResult {
    #[new]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchResult(ops={}, entries={}, exits={}, updates={}, fees={}, errors={})",
            self.operations_processed,
            self.entries_registered,
            self.exits_registered,
            self.updates_performed,
            self.fees_registered,
            self.errors.len()
        )
    }
}

impl BatchOperation {
    /// Parse a batch operation from a Python dict.
    ///
    /// Expected dict format:
    /// ```python
    /// # Entry operation
    /// {"type": "entry", "position": {...}, "fee": 3.0, "fee_kind": "entry"}
    ///
    /// # Exit operation
    /// {"type": "exit", "position_idx": 0, "price": 1.102, "time": 1704067260, "reason": "take_profit", "fee": 3.0}
    ///
    /// # Update operation
    /// {"type": "update", "time": 1704067260}
    ///
    /// # Fee operation
    /// {"type": "fee", "amount": 3.0, "time": 1704067200, "kind": "entry"}
    /// ```
    pub fn from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let op_type: String = dict
            .get_item("type")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'type' key in operation")
            })?
            .extract()?;

        match op_type.as_str() {
            "entry" => {
                let pos_dict = dict.get_item("position")?.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Missing 'position' key for entry operation",
                    )
                })?;
                let position = position_from_pydict(pos_dict.downcast::<PyDict>()?)?;
                let fee: Option<f64> = dict
                    .get_item("fee")?
                    .and_then(|v| v.extract().ok());
                let fee_kind: Option<String> = dict
                    .get_item("fee_kind")?
                    .and_then(|v| v.extract().ok());

                Ok(BatchOperation::RegisterEntry {
                    position,
                    fee,
                    fee_kind,
                })
            }
            "exit" => {
                let position_idx: usize = dict
                    .get_item("position_idx")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'position_idx' for exit operation",
                        )
                    })?
                    .extract()?;
                let price: f64 = dict
                    .get_item("price")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'price' for exit operation",
                        )
                    })?
                    .extract()?;
                let time: i64 = dict
                    .get_item("time")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'time' for exit operation",
                        )
                    })?
                    .extract()?;
                let reason: String = dict
                    .get_item("reason")?
                    .map_or_else(|| "signal".to_string(), |v| {
                        v.extract().unwrap_or_else(|_| "signal".to_string())
                    });
                let fee: Option<f64> = dict
                    .get_item("fee")?
                    .and_then(|v| v.extract().ok());

                Ok(BatchOperation::RegisterExit {
                    position_idx,
                    price,
                    time,
                    reason,
                    fee,
                })
            }
            "update" => {
                let time: i64 = dict
                    .get_item("time")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'time' for update operation",
                        )
                    })?
                    .extract()?;

                Ok(BatchOperation::Update { time })
            }
            "fee" => {
                let amount: f64 = dict
                    .get_item("amount")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'amount' for fee operation",
                        )
                    })?
                    .extract()?;
                let time: i64 = dict
                    .get_item("time")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'time' for fee operation",
                        )
                    })?
                    .extract()?;
                let kind: String = dict
                    .get_item("kind")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Missing 'kind' for fee operation",
                        )
                    })?
                    .extract()?;

                Ok(BatchOperation::RegisterFee { amount, time, kind })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown operation type: '{}'",
                op_type
            ))),
        }
    }
}

/// Helper to create a PositionRust from a Python dict.
fn position_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<PositionRust> {
    let entry_time: i64 = dict
        .get_item("entry_time")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'entry_time'"))?
        .extract()?;
    let direction: i8 = dict
        .get_item("direction")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'direction'"))?
        .extract()?;
    let symbol: String = dict
        .get_item("symbol")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'symbol'"))?
        .extract()?;
    let entry_price: f64 = dict
        .get_item("entry_price")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'entry_price'"))?
        .extract()?;
    let stop_loss: f64 = dict
        .get_item("stop_loss")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'stop_loss'"))?
        .extract()?;
    let take_profit: f64 = dict
        .get_item("take_profit")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'take_profit'"))?
        .extract()?;
    let size: f64 = dict
        .get_item("size")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'size'"))?
        .extract()?;
    let risk_per_trade: f64 = dict
        .get_item("risk_per_trade")?
        .map_or(100.0, |v| v.extract().unwrap_or(100.0));

    let mut pos = PositionRust::new(
        entry_time,
        direction,
        symbol,
        entry_price,
        stop_loss,
        take_profit,
        size,
        risk_per_trade,
    );

    // Optional fields
    if let Some(val) = dict.get_item("initial_stop_loss")? {
        if let Ok(v) = val.extract::<f64>() {
            pos.initial_stop_loss = Some(v);
        }
    }
    if let Some(val) = dict.get_item("initial_take_profit")? {
        if let Ok(v) = val.extract::<f64>() {
            pos.initial_take_profit = Some(v);
        }
    }
    if let Some(val) = dict.get_item("order_type")? {
        if let Ok(v) = val.extract::<String>() {
            pos.order_type = v;
        }
    }
    if let Some(val) = dict.get_item("status")? {
        if let Ok(v) = val.extract::<String>() {
            pos.status = v;
        }
    }

    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_result_default() {
        let result = BatchResult::new();
        assert_eq!(result.operations_processed, 0);
        assert_eq!(result.entries_registered, 0);
        assert_eq!(result.exits_registered, 0);
        assert_eq!(result.updates_performed, 0);
        assert_eq!(result.fees_registered, 0);
        assert!(result.errors.is_empty());
    }
}
