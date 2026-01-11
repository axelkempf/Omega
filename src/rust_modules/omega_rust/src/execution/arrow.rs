//! Arrow IPC serialization for execution data.
//!
//! Provides utilities for encoding and decoding trade signals,
//! positions, and candles using Apache Arrow IPC format for
//! zero-copy transfer between Python and Rust.
//!
//! ## Schema Registry
//!
//! Schemas are defined in `src/shared/arrow_schemas.py` and must be
//! kept in sync with the Rust implementations here.
//!
//! ## Schema Version: 2.0.0
//!
//! Dictionary encoding uses int32 index type for scalability.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, DictionaryArray, Float64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, TimeUnit};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;

use arrow::array::BooleanArray;

use crate::error::{OmegaError, Result};
use crate::execution::position::{Direction, OrderType, Position};
use crate::execution::signal::TradeSignal;
use crate::execution::trigger::Candle;

/// TRADE_SIGNAL_SCHEMA field names (from arrow_schemas.py v2.0.0)
const SIGNAL_FIELD_TIMESTAMP: &str = "timestamp";
const SIGNAL_FIELD_DIRECTION: &str = "direction";
const SIGNAL_FIELD_ENTRY: &str = "entry";
const SIGNAL_FIELD_SL: &str = "sl";
const SIGNAL_FIELD_TP: &str = "tp";
const SIGNAL_FIELD_SIZE: &str = "size";
const SIGNAL_FIELD_SYMBOL: &str = "symbol";
const SIGNAL_FIELD_ORDER_TYPE: &str = "order_type";
const SIGNAL_FIELD_REASON: &str = "reason";
const SIGNAL_FIELD_SCENARIO: &str = "scenario";

/// POSITION_SCHEMA field names
const POS_FIELD_ENTRY_TIME: &str = "entry_time";
const POS_FIELD_EXIT_TIME: &str = "exit_time";
const POS_FIELD_DIRECTION: &str = "direction";
const POS_FIELD_SYMBOL: &str = "symbol";
const POS_FIELD_ENTRY_PRICE: &str = "entry_price";
const POS_FIELD_EXIT_PRICE: &str = "exit_price";
const POS_FIELD_INITIAL_SL: &str = "initial_sl";
const POS_FIELD_CURRENT_SL: &str = "current_sl";
const POS_FIELD_TP: &str = "tp";
const POS_FIELD_SIZE: &str = "size";
const POS_FIELD_RESULT: &str = "result";
const POS_FIELD_R_MULTIPLE: &str = "r_multiple";
const POS_FIELD_STATUS: &str = "status";

/// OHLCV_SCHEMA field names (from arrow_schemas.py v2.0.0)
const OHLCV_FIELD_TIMESTAMP: &str = "timestamp";
const OHLCV_FIELD_OPEN: &str = "open";
const OHLCV_FIELD_HIGH: &str = "high";
const OHLCV_FIELD_LOW: &str = "low";
const OHLCV_FIELD_CLOSE: &str = "close";
const OHLCV_FIELD_VOLUME: &str = "volume";
const OHLCV_FIELD_VALID: &str = "valid";

/// Build the expected OHLCV_SCHEMA for candle data.
pub fn build_ohlcv_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            OHLCV_FIELD_TIMESTAMP,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(OHLCV_FIELD_OPEN, DataType::Float64, false),
        Field::new(OHLCV_FIELD_HIGH, DataType::Float64, false),
        Field::new(OHLCV_FIELD_LOW, DataType::Float64, false),
        Field::new(OHLCV_FIELD_CLOSE, DataType::Float64, false),
        Field::new(OHLCV_FIELD_VOLUME, DataType::Float64, false),
        Field::new(OHLCV_FIELD_VALID, DataType::Boolean, false),
    ])
}

/// Build the expected TRADE_SIGNAL_SCHEMA for validation.
pub fn build_signal_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            SIGNAL_FIELD_TIMESTAMP,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            SIGNAL_FIELD_DIRECTION,
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(SIGNAL_FIELD_ENTRY, DataType::Float64, false),
        Field::new(SIGNAL_FIELD_SL, DataType::Float64, false),
        Field::new(SIGNAL_FIELD_TP, DataType::Float64, false),
        Field::new(SIGNAL_FIELD_SIZE, DataType::Float64, false),
        Field::new(
            SIGNAL_FIELD_SYMBOL,
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            SIGNAL_FIELD_ORDER_TYPE,
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(SIGNAL_FIELD_REASON, DataType::Utf8, true),
        Field::new(SIGNAL_FIELD_SCENARIO, DataType::Utf8, true),
    ])
}

/// Build the POSITION_SCHEMA for encoding positions.
pub fn build_position_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            POS_FIELD_ENTRY_TIME,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            POS_FIELD_EXIT_TIME,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            POS_FIELD_DIRECTION,
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            POS_FIELD_SYMBOL,
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(POS_FIELD_ENTRY_PRICE, DataType::Float64, false),
        Field::new(POS_FIELD_EXIT_PRICE, DataType::Float64, true),
        Field::new(POS_FIELD_INITIAL_SL, DataType::Float64, false),
        Field::new(POS_FIELD_CURRENT_SL, DataType::Float64, false),
        Field::new(POS_FIELD_TP, DataType::Float64, false),
        Field::new(POS_FIELD_SIZE, DataType::Float64, false),
        Field::new(POS_FIELD_RESULT, DataType::Float64, true),
        Field::new(POS_FIELD_R_MULTIPLE, DataType::Float64, true),
        Field::new(
            POS_FIELD_STATUS,
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
    ])
}

/// Helper to extract string from dictionary array.
fn get_dict_string(array: &DictionaryArray<Int32Type>, idx: usize) -> Result<String> {
    let values = array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            OmegaError::ArrowError("Dictionary values must be StringArray".to_string())
        })?;

    let key = array.key(idx).ok_or_else(|| {
        OmegaError::ArrowError(format!("Null key at index {}", idx))
    })?;

    Ok(values.value(key as usize).to_string())
}

/// Helper to get optional string from nullable StringArray.
fn get_optional_string(array: &StringArray, idx: usize) -> Option<String> {
    if array.is_null(idx) {
        None
    } else {
        Some(array.value(idx).to_string())
    }
}

/// Decode trade signals from Arrow IPC bytes.
///
/// Expects IPC stream format with TRADE_SIGNAL_SCHEMA (v2.0.0).
pub fn decode_trade_signals(ipc_bytes: &[u8]) -> Result<Vec<TradeSignal>> {
    let cursor = Cursor::new(ipc_bytes);
    let reader = StreamReader::try_new(cursor, None).map_err(|e| {
        OmegaError::ArrowError(format!("Failed to create StreamReader: {}", e))
    })?;

    let mut signals = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            OmegaError::ArrowError(format!("Failed to read batch: {}", e))
        })?;

        let num_rows = batch.num_rows();

        // Extract columns by name
        let timestamp_col = batch
            .column_by_name(SIGNAL_FIELD_TIMESTAMP)
            .ok_or_else(|| OmegaError::ArrowError("Missing timestamp column".into()))?
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| OmegaError::ArrowError("timestamp must be Timestamp[us]".into()))?;

        let direction_col = batch
            .column_by_name(SIGNAL_FIELD_DIRECTION)
            .ok_or_else(|| OmegaError::ArrowError("Missing direction column".into()))?
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .ok_or_else(|| OmegaError::ArrowError("direction must be Dict<Int32,Utf8>".into()))?;

        let entry_col = batch
            .column_by_name(SIGNAL_FIELD_ENTRY)
            .ok_or_else(|| OmegaError::ArrowError("Missing entry column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("entry must be Float64".into()))?;

        let sl_col = batch
            .column_by_name(SIGNAL_FIELD_SL)
            .ok_or_else(|| OmegaError::ArrowError("Missing sl column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("sl must be Float64".into()))?;

        let tp_col = batch
            .column_by_name(SIGNAL_FIELD_TP)
            .ok_or_else(|| OmegaError::ArrowError("Missing tp column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("tp must be Float64".into()))?;

        let size_col = batch
            .column_by_name(SIGNAL_FIELD_SIZE)
            .ok_or_else(|| OmegaError::ArrowError("Missing size column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("size must be Float64".into()))?;

        let symbol_col = batch
            .column_by_name(SIGNAL_FIELD_SYMBOL)
            .ok_or_else(|| OmegaError::ArrowError("Missing symbol column".into()))?
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .ok_or_else(|| OmegaError::ArrowError("symbol must be Dict<Int32,Utf8>".into()))?;

        let order_type_col = batch
            .column_by_name(SIGNAL_FIELD_ORDER_TYPE)
            .ok_or_else(|| OmegaError::ArrowError("Missing order_type column".into()))?
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .ok_or_else(|| OmegaError::ArrowError("order_type must be Dict<Int32,Utf8>".into()))?;

        let reason_col = batch
            .column_by_name(SIGNAL_FIELD_REASON)
            .ok_or_else(|| OmegaError::ArrowError("Missing reason column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| OmegaError::ArrowError("reason must be Utf8".into()))?;

        let scenario_col = batch
            .column_by_name(SIGNAL_FIELD_SCENARIO)
            .ok_or_else(|| OmegaError::ArrowError("Missing scenario column".into()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| OmegaError::ArrowError("scenario must be Utf8".into()))?;

        for idx in 0..num_rows {
            let direction_str = get_dict_string(direction_col, idx)?;
            let order_type_str = get_dict_string(order_type_col, idx)?;

            let direction = Direction::from_str(&direction_str)?;
            let order_type = OrderType::from_str(&order_type_str)?;

            let mut signal = TradeSignal::new(
                timestamp_col.value(idx),
                get_dict_string(symbol_col, idx)?,
                direction,
                order_type,
                entry_col.value(idx),
                sl_col.value(idx),
                tp_col.value(idx),
            );

            signal.size = size_col.value(idx);
            signal.reason = get_optional_string(reason_col, idx);
            signal.scenario = get_optional_string(scenario_col, idx);

            signals.push(signal);
        }
    }

    Ok(signals)
}

/// Decode candles from Arrow IPC bytes.
///
/// Expects IPC stream format with OHLCV_SCHEMA (v2.0.0).
/// Only returns candles where `valid` is true.
pub fn decode_candles(ipc_bytes: &[u8]) -> Result<Vec<Candle>> {
    let cursor = Cursor::new(ipc_bytes);
    let reader = StreamReader::try_new(cursor, None).map_err(|e| {
        OmegaError::ArrowError(format!("Failed to create StreamReader for candles: {}", e))
    })?;

    let mut candles = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            OmegaError::ArrowError(format!("Failed to read candle batch: {}", e))
        })?;

        let num_rows = batch.num_rows();

        // Extract columns by name
        let timestamp_col = batch
            .column_by_name(OHLCV_FIELD_TIMESTAMP)
            .ok_or_else(|| OmegaError::ArrowError("Missing timestamp column".into()))?
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| OmegaError::ArrowError("timestamp must be Timestamp[us]".into()))?;

        let open_col = batch
            .column_by_name(OHLCV_FIELD_OPEN)
            .ok_or_else(|| OmegaError::ArrowError("Missing open column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("open must be Float64".into()))?;

        let high_col = batch
            .column_by_name(OHLCV_FIELD_HIGH)
            .ok_or_else(|| OmegaError::ArrowError("Missing high column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("high must be Float64".into()))?;

        let low_col = batch
            .column_by_name(OHLCV_FIELD_LOW)
            .ok_or_else(|| OmegaError::ArrowError("Missing low column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("low must be Float64".into()))?;

        let close_col = batch
            .column_by_name(OHLCV_FIELD_CLOSE)
            .ok_or_else(|| OmegaError::ArrowError("Missing close column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("close must be Float64".into()))?;

        let volume_col = batch
            .column_by_name(OHLCV_FIELD_VOLUME)
            .ok_or_else(|| OmegaError::ArrowError("Missing volume column".into()))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| OmegaError::ArrowError("volume must be Float64".into()))?;

        let valid_col = batch
            .column_by_name(OHLCV_FIELD_VALID)
            .ok_or_else(|| OmegaError::ArrowError("Missing valid column".into()))?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| OmegaError::ArrowError("valid must be Boolean".into()))?;

        for idx in 0..num_rows {
            // Skip invalid candles (valid == false means None/NaN data)
            if !valid_col.value(idx) {
                continue;
            }

            let candle = Candle::new(
                timestamp_col.value(idx),
                open_col.value(idx),
                high_col.value(idx),
                low_col.value(idx),
                close_col.value(idx),
                volume_col.value(idx),
            );

            candles.push(candle);
        }
    }

    Ok(candles)
}

/// Encode positions to Arrow IPC bytes.
pub fn encode_positions(positions: &[Position]) -> Result<Vec<u8>> {
    use arrow::array::{
        ArrayRef, Float64Builder,
        StringDictionaryBuilder,
    };
    use arrow::array::TimestampMicrosecondBuilder;

    let num_rows = positions.len();
    let schema = Arc::new(build_position_schema());

    // Build arrays using proper timestamp types (not Int64!)
    // This fixes the Arrow Schema Mismatch: Timestamp(Microsecond, UTC) vs Int64
    let mut entry_time_builder =
        TimestampMicrosecondBuilder::with_capacity(num_rows).with_timezone("UTC");
    let mut exit_time_builder =
        TimestampMicrosecondBuilder::with_capacity(num_rows).with_timezone("UTC");
    let mut direction_builder: StringDictionaryBuilder<Int32Type> =
        StringDictionaryBuilder::new();
    let mut symbol_builder: StringDictionaryBuilder<Int32Type> =
        StringDictionaryBuilder::new();
    let mut entry_price_builder = Float64Builder::with_capacity(num_rows);
    let mut exit_price_builder = Float64Builder::with_capacity(num_rows);
    let mut initial_sl_builder = Float64Builder::with_capacity(num_rows);
    let mut current_sl_builder = Float64Builder::with_capacity(num_rows);
    let mut tp_builder = Float64Builder::with_capacity(num_rows);
    let mut size_builder = Float64Builder::with_capacity(num_rows);
    let mut result_builder = Float64Builder::with_capacity(num_rows);
    let mut r_multiple_builder = Float64Builder::with_capacity(num_rows);
    let mut status_builder: StringDictionaryBuilder<Int32Type> =
        StringDictionaryBuilder::new();

    for pos in positions {
        entry_time_builder.append_value(pos.entry_time_us);

        if let Some(exit_time) = pos.exit_time_us {
            exit_time_builder.append_value(exit_time);
        } else {
            exit_time_builder.append_null();
        }

        direction_builder.append_value(pos.direction.as_str());
        symbol_builder.append_value(&pos.symbol);
        entry_price_builder.append_value(pos.entry_price);

        if let Some(exit_price) = pos.exit_price {
            exit_price_builder.append_value(exit_price);
        } else {
            exit_price_builder.append_null();
        }

        initial_sl_builder.append_value(pos.initial_stop_loss);
        current_sl_builder.append_value(pos.stop_loss);
        tp_builder.append_value(pos.take_profit);
        size_builder.append_value(pos.size);

        if let Some(result) = pos.result {
            result_builder.append_value(result);
        } else {
            result_builder.append_null();
        }

        // r_multiple requires unit_value for calculation, use stored result / risk
        // For now, we compute a simplified version or leave as None
        // Full calculation requires unit_value from SymbolSpec
        r_multiple_builder.append_null();

        status_builder.append_value(pos.status.as_str());
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(entry_time_builder.finish()),
        Arc::new(exit_time_builder.finish()),
        Arc::new(direction_builder.finish()),
        Arc::new(symbol_builder.finish()),
        Arc::new(entry_price_builder.finish()),
        Arc::new(exit_price_builder.finish()),
        Arc::new(initial_sl_builder.finish()),
        Arc::new(current_sl_builder.finish()),
        Arc::new(tp_builder.finish()),
        Arc::new(size_builder.finish()),
        Arc::new(result_builder.finish()),
        Arc::new(r_multiple_builder.finish()),
        Arc::new(status_builder.finish()),
    ];

    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| {
        OmegaError::ArrowError(format!("Failed to create RecordBatch: {}", e))
    })?;

    // Write to IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).map_err(|e| {
            OmegaError::ArrowError(format!("Failed to create StreamWriter: {}", e))
        })?;

        writer.write(&batch).map_err(|e| {
            OmegaError::ArrowError(format!("Failed to write batch: {}", e))
        })?;

        writer.finish().map_err(|e| {
            OmegaError::ArrowError(format!("Failed to finish stream: {}", e))
        })?;
    }

    Ok(buffer)
}

/// Schema validation result.
#[derive(Clone, Debug)]
pub struct SchemaValidation {
    /// Whether the schema is valid
    pub valid: bool,
    /// Error message if invalid
    pub error: Option<String>,
    /// Schema fingerprint
    pub fingerprint: Option<String>,
}

impl SchemaValidation {
    /// Create a valid result.
    pub fn valid(fingerprint: String) -> Self {
        Self {
            valid: true,
            error: None,
            fingerprint: Some(fingerprint),
        }
    }

    /// Create an invalid result.
    pub fn invalid(error: impl Into<String>) -> Self {
        Self {
            valid: false,
            error: Some(error.into()),
            fingerprint: None,
        }
    }
}

/// Validate schema fingerprint against expected.
///
/// Computes a fingerprint of the schema and compares it.
pub fn validate_schema(schema: &Schema, expected_fingerprint: &str) -> SchemaValidation {
    let fingerprint = compute_schema_fingerprint(schema);
    if fingerprint == expected_fingerprint {
        SchemaValidation::valid(fingerprint)
    } else {
        SchemaValidation::invalid(format!(
            "Schema fingerprint mismatch: expected '{}', got '{}'",
            expected_fingerprint, fingerprint
        ))
    }
}

/// Compute a fingerprint for a schema.
///
/// Simple hash based on field names and types.
fn compute_schema_fingerprint(schema: &Schema) -> String {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    for field in schema.fields() {
        field.name().hash(&mut hasher);
        format!("{:?}", field.data_type()).hash(&mut hasher);
        field.is_nullable().hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::position::Direction;

    #[test]
    fn test_signal_schema_fields() {
        let schema = build_signal_schema();
        assert_eq!(schema.fields().len(), 10);
        assert!(schema.field_with_name(SIGNAL_FIELD_TIMESTAMP).is_ok());
        assert!(schema.field_with_name(SIGNAL_FIELD_DIRECTION).is_ok());
        assert!(schema.field_with_name(SIGNAL_FIELD_SYMBOL).is_ok());
    }

    #[test]
    fn test_position_schema_fields() {
        let schema = build_position_schema();
        assert_eq!(schema.fields().len(), 13);
        assert!(schema.field_with_name(POS_FIELD_ENTRY_TIME).is_ok());
        assert!(schema.field_with_name(POS_FIELD_STATUS).is_ok());
    }

    #[test]
    fn test_encode_empty_positions() {
        let positions: Vec<Position> = vec![];
        let ipc_bytes = encode_positions(&positions).unwrap();
        assert!(!ipc_bytes.is_empty()); // IPC header is always written
    }

    #[test]
    fn test_encode_single_position() {
        use crate::execution::position::OrderType;

        let mut pos = Position::new_pending(
            1,
            1704067200_000_000,
            "EURUSD".to_string(),
            Direction::Long,
            OrderType::Market,
            1.1000,
            1.0950,
            1.1100,
            100.0,
        );
        pos.size = 0.1;

        let ipc_bytes = encode_positions(&[pos]).unwrap();
        assert!(!ipc_bytes.is_empty());
    }

    #[test]
    fn test_schema_fingerprint_consistency() {
        let schema1 = build_signal_schema();
        let schema2 = build_signal_schema();

        let fp1 = compute_schema_fingerprint(&schema1);
        let fp2 = compute_schema_fingerprint(&schema2);

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_ohlcv_schema_fields() {
        let schema = build_ohlcv_schema();
        assert_eq!(schema.fields().len(), 7);
        assert!(schema.field_with_name(OHLCV_FIELD_TIMESTAMP).is_ok());
        assert!(schema.field_with_name(OHLCV_FIELD_OPEN).is_ok());
        assert!(schema.field_with_name(OHLCV_FIELD_HIGH).is_ok());
        assert!(schema.field_with_name(OHLCV_FIELD_LOW).is_ok());
        assert!(schema.field_with_name(OHLCV_FIELD_CLOSE).is_ok());
        assert!(schema.field_with_name(OHLCV_FIELD_VOLUME).is_ok());
        assert!(schema.field_with_name(OHLCV_FIELD_VALID).is_ok());
    }

    #[test]
    fn test_ohlcv_schema_fingerprint_consistency() {
        let schema1 = build_ohlcv_schema();
        let schema2 = build_ohlcv_schema();

        let fp1 = compute_schema_fingerprint(&schema1);
        let fp2 = compute_schema_fingerprint(&schema2);

        assert_eq!(fp1, fp2);
    }
}
