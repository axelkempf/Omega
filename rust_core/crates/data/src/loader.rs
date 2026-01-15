use std::collections::HashMap;
use std::path::{Path, PathBuf};

use arrow::array::{Float64Array, Int64Array, TimestampNanosecondArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::alignment::align_bid_ask;
use crate::error::DataError;
use crate::validation::{validate_candles, validate_spread};
use omega_types::Candle;

/// Resolve a Parquet candle path using the canonical layout or an env override.
pub fn resolve_data_path(symbol: &str, timeframe: &str, side: &str) -> PathBuf {
    let root =
        std::env::var("OMEGA_DATA_PARQUET_ROOT").unwrap_or_else(|_| "data/parquet".to_string());

    PathBuf::from(root)
        .join(symbol)
        .join(format!("{symbol}_{timeframe}_{side}.parquet"))
}

/// Loads candles from a Parquet file with schema:
/// `UTC time` (timestamp ns), `Open`, `High`, `Low`, `Close`, `Volume`.
/// Duplicates with identical OHLCV are deduplicated (keep-first); divergent duplicates error.
pub fn load_candles(path: &Path) -> Result<Vec<Candle>, DataError> {
    let file = std::fs::File::open(path)
        .map_err(|e| DataError::FileNotFound(path.display().to_string(), e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    let reader = builder
        .build()
        .map_err(|e| DataError::ParseError(e.to_string()))?;

    let mut candles = Vec::new();
    let mut seen: HashMap<i64, Candle> = HashMap::new();
    let mut last_ts: Option<i64> = None;
    let mut processed_rows = 0usize;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| DataError::ParseError(e.to_string()))?;

        let ts_col = batch
            .column_by_name("UTC time")
            .ok_or_else(|| DataError::MissingColumn("UTC time".to_string()))?;
        let ts_arr = ts_col
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .ok_or_else(|| DataError::InvalidColumnType("UTC time".to_string()))?;

        let open_arr = numeric_f64_column(&batch, "Open")?;
        let high_arr = numeric_f64_column(&batch, "High")?;
        let low_arr = numeric_f64_column(&batch, "Low")?;
        let close_arr = numeric_f64_column(&batch, "Close")?;
        let volume_arr = numeric_f64_or_i64_column(&batch, "Volume")?;

        for row_idx in 0..batch.num_rows() {
            let ts = ts_arr.value(row_idx);
            if let Some(prev) = last_ts
                && ts < prev
            {
                return Err(DataError::CorruptData(format!(
                    "Out-of-order timestamp at row {}: {} < {}",
                    processed_rows + row_idx,
                    ts,
                    prev
                )));
            }

            let candle = Candle {
                timestamp_ns: ts,
                open: open_arr.value(row_idx),
                high: high_arr.value(row_idx),
                low: low_arr.value(row_idx),
                close: close_arr.value(row_idx),
                volume: volume_arr.value(row_idx),
            };

            match seen.get(&ts) {
                Some(existing) => {
                    if !same_candle(existing, &candle) {
                        return Err(DataError::CorruptData(format!(
                            "Divergent duplicate timestamp {} at row {}",
                            ts,
                            processed_rows + row_idx
                        )));
                    }
                }
                None => {
                    seen.insert(ts, candle);
                    last_ts = Some(ts);
                    candles.push(candle);
                }
            }
        }

        processed_rows += batch.num_rows();
    }

    if candles.is_empty() {
        return Err(DataError::EmptyData);
    }

    Ok(candles)
}

/// Convenience: load and validate candles.
pub fn load_and_validate(path: &Path) -> Result<Vec<Candle>, DataError> {
    let candles = load_candles(path)?;
    validate_candles(&candles)?;
    Ok(candles)
}

/// Convenience: load, validate, align, and validate spread for a bid/ask pair.
pub fn load_and_validate_bid_ask(
    bid_path: &Path,
    ask_path: &Path,
) -> Result<crate::alignment::AlignedData, DataError> {
    let bid = load_and_validate(bid_path)?;
    let ask = load_and_validate(ask_path)?;

    let aligned = align_bid_ask(bid, ask)?;
    validate_spread(&aligned.bid, &aligned.ask)?;
    Ok(aligned)
}

fn same_candle(a: &Candle, b: &Candle) -> bool {
    a.open == b.open
        && a.high == b.high
        && a.low == b.low
        && a.close == b.close
        && a.volume == b.volume
}

enum NumericAccessor<'a> {
    F64(&'a Float64Array),
    I64(&'a Int64Array),
}

impl<'a> NumericAccessor<'a> {
    fn value(&self, idx: usize) -> f64 {
        match self {
            NumericAccessor::F64(arr) => arr.value(idx),
            NumericAccessor::I64(arr) => arr.value(idx) as f64,
        }
    }
}

fn numeric_f64_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<&'a Float64Array, DataError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| DataError::MissingColumn(name.to_string()))?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| DataError::InvalidColumnType(name.to_string()))
}

fn numeric_f64_or_i64_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<NumericAccessor<'a>, DataError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| DataError::MissingColumn(name.to_string()))?;

    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        Ok(NumericAccessor::F64(arr))
    } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        Ok(NumericAccessor::I64(arr))
    } else {
        Err(DataError::InvalidColumnType(name.to_string()))
    }
}
