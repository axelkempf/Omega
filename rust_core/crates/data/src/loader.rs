//! Parquet loading and date-range filtering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use arrow::array::{Float64Array, Int64Array, TimestampNanosecondArray};
use arrow::datatypes::{DataType, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::alignment::align_bid_ask;
use crate::error::DataError;
use crate::validation::{validate_candles, validate_spread};
use omega_types::{Candle, Timeframe};

/// Resolve a Parquet candle path using the canonical layout or an env override.
#[must_use]
pub fn resolve_data_path(symbol: &str, timeframe: &str, side: &str) -> PathBuf {
    let root =
        std::env::var("OMEGA_DATA_PARQUET_ROOT").unwrap_or_else(|_| "data/parquet".to_string());

    PathBuf::from(root)
        .join(symbol)
        .join(format!("{symbol}_{timeframe}_{side}.parquet"))
}

/// Loads candles from a Parquet file with schema:
/// `UTC time` (timestamp ns, UTC), `Open`, `High`, `Low`, `Close`, `Volume`.
///
/// Duplicates with identical OHLCV are deduplicated (keep-first); divergent duplicates error.
///
/// # Errors
/// - [`DataError::FileNotFound`] when the file cannot be opened.
/// - [`DataError::ParseError`] when Parquet decoding fails.
/// - [`DataError::MissingColumn`] or [`DataError::InvalidColumnType`] when schema is invalid.
/// - [`DataError::InvalidTimezone`] when `UTC time` is missing timezone or not UTC.
/// - [`DataError::CorruptData`] for out-of-order or divergent duplicates.
/// - [`DataError::EmptyData`] when no rows are loaded.
pub fn load_candles(path: &Path, timeframe: Timeframe) -> Result<Vec<Candle>, DataError> {
    let file = std::fs::File::open(path)
        .map_err(|e| DataError::FileNotFound(path.display().to_string(), e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    let reader = builder
        .build()
        .map_err(|e| DataError::ParseError(e.to_string()))?;

    let duration_ns = timeframe_duration_ns(timeframe)?;
    let mut candles = Vec::new();
    let mut seen: HashMap<i64, Candle> = HashMap::new();
    let mut last_ts: Option<i64> = None;
    let mut processed_rows = 0usize;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| DataError::ParseError(e.to_string()))?;

        let ts_col = batch
            .column_by_name("UTC time")
            .ok_or_else(|| DataError::MissingColumn("UTC time".to_string()))?;
        ensure_utc_timezone("UTC time", ts_col.data_type())?;
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

            let close_time_ns = close_time_from_open(ts, duration_ns, processed_rows + row_idx)?;
            let candle = Candle {
                timestamp_ns: ts,
                close_time_ns,
                open: open_arr.value(row_idx),
                high: high_arr.value(row_idx),
                low: low_arr.value(row_idx),
                close: close_arr.value(row_idx),
                volume: volume_arr.value(row_idx)?,
            };

            if let Some(existing) = seen.get(&ts) {
                if !same_candle(existing, &candle) {
                    return Err(DataError::CorruptData(format!(
                        "Divergent duplicate timestamp {ts} at row {}",
                        processed_rows + row_idx
                    )));
                }
            } else {
                seen.insert(ts, candle);
                last_ts = Some(ts);
                candles.push(candle);
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
///
/// # Errors
/// - Propagates [`load_candles`] errors.
/// - Propagates [`validate_candles`] errors.
pub fn load_and_validate(path: &Path, timeframe: Timeframe) -> Result<Vec<Candle>, DataError> {
    let candles = load_candles(path, timeframe)?;
    validate_candles(&candles)?;
    Ok(candles)
}

/// Convenience: load, validate, align, and validate spread for a bid/ask pair.
///
/// # Errors
/// - Propagates [`load_and_validate`] errors.
/// - Propagates [`align_bid_ask`] errors.
/// - Propagates [`validate_spread`] errors.
pub fn load_and_validate_bid_ask(
    bid_path: &Path,
    ask_path: &Path,
    timeframe: Timeframe,
) -> Result<crate::alignment::AlignedData, DataError> {
    let bid = load_and_validate(bid_path, timeframe)?;
    let ask = load_and_validate(ask_path, timeframe)?;

    let aligned = align_bid_ask(bid, ask)?;
    validate_spread(&aligned.bid, &aligned.ask)?;
    Ok(aligned)
}

/// Filter candles to the inclusive `[start_ns, end_ns]` range.
///
/// # Errors
/// - [`DataError::CorruptData`] when `start_ns > end_ns`.
/// - [`DataError::DateRangeEmpty`] when the filter yields no rows.
pub fn filter_by_date_range(
    candles: &[Candle],
    start_ns: i64,
    end_ns: i64,
) -> Result<Vec<Candle>, DataError> {
    if start_ns > end_ns {
        return Err(DataError::CorruptData(format!(
            "Invalid date range: start_ns={start_ns} > end_ns={end_ns}"
        )));
    }

    let filtered: Vec<Candle> = candles
        .iter()
        .copied()
        .filter(|c| c.timestamp_ns >= start_ns && c.timestamp_ns <= end_ns)
        .collect();

    if filtered.is_empty() {
        return Err(DataError::DateRangeEmpty { start_ns, end_ns });
    }

    Ok(filtered)
}

fn same_candle(a: &Candle, b: &Candle) -> bool {
    a.close_time_ns == b.close_time_ns
        && a.open.to_bits() == b.open.to_bits()
        && a.high.to_bits() == b.high.to_bits()
        && a.low.to_bits() == b.low.to_bits()
        && a.close.to_bits() == b.close.to_bits()
        && a.volume.to_bits() == b.volume.to_bits()
}

fn timeframe_duration_ns(timeframe: Timeframe) -> Result<i64, DataError> {
    let seconds = i64::try_from(timeframe.to_seconds()).map_err(|_| {
        DataError::CorruptData(format!(
            "Timeframe seconds overflow: {}",
            timeframe.as_str()
        ))
    })?;
    seconds
        .checked_mul(1_000_000_000)
        .ok_or_else(|| {
            DataError::CorruptData(format!(
                "Timeframe duration overflow: {}",
                timeframe.as_str()
            ))
        })
}

fn close_time_from_open(
    open_ns: i64,
    duration_ns: i64,
    row_idx: usize,
) -> Result<i64, DataError> {
    open_ns
        .checked_add(duration_ns)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            DataError::CorruptData(format!(
                "Close time overflow at row {row_idx}: open_ns={open_ns}, duration_ns={duration_ns}"
            ))
        })
}

enum NumericAccessor<'a> {
    F64(&'a Float64Array),
    I64(&'a Int64Array),
}

impl NumericAccessor<'_> {
    fn value(&self, idx: usize) -> Result<f64, DataError> {
        match self {
            NumericAccessor::F64(arr) => Ok(arr.value(idx)),
            NumericAccessor::I64(arr) => {
                let value = arr.value(idx);
                if value < 0 {
                    return Err(DataError::CorruptData(format!(
                        "Negative volume at row {idx}: {value}"
                    )));
                }
                #[allow(clippy::cast_precision_loss)]
                let value_f64 = value as f64;
                Ok(value_f64)
            }
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

fn ensure_utc_timezone(column: &str, data_type: &DataType) -> Result<(), DataError> {
    match data_type {
        DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)) if tz.as_ref() == "UTC" => Ok(()),
        DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)) => Err(DataError::InvalidTimezone {
            column: column.to_string(),
            timezone: tz.to_string(),
        }),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => Err(DataError::InvalidTimezone {
            column: column.to_string(),
            timezone: "<none>".to_string(),
        }),
        _ => Err(DataError::InvalidColumnType(column.to_string())),
    }
}
