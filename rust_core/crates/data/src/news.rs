//! News calendar loading and normalization.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use arrow::array::{Array, Int64Array, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::DataError;

/// Normalized news calendar event.
#[derive(Debug, Clone, PartialEq)]
pub struct NewsEvent {
    /// Unique event identifier.
    pub id: i64,
    /// Event timestamp in epoch-nanoseconds UTC.
    pub timestamp_ns: i64,
    /// Event name.
    pub name: String,
    /// Normalized impact (LOW|MEDIUM|HIGH).
    pub impact: String,
    /// Currency code (uppercase, 3-letter).
    pub currency: String,
}

/// Resolve the news calendar file path from env or default.
#[must_use]
pub fn resolve_news_calendar_path() -> PathBuf {
    let path = std::env::var("OMEGA_NEWS_CALENDAR_FILE")
        .unwrap_or_else(|_| "data/news/news_calender_history.parquet".to_string());
    PathBuf::from(path)
}

/// Load news calendar events from a Parquet file.
///
/// Expected schema: `UTC time` (timestamp ns, UTC), `Id` (int), `Name` (string),
/// `Impact` (string), `Currency` (string).
///
/// # Errors
/// - [`DataError::FileNotFound`] when the file cannot be opened.
/// - [`DataError::ParseError`] when Parquet decoding fails.
/// - [`DataError::MissingColumn`] or [`DataError::InvalidColumnType`] when schema is invalid.
/// - [`DataError::InvalidTimezone`] when `UTC time` is missing timezone or not UTC.
/// - [`DataError::CorruptData`] for out-of-order timestamps or invalid fields.
/// - [`DataError::EmptyData`] when no rows are loaded.
pub fn load_news_calendar(path: &Path) -> Result<Vec<NewsEvent>, DataError> {
    let file = std::fs::File::open(path)
        .map_err(|e| DataError::FileNotFound(path.display().to_string(), e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    let reader = builder
        .build()
        .map_err(|e| DataError::ParseError(e.to_string()))?;

    let mut events = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut last_ts: Option<i64> = None;

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
        let id_arr = batch
            .column_by_name("Id")
            .ok_or_else(|| DataError::MissingColumn("Id".to_string()))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataError::InvalidColumnType("Id".to_string()))?;
        let name_arr = batch
            .column_by_name("Name")
            .ok_or_else(|| DataError::MissingColumn("Name".to_string()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataError::InvalidColumnType("Name".to_string()))?;
        let impact_arr = batch
            .column_by_name("Impact")
            .ok_or_else(|| DataError::MissingColumn("Impact".to_string()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataError::InvalidColumnType("Impact".to_string()))?;
        let currency_arr = batch
            .column_by_name("Currency")
            .ok_or_else(|| DataError::MissingColumn("Currency".to_string()))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataError::InvalidColumnType("Currency".to_string()))?;

        for row_idx in 0..batch.num_rows() {
            if ts_arr.is_null(row_idx) || id_arr.is_null(row_idx) {
                return Err(DataError::CorruptData(format!(
                    "Null timestamp/id at row {row_idx}"
                )));
            }

            let ts = ts_arr.value(row_idx);
            if let Some(prev) = last_ts
                && ts < prev
            {
                return Err(DataError::CorruptData(format!(
                    "Out-of-order news timestamp at row {row_idx}: {ts} < {prev}"
                )));
            }

            let id = id_arr.value(row_idx);
            if seen_ids.contains(&id) {
                continue;
            }
            seen_ids.insert(id);

            if name_arr.is_null(row_idx)
                || impact_arr.is_null(row_idx)
                || currency_arr.is_null(row_idx)
            {
                return Err(DataError::CorruptData(format!(
                    "Null news fields at row {row_idx}"
                )));
            }

            let name = name_arr.value(row_idx).to_string();
            let impact = normalize_impact(impact_arr.value(row_idx))?;
            let currency = normalize_currency(currency_arr.value(row_idx))?;

            events.push(NewsEvent {
                id,
                timestamp_ns: ts,
                name,
                impact,
                currency,
            });
            last_ts = Some(ts);
        }
    }

    if events.is_empty() {
        return Err(DataError::EmptyData);
    }

    Ok(events)
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

fn normalize_impact(impact: &str) -> Result<String, DataError> {
    let up = impact.to_ascii_uppercase();
    match up.as_str() {
        "LOW" | "MEDIUM" | "HIGH" => Ok(up),
        _ => Err(DataError::CorruptData(format!(
            "Invalid impact value: {impact}"
        ))),
    }
}

fn normalize_currency(currency: &str) -> Result<String, DataError> {
    let up = currency.to_ascii_uppercase();
    if up.len() == 3 && up.chars().all(|c| c.is_ascii_alphabetic()) {
        Ok(up)
    } else {
        Err(DataError::CorruptData(format!(
            "Invalid currency value: {currency}"
        )))
    }
}
