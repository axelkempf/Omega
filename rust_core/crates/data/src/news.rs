use std::collections::HashSet;
use std::path::Path;

use arrow::array::{Array, Int64Array, StringArray, TimestampNanosecondArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::DataError;

#[derive(Debug, Clone, PartialEq)]
pub struct NewsEvent {
    pub id: i64,
    pub timestamp_ns: i64,
    pub name: String,
    pub impact: String,
    pub currency: String,
}

/// Load news calendar events from a Parquet file.
/// Expected schema: `UTC time` (timestamp ns), `Id` (int), `Name` (string),
/// `Impact` (string), `Currency` (string).
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

        let ts_arr = batch
            .column_by_name("UTC time")
            .ok_or_else(|| DataError::MissingColumn("UTC time".to_string()))?
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
                    "Null timestamp/id at row {}",
                    row_idx
                )));
            }

            let ts = ts_arr.value(row_idx);
            if let Some(prev) = last_ts
                && ts < prev
            {
                return Err(DataError::CorruptData(format!(
                    "Out-of-order news timestamp at row {}: {} < {}",
                    row_idx, ts, prev
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
                    "Null news fields at row {}",
                    row_idx
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

fn normalize_impact(impact: &str) -> Result<String, DataError> {
    let up = impact.to_ascii_uppercase();
    match up.as_str() {
        "LOW" | "MEDIUM" | "HIGH" => Ok(up),
        _ => Err(DataError::CorruptData(format!(
            "Invalid impact value: {}",
            impact
        ))),
    }
}

fn normalize_currency(currency: &str) -> Result<String, DataError> {
    let up = currency.to_ascii_uppercase();
    if up.len() == 3 && up.chars().all(|c| c.is_ascii_alphabetic()) {
        Ok(up)
    } else {
        Err(DataError::CorruptData(format!(
            "Invalid currency value: {}",
            currency
        )))
    }
}
