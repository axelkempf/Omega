use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use omega_types::Candle;
use parquet::arrow::arrow_writer::ArrowWriter;

pub fn write_candle_parquet(
    path: &Path,
    candles: &[Candle],
) -> Result<(), Box<dyn std::error::Error>> {
    let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp_ns).collect();
    let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let fields = vec![
        Field::new(
            "UTC time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("Open", DataType::Float64, false),
        Field::new("High", DataType::Float64, false),
        Field::new("Low", DataType::Float64, false),
        Field::new("Close", DataType::Float64, false),
        Field::new("Volume", DataType::Float64, false),
    ];

    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampNanosecondArray::from(timestamps).with_timezone("UTC")),
        Arc::new(Float64Array::from(opens)),
        Arc::new(Float64Array::from(highs)),
        Arc::new(Float64Array::from(lows)),
        Arc::new(Float64Array::from(closes)),
        Arc::new(Float64Array::from(volumes)),
    ];

    write_custom_parquet(path, fields, columns)
}

pub fn write_custom_parquet(
    path: &Path,
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close().map(|_| ()).map_err(|e| e.into())
}

pub fn sample_candles() -> Vec<Candle> {
    vec![
        Candle {
            timestamp_ns: 1_700_000_000_000_000_000,
            open: 1.1,
            high: 1.2,
            low: 1.0,
            close: 1.15,
            volume: 100.0,
        },
        Candle {
            timestamp_ns: 1_700_000_060_000_000_000,
            open: 1.15,
            high: 1.25,
            low: 1.05,
            close: 1.2,
            volume: 120.0,
        },
    ]
}

pub fn string_column(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec())) as ArrayRef
}
