use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use omega_types::Candle;
use parquet::arrow::arrow_writer::ArrowWriter;

const SYMBOL: &str = "EURUSD";
const BASE_DATE: &str = "2025-01-01";
const BASE_TS_NS: i64 = 1_735_689_600_000_000_000; // 2025-01-01T00:00:00Z
const STEP_NS: i64 = 60_000_000_000; // 1 minute
const SPREAD: f64 = 0.0001;
const M1_BARS: usize = 120;
const M5_GROUP: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = repo_root();
    let fixtures_root = std::env::var("OMEGA_FIXTURES_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("python/tests/fixtures/data"));

    let parquet_root = fixtures_root.join("parquet").join(SYMBOL);
    let csv_root = fixtures_root.join("csv").join(SYMBOL);
    fs::create_dir_all(&parquet_root)?;
    fs::create_dir_all(&csv_root)?;

    let m1_bid = generate_bid_series(M1_BARS, STEP_NS);
    let m1_ask = apply_spread(&m1_bid, SPREAD);
    let m5_bid = aggregate_timeframe(&m1_bid, M5_GROUP, STEP_NS);
    let m5_ask = aggregate_timeframe(&m1_ask, M5_GROUP, STEP_NS);

    write_candle_parquet(
        &parquet_root.join(format!("{SYMBOL}_M1_BID.parquet")),
        &m1_bid,
    )?;
    write_candle_parquet(
        &parquet_root.join(format!("{SYMBOL}_M1_ASK.parquet")),
        &m1_ask,
    )?;
    write_candle_parquet(
        &parquet_root.join(format!("{SYMBOL}_M5_BID.parquet")),
        &m5_bid,
    )?;
    write_candle_parquet(
        &parquet_root.join(format!("{SYMBOL}_M5_ASK.parquet")),
        &m5_ask,
    )?;

    write_candle_csv(&csv_root.join(format!("{SYMBOL}_M1_BID.csv")), &m1_bid)?;
    write_candle_csv(&csv_root.join(format!("{SYMBOL}_M1_ASK.csv")), &m1_ask)?;
    write_candle_csv(&csv_root.join(format!("{SYMBOL}_M5_BID.csv")), &m5_bid)?;
    write_candle_csv(&csv_root.join(format!("{SYMBOL}_M5_ASK.csv")), &m5_ask)?;

    Ok(())
}

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../../..")
}

fn generate_bid_series(count: usize, step_ns: i64) -> Vec<Candle> {
    let mut candles = Vec::with_capacity(count);
    let mut prev_close = round_price(1.1000);
    for i in 0..count {
        let timestamp_ns = BASE_TS_NS + i as i64 * step_ns;
        let wave = ((i as f64) / 6.0).sin() * 0.0006;
        let drift = (i as f64) * 0.000002;
        let close = round_price(1.1000 + wave + drift);
        let open = if i == 0 { close } else { prev_close };
        let high = round_price(open.max(close) + 0.0002);
        let low = round_price(open.min(close) - 0.0002);
        let volume = 100.0 + (i % 10) as f64;
        let close_time_ns = timestamp_ns + step_ns - 1;

        candles.push(Candle {
            timestamp_ns,
            close_time_ns,
            open,
            high,
            low,
            close,
            volume,
        });
        prev_close = close;
    }
    candles
}

fn apply_spread(bid: &[Candle], spread: f64) -> Vec<Candle> {
    bid.iter()
        .map(|c| Candle {
            timestamp_ns: c.timestamp_ns,
            close_time_ns: c.close_time_ns,
            open: round_price(c.open + spread),
            high: round_price(c.high + spread),
            low: round_price(c.low + spread),
            close: round_price(c.close + spread),
            volume: c.volume,
        })
        .collect()
}

fn aggregate_timeframe(candles: &[Candle], group: usize, step_ns: i64) -> Vec<Candle> {
    let mut out = Vec::new();
    if group == 0 {
        return out;
    }
    for chunk in candles.chunks(group) {
        if chunk.len() < group {
            break;
        }
        let open = chunk.first().map(|c| c.open).unwrap_or(0.0);
        let close = chunk.last().map(|c| c.close).unwrap_or(0.0);
        let mut high = f64::MIN;
        let mut low = f64::MAX;
        let mut volume = 0.0;
        for candle in chunk {
            high = high.max(candle.high);
            low = low.min(candle.low);
            volume += candle.volume;
        }
        let timestamp_ns = chunk.first().map(|c| c.timestamp_ns).unwrap_or(0);
        let close_time_ns = timestamp_ns + (group as i64 * step_ns) - 1;
        out.push(Candle {
            timestamp_ns,
            close_time_ns,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    out
}

fn write_candle_parquet(path: &Path, candles: &[Candle]) -> Result<(), Box<dyn std::error::Error>> {
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

fn write_custom_parquet(
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

fn write_candle_csv(path: &Path, candles: &[Candle]) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "UTC time,Open,High,Low,Close,Volume")?;
    for candle in candles {
        let offset_minutes = (candle.timestamp_ns - BASE_TS_NS) / STEP_NS;
        let hour = (offset_minutes / 60) % 24;
        let minute = offset_minutes % 60;
        let timestamp = format!("{BASE_DATE} {hour:02}:{minute:02}:00+00:00");
        writeln!(
            writer,
            "{},{:.5},{:.5},{:.5},{:.5},{:.1}",
            timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume
        )?;
    }
    Ok(())
}

fn round_price(value: f64) -> f64 {
    (value * 100_000.0).round() / 100_000.0
}
