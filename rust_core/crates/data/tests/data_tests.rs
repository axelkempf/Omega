use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, TimeUnit};
use tempfile::tempdir;

use omega_data::{
    DataError, align_bid_ask, load_and_validate, load_and_validate_bid_ask, load_candles,
    resolve_data_path, validate_spread,
};
use omega_types::Candle;

mod common;
use common::{sample_candles, string_column, write_candle_parquet, write_custom_parquet};
mod generators;
use proptest::prelude::*;

#[test]
fn test_load_candles_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("candles.parquet");
    let candles = sample_candles();
    write_candle_parquet(&path, &candles).unwrap();

    let loaded = load_candles(&path).unwrap();
    assert_eq!(loaded, candles);
}

#[test]
fn test_load_and_validate_rejects_nan() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nan.parquet");
    let mut candles = sample_candles();
    candles[0].open = f64::NAN;
    write_candle_parquet(&path, &candles).unwrap();

    let err = load_and_validate(&path).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_align_bid_ask_success() {
    let candles = sample_candles();
    let aligned = align_bid_ask(candles.clone(), candles.clone()).unwrap();
    assert_eq!(aligned.bid.len(), candles.len());
    assert_eq!(aligned.ask.len(), candles.len());
    assert!(aligned.alignment_stats.alignment_loss <= 0.01);
}

#[test]
fn test_align_bid_ask_rejects_high_loss() {
    let bid = vec![Candle {
        timestamp_ns: 1,
        open: 1.0,
        high: 1.1,
        low: 0.9,
        close: 1.05,
        volume: 1.0,
    }];
    let ask = vec![Candle {
        timestamp_ns: 2,
        open: 1.0,
        high: 1.1,
        low: 0.9,
        close: 1.05,
        volume: 1.0,
    }];

    let err = align_bid_ask(bid, ask).unwrap_err();
    assert!(matches!(err, DataError::AlignmentFailure(_)));
}

#[test]
fn test_validate_spread_rejects_bid_gt_ask() {
    let bid = vec![Candle {
        timestamp_ns: 1,
        open: 1.1,
        high: 1.1,
        low: 1.1,
        close: 1.1,
        volume: 1.0,
    }];
    let ask = vec![Candle {
        timestamp_ns: 1,
        open: 1.0,
        high: 1.0,
        low: 1.0,
        close: 1.0,
        volume: 1.0,
    }];

    let err = validate_spread(&bid, &ask).unwrap_err();
    assert!(matches!(err, DataError::InvalidSpread(_)));
}

#[test]
fn test_resolve_data_path_env_override() {
    let dir = tempdir().unwrap();
    unsafe {
        env::set_var("OMEGA_DATA_PARQUET_ROOT", dir.path());
    }

    let path = resolve_data_path("EURUSD", "M1", "BID");
    assert!(path.starts_with(dir.path()));
    assert!(path.ends_with(PathBuf::from("EURUSD/EURUSD_M1_BID.parquet")));

    unsafe {
        env::remove_var("OMEGA_DATA_PARQUET_ROOT");
    }
}

#[test]
fn test_load_candles_rejects_divergent_duplicate() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dup.parquet");

    // Two rows with same timestamp but different prices.
    let timestamps =
        Arc::new(TimestampNanosecondArray::from(vec![1_i64, 1_i64]).with_timezone("UTC"))
            as ArrayRef;
    let opens = Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef;
    let highs = Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef;
    let lows = Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef;
    let closes = Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef;
    let volumes = Arc::new(Float64Array::from(vec![1.0, 1.0])) as ArrayRef;

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

    write_custom_parquet(
        &path,
        fields,
        vec![timestamps, opens, highs, lows, closes, volumes],
    )
    .unwrap();

    let err = load_candles(&path).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_and_validate_bid_ask_checks_spread() {
    let dir = tempdir().unwrap();
    let bid_path = dir.path().join("bid.parquet");
    let ask_path = dir.path().join("ask.parquet");

    let bid_candles = sample_candles();
    // ask close/open lower to trigger spread validation
    let mut ask_candles = bid_candles.clone();
    for c in &mut ask_candles {
        c.open -= 0.05;
        c.close -= 0.05;
        // Keep OHLC consistent after adjustment
        c.low = c.low.min(c.open).min(c.close);
        c.high = c.high.max(c.open).max(c.close);
    }

    write_candle_parquet(&bid_path, &bid_candles).unwrap();
    write_candle_parquet(&ask_path, &ask_candles).unwrap();

    let err = load_and_validate_bid_ask(&bid_path, &ask_path).unwrap_err();
    assert!(matches!(err, DataError::InvalidSpread(_)));
}

#[test]
fn test_load_and_validate_detects_missing_column() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing.parquet");
    let candles = sample_candles();

    let timestamps: ArrayRef = Arc::new(
        TimestampNanosecondArray::from(candles.iter().map(|c| c.timestamp_ns).collect::<Vec<_>>())
            .with_timezone("UTC"),
    );
    let opens: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.open).collect::<Vec<_>>(),
    ));
    let highs: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.high).collect::<Vec<_>>(),
    ));
    let lows: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.low).collect::<Vec<_>>(),
    ));
    // Close column intentionally omitted
    let volumes: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.volume).collect::<Vec<_>>(),
    ));

    let fields = vec![
        Field::new(
            "UTC time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("Open", DataType::Float64, false),
        Field::new("High", DataType::Float64, false),
        Field::new("Low", DataType::Float64, false),
        Field::new("Volume", DataType::Float64, false),
    ];

    write_custom_parquet(&path, fields, vec![timestamps, opens, highs, lows, volumes]).unwrap();
    let err = load_candles(&path).unwrap_err();
    assert!(matches!(err, DataError::MissingColumn(_)));
}

#[test]
fn test_load_and_validate_rejects_wrong_column_type() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wrong_type.parquet");
    let candles = sample_candles();

    let timestamps: ArrayRef = Arc::new(
        TimestampNanosecondArray::from(candles.iter().map(|c| c.timestamp_ns).collect::<Vec<_>>())
            .with_timezone("UTC"),
    );
    let opens = string_column(&["1.0", "1.1"]); // wrong type
    let highs: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.high).collect::<Vec<_>>(),
    ));
    let lows: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.low).collect::<Vec<_>>(),
    ));
    let closes: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.close).collect::<Vec<_>>(),
    ));
    let volumes: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.volume).collect::<Vec<_>>(),
    ));

    let fields = vec![
        Field::new(
            "UTC time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("Open", DataType::Utf8, false),
        Field::new("High", DataType::Float64, false),
        Field::new("Low", DataType::Float64, false),
        Field::new("Close", DataType::Float64, false),
        Field::new("Volume", DataType::Float64, false),
    ];

    write_custom_parquet(
        &path,
        fields,
        vec![timestamps, opens, highs, lows, closes, volumes],
    )
    .unwrap();

    let err = load_candles(&path).unwrap_err();
    assert!(matches!(err, DataError::InvalidColumnType(_)));
}

proptest! {
    #[test]
    fn prop_valid_sequences_pass_validation(seq in generators::valid_candle_sequence(5)) {
        omega_data::validate_candles(&seq).unwrap();
    }
}
