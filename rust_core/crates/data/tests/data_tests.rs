use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, TimeUnit};
use temp_env::with_var;
use tempfile::tempdir;

use omega_data::{
    DataError, align_bid_ask, analyze_gaps, filter_by_date_range, load_and_validate,
    load_and_validate_bid_ask, load_candles, load_news_calendar, resolve_data_path,
    resolve_news_calendar_path, validate_spread,
};
use omega_types::{Candle, Timeframe};

mod common;
use common::{
    sample_candles, string_column, write_candle_parquet, write_candle_parquet_with_timezone,
    write_custom_parquet, write_news_parquet,
};
mod generators;
use proptest::prelude::*;

const TEST_STEP_NS: i64 = 60_000_000_000;

#[test]
fn test_load_candles_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("candles.parquet");
    let candles = sample_candles();
    write_candle_parquet(&path, &candles).unwrap();

    let loaded = load_candles(&path, Timeframe::M1).unwrap();
    assert_eq!(loaded, candles);
}

#[test]
fn test_load_and_validate_rejects_nan() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nan.parquet");
    let mut candles = sample_candles();
    candles[0].open = f64::NAN;
    write_candle_parquet(&path, &candles).unwrap();

    let err = load_and_validate(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_and_validate_rejects_non_monotonic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("non_monotonic.parquet");

    let candles = vec![
        Candle {
            timestamp_ns: 2,
            close_time_ns: 2 + TEST_STEP_NS - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.05,
            volume: 1.0,
        },
        Candle {
            timestamp_ns: 1,
            close_time_ns: 1 + TEST_STEP_NS - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.05,
            volume: 1.0,
        },
    ];

    write_candle_parquet(&path, &candles).unwrap();
    let err = load_and_validate(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_and_validate_rejects_invalid_ohlc() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("invalid_ohlc.parquet");

    let mut candles = sample_candles();
    candles[0].low = candles[0].high + 0.1;
    write_candle_parquet(&path, &candles).unwrap();

    let err = load_and_validate(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_and_validate_rejects_negative_volume() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("negative_volume.parquet");

    let mut candles = sample_candles();
    candles[1].volume = -1.0;
    write_candle_parquet(&path, &candles).unwrap();

    let err = load_and_validate(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_validate_candles_rejects_close_time_before_open() {
    let candles = vec![Candle {
        timestamp_ns: 10,
        close_time_ns: 9,
        open: 1.0,
        high: 1.1,
        low: 0.9,
        close: 1.05,
        volume: 1.0,
    }];

    let err = omega_data::validate_candles(&candles).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_validate_candles_rejects_non_monotonic_close_time() {
    let candles = vec![
        Candle {
            timestamp_ns: 1,
            close_time_ns: 10,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.05,
            volume: 1.0,
        },
        Candle {
            timestamp_ns: 2,
            close_time_ns: 9,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.05,
            volume: 1.0,
        },
    ];

    let err = omega_data::validate_candles(&candles).unwrap_err();
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
        close_time_ns: 1 + TEST_STEP_NS - 1,
        open: 1.0,
        high: 1.1,
        low: 0.9,
        close: 1.05,
        volume: 1.0,
    }];
    let ask = vec![Candle {
        timestamp_ns: 2,
        close_time_ns: 2 + TEST_STEP_NS - 1,
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
        close_time_ns: 1 + TEST_STEP_NS - 1,
        open: 1.1,
        high: 1.1,
        low: 1.1,
        close: 1.1,
        volume: 1.0,
    }];
    let ask = vec![Candle {
        timestamp_ns: 1,
        close_time_ns: 1 + TEST_STEP_NS - 1,
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
    with_var("OMEGA_DATA_PARQUET_ROOT", Some(dir.path()), || {
        let path = resolve_data_path("EURUSD", "M1", "BID");
        assert!(path.starts_with(dir.path()));
        assert!(path.ends_with(PathBuf::from("EURUSD/EURUSD_M1_BID.parquet")));
    });
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

    let err = load_candles(&path, Timeframe::M1).unwrap_err();
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

    let err = load_and_validate_bid_ask(&bid_path, &ask_path, Timeframe::M1).unwrap_err();
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
    let err = load_candles(&path, Timeframe::M1).unwrap_err();
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

    let err = load_candles(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::InvalidColumnType(_)));
}

#[test]
fn test_timezone_contract_rejects_missing_timezone() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("no_tz.parquet");
    let candles = sample_candles();

    write_candle_parquet_with_timezone(&path, &candles, None).unwrap();
    let err = load_candles(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::InvalidTimezone { .. }));
}

#[test]
fn test_timezone_contract_rejects_non_utc_timezone() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_tz.parquet");
    let candles = sample_candles();

    write_candle_parquet_with_timezone(&path, &candles, Some("Europe/Berlin")).unwrap();
    let err = load_candles(&path, Timeframe::M1).unwrap_err();
    assert!(matches!(err, DataError::InvalidTimezone { .. }));
}

#[test]
fn test_gap_analysis_session_aware() {
    let step_ns = 60_000_000_000i64;
    let base = 1_704_067_200_000_000_000i64; // 2024-01-01 00:00:00 UTC

    let candles = vec![
        Candle {
            timestamp_ns: base,
            close_time_ns: base + step_ns - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: 1.0,
        },
        Candle {
            timestamp_ns: base + step_ns,
            close_time_ns: base + step_ns + step_ns - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: 1.0,
        },
        Candle {
            timestamp_ns: base + step_ns * 3,
            close_time_ns: base + step_ns * 3 + step_ns - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: 1.0,
        },
    ];

    let stats = analyze_gaps(&candles, step_ns, None).unwrap();
    assert_eq!(stats.expected_bars, 4);
    assert_eq!(stats.missing_bars, 1);
    assert!((stats.gap_loss - 0.25).abs() < 1e-10);
}

#[test]
fn test_gap_analysis_respects_sessions() {
    let step_ns = 60_000_000_000i64;
    let base = 1_704_067_200_000_000_000i64; // 2024-01-01 00:00:00 UTC

    let candles = vec![
        Candle {
            timestamp_ns: base + step_ns * 479, // 07:59
            close_time_ns: base + step_ns * 479 + step_ns - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: 1.0,
        },
        Candle {
            timestamp_ns: base + step_ns * 480, // 08:00
            close_time_ns: base + step_ns * 480 + step_ns - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: 1.0,
        },
        Candle {
            timestamp_ns: base + step_ns * 482, // 08:02
            close_time_ns: base + step_ns * 482 + step_ns - 1,
            open: 1.0,
            high: 1.1,
            low: 0.9,
            close: 1.0,
            volume: 1.0,
        },
    ];

    let sessions = vec![omega_types::SessionConfig {
        start: "08:00".to_string(),
        end: "10:00".to_string(),
    }];

    let stats = analyze_gaps(&candles, step_ns, Some(&sessions)).unwrap();
    assert_eq!(stats.expected_bars, 3);
    assert_eq!(stats.missing_bars, 1);
}

#[test]
fn test_filter_by_date_range_inclusive() {
    let candles = sample_candles();
    let start = candles[0].timestamp_ns;
    let end = candles[0].timestamp_ns;

    let filtered = filter_by_date_range(&candles, start, end).unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].timestamp_ns, start);
}

#[test]
fn test_filter_by_date_range_empty_result() {
    let candles = sample_candles();
    let start = candles[1].timestamp_ns + 1;
    let end = candles[1].timestamp_ns + 2;

    let err = filter_by_date_range(&candles, start, end).unwrap_err();
    assert!(matches!(err, DataError::DateRangeEmpty { .. }));
}

#[test]
fn test_load_news_calendar_rejects_missing_column() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_missing.parquet");

    let timestamps: ArrayRef =
        Arc::new(TimestampNanosecondArray::from(vec![1_i64]).with_timezone("UTC"));
    let ids: ArrayRef = Arc::new(StringArray::from(vec!["1"]));
    let names: ArrayRef = Arc::new(StringArray::from(vec!["Event"]));
    let impacts: ArrayRef = Arc::new(StringArray::from(vec!["HIGH"]));
    // Currency column intentionally omitted

    let fields = vec![
        Field::new(
            "UTC time",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("Id", DataType::Utf8, false),
        Field::new("Name", DataType::Utf8, false),
        Field::new("Impact", DataType::Utf8, false),
    ];

    write_custom_parquet(&path, fields, vec![timestamps, ids, names, impacts]).unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::MissingColumn(_)));
}

#[test]
fn test_load_news_calendar_rejects_invalid_impact() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_invalid_impact.parquet");

    write_news_parquet(
        &path,
        &[1_i64],
        &["1"],
        &["Event"],
        &["SEVERE"],
        &["USD"],
        Some("UTC"),
    )
    .unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_news_calendar_rejects_invalid_currency() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_invalid_currency.parquet");

    write_news_parquet(
        &path,
        &[1_i64],
        &["1"],
        &["Event"],
        &["HIGH"],
        &["EURO"],
        Some("UTC"),
    )
    .unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_news_calendar_rejects_non_monotonic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_non_monotonic.parquet");

    write_news_parquet(
        &path,
        &[2_i64, 1_i64],
        &["1", "2"],
        &["Event A", "Event B"],
        &["HIGH", "LOW"],
        &["USD", "EUR"],
        Some("UTC"),
    )
    .unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::CorruptData(_)));
}

#[test]
fn test_load_news_calendar_rejects_empty() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_empty.parquet");

    write_news_parquet(&path, &[], &[], &[], &[], &[], Some("UTC")).unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::EmptyData));
}

#[test]
fn test_news_timezone_contract_rejects_missing_timezone() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_no_tz.parquet");

    write_news_parquet(
        &path,
        &[1_i64],
        &["1"],
        &["Event"],
        &["HIGH"],
        &["USD"],
        None,
    )
    .unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::InvalidTimezone { .. }));
}

#[test]
fn test_news_timezone_contract_rejects_non_utc_timezone() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("news_bad_tz.parquet");

    write_news_parquet(
        &path,
        &[1_i64],
        &["1"],
        &["Event"],
        &["HIGH"],
        &["USD"],
        Some("Europe/Berlin"),
    )
    .unwrap();

    let err = load_news_calendar(&path).unwrap_err();
    assert!(matches!(err, DataError::InvalidTimezone { .. }));
}

#[test]
fn test_resolve_news_calendar_path_env_override() {
    let dir = tempdir().unwrap();
    let custom_path = dir.path().join("calendar.parquet");

    with_var(
        "OMEGA_NEWS_CALENDAR_FILE",
        Some(custom_path.as_os_str()),
        || {
            let resolved = resolve_news_calendar_path();
            assert_eq!(resolved, custom_path);
        },
    );
}

proptest! {
    #[test]
    fn prop_valid_sequences_pass_validation(seq in generators::valid_candle_sequence(5)) {
        omega_data::validate_candles(&seq).unwrap();
    }
}
