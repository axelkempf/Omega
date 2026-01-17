//! Integration tests for the backtest engine.
//!
//! Tests cover:
//! - Config validation (execution_variant, additional_source, sessions, etc.)
//! - Warmup validation (insufficient data)
//! - HTF lookahead prevention
//! - Session/News gates
//! - Event loop ordering

use omega_backtest::{BacktestError, run_backtest_from_json};

// ============================================================================
// CONFIG VALIDATION TESTS
// ============================================================================

#[test]
fn test_config_validation_execution_variant_v1_parity_requires_dev_mode() {
    // v1_parity is only allowed in dev mode
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "prod",
        "data_mode": "candle",
        "execution_variant": "v1_parity",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10
    }"#;

    let result = run_backtest_from_json(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("v1_parity")),
        "Expected v1_parity error, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_execution_variant_v1_parity_allowed_in_dev_mode() {
    // v1_parity is allowed in dev mode (will fail later due to missing data)
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v1_parity",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10
    }"#;

    let result = run_backtest_from_json(config);
    // Should pass config validation but fail on data loading (no parquet files)
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Should NOT be a config validation error about v1_parity
    assert!(
        !matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("v1_parity")),
        "Should not fail on v1_parity in dev mode, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_additional_source_must_be_separate_parquet() {
    // Only "separate_parquet" is supported in MVP
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "aggregate_from_primary"},
        "warmup_bars": 10
    }"#;

    let result = run_backtest_from_json(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("additional_source")),
        "Expected additional_source error, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_warmup_bars_zero_is_allowed() {
    // Per CONFIG_SCHEMA_PLAN 5.1: warmup_bars >= 0 (not > 0)
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 0
    }"#;

    let result = run_backtest_from_json(config);
    // Should pass config validation (will fail on data loading)
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Should NOT be a config validation error about warmup_bars
    assert!(
        !matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("warmup_bars")),
        "warmup_bars=0 should be allowed, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_session_time_format_valid() {
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10,
        "sessions": [{"start": "08:00", "end": "17:00"}]
    }"#;

    let result = run_backtest_from_json(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Should NOT be a session time format error
    assert!(
        !matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("session time")),
        "Valid session times should be accepted, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_session_time_format_invalid_hours() {
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10,
        "sessions": [{"start": "25:00", "end": "17:00"}]
    }"#;

    let result = run_backtest_from_json(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("session time") || msg.contains("hours")),
        "Expected session time hours error, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_session_time_format_invalid_minutes() {
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10,
        "sessions": [{"start": "08:75", "end": "17:00"}]
    }"#;

    let result = run_backtest_from_json(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("session time") || msg.contains("minutes")),
        "Expected session time minutes error, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_account_currency_must_be_3_letters() {
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10,
        "account": {"initial_balance": 10000.0, "account_currency": "EURO", "risk_per_trade": 100.0}
    }"#;

    let result = run_backtest_from_json(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("account_currency")),
        "Expected account_currency error, got: {:?}",
        err
    );
}

#[test]
fn test_config_validation_account_currency_must_be_uppercase() {
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10,
        "account": {"initial_balance": 10000.0, "account_currency": "eur", "risk_per_trade": 100.0}
    }"#;

    let result = run_backtest_from_json(config);
    // Should normalize to uppercase and pass (or fail with data error, not currency error)
    assert!(result.is_err());
    let err = result.unwrap_err();
    // After normalization, "eur" becomes "EUR", so should NOT fail on currency validation
    assert!(
        !matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("account_currency") && msg.contains("uppercase")),
        "Lowercase currency should be normalized, got: {:?}",
        err
    );
}

#[test]
fn test_config_normalization_symbol_trimmed_and_uppercased() {
    // Symbol with whitespace and lowercase should be normalized
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "  eurusd  ",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional_source": "separate_parquet"},
        "warmup_bars": 10
    }"#;

    let result = run_backtest_from_json(config);
    // Should pass config validation (will fail on data loading)
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Should NOT fail because symbol is empty after normalization
    assert!(
        !matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("symbol is empty")),
        "Symbol should be normalized, got: {:?}",
        err
    );
}

#[test]
fn test_config_normalization_timeframes_deduplicated() {
    // Additional timeframes should be deduplicated and primary should be removed
    let config = r#"{
        "schema_version": "2",
        "strategy_name": "mean_reversion_z_score",
        "symbol": "EURUSD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "run_mode": "dev",
        "data_mode": "candle",
        "execution_variant": "v2",
        "timeframes": {"primary": "M5", "additional": ["H1", "h1", "M5", "D1", "d1"], "additional_source": "separate_parquet"},
        "warmup_bars": 10
    }"#;

    let result = run_backtest_from_json(config);
    // Should pass config validation (will fail on data loading)
    assert!(result.is_err());
    // Just verify it doesn't fail on timeframe validation
    let err = result.unwrap_err();
    assert!(
        !matches!(err, BacktestError::ConfigValidation(ref msg) if msg.contains("timeframe")),
        "Timeframes should be normalized and deduplicated, got: {:?}",
        err
    );
}

// ============================================================================
// DATA FLOW / HTF LOOKAHEAD TESTS (Unit-level)
// ============================================================================

mod htf_lookahead {
    use omega_data::{CandleStore, MultiTfStore};
    use omega_types::{Candle, Timeframe};

    /// Creates a test candle at given timestamp.
    fn make_candle(timestamp_ns: i64) -> Candle {
        Candle {
            timestamp_ns,
            close_time_ns: timestamp_ns + 299_999_999_999, // ~5min bar
            open: 1.1000,
            high: 1.1010,
            low: 1.0990,
            close: 1.1005,
            volume: 100.0,
        }
    }

    /// Creates a candle store with given timestamps.
    fn make_store(timestamps: Vec<i64>, tf: Timeframe) -> CandleStore {
        let bid: Vec<Candle> = timestamps.iter().map(|&ts| make_candle(ts)).collect();
        let ask = bid.clone();
        CandleStore {
            bid,
            ask,
            timestamps,
            timeframe: tf,
            symbol: "EURUSD".to_string(),
            warmup_bars: 0,
        }
    }

    #[test]
    fn test_htf_index_mapping_returns_none_when_no_htf() {
        let primary = make_store(
            vec![
                1_704_067_200_000_000_000, // 2024-01-01 00:00:00
                1_704_067_500_000_000_000, // 2024-01-01 00:05:00
            ],
            Timeframe::M5,
        );

        let store = MultiTfStore::new(primary, None, vec![]);

        // No HTF store, should return None
        assert!(store.htf_index_at(0).is_none());
        assert!(store.htf_index_at(1).is_none());
    }

    #[test]
    fn test_htf_index_mapping_basic() {
        // Primary: M5 bars
        let primary = make_store(
            vec![
                1_704_067_200_000_000_000, // 2024-01-01 00:00:00
                1_704_067_500_000_000_000, // 2024-01-01 00:05:00
                1_704_067_800_000_000_000, // 2024-01-01 00:10:00
                1_704_068_100_000_000_000, // 2024-01-01 00:15:00
                1_704_070_800_000_000_000, // 2024-01-01 01:00:00 (new H1 bar)
                1_704_071_100_000_000_000, // 2024-01-01 01:05:00
            ],
            Timeframe::M5,
        );

        // HTF: H1 bars
        let htf = make_store(
            vec![
                1_704_067_200_000_000_000, // 2024-01-01 00:00:00
                1_704_070_800_000_000_000, // 2024-01-01 01:00:00
            ],
            Timeframe::H1,
        );

        let store = MultiTfStore::new(primary, Some(htf), vec![]);

        // At primary idx 0 (00:00:00), HTF idx 0 is "current" (not completed)
        // The index_map should return Some(0), but per lookahead rules,
        // the engine should use completed_idx = 0 - 1 = None
        assert_eq!(store.htf_index_at(0), Some(0));

        // At primary idx 4 (01:00:00), HTF idx 1 is "current"
        // HTF idx 0 is completed
        assert_eq!(store.htf_index_at(4), Some(1));
    }

    #[test]
    fn test_htf_completed_bar_logic() {
        // This tests the logic that should be applied in build_htf_context:
        // completed_idx = htf_idx.checked_sub(1)

        let htf_idx_at_first_bar: usize = 0;
        let htf_idx_at_later_bar: usize = 1;

        // At first primary bar, htf_idx = 0, completed = 0 - 1 = None
        let completed_first = htf_idx_at_first_bar.checked_sub(1);
        assert!(completed_first.is_none());

        // At later primary bar, htf_idx = 1, completed = 1 - 1 = 0
        let completed_later = htf_idx_at_later_bar.checked_sub(1);
        assert_eq!(completed_later, Some(0));
    }
}

// ============================================================================
// ENV OVERRIDE TESTS
// ============================================================================

mod env_overrides {
    use std::env;
    use std::path::PathBuf;

    // Note: These tests verify the env var reading mechanism works.
    // The actual path resolution functions are internal to the engine.
    // Full integration testing would require actual config files.

    struct EnvVarGuard {
        key: String,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &str) -> Self {
            Self {
                key: key.to_string(),
                original: env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => env::set_var(&self.key, value),
                    None => env::remove_var(&self.key),
                }
            }
        }
    }

    fn with_env_var<T>(key: &str, value: &str, f: impl FnOnce() -> T) -> T {
        let _guard = EnvVarGuard::new(key);
        unsafe {
            env::set_var(key, value);
        }
        f()
    }

    #[test]
    fn test_env_var_round_trip_execution_costs() {
        let test_path = PathBuf::from("/tmp/omega_execution_costs.yaml");
        let test_path_str = test_path.to_string_lossy().to_string();

        with_env_var("OMEGA_EXECUTION_COSTS_FILE", &test_path_str, || {
            let value = env::var("OMEGA_EXECUTION_COSTS_FILE")
                .expect("OMEGA_EXECUTION_COSTS_FILE should be set");
            assert_eq!(value, test_path_str);
        });
    }

    #[test]
    fn test_env_var_round_trip_symbol_specs() {
        let test_path = PathBuf::from("/tmp/omega_symbol_specs.yaml");
        let test_path_str = test_path.to_string_lossy().to_string();

        with_env_var("OMEGA_SYMBOL_SPECS_FILE", &test_path_str, || {
            let value =
                env::var("OMEGA_SYMBOL_SPECS_FILE").expect("OMEGA_SYMBOL_SPECS_FILE should be set");
            assert_eq!(value, test_path_str);
        });
    }
}

// ============================================================================
// WARMUP VALIDATION TESTS
// ============================================================================

mod warmup_validation {
    use omega_backtest::warmup::{validate_htf_warmup, validate_warmup};
    use omega_data::CandleStore;
    use omega_types::{Candle, Timeframe};

    fn make_candle(timestamp_ns: i64) -> Candle {
        Candle {
            timestamp_ns,
            close_time_ns: timestamp_ns + 299_999_999_999,
            open: 1.1000,
            high: 1.1010,
            low: 1.0990,
            close: 1.1005,
            volume: 100.0,
        }
    }

    fn make_store(count: usize) -> CandleStore {
        let timestamps: Vec<i64> = (0..count)
            .map(|i| 1_704_067_200_000_000_000 + (i as i64) * 300_000_000_000)
            .collect();
        let bid: Vec<Candle> = timestamps.iter().map(|&ts| make_candle(ts)).collect();
        let ask = bid.clone();
        CandleStore {
            bid,
            ask,
            timestamps,
            timeframe: Timeframe::M5,
            symbol: "EURUSD".to_string(),
            warmup_bars: 0,
        }
    }

    #[test]
    fn test_warmup_validation_passes_with_sufficient_data() {
        let store = make_store(100);
        let result = validate_warmup(&store, 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_warmup_validation_fails_with_insufficient_data() {
        let store = make_store(50);
        let result = validate_warmup(&store, 50);
        assert!(result.is_err());
    }

    #[test]
    fn test_warmup_validation_fails_when_data_equals_warmup() {
        // Need more than warmup_bars, not equal
        let store = make_store(10);
        let result = validate_warmup(&store, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_warmup_validation_passes_with_zero_warmup() {
        let store = make_store(1);
        let result = validate_warmup(&store, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_htf_warmup_validation_passes_when_no_htf() {
        let result = validate_htf_warmup(None, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_htf_warmup_validation_passes_with_sufficient_htf_data() {
        let store = make_store(100);
        let result = validate_htf_warmup(Some(&store), 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_htf_warmup_validation_fails_with_insufficient_htf_data() {
        let store = make_store(30);
        let result = validate_htf_warmup(Some(&store), 50);
        assert!(result.is_err());
    }
}

// ============================================================================
// W5 ORCHESTRATION TESTS
// ============================================================================

mod wave5_orchestration {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float64Array, StringArray, TimestampNanosecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use omega_types::{BacktestResult, Candle, ExitReason};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use serde_json::json;
    use temp_env::with_var;
    use tempfile::tempdir;

    use super::run_backtest_from_json;

    const STEP_NS: i64 = 60_000_000_000;
    const BASE_TS: i64 = 1_704_067_200_000_000_000;
    const SYMBOL: &str = "EURUSD";
    const TIMEFRAME: &str = "M1";
    const SPREAD: f64 = 0.0002;

    #[derive(Debug, Clone)]
    struct NewsFixture {
        timestamps: Vec<i64>,
        ids: Vec<String>,
        names: Vec<&'static str>,
        impacts: Vec<&'static str>,
        currencies: Vec<&'static str>,
    }

    fn make_candles(closes: &[f64], high_delta: f64, low_delta: f64) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(idx, close)| {
                let ts = BASE_TS + (idx as i64) * STEP_NS;
                Candle {
                    timestamp_ns: ts,
                    close_time_ns: ts + STEP_NS - 1,
                    open: *close,
                    high: *close + high_delta,
                    low: *close - low_delta,
                    close: *close,
                    volume: 100.0,
                }
            })
            .collect()
    }

    fn set_range(candle: &mut Candle, high_delta: f64, low_delta: f64) {
        candle.high = candle.close + high_delta;
        candle.low = candle.close - low_delta;
    }

    fn make_ask(bid: &[Candle], spread: f64) -> Vec<Candle> {
        bid.iter()
            .map(|c| Candle {
                timestamp_ns: c.timestamp_ns,
                close_time_ns: c.close_time_ns,
                open: c.open + spread,
                high: c.high + spread,
                low: c.low + spread,
                close: c.close + spread,
                volume: c.volume,
            })
            .collect()
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

    fn write_candle_parquet(
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

    fn write_news_parquet(
        path: &Path,
        fixture: &NewsFixture,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fields = vec![
            Field::new(
                "UTC time",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                false,
            ),
            Field::new("Id", DataType::Utf8, false),
            Field::new("Name", DataType::Utf8, false),
            Field::new("Impact", DataType::Utf8, false),
            Field::new("Currency", DataType::Utf8, false),
        ];

        let columns: Vec<ArrayRef> = vec![
            Arc::new(
                TimestampNanosecondArray::from(fixture.timestamps.clone()).with_timezone("UTC"),
            ),
            Arc::new(StringArray::from(fixture.ids.clone())),
            Arc::new(StringArray::from(fixture.names.to_vec())),
            Arc::new(StringArray::from(fixture.impacts.to_vec())),
            Arc::new(StringArray::from(fixture.currencies.to_vec())),
        ];

        write_custom_parquet(path, fields, columns)
    }

    fn write_bid_ask_fixture(
        root: &Path,
        bid: &[Candle],
        ask: &[Candle],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let symbol_dir = root.join(SYMBOL);
        std::fs::create_dir_all(&symbol_dir)?;

        let bid_path = symbol_dir.join(format!("{SYMBOL}_{TIMEFRAME}_BID.parquet"));
        let ask_path = symbol_dir.join(format!("{SYMBOL}_{TIMEFRAME}_ASK.parquet"));
        write_candle_parquet(&bid_path, bid)?;
        write_candle_parquet(&ask_path, ask)?;
        Ok(())
    }

    fn build_config(
        warmup_bars: usize,
        rng_seed: u64,
        sessions: Option<serde_json::Value>,
        news_filter: Option<serde_json::Value>,
        trade_management: Option<serde_json::Value>,
    ) -> serde_json::Value {
        let mut config = json!({
            "schema_version": "2",
            "strategy_name": "mean_reversion_z_score",
            "symbol": SYMBOL,
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T00:04:00Z",
            "run_mode": "dev",
            "data_mode": "candle",
            "execution_variant": "v2",
            "timeframes": {
                "primary": TIMEFRAME,
                "additional": [],
                "additional_source": "separate_parquet"
            },
            "warmup_bars": warmup_bars,
            "rng_seed": rng_seed,
            "account": {
                "initial_balance": 10000.0,
                "account_currency": "EUR",
                "risk_per_trade": 10000.0,
                "max_positions": 1
            },
            "costs": {
                "enabled": false
            },
            "strategy_parameters": {
                "ema_length": 2,
                "atr_length": 1,
                "atr_mult": 1.0,
                "window_length": 2,
                "z_score_long": -0.5,
                "z_score_short": 0.5,
                "htf_filter": "none",
                "extra_htf_filter": "none",
                "enabled_scenarios": [1]
            }
        });

        if let Some(obj) = config.as_object_mut() {
            if let Some(sessions) = sessions {
                obj.insert("sessions".to_string(), sessions);
            }
            if let Some(news_filter) = news_filter {
                obj.insert("news_filter".to_string(), news_filter);
            }
            if let Some(trade_management) = trade_management {
                obj.insert("trade_management".to_string(), trade_management);
            }
        }

        config
    }

    fn run_backtest_fixture(
        config: serde_json::Value,
        bid: &[Candle],
        ask: &[Candle],
        news: Option<&NewsFixture>,
    ) -> BacktestResult {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();

        write_bid_ask_fixture(root, bid, ask).expect("fixture write");
        let config_json = serde_json::to_string(&config).expect("config json");

        let result_json = with_var("OMEGA_DATA_PARQUET_ROOT", Some(root), || {
            if let Some(news_fixture) = news {
                let news_path = root.join("news_calendar.parquet");
                write_news_parquet(&news_path, news_fixture).expect("news parquet");
                with_var("OMEGA_NEWS_CALENDAR_FILE", Some(&news_path), || {
                    run_backtest_from_json(&config_json).expect("backtest run")
                })
            } else {
                run_backtest_from_json(&config_json).expect("backtest run")
            }
        });

        serde_json::from_str(&result_json).expect("result json")
    }

    #[test]
    fn test_event_loop_market_order_tp_not_in_entry_bar() {
        let mut bid = make_candles(&[1.0, 1.1, 0.9, 0.9, 0.9], 0.01, 0.01);
        set_range(&mut bid[2], 0.12, 0.02);
        set_range(&mut bid[3], 0.12, 0.02);
        let ask = make_ask(&bid, SPREAD);

        let config = build_config(2, 42, None, None, None);
        let result = run_backtest_fixture(config, &bid, &ask, None);
        assert!(result.ok);

        let trades = result.trades.unwrap_or_default();
        assert_eq!(trades.len(), 1);

        let trade = &trades[0];
        assert_eq!(trade.entry_time_ns, bid[2].timestamp_ns);
        assert_eq!(trade.exit_time_ns, bid[3].timestamp_ns);
        assert!(bid[2].high >= trade.take_profit);
        assert!(trade.exit_time_ns > trade.entry_time_ns);
        let in_entry = trade
            .meta
            .get("in_entry_candle")
            .and_then(serde_json::Value::as_bool);
        assert_eq!(in_entry, Some(false));
    }

    #[test]
    fn test_trade_management_runs_before_signal() {
        let mut bid = make_candles(&[1.0, 1.1, 0.9, 0.9, 0.9], 0.01, 0.01);
        set_range(&mut bid[2], 0.12, 0.02);
        set_range(&mut bid[3], 0.03, 0.02);
        set_range(&mut bid[4], 0.12, 0.02);
        let ask = make_ask(&bid, SPREAD);

        let trade_management = json!({
            "enabled": true,
            "rules": {
                "max_holding_time": {
                    "enabled": true,
                    "max_holding_minutes": 1
                }
            }
        });

        let config = build_config(2, 7, None, None, Some(trade_management));
        let result = run_backtest_fixture(config, &bid, &ask, None);
        assert!(result.ok);

        let trades = result.trades.unwrap_or_default();
        assert_eq!(trades.len(), 2);
        assert_eq!(trades[0].reason, ExitReason::Timeout);
        assert_eq!(trades[0].exit_time_ns, bid[3].timestamp_ns);
        assert_eq!(trades[1].entry_time_ns, bid[3].timestamp_ns);
        assert_eq!(trades[1].reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_session_gate_blocks_entries() {
        let mut bid = make_candles(&[1.0, 1.1, 0.9, 0.9, 0.9], 0.01, 0.01);
        set_range(&mut bid[4], 0.06, 0.01);
        let ask = make_ask(&bid, SPREAD);

        let sessions = json!([
            {"start": "00:00", "end": "00:02"}
        ]);
        let config = build_config(2, 42, Some(sessions), None, None);
        let result = run_backtest_fixture(config, &bid, &ask, None);
        assert!(result.ok);
        let trades = result.trades.unwrap_or_default();
        assert!(trades.is_empty());

        let config = build_config(2, 42, None, None, None);
        let result = run_backtest_fixture(config, &bid, &ask, None);
        assert!(result.ok);
        let trades = result.trades.unwrap_or_default();
        assert!(!trades.is_empty());
    }

    #[test]
    fn test_news_gate_blocks_entries() {
        let mut bid = make_candles(&[1.0, 1.1, 0.9, 0.9, 0.9], 0.01, 0.01);
        set_range(&mut bid[4], 0.06, 0.01);
        let ask = make_ask(&bid, SPREAD);

        let news = NewsFixture {
            timestamps: vec![bid[2].timestamp_ns],
            ids: vec!["1".to_string()],
            names: vec!["NFP"],
            impacts: vec!["HIGH"],
            currencies: vec!["EUR"],
        };

        let news_filter = json!({
            "enabled": true,
            "minutes_before": 1,
            "minutes_after": 1,
            "min_impact": "medium"
        });

        let config = build_config(2, 42, None, Some(news_filter), None);
        let result = run_backtest_fixture(config, &bid, &ask, Some(&news));
        assert!(result.ok);
        let trades = result.trades.unwrap_or_default();
        assert!(trades.is_empty());
    }

    #[test]
    fn test_determinism_same_seed_trades_and_equity_match() {
        let mut bid = make_candles(&[1.0, 1.1, 0.9, 0.9, 0.9], 0.01, 0.01);
        set_range(&mut bid[4], 0.06, 0.01);
        let ask = make_ask(&bid, SPREAD);

        let config = build_config(2, 123, None, None, None);
        let result_a = run_backtest_fixture(config.clone(), &bid, &ask, None);
        let result_b = run_backtest_fixture(config, &bid, &ask, None);
        assert!(result_a.ok && result_b.ok);

        let trades_a = serde_json::to_string(&result_a.trades.unwrap_or_default()).unwrap();
        let trades_b = serde_json::to_string(&result_b.trades.unwrap_or_default()).unwrap();
        assert_eq!(trades_a, trades_b);

        let equity_a = serde_json::to_string(&result_a.equity_curve.unwrap_or_default()).unwrap();
        let equity_b = serde_json::to_string(&result_b.equity_curve.unwrap_or_default()).unwrap();
        assert_eq!(equity_a, equity_b);
    }

    #[test]
    fn test_mini_e2e_fixture_equity_per_bar() {
        let mut bid = make_candles(&[1.0, 1.1, 0.9, 0.9, 0.9], 0.01, 0.01);
        set_range(&mut bid[4], 0.06, 0.01);
        let ask = make_ask(&bid, SPREAD);

        let config = build_config(2, 42, None, None, None);
        let result = run_backtest_fixture(config, &bid, &ask, None);
        assert!(result.ok);

        let trades = result.trades.unwrap_or_default();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].entry_time_ns, bid[2].timestamp_ns);
        assert_eq!(trades[0].exit_time_ns, bid[4].timestamp_ns);

        let equity = result.equity_curve.unwrap_or_default();
        assert_eq!(equity.len(), bid.len().saturating_sub(2));
        assert_eq!(equity[0].timestamp_ns, bid[2].timestamp_ns);
    }
}
