//! High-level backtest runner helpers.

use std::collections::HashSet;
use std::str::FromStr;

use omega_types::{BacktestConfig, DataMode, ExecutionVariant, RunMode, Timeframe};

use crate::engine::{BacktestEngine, DateBoundary, parse_datetime_ns};
use crate::error::BacktestError;

/// Main entry point: receives config JSON, returns result JSON.
///
/// # Errors
/// - [`BacktestError::ConfigParse`] when JSON parsing fails.
/// - [`BacktestError::ConfigValidation`] for invalid configuration values.
/// - Any errors from engine initialization or execution.
pub fn run_backtest_from_json(config_json: &str) -> Result<String, BacktestError> {
    let config: BacktestConfig =
        serde_json::from_str(config_json).map_err(|e| BacktestError::ConfigParse(e.to_string()))?;

    let config = normalize_config(config);
    validate_config(&config)?;

    let engine = BacktestEngine::new(config)?;
    let result = engine.run()?;

    serde_json::to_string(&result).map_err(|e| BacktestError::ResultSerialize(e.to_string()))
}

/// Normalizes config fields (trim, uppercase, deduplication).
fn normalize_config(mut config: BacktestConfig) -> BacktestConfig {
    // Normalize symbol (trim + uppercase)
    config.symbol = config.symbol.trim().to_uppercase();

    // Normalize strategy_name (trim only)
    config.strategy_name = config.strategy_name.trim().to_string();

    // Normalize primary timeframe (trim + uppercase)
    config.timeframes.primary = config.timeframes.primary.trim().to_uppercase();

    // Normalize and deduplicate additional timeframes
    let primary_upper = config.timeframes.primary.clone();
    let mut seen = HashSet::new();
    seen.insert(primary_upper.clone());

    config.timeframes.additional = config
        .timeframes
        .additional
        .into_iter()
        .map(|tf| tf.trim().to_uppercase())
        .filter(|tf| !tf.is_empty() && seen.insert(tf.clone()))
        .collect();

    // Normalize additional_source (trim + lowercase)
    config.timeframes.additional_source = config.timeframes.additional_source.trim().to_lowercase();

    // Normalize account_currency (trim + uppercase)
    config.account.account_currency = config.account.account_currency.trim().to_uppercase();

    config
}

fn validate_config(config: &BacktestConfig) -> Result<(), BacktestError> {
    // Symbol must not be empty
    if config.symbol.trim().is_empty() {
        return Err(BacktestError::ConfigValidation(
            "symbol is empty".to_string(),
        ));
    }
    // Strategy name must not be empty
    if config.strategy_name.trim().is_empty() {
        return Err(BacktestError::ConfigValidation(
            "strategy_name is empty".to_string(),
        ));
    }

    // Only candle data_mode is supported in MVP
    if config.data_mode != DataMode::Candle {
        return Err(BacktestError::ConfigValidation(
            "only candle data_mode is supported".to_string(),
        ));
    }

    // execution_variant = v1_parity is only allowed in dev mode (CONFIG_SCHEMA_PLAN 5.1)
    if config.execution_variant == ExecutionVariant::V1Parity && config.run_mode != RunMode::Dev {
        return Err(BacktestError::ConfigValidation(
            "execution_variant 'v1_parity' is only allowed when run_mode = 'dev'".to_string(),
        ));
    }

    // additional_source must be "separate_parquet" in MVP (CONFIG_SCHEMA_PLAN 4.1)
    if config.timeframes.additional_source != "separate_parquet" {
        return Err(BacktestError::ConfigValidation(format!(
            "timeframes.additional_source must be 'separate_parquet', got '{}'",
            config.timeframes.additional_source
        )));
    }

    // warmup_bars >= 0 (CONFIG_SCHEMA_PLAN 5.1 says >= 0, not > 0)
    // Note: warmup_bars is usize so cannot be negative, no validation needed for >= 0

    // Account validations
    if config.account.initial_balance <= 0.0 {
        return Err(BacktestError::ConfigValidation(
            "account.initial_balance must be > 0".to_string(),
        ));
    }
    if config.account.risk_per_trade <= 0.0 {
        return Err(BacktestError::ConfigValidation(
            "account.risk_per_trade must be > 0".to_string(),
        ));
    }

    // account_currency must be 3-letter uppercase (CONFIG_SCHEMA_PLAN 4.3)
    validate_account_currency(&config.account.account_currency)?;

    // Costs validations
    if config.costs.fee_multiplier < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.fee_multiplier must be >= 0".to_string(),
        ));
    }
    if config.costs.slippage_multiplier < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.slippage_multiplier must be >= 0".to_string(),
        ));
    }
    if config.costs.spread_multiplier < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.spread_multiplier must be >= 0".to_string(),
        ));
    }
    if let Some(pip_size) = config.costs.pip_size
        && pip_size <= 0.0
    {
        return Err(BacktestError::ConfigValidation(
            "costs.pip_size must be > 0".to_string(),
        ));
    }
    if config.costs.pip_buffer_factor < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.pip_buffer_factor must be >= 0".to_string(),
        ));
    }

    // Date range validation
    let start_ns = parse_datetime_ns(&config.start_date, DateBoundary::Start)
        .map_err(|e| BacktestError::ConfigValidation(e.to_string()))?;
    let end_ns = parse_datetime_ns(&config.end_date, DateBoundary::End)
        .map_err(|e| BacktestError::ConfigValidation(e.to_string()))?;
    if start_ns >= end_ns {
        return Err(BacktestError::ConfigValidation(
            "start_date must be before end_date".to_string(),
        ));
    }

    // Session validation (CONFIG_SCHEMA_PLAN 4.2)
    if let Some(sessions) = &config.sessions {
        for session in sessions {
            validate_session_time(&session.start)?;
            validate_session_time(&session.end)?;
        }
    }

    // News filter validation (CONFIG_SCHEMA_PLAN 4.5)
    // minutes_before and minutes_after are u32, so >= 0 is guaranteed
    // min_impact is validated by serde enum deserialization
    // No additional validation needed for news_filter

    // Timeframe validations
    validate_timeframe(&config.timeframes.primary)?;
    for tf in &config.timeframes.additional {
        validate_timeframe(tf)?;
    }

    Ok(())
}

/// Validates session time format (HH:MM) and range.
fn validate_session_time(time: &str) -> Result<(), BacktestError> {
    let parts: Vec<&str> = time.split(':').collect();
    if parts.len() != 2 {
        return Err(BacktestError::ConfigValidation(format!(
            "invalid session time format '{time}', expected HH:MM"
        )));
    }

    let hours: u32 = parts[0].parse().map_err(|_| {
        BacktestError::ConfigValidation(format!("invalid hours in session time '{time}'"))
    })?;
    let minutes: u32 = parts[1].parse().map_err(|_| {
        BacktestError::ConfigValidation(format!("invalid minutes in session time '{time}'"))
    })?;

    if hours > 23 {
        return Err(BacktestError::ConfigValidation(format!(
            "session time hours must be 0-23, got {hours} in '{time}'"
        )));
    }
    if minutes > 59 {
        return Err(BacktestError::ConfigValidation(format!(
            "session time minutes must be 0-59, got {minutes} in '{time}'"
        )));
    }

    Ok(())
}

/// Validates `account_currency` is 3-letter uppercase.
fn validate_account_currency(currency: &str) -> Result<(), BacktestError> {
    if currency.len() != 3 {
        return Err(BacktestError::ConfigValidation(format!(
            "account_currency must be 3 letters, got '{currency}'"
        )));
    }
    if !currency.chars().all(|c| c.is_ascii_uppercase()) {
        return Err(BacktestError::ConfigValidation(format!(
            "account_currency must be uppercase letters, got '{currency}'"
        )));
    }
    Ok(())
}

fn validate_timeframe(value: &str) -> Result<(), BacktestError> {
    Timeframe::from_str(value)
        .map_err(|_| BacktestError::ConfigValidation(format!("invalid timeframe '{value}'")))?;
    Ok(())
}
