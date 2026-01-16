//! High-level backtest runner helpers.

use std::str::FromStr;

use omega_types::{BacktestConfig, DataMode, Timeframe};

use crate::engine::{BacktestEngine, DateBoundary, parse_datetime_ns};
use crate::error::BacktestError;

/// Main entry point: receives config JSON, returns result JSON.
///
/// # Errors
/// - [`BacktestError::ConfigParse`] when JSON parsing fails.
/// - [`BacktestError::ConfigValidation`] for invalid configuration values.
/// - Any errors from engine initialization or execution.
pub fn run_backtest_from_json(config_json: &str) -> Result<String, BacktestError> {
    let config: BacktestConfig = serde_json::from_str(config_json)
        .map_err(|e| BacktestError::ConfigParse(e.to_string()))?;

    validate_config(&config)?;

    let engine = BacktestEngine::new(config)?;
    let result = engine.run();

    serde_json::to_string(&result)
        .map_err(|e| BacktestError::ResultSerialize(e.to_string()))
}

fn validate_config(config: &BacktestConfig) -> Result<(), BacktestError> {
    if config.symbol.trim().is_empty() {
        return Err(BacktestError::ConfigValidation("symbol is empty".to_string()));
    }
    if config.strategy_name.trim().is_empty() {
        return Err(BacktestError::ConfigValidation(
            "strategy_name is empty".to_string(),
        ));
    }

    if config.data_mode != DataMode::Candle {
        return Err(BacktestError::ConfigValidation(
            "only candle data_mode is supported".to_string(),
        ));
    }

    if config.warmup_bars == 0 {
        return Err(BacktestError::ConfigValidation(
            "warmup_bars must be > 0".to_string(),
        ));
    }

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

    let start_ns = parse_datetime_ns(&config.start_date, DateBoundary::Start)
        .map_err(|e| BacktestError::ConfigValidation(e.to_string()))?;
    let end_ns = parse_datetime_ns(&config.end_date, DateBoundary::End)
        .map_err(|e| BacktestError::ConfigValidation(e.to_string()))?;
    if start_ns >= end_ns {
        return Err(BacktestError::ConfigValidation(
            "start_date must be before end_date".to_string(),
        ));
    }

    validate_timeframe(&config.timeframes.primary)?;
    for tf in &config.timeframes.additional {
        validate_timeframe(tf)?;
    }

    Ok(())
}

fn validate_timeframe(value: &str) -> Result<(), BacktestError> {
    Timeframe::from_str(value).map_err(|_| {
        BacktestError::ConfigValidation(format!("invalid timeframe '{value}'"))
    })?;
    Ok(())
}
