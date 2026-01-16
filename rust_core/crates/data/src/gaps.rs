//! Gap analysis for candle data.

use std::collections::HashSet;

use omega_types::{Candle, SessionConfig};

use crate::error::DataError;

/// Statistics describing missing bars within active sessions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapStats {
    /// Number of expected bars within active sessions.
    pub expected_bars: usize,
    /// Number of missing bars within active sessions.
    pub missing_bars: usize,
    /// Missing bars divided by expected bars (0.0 when `expected_bars` == 0).
    pub gap_loss: f64,
}

/// Analyze gaps within active sessions for a candle sequence.
///
/// The analysis is session-aware, deterministic, and does not interpolate.
/// It reports missing bars based on the expected timeframe step.
///
/// If sessions are `None` or empty, a 24/5 default is applied
/// (Mon–Fri full-day sessions, weekend inactive).
///
/// # Errors
/// - [`DataError::EmptyData`] when `candles` is empty.
/// - [`DataError::CorruptData`] when `step_ns` is non-positive or overflow occurs.
/// - [`DataError::CorruptData`] when timestamp conversion fails.
pub fn analyze_gaps(
    candles: &[Candle],
    step_ns: i64,
    sessions: Option<&[SessionConfig]>,
) -> Result<GapStats, DataError> {
    if candles.is_empty() {
        return Err(DataError::EmptyData);
    }

    if step_ns <= 0 {
        return Err(DataError::CorruptData(format!(
            "Invalid timeframe step (ns): {step_ns}"
        )));
    }

    let start_ts = candles
        .first()
        .map(|c| c.timestamp_ns)
        .ok_or(DataError::EmptyData)?;
    let end_ts = candles
        .last()
        .map(|c| c.timestamp_ns)
        .ok_or(DataError::EmptyData)?;

    let actual: HashSet<i64> = candles.iter().map(|c| c.timestamp_ns).collect();

    let mut expected_bars = 0usize;
    let mut missing_bars = 0usize;
    let mut ts = start_ts;

    while ts <= end_ts {
        if is_session_active(ts, sessions)? {
            expected_bars += 1;
            if !actual.contains(&ts) {
                missing_bars += 1;
            }
        }

        ts = ts.checked_add(step_ns).ok_or_else(|| {
            DataError::CorruptData("Timestamp overflow while scanning gaps".to_string())
        })?;
    }

    let gap_loss = if expected_bars == 0 {
        0.0
    } else {
        let missing_f = count_to_f64(missing_bars, "missing_bars")?;
        let expected_f = count_to_f64(expected_bars, "expected_bars")?;
        missing_f / expected_f
    };

    if expected_bars > 0 && gap_loss > 0.05 {
        tracing::warn!(
            "Large gaps detected: {} missing bars ({:.2}%)",
            missing_bars,
            gap_loss * 100.0
        );
    }

    Ok(GapStats {
        expected_bars,
        missing_bars,
        gap_loss,
    })
}

fn is_session_active(timestamp_ns: i64, sessions: Option<&[SessionConfig]>) -> Result<bool, DataError> {
    match sessions {
        Some(configured) if !configured.is_empty() => {
            let seconds = seconds_since_midnight_utc(timestamp_ns)?;
            Ok(configured.iter().any(|s| s.contains(seconds)))
        }
        _ => is_weekday_utc(timestamp_ns),
    }
}

fn seconds_since_midnight_utc(timestamp_ns: i64) -> Result<u32, DataError> {
    let seconds = timestamp_ns.div_euclid(1_000_000_000);
    let seconds_in_day = 86_400;
    let rem = seconds.rem_euclid(seconds_in_day);
    u32_from_i64(rem)
}

fn is_weekday_utc(timestamp_ns: i64) -> Result<bool, DataError> {
    Ok(weekday_index_utc(timestamp_ns)? < 5)
}

fn weekday_index_utc(timestamp_ns: i64) -> Result<u32, DataError> {
    let seconds = timestamp_ns.div_euclid(1_000_000_000);
    let days_since_epoch = seconds.div_euclid(86_400);
    let weekday = (days_since_epoch + 3).rem_euclid(7);
    u32_from_i64(weekday)
}

fn count_to_f64(value: usize, label: &str) -> Result<f64, DataError> {
    let value_u64 = u64::try_from(value).map_err(|_| {
        DataError::CorruptData(format!("Count overflow for {label}: {value}"))
    })?;
    #[allow(clippy::cast_precision_loss)]
    let value_as_f64 = value_u64 as f64;
    Ok(value_as_f64)
}

fn u32_from_i64(value: i64) -> Result<u32, DataError> {
    u32::try_from(value).map_err(|_| {
        DataError::CorruptData(format!("Invalid u32 conversion from i64: {value}"))
    })
}
