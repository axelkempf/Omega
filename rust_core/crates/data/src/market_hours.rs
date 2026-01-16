//! Session handling helpers.

use omega_types::{Candle, SessionConfig};

use crate::error::DataError;

/// Filters candles to those inside configured trading sessions (UTC).
///
/// If `sessions` is empty, a 24/5 default is applied (Mon–Fri, full-day).
///
/// # Errors
/// - [`DataError::CorruptData`] when timestamp conversion fails.
pub fn filter_by_sessions(
    candles: Vec<Candle>,
    sessions: &[SessionConfig],
) -> Result<Vec<Candle>, DataError> {
    if sessions.is_empty() {
        let mut filtered = Vec::with_capacity(candles.len());
        for candle in candles {
            if is_weekday_utc(candle.timestamp_ns)? {
                filtered.push(candle);
            }
        }
        return Ok(filtered);
    }

    let mut filtered = Vec::with_capacity(candles.len());
    for candle in candles {
        let seconds = seconds_since_midnight_utc(candle.timestamp_ns)?;
        if sessions.iter().any(|s| s.contains(seconds)) {
            filtered.push(candle);
        }
    }

    Ok(filtered)
}

fn seconds_since_midnight_utc(timestamp_ns: i64) -> Result<u32, DataError> {
    let seconds = timestamp_ns / 1_000_000_000;
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

fn u32_from_i64(value: i64) -> Result<u32, DataError> {
    u32::try_from(value)
        .map_err(|_| DataError::CorruptData(format!("Invalid u32 conversion from i64: {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_by_sessions() {
        let sessions = vec![SessionConfig {
            start: "08:00".to_string(),
            end: "10:00".to_string(),
        }];

        let candles = vec![
            Candle {
                timestamp_ns: 0, // 00:00 UTC
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            },
            Candle {
                timestamp_ns: 8 * 3_600 * 1_000_000_000, // 08:00 UTC
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            },
        ];

        let filtered = filter_by_sessions(candles, &sessions).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].timestamp_ns, 8 * 3_600 * 1_000_000_000);
    }

    #[test]
    fn test_filter_by_sessions_default_24_5() {
        let monday_ts = 1_704_067_200_000_000_000i64; // 2024-01-01 00:00:00 UTC
        let saturday_ts = 1_704_499_200_000_000_000i64; // 2024-01-06 00:00:00 UTC

        let candles = vec![
            Candle {
                timestamp_ns: monday_ts,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            },
            Candle {
                timestamp_ns: saturday_ts,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            },
        ];

        let filtered = filter_by_sessions(candles, &[]).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].timestamp_ns, monday_ts);
    }

    #[test]
    fn test_u32_from_i64_rejects_negative() {
        let err = u32_from_i64(-1).unwrap_err();
        assert!(matches!(err, DataError::CorruptData(_)));
    }
}
