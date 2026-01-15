use omega_types::{Candle, SessionConfig};

/// Filters candles to those inside configured trading sessions (UTC).
/// If `sessions` is empty, the input is returned unchanged.
pub fn filter_by_sessions(candles: Vec<Candle>, sessions: &[SessionConfig]) -> Vec<Candle> {
    if sessions.is_empty() {
        return candles;
    }

    candles
        .into_iter()
        .filter(|c| {
            let seconds = seconds_since_midnight_utc(c.timestamp_ns);
            sessions.iter().any(|s| s.contains(seconds))
        })
        .collect()
}

fn seconds_since_midnight_utc(timestamp_ns: i64) -> u32 {
    let seconds = timestamp_ns / 1_000_000_000;
    let seconds_in_day = 86_400;
    let rem = seconds.rem_euclid(seconds_in_day);
    rem as u32
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

        let filtered = filter_by_sessions(candles, &sessions);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].timestamp_ns, 8 * 3_600 * 1_000_000_000);
    }
}
