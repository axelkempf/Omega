//! Run context helpers (session/news masks, bar duration).

use std::collections::HashSet;

use omega_data::{NewsEvent, load_news_calendar, resolve_news_calendar_path};
use omega_types::{NewsFilterConfig, NewsImpact, SessionConfig};

use crate::error::BacktestError;

/// Precomputed runtime context for the event loop.
#[derive(Debug, Clone)]
pub struct RunContext {
    session_mask: Option<Vec<bool>>,
    news_mask: Option<Vec<bool>>,
    bar_duration_ns: i64,
}

impl RunContext {
    /// Builds a new `RunContext` from timestamps and config.
    ///
    /// # Errors
    /// Returns an error if the news calendar cannot be loaded or parsed.
    pub fn new(
        timestamps: &[i64],
        sessions: Option<&[SessionConfig]>,
        news_filter: Option<&NewsFilterConfig>,
        symbol: &str,
        bar_duration_ns: i64,
    ) -> Result<Self, BacktestError> {
        let session_mask = build_session_mask(timestamps, sessions);
        let news_mask = build_news_mask(timestamps, news_filter, symbol)?;

        Ok(Self {
            session_mask,
            news_mask,
            bar_duration_ns,
        })
    }

    /// Returns whether the session is open for a given primary index.
    #[must_use]
    pub fn session_open(&self, idx: usize) -> bool {
        self.session_mask
            .as_ref()
            .and_then(|mask| mask.get(idx).copied())
            .unwrap_or(true)
    }

    /// Returns whether news filter blocks entries for a given primary index.
    #[must_use]
    pub fn news_blocked(&self, idx: usize) -> bool {
        self.news_mask
            .as_ref()
            .and_then(|mask| mask.get(idx).copied())
            .unwrap_or(false)
    }

    /// Returns bar duration in nanoseconds.
    #[must_use]
    pub fn bar_duration_ns(&self) -> i64 {
        self.bar_duration_ns
    }
}

/// Builds a session mask aligned to primary timestamps.
#[must_use]
pub fn build_session_mask(
    timestamps: &[i64],
    sessions: Option<&[SessionConfig]>,
) -> Option<Vec<bool>> {
    let sessions = sessions?;
    if sessions.is_empty() {
        return None;
    }

    let mut mask = Vec::with_capacity(timestamps.len());
    for &ts in timestamps {
        let seconds = seconds_since_midnight_utc(ts);
        let open = seconds
            .map(|s| sessions.iter().any(|session| session.contains(s)))
            .unwrap_or(true);
        mask.push(open);
    }

    Some(mask)
}

/// Builds a news mask aligned to primary timestamps.
///
/// # Errors
/// Returns an error if the news calendar cannot be loaded.
pub fn build_news_mask(
    timestamps: &[i64],
    news_filter: Option<&NewsFilterConfig>,
    symbol: &str,
) -> Result<Option<Vec<bool>>, BacktestError> {
    let Some(filter) = news_filter else {
        return Ok(None);
    };
    if !filter.enabled {
        return Ok(None);
    }

    let currencies = filter
        .currencies
        .clone()
        .unwrap_or_else(|| derive_currencies_from_symbol(symbol));
    let currency_set: HashSet<String> = currencies
        .into_iter()
        .map(|c| c.trim().to_uppercase())
        .filter(|c| !c.is_empty())
        .collect();

    let path = resolve_news_calendar_path();
    let events = load_news_calendar(&path)?;
    let min_rank = impact_rank(filter.min_impact);

    let mut filtered: Vec<NewsEvent> = events
        .into_iter()
        .filter(|event| impact_rank_str(&event.impact) >= min_rank)
        .filter(|event| {
            if currency_set.is_empty() {
                return true;
            }
            currency_set.contains(&event.currency)
        })
        .collect();

    if filtered.is_empty() {
        return Ok(Some(vec![false; timestamps.len()]));
    }

    filtered.sort_by_key(|event| event.timestamp_ns);

    let pre_ns = minutes_to_ns(filter.minutes_before);
    let post_ns = minutes_to_ns(filter.minutes_after);

    let mut mask = vec![false; timestamps.len()];
    let mut event_idx = 0usize;

    for (idx, &ts) in timestamps.iter().enumerate() {
        while event_idx < filtered.len() {
            let event_ts = filtered[event_idx].timestamp_ns;
            let event_end = event_ts.saturating_add(post_ns);
            if event_end < ts {
                event_idx += 1;
            } else {
                break;
            }
        }

        if event_idx < filtered.len() {
            let event_ts = filtered[event_idx].timestamp_ns;
            let event_start = event_ts.saturating_sub(pre_ns);
            let event_end = event_ts.saturating_add(post_ns);
            if ts >= event_start && ts <= event_end {
                mask[idx] = true;
            }
        }
    }

    Ok(Some(mask))
}

fn impact_rank(impact: NewsImpact) -> u8 {
    match impact {
        NewsImpact::Low => 1,
        NewsImpact::Medium => 2,
        NewsImpact::High => 3,
    }
}

fn impact_rank_str(value: &str) -> u8 {
    match value.trim().to_ascii_uppercase().as_str() {
        "LOW" => 1,
        "MEDIUM" => 2,
        "HIGH" => 3,
        _ => 0,
    }
}

fn derive_currencies_from_symbol(symbol: &str) -> Vec<String> {
    let up = symbol.trim().to_uppercase();
    if up.len() >= 6 {
        vec![up[0..3].to_string(), up[3..6].to_string()]
    } else {
        Vec::new()
    }
}

fn minutes_to_ns(minutes: u32) -> i64 {
    let secs = i64::from(minutes).saturating_mul(60);
    secs.saturating_mul(1_000_000_000)
}

fn seconds_since_midnight_utc(timestamp_ns: i64) -> Result<u32, BacktestError> {
    let seconds = timestamp_ns.div_euclid(1_000_000_000);
    let rem = seconds.rem_euclid(86_400);
    u32::try_from(rem).map_err(|_| BacktestError::Runtime("invalid timestamp".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_mask_basic() {
        let sessions = vec![SessionConfig {
            start: "08:00".to_string(),
            end: "10:00".to_string(),
        }];
        let timestamps = vec![0, 8 * 3_600 * 1_000_000_000];
        let mask = build_session_mask(&timestamps, Some(&sessions)).unwrap();
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn test_news_mask_empty_filter() {
        let timestamps = vec![0, 1, 2];
        let mask = build_news_mask(&timestamps, None, "EURUSD").unwrap();
        assert!(mask.is_none());
    }
}
