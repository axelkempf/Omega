//! Helpers for trade-derived metrics.

use std::collections::HashSet;

use omega_types::Trade;

const NS_PER_DAY: i64 = 86_400_000_000_000;

/// Counts total trades, wins, and losses.
#[must_use]
pub(crate) fn count_trades(trades: &[Trade]) -> (u64, u64, u64) {
    let total = usize_to_u64(trades.len());
    let wins = usize_to_u64(trades.iter().filter(|trade| trade.result > 0.0).count());
    let losses = usize_to_u64(trades.iter().filter(|trade| trade.result < 0.0).count());
    (total, wins, losses)
}

/// Computes gross profit and gross loss (absolute sum of negative results).
#[must_use]
pub(crate) fn gross_profit_loss(trades: &[Trade]) -> (f64, f64) {
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    for trade in trades {
        if trade.result > 0.0 {
            gross_profit += trade.result;
        } else if trade.result < 0.0 {
            gross_loss += trade.result.abs();
        }
    }

    (gross_profit, gross_loss)
}

/// Counts unique active days based on trade entry timestamps.
#[must_use]
pub(crate) fn active_days(trades: &[Trade]) -> u64 {
    let unique_days: HashSet<i64> = trades
        .iter()
        .map(|trade| trade.entry_time_ns / NS_PER_DAY)
        .collect();
    usize_to_u64(unique_days.len())
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
