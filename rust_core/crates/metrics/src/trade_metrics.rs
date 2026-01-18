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

/// Computes average win, average loss, largest win, and largest loss.
///
/// Returns (`avg_win`, `avg_loss`, `largest_win`, `largest_loss`):
/// - `avg_win`: average of positive results (0 if no wins)
/// - `avg_loss`: absolute value of average negative results (0 if no losses)
/// - `largest_win`: maximum positive result (0 if no wins)
/// - `largest_loss`: absolute value of minimum negative result (0 if no losses)
#[must_use]
#[allow(clippy::cast_precision_loss)] // Trade counts never exceed f64 mantissa precision
pub(crate) fn win_loss_stats(trades: &[Trade]) -> (f64, f64, f64, f64) {
    let mut sum_wins = 0.0;
    let mut sum_losses = 0.0;
    let mut win_count = 0usize;
    let mut loss_count = 0usize;
    let mut largest_win = 0.0_f64;
    let mut largest_loss = 0.0_f64;

    for trade in trades {
        if trade.result > 0.0 {
            sum_wins += trade.result;
            win_count += 1;
            largest_win = largest_win.max(trade.result);
        } else if trade.result < 0.0 {
            sum_losses += trade.result.abs();
            loss_count += 1;
            largest_loss = largest_loss.max(trade.result.abs());
        }
    }

    let avg_win = if win_count > 0 {
        sum_wins / win_count as f64
    } else {
        0.0
    };

    let avg_loss = if loss_count > 0 {
        sum_losses / loss_count as f64
    } else {
        0.0
    };

    (avg_win, avg_loss, largest_win, largest_loss)
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
