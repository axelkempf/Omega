use std::collections::HashSet;

use crate::error::DataError;
use omega_types::Candle;

/// Ergebnis des Bid/Ask Alignments
#[derive(Debug)]
pub struct AlignedData {
    pub bid: Vec<Candle>,
    pub ask: Vec<Candle>,
    pub timestamps: Vec<i64>,
    pub alignment_stats: AlignmentStats,
}

#[derive(Debug)]
pub struct AlignmentStats {
    pub bid_count_before: usize,
    pub ask_count_before: usize,
    pub aligned_count: usize,
    pub discarded_count: usize,
    pub alignment_loss: f64,
}

/// Führt Inner Join auf Timestamps durch.
/// Kritisch: Keine Interpolation – nur gemeinsame Timestamps werden behalten.
pub fn align_bid_ask(bid: Vec<Candle>, ask: Vec<Candle>) -> Result<AlignedData, DataError> {
    let bid_count_before = bid.len();
    let ask_count_before = ask.len();
    if bid_count_before == 0 || ask_count_before == 0 {
        return Err(DataError::InsufficientData {
            required: 1,
            available: bid_count_before.min(ask_count_before),
        });
    }

    let bid_timestamps: HashSet<i64> = bid.iter().map(|c| c.timestamp_ns).collect();
    let ask_timestamps: HashSet<i64> = ask.iter().map(|c| c.timestamp_ns).collect();

    let common: HashSet<i64> = bid_timestamps
        .intersection(&ask_timestamps)
        .copied()
        .collect();
    if common.is_empty() {
        return Err(DataError::AlignmentFailure(
            "No common timestamps between bid and ask".to_string(),
        ));
    }

    let mut aligned_bid: Vec<Candle> = bid
        .into_iter()
        .filter(|c| common.contains(&c.timestamp_ns))
        .collect();
    let mut aligned_ask: Vec<Candle> = ask
        .into_iter()
        .filter(|c| common.contains(&c.timestamp_ns))
        .collect();

    aligned_bid.sort_by_key(|c| c.timestamp_ns);
    aligned_ask.sort_by_key(|c| c.timestamp_ns);

    let aligned_count = aligned_bid.len();
    let discarded_count = bid_count_before + ask_count_before - 2 * aligned_count;
    let loss_bid = 1.0 - (aligned_count as f64 / bid_count_before as f64);
    let loss_ask = 1.0 - (aligned_count as f64 / ask_count_before as f64);
    let alignment_loss = loss_bid.max(loss_ask);

    if alignment_loss > 0.01 {
        return Err(DataError::AlignmentFailure(format!(
            "Alignment loss too high: {:.4}",
            alignment_loss
        )));
    } else if alignment_loss > 0.0 {
        tracing::warn!(
            "Alignment discarded {} bars (loss {:.2}%)",
            discarded_count,
            alignment_loss * 100.0
        );
    }

    let timestamps: Vec<i64> = aligned_bid.iter().map(|c| c.timestamp_ns).collect();

    Ok(AlignedData {
        bid: aligned_bid,
        ask: aligned_ask,
        timestamps,
        alignment_stats: AlignmentStats {
            bid_count_before,
            ask_count_before,
            aligned_count,
            discarded_count,
            alignment_loss,
        },
    })
}
