//! Governance validation helpers.

use crate::error::DataError;
use omega_types::Candle;

/// Validates a sequence of candles according to the governance contract.
///
/// # Errors
/// - [`DataError::EmptyData`] when `candles` is empty.
/// - [`DataError::CorruptData`] when data violates governance invariants.
pub fn validate_candles(candles: &[Candle]) -> Result<(), DataError> {
    if candles.is_empty() {
        return Err(DataError::EmptyData);
    }

    for (i, candle) in candles.iter().enumerate() {
        if !candle.open.is_finite()
            || !candle.high.is_finite()
            || !candle.low.is_finite()
            || !candle.close.is_finite()
            || !candle.volume.is_finite()
        {
            return Err(DataError::CorruptData(format!(
                "NaN/Inf at index {i}: {candle:?}"
            )));
        }

        if candle.volume < 0.0 {
            return Err(DataError::CorruptData(format!(
                "Negative volume at index {i}: {}",
                candle.volume
            )));
        }

        if candle.low > candle.open
            || candle.low > candle.close
            || candle.high < candle.open
            || candle.high < candle.close
            || candle.low > candle.high
        {
            return Err(DataError::CorruptData(format!(
                "Invalid OHLC at index {i}: low={}, high={}, open={}, close={}",
                candle.low, candle.high, candle.open, candle.close
            )));
        }

        if i > 0 && candle.timestamp_ns <= candles[i - 1].timestamp_ns {
            return Err(DataError::CorruptData(format!(
                "Non-monotonic timestamp at index {i}: {} <= {}",
                candle.timestamp_ns,
                candles[i - 1].timestamp_ns
            )));
        }
    }

    Ok(())
}

/// Validates bid/ask spread ordering (Open/Close must satisfy Bid <= Ask).
///
/// # Errors
/// - [`DataError::InvalidSpread`] when bid open/close exceeds ask open/close.
pub fn validate_spread(bid: &[Candle], ask: &[Candle]) -> Result<(), DataError> {
    for (i, (b, a)) in bid.iter().zip(ask.iter()).enumerate() {
        if b.open > a.open || b.close > a.close {
            return Err(DataError::InvalidSpread(format!(
                "Invalid spread at index {}: bid_close={} > ask_close={}",
                i, b.close, a.close
            )));
        }
    }
    Ok(())
}
