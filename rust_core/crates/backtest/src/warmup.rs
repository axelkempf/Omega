//! Warmup validation helpers.

use omega_data::CandleStore;

use crate::error::BacktestError;

/// Validates that sufficient data exists for warmup.
pub fn validate_warmup(data: &CandleStore, warmup_bars: usize) -> Result<(), BacktestError> {
    if data.len() <= warmup_bars {
        return Err(BacktestError::InsufficientData {
            required: warmup_bars + 1,
            available: data.len(),
        });
    }

    tracing::info!(
        "Warmup validated: {} bars required, {} available ({} trading bars)",
        warmup_bars,
        data.len(),
        data.len().saturating_sub(warmup_bars)
    );

    Ok(())
}

/// Validates HTF warmup (if HTF enabled).
pub fn validate_htf_warmup(
    htf_data: Option<&CandleStore>,
    warmup_bars: usize,
) -> Result<(), BacktestError> {
    if let Some(htf) = htf_data
        && htf.len() < warmup_bars
    {
        return Err(BacktestError::InsufficientHtfData {
            required: warmup_bars,
            available: htf.len(),
        });
    }
    Ok(())
}
