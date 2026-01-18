//! Warmup validation helpers.

use omega_data::CandleStore;
use omega_strategy::IndicatorRequirement;
use omega_types::Timeframe;

use crate::error::BacktestError;

/// Validates that sufficient data exists for warmup.
///
/// # Errors
/// Returns an error if the available data is shorter than the warmup period.
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
///
/// # Errors
/// Returns an error if HTF data is present but shorter than the warmup period.
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

/// Computes the effective warmup for aligned indicators.
///
/// **DEPRECATED**: This function is no longer needed since HTF indicators are now stored
/// only in the `htf_cache` at their native length (not stretched to primary length).
/// Each timeframe has its own independent warmup period.
///
/// When HTF indicators (e.g., D1-EMA) are aligned to the primary timeframe (e.g., H1),
/// the alignment stretches the NaN period by the timeframe ratio.
/// For example, D1-EMA(75) aligned to H1 produces 75 × 24 = 1800 NaN values.
///
/// This function calculates the maximum effective warmup needed for all aligned HTF indicators,
/// ensuring `validate_indicators` correctly skips all NaN values.
///
/// # Arguments
/// * `warmup_bars` - The configured warmup per timeframe
/// * `primary_tf` - The primary timeframe
/// * `requirements` - The indicator requirements with optional timeframe
///
/// # Returns
/// The effective warmup for the aligned primary cache
#[allow(dead_code)]
#[must_use]
pub fn compute_aligned_warmup(
    warmup_bars: usize,
    primary_tf: Timeframe,
    requirements: &[IndicatorRequirement],
) -> usize {
    let primary_seconds = primary_tf.to_seconds();

    requirements
        .iter()
        .filter_map(|req| {
            req.timeframe.as_deref().and_then(|tf_str| {
                tf_str.parse::<Timeframe>().ok().map(|tf| {
                    let htf_seconds = tf.to_seconds();
                    if htf_seconds > primary_seconds {
                        // HTF indicator: scale warmup by tf_ratio
                        let tf_ratio = htf_seconds / primary_seconds;
                        let ratio = usize::try_from(tf_ratio).unwrap_or(usize::MAX);
                        warmup_bars.saturating_mul(ratio)
                    } else {
                        // Same or lower TF: use regular warmup
                        warmup_bars
                    }
                })
            })
        })
        .max()
        .unwrap_or(warmup_bars)
        .max(warmup_bars) // Always at least warmup_bars
}
