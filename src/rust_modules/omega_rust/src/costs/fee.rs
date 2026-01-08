//! Fee calculation for trade execution simulation.
//!
//! Supports fee calculation based on per-million notional with optional minimum fee.
//!
//! ## Formula
//!
//! ```text
//! notional = volume_lots × contract_size × price
//! fee = max(notional / 1_000_000 × per_million, min_fee)
//! ```

use pyo3::prelude::*;

use crate::error::{OmegaError, Result};

/// Calculate fee for a single trade.
///
/// Formula: `fee = max(notional / 1_000_000 × per_million, min_fee)`
/// where `notional = volume_lots × contract_size × price`
///
/// # Arguments
/// * `volume_lots` - Trade volume in lots
/// * `price` - Price per unit (quote currency)
/// * `contract_size` - Contract size per lot (e.g., 100,000 for FX)
/// * `per_million` - Commission per 1 million notional
/// * `min_fee` - Minimum fee per transaction
///
/// # Returns
/// Absolute fee amount (same currency as price)
///
/// # Errors
/// Returns error if any input is negative or `contract_size` is zero
///
/// # Example
///
/// ```python
/// from omega._rust import calculate_fee
///
/// # Calculate fee for 1 lot EURUSD
/// fee = calculate_fee(
///     volume_lots=1.0,
///     price=1.10000,
///     contract_size=100_000.0,
///     per_million=30.0,
///     min_fee=0.01
/// )
/// # fee = 3.3 (110,000 / 1,000,000 * 30)
/// ```
#[pyfunction]
#[pyo3(signature = (volume_lots, price, contract_size, per_million, min_fee=0.0))]
pub fn calculate_fee(
    volume_lots: f64,
    price: f64,
    contract_size: f64,
    per_million: f64,
    min_fee: f64,
) -> PyResult<f64> {
    calculate_fee_impl(volume_lots, price, contract_size, per_million, min_fee).map_err(Into::into)
}

/// Internal implementation for fee calculation.
pub fn calculate_fee_impl(
    volume_lots: f64,
    price: f64,
    contract_size: f64,
    per_million: f64,
    min_fee: f64,
) -> Result<f64> {
    // Input validation
    if volume_lots < 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: format!("volume_lots cannot be negative: {volume_lots}"),
        });
    }
    if price < 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: format!("price cannot be negative: {price}"),
        });
    }
    if contract_size <= 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: format!("contract_size must be positive: {contract_size}"),
        });
    }
    if per_million < 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: format!("per_million cannot be negative: {per_million}"),
        });
    }
    if min_fee < 0.0 {
        return Err(OmegaError::InvalidParameter {
            reason: format!("min_fee cannot be negative: {min_fee}"),
        });
    }

    let notional = volume_lots * contract_size * price;
    let fee = (notional / 1_000_000.0) * per_million;

    // Apply minimum fee
    let final_fee = if min_fee > 0.0 { fee.max(min_fee) } else { fee };

    Ok(final_fee)
}

/// Calculate fees for a batch of trades.
///
/// Optimized for processing multiple trades efficiently.
///
/// # Arguments
/// * `volume_lots` - Vector of trade volumes
/// * `prices` - Vector of prices
/// * `contract_size` - Contract size per lot (same for all)
/// * `per_million` - Commission per 1 million notional
/// * `min_fee` - Minimum fee per transaction
///
/// # Returns
/// Vector of fee amounts
///
/// # Errors
/// Returns error if `volume_lots` and prices have different lengths
#[pyfunction]
#[pyo3(signature = (volume_lots, prices, contract_size, per_million, min_fee=0.0))]
#[allow(clippy::needless_pass_by_value)]
pub fn calculate_fee_batch(
    volume_lots: Vec<f64>,
    prices: Vec<f64>,
    contract_size: f64,
    per_million: f64,
    min_fee: f64,
) -> PyResult<Vec<f64>> {
    calculate_fee_batch_impl(&volume_lots, &prices, contract_size, per_million, min_fee)
        .map_err(Into::into)
}

/// Internal implementation for batch fee calculation.
pub fn calculate_fee_batch_impl(
    volume_lots: &[f64],
    prices: &[f64],
    contract_size: f64,
    per_million: f64,
    min_fee: f64,
) -> Result<Vec<f64>> {
    if volume_lots.len() != prices.len() {
        return Err(OmegaError::InvalidParameter {
            reason: format!(
                "volume_lots and prices must have same length: {} vs {}",
                volume_lots.len(),
                prices.len()
            ),
        });
    }

    volume_lots
        .iter()
        .zip(prices.iter())
        .map(|(&vol, &price)| calculate_fee_impl(vol, price, contract_size, per_million, min_fee))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fee_basic_calculation() {
        // 1 lot × 100,000 contract × 1.10 price = 110,000 notional
        // 110,000 / 1,000,000 × 30 = 3.3
        let fee = calculate_fee_impl(1.0, 1.10, 100_000.0, 30.0, 0.0).unwrap();
        assert_relative_eq!(fee, 3.3, epsilon = 1e-10);
    }

    #[test]
    fn test_fee_with_minimum() {
        // Very small trade should hit minimum
        // 0.001 × 100,000 × 1.0 = 100 notional
        // 100 / 1,000,000 × 30 = 0.003
        // min_fee = 1.0 > 0.003, so fee = 1.0
        let fee = calculate_fee_impl(0.001, 1.0, 100_000.0, 30.0, 1.0).unwrap();
        assert_relative_eq!(fee, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fee_deterministic() {
        let fee1 = calculate_fee_impl(1.0, 1.10, 100_000.0, 30.0, 0.01).unwrap();
        let fee2 = calculate_fee_impl(1.0, 1.10, 100_000.0, 30.0, 0.01).unwrap();
        assert_relative_eq!(fee1, fee2, epsilon = 1e-10);
    }

    #[test]
    fn test_fee_zero_volume() {
        let fee = calculate_fee_impl(0.0, 1.10, 100_000.0, 30.0, 0.0).unwrap();
        assert_relative_eq!(fee, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fee_negative_volume_error() {
        let result = calculate_fee_impl(-1.0, 1.10, 100_000.0, 30.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fee_zero_contract_size_error() {
        let result = calculate_fee_impl(1.0, 1.10, 0.0, 30.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fee_batch() {
        let volumes = vec![0.01, 0.1, 1.0];
        let prices = vec![1.1, 1.1, 1.1];

        let fees =
            calculate_fee_batch_impl(&volumes, &prices, 100_000.0, 30.0, 0.01).unwrap();

        assert_eq!(fees.len(), 3);
        // 0.01 × 100,000 × 1.1 / 1,000,000 × 30 = 0.033
        assert_relative_eq!(fees[0], 0.033, epsilon = 1e-10);
        // 0.1 × 100,000 × 1.1 / 1,000,000 × 30 = 0.33
        assert_relative_eq!(fees[1], 0.33, epsilon = 1e-10);
        // 1.0 × 100,000 × 1.1 / 1,000,000 × 30 = 3.3
        assert_relative_eq!(fees[2], 3.3, epsilon = 1e-10);
    }

    #[test]
    fn test_fee_batch_length_mismatch() {
        let volumes = vec![0.01, 0.1, 1.0];
        let prices = vec![1.1, 1.1]; // Shorter than volumes

        let result = calculate_fee_batch_impl(&volumes, &prices, 100_000.0, 30.0, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_fee_matches_golden_reference() {
        // From golden file: volume=1.0, price=1.1, contract=100000, per_million=30.0, min_fee=0.01
        // Expected: 3.3
        let fee = calculate_fee_impl(1.0, 1.1, 100_000.0, 30.0, 0.01).unwrap();
        assert_relative_eq!(fee, 3.3, epsilon = 1e-8);

        // From golden file: volume=0.01, price=1.1, contract=100000, per_million=30.0, min_fee=0.01
        // Expected: 0.033
        let fee = calculate_fee_impl(0.01, 1.1, 100_000.0, 30.0, 0.01).unwrap();
        assert_relative_eq!(fee, 0.033, epsilon = 1e-8);

        // From golden file: volume=10.0, price=1.2, contract=100000, per_million=30.0, min_fee=0.01
        // Expected: 36.0
        let fee = calculate_fee_impl(10.0, 1.2, 100_000.0, 30.0, 0.01).unwrap();
        assert_relative_eq!(fee, 36.0, epsilon = 1e-8);
    }
}
