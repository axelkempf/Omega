//! Slippage calculation for trade execution simulation.
//!
//! Provides deterministic slippage calculation with optional random component.
//! Uses `ChaCha8` RNG for reproducible results across platforms.
//!
//! ## Formula
//!
//! ```text
//! slippage = fixed_pips + random(0, random_pips)
//! adjusted_price = price ± (slippage × pip_size)
//!     + for long (buy at higher price)
//!     - for short (sell at lower price)
//! ```

use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{OmegaError, Result};

/// Direction indicator for trade.
/// 1 = Long (buy), -1 = Short (sell)
pub type Direction = i8;

/// Long direction constant (buy)
pub const DIRECTION_LONG: Direction = 1;
/// Short direction constant (sell)
pub const DIRECTION_SHORT: Direction = -1;

/// Calculate slippage for a single trade.
///
/// # Arguments
/// * `price` - Original order price
/// * `direction` - Trade direction: 1 for long, -1 for short
/// * `pip_size` - Price increment per pip (e.g., 0.0001 for EURUSD)
/// * `fixed_pips` - Fixed slippage in pips
/// * `random_pips` - Maximum additional random slippage in pips
/// * `seed` - Optional seed for deterministic random component
///
/// # Returns
/// Execution price after slippage
///
/// # Errors
/// Returns error if direction is invalid (not 1 or -1)
///
/// # Example
///
/// ```python
/// from omega._rust import calculate_slippage
///
/// # Long trade with fixed + random slippage
/// price = calculate_slippage(
///     price=1.10000,
///     direction=1,  # long
///     pip_size=0.0001,
///     fixed_pips=0.5,
///     random_pips=1.0,
///     seed=42  # for reproducibility
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (price, direction, pip_size, fixed_pips, random_pips, seed=None))]
pub fn calculate_slippage(
    price: f64,
    direction: i8,
    pip_size: f64,
    fixed_pips: f64,
    random_pips: f64,
    seed: Option<u64>,
) -> PyResult<f64> {
    calculate_slippage_impl(price, direction, pip_size, fixed_pips, random_pips, seed)
        .map_err(Into::into)
}

/// Internal implementation for slippage calculation.
pub fn calculate_slippage_impl(
    price: f64,
    direction: i8,
    pip_size: f64,
    fixed_pips: f64,
    random_pips: f64,
    seed: Option<u64>,
) -> Result<f64> {
    // Validate direction
    if direction != DIRECTION_LONG && direction != DIRECTION_SHORT {
        return Err(OmegaError::InvalidParameter {
            reason: format!("direction must be 1 (long) or -1 (short), got {direction}"),
        });
    }

    // Calculate random component
    let random_component = if random_pips > 0.0 {
        if let Some(s) = seed {
            let mut rng = ChaCha8Rng::seed_from_u64(s);
            rng.random_range(0.0..random_pips)
        } else {
            let mut rng = rand::rng();
            rng.random_range(0.0..random_pips)
        }
    } else {
        0.0
    };

    let total_slippage = fixed_pips + random_component;
    let slippage_amount = total_slippage * pip_size;

    // Long: price increases (buy at higher price)
    // Short: price decreases (sell at lower price)
    let adjusted_price = if direction == DIRECTION_LONG {
        price + slippage_amount
    } else {
        price - slippage_amount
    };

    Ok(adjusted_price)
}

/// Calculate slippage for a batch of trades.
///
/// Optimized for processing multiple trades efficiently.
/// Each trade gets a unique seed derived from `base_seed + index`.
///
/// # Arguments
/// * `prices` - Vector of original prices
/// * `directions` - Vector of directions (1=long, -1=short)
/// * `pip_size` - Price increment per pip
/// * `fixed_pips` - Fixed slippage in pips
/// * `random_pips` - Maximum additional random slippage
/// * `seed` - Base seed for deterministic results
///
/// # Returns
/// Vector of adjusted prices after slippage
///
/// # Errors
/// Returns error if prices and directions have different lengths
#[pyfunction]
#[pyo3(signature = (prices, directions, pip_size, fixed_pips, random_pips, seed=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn calculate_slippage_batch(
    prices: Vec<f64>,
    directions: Vec<i8>,
    pip_size: f64,
    fixed_pips: f64,
    random_pips: f64,
    seed: Option<u64>,
) -> PyResult<Vec<f64>> {
    calculate_slippage_batch_impl(&prices, &directions, pip_size, fixed_pips, random_pips, seed)
        .map_err(Into::into)
}

/// Internal implementation for batch slippage calculation.
pub fn calculate_slippage_batch_impl(
    prices: &[f64],
    directions: &[i8],
    pip_size: f64,
    fixed_pips: f64,
    random_pips: f64,
    seed: Option<u64>,
) -> Result<Vec<f64>> {
    if prices.len() != directions.len() {
        return Err(OmegaError::InvalidParameter {
            reason: format!(
                "prices and directions must have same length: {} vs {}",
                prices.len(),
                directions.len()
            ),
        });
    }

    let base_seed = seed.unwrap_or(42);

    prices
        .iter()
        .zip(directions.iter())
        .enumerate()
        .map(|(i, (&price, &direction))| {
            // Each trade gets a unique seed derived from base seed + index
            let trade_seed = Some(base_seed.wrapping_add(i as u64));
            calculate_slippage_impl(price, direction, pip_size, fixed_pips, random_pips, trade_seed)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_slippage_deterministic_with_seed() {
        let price = 1.10000;
        let pip_size = 0.0001;
        let fixed_pips = 0.5;
        let random_pips = 1.0;
        let seed = Some(42_u64);

        let result1 =
            calculate_slippage_impl(price, DIRECTION_LONG, pip_size, fixed_pips, random_pips, seed)
                .unwrap();
        let result2 =
            calculate_slippage_impl(price, DIRECTION_LONG, pip_size, fixed_pips, random_pips, seed)
                .unwrap();

        assert_relative_eq!(result1, result2, epsilon = 1e-10);
    }

    #[test]
    fn test_slippage_direction_awareness() {
        let price = 1.10000;
        let pip_size = 0.0001;
        let fixed_pips = 1.0;
        let random_pips = 0.0; // No random for deterministic test

        let long_price = calculate_slippage_impl(
            price,
            DIRECTION_LONG,
            pip_size,
            fixed_pips,
            random_pips,
            None,
        )
        .unwrap();
        let short_price = calculate_slippage_impl(
            price,
            DIRECTION_SHORT,
            pip_size,
            fixed_pips,
            random_pips,
            None,
        )
        .unwrap();

        assert!(long_price > price, "Long slippage should increase price");
        assert!(short_price < price, "Short slippage should decrease price");
    }

    #[test]
    fn test_slippage_fixed_only() {
        let price = 1.10000;
        let pip_size = 0.0001;
        let fixed_pips = 0.5;
        let random_pips = 0.0;

        let long_price = calculate_slippage_impl(
            price,
            DIRECTION_LONG,
            pip_size,
            fixed_pips,
            random_pips,
            None,
        )
        .unwrap();
        let short_price = calculate_slippage_impl(
            price,
            DIRECTION_SHORT,
            pip_size,
            fixed_pips,
            random_pips,
            None,
        )
        .unwrap();

        // Expected: price + 0.5 * 0.0001 = 1.10005
        assert_relative_eq!(long_price, 1.10005, epsilon = 1e-10);
        // Expected: price - 0.5 * 0.0001 = 1.09995
        assert_relative_eq!(short_price, 1.09995, epsilon = 1e-10);
    }

    #[test]
    fn test_slippage_invalid_direction() {
        let result = calculate_slippage_impl(1.0, 0, 0.0001, 0.5, 1.0, None);
        assert!(result.is_err());

        let result = calculate_slippage_impl(1.0, 2, 0.0001, 0.5, 1.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_slippage_batch_deterministic() {
        let prices = vec![1.1, 1.2, 1.3];
        let directions = vec![1, -1, 1];
        let seed = Some(42_u64);

        let result1 =
            calculate_slippage_batch_impl(&prices, &directions, 0.0001, 0.5, 1.0, seed).unwrap();
        let result2 =
            calculate_slippage_batch_impl(&prices, &directions, 0.0001, 0.5, 1.0, seed).unwrap();

        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_relative_eq!(r1, r2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_slippage_batch_length_mismatch() {
        let prices = vec![1.1, 1.2, 1.3];
        let directions = vec![1, -1]; // Shorter than prices

        let result =
            calculate_slippage_batch_impl(&prices, &directions, 0.0001, 0.5, 1.0, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_slippage_zero_pips() {
        let price = 1.10000;
        let result =
            calculate_slippage_impl(price, DIRECTION_LONG, 0.0001, 0.0, 0.0, None).unwrap();
        assert_relative_eq!(result, price, epsilon = 1e-10);
    }
}
