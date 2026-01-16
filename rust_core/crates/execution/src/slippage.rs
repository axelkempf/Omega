//! Slippage models for order execution simulation.
//!
//! Slippage represents the difference between the expected price of a trade
//! and the actual price at which the trade is executed.

use omega_types::Direction;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Trait for slippage calculation models.
///
/// Implementations must be deterministic when given the same RNG state.
pub trait SlippageModel: Send + Sync {
    /// Calculates slippage for a fill.
    ///
    /// # Arguments
    /// * `price` - The signal/entry price
    /// * `direction` - Trade direction (Long or Short)
    /// * `rng` - Deterministic random number generator
    ///
    /// # Returns
    /// Slippage in price units (positive = adverse for the trader)
    fn calculate(&self, price: f64, direction: Direction, rng: &mut ChaCha8Rng) -> f64;

    /// Returns the model name for logging/debugging.
    fn name(&self) -> &'static str;
}

/// Fixed slippage in pips.
///
/// Always applies the same slippage regardless of market conditions.
/// Useful for conservative backtesting assumptions.
#[derive(Debug, Clone)]
pub struct FixedSlippage {
    /// Slippage amount in pips
    pub pips: f64,
    /// Pip size for the instrument (e.g., 0.0001 for EUR/USD)
    pub pip_size: f64,
}

impl FixedSlippage {
    /// Creates a new fixed slippage model.
    #[must_use]
    pub fn new(pips: f64, pip_size: f64) -> Self {
        Self { pips, pip_size }
    }
}

impl SlippageModel for FixedSlippage {
    fn calculate(&self, _price: f64, direction: Direction, _rng: &mut ChaCha8Rng) -> f64 {
        let base = self.pips * self.pip_size;
        match direction {
            Direction::Long => base,   // Buy at higher price
            Direction::Short => -base, // Sell at lower price
        }
    }

    fn name(&self) -> &'static str {
        "FixedSlippage"
    }
}

/// Volatility-based slippage with randomness.
///
/// Adds jitter around a base slippage value to simulate
/// realistic market conditions where slippage varies.
#[derive(Debug, Clone)]
pub struct VolatilitySlippage {
    /// Base slippage amount in pips
    pub base_pips: f64,
    /// Pip size for the instrument
    pub pip_size: f64,
    /// Jitter factor (0.0 - 1.0), controls variance around base
    pub jitter_factor: f64,
}

impl VolatilitySlippage {
    /// Creates a new volatility-based slippage model.
    ///
    /// # Arguments
    /// * `base_pips` - Base slippage in pips
    /// * `pip_size` - Pip size for the instrument
    /// * `jitter_factor` - Variance factor (0.0 = no jitter, 1.0 = full variance)
    #[must_use]
    pub fn new(base_pips: f64, pip_size: f64, jitter_factor: f64) -> Self {
        Self {
            base_pips,
            pip_size,
            jitter_factor: jitter_factor.clamp(0.0, 1.0),
        }
    }
}

impl SlippageModel for VolatilitySlippage {
    fn calculate(&self, _price: f64, direction: Direction, rng: &mut ChaCha8Rng) -> f64 {
        let jitter: f64 = rng.gen_range(-self.jitter_factor..self.jitter_factor);
        let pips = self.base_pips * (1.0 + jitter);
        let base = pips * self.pip_size;
        match direction {
            Direction::Long => base,
            Direction::Short => -base,
        }
    }

    fn name(&self) -> &'static str {
        "VolatilitySlippage"
    }
}

/// Zero slippage model for testing or ideal conditions.
#[derive(Debug, Clone, Default)]
pub struct NoSlippage;

impl SlippageModel for NoSlippage {
    fn calculate(&self, _price: f64, _direction: Direction, _rng: &mut ChaCha8Rng) -> f64 {
        0.0
    }

    fn name(&self) -> &'static str {
        "NoSlippage"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;

    #[test]
    fn test_fixed_slippage_long() {
        let model = FixedSlippage::new(1.0, 0.0001);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let slippage = model.calculate(1.2000, Direction::Long, &mut rng);
        assert_relative_eq!(slippage, 0.0001, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_slippage_short() {
        let model = FixedSlippage::new(1.0, 0.0001);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let slippage = model.calculate(1.2000, Direction::Short, &mut rng);
        assert_relative_eq!(slippage, -0.0001, epsilon = 1e-10);
    }

    #[test]
    fn test_volatility_slippage_deterministic() {
        let model = VolatilitySlippage::new(1.0, 0.0001, 0.5);

        // Same seed should produce same result
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let s1 = model.calculate(1.2000, Direction::Long, &mut rng1);
        let s2 = model.calculate(1.2000, Direction::Long, &mut rng2);

        assert_relative_eq!(s1, s2, epsilon = 1e-10);
    }

    #[test]
    fn test_volatility_slippage_varies() {
        let model = VolatilitySlippage::new(1.0, 0.0001, 0.5);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let s1 = model.calculate(1.2000, Direction::Long, &mut rng);
        let s2 = model.calculate(1.2000, Direction::Long, &mut rng);

        // Different calls should produce different results
        assert!((s1 - s2).abs() > 1e-12);
    }

    #[test]
    fn test_no_slippage() {
        let model = NoSlippage;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        assert_relative_eq!(
            model.calculate(1.2000, Direction::Long, &mut rng),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            model.calculate(1.2000, Direction::Short, &mut rng),
            0.0,
            epsilon = 1e-10
        );
    }
}
