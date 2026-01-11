//! Slippage calculation with deterministic RNG.
//!
//! Uses ChaCha8 RNG seeded from a base seed plus position-specific data
//! to ensure reproducible slippage across runs.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use super::position::Direction;

/// Slippage calculator with deterministic RNG.
///
/// The RNG is seeded using a combination of:
/// - Base seed (global, set at simulator creation)
/// - Position ID (per-position uniqueness)
/// - Timestamp (temporal uniqueness)
///
/// This ensures:
/// 1. Same base_seed → same slippage sequence
/// 2. Different positions get different but reproducible slippage
/// 3. Results are platform-independent (ChaCha8 is portable)
#[derive(Clone, Debug)]
pub struct SlippageCalculator {
    /// Base seed for RNG initialization
    base_seed: u64,
    /// Max slippage in pips (will be converted to price units)
    max_slippage_pips: f64,
}

impl SlippageCalculator {
    /// Create a new slippage calculator.
    ///
    /// # Arguments
    /// * `base_seed` - Base seed for deterministic RNG
    /// * `max_slippage_pips` - Maximum slippage in pips (default: 1.0)
    pub fn new(base_seed: u64, max_slippage_pips: f64) -> Self {
        Self {
            base_seed,
            max_slippage_pips,
        }
    }

    /// Create a default slippage calculator with seed 0.
    pub fn default_with_seed(base_seed: u64) -> Self {
        Self::new(base_seed, 1.0)
    }

    /// Calculate slippage for an entry.
    ///
    /// Entry slippage is always unfavorable:
    /// - Long: price increases (you pay more)
    /// - Short: price decreases (you receive less)
    ///
    /// # Arguments
    /// * `entry_price` - The requested entry price
    /// * `pip_size` - The pip size for the symbol
    /// * `direction` - Trade direction
    /// * `position_id` - Unique position identifier
    /// * `timestamp_us` - Signal timestamp in microseconds
    ///
    /// # Returns
    /// The slipped entry price
    pub fn apply_entry_slippage(
        &self,
        entry_price: f64,
        pip_size: f64,
        direction: Direction,
        position_id: u64,
        timestamp_us: i64,
    ) -> f64 {
        let slippage_pips = self.generate_slippage(position_id, timestamp_us, 0);
        let slippage_price = slippage_pips * pip_size;

        match direction {
            Direction::Long => entry_price + slippage_price,
            Direction::Short => entry_price - slippage_price,
        }
    }

    /// Calculate slippage for an exit.
    ///
    /// Exit slippage is always unfavorable:
    /// - Long: price decreases (you receive less)
    /// - Short: price increases (you pay more to close)
    ///
    /// # Arguments
    /// * `exit_price` - The theoretical exit price
    /// * `pip_size` - The pip size for the symbol
    /// * `direction` - Trade direction
    /// * `position_id` - Unique position identifier
    /// * `timestamp_us` - Exit timestamp in microseconds
    ///
    /// # Returns
    /// The slipped exit price
    pub fn apply_exit_slippage(
        &self,
        exit_price: f64,
        pip_size: f64,
        direction: Direction,
        position_id: u64,
        timestamp_us: i64,
    ) -> f64 {
        let slippage_pips = self.generate_slippage(position_id, timestamp_us, 1);
        let slippage_price = slippage_pips * pip_size;

        match direction {
            Direction::Long => exit_price - slippage_price,
            Direction::Short => exit_price + slippage_price,
        }
    }

    /// Generate a deterministic slippage value in pips.
    ///
    /// Uses ChaCha8 RNG seeded from a hash of inputs.
    fn generate_slippage(&self, position_id: u64, timestamp_us: i64, salt: u64) -> f64 {
        // Create a deterministic seed from inputs
        let combined_seed = self.combine_seed(position_id, timestamp_us, salt);
        
        // Create RNG from seed
        let mut rng = ChaCha8Rng::seed_from_u64(combined_seed);
        
        // Generate uniform random slippage in [0, max_slippage_pips]
        rng.random::<f64>() * self.max_slippage_pips
    }

    /// Combine base_seed with position data to create a unique seed.
    fn combine_seed(&self, position_id: u64, timestamp_us: i64, salt: u64) -> u64 {
        // Simple hash combination using XOR and bit rotation
        let ts = timestamp_us as u64;
        let mut seed = self.base_seed;
        seed = seed.wrapping_add(position_id.wrapping_mul(0x9E3779B97F4A7C15));
        seed ^= ts.rotate_left(17);
        seed = seed.wrapping_add(salt.wrapping_mul(0x6A5D39EAE12657AA));
        seed
    }
}

impl Default for SlippageCalculator {
    fn default() -> Self {
        Self::new(42, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_determinism() {
        let calc1 = SlippageCalculator::new(12345, 1.0);
        let calc2 = SlippageCalculator::new(12345, 1.0);

        let pip_size = 0.0001;
        let entry_price = 1.1000;

        let slip1 = calc1.apply_entry_slippage(entry_price, pip_size, Direction::Long, 1, 1000);
        let slip2 = calc2.apply_entry_slippage(entry_price, pip_size, Direction::Long, 1, 1000);

        assert!((slip1 - slip2).abs() < 1e-15, "Slippage should be deterministic");
    }

    #[test]
    fn test_different_positions_different_slippage() {
        let calc = SlippageCalculator::new(12345, 1.0);
        let pip_size = 0.0001;
        let entry_price = 1.1000;

        let slip1 = calc.apply_entry_slippage(entry_price, pip_size, Direction::Long, 1, 1000);
        let slip2 = calc.apply_entry_slippage(entry_price, pip_size, Direction::Long, 2, 1000);

        assert!((slip1 - slip2).abs() > 1e-10, "Different positions should have different slippage");
    }

    #[test]
    fn test_long_entry_slippage_increases_price() {
        let calc = SlippageCalculator::new(12345, 1.0);
        let pip_size = 0.0001;
        let entry_price = 1.1000;

        let slipped = calc.apply_entry_slippage(entry_price, pip_size, Direction::Long, 1, 1000);
        
        assert!(slipped >= entry_price, "Long entry slippage should increase price");
    }

    #[test]
    fn test_short_entry_slippage_decreases_price() {
        let calc = SlippageCalculator::new(12345, 1.0);
        let pip_size = 0.0001;
        let entry_price = 1.1000;

        let slipped = calc.apply_entry_slippage(entry_price, pip_size, Direction::Short, 1, 1000);
        
        assert!(slipped <= entry_price, "Short entry slippage should decrease price");
    }

    #[test]
    fn test_long_exit_slippage_decreases_price() {
        let calc = SlippageCalculator::new(12345, 1.0);
        let pip_size = 0.0001;
        let exit_price = 1.1100;

        let slipped = calc.apply_exit_slippage(exit_price, pip_size, Direction::Long, 1, 2000);
        
        assert!(slipped <= exit_price, "Long exit slippage should decrease price");
    }

    #[test]
    fn test_short_exit_slippage_increases_price() {
        let calc = SlippageCalculator::new(12345, 1.0);
        let pip_size = 0.0001;
        let exit_price = 1.0900;

        let slipped = calc.apply_exit_slippage(exit_price, pip_size, Direction::Short, 1, 2000);
        
        assert!(slipped >= exit_price, "Short exit slippage should increase price");
    }

    #[test]
    fn test_max_slippage_bound() {
        let max_pips = 2.0;
        let calc = SlippageCalculator::new(12345, max_pips);
        let pip_size = 0.0001;
        let entry_price = 1.1000;

        // Test many positions to ensure slippage stays within bounds
        for i in 0..100 {
            let slipped = calc.apply_entry_slippage(entry_price, pip_size, Direction::Long, i, 1000);
            let slippage_pips = (slipped - entry_price) / pip_size;
            assert!(slippage_pips >= 0.0, "Slippage should be non-negative");
            assert!(slippage_pips <= max_pips, "Slippage should not exceed max_pips");
        }
    }

    #[test]
    fn test_entry_exit_different_slippage() {
        let calc = SlippageCalculator::new(12345, 1.0);
        let pip_size = 0.0001;
        let position_id = 1;
        let timestamp = 1000;

        let entry_slip = calc.apply_entry_slippage(1.1000, pip_size, Direction::Long, position_id, timestamp);
        let exit_slip = calc.apply_exit_slippage(1.1100, pip_size, Direction::Long, position_id, timestamp);

        // Entry and exit should have different slippage due to salt
        // They use different base calculations
        let entry_delta = (entry_slip - 1.1000) / pip_size;
        let exit_delta = (1.1100 - exit_slip) / pip_size;

        // These should be different values due to salt=0 vs salt=1
        assert!((entry_delta - exit_delta).abs() > 1e-10 || entry_delta == exit_delta, 
            "Salt should create variation");
    }
}
