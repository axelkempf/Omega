//! Fee models for order execution simulation.
//!
//! Fees represent trading costs charged by brokers and exchanges.

/// Trait for fee calculation models.
pub trait FeeModel: Send + Sync {
    /// Calculates the fee for an order.
    ///
    /// # Arguments
    /// * `size` - Position size in lots
    /// * `price` - Fill price
    ///
    /// # Returns
    /// Fee amount in account currency
    fn calculate(&self, size: f64, price: f64) -> f64;

    /// Returns the model name for logging/debugging.
    fn name(&self) -> &'static str;
}

/// Fee wrapper that enforces a minimum fee.
pub struct MinFee {
    /// Inner fee model
    inner: Box<dyn FeeModel>,
    /// Minimum fee to apply
    min_fee: f64,
}

impl MinFee {
    /// Creates a new minimum fee wrapper.
    #[must_use]
    pub fn new(inner: Box<dyn FeeModel>, min_fee: f64) -> Self {
        Self {
            inner,
            min_fee: min_fee.max(0.0),
        }
    }
}

impl FeeModel for MinFee {
    fn calculate(&self, size: f64, price: f64) -> f64 {
        let fee = self.inner.calculate(size, price);
        if self.min_fee > 0.0 {
            fee.max(self.min_fee)
        } else {
            fee
        }
    }

    fn name(&self) -> &'static str {
        "MinFee"
    }
}

/// Percentage-based fee model.
///
/// Calculates fee as a percentage of the notional value.
#[derive(Debug, Clone)]
pub struct PercentageFee {
    /// Fee percentage (e.g., 0.001 for 0.1%)
    pub percent: f64,
}

impl PercentageFee {
    /// Creates a new percentage-based fee model.
    ///
    /// # Arguments
    /// * `percent` - Fee as a decimal (e.g., 0.001 for 0.1%)
    #[must_use]
    pub fn new(percent: f64) -> Self {
        Self { percent }
    }

    /// Creates a percentage fee from basis points.
    ///
    /// # Arguments
    /// * `bps` - Fee in basis points (e.g., 10 bps = 0.1%)
    #[must_use]
    pub fn from_bps(bps: f64) -> Self {
        Self {
            percent: bps / 10_000.0,
        }
    }
}

impl FeeModel for PercentageFee {
    fn calculate(&self, size: f64, price: f64) -> f64 {
        size * price * self.percent
    }

    fn name(&self) -> &'static str {
        "PercentageFee"
    }
}

/// Fixed fee per lot.
///
/// Common for forex and CFD trading where brokers charge
/// a fixed commission per lot traded.
#[derive(Debug, Clone)]
pub struct FixedFee {
    /// Fee per lot in account currency
    pub fee_per_lot: f64,
}

impl FixedFee {
    /// Creates a new fixed fee model.
    #[must_use]
    pub fn new(fee_per_lot: f64) -> Self {
        Self { fee_per_lot }
    }
}

impl FeeModel for FixedFee {
    fn calculate(&self, size: f64, _price: f64) -> f64 {
        size * self.fee_per_lot
    }

    fn name(&self) -> &'static str {
        "FixedFee"
    }
}

/// Fee model based on per-million notional.
///
/// Calculates fee as a rate per 1,000,000 notional, where notional is
/// `size (lots) * lot_size * price`.
#[derive(Debug, Clone)]
pub struct PerMillionNotionalFee {
    /// Fee rate per 1,000,000 notional
    pub rate_per_million: f64,
    /// Lot size (contract size) used to convert lots to notional
    pub lot_size: f64,
}

impl PerMillionNotionalFee {
    /// Creates a new per-million notional fee model.
    #[must_use]
    pub fn new(rate_per_million: f64, lot_size: f64) -> Self {
        Self {
            rate_per_million,
            lot_size: lot_size.max(0.0),
        }
    }
}

impl FeeModel for PerMillionNotionalFee {
    fn calculate(&self, size: f64, price: f64) -> f64 {
        if self.rate_per_million <= 0.0 || self.lot_size <= 0.0 {
            return 0.0;
        }
        let notional = size * self.lot_size * price;
        (notional / 1_000_000.0) * self.rate_per_million
    }

    fn name(&self) -> &'static str {
        "PerMillionNotionalFee"
    }
}

/// Tiered fee model with volume-based discounts.
///
/// Applies different rates based on trading volume tiers.
#[derive(Debug)]
pub struct TieredFee {
    /// Tiers as (`volume_threshold`, `fee_per_lot`)
    /// Must be sorted by volume threshold ascending
    pub tiers: Vec<(f64, f64)>,
    /// Cumulative volume for tier calculation
    cumulative_volume: std::sync::Mutex<f64>,
}

impl TieredFee {
    /// Creates a new tiered fee model.
    ///
    /// # Arguments
    /// * `tiers` - Vector of (`volume_threshold`, `fee_per_lot`) tuples,
    ///   sorted by volume threshold ascending
    #[must_use]
    pub fn new(tiers: Vec<(f64, f64)>) -> Self {
        Self {
            tiers,
            cumulative_volume: std::sync::Mutex::new(0.0),
        }
    }

    /// Resets the cumulative volume (e.g., at month start).
    pub fn reset_volume(&mut self) {
        let mut guard = self
            .cumulative_volume
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = 0.0;
    }

    fn get_current_rate_for(&self, volume: f64) -> f64 {
        for (threshold, rate) in self.tiers.iter().rev() {
            if volume >= *threshold {
                return *rate;
            }
        }
        // Default to highest rate (first tier)
        self.tiers.first().map_or(0.0, |(_, r)| *r)
    }
}

impl Clone for TieredFee {
    fn clone(&self) -> Self {
        let volume = self
            .cumulative_volume
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self {
            tiers: self.tiers.clone(),
            cumulative_volume: std::sync::Mutex::new(*volume),
        }
    }
}

impl FeeModel for TieredFee {
    fn calculate(&self, size: f64, _price: f64) -> f64 {
        let mut guard = self
            .cumulative_volume
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rate = self.get_current_rate_for(*guard);
        *guard += size;
        size * rate
    }

    fn name(&self) -> &'static str {
        "TieredFee"
    }
}

/// Combined fee model that adds multiple fee components.
#[derive(Debug, Clone)]
pub struct CombinedFee {
    /// Percentage component
    pub percentage: Option<PercentageFee>,
    /// Fixed component
    pub fixed: Option<FixedFee>,
}

impl CombinedFee {
    /// Creates a combined fee with both percentage and fixed components.
    #[must_use]
    pub fn new(percentage_fee: Option<PercentageFee>, fixed_fee: Option<FixedFee>) -> Self {
        Self {
            percentage: percentage_fee,
            fixed: fixed_fee,
        }
    }
}

impl FeeModel for CombinedFee {
    fn calculate(&self, size: f64, price: f64) -> f64 {
        let pct = self
            .percentage
            .as_ref()
            .map_or(0.0, |f| f.calculate(size, price));
        let fixed = self
            .fixed
            .as_ref()
            .map_or(0.0, |f| f.calculate(size, price));
        pct + fixed
    }

    fn name(&self) -> &'static str {
        "CombinedFee"
    }
}

/// Zero fee model for testing.
#[derive(Debug, Clone, Default)]
pub struct NoFee;

impl FeeModel for NoFee {
    fn calculate(&self, _size: f64, _price: f64) -> f64 {
        0.0
    }

    fn name(&self) -> &'static str {
        "NoFee"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_percentage_fee() {
        let model = PercentageFee::new(0.001); // 0.1%
        let fee = model.calculate(1.0, 100_000.0);
        assert_relative_eq!(fee, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_percentage_fee_from_bps() {
        let model = PercentageFee::from_bps(10.0); // 10 bps = 0.1%
        let fee = model.calculate(1.0, 100_000.0);
        assert_relative_eq!(fee, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_fee() {
        let model = FixedFee::new(7.0); // $7 per lot
        let fee = model.calculate(2.5, 1.2000); // 2.5 lots
        assert_relative_eq!(fee, 17.5, epsilon = 1e-10);
    }

    #[test]
    fn test_per_million_notional_fee() {
        let model = PerMillionNotionalFee::new(10.0, 100_000.0);
        let fee = model.calculate(1.0, 1.2000); // 120k notional
        assert_relative_eq!(fee, 1.2, epsilon = 1e-10);
    }

    #[test]
    fn test_combined_fee() {
        let model = CombinedFee::new(
            Some(PercentageFee::new(0.0001)), // 1 bps
            Some(FixedFee::new(3.0)),         // $3 per lot
        );

        let fee = model.calculate(1.0, 100_000.0);
        // 10 (percentage) + 3 (fixed) = 13
        assert_relative_eq!(fee, 13.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tiered_fee() {
        let model = TieredFee::new(vec![
            (0.0, 7.0),   // 0-99 lots: $7/lot
            (100.0, 6.0), // 100-499 lots: $6/lot
            (500.0, 5.0), // 500+ lots: $5/lot
        ]);

        // At 0 cumulative volume, should use first tier rate
        let fee = model.calculate(1.0, 1.0);
        assert_relative_eq!(fee, 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tiered_fee_advances_tier() {
        let model = TieredFee::new(vec![(0.0, 7.0), (100.0, 6.0)]);

        let fee1 = model.calculate(50.0, 1.0);
        assert_relative_eq!(fee1, 350.0, epsilon = 1e-10);

        let fee2 = model.calculate(60.0, 1.0);
        assert_relative_eq!(fee2, 420.0, epsilon = 1e-10);

        let fee3 = model.calculate(10.0, 1.0);
        assert_relative_eq!(fee3, 60.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_fee() {
        let model = NoFee;
        assert_relative_eq!(model.calculate(100.0, 100_000.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_min_fee_applies_floor() {
        let model = MinFee::new(Box::new(FixedFee::new(1.0)), 5.0);

        let fee = model.calculate(1.0, 100.0);
        assert_relative_eq!(fee, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_min_fee_allows_higher_fee() {
        let model = MinFee::new(Box::new(PercentageFee::new(0.01)), 5.0);

        let fee = model.calculate(1.0, 1_000.0);
        assert_relative_eq!(fee, 10.0, epsilon = 1e-10);
    }
}
