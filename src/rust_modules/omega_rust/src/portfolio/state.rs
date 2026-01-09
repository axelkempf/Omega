//! Internal state structures for Portfolio.
//!
//! These structures track portfolio state, equity curve, and fee logs.

/// Internal state tracking for portfolio
#[derive(Clone, Debug, Default)]
pub struct PortfolioState {
    pub initial_balance: f64,
    pub cash: f64,
    pub equity: f64,
    pub max_equity: f64,
    pub max_drawdown: f64,
    pub initial_max_drawdown: f64,
    pub total_fees: f64,
    pub start_timestamp: Option<i64>,
}

impl PortfolioState {
    /// Create a new portfolio state with the given initial balance.
    #[must_use]
    pub const fn new(initial_balance: f64) -> Self {
        Self {
            initial_balance,
            cash: initial_balance,
            equity: initial_balance,
            max_equity: initial_balance,
            max_drawdown: 0.0,
            initial_max_drawdown: 0.0,
            total_fees: 0.0,
            start_timestamp: None,
        }
    }
}

/// Entry for equity curve tracking
#[derive(Clone, Debug)]
pub struct EquityPoint {
    /// Unix timestamp in microseconds
    pub timestamp: i64,
    /// Equity value at this point
    pub equity: f64,
}

/// Entry for fee logging
#[derive(Clone, Debug)]
pub struct FeeLogEntry {
    /// Unix timestamp in microseconds
    pub time: i64,
    /// Fee type (e.g., "entry", "exit", "swap")
    pub kind: String,
    /// Symbol associated with the fee (optional)
    pub symbol: Option<String>,
    /// Position size associated with the fee (optional)
    pub size: Option<f64>,
    /// Fee amount
    pub fee: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_portfolio_state_new() {
        let state = PortfolioState::new(100_000.0);
        assert_relative_eq!(state.initial_balance, 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(state.cash, 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(state.equity, 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(state.max_equity, 100_000.0, epsilon = 1e-8);
        assert_relative_eq!(state.max_drawdown, 0.0, epsilon = 1e-8);
        assert_relative_eq!(state.total_fees, 0.0, epsilon = 1e-8);
        assert!(state.start_timestamp.is_none());
    }

    #[test]
    fn test_equity_point_creation() {
        let point = EquityPoint {
            timestamp: 1_704_067_200_000_000, // 2024-01-01 00:00:00 UTC in microseconds
            equity: 105_000.0,
        };
        assert_eq!(point.timestamp, 1_704_067_200_000_000);
        assert_relative_eq!(point.equity, 105_000.0, epsilon = 1e-8);
    }

    #[test]
    fn test_fee_log_entry_creation() {
        let entry = FeeLogEntry {
            time: 1_704_067_200_000_000,
            kind: "entry".to_string(),
            symbol: Some("EURUSD".to_string()),
            size: Some(1.0),
            fee: 3.0,
        };
        assert_eq!(entry.kind, "entry");
        assert_eq!(entry.symbol, Some("EURUSD".to_string()));
        assert_relative_eq!(entry.fee, 3.0, epsilon = 1e-8);
    }
}
