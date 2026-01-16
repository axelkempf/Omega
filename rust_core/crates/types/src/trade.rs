use crate::signal::Direction;

/// Reason for trade exit
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    /// Hit take profit
    TakeProfit,
    /// Hit stop loss
    StopLoss,
    /// Timeout / max holding time
    Timeout,
    /// Break-even stop loss triggered
    BreakEvenStopLoss,
    /// Trailing stop loss triggered
    TrailingStopLoss,
    /// Manual exit
    Manual,
}

/// Completed trade
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Trade {
    /// Entry timestamp in nanoseconds
    pub entry_time_ns: i64,
    /// Exit timestamp in nanoseconds
    pub exit_time_ns: i64,
    /// Direction of the trade
    pub direction: Direction,
    /// Trading symbol
    pub symbol: String,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Stop loss price
    pub stop_loss: f64,
    /// Take profit price
    pub take_profit: f64,
    /// Position size
    pub size: f64,
    /// `PnL` in account currency
    pub result: f64,
    /// R-multiple (`PnL` / risk)
    pub r_multiple: f64,
    /// Reason for exit
    pub reason: ExitReason,
    /// Scenario ID for categorization
    pub scenario_id: u8,
    /// Additional metadata as JSON
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_serde_roundtrip() {
        let trade = Trade {
            entry_time_ns: 1234567890000000000,
            exit_time_ns: 1234567900000000000,
            direction: Direction::Long,
            symbol: "EURUSD".to_string(),
            entry_price: 1.1000,
            exit_price: 1.1050,
            stop_loss: 1.0950,
            take_profit: 1.1100,
            size: 0.1,
            result: 50.0,
            r_multiple: 1.0,
            reason: ExitReason::TakeProfit,
            scenario_id: 1,
            meta: serde_json::json!({}),
        };

        let json = serde_json::to_string(&trade).unwrap();
        let deserialized: Trade = serde_json::from_str(&json).unwrap();

        assert_eq!(trade.symbol, deserialized.symbol);
        assert_eq!(trade.reason, deserialized.reason);
        assert_eq!(trade.entry_time_ns, deserialized.entry_time_ns);
    }

    #[test]
    fn test_exit_reason_serialization() {
        assert_eq!(
            serde_json::to_string(&ExitReason::TakeProfit).unwrap(),
            "\"take_profit\""
        );
        assert_eq!(
            serde_json::to_string(&ExitReason::StopLoss).unwrap(),
            "\"stop_loss\""
        );
    }
}
