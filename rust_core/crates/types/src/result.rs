use crate::trade::Trade;

/// Backtest result container
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BacktestResult {
    /// Success flag
    pub ok: bool,
    /// Error information if not ok
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorResult>,
    /// List of completed trades
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trades: Option<Vec<Trade>>,
    /// Performance metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    /// Equity curve data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equity_curve: Option<Vec<EquityPoint>>,
    /// Result metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ResultMeta>,
}

/// Error result information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorResult {
    /// Error category
    pub category: String,
    /// Error message
    pub message: String,
    /// Additional error details
    #[serde(default)]
    pub details: serde_json::Value,
}

/// Point in equity curve
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EquityPoint {
    /// Timestamp in nanoseconds
    pub timestamp_ns: i64,
    /// Current equity
    pub equity: f64,
    /// Current balance
    pub balance: f64,
    /// Drawdown percentage
    pub drawdown: f64,
    /// High water mark
    pub high_water: f64,
}

/// Performance metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    /// Total number of trades
    pub total_trades: u64,
    /// Number of winning trades
    pub wins: u64,
    /// Number of losing trades
    pub losses: u64,
    /// Win rate (wins / `total_trades`)
    pub win_rate: f64,
    /// Gross profit (before costs)
    pub profit_gross: f64,
    /// Net profit (after costs)
    pub profit_net: f64,
    /// Total fees paid
    pub fees_total: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Maximum drawdown absolute value
    pub max_drawdown_abs: f64,
    /// Average R-multiple
    pub avg_r_multiple: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average win amount
    #[serde(default)]
    pub avg_win: f64,
    /// Average loss amount
    #[serde(default)]
    pub avg_loss: f64,
    /// Largest win
    #[serde(default)]
    pub largest_win: f64,
    /// Largest loss
    #[serde(default)]
    pub largest_loss: f64,
    /// Sharpe ratio
    #[serde(default)]
    pub sharpe_ratio: f64,
    /// Sortino ratio
    #[serde(default)]
    pub sortino_ratio: f64,
    /// Calmar ratio
    #[serde(default)]
    pub calmar_ratio: f64,
}

/// Result metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResultMeta {
    /// Backtest runtime in seconds
    #[serde(default)]
    pub runtime_seconds: f64,
    /// Number of candles processed
    #[serde(default)]
    pub candles_processed: u64,
    /// Start timestamp
    #[serde(default)]
    pub start_timestamp: Option<i64>,
    /// End timestamp
    #[serde(default)]
    pub end_timestamp: Option<i64>,
    /// Additional metadata
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Direction;
    use crate::trade::ExitReason;

    #[test]
    fn test_backtest_result_success_serde() {
        let result = BacktestResult {
            ok: true,
            error: None,
            trades: Some(vec![Trade {
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
            }]),
            metrics: Some(Metrics {
                total_trades: 1,
                wins: 1,
                losses: 0,
                win_rate: 1.0,
                profit_gross: 50.0,
                profit_net: 45.0,
                fees_total: 5.0,
                max_drawdown: 0.0,
                max_drawdown_abs: 0.0,
                avg_r_multiple: 1.0,
                profit_factor: 0.0,
                avg_win: 50.0,
                avg_loss: 0.0,
                largest_win: 50.0,
                largest_loss: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                calmar_ratio: 0.0,
            }),
            equity_curve: None,
            meta: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BacktestResult = serde_json::from_str(&json).unwrap();

        assert!(deserialized.ok);
        assert!(deserialized.error.is_none());
        assert!(deserialized.trades.is_some());
    }

    #[test]
    fn test_backtest_result_error_serde() {
        let result = BacktestResult {
            ok: false,
            error: Some(ErrorResult {
                category: "config".to_string(),
                message: "Invalid parameter".to_string(),
                details: serde_json::json!({"field": "symbol"}),
            }),
            trades: None,
            metrics: None,
            equity_curve: None,
            meta: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BacktestResult = serde_json::from_str(&json).unwrap();

        assert!(!deserialized.ok);
        assert!(deserialized.error.is_some());
        assert_eq!(deserialized.error.unwrap().category, "config");
    }

    #[test]
    fn test_equity_point_serde() {
        let point = EquityPoint {
            timestamp_ns: 1234567890000000000,
            equity: 10500.0,
            balance: 10000.0,
            drawdown: 0.05,
            high_water: 11000.0,
        };

        let json = serde_json::to_string(&point).unwrap();
        let deserialized: EquityPoint = serde_json::from_str(&json).unwrap();

        assert_eq!(point.timestamp_ns, deserialized.timestamp_ns);
        assert_eq!(point.equity, deserialized.equity);
    }
}
