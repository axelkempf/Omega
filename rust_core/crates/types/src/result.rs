use std::collections::BTreeMap;

use serde_json::Value;

use crate::trade::Trade;

/// Backtest result container.
/// Uses `BTreeMap` for deterministic (sorted) key order in JSON output.
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
    /// Metric definitions (output contract, sorted for determinism)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_definitions: Option<BTreeMap<String, MetricDefinition>>,
    /// Equity curve data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equity_curve: Option<Vec<EquityPoint>>,
    /// Profiling information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<Value>,
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

/// Metric value supporting numeric values or "n/a" strings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MetricValue {
    /// Numeric metric value.
    Number(f64),
    /// String metric value (e.g. "n/a").
    Text(String),
}

impl Default for MetricValue {
    fn default() -> Self {
        Self::Number(0.0)
    }
}

/// Performance metrics
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    /// Maximum drawdown duration in bars
    #[serde(default)]
    pub max_drawdown_duration_bars: u64,
    /// Average R-multiple
    pub avg_r_multiple: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average trade `PnL` (`profit_net` / `total_trades`)
    #[serde(default)]
    pub avg_trade_pnl: f64,
    /// Expectancy in R-multiples
    #[serde(default)]
    pub expectancy: f64,
    /// Count of unique active trading days
    #[serde(default)]
    pub active_days: u64,
    /// Trades per active day
    #[serde(default)]
    pub trades_per_day: f64,
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
    /// Trade-based Sharpe ratio (R-multiples, no annualization)
    #[serde(default)]
    pub sharpe_trade_r: MetricValue,
    /// Trade-based Sortino ratio (R-multiples, no annualization)
    #[serde(default)]
    pub sortino_trade_r: MetricValue,
    /// Equity-based Sharpe ratio (daily returns, annualized with sqrt(252))
    #[serde(default)]
    pub sharpe_equity_daily: MetricValue,
    /// Equity-based Sortino ratio (daily returns, annualized with sqrt(252))
    #[serde(default)]
    pub sortino_equity_daily: MetricValue,
    /// Calmar ratio (annualized return / max drawdown)
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

/// Metric definition metadata for output consumers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricDefinition {
    /// Unit of the metric (e.g. ratio, `account_currency`)
    pub unit: String,
    /// Human-readable description
    pub description: String,
    /// Allowed domain of values
    pub domain: String,
    /// Source of the metric (trades, equity, etc.)
    pub source: String,
    /// Value type for serialization
    #[serde(rename = "type")]
    pub value_type: String,
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
                entry_time_ns: 1_234_567_890_000_000_000,
                exit_time_ns: 1_234_567_900_000_000_000,
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
                ..Metrics::default()
            }),
            metric_definitions: None,
            equity_curve: None,
            profiling: None,
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
            metric_definitions: None,
            equity_curve: None,
            profiling: None,
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
            timestamp_ns: 1_234_567_890_000_000_000,
            equity: 10500.0,
            balance: 10000.0,
            drawdown: 0.05,
            high_water: 11000.0,
        };

        let json = serde_json::to_string(&point).unwrap();
        let deserialized: EquityPoint = serde_json::from_str(&json).unwrap();

        assert_eq!(point.timestamp_ns, deserialized.timestamp_ns);
        assert!((point.equity - deserialized.equity).abs() < f64::EPSILON);
    }
}
