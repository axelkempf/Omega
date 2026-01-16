//! Trade context types for rule evaluation
//!
//! Provides read-only views and context for trade management rules
//! without cross-cutting dependencies on portfolio/execution crates.

use omega_types::Direction;
use serde::{Deserialize, Serialize};

/// Read-only snapshot of a position for trade management.
///
/// Contains only the information needed for rule evaluation,
/// avoiding dependencies on the full Position type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionView {
    /// Unique position ID
    pub position_id: u64,
    /// Trading symbol
    pub symbol: String,
    /// Trade direction
    pub direction: Direction,
    /// Entry timestamp in nanoseconds
    pub entry_time_ns: i64,
    /// Entry price
    pub entry_price: f64,
    /// Position size
    pub size: f64,
    /// Current stop loss level
    pub stop_loss: Option<f64>,
    /// Current take profit level
    pub take_profit: Option<f64>,
    /// Scenario ID (1-6 for MRZ)
    pub scenario_id: u8,
}

impl PositionView {
    /// Creates a new PositionView.
    pub fn new(
        position_id: u64,
        symbol: impl Into<String>,
        direction: Direction,
        entry_time_ns: i64,
        entry_price: f64,
        size: f64,
    ) -> Self {
        Self {
            position_id,
            symbol: symbol.into(),
            direction,
            entry_time_ns,
            entry_price,
            size,
            stop_loss: None,
            take_profit: None,
            scenario_id: 1,
        }
    }

    /// Sets the stop loss level.
    pub fn with_stop_loss(mut self, sl: f64) -> Self {
        self.stop_loss = Some(sl);
        self
    }

    /// Sets the take profit level.
    pub fn with_take_profit(mut self, tp: f64) -> Self {
        self.take_profit = Some(tp);
        self
    }

    /// Sets the scenario ID.
    pub fn with_scenario(mut self, scenario_id: u8) -> Self {
        self.scenario_id = scenario_id;
        self
    }
}

/// Market snapshot for trade management decisions.
///
/// Contains bid/ask OHLC data from the current bar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarketView {
    /// Current bar timestamp in nanoseconds (bar open time)
    pub timestamp_ns: i64,
    /// Bid candle open
    pub bid_open: f64,
    /// Bid candle high
    pub bid_high: f64,
    /// Bid candle low
    pub bid_low: f64,
    /// Bid candle close
    pub bid_close: f64,
    /// Ask candle open
    pub ask_open: f64,
    /// Ask candle high
    pub ask_high: f64,
    /// Ask candle low
    pub ask_low: f64,
    /// Ask candle close
    pub ask_close: f64,
}

impl MarketView {
    /// Creates a MarketView from bid/ask close prices (simplified).
    pub fn from_close(timestamp_ns: i64, bid_close: f64, ask_close: f64) -> Self {
        Self {
            timestamp_ns,
            bid_open: bid_close,
            bid_high: bid_close,
            bid_low: bid_close,
            bid_close,
            ask_open: ask_close,
            ask_high: ask_close,
            ask_low: ask_close,
            ask_close,
        }
    }

    /// Returns exit price for the given direction.
    ///
    /// For long positions: bid_close (selling at bid)
    /// For short positions: ask_close (buying at ask)
    pub fn exit_price(&self, direction: Direction) -> f64 {
        match direction {
            Direction::Long => self.bid_close,
            Direction::Short => self.ask_close,
        }
    }

    /// Returns the current price for the given direction.
    ///
    /// For long positions: bid_close
    /// For short positions: ask_close
    pub fn current_price(&self, direction: Direction) -> f64 {
        self.exit_price(direction)
    }
}

/// Trade management evaluation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeMgmtMode {
    /// Candle-based evaluation (default for backtesting)
    #[default]
    Candle,
    /// Tick-based evaluation (for live trading)
    Tick,
}

/// Context for trade management rule evaluation.
///
/// Contains all information needed by rules to make decisions
/// without cross-cutting dependencies.
#[derive(Debug, Clone)]
pub struct TradeContext {
    /// Current bar index
    pub idx: usize,
    /// Market snapshot
    pub market: MarketView,
    /// Whether trading session is open
    pub session_open: bool,
    /// Whether news filter is blocking
    pub news_blocked: bool,
    /// Evaluation mode
    pub mode: TradeMgmtMode,
    /// Bar duration in nanoseconds (for time-based calculations)
    pub bar_duration_ns: i64,
}

impl TradeContext {
    /// Creates a new TradeContext.
    pub fn new(idx: usize, market: MarketView, bar_duration_ns: i64) -> Self {
        Self {
            idx,
            market,
            session_open: true,
            news_blocked: false,
            mode: TradeMgmtMode::default(),
            bar_duration_ns,
        }
    }

    /// Sets session open status.
    pub fn with_session(mut self, open: bool) -> Self {
        self.session_open = open;
        self
    }

    /// Sets news blocked status.
    pub fn with_news_blocked(mut self, blocked: bool) -> Self {
        self.news_blocked = blocked;
        self
    }

    /// Sets evaluation mode.
    pub fn with_mode(mut self, mode: TradeMgmtMode) -> Self {
        self.mode = mode;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_view_builder() {
        let view = PositionView::new(
            42,
            "EURUSD",
            Direction::Long,
            1000000000,
            1.1000,
            0.1,
        )
        .with_stop_loss(1.0950)
        .with_take_profit(1.1100)
        .with_scenario(3);

        assert_eq!(view.position_id, 42);
        assert_eq!(view.symbol, "EURUSD");
        assert_eq!(view.stop_loss, Some(1.0950));
        assert_eq!(view.take_profit, Some(1.1100));
        assert_eq!(view.scenario_id, 3);
    }

    #[test]
    fn test_market_view_exit_price() {
        let market = MarketView::from_close(0, 1.1000, 1.1002);

        assert_eq!(market.exit_price(Direction::Long), 1.1000);
        assert_eq!(market.exit_price(Direction::Short), 1.1002);
    }

    #[test]
    fn test_trade_context_builder() {
        let market = MarketView::from_close(1000000000, 1.1000, 1.1002);
        let ctx = TradeContext::new(10, market, 60_000_000_000)
            .with_session(false)
            .with_news_blocked(true);

        assert_eq!(ctx.idx, 10);
        assert!(!ctx.session_open);
        assert!(ctx.news_blocked);
    }
}
