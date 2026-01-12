//! Type Definitions for Pure Rust Strategy Module
//!
//! Enthält alle Datenstrukturen für das RustStrategy-Pattern gemäß
//! docs/ffi/rust_strategy.md Spezifikation.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Zeitrahmen für Candle-Daten
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1,
}

impl Timeframe {
    /// Konvertiert String zu Timeframe
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "M1" => Some(Timeframe::M1),
            "M5" => Some(Timeframe::M5),
            "M15" => Some(Timeframe::M15),
            "M30" => Some(Timeframe::M30),
            "H1" => Some(Timeframe::H1),
            "H4" => Some(Timeframe::H4),
            "D1" => Some(Timeframe::D1),
            _ => None,
        }
    }

    /// Konvertiert Timeframe zu String
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "M1",
            Timeframe::M5 => "M5",
            Timeframe::M15 => "M15",
            Timeframe::M30 => "M30",
            Timeframe::H1 => "H1",
            Timeframe::H4 => "H4",
            Timeframe::D1 => "D1",
        }
    }

    /// Minuten pro Timeframe
    pub fn minutes(&self) -> u32 {
        match self {
            Timeframe::M1 => 1,
            Timeframe::M5 => 5,
            Timeframe::M15 => 15,
            Timeframe::M30 => 30,
            Timeframe::H1 => 60,
            Timeframe::H4 => 240,
            Timeframe::D1 => 1440,
        }
    }
}

#[pymethods]
impl Timeframe {
    fn __str__(&self) -> &'static str {
        self.as_str()
    }

    fn __repr__(&self) -> String {
        format!("Timeframe.{}", self.as_str())
    }
}

/// Einzelne OHLCV Candle
#[pyclass]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CandleData {
    /// Unix timestamp in Mikrosekunden
    #[pyo3(get)]
    pub timestamp_us: i64,
    #[pyo3(get)]
    pub open: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub volume: f64,
}

impl CandleData {
    pub fn new(timestamp_us: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp_us,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Validiert OHLCV-Constraints
    pub fn is_valid(&self) -> bool {
        self.high >= self.low
            && self.high >= self.open
            && self.high >= self.close
            && self.low <= self.open
            && self.low <= self.close
            && self.volume >= 0.0
    }
}

#[pymethods]
impl CandleData {
    #[new]
    fn py_new(
        timestamp_us: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self::new(timestamp_us, open, high, low, close, volume)
    }

    fn __repr__(&self) -> String {
        format!(
            "CandleData(ts={}, o={:.5}, h={:.5}, l={:.5}, c={:.5}, v={:.0})",
            self.timestamp_us, self.open, self.high, self.low, self.close, self.volume
        )
    }
}

/// Trade-Richtung
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
}

#[pymethods]
impl Direction {
    fn __str__(&self) -> &'static str {
        match self {
            Direction::Long => "long",
            Direction::Short => "short",
        }
    }

    fn __repr__(&self) -> String {
        format!("Direction.{:?}", self)
    }
}

/// Trade-Signal aus Strategy.evaluate()
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    #[pyo3(get)]
    pub direction: Direction,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub stop_loss: f64,
    #[pyo3(get)]
    pub take_profit: f64,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub scenario: u8,
    #[pyo3(get)]
    pub reason: Option<String>,
    #[pyo3(get)]
    pub timestamp_us: i64,
}

impl TradeSignal {
    pub fn long(
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        symbol: String,
        scenario: u8,
        timestamp_us: i64,
    ) -> Self {
        Self {
            direction: Direction::Long,
            entry_price,
            stop_loss,
            take_profit,
            symbol,
            scenario,
            reason: None,
            timestamp_us,
        }
    }

    pub fn short(
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        symbol: String,
        scenario: u8,
        timestamp_us: i64,
    ) -> Self {
        Self {
            direction: Direction::Short,
            entry_price,
            stop_loss,
            take_profit,
            symbol,
            scenario,
            reason: None,
            timestamp_us,
        }
    }

    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = Some(reason.to_string());
        self
    }

    /// Validiert Signal-Logik
    pub fn is_valid(&self) -> bool {
        match self.direction {
            Direction::Long => {
                self.stop_loss < self.entry_price && self.take_profit > self.entry_price
            }
            Direction::Short => {
                self.stop_loss > self.entry_price && self.take_profit < self.entry_price
            }
        }
    }

    /// Risk-Reward Ratio
    pub fn risk_reward(&self) -> f64 {
        let risk = (self.entry_price - self.stop_loss).abs();
        let reward = (self.take_profit - self.entry_price).abs();
        if risk > 0.0 {
            reward / risk
        } else {
            0.0
        }
    }
}

#[pymethods]
impl TradeSignal {
    #[new]
    #[pyo3(signature = (direction, entry_price, stop_loss, take_profit, symbol, scenario, timestamp_us, reason=None))]
    fn py_new(
        direction: Direction,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        symbol: String,
        scenario: u8,
        timestamp_us: i64,
        reason: Option<String>,
    ) -> Self {
        Self {
            direction,
            entry_price,
            stop_loss,
            take_profit,
            symbol,
            scenario,
            reason,
            timestamp_us,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TradeSignal({:?}, entry={:.5}, sl={:.5}, tp={:.5}, scenario={})",
            self.direction, self.entry_price, self.stop_loss, self.take_profit, self.scenario
        )
    }
}

/// Offene Position für Position-Management
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    #[pyo3(get)]
    pub id: u64,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub direction: Direction,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub current_stop_loss: f64,
    #[pyo3(get)]
    pub current_take_profit: f64,
    #[pyo3(get)]
    pub size: f64,
    #[pyo3(get)]
    pub entry_timestamp_us: i64,
    #[pyo3(get)]
    pub scenario: u8,
}

impl Position {
    pub fn new(
        id: u64,
        signal: &TradeSignal,
        size: f64,
    ) -> Self {
        Self {
            id,
            symbol: signal.symbol.clone(),
            direction: signal.direction,
            entry_price: signal.entry_price,
            current_stop_loss: signal.stop_loss,
            current_take_profit: signal.take_profit,
            size,
            entry_timestamp_us: signal.timestamp_us,
            scenario: signal.scenario,
        }
    }

    /// Unrealisierter P&L in Kontowährung (für Forex: Pips * PipValue * Lots)
    /// 
    /// Für EURUSD und andere 4-stellige Paare:
    /// - 1 Pip = 0.0001 (für JPY-Paare: 0.01)
    /// - PipValue für 1 Standard-Lot ≈ $10 (für EURUSD)
    /// 
    /// Formel: (Exit - Entry) * 10000 * PipValue * Lots
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let pips = (current_price - self.entry_price) * self.pip_multiplier();
        let pip_value = self.pip_value_per_lot();
        match self.direction {
            Direction::Long => pips * pip_value * self.size,
            Direction::Short => -pips * pip_value * self.size,
        }
    }
    
    /// Pip-Multiplikator basierend auf Symbol (10000 für 4-stellig, 100 für JPY)
    fn pip_multiplier(&self) -> f64 {
        if self.symbol.contains("JPY") {
            100.0
        } else {
            10_000.0
        }
    }
    
    /// Pip-Wert pro Standard-Lot in USD
    fn pip_value_per_lot(&self) -> f64 {
        // Vereinfacht: $10 pro Pip für die meisten Paare
        // In der Realität variiert dies je nach Quotewährung
        10.0
    }

    /// Check ob SL/TP erreicht
    pub fn check_exit(&self, bid: f64, ask: f64) -> Option<PositionAction> {
        match self.direction {
            Direction::Long => {
                if bid <= self.current_stop_loss {
                    Some(PositionAction::Close {
                        reason: "stop_loss".to_string(),
                        exit_price: bid,
                    })
                } else if bid >= self.current_take_profit {
                    Some(PositionAction::Close {
                        reason: "take_profit".to_string(),
                        exit_price: bid,
                    })
                } else {
                    None
                }
            }
            Direction::Short => {
                if ask >= self.current_stop_loss {
                    Some(PositionAction::Close {
                        reason: "stop_loss".to_string(),
                        exit_price: ask,
                    })
                } else if ask <= self.current_take_profit {
                    Some(PositionAction::Close {
                        reason: "take_profit".to_string(),
                        exit_price: ask,
                    })
                } else {
                    None
                }
            }
        }
    }
}

#[pymethods]
impl Position {
    fn __repr__(&self) -> String {
        format!(
            "Position(id={}, {:?}, entry={:.5}, sl={:.5}, tp={:.5})",
            self.id, self.direction, self.entry_price, self.current_stop_loss, self.current_take_profit
        )
    }
}

/// Aktion für Position-Management
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionAction {
    /// Position halten - leere Tuple-Variante für PyO3 Kompatibilität
    Hold(),
    /// Stop-Loss anpassen
    ModifyStopLoss { new_stop_loss: f64 },
    /// Take-Profit anpassen
    ModifyTakeProfit { new_take_profit: f64 },
    /// Position schließen
    Close { reason: String, exit_price: f64 },
}

impl PositionAction {
    pub fn hold() -> Self {
        PositionAction::Hold()
    }

    pub fn modify_sl(new_stop_loss: f64) -> Self {
        PositionAction::ModifyStopLoss { new_stop_loss }
    }

    pub fn modify_tp(new_take_profit: f64) -> Self {
        PositionAction::ModifyTakeProfit { new_take_profit }
    }

    pub fn close(reason: &str, exit_price: f64) -> Self {
        PositionAction::Close {
            reason: reason.to_string(),
            exit_price,
        }
    }
}

/// Snapshot der Marktdaten für evaluate()
#[pyclass]
#[derive(Debug, Clone)]
pub struct DataSlice {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub primary_timeframe: Timeframe,
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub timestamp_us: i64,

    /// Bid-Candles per Timeframe
    pub bid_candles: HashMap<Timeframe, Vec<CandleData>>,
    /// Ask-Candles per Timeframe
    pub ask_candles: HashMap<Timeframe, Vec<CandleData>>,

    /// Gecachte Indikatoren: Key = "ema_20_M5_bid"
    indicators: HashMap<String, Vec<f64>>,
}

impl DataSlice {
    pub fn new(
        symbol: String,
        primary_timeframe: Timeframe,
        index: usize,
        timestamp_us: i64,
    ) -> Self {
        Self {
            symbol,
            primary_timeframe,
            index,
            timestamp_us,
            bid_candles: HashMap::new(),
            ask_candles: HashMap::new(),
            indicators: HashMap::new(),
        }
    }

    /// Setzt Candle-Daten für einen Timeframe
    pub fn set_candles(&mut self, timeframe: Timeframe, bid: Vec<CandleData>, ask: Vec<CandleData>) {
        self.bid_candles.insert(timeframe, bid);
        self.ask_candles.insert(timeframe, ask);
    }

    /// Aktuelle Bid-Candle
    pub fn current_bid(&self) -> Option<&CandleData> {
        self.bid_candles
            .get(&self.primary_timeframe)
            .and_then(|candles| candles.get(self.index))
    }

    /// Aktuelle Ask-Candle
    pub fn current_ask(&self) -> Option<&CandleData> {
        self.ask_candles
            .get(&self.primary_timeframe)
            .and_then(|candles| candles.get(self.index))
    }

    /// Lookback für Candles
    pub fn bid_lookback(&self, timeframe: Timeframe, periods: usize) -> Option<&[CandleData]> {
        self.bid_candles.get(&timeframe).and_then(|candles| {
            if self.index >= periods && candles.len() > self.index {
                Some(&candles[self.index - periods..=self.index])
            } else {
                None
            }
        })
    }

    /// Setzt gecachten Indikator
    pub fn set_indicator(&mut self, key: &str, values: Vec<f64>) {
        self.indicators.insert(key.to_string(), values);
    }

    /// Holt Indikator-Wert am aktuellen Index
    pub fn indicator(&self, key: &str) -> Option<f64> {
        self.indicators
            .get(key)
            .and_then(|values| values.get(self.index).copied())
    }

    /// Holt Indikator-Wert mit Lookback
    pub fn indicator_lookback(&self, key: &str, lookback: usize) -> Option<f64> {
        if self.index >= lookback {
            self.indicators
                .get(key)
                .and_then(|values| values.get(self.index - lookback).copied())
        } else {
            None
        }
    }
}

#[pymethods]
impl DataSlice {
    fn __repr__(&self) -> String {
        format!(
            "DataSlice({}, {}, index={}, ts={})",
            self.symbol,
            self.primary_timeframe.as_str(),
            self.index,
            self.timestamp_us
        )
    }
}

/// Strategy Konfiguration von Python
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub primary_timeframe: String,
    #[pyo3(get)]
    pub initial_capital: f64,
    #[pyo3(get)]
    pub risk_per_trade: f64,
    /// Strategy-spezifische Parameter
    #[pyo3(get)]
    pub params: HashMap<String, f64>,
}

impl StrategyConfig {
    /// Holt Parameter mit Default (public method for Rust access)
    pub fn get_param(&self, key: &str, default: f64) -> f64 {
        self.params.get(key).copied().unwrap_or(default)
    }
}

#[pymethods]
impl StrategyConfig {
    #[new]
    #[pyo3(signature = (symbol, primary_timeframe, initial_capital=10000.0, risk_per_trade=0.02, params=None))]
    fn py_new(
        symbol: String,
        primary_timeframe: String,
        initial_capital: f64,
        risk_per_trade: f64,
        params: Option<HashMap<String, f64>>,
    ) -> Self {
        Self {
            symbol,
            primary_timeframe,
            initial_capital,
            risk_per_trade,
            params: params.unwrap_or_default(),
        }
    }

    /// Python wrapper for get_param
    fn get_param_py(&self, key: &str, default: f64) -> f64 {
        self.get_param(key, default)
    }

    fn __repr__(&self) -> String {
        format!(
            "StrategyConfig({}, {}, capital={:.0}, risk={:.2}%)",
            self.symbol, self.primary_timeframe, self.initial_capital, self.risk_per_trade * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_from_str() {
        assert_eq!(Timeframe::from_str("M5"), Some(Timeframe::M5));
        assert_eq!(Timeframe::from_str("h1"), Some(Timeframe::H1));
        assert_eq!(Timeframe::from_str("INVALID"), None);
    }

    #[test]
    fn test_candle_validity() {
        let valid = CandleData::new(0, 1.1, 1.2, 1.0, 1.15, 1000.0);
        assert!(valid.is_valid());

        let invalid = CandleData::new(0, 1.1, 1.0, 1.2, 1.15, 1000.0); // High < Low
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_trade_signal_validity() {
        let long = TradeSignal::long(1.10, 1.09, 1.12, "EURUSD".into(), 1, 0);
        assert!(long.is_valid());

        let short = TradeSignal::short(1.10, 1.11, 1.08, "EURUSD".into(), 2, 0);
        assert!(short.is_valid());

        let invalid_long = TradeSignal::long(1.10, 1.11, 1.09, "EURUSD".into(), 1, 0); // SL > Entry
        assert!(!invalid_long.is_valid());
    }

    #[test]
    fn test_risk_reward() {
        let signal = TradeSignal::long(1.10, 1.09, 1.13, "EURUSD".into(), 1, 0);
        let rr = signal.risk_reward();
        assert!((rr - 3.0).abs() < 0.001); // Reward=0.03, Risk=0.01 → RR=3
    }

    #[test]
    fn test_position_unrealized_pnl() {
        let signal = TradeSignal::long(1.10, 1.09, 1.12, "EURUSD".into(), 1, 0);
        let pos = Position::new(1, &signal, 100_000.0); // 1 Lot

        let pnl = pos.unrealized_pnl(1.1050);
        assert!((pnl - 500.0).abs() < 0.01); // 50 Pips × 100k = 500

        let signal_short = TradeSignal::short(1.10, 1.11, 1.08, "EURUSD".into(), 1, 0);
        let pos_short = Position::new(2, &signal_short, 100_000.0);

        let pnl_short = pos_short.unrealized_pnl(1.0950);
        assert!((pnl_short - 500.0).abs() < 0.01); // 50 Pips gewinn für Short
    }

    #[test]
    fn test_position_check_exit() {
        let signal = TradeSignal::long(1.10, 1.09, 1.12, "EURUSD".into(), 1, 0);
        let pos = Position::new(1, &signal, 100_000.0);

        // Kein Exit
        assert!(pos.check_exit(1.1050, 1.1051).is_none());

        // SL Hit
        match pos.check_exit(1.0899, 1.0900) {
            Some(PositionAction::Close { reason, .. }) => assert_eq!(reason, "stop_loss"),
            _ => panic!("Expected stop_loss exit"),
        }

        // TP Hit
        match pos.check_exit(1.1201, 1.1202) {
            Some(PositionAction::Close { reason, .. }) => assert_eq!(reason, "take_profit"),
            _ => panic!("Expected take_profit exit"),
        }
    }
}
