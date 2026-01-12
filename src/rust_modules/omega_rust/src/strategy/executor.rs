//! Pure Rust Backtest Executor
//!
//! Führt die komplette Backtest-Schleife in Rust aus, ohne FFI-Overhead.
//! Ziel: ≥10x Performance-Verbesserung gegenüber Python-Callback-Ansatz.

use std::collections::HashMap;
use std::time::Instant;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::indicators::{IndicatorCache, PyIndicatorCache};
use super::traits::{RustStrategy, StrategyError};
use super::types::{CandleData, DataSlice, Direction, Position, PositionAction, StrategyConfig, Timeframe, TradeSignal};
use super::registry::create_strategy;

/// Ergebnis eines einzelnen Trades
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeResult {
    /// Trade-ID
    #[pyo3(get)]
    pub id: u64,
    
    /// Handelssymbol
    #[pyo3(get)]
    pub symbol: String,
    
    /// Richtung: "long" oder "short"
    #[pyo3(get)]
    pub direction: String,
    
    /// Entry-Preis
    #[pyo3(get)]
    pub entry_price: f64,
    
    /// Exit-Preis
    #[pyo3(get)]
    pub exit_price: f64,
    
    /// Entry-Zeitstempel (Microseconds since epoch)
    #[pyo3(get)]
    pub entry_timestamp_us: i64,
    
    /// Exit-Zeitstempel (Microseconds since epoch)
    #[pyo3(get)]
    pub exit_timestamp_us: i64,
    
    /// Positionsgröße
    #[pyo3(get)]
    pub size: f64,
    
    /// Profit/Loss in Währung
    #[pyo3(get)]
    pub pnl: f64,
    
    /// Profit/Loss in Pips
    #[pyo3(get)]
    pub pnl_pips: f64,
    
    /// Exit-Grund: "stop_loss", "take_profit", "manual", "timeout"
    #[pyo3(get)]
    pub exit_reason: String,
    
    /// Szenario (1-6)
    #[pyo3(get)]
    pub scenario: u8,
}

#[pymethods]
impl TradeResult {
    /// Erstellt ein TradeResult aus den gegebenen Parametern
    #[new]
    pub fn new(
        id: u64,
        symbol: String,
        direction: String,
        entry_price: f64,
        exit_price: f64,
        entry_timestamp_us: i64,
        exit_timestamp_us: i64,
        size: f64,
        pnl: f64,
        pnl_pips: f64,
        exit_reason: String,
        scenario: u8,
    ) -> Self {
        Self {
            id,
            symbol,
            direction,
            entry_price,
            exit_price,
            entry_timestamp_us,
            exit_timestamp_us,
            size,
            pnl,
            pnl_pips,
            exit_reason,
            scenario,
        }
    }
    
    /// Gewinn-/Verlust-Faktor
    #[getter]
    pub fn r_multiple(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        self.pnl / (self.entry_price * self.size).abs() * 100.0
    }
    
    /// Trade-Dauer in Sekunden
    #[getter]
    pub fn duration_seconds(&self) -> f64 {
        (self.exit_timestamp_us - self.entry_timestamp_us) as f64 / 1_000_000.0
    }
    
    /// Gewinn oder Verlust?
    #[getter]
    pub fn is_winner(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Gesamtergebnis eines Backtests
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Strategie-Name
    #[pyo3(get)]
    pub strategy_name: String,
    
    /// Gehandeltes Symbol
    #[pyo3(get)]
    pub symbol: String,
    
    /// Alle Trades
    #[pyo3(get)]
    pub trades: Vec<TradeResult>,
    
    /// Start-Kapital
    #[pyo3(get)]
    pub initial_capital: f64,
    
    /// End-Kapital
    #[pyo3(get)]
    pub final_capital: f64,
    
    /// Anzahl Bars verarbeitet
    #[pyo3(get)]
    pub bars_processed: u64,
    
    /// Laufzeit in Millisekunden
    #[pyo3(get)]
    pub execution_time_ms: f64,
    
    /// Laufzeit der reinen Strategie-Evaluation (ms)
    #[pyo3(get)]
    pub strategy_time_ms: f64,
    
    /// Anzahl offene Positionen am Ende
    #[pyo3(get)]
    pub open_positions: usize,
}

#[pymethods]
impl BacktestResult {
    /// Gesamtzahl der Trades
    #[getter]
    pub fn total_trades(&self) -> usize {
        self.trades.len()
    }
    
    /// Anzahl Gewinntrades
    #[getter]
    pub fn winning_trades(&self) -> usize {
        self.trades.iter().filter(|t| t.is_winner()).count()
    }
    
    /// Anzahl Verlusttrades
    #[getter]
    pub fn losing_trades(&self) -> usize {
        self.trades.iter().filter(|t| !t.is_winner()).count()
    }
    
    /// Gewinnrate (0-1)
    #[getter]
    pub fn win_rate(&self) -> f64 {
        let total = self.total_trades();
        if total == 0 {
            return 0.0;
        }
        self.winning_trades() as f64 / total as f64
    }
    
    /// Gesamtgewinn/-verlust
    #[getter]
    pub fn total_pnl(&self) -> f64 {
        self.trades.iter().map(|t| t.pnl).sum()
    }
    
    /// Durchschnittlicher Trade
    #[getter]
    pub fn avg_trade(&self) -> f64 {
        let total = self.total_trades();
        if total == 0 {
            return 0.0;
        }
        self.total_pnl() / total as f64
    }
    
    /// Profit Faktor (Gewinne / Verluste)
    #[getter]
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = self.trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        
        if gross_loss == 0.0 {
            return if gross_profit > 0.0 { f64::INFINITY } else { 0.0 };
        }
        gross_profit / gross_loss
    }
    
    /// Return on Investment
    #[getter]
    pub fn roi(&self) -> f64 {
        if self.initial_capital == 0.0 {
            return 0.0;
        }
        (self.final_capital - self.initial_capital) / self.initial_capital * 100.0
    }
    
    /// Durchsatz in Bars pro Sekunde
    #[getter]
    pub fn bars_per_second(&self) -> f64 {
        if self.execution_time_ms == 0.0 {
            return 0.0;
        }
        self.bars_processed as f64 / (self.execution_time_ms / 1000.0)
    }
}

/// Pure Rust Backtest Executor
pub struct RustExecutor {
    strategy: Box<dyn RustStrategy>,
    config: StrategyConfig,
    positions: Vec<Position>,
    trades: Vec<TradeResult>,
    capital: f64,
    next_trade_id: u64,
}

impl RustExecutor {
    /// Erstellt einen neuen Executor mit der gegebenen Strategie.
    pub fn new(strategy: Box<dyn RustStrategy>, config: StrategyConfig) -> Self {
        let capital = config.initial_capital;
        Self {
            strategy,
            config,
            positions: Vec::new(),
            trades: Vec::new(),
            capital,
            next_trade_id: 1,
        }
    }

    /// Führt den kompletten Backtest aus.
    ///
    /// # Arguments
    ///
    /// * `bid_candles` - Bid-Candle-Daten nach Timeframe
    /// * `ask_candles` - Ask-Candle-Daten nach Timeframe
    /// * `indicator_cache` - Pre-computed Indicators
    ///
    /// # Returns
    ///
    /// Vollständiges BacktestResult mit allen Trades und Metriken.
    pub fn run(
        &mut self,
        bid_candles: HashMap<Timeframe, Vec<CandleData>>,
        ask_candles: HashMap<Timeframe, Vec<CandleData>>,
        indicator_cache: &mut IndicatorCache,
    ) -> Result<BacktestResult, StrategyError> {
        let start_time = Instant::now();
        let mut strategy_time_ns: u64 = 0;
        
        // Primary timeframe data
        let primary_tf = self.strategy.primary_timeframe();
        let bid_data = bid_candles.get(&primary_tf).ok_or_else(|| {
            StrategyError::ConfigError(format!("No bid data for timeframe {:?}", primary_tf))
        })?;
        let ask_data = ask_candles.get(&primary_tf).ok_or_else(|| {
            StrategyError::ConfigError(format!("No ask data for timeframe {:?}", primary_tf))
        })?;
        
        if bid_data.len() != ask_data.len() {
            return Err(StrategyError::ConfigError(
                "Bid and ask data length mismatch".to_string(),
            ));
        }
        
        let warmup = self.strategy.warmup_bars();
        let total_bars = bid_data.len();
        
        if total_bars < warmup {
            return Err(StrategyError::InsufficientData {
                required: warmup,
                available: total_bars,
            });
        }
        
        // Main loop: iterate over bars
        for i in warmup..total_bars {
            let bid = &bid_data[i];
            let ask = &ask_data[i];
            
            // Build DataSlice for current bar using constructor
            let mut slice = DataSlice::new(
                self.config.symbol.clone(),
                primary_tf.clone(),
                i,
                bid.timestamp_us,
            );
            
            // Set candle data for primary timeframe (entire vectors, index determines current)
            slice.set_candles(primary_tf.clone(), bid_data.clone(), ask_data.clone());
            
            // 1. Exit-Management for existing positions
            self.process_exits(&slice, bid, ask);
            
            // 2. Position management (trailing stop, break-even, etc.)
            self.manage_positions(&slice);
            
            // 3. Strategy evaluation
            if self.positions.len() < self.strategy.max_positions() {
                let eval_start = Instant::now();
                if let Some(signal) = self.strategy.evaluate(&slice, indicator_cache) {
                    self.process_signal(signal, bid, ask);
                }
                strategy_time_ns += eval_start.elapsed().as_nanos() as u64;
            }
        }
        
        let execution_time = start_time.elapsed();
        
        Ok(BacktestResult {
            strategy_name: self.strategy.name().to_string(),
            symbol: self.config.symbol.clone(),
            trades: self.trades.clone(),
            initial_capital: self.config.initial_capital,
            final_capital: self.capital,
            bars_processed: (total_bars - warmup) as u64,
            execution_time_ms: execution_time.as_secs_f64() * 1000.0,
            strategy_time_ms: strategy_time_ns as f64 / 1_000_000.0,
            open_positions: self.positions.len(),
        })
    }
    
    /// Verarbeitet Exit-Bedingungen (SL/TP Hit)
    fn process_exits(&mut self, slice: &DataSlice, bid: &CandleData, ask: &CandleData) {
        let mut closed_indices = Vec::new();
        
        for (idx, position) in self.positions.iter().enumerate() {
            // check_exit expects f64 prices, not CandleData
            if let Some(action) = position.check_exit(bid.close, ask.close) {
                if let PositionAction::Close { reason, exit_price } = action {
                    // Calculate PnL
                    let pnl = position.unrealized_pnl(exit_price);
                    let pnl_pips = (exit_price - position.entry_price).abs() * 10_000.0;
                    
                    // Record trade
                    let trade = TradeResult {
                        id: position.id,
                        symbol: position.symbol.clone(),
                        direction: match position.direction {
                            Direction::Long => "long".to_string(),
                            Direction::Short => "short".to_string(),
                        },
                        entry_price: position.entry_price,
                        exit_price,
                        entry_timestamp_us: position.entry_timestamp_us,
                        exit_timestamp_us: slice.timestamp_us,
                        size: position.size,
                        pnl,
                        pnl_pips,
                        exit_reason: reason,
                        scenario: position.scenario,
                    };
                    
                    self.capital += pnl;
                    self.trades.push(trade);
                    closed_indices.push(idx);
                }
            }
        }
        
        // Remove closed positions (reverse order to maintain indices)
        for idx in closed_indices.into_iter().rev() {
            self.positions.remove(idx);
        }
    }
    
    /// Verarbeitet Position-Management (Trailing Stop, etc.)
    fn manage_positions(&mut self, slice: &DataSlice) {
        for position in &mut self.positions {
            match self.strategy.manage_position(position, slice) {
                PositionAction::Hold() => {}
                PositionAction::ModifyStopLoss { new_stop_loss } => {
                    position.current_stop_loss = new_stop_loss;
                }
                PositionAction::ModifyTakeProfit { new_take_profit } => {
                    position.current_take_profit = new_take_profit;
                }
                PositionAction::Close { reason: _, exit_price } => {
                    // Mark for closing at specified price
                    // (handled in next process_exits call)
                    position.current_stop_loss = exit_price;
                }
            }
        }
    }
    
    /// Verarbeitet ein neues Trade-Signal
    fn process_signal(&mut self, signal: TradeSignal, _bid: &CandleData, _ask: &CandleData) {
        // Validate signal
        if !signal.is_valid() {
            return;
        }
        
        // Check direction filter
        let direction_str = match signal.direction {
            Direction::Long => "long",
            Direction::Short => "short",
        };
        if !self.strategy.is_direction_allowed(direction_str) {
            return;
        }
        
        // Check scenario filter
        if !self.strategy.is_scenario_enabled(signal.scenario) {
            return;
        }
        
        // Calculate position size based on risk
        let risk_amount = self.capital * self.config.risk_per_trade;
        let stop_distance = (signal.entry_price - signal.stop_loss).abs();
        let pip_value = 10.0; // Simplified: 10 USD per pip for standard lot
        let size = if stop_distance > 0.0 {
            risk_amount / (stop_distance * 10_000.0 * pip_value)
        } else {
            0.01 // Minimum size
        };
        
        // Create position
        let position = Position {
            id: self.next_trade_id,
            symbol: signal.symbol.clone(),
            direction: signal.direction,
            entry_price: signal.entry_price,
            current_stop_loss: signal.stop_loss,
            current_take_profit: signal.take_profit,
            size,
            entry_timestamp_us: signal.timestamp_us,
            scenario: signal.scenario,
        };
        
        self.positions.push(position);
        self.next_trade_id += 1;
    }
}

// ============================================================================
// PyO3 Bindings
// ============================================================================

/// Führt einen Pure-Rust-Backtest aus.
///
/// # Arguments
///
/// * `strategy_name` - Name der registrierten Strategie
/// * `config` - Strategie-Konfiguration
/// * `bid_candles` - Dict[Timeframe, List[CandleData]]
/// * `ask_candles` - Dict[Timeframe, List[CandleData]]
/// * `indicator_cache` - IndicatorCacheRust Instanz
///
/// # Returns
///
/// BacktestResult mit allen Trades und Metriken.
#[pyfunction]
pub fn run_backtest_rust(
    _py: Python<'_>,
    strategy_name: &str,
    config: StrategyConfig,
    bid_candles_py: HashMap<String, Vec<CandleData>>,
    ask_candles_py: HashMap<String, Vec<CandleData>>,
    indicator_cache: &PyIndicatorCache,
) -> PyResult<BacktestResult> {
    // Convert timeframe strings to enum
    let bid_candles: HashMap<Timeframe, Vec<CandleData>> = bid_candles_py
        .into_iter()
        .filter_map(|(tf_str, candles)| {
            Timeframe::from_str(&tf_str).map(|tf| (tf, candles))
        })
        .collect();
    
    let ask_candles: HashMap<Timeframe, Vec<CandleData>> = ask_candles_py
        .into_iter()
        .filter_map(|(tf_str, candles)| {
            Timeframe::from_str(&tf_str).map(|tf| (tf, candles))
        })
        .collect();
    
    // Create strategy from registry
    let strategy = create_strategy(strategy_name, &config)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    // Lock the indicator cache for mutable access
    let mut cache_guard = indicator_cache.inner.lock()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
    
    // Run backtest
    let mut executor = RustExecutor::new(strategy, config);
    executor
        .run(bid_candles, ask_candles, &mut cache_guard)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_result_metrics() {
        let trade = TradeResult {
            id: 1,
            symbol: "EURUSD".to_string(),
            direction: "long".to_string(),
            entry_price: 1.1000,
            exit_price: 1.1100,
            entry_timestamp_us: 1000000,
            exit_timestamp_us: 2000000,
            size: 0.1,
            pnl: 100.0,
            pnl_pips: 100.0,
            exit_reason: "take_profit".to_string(),
            scenario: 1,
        };
        
        assert!(trade.is_winner());
        assert_eq!(trade.duration_seconds(), 1.0);
    }

    #[test]
    fn test_backtest_result_metrics() {
        let trades = vec![
            TradeResult {
                id: 1,
                symbol: "EURUSD".to_string(),
                direction: "long".to_string(),
                entry_price: 1.1000,
                exit_price: 1.1100,
                entry_timestamp_us: 1000000,
                exit_timestamp_us: 2000000,
                size: 0.1,
                pnl: 100.0,
                pnl_pips: 100.0,
                exit_reason: "take_profit".to_string(),
                scenario: 1,
            },
            TradeResult {
                id: 2,
                symbol: "EURUSD".to_string(),
                direction: "short".to_string(),
                entry_price: 1.1100,
                exit_price: 1.1150,
                entry_timestamp_us: 3000000,
                exit_timestamp_us: 4000000,
                size: 0.1,
                pnl: -50.0,
                pnl_pips: 50.0,
                exit_reason: "stop_loss".to_string(),
                scenario: 2,
            },
        ];
        
        let result = BacktestResult {
            strategy_name: "test".to_string(),
            symbol: "EURUSD".to_string(),
            trades,
            initial_capital: 10000.0,
            final_capital: 10050.0,
            bars_processed: 1000,
            execution_time_ms: 100.0,
            strategy_time_ms: 10.0,
            open_positions: 0,
        };
        
        assert_eq!(result.total_trades(), 2);
        assert_eq!(result.winning_trades(), 1);
        assert_eq!(result.losing_trades(), 1);
        assert_eq!(result.win_rate(), 0.5);
        assert_eq!(result.total_pnl(), 50.0);
        assert_eq!(result.profit_factor(), 2.0);  // 100 / 50
        assert_eq!(result.roi(), 0.5);  // 50 / 10000 * 100
    }
}
