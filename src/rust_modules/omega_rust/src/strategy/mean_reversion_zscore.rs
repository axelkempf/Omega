//! Mean Reversion Z-Score Strategie - Pure Rust Implementation
//!
//! Port der Python-Strategie `strategies/mean_reversion_z_score/strategy/mean_reversion_z_score.py`
//! zu reinem Rust für maximale Performance ohne FFI-Overhead.
//!
//! ## Szenarien
//!
//! Die Strategie unterstützt 6 verschiedene Szenarien basierend auf Market-State:
//!
//! | Szenario | Markt-Zustand | Beschreibung |
//! |----------|---------------|--------------|
//! | 1 | Low Volatility | Ruhiger Markt, enge Ranges |
//! | 2 | Expanding Volatility | Zunehmende Volatilität |
//! | 3 | High Volatility | Hohe Volatilität, weite Ranges |
//! | 4 | Mean Reverting | Seitwärts-Markt |
//! | 5 | Strong Trend | Starker Trend |
//! | 6 | Volatile Trend | Trend mit hoher Volatilität |

use std::sync::RwLock;

use crate::indicators::IndicatorCache;
use super::traits::{RustStrategy, StrategyError, StrategyFactory};
use super::types::{DataSlice, Direction, Position, PositionAction, StrategyConfig, Timeframe, TradeSignal};
use super::registry::register_strategy;

/// Parameter für die Mean Reversion Z-Score Strategie
#[derive(Clone, Debug)]
pub struct MeanReversionParams {
    // Z-Score Parameter
    pub z_score_lookback: usize,
    pub z_score_entry_threshold: f64,
    pub z_score_exit_threshold: f64,
    
    // Kalman-Filter Parameter
    pub kalman_q: f64,
    pub kalman_r: f64,
    
    // ATR Parameter
    pub atr_period: usize,
    pub atr_sl_multiplier: f64,
    pub atr_tp_multiplier: f64,
    
    // Trend-Filter Parameter
    pub ema_fast_period: usize,
    pub ema_slow_period: usize,
    pub adx_period: usize,
    pub adx_threshold: f64,
    
    // Szenario-Thresholds
    pub volatility_low_threshold: f64,
    pub volatility_high_threshold: f64,
    pub trend_strength_threshold: f64,
    
    // Risk Management
    pub max_daily_trades: usize,
    pub min_rr_ratio: f64,
    pub max_spread_pips: f64,
}

impl Default for MeanReversionParams {
    fn default() -> Self {
        Self {
            z_score_lookback: 100,
            z_score_entry_threshold: 2.0,
            z_score_exit_threshold: 0.5,
            kalman_q: 0.01,
            kalman_r: 1.0,
            atr_period: 14,
            atr_sl_multiplier: 2.0,
            atr_tp_multiplier: 3.0,
            ema_fast_period: 20,
            ema_slow_period: 50,
            adx_period: 14,
            adx_threshold: 25.0,
            volatility_low_threshold: 0.3,
            volatility_high_threshold: 0.7,
            trend_strength_threshold: 0.6,
            max_daily_trades: 5,
            min_rr_ratio: 1.5,
            max_spread_pips: 3.0,
        }
    }
}

impl MeanReversionParams {
    /// Erstellt Parameter aus StrategyConfig
    pub fn from_config(config: &StrategyConfig) -> Self {
        let defaults = Self::default();
        
        Self {
            // Z-Score
            z_score_lookback: config.get_param("z_score_lookback", defaults.z_score_lookback as f64) as usize,
            z_score_entry_threshold: config.get_param("z_score_entry_threshold", defaults.z_score_entry_threshold),
            z_score_exit_threshold: config.get_param("z_score_exit_threshold", defaults.z_score_exit_threshold),
            
            // Kalman
            kalman_q: config.get_param("kalman_q", defaults.kalman_q),
            kalman_r: config.get_param("kalman_r", defaults.kalman_r),
            
            // ATR
            atr_period: config.get_param("atr_period", defaults.atr_period as f64) as usize,
            atr_sl_multiplier: config.get_param("atr_sl_multiplier", defaults.atr_sl_multiplier),
            atr_tp_multiplier: config.get_param("atr_tp_multiplier", defaults.atr_tp_multiplier),
            
            // EMA/ADX
            ema_fast_period: config.get_param("ema_fast_period", defaults.ema_fast_period as f64) as usize,
            ema_slow_period: config.get_param("ema_slow_period", defaults.ema_slow_period as f64) as usize,
            adx_period: config.get_param("adx_period", defaults.adx_period as f64) as usize,
            adx_threshold: config.get_param("adx_threshold", defaults.adx_threshold),
            
            // Szenario-Thresholds
            volatility_low_threshold: config.get_param("volatility_low_threshold", defaults.volatility_low_threshold),
            volatility_high_threshold: config.get_param("volatility_high_threshold", defaults.volatility_high_threshold),
            trend_strength_threshold: config.get_param("trend_strength_threshold", defaults.trend_strength_threshold),
            
            // Risk Management
            max_daily_trades: config.get_param("max_daily_trades", defaults.max_daily_trades as f64) as usize,
            min_rr_ratio: config.get_param("min_rr_ratio", defaults.min_rr_ratio),
            max_spread_pips: config.get_param("max_spread_pips", defaults.max_spread_pips),
        }
    }
}

/// Market-State-Klassifikation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarketState {
    /// Szenario 1: Low Volatility
    LowVolatility,
    /// Szenario 2: Expanding Volatility
    ExpandingVolatility,
    /// Szenario 3: High Volatility
    HighVolatility,
    /// Szenario 4: Mean Reverting
    MeanReverting,
    /// Szenario 5: Strong Trend
    StrongTrend,
    /// Szenario 6: Volatile Trend
    VolatileTrend,
}

impl MarketState {
    /// Konvertiert zu Szenario-Nummer (1-6)
    pub fn to_scenario(&self) -> u8 {
        match self {
            Self::LowVolatility => 1,
            Self::ExpandingVolatility => 2,
            Self::HighVolatility => 3,
            Self::MeanReverting => 4,
            Self::StrongTrend => 5,
            Self::VolatileTrend => 6,
        }
    }
}

/// Mean Reversion Z-Score Strategie
pub struct MeanReversionZScore {
    params: MeanReversionParams,
    enabled_scenarios: Vec<u8>,
    direction_filter: String,
    
    // Mutable state wrapped in RwLock for interior mutability + Send+Sync
    // This allows &self methods to update internal state
    daily_trade_count: RwLock<usize>,
    last_trade_day: RwLock<i64>,
    
    // Kalman-Filter State (also needs RwLock for interior mutability + Send+Sync)
    kalman_mean: RwLock<f64>,
    kalman_variance: RwLock<f64>,
}

impl MeanReversionZScore {
    /// Erstellt eine neue Instanz mit den gegebenen Parametern.
    pub fn new(params: MeanReversionParams) -> Self {
        Self {
            params,
            enabled_scenarios: vec![1, 2, 3, 4, 5, 6],
            direction_filter: "both".to_string(),
            daily_trade_count: RwLock::new(0),
            last_trade_day: RwLock::new(0),
            kalman_mean: RwLock::new(0.0),
            kalman_variance: RwLock::new(1.0),
        }
    }
    
    /// Erstellt eine Instanz aus StrategyConfig.
    /// 
    /// Note: enabled_scenarios and direction_filter use defaults since
    /// StrategyConfig.params only supports f64 values. All 6 scenarios
    /// are enabled by default, and direction_filter defaults to "both".
    pub fn from_config(config: &StrategyConfig) -> Self {
        let params = MeanReversionParams::from_config(config);
        Self::new(params)
        // All scenarios enabled by default, direction_filter = "both"
    }
    
    /// Klassifiziert den aktuellen Markt-Zustand.
    fn classify_market_state(
        &self,
        volatility_percentile: f64,
        volatility_expanding: bool,
        trend_strength: f64,
        is_trending: bool,
    ) -> MarketState {
        // Volatility classification
        let is_low_vol = volatility_percentile < self.params.volatility_low_threshold;
        let is_high_vol = volatility_percentile > self.params.volatility_high_threshold;
        let is_strong_trend = trend_strength > self.params.trend_strength_threshold;
        
        if is_low_vol && !is_trending {
            MarketState::LowVolatility
        } else if volatility_expanding && !is_high_vol {
            MarketState::ExpandingVolatility
        } else if is_high_vol && is_trending && is_strong_trend {
            MarketState::VolatileTrend
        } else if is_high_vol && !is_trending {
            MarketState::HighVolatility
        } else if is_trending && is_strong_trend {
            MarketState::StrongTrend
        } else {
            MarketState::MeanReverting
        }
    }
    
    /// Berechnet Z-Score mit Kalman-Filter.
    /// Uses interior mutability via RwLock to allow mutation through &self
    fn calculate_kalman_zscore(&self, price: f64, initial_warmup: bool) -> f64 {
        // Kalman Filter Update
        let q = self.params.kalman_q;
        let r = self.params.kalman_r;
        
        // Get current state
        let current_mean = *self.kalman_mean.read().unwrap();
        let current_variance = *self.kalman_variance.read().unwrap();
        
        // Initialize with first price if mean is still at default
        if current_mean == 0.0 || initial_warmup {
            *self.kalman_mean.write().unwrap() = price;
            *self.kalman_variance.write().unwrap() = (price * 0.01).powi(2); // 1% of price as initial variance
            return 0.0; // No signal during initialization
        }
        
        // Prediction
        let pred_variance = current_variance + q;
        
        // Update
        let kalman_gain = pred_variance / (pred_variance + r);
        let new_mean = current_mean + kalman_gain * (price - current_mean);
        let new_variance = (1.0 - kalman_gain) * pred_variance;
        
        // Store updated state
        *self.kalman_mean.write().unwrap() = new_mean;
        *self.kalman_variance.write().unwrap() = new_variance;
        
        // Z-Score based on price deviation from Kalman mean
        // Use a more realistic volatility measure: ATR-based or rolling std
        let price_range = price * 0.001; // ~10 pips for EURUSD 
        let std_dev = new_variance.sqrt().max(price_range);
        (price - new_mean) / std_dev
    }
    
    /// Berechnet klassischen Z-Score aus Preis-History.
    /// Primary method for signal generation - more stable than Kalman.
    fn calculate_classic_zscore(&self, closes: &[f64], lookback: usize, current_idx: usize) -> Option<f64> {
        if current_idx < lookback {
            return None;
        }
        
        let start = current_idx.saturating_sub(lookback);
        let recent = &closes[start..current_idx];
        let current_price = *closes.get(current_idx)?;
        
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev < 1e-10 {
            return None;
        }
        
        Some((current_price - mean) / std_dev)
    }
    
    /// Berechnet Volatilitäts-Perzentil.
    fn calculate_volatility_percentile(&self, atr_values: &[f64], lookback: usize) -> f64 {
        if atr_values.len() < lookback {
            return 0.5;
        }
        
        let current_atr = *atr_values.last().unwrap_or(&0.0);
        let recent = &atr_values[atr_values.len() - lookback..];
        
        let count_below = recent.iter().filter(|&&x| x < current_atr).count();
        count_below as f64 / lookback as f64
    }
    
    /// Prüft ob Volatilität expandiert.
    fn is_volatility_expanding(&self, atr_values: &[f64], short_period: usize, long_period: usize) -> bool {
        if atr_values.len() < long_period {
            return false;
        }
        
        let short_avg: f64 = atr_values[atr_values.len() - short_period..].iter().sum::<f64>() / short_period as f64;
        let long_avg: f64 = atr_values[atr_values.len() - long_period..].iter().sum::<f64>() / long_period as f64;
        
        short_avg > long_avg * 1.1  // 10% höher
    }
    
    /// Berechnet Trend-Stärke basierend auf EMA-Differenz.
    fn calculate_trend_strength(&self, ema_fast: f64, ema_slow: f64, atr: f64) -> f64 {
        if atr < 1e-10 {
            return 0.0;
        }
        ((ema_fast - ema_slow) / atr).abs().min(1.0)
    }
    
    /// Prüft ob der Markt trendet.
    fn is_trending(&self, adx: f64) -> bool {
        adx > self.params.adx_threshold
    }
    
    /// Berechnet Entry-Parameter für einen Trade.
    fn calculate_entry(
        &self,
        direction: Direction,
        bid_price: f64,
        ask_price: f64,
        atr: f64,
    ) -> (f64, f64, f64) {
        let entry_price = match direction {
            Direction::Long => ask_price,
            Direction::Short => bid_price,
        };
        
        let (stop_loss, take_profit) = match direction {
            Direction::Long => (
                entry_price - atr * self.params.atr_sl_multiplier,
                entry_price + atr * self.params.atr_tp_multiplier,
            ),
            Direction::Short => (
                entry_price + atr * self.params.atr_sl_multiplier,
                entry_price - atr * self.params.atr_tp_multiplier,
            ),
        };
        
        (entry_price, stop_loss, take_profit)
    }
    
    /// Validiert Trade-Setup.
    fn validate_setup(
        &self,
        entry_price: f64,
        stop_loss: f64,
        take_profit: f64,
        spread: f64,
    ) -> bool {
        // Spread-Check
        if spread > self.params.max_spread_pips * 0.0001 {
            return false;
        }
        
        // Risk-Reward Check
        let risk = (entry_price - stop_loss).abs();
        let reward = (take_profit - entry_price).abs();
        if risk < 1e-10 || reward / risk < self.params.min_rr_ratio {
            return false;
        }
        
        true
    }
    
    /// Aktualisiert den täglichen Trade-Counter.
    /// Uses interior mutability via RwLock for thread safety.
    fn update_daily_counter(&self, timestamp_us: i64) {
        let day = timestamp_us / (24 * 60 * 60 * 1_000_000);
        if day != *self.last_trade_day.read().unwrap() {
            *self.daily_trade_count.write().unwrap() = 0;
            *self.last_trade_day.write().unwrap() = day;
        }
    }
}

impl RustStrategy for MeanReversionZScore {
    fn name(&self) -> &str {
        "mean_reversion_zscore"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn description(&self) -> &str {
        "Mean Reversion Strategy using Kalman-filtered Z-Score with 6 market state scenarios"
    }
    
    fn primary_timeframe(&self) -> Timeframe {
        Timeframe::M5
    }
    
    fn warmup_bars(&self) -> usize {
        self.params.z_score_lookback.max(self.params.ema_slow_period).max(100)
    }
    
    fn max_positions(&self) -> usize {
        1
    }
    
    fn direction_filter(&self) -> &str {
        &self.direction_filter
    }
    
    fn enabled_scenarios(&self) -> &[u8] {
        &self.enabled_scenarios
    }
    
    fn on_init(&mut self, _config: &StrategyConfig) -> Result<(), StrategyError> {
        // Uses interior mutability via RwLock for thread safety
        *self.kalman_mean.write().unwrap() = 0.0;
        *self.kalman_variance.write().unwrap() = 1.0;
        *self.daily_trade_count.write().unwrap() = 0;
        *self.last_trade_day.write().unwrap() = 0;
        Ok(())
    }
    
    fn evaluate(
        &self,
        slice: &DataSlice,
        indicator_cache: &mut IndicatorCache,
    ) -> Option<TradeSignal> {
        // Update daily counter
        self.update_daily_counter(slice.timestamp_us);
        
        // Daily trade limit
        if *self.daily_trade_count.read().unwrap() >= self.params.max_daily_trades {
            return None;
        }
        
        // Get current prices using helper methods
        let bid_candle = slice.current_bid()?;
        let ask_candle = slice.current_ask()?;
        let mid_price = (bid_candle.close + ask_candle.close) / 2.0;
        let spread = ask_candle.close - bid_candle.close;
        
        // Get indicators from cache using actual API
        // Note: price_type must match registration case (uppercase)
        let symbol = &slice.symbol;
        let tf_str = self.primary_timeframe().as_str();
        
        // EMA: cache.ema(symbol, timeframe, price_type, span, start_idx) -> Vec<f64>
        let ema_fast_vec = indicator_cache.ema(symbol, tf_str, "BID", self.params.ema_fast_period, None).ok()?;
        let ema_slow_vec = indicator_cache.ema(symbol, tf_str, "BID", self.params.ema_slow_period, None).ok()?;
        
        // ATR: cache.atr(symbol, timeframe, price_type, period) -> Vec<f64>
        let atr_vec = indicator_cache.atr(symbol, tf_str, "BID", self.params.atr_period).ok()?;
        
        // DMI: cache.dmi(symbol, timeframe, price_type, period) -> (+DI, -DI, ADX)
        let (_, _, adx_vec) = indicator_cache.dmi(symbol, tf_str, "BID", self.params.adx_period).ok()?;
        
        // Get close prices for Z-Score calculation
        let closes = indicator_cache.closes(symbol, tf_str, "BID").ok()?;
        
        // Get values at current index
        let idx = slice.index;
        let ema_fast = *ema_fast_vec.get(idx)?;
        let ema_slow = *ema_slow_vec.get(idx)?;
        let atr = *atr_vec.get(idx)?;
        let adx = *adx_vec.get(idx)?;
        
        // Z-Score calculation using classic rolling method for reliable signals
        let z_score = self.calculate_classic_zscore(&closes, self.params.z_score_lookback, idx)?;
        
        // Also update Kalman filter for future use/comparison
        let _ = self.calculate_kalman_zscore(mid_price, idx < self.params.z_score_lookback);
        
        // Market state classification - use last 100 ATR values for history
        let history_start = idx.saturating_sub(100);
        let atr_history: Vec<f64> = atr_vec[history_start..=idx].to_vec();
        let volatility_percentile = self.calculate_volatility_percentile(&atr_history, 100);
        let volatility_expanding = self.is_volatility_expanding(&atr_history, 5, 20);
        let trend_strength = self.calculate_trend_strength(ema_fast, ema_slow, atr);
        let is_trending = self.is_trending(adx);
        
        let market_state = self.classify_market_state(
            volatility_percentile,
            volatility_expanding,
            trend_strength,
            is_trending,
        );
        let scenario = market_state.to_scenario();
        
        // Check if scenario is enabled
        if !self.is_scenario_enabled(scenario) {
            return None;
        }
        
        // Entry logic based on Z-Score
        let direction = if z_score < -self.params.z_score_entry_threshold {
            // Price below mean → Long
            Some(Direction::Long)
        } else if z_score > self.params.z_score_entry_threshold {
            // Price above mean → Short
            Some(Direction::Short)
        } else {
            None
        }?;
        
        // Direction filter
        if !self.is_direction_allowed(match direction {
            Direction::Long => "long",
            Direction::Short => "short",
        }) {
            return None;
        }
        
        // Calculate entry parameters
        let (entry_price, stop_loss, take_profit) = self.calculate_entry(
            direction.clone(),
            bid_candle.close,
            ask_candle.close,
            atr,
        );
        
        // Validate setup
        if !self.validate_setup(entry_price, stop_loss, take_profit, spread) {
            return None;
        }
        
        // Update trade counter (interior mutability via RwLock)
        *self.daily_trade_count.write().unwrap() += 1;
        
        Some(TradeSignal {
            symbol: slice.symbol.clone(),
            direction,
            entry_price,
            stop_loss,
            take_profit,
            timestamp_us: slice.timestamp_us,
            scenario,
            reason: Some(format!(
                "Z-Score: {:.2}, Market: {:?}, Scenario: {}",
                z_score, market_state, scenario
            )),
        })
    }
    
    fn manage_position(
        &self,
        position: &Position,
        slice: &DataSlice,
    ) -> PositionAction {
        // Get current prices using helper methods
        let bid_candle = match slice.current_bid() {
            Some(c) => c,
            None => return PositionAction::Hold(),
        };
        let ask_candle = match slice.current_ask() {
            Some(c) => c,
            None => return PositionAction::Hold(),
        };
        
        let current_price = match position.direction {
            Direction::Long => bid_candle.close,
            Direction::Short => ask_candle.close,
        };
        
        // Calculate unrealized PnL
        let unrealized = position.unrealized_pnl(current_price);
        let entry_risk = (position.entry_price - position.current_stop_loss).abs();
        
        // Break-even after 1R profit
        if unrealized > entry_risk && position.current_stop_loss != position.entry_price {
            return PositionAction::ModifyStopLoss { new_stop_loss: position.entry_price };
        }
        
        // Trailing stop after 2R profit
        if unrealized > 2.0 * entry_risk {
            let trail_distance = entry_risk * 0.5;  // Trail at 0.5R
            let new_sl = match position.direction {
                Direction::Long => current_price - trail_distance,
                Direction::Short => current_price + trail_distance,
            };
            
            // Only update if it improves the stop
            let better_sl = match position.direction {
                Direction::Long => new_sl > position.current_stop_loss,
                Direction::Short => new_sl < position.current_stop_loss,
            };
            
            if better_sl {
                return PositionAction::ModifyStopLoss { new_stop_loss: new_sl };
            }
        }
        
        PositionAction::Hold()
    }
}

// ============================================================================
// Factory Registration
// ============================================================================

/// Factory für MeanReversionZScore
pub struct MeanReversionZScoreFactory;

impl StrategyFactory for MeanReversionZScoreFactory {
    fn name(&self) -> &str {
        "mean_reversion_zscore"
    }
    
    fn create(&self, config: &StrategyConfig) -> Result<Box<dyn RustStrategy>, StrategyError> {
        let strategy = MeanReversionZScore::from_config(config);
        Ok(Box::new(strategy))
    }
}

/// Registriert die MeanReversionZScore-Strategie im globalen Registry.
pub fn register_mean_reversion_zscore() {
    register_strategy(Box::new(MeanReversionZScoreFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_default_params() {
        let params = MeanReversionParams::default();
        assert_eq!(params.z_score_lookback, 100);
        assert_eq!(params.z_score_entry_threshold, 2.0);
        assert_eq!(params.atr_period, 14);
    }

    #[test]
    fn test_market_state_classification() {
        let strategy = MeanReversionZScore::new(MeanReversionParams::default());
        
        // Low volatility, no trend
        let state = strategy.classify_market_state(0.2, false, 0.3, false);
        assert_eq!(state, MarketState::LowVolatility);
        assert_eq!(state.to_scenario(), 1);
        
        // High volatility, trending
        let state = strategy.classify_market_state(0.8, false, 0.8, true);
        assert_eq!(state, MarketState::VolatileTrend);
        assert_eq!(state.to_scenario(), 6);
        
        // Mean reverting
        let state = strategy.classify_market_state(0.5, false, 0.3, false);
        assert_eq!(state, MarketState::MeanReverting);
        assert_eq!(state.to_scenario(), 4);
    }

    #[test]
    fn test_zscore_calculation() {
        let strategy = MeanReversionZScore::new(MeanReversionParams::default());
        
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let zscore = strategy.calculate_zscore(&prices, 20);
        
        assert!(zscore.is_some());
        // Last price is highest, so z-score should be positive
        assert!(zscore.unwrap() > 0.0);
    }

    #[test]
    fn test_kalman_zscore() {
        let strategy = MeanReversionZScore::new(MeanReversionParams::default());
        
        // Feed constant price
        for _ in 0..100 {
            strategy.calculate_kalman_zscore(100.0);
        }
        
        // Z-score should be near 0 for constant price
        let z = strategy.calculate_kalman_zscore(100.0);
        assert!(z.abs() < 0.5);
        
        // Large deviation should give large z-score
        let z_high = strategy.calculate_kalman_zscore(110.0);
        assert!(z_high > 2.0);
    }

    #[test]
    fn test_strategy_trait_methods() {
        let strategy = MeanReversionZScore::new(MeanReversionParams::default());
        
        assert_eq!(strategy.name(), "mean_reversion_zscore");
        assert_eq!(strategy.version(), "1.0.0");
        assert_eq!(strategy.primary_timeframe(), Timeframe::M5);
        assert_eq!(strategy.max_positions(), 1);
        assert!(strategy.warmup_bars() >= 100);
    }

    #[test]
    fn test_entry_calculation() {
        let strategy = MeanReversionZScore::new(MeanReversionParams::default());
        
        let (entry, sl, tp) = strategy.calculate_entry(
            Direction::Long,
            1.1000,  // bid
            1.1002,  // ask
            0.0020,  // ATR
        );
        
        // Long entry at ask
        assert_eq!(entry, 1.1002);
        // SL below entry
        assert!(sl < entry);
        // TP above entry
        assert!(tp > entry);
    }

    #[test]
    fn test_factory_registration() {
        let factory = MeanReversionZScoreFactory;
        assert_eq!(factory.name(), "mean_reversion_zscore");
        
        let config = StrategyConfig {
            symbol: "EURUSD".to_string(),
            primary_timeframe: "M5".to_string(),
            initial_capital: 10000.0,
            risk_per_trade: 0.01,
            params: HashMap::new(),
        };
        
        let result = factory.create(&config);
        assert!(result.is_ok());
        
        let strategy = result.unwrap();
        assert_eq!(strategy.name(), "mean_reversion_zscore");
    }
}
