//! Mean Reversion Z-Score Strategy
//!
//! Implements all 6 scenarios for the Mean Reversion Z-Score trading strategy.
//!
//! # Scenarios
//! 1. Z-Score + EMA Take Profit
//! 2. Kalman-Z + Bollinger, TP = BB-Mid
//! 3. Limit Entry + Minimum TP Distance
//! 4. Same-Bar SL/TP Tie → SL-Priorität (GARCH-based)
//! 5. Entry Candle Rule + Intraday Vol Clustering
//! 6. Multi-TF + Sessions/Warmup Mix

use crate::context::BarContext;
use crate::error::StrategyError;
use crate::traits::{IndicatorRequirement, Strategy};
use omega_indicators::{GarchLocalParams, VolClusterRequest, VolFeatureSeries};
use omega_types::{Direction, OrderType, PriceType, Signal};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// HTF filter mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HtfFilter {
    /// Allow both directions
    #[default]
    Both,
    /// Only trade when price is above HTF EMA
    Above,
    /// Only trade when price is below HTF EMA
    Below,
    /// No HTF filter
    None,
}

/// Direction filter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionFilter {
    /// Allow both long and short
    #[default]
    Both,
    /// Only long trades
    Long,
    /// Only short trades
    Short,
}

/// Scenario 6 multi-TF agreement mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scenario6Mode {
    /// All timeframes must agree
    #[default]
    All,
    /// Any timeframe can trigger
    Any,
}

impl Scenario6Mode {
    /// Returns string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Scenario6Mode::All => "all",
            Scenario6Mode::Any => "any",
        }
    }
}

/// Mean Reversion Z-Score parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MrzParams {
    // Core parameters
    /// EMA period for trend/TP
    #[serde(default = "default_ema_length")]
    pub ema_length: usize,
    /// ATR period for volatility
    #[serde(default = "default_atr_length")]
    pub atr_length: usize,
    /// ATR multiplier for SL
    #[serde(default = "default_atr_mult")]
    pub atr_mult: f64,
    /// Bollinger Bands period
    #[serde(default = "default_bb_length")]
    pub b_b_length: usize,
    /// Bollinger Bands std factor
    #[serde(default = "default_std_factor")]
    pub std_factor: f64,
    /// Z-Score window length
    #[serde(default = "default_window_length")]
    pub window_length: usize,
    /// Z-Score threshold for long (negative)
    #[serde(default = "default_z_score_long")]
    pub z_score_long: f64,
    /// Z-Score threshold for short (positive)
    #[serde(default = "default_z_score_short")]
    pub z_score_short: f64,
    /// Kalman filter R (measurement noise)
    #[serde(default = "default_kalman_r")]
    pub kalman_r: f64,
    /// Kalman filter Q (process noise)
    #[serde(default = "default_kalman_q")]
    pub kalman_q: f64,

    // HTF parameters
    /// HTF timeframe (e.g., "H4")
    #[serde(default = "default_htf_tf")]
    pub htf_tf: String,
    /// HTF EMA period
    #[serde(default = "default_htf_ema")]
    pub htf_ema: usize,
    /// HTF filter mode
    #[serde(default)]
    pub htf_filter: HtfFilter,

    // Extra HTF parameters (Bias B)
    /// Extra HTF timeframe (e.g., "H1" or "NONE")
    #[serde(default = "default_extra_htf_tf")]
    pub extra_htf_tf: String,
    /// Extra HTF EMA period
    #[serde(default = "default_extra_htf_ema")]
    pub extra_htf_ema: usize,
    /// Extra HTF filter mode
    #[serde(default)]
    pub extra_htf_filter: HtfFilter,

    // GARCH parameters (Scenario 4/5)
    /// GARCH alpha
    #[serde(default = "default_garch_alpha")]
    pub garch_alpha: f64,
    /// GARCH beta
    #[serde(default = "default_garch_beta")]
    pub garch_beta: f64,
    /// GARCH omega
    #[serde(default = "default_garch_omega")]
    pub garch_omega: f64,
    /// Use log returns for GARCH
    #[serde(default = "default_garch_use_log_returns")]
    pub garch_use_log_returns: bool,
    /// Scale factor for GARCH returns
    #[serde(default = "default_garch_scale")]
    pub garch_scale: f64,
    /// Minimum periods for GARCH initialization
    #[serde(default = "default_garch_min_periods")]
    pub garch_min_periods: usize,
    /// GARCH sigma floor
    #[serde(default = "default_garch_sigma_floor")]
    pub garch_sigma_floor: f64,

    // Scenario 3 parameters
    /// Minimum TP distance (price units, not pips!)
    #[serde(default = "default_tp_min_distance")]
    pub tp_min_distance: f64,

    // Scenario 5 parameters (Vol Cluster)
    /// Intraday volatility cluster window
    #[serde(default = "default_vol_cluster_window")]
    pub intraday_vol_cluster_window: usize,
    /// Number of volatility clusters
    #[serde(default = "default_vol_cluster_k")]
    pub intraday_vol_cluster_k: usize,
    /// Intraday volatility feature (e.g., "`garch_forecast`")
    #[serde(default = "default_intraday_vol_feature")]
    pub intraday_vol_feature: String,
    /// Minimum points required for volatility clustering
    #[serde(default = "default_intraday_vol_min_points")]
    pub intraday_vol_min_points: usize,
    /// Whether to log-transform vol feature before clustering
    #[serde(default = "default_intraday_vol_log_transform")]
    pub intraday_vol_log_transform: bool,
    /// GARCH lookback for intraday vol clustering
    #[serde(default = "default_intraday_vol_garch_lookback")]
    pub intraday_vol_garch_lookback: usize,
    /// Allowed volatility clusters
    #[serde(default = "default_intraday_vol_allowed")]
    pub intraday_vol_allowed: Vec<String>,
    /// Cluster hysteresis bars
    #[serde(default = "default_cluster_hysteresis_bars")]
    pub cluster_hysteresis_bars: usize,

    // Scenario 6 parameters (Multi-TF)
    /// Multi-TF agreement mode
    #[serde(default)]
    pub scenario6_mode: Scenario6Mode,
    /// Additional timeframes for Scenario 6
    #[serde(default)]
    pub scenario6_timeframes: Vec<String>,
    /// Scenario 6 extra parameters
    #[serde(default)]
    pub scenario6_params: serde_json::Value,

    // Gating
    /// Legacy flag for enabling trade management (compat mapping).
    #[serde(default = "default_use_position_manager")]
    pub use_position_manager: bool,
    /// Maximum holding time in minutes (legacy compat; 0 disables).
    #[serde(default = "default_max_holding_minutes")]
    pub max_holding_minutes: u64,
    /// Direction filter
    #[serde(default)]
    pub direction_filter: DirectionFilter,
    /// Enabled scenarios (empty = all)
    #[serde(default = "default_enabled_scenarios")]
    pub enabled_scenarios: Vec<u8>,
}

// Default value functions
fn default_ema_length() -> usize {
    20
}
fn default_atr_length() -> usize {
    14
}
fn default_atr_mult() -> f64 {
    2.0
}
fn default_bb_length() -> usize {
    20
}
fn default_std_factor() -> f64 {
    2.0
}
fn default_window_length() -> usize {
    20
}
fn default_z_score_long() -> f64 {
    -2.0
}
fn default_z_score_short() -> f64 {
    2.0
}
fn default_kalman_r() -> f64 {
    1.0
}
fn default_kalman_q() -> f64 {
    0.1
}
fn default_htf_tf() -> String {
    "H4".to_string()
}
fn default_htf_ema() -> usize {
    200
}
fn default_extra_htf_tf() -> String {
    "NONE".to_string()
}
fn default_extra_htf_ema() -> usize {
    50
}
fn default_garch_alpha() -> f64 {
    0.1
}
fn default_garch_beta() -> f64 {
    0.8
}
fn default_garch_omega() -> f64 {
    0.00001
}
fn default_garch_use_log_returns() -> bool {
    true
}
fn default_garch_scale() -> f64 {
    100.0
}
fn default_garch_min_periods() -> usize {
    50
}
fn default_garch_sigma_floor() -> f64 {
    1e-6
}
fn default_tp_min_distance() -> f64 {
    0.0010
}
fn default_vol_cluster_window() -> usize {
    20
}
fn default_vol_cluster_k() -> usize {
    3
}
fn default_intraday_vol_feature() -> String {
    "garch_forecast".to_string()
}
fn default_intraday_vol_min_points() -> usize {
    60
}
fn default_intraday_vol_log_transform() -> bool {
    true
}
fn default_intraday_vol_garch_lookback() -> usize {
    500
}
fn default_intraday_vol_allowed() -> Vec<String> {
    vec!["low".to_string(), "mid".to_string()]
}
fn default_cluster_hysteresis_bars() -> usize {
    1
}
fn default_use_position_manager() -> bool {
    false
}
fn default_max_holding_minutes() -> u64 {
    0
}
fn default_enabled_scenarios() -> Vec<u8> {
    vec![1]
}

impl Default for MrzParams {
    fn default() -> Self {
        Self {
            ema_length: default_ema_length(),
            atr_length: default_atr_length(),
            atr_mult: default_atr_mult(),
            b_b_length: default_bb_length(),
            std_factor: default_std_factor(),
            window_length: default_window_length(),
            z_score_long: default_z_score_long(),
            z_score_short: default_z_score_short(),
            kalman_r: default_kalman_r(),
            kalman_q: default_kalman_q(),
            htf_tf: default_htf_tf(),
            htf_ema: default_htf_ema(),
            htf_filter: HtfFilter::default(),
            extra_htf_tf: default_extra_htf_tf(),
            extra_htf_ema: default_extra_htf_ema(),
            extra_htf_filter: HtfFilter::default(),
            garch_alpha: default_garch_alpha(),
            garch_beta: default_garch_beta(),
            garch_omega: default_garch_omega(),
            garch_use_log_returns: default_garch_use_log_returns(),
            garch_scale: default_garch_scale(),
            garch_min_periods: default_garch_min_periods(),
            garch_sigma_floor: default_garch_sigma_floor(),
            tp_min_distance: default_tp_min_distance(),
            intraday_vol_cluster_window: default_vol_cluster_window(),
            intraday_vol_cluster_k: default_vol_cluster_k(),
            intraday_vol_feature: default_intraday_vol_feature(),
            intraday_vol_min_points: default_intraday_vol_min_points(),
            intraday_vol_log_transform: default_intraday_vol_log_transform(),
            intraday_vol_garch_lookback: default_intraday_vol_garch_lookback(),
            intraday_vol_allowed: default_intraday_vol_allowed(),
            cluster_hysteresis_bars: default_cluster_hysteresis_bars(),
            scenario6_mode: Scenario6Mode::default(),
            scenario6_timeframes: Vec::new(),
            scenario6_params: serde_json::Value::Null,
            use_position_manager: default_use_position_manager(),
            max_holding_minutes: default_max_holding_minutes(),
            direction_filter: DirectionFilter::default(),
            enabled_scenarios: default_enabled_scenarios(),
        }
    }
}

/// Internal state for MRZ strategy
#[derive(Debug, Clone, Default)]
struct MrzState;

/// Multi-TF chain result for Scenario 6
#[derive(Debug, Clone)]
struct TfChainResult {
    tf: String,
    ok: bool,
    kalman_z: Option<f64>,
    threshold: f64,
    price: Option<f64>,
    band_upper: Option<f64>,
    band_lower: Option<f64>,
    status: String,
    params: serde_json::Value,
}

/// Mean Reversion Z-Score Strategy
///
/// Implements all 6 scenarios for mean reversion trading based on
/// Z-Score deviations from moving averages.
#[derive(Debug, Clone)]
pub struct MeanReversionZScore {
    /// Strategy parameters
    pub params: MrzParams,
    /// Internal state
    state: MrzState,
}

impl MeanReversionZScore {
    /// Creates a new strategy with the given parameters.
    #[must_use]
    pub fn new(params: MrzParams) -> Self {
        Self {
            params,
            state: MrzState,
        }
    }

    /// Creates a strategy from JSON parameters.
    ///
    /// # Errors
    /// Returns `StrategyError::Json` if the parameters are invalid JSON or
    /// `StrategyError::InvalidParams` if required constraints are violated.
    pub fn from_params(params: &serde_json::Value) -> Result<Self, StrategyError> {
        let mut params: MrzParams =
            serde_json::from_value(params.clone()).map_err(StrategyError::Json)?;

        // Validate parameters
        if params.z_score_long >= 0.0 {
            return Err(StrategyError::InvalidParams(
                "z_score_long must be negative".to_string(),
            ));
        }
        if params.z_score_short <= 0.0 {
            return Err(StrategyError::InvalidParams(
                "z_score_short must be positive".to_string(),
            ));
        }

        params.scenario6_timeframes = normalize_timeframe_list(&params.scenario6_timeframes);
        params.scenario6_params = normalize_scenario6_params(&params.scenario6_params);

        Ok(Self::new(params))
    }

    /// Tries a specific scenario.
    fn try_scenario(&mut self, scenario_id: u8, ctx: &BarContext) -> Option<Signal> {
        match scenario_id {
            1 => self.scenario_1(ctx),
            2 => self.scenario_2(ctx),
            3 => self.scenario_3(ctx),
            4 => self.scenario_4(ctx),
            5 => self.scenario_5(ctx),
            6 => self.scenario_6(ctx),
            _ => None,
        }
    }

    /// Scenario 1: Z-Score + EMA Take Profit
    ///
    /// Simple mean reversion: enter when Z-Score exceeds threshold,
    /// take profit at EMA, stop loss based on ATR.
    fn scenario_1(&self, ctx: &BarContext) -> Option<Signal> {
        let zscore = ctx.get_indicator(
            "Z_SCORE",
            &serde_json::json!({
                "window": self.params.window_length,
                "mean_source": "ema",
                "ema_period": self.params.ema_length
            }),
        )?;
        let ema = ctx.get_indicator_with_price_type(
            "EMA",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.ema_length}),
        )?;
        let atr = ctx.get_indicator_with_price_type(
            "ATR",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.atr_length}),
        )?;

        // Long Entry
        if self.params.direction_filter != DirectionFilter::Short
            && zscore <= self.params.z_score_long
        {
            let sl = ctx.bid.low - self.params.atr_mult * atr;
            let tp = ema;
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 1,
                tags: vec!["scenario1".to_string(), "long_1".to_string()],
                meta: serde_json::json!({"zscore": zscore}),
            });
        }

        // Short Entry
        if self.params.direction_filter != DirectionFilter::Long
            && zscore >= self.params.z_score_short
        {
            let atr_ask = ctx.get_indicator_with_price_type(
                "ATR",
                PriceType::Ask,
                &serde_json::json!({"period": self.params.atr_length}),
            )?;
            let sl = ctx.bid.high + self.params.atr_mult * atr_ask;
            let tp = ema;
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 1,
                tags: vec!["scenario1".to_string(), "short_1".to_string()],
                meta: serde_json::json!({"zscore": zscore}),
            });
        }

        None
    }

    /// Scenario 2: Kalman-Z + Bollinger, TP = BB-Mid
    ///
    /// Uses Kalman-filtered Z-Score and Bollinger Bands for entry,
    /// with HTF trend filter.
    fn scenario_2(&self, ctx: &BarContext) -> Option<Signal> {
        // HTF Filter Check
        if !self.check_htf_bias(ctx) {
            return None;
        }

        let kalman_z = ctx.get_indicator(
            "KALMAN_Z",
            &serde_json::json!({
                "window": self.params.window_length,
                "r": self.params.kalman_r,
                "q": self.params.kalman_q
            }),
        )?;

        let bb_lower = ctx.get_indicator_output(
            "BOLLINGER",
            "lower",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_upper = ctx.get_indicator_output(
            "BOLLINGER",
            "upper",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_mid = ctx.get_indicator_output(
            "BOLLINGER",
            "middle",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;

        let atr = ctx.get_indicator_with_price_type(
            "ATR",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.atr_length}),
        )?;
        let close = ctx.bid.close;

        // Long
        if self.params.direction_filter != DirectionFilter::Short
            && kalman_z <= self.params.z_score_long
            && close <= bb_lower
        {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 2,
                tags: vec![
                    "scenario2".to_string(),
                    "long_2".to_string(),
                    "kalman".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_lower": bb_lower
                }),
            });
        }

        // Short
        if self.params.direction_filter != DirectionFilter::Long
            && kalman_z >= self.params.z_score_short
            && close >= bb_upper
        {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.bid.high + self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 2,
                tags: vec![
                    "scenario2".to_string(),
                    "short_2".to_string(),
                    "kalman".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_upper": bb_upper
                }),
            });
        }

        None
    }

    /// Scenario 3: Kalman-Z + Bollinger Entry, TP=EMA mit Mindestabstand
    ///
    /// Like Scenario 2 but with EMA as TP target and minimum TP distance as
    /// additional entry condition.
    fn scenario_3(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        let kalman_z = ctx.get_indicator(
            "KALMAN_Z",
            &serde_json::json!({
                "window": self.params.window_length,
                "r": self.params.kalman_r,
                "q": self.params.kalman_q
            }),
        )?;

        let bb_lower = ctx.get_indicator_output(
            "BOLLINGER",
            "lower",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_upper = ctx.get_indicator_output(
            "BOLLINGER",
            "upper",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;

        let atr = ctx.get_indicator_with_price_type(
            "ATR",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.atr_length}),
        )?;
        let ema = ctx.get_indicator_with_price_type(
            "EMA",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.ema_length}),
        )?;
        let close = ctx.bid.close;

        // Long: Kalman-Z + Bollinger Entry + TP min distance condition
        // Entry only if ema_now > ask_close + tp_min_distance
        if self.params.direction_filter != DirectionFilter::Short
            && kalman_z <= self.params.z_score_long
            && close <= bb_lower
            && ema > ctx.ask.close + self.params.tp_min_distance
        {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: ema,
                size: None,
                scenario_id: 3,
                tags: vec![
                    "scenario3".to_string(),
                    "long_3".to_string(),
                    "kalman".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_lower": bb_lower,
                    "ema": ema,
                    "tp_min_distance": self.params.tp_min_distance
                }),
            });
        }

        // Short: Kalman-Z + Bollinger Entry + TP min distance condition
        // Entry only if ema_now < bid_close - tp_min_distance
        if self.params.direction_filter != DirectionFilter::Long
            && kalman_z >= self.params.z_score_short
            && close >= bb_upper
            && ema < ctx.bid.close - self.params.tp_min_distance
        {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.bid.high + self.params.atr_mult * atr,
                take_profit: ema,
                size: None,
                scenario_id: 3,
                tags: vec![
                    "scenario3".to_string(),
                    "short_3".to_string(),
                    "kalman".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_upper": bb_upper,
                    "ema": ema,
                    "tp_min_distance": self.params.tp_min_distance
                }),
            });
        }

        None
    }

    /// Scenario 4: Kalman+GARCH Z + Bollinger Entry, TP=BB-Mid
    ///
    /// Like Scenario 2 but uses Kalman+GARCH Z-Score for signal generation.
    /// The GARCH component normalizes the Kalman residual by GARCH sigma.
    fn scenario_4(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        // Kalman+GARCH Z-Score (Kalman residual / GARCH sigma)
        let kalman_garch_z = ctx.get_indicator(
            "KALMAN_GARCH_Z",
            &serde_json::json!({
                "window": self.params.window_length,
                "r": self.params.kalman_r,
                "q": self.params.kalman_q,
                "alpha": self.params.garch_alpha,
                "beta": self.params.garch_beta,
                "omega": self.params.garch_omega,
                "use_log_returns": self.params.garch_use_log_returns,
                "scale": self.params.garch_scale,
                "min_periods": self.params.garch_min_periods,
                "sigma_floor": self.params.garch_sigma_floor
            }),
        )?;

        let bb_lower = ctx.get_indicator_output(
            "BOLLINGER",
            "lower",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_upper = ctx.get_indicator_output(
            "BOLLINGER",
            "upper",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_mid = ctx.get_indicator_output(
            "BOLLINGER",
            "middle",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;

        let atr = ctx.get_indicator_with_price_type(
            "ATR",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.atr_length}),
        )?;
        let close = ctx.bid.close;

        // Long: Kalman+GARCH-Z + Bollinger Entry
        if self.params.direction_filter != DirectionFilter::Short
            && kalman_garch_z <= self.params.z_score_long
            && close <= bb_lower
        {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 4,
                tags: vec![
                    "scenario4".to_string(),
                    "long_4".to_string(),
                    "kalman_garch".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_garch_z": kalman_garch_z,
                    "bb_lower": bb_lower,
                    "bb_mid": bb_mid
                }),
            });
        }

        // Short: Kalman+GARCH-Z + Bollinger Entry
        if self.params.direction_filter != DirectionFilter::Long
            && kalman_garch_z >= self.params.z_score_short
            && close >= bb_upper
        {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.bid.high + self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 4,
                tags: vec![
                    "scenario4".to_string(),
                    "short_4".to_string(),
                    "kalman_garch".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_garch_z": kalman_garch_z,
                    "bb_upper": bb_upper,
                    "bb_mid": bb_mid
                }),
            });
        }

        None
    }

    /// Scenario 5: Szenario 2 Basis + Intraday Vol Cluster Guard
    ///
    /// Same as Scenario 2 (Kalman-Z + Bollinger, TP=BB-Mid) but with
    /// an additional volatility cluster guard using GARCH-forecast.
    #[allow(clippy::too_many_lines)]
    fn scenario_5(&mut self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        let vol_cluster_meta = self.vol_cluster_guard(ctx)?;
        let vol_cluster_status = vol_cluster_meta
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        if vol_cluster_status != "ok" {
            return None;
        }

        // Same entry logic as Scenario 2: Kalman-Z + Bollinger
        let kalman_z = ctx.get_indicator(
            "KALMAN_Z",
            &serde_json::json!({
                "window": self.params.window_length,
                "r": self.params.kalman_r,
                "q": self.params.kalman_q
            }),
        )?;

        let bb_lower = ctx.get_indicator_output(
            "BOLLINGER",
            "lower",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_upper = ctx.get_indicator_output(
            "BOLLINGER",
            "upper",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_mid = ctx.get_indicator_output(
            "BOLLINGER",
            "middle",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;

        let atr = ctx.get_indicator_with_price_type(
            "ATR",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.atr_length}),
        )?;
        let close = ctx.bid.close;

        let mut vol_cluster_long = vol_cluster_meta.clone();
        if let Some(obj) = vol_cluster_long.as_object_mut() {
            obj.insert("direction".to_string(), serde_json::json!("long"));
        }

        // Long: Kalman-Z + Bollinger + Vol-Cluster Guard
        if self.params.direction_filter != DirectionFilter::Short
            && kalman_z <= self.params.z_score_long
            && close <= bb_lower
        {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 5,
                tags: vec![
                    "scenario5".to_string(),
                    "long_5".to_string(),
                    "kalman".to_string(),
                    "vol_cluster".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_lower": bb_lower,
                    "vol_cluster": vol_cluster_long
                }),
            });
        }

        // Short: Kalman-Z + Bollinger + Vol-Cluster Guard
        if self.params.direction_filter != DirectionFilter::Long
            && kalman_z >= self.params.z_score_short
            && close >= bb_upper
        {
            let mut vol_cluster_short = vol_cluster_meta.clone();
            if let Some(obj) = vol_cluster_short.as_object_mut() {
                obj.insert("direction".to_string(), serde_json::json!("short"));
            }
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.bid.high + self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 5,
                tags: vec![
                    "scenario5".to_string(),
                    "short_5".to_string(),
                    "kalman".to_string(),
                    "vol_cluster".to_string(),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_upper": bb_upper,
                    "vol_cluster": vol_cluster_short
                }),
            });
        }

        None
    }

    /// Scenario 6: Multi-TF Overlay with Kalman-Z + Bollinger per TF
    ///
    /// Same as Scenario 2 basis, but requires multi-TF agreement.
    /// Each TF checks: `kalman_z` threshold AND price vs Bollinger band.
    ///
    /// All: All TFs must give signal
    /// Any: At least one TF gives signal
    #[allow(clippy::too_many_lines)]
    fn scenario_6(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }
        if self.params.scenario6_timeframes.is_empty() {
            return None;
        }

        // Primary TF: Kalman-Z + Bollinger (like Scenario 2)
        let kalman_z = ctx.get_indicator(
            "KALMAN_Z",
            &serde_json::json!({
                "window": self.params.window_length,
                "r": self.params.kalman_r,
                "q": self.params.kalman_q
            }),
        )?;

        let bb_lower = ctx.get_indicator_output(
            "BOLLINGER",
            "lower",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_upper = ctx.get_indicator_output(
            "BOLLINGER",
            "upper",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;
        let bb_mid = ctx.get_indicator_output(
            "BOLLINGER",
            "middle",
            &serde_json::json!({
                "period": self.params.b_b_length,
                "std_factor": self.params.std_factor
            }),
        )?;

        let atr = ctx.get_indicator_with_price_type(
            "ATR",
            PriceType::Bid,
            &serde_json::json!({"period": self.params.atr_length}),
        )?;
        let close = ctx.bid.close;

        // Determine primary TF direction
        let primary_long = kalman_z <= self.params.z_score_long && close <= bb_lower;
        let primary_short = kalman_z >= self.params.z_score_short && close >= bb_upper;

        if !primary_long && !primary_short {
            return None;
        }

        // Multi-TF Agreement Check (Kalman-Z + Bollinger per TF)
        let tf_chain = self.check_multi_tf_signals_v2(ctx, primary_long);

        let has_agreement = match self.params.scenario6_mode {
            Scenario6Mode::All => tf_chain.iter().all(|c| c.ok),
            Scenario6Mode::Any => tf_chain.iter().any(|c| c.ok),
        };

        if !has_agreement {
            return None;
        }

        // Direction filter
        let mode_str = self.params.scenario6_mode.as_str();
        let chain_json: Vec<serde_json::Value> = tf_chain
            .iter()
            .map(|c| {
                serde_json::json!({
                    "tf": c.tf,
                    "ok": c.ok,
                    "status": c.status,
                    "kalman_z": c.kalman_z,
                    "threshold": c.threshold,
                    "price": c.price,
                    "upper": c.band_upper,
                    "lower": c.band_lower,
                    "params": c.params
                })
            })
            .collect();

        // Long
        if self.params.direction_filter != DirectionFilter::Short && primary_long {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 6,
                tags: vec![
                    "scenario6".to_string(),
                    "long_6".to_string(),
                    "kalman".to_string(),
                    format!("multi_tf_{}", mode_str),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_lower": bb_lower,
                    "scenario6": {
                        "mode": mode_str,
                        "chain": chain_json
                    }
                }),
            });
        }

        // Short
        if self.params.direction_filter != DirectionFilter::Long && primary_short {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.bid.high + self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 6,
                tags: vec![
                    "scenario6".to_string(),
                    "short_6".to_string(),
                    "kalman".to_string(),
                    format!("multi_tf_{}", mode_str),
                ],
                meta: serde_json::json!({
                    "kalman_z": kalman_z,
                    "bb_upper": bb_upper,
                    "scenario6": {
                        "mode": mode_str,
                        "chain": chain_json
                    }
                }),
            });
        }

        None
    }

    // ==================== Helper Methods ====================

    /// Checks Multi-TF signals using Kalman-Z + Bollinger per TF.
    #[allow(clippy::too_many_lines)]
    fn check_multi_tf_signals_v2(&self, ctx: &BarContext, is_long: bool) -> Vec<TfChainResult> {
        let direction = if is_long {
            Direction::Long
        } else {
            Direction::Short
        };
        self.params
            .scenario6_timeframes
            .iter()
            .map(|tf| {
                // Get TF-specific params from scenario6_params or use defaults
                let tf_params = self.scenario6_params_for_tf(tf, Some(direction));

                let window = tf_params
                    .get("window_length")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(self.params.window_length);
                let kalman_r = tf_params
                    .get("kalman_r")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.kalman_r);
                let kalman_q = tf_params
                    .get("kalman_q")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.kalman_q);
                let bb_period = tf_params
                    .get("b_b_length")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(self.params.b_b_length);
                let std_factor = tf_params
                    .get("std_factor")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.std_factor);
                let z_long = tf_params
                    .get("z_score_long")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.z_score_long);
                let z_short = tf_params
                    .get("z_score_short")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.z_score_short);

                let params_json = serde_json::json!({
                    "window_length": window,
                    "b_b_length": bb_period,
                    "std_factor": std_factor,
                    "kalman_r": kalman_r,
                    "kalman_q": kalman_q,
                    "z_score_long": z_long,
                    "z_score_short": z_short
                });

                // Get TF-specific Kalman-Z
                let kalman_z =
                    ctx.get_stepwise_kalman_zscore(tf, PriceType::Bid, window, kalman_r, kalman_q);

                // Get TF-specific price and Bollinger
                let tf_price = ctx.get_tf_close(tf, PriceType::Bid);
                let threshold = if is_long { z_long } else { z_short };
                let mut band_upper = None;
                let mut band_lower = None;
                let mut status = "ok".to_string();
                let mut ok = false;

                match (kalman_z, tf_price) {
                    (Some(z), Some(price)) => {
                        if is_long {
                            if z > z_long {
                                status = "z_above_threshold".to_string();
                            } else {
                                band_lower = ctx.get_stepwise_bollinger_output(
                                    tf,
                                    PriceType::Bid,
                                    bb_period,
                                    std_factor,
                                    "lower",
                                );
                                match band_lower {
                                    Some(b) if price <= b => {
                                        ok = true;
                                    }
                                    Some(_) => {
                                        status = "price_above_lower".to_string();
                                    }
                                    None => {
                                        status = "no_lower_band".to_string();
                                    }
                                }
                            }
                        } else if z < z_short {
                            status = "z_below_threshold".to_string();
                        } else {
                            band_upper = ctx.get_stepwise_bollinger_output(
                                tf,
                                PriceType::Bid,
                                bb_period,
                                std_factor,
                                "upper",
                            );
                            match band_upper {
                                Some(b) if price >= b => {
                                    ok = true;
                                }
                                Some(_) => {
                                    status = "price_below_upper".to_string();
                                }
                                None => {
                                    status = "no_upper_band".to_string();
                                }
                            }
                        }
                    }
                    (None, _) => {
                        status = "no_zscore".to_string();
                    }
                    (_, None) => {
                        status = "no_price".to_string();
                    }
                }

                TfChainResult {
                    tf: tf.clone(),
                    ok,
                    kalman_z,
                    threshold,
                    price: tf_price,
                    band_upper,
                    band_lower,
                    status,
                    params: params_json,
                }
            })
            .collect()
    }

    fn scenario6_params_for_tf(
        &self,
        tf: &str,
        direction: Option<Direction>,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut resolved = serde_json::Map::new();
        let Some(map) = self.params.scenario6_params.as_object() else {
            return resolved;
        };

        let normalized_tf = normalize_timeframe_name(tf);
        let direction_key = direction.map(|dir| match dir {
            Direction::Long => "long",
            Direction::Short => "short",
        });

        let wildcard_value = find_scenario6_param_value(map, "*");
        if let Some(wildcard) = wildcard_value.and_then(|v| v.as_object())
            && !is_direction_map(wildcard)
        {
            merge_params(&mut resolved, wildcard);
        }

        if let Some(dir_key) = direction_key {
            let wildcard_dir_key = format!("*.{dir_key}");
            if let Some(obj) = find_scenario6_param_value(map, &wildcard_dir_key)
                .and_then(|v| v.as_object())
            {
                merge_params(&mut resolved, obj);
            } else if let Some(obj) = wildcard_value
                .and_then(|v| v.as_object())
                .and_then(|v| direction_object(v, dir_key))
            {
                merge_params(&mut resolved, obj);
            }
        }

        let tf_value = find_scenario6_param_value(map, &normalized_tf);
        if let Some(tf_map) = tf_value.and_then(|v| v.as_object())
            && !is_direction_map(tf_map)
        {
            merge_params(&mut resolved, tf_map);
        }

        if let Some(dir_key) = direction_key {
            if let Some(tf_map) = tf_value.and_then(|v| v.as_object())
                && let Some(obj) = direction_object(tf_map, dir_key)
            {
                merge_params(&mut resolved, obj);
            }

            let tf_dir_key = format!("{normalized_tf}.{dir_key}");
            if let Some(obj) = find_scenario6_param_value(map, &tf_dir_key)
                .and_then(|v| v.as_object())
            {
                merge_params(&mut resolved, obj);
            }
        }

        resolved
    }

    /// Checks HTF trend bias (Bias A + optional Bias B).
    fn check_htf_bias(&self, ctx: &BarContext) -> bool {
        let ok_a = Self::check_htf_filter(
            ctx,
            &self.params.htf_tf,
            self.params.htf_ema,
            self.params.htf_filter,
        );
        let ok_b = Self::check_htf_filter(
            ctx,
            &self.params.extra_htf_tf,
            self.params.extra_htf_ema,
            self.params.extra_htf_filter,
        );
        ok_a && ok_b
    }

    fn check_htf_filter(
        ctx: &BarContext,
        timeframe: &str,
        ema_len: usize,
        filter: HtfFilter,
    ) -> bool {
        if filter == HtfFilter::None || filter == HtfFilter::Both {
            return true;
        }
        if is_htf_disabled(timeframe) {
            return true;
        }
        let ema = ctx.get_stepwise_ema(timeframe, PriceType::Bid, ema_len);
        let price = ctx.get_tf_close(timeframe, PriceType::Bid);
        match (ema, price) {
            (Some(ema), Some(price)) => match filter {
                HtfFilter::Above => price > ema,
                HtfFilter::Below => price < ema,
                HtfFilter::Both | HtfFilter::None => true,
            },
            _ => true,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn vol_cluster_guard(&self, ctx: &BarContext) -> Option<serde_json::Value> {
        let feature = self.params.intraday_vol_feature.trim().to_lowercase();
        let window = self.params.intraday_vol_cluster_window.max(1);
        let k = self.params.intraday_vol_cluster_k.max(1);
        let min_points = self.params.intraday_vol_min_points.max(1);
        let log_transform = self.params.intraday_vol_log_transform;
        let hysteresis_bars = self.params.cluster_hysteresis_bars;

        let mut meta = serde_json::Map::new();
        meta.insert("feature".to_string(), serde_json::json!(feature));
        meta.insert("window".to_string(), serde_json::json!(window));
        meta.insert("k".to_string(), serde_json::json!(k));
        meta.insert("min_points".to_string(), serde_json::json!(min_points));
        meta.insert(
            "log_transform".to_string(),
            serde_json::json!(log_transform),
        );
        meta.insert(
            "hysteresis_bars".to_string(),
            serde_json::json!(hysteresis_bars),
        );
        let feature_supported = matches!(feature.as_str(), "garch_forecast" | "atr_points");
        if !feature_supported {
            meta.insert(
                "status".to_string(),
                serde_json::json!("feature_not_supported"),
            );
            meta.insert("allowed_now".to_string(), serde_json::json!(false));
            return Some(serde_json::Value::Object(meta));
        }

        let series = self.vol_cluster_feature_series(ctx, &feature)?;
        let mut values = match series {
            VolFeatureSeries::Full(values) | VolFeatureSeries::Local { values, .. } => values,
        };
        if values.is_empty() {
            meta.insert(
                "status".to_string(),
                serde_json::json!("series_unavailable"),
            );
            return None;
        }

        let end = ctx.idx.min(values.len().saturating_sub(1)) + 1;
        values.truncate(end);
        let mut cleaned: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
        let sample_size = cleaned.len();
        meta.insert("sample_size".to_string(), serde_json::json!(sample_size));
        if sample_size < min_points {
            meta.insert(
                "status".to_string(),
                serde_json::json!("insufficient_points"),
            );
            return None;
        }

        let start = sample_size.saturating_sub(window);
        let tail = cleaned.split_off(start);
        if tail.len() < k {
            meta.insert(
                "status".to_string(),
                serde_json::json!("insufficient_unique"),
            );
            return None;
        }

        let sigma = tail.last().copied().filter(|v| v.is_finite());
        if let Some(sigma) = sigma {
            meta.insert("sigma".to_string(), serde_json::json!(sigma));
        }

        let mut cluster_values = tail.clone();
        if log_transform {
            for v in &mut cluster_values {
                *v = v.max(1e-12).ln();
            }
        }

        let Some((centers, labels)) = kmeans_1d(&cluster_values, k) else {
            meta.insert("status".to_string(), serde_json::json!("clustering_failed"));
            return None;
        };

        let mut order: Vec<usize> = (0..centers.len()).collect();
        order.sort_by(|a, b| {
            centers[*a]
                .partial_cmp(&centers[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let label_names = ["low", "mid", "high", "very_high", "extreme"];
        let mut mapping = vec![String::new(); centers.len()];
        for (rank, idx_c) in order.iter().enumerate() {
            let label = if rank < label_names.len() {
                label_names[rank].to_string()
            } else {
                format!("cluster_{rank}")
            };
            mapping[*idx_c] = label;
        }

        let mut centers_sorted = Vec::with_capacity(order.len());
        let mut centers_sorted_log = Vec::with_capacity(order.len());
        for idx_c in &order {
            let c = centers[*idx_c];
            if log_transform {
                centers_sorted.push(c.exp());
                centers_sorted_log.push(c);
            } else {
                centers_sorted.push(c);
            }
        }

        if !centers_sorted.is_empty() {
            meta.insert("centers".to_string(), serde_json::json!(centers_sorted));
        }
        if log_transform && !centers_sorted_log.is_empty() {
            meta.insert(
                "centers_log".to_string(),
                serde_json::json!(centers_sorted_log),
            );
        }

        let current_idx = labels.last().copied().unwrap_or(0);
        let current_label = mapping
            .get(current_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        meta.insert("label".to_string(), serde_json::json!(current_label));

        let allowed_labels: Vec<String> = self
            .params
            .intraday_vol_allowed
            .iter()
            .map(|label| label.trim().to_lowercase())
            .collect();
        if !allowed_labels.is_empty() {
            meta.insert(
                "allowed_labels".to_string(),
                serde_json::json!(allowed_labels),
            );
        }

        let allowed_now =
            allowed_labels.is_empty() || allowed_labels.contains(&current_label.to_lowercase());
        meta.insert("allowed_now".to_string(), serde_json::json!(allowed_now));

        let hysteresis_ok = if hysteresis_bars > 1 {
            if labels.len() < hysteresis_bars {
                false
            } else {
                labels[labels.len() - hysteresis_bars..]
                    .iter()
                    .all(|lbl| *lbl == current_idx)
            }
        } else {
            true
        };
        meta.insert(
            "hysteresis_ok".to_string(),
            serde_json::json!(hysteresis_ok),
        );

        if !hysteresis_ok {
            meta.insert("status".to_string(), serde_json::json!("hysteresis_block"));
            return None;
        }
        if !allowed_now {
            meta.insert("status".to_string(), serde_json::json!("label_block"));
            return None;
        }

        meta.insert("status".to_string(), serde_json::json!("ok"));
        Some(serde_json::Value::Object(meta))
    }

    fn vol_cluster_feature_series(
        &self,
        ctx: &BarContext,
        feature: &str,
    ) -> Option<VolFeatureSeries> {
        let cache = ctx.multi_tf?;
        let primary_tf = cache.borrow().primary_timeframe();
        let garch_params = GarchLocalParams {
            alpha: self.params.garch_alpha,
            beta: self.params.garch_beta,
            omega: Some(self.params.garch_omega),
            use_log_returns: self.params.garch_use_log_returns,
            scale: self.params.garch_scale,
            min_periods: self.params.garch_min_periods,
            sigma_floor: self.params.garch_sigma_floor,
        };
        let request = VolClusterRequest {
            timeframe: primary_tf,
            price_type: PriceType::Bid,
            idx: ctx.idx,
            feature,
            atr_length: self.params.atr_length,
            garch_lookback: self.params.intraday_vol_garch_lookback.max(1),
            garch_params,
        };
        cache.borrow_mut().vol_cluster_series(request)
    }
}

impl Strategy for MeanReversionZScore {
    fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal> {
        // Entry-Gates
        if !ctx.session_open || ctx.news_blocked {
            return None;
        }

        // Try each enabled scenario
        for scenario_id in self.params.enabled_scenarios.clone() {
            if let Some(signal) = self.try_scenario(scenario_id, ctx) {
                return Some(signal);
            }
        }

        None
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "mean_reversion_z_score"
    }

    #[allow(clippy::too_many_lines)]
    fn required_indicators(&self) -> Vec<IndicatorRequirement> {
        let mut reqs = vec![
            IndicatorRequirement::new("EMA", serde_json::json!({"period": self.params.ema_length})),
            IndicatorRequirement::new("ATR", serde_json::json!({"period": self.params.atr_length})),
            IndicatorRequirement::new(
                "Z_SCORE",
                serde_json::json!({
                    "window": self.params.window_length,
                    "mean_source": "ema",
                    "ema_period": self.params.ema_length
                }),
            ),
        ];

        // Add Bollinger if scenarios 2, 3, 4, 5, or 6 are enabled
        if self
            .params
            .enabled_scenarios
            .iter()
            .any(|&s| matches!(s, 2..=6))
        {
            reqs.push(IndicatorRequirement::new(
                "BOLLINGER",
                serde_json::json!({
                    "period": self.params.b_b_length,
                    "std_factor": self.params.std_factor
                }),
            ));
        }

        // Add Kalman Z if scenarios 2, 3, 5, or 6 are enabled
        if self
            .params
            .enabled_scenarios
            .iter()
            .any(|&s| matches!(s, 2 | 3 | 5 | 6))
        {
            reqs.push(IndicatorRequirement::new(
                "KALMAN_Z",
                serde_json::json!({
                    "window": self.params.window_length,
                    "r": self.params.kalman_r,
                    "q": self.params.kalman_q
                }),
            ));
        }

        // Add Kalman+GARCH Z if scenario 4 is enabled
        if self.params.enabled_scenarios.contains(&4) {
            reqs.push(IndicatorRequirement::new(
                "KALMAN_GARCH_Z",
                serde_json::json!({
                    "window": self.params.window_length,
                    "r": self.params.kalman_r,
                    "q": self.params.kalman_q,
                    "alpha": self.params.garch_alpha,
                    "beta": self.params.garch_beta,
                    "omega": self.params.garch_omega,
                    "use_log_returns": self.params.garch_use_log_returns,
                    "scale": self.params.garch_scale,
                    "min_periods": self.params.garch_min_periods,
                    "sigma_floor": self.params.garch_sigma_floor
                }),
            ));
        }

        // Add GARCH VOL if scenario 5 is enabled (for vol_cluster meta)
        if self.params.enabled_scenarios.contains(&5) {
            reqs.push(IndicatorRequirement::new(
                "GARCH_VOL",
                serde_json::json!({
                    "alpha": self.params.garch_alpha,
                    "beta": self.params.garch_beta,
                    "omega": self.params.garch_omega,
                    "use_log_returns": self.params.garch_use_log_returns,
                    "scale": self.params.garch_scale,
                    "min_periods": self.params.garch_min_periods,
                    "sigma_floor": self.params.garch_sigma_floor
                }),
            ));
        }

        // Add Vol Cluster if scenario 5 is enabled
        if self.params.enabled_scenarios.contains(&5) {
            reqs.push(IndicatorRequirement::new(
                "VOL_CLUSTER",
                serde_json::json!({
                    "feature": self.params.intraday_vol_feature,
                    "window": self.params.intraday_vol_cluster_window,
                    "k": self.params.intraday_vol_cluster_k,
                    "min_points": self.params.intraday_vol_min_points,
                    "log_transform": self.params.intraday_vol_log_transform,
                    "garch_lookback": self.params.intraday_vol_garch_lookback,
                    "alpha": self.params.garch_alpha,
                    "beta": self.params.garch_beta,
                    "omega": self.params.garch_omega,
                    "use_log_returns": self.params.garch_use_log_returns,
                    "scale": self.params.garch_scale,
                    "min_periods": self.params.garch_min_periods,
                    "sigma_floor": self.params.garch_sigma_floor
                }),
            ));
        }

        // Add HTF EMA if HTF filter is enabled
        if self.params.htf_filter != HtfFilter::None {
            reqs.push(IndicatorRequirement::with_timeframe(
                "EMA",
                &self.params.htf_tf,
                serde_json::json!({"period": self.params.htf_ema}),
            ));
        }

        if self.params.extra_htf_filter != HtfFilter::None
            && !is_htf_disabled(&self.params.extra_htf_tf)
        {
            reqs.push(IndicatorRequirement::with_timeframe(
                "EMA",
                &self.params.extra_htf_tf,
                serde_json::json!({"period": self.params.extra_htf_ema}),
            ));
        }

        // Add TF-specific indicators for scenario 6
        if self.params.enabled_scenarios.contains(&6) {
            for tf in &self.params.scenario6_timeframes {
                // Get TF-specific params or defaults
                let tf_params = self.scenario6_params_for_tf(tf, None);

                let window = tf_params
                    .get("window_length")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(self.params.window_length);
                let kalman_r = tf_params
                    .get("kalman_r")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.kalman_r);
                let kalman_q = tf_params
                    .get("kalman_q")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.kalman_q);
                let bb_period = tf_params
                    .get("b_b_length")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(self.params.b_b_length);
                let std_factor = tf_params
                    .get("std_factor")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(self.params.std_factor);

                // TF-specific Kalman-Z
                reqs.push(IndicatorRequirement::with_timeframe(
                    format!("KALMAN_Z_{tf}"),
                    tf,
                    serde_json::json!({
                        "window": window,
                        "r": kalman_r,
                        "q": kalman_q
                    }),
                ));

                // TF-specific Bollinger
                reqs.push(IndicatorRequirement::with_timeframe(
                    format!("BOLLINGER_{tf}"),
                    tf,
                    serde_json::json!({
                        "period": bb_period,
                        "std_factor": std_factor
                    }),
                ));

                // TF-specific Close price
                reqs.push(IndicatorRequirement::with_timeframe(
                    format!("CLOSE_{tf}"),
                    tf,
                    serde_json::json!({}),
                ));
            }
        }

        reqs
    }

    fn required_htf_timeframes(&self) -> Vec<String> {
        let mut tfs = Vec::new();

        // Add primary HTF if filter is enabled
        if self.params.htf_filter != HtfFilter::None {
            tfs.push(self.params.htf_tf.clone());
        }

        if self.params.extra_htf_filter != HtfFilter::None
            && !is_htf_disabled(&self.params.extra_htf_tf)
            && !tfs.contains(&self.params.extra_htf_tf)
        {
            tfs.push(self.params.extra_htf_tf.clone());
        }

        // Add scenario 6 timeframes
        if self.params.enabled_scenarios.contains(&6) {
            for tf in &self.params.scenario6_timeframes {
                if !tfs.contains(tf) {
                    tfs.push(tf.clone());
                }
            }
        }

        tfs
    }

    fn reset(&mut self) {
        self.state = MrzState;
    }
}

fn is_htf_disabled(timeframe: &str) -> bool {
    let tf = timeframe.trim();
    tf.is_empty() || tf.eq_ignore_ascii_case("NONE")
}

fn normalize_timeframe_name(value: &str) -> String {
    value.trim().to_uppercase()
}

fn normalize_timeframe_list(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();

    for value in values {
        let tf = normalize_timeframe_name(value);
        if tf.is_empty() {
            continue;
        }
        if seen.insert(tf.clone()) {
            normalized.push(tf);
        }
    }

    normalized
}

fn normalize_scenario6_params(value: &serde_json::Value) -> serde_json::Value {
    let Some(map) = value.as_object() else {
        return value.clone();
    };

    let mut normalized = serde_json::Map::new();
    for (key, val) in map {
        if let Some(normalized_key) = normalize_scenario6_key(key) {
            normalized.insert(normalized_key, val.clone());
        }
    }

    serde_json::Value::Object(normalized)
}

fn normalize_scenario6_key(key: &str) -> Option<String> {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "*" {
        return Some("*".to_string());
    }

    if let Some((tf_part, dir_part)) = trimmed.split_once('.') {
        let tf_part = tf_part.trim();
        let dir_part = dir_part.trim();
        if tf_part.is_empty() || dir_part.is_empty() {
            return None;
        }
        let tf_normalized = if tf_part == "*" {
            "*".to_string()
        } else {
            normalize_timeframe_name(tf_part)
        };
        let dir_normalized = dir_part.to_ascii_lowercase();
        return Some(format!("{tf_normalized}.{dir_normalized}"));
    }

    Some(normalize_timeframe_name(trimmed))
}

fn find_scenario6_param_value<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(value) = map.get(key) {
        return Some(value);
    }

    map.iter().find_map(|(raw_key, value)| {
        normalize_scenario6_key(raw_key)
            .as_ref()
            .filter(|normalized| normalized.as_str() == key)
            .map(|_| value)
    })
}

fn is_direction_map(map: &serde_json::Map<String, serde_json::Value>) -> bool {
    map.keys().any(|key| {
        let trimmed = key.trim();
        trimmed.eq_ignore_ascii_case("long") || trimmed.eq_ignore_ascii_case("short")
    })
}

fn direction_object<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    direction: &str,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    map.iter().find_map(|(key, value)| {
        if key.trim().eq_ignore_ascii_case(direction) {
            value.as_object()
        } else {
            None
        }
    })
}

fn merge_params(
    target: &mut serde_json::Map<String, serde_json::Value>,
    source: &serde_json::Map<String, serde_json::Value>,
) {
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn kmeans_1d(values: &[f64], k: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique = sorted.clone();
    unique.dedup();
    let k = k.min(unique.len().max(1));
    if k == 1 {
        let mean = values.iter().sum::<f64>() / usize_to_f64(values.len());
        return Some((vec![mean], vec![0; values.len()]));
    }

    let mut centers: Vec<f64> = (1..=k)
        .map(|i| {
            let q = usize_to_f64(i) / (usize_to_f64(k) + 1.0);
            quantile_sorted(&sorted, q)
        })
        .collect();

    let mut labels = vec![0usize; values.len()];
    for _ in 0..30 {
        for (i, v) in values.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = (v - centers[0]).abs();
            for (j, c) in centers.iter().enumerate().skip(1) {
                let dist = (v - c).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            labels[i] = best_idx;
        }

        let mut sums = vec![0.0; k];
        let mut counts = vec![0usize; k];
        for (label, value) in labels.iter().zip(values.iter()) {
            sums[*label] += *value;
            counts[*label] += 1;
        }

        let mut new_centers = centers.clone();
        for j in 0..k {
            if counts[j] > 0 {
                new_centers[j] = sums[j] / usize_to_f64(counts[j]);
            }
        }

        let mut converged = true;
        for (a, b) in centers.iter().zip(new_centers.iter()) {
            if (a - b).abs() > 1e-6 {
                converged = false;
                break;
            }
        }
        centers = new_centers;
        if converged {
            break;
        }
    }

    Some((centers, labels))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let len_minus_one = sorted.len().saturating_sub(1);
    let pos = q.clamp(0.0, 1.0) * usize_to_f64(len_minus_one);
    let idx = pos.floor().clamp(0.0, usize_to_f64(len_minus_one)) as usize;
    let frac = pos - usize_to_f64(idx);
    let next = (idx + 1).min(sorted.len() - 1);
    sorted[idx] + frac * (sorted[next] - sorted[idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::HtfContext;
    use omega_indicators::{
        ATR, EMA, IndicatorCache, IndicatorParams, IndicatorSpec, MultiTfIndicatorCache, ZScore,
    };
    use omega_types::{Candle, Timeframe};
    use std::cell::RefCell;

    fn make_candle(close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            close_time_ns: 300_000_000_000 - 1,
            open: close - 0.001,
            high: close + 0.002,
            low: close - 0.002,
            close,
            volume: 100.0,
        }
    }

    fn make_multi_tf_cache(closes: &[f64]) -> RefCell<MultiTfIndicatorCache> {
        let bid: Vec<Candle> = closes.iter().map(|&c| make_candle(c)).collect();
        let ask: Vec<Candle> = closes.iter().map(|&c| make_candle(c + 0.0002)).collect();
        RefCell::new(MultiTfIndicatorCache::new(
            Timeframe::M5,
            bid,
            ask,
            Vec::new(),
        ))
    }

    fn make_candles(closes: &[f64]) -> Vec<Candle> {
        closes.iter().map(|&c| make_candle(c)).collect()
    }

    fn setup_cache(candles: &[Candle]) -> IndicatorCache {
        let mut cache = IndicatorCache::new();

        // Pre-compute required indicators
        let ema = EMA::new(20);
        let ema_spec = IndicatorSpec::new("EMA", IndicatorParams::Period(20));
        cache.get_or_compute(&ema_spec, candles, &ema);

        let atr = ATR::new(14);
        let atr_spec = IndicatorSpec::new("ATR", IndicatorParams::Period(14));
        cache.get_or_compute(&atr_spec, candles, &atr);

        let zscore = ZScore::new(20);
        let zscore_spec = IndicatorSpec::new("Z_SCORE", IndicatorParams::Period(20));
        cache.get_or_compute(&zscore_spec, candles, &zscore);

        cache
    }

    fn insert_series(cache: &mut IndicatorCache, spec: IndicatorSpec, idx: usize, value: f64) {
        let mut values = vec![0.0; idx + 1];
        values[idx] = value;
        cache.insert(spec, values);
    }

    fn insert_period(
        cache: &mut IndicatorCache,
        name: &str,
        period: usize,
        idx: usize,
        value: f64,
    ) {
        insert_series(
            cache,
            IndicatorSpec::new(name, IndicatorParams::Period(period)),
            idx,
            value,
        );
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn scale_to_u32(value: f64, scale: f64) -> u32 {
        if !value.is_finite() {
            return 0;
        }
        let scaled = (value * scale).round();
        let clamped = scaled.clamp(0.0, f64::from(u32::MAX));
        clamped as u32
    }

    #[derive(Copy, Clone)]
    struct BollingerValues {
        lower: f64,
        middle: f64,
        upper: f64,
    }

    #[derive(Copy, Clone)]
    struct KalmanGarchInsert {
        r: f64,
        q: f64,
        alpha: f64,
        beta: f64,
        omega: f64,
        use_log_returns: bool,
        scale: f64,
        min_periods: usize,
        sigma_floor: f64,
        value: f64,
    }

    fn insert_bollinger(
        cache: &mut IndicatorCache,
        name: &str,
        period: usize,
        std_factor: f64,
        idx: usize,
        values: BollingerValues,
    ) {
        let spec = IndicatorSpec::new(
            name,
            IndicatorParams::Bollinger {
                period,
                std_factor_x100: scale_to_u32(std_factor, 100.0),
            },
        );
        insert_series(cache, spec.with_output_suffix("lower"), idx, values.lower);
        insert_series(cache, spec.with_output_suffix("middle"), idx, values.middle);
        insert_series(cache, spec.with_output_suffix("upper"), idx, values.upper);
    }

    fn insert_kalman(
        cache: &mut IndicatorCache,
        name: &str,
        window: usize,
        r: f64,
        q: f64,
        idx: usize,
        value: f64,
    ) {
        insert_series(
            cache,
            IndicatorSpec::new(
                name,
                IndicatorParams::Kalman {
                    window,
                    r_x1000: scale_to_u32(r, 1000.0),
                    q_x1000: scale_to_u32(q, 1000.0),
                },
            ),
            idx,
            value,
        );
    }

    fn insert_kalman_garch(
        cache: &mut IndicatorCache,
        name: &str,
        window: usize,
        idx: usize,
        params: KalmanGarchInsert,
    ) {
        insert_series(
            cache,
            IndicatorSpec::new(
                name,
                IndicatorParams::KalmanGarch {
                    window,
                    r_x1000: scale_to_u32(params.r, 1000.0),
                    q_x1000: scale_to_u32(params.q, 1000.0),
                    alpha_x1000: scale_to_u32(params.alpha, 1000.0),
                    beta_x1000: scale_to_u32(params.beta, 1000.0),
                    omega_x1000000: scale_to_u32(params.omega, 1_000_000.0),
                    use_log_returns: params.use_log_returns,
                    scale_x100: scale_to_u32(params.scale, 100.0),
                    min_periods: params.min_periods,
                    sigma_floor_x1e8: scale_to_u32(params.sigma_floor, 1e8),
                },
            ),
            idx,
            params.value,
        );
    }

    #[test]
    fn test_mrz_from_params_valid() {
        let params = serde_json::json!({
            "ema_length": 20,
            "z_score_long": -2.0,
            "z_score_short": 2.0
        });

        let strategy = MeanReversionZScore::from_params(&params);
        assert!(strategy.is_ok());
    }

    #[test]
    fn test_mrz_from_params_invalid_zscore_long() {
        let params = serde_json::json!({
            "z_score_long": 1.0  // Must be negative
        });

        let strategy = MeanReversionZScore::from_params(&params);
        assert!(matches!(strategy, Err(StrategyError::InvalidParams(_))));
    }

    #[test]
    fn test_mrz_from_params_invalid_zscore_short() {
        let params = serde_json::json!({
            "z_score_long": -2.0,
            "z_score_short": -1.0  // Must be positive
        });

        let strategy = MeanReversionZScore::from_params(&params);
        assert!(matches!(strategy, Err(StrategyError::InvalidParams(_))));
    }

    #[test]
    fn test_mrz_name() {
        let strategy = MeanReversionZScore::new(MrzParams::default());
        assert_eq!(strategy.name(), "mean_reversion_z_score");
    }

    #[test]
    fn test_mrz_required_indicators() {
        let params = MrzParams {
            enabled_scenarios: vec![1, 2, 4, 5],
            htf_filter: HtfFilter::Above,
            ..Default::default()
        };

        let strategy = MeanReversionZScore::new(params);
        let reqs = strategy.required_indicators();

        // Should include EMA, ATR, Z_SCORE, BOLLINGER, KALMAN_Z, GARCH_VOL, HTF EMA
        assert!(reqs.iter().any(|r| r.name == "EMA"));
        assert!(reqs.iter().any(|r| r.name == "ATR"));
        assert!(reqs.iter().any(|r| r.name == "Z_SCORE"));
        assert!(reqs.iter().any(|r| r.name == "BOLLINGER"));
        assert!(reqs.iter().any(|r| r.name == "KALMAN_Z"));
        assert!(reqs.iter().any(|r| r.name == "GARCH_VOL"));
        assert!(reqs.iter().any(|r| r.timeframe.is_some())); // HTF
    }

    #[test]
    fn test_mrz_session_gate() {
        let params = MrzParams::default();
        let mut strategy = MeanReversionZScore::new(params);

        // Create a downtrend (should trigger long)
        let closes: Vec<f64> = (0..50)
            .map(|i| 1.1000 - f64::from(i) * 0.001)
            .collect();
        let candles = make_candles(&closes);
        let cache = setup_cache(&candles);

        let bid = &candles[49];
        let ask = make_candle(bid.close + 0.0002);

        // Session closed - should not generate signal
        let ctx = BarContext::new(49, 1_234_567_890, bid, &ask, &cache).with_session(false);

        let signal = strategy.on_bar(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_mrz_news_gate() {
        let params = MrzParams::default();
        let mut strategy = MeanReversionZScore::new(params);

        let closes: Vec<f64> = (0..50)
            .map(|i| 1.1000 - f64::from(i) * 0.001)
            .collect();
        let candles = make_candles(&closes);
        let cache = setup_cache(&candles);

        let bid = &candles[49];
        let ask = make_candle(bid.close + 0.0002);

        // News blocked - should not generate signal
        let ctx = BarContext::new(49, 1_234_567_890, bid, &ask, &cache).with_news_blocked(true);

        let signal = strategy.on_bar(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_mrz_direction_filter_long_only() {
        let params = MrzParams {
            direction_filter: DirectionFilter::Long,
            z_score_short: 0.5,
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        // Uptrend (should trigger short normally, but filter blocks)
        let closes: Vec<f64> = (0..50)
            .map(|i| 1.0000 + f64::from(i) * 0.001)
            .collect();
        let candles = make_candles(&closes);
        let cache = setup_cache(&candles);

        let bid = &candles[49];
        let ask = make_candle(bid.close + 0.0002);

        let ctx = BarContext::new(49, 1_234_567_890, bid, &ask, &cache);

        // Scenario 1 short should be blocked by direction filter
        let signal = strategy.scenario_1(&ctx);
        assert!(signal.is_none() || signal.as_ref().map(|s| s.direction) == Some(Direction::Long));
    }

    #[test]
    fn test_mrz_reset() {
        let mut strategy = MeanReversionZScore::new(MrzParams::default());
        let before = format!("{:?}", strategy.state);
        strategy.reset();
        let after = format!("{:?}", strategy.state);

        assert_eq!(before, after);
    }

    #[test]
    fn test_htf_filter_enum_serde() {
        let filter = HtfFilter::Above;
        let json = serde_json::to_string(&filter).unwrap();
        assert_eq!(json, "\"above\"");

        let deserialized: HtfFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, HtfFilter::Above);
    }

    #[test]
    fn test_direction_filter_enum_serde() {
        let filter = DirectionFilter::Short;
        let json = serde_json::to_string(&filter).unwrap();
        assert_eq!(json, "\"short\"");
    }

    #[test]
    fn test_scenario6_mode_as_str() {
        assert_eq!(Scenario6Mode::All.as_str(), "all");
        assert_eq!(Scenario6Mode::Any.as_str(), "any");
    }

    #[test]
    fn test_mrz_params_default() {
        let params = MrzParams::default();
        assert_eq!(params.ema_length, 20);
        assert_eq!(params.atr_length, 14);
        assert!((params.z_score_long + 2.0).abs() < 1e-10);
        assert!((params.z_score_short - 2.0).abs() < 1e-10);
        assert_eq!(params.cluster_hysteresis_bars, 1);
        assert_eq!(
            params.intraday_vol_allowed,
            vec!["low".to_string(), "mid".to_string()]
        );
        assert!(!params.use_position_manager);
        assert_eq!(params.max_holding_minutes, 0);
        assert_eq!(params.enabled_scenarios, vec![1]);
    }

    #[test]
    fn test_htf_filter_above_allows_when_price_above_ema() {
        let ema_len = 2;
        let mut htf_cache = IndicatorCache::new();
        insert_period(&mut htf_cache, "EMA", ema_len, 0, 1.1000);

        let htf_bid = make_candle(1.1050);
        let htf_ask = make_candle(1.1052);
        let htf_ctx = HtfContext::new(&htf_bid, &htf_ask, &htf_cache, 0, "H1");

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();
        let ctx = BarContext::new(0, 1_000_000_000, &bid, &ask, &cache).with_htf(htf_ctx);

        assert!(MeanReversionZScore::check_htf_filter(
            &ctx,
            "H1",
            ema_len,
            HtfFilter::Above,
        ));
    }

    #[test]
    fn test_htf_filter_below_allows_when_price_below_ema() {
        let ema_len = 2;
        let mut htf_cache = IndicatorCache::new();
        insert_period(&mut htf_cache, "EMA", ema_len, 0, 1.2000);

        let htf_bid = make_candle(1.1950);
        let htf_ask = make_candle(1.1952);
        let htf_ctx = HtfContext::new(&htf_bid, &htf_ask, &htf_cache, 0, "H1");

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();
        let ctx = BarContext::new(0, 1_000_000_000, &bid, &ask, &cache).with_htf(htf_ctx);

        assert!(MeanReversionZScore::check_htf_filter(
            &ctx,
            "H1",
            ema_len,
            HtfFilter::Below,
        ));
    }

    #[test]
    fn test_htf_filter_both_and_none_are_pass_through() {
        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();
        let ctx = BarContext::new(0, 1_000_000_000, &bid, &ask, &cache);

        assert!(MeanReversionZScore::check_htf_filter(
            &ctx,
            "H1",
            200,
            HtfFilter::Both,
        ));
        assert!(MeanReversionZScore::check_htf_filter(
            &ctx,
            "H1",
            200,
            HtfFilter::None,
        ));
    }

    #[test]
    fn test_htf_bias_requires_both_filters() {
        let ema_len = 2;
        let mut htf_cache = IndicatorCache::new();
        insert_period(&mut htf_cache, "EMA", ema_len, 0, 1.1000);

        let htf_bid = make_candle(1.1050);
        let htf_ask = make_candle(1.1052);
        let htf_ctx = HtfContext::new(&htf_bid, &htf_ask, &htf_cache, 0, "H1");

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();
        let ctx = BarContext::new(0, 1_000_000_000, &bid, &ask, &cache).with_htf(htf_ctx);

        let params = MrzParams {
            htf_tf: "H1".to_string(),
            htf_ema: ema_len,
            htf_filter: HtfFilter::Above,
            extra_htf_tf: "H1".to_string(),
            extra_htf_ema: ema_len,
            extra_htf_filter: HtfFilter::Below,
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        assert!(!strategy.check_htf_bias(&ctx));
    }

    #[test]
    fn test_scenario6_timeframe_normalization_and_param_keys() {
        let params = serde_json::json!({
            "scenario6_timeframes": [" h1 ", "H1"],
            "scenario6_params": {
                "h1": {"z_score_long": -1.5}
            }
        });

        let strategy = MeanReversionZScore::from_params(&params).expect("valid params");

        assert_eq!(strategy.params.scenario6_timeframes, vec!["H1".to_string()]);
        assert!(strategy.params.scenario6_params.get("H1").is_some());
    }

    #[test]
    fn test_scenario6_params_wildcard_fallback() {
        let params = MrzParams {
            scenario6_timeframes: vec!["H1".to_string()],
            scenario6_params: serde_json::json!({
                "*": {"z_score_long": -1.25}
            }),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -2.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1000,
                middle: 1.1010,
                upper: 1.1020,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.0990);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let chain = strategy.check_multi_tf_signals_v2(&ctx, true);
        assert_eq!(chain.len(), 1);
        assert!((chain[0].threshold + 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_scenario6_params_specific_overrides_wildcard() {
        let params = MrzParams {
            scenario6_timeframes: vec!["H1".to_string()],
            scenario6_params: serde_json::json!({
                "*": {"z_score_long": -1.0},
                "H1": {"z_score_long": -3.0}
            }),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -4.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1000,
                middle: 1.1010,
                upper: 1.1020,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.0990);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let chain = strategy.check_multi_tf_signals_v2(&ctx, true);
        assert_eq!(chain.len(), 1);
        assert!((chain[0].threshold + 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scenario6_params_direction_override_priority() {
        let params = MrzParams {
            scenario6_timeframes: vec!["H1".to_string()],
            scenario6_params: serde_json::json!({
                "*": {"z_score_long": -1.0},
                "*.Long": {"z_score_long": -1.5},
                "h1": {"z_score_long": -2.0},
                "H1.Long": {"z_score_long": -3.0}
            }),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -4.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1000,
                middle: 1.1010,
                upper: 1.1020,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.0990);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let chain = strategy.check_multi_tf_signals_v2(&ctx, true);
        assert_eq!(chain.len(), 1);
        assert!((chain[0].threshold + 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scenario6_params_direction_tf_overrides_wildcard_dir() {
        let params = MrzParams {
            scenario6_timeframes: vec!["H1".to_string()],
            scenario6_params: serde_json::json!({
                "*.long": {"z_score_long": -1.5},
                "H1": {"z_score_long": -2.5}
            }),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1000,
                middle: 1.1010,
                upper: 1.1020,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.0990);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let chain = strategy.check_multi_tf_signals_v2(&ctx, true);
        assert_eq!(chain.len(), 1);
        assert!((chain[0].threshold + 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_scenario6_params_wildcard_direction_without_tf_key() {
        let params = MrzParams {
            scenario6_timeframes: vec!["H1".to_string()],
            scenario6_params: serde_json::json!({
                "*": {"z_score_long": -0.75},
                "*.long": {"z_score_long": -1.75}
            }),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -2.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1000,
                middle: 1.1010,
                upper: 1.1020,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.0990);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let chain = strategy.check_multi_tf_signals_v2(&ctx, true);
        assert_eq!(chain.len(), 1);
        assert!((chain[0].threshold + 1.75).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_1_long_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![1],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_period(&mut cache, "Z_SCORE", 20, idx, -3.0);
        insert_period(&mut cache, "EMA", 20, idx, 1.1050);
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_1(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.scenario_id, 1);
        assert!((signal.take_profit - 1.1050).abs() < 1e-10);
        assert!((signal.entry_price - ask.close).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_1_long_no_signal_when_zscore_above_threshold() {
        let params = MrzParams {
            enabled_scenarios: vec![1],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_period(&mut cache, "Z_SCORE", 20, idx, -1.9);
        insert_period(&mut cache, "EMA", 20, idx, 1.1050);
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_1(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_1_short_no_signal_when_zscore_below_threshold() {
        let params = MrzParams {
            enabled_scenarios: vec![1],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_period(&mut cache, "Z_SCORE", 20, idx, 1.9);
        insert_period(&mut cache, "EMA", 20, idx, 1.0950);
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_1(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_2_short_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![2],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, 3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0980,
                middle: 1.0990,
                upper: 1.1005,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1010);
        let ask = make_candle(1.1012);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_2(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Short);
        assert_eq!(signal.scenario_id, 2);
        assert!((signal.take_profit - 1.0990).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_2_long_no_signal_when_kalman_z_above_threshold() {
        let params = MrzParams {
            enabled_scenarios: vec![2],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -1.9);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1000,
                middle: 1.1010,
                upper: 1.1020,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.0990);
        let ask = make_candle(1.0992);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_2(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_2_short_no_signal_when_price_below_upper_band() {
        let params = MrzParams {
            enabled_scenarios: vec![2],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, 2.1);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0980,
                middle: 1.0990,
                upper: 1.1005,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_2(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_3_tp_min_distance_respected() {
        let params = MrzParams {
            enabled_scenarios: vec![3],
            tp_min_distance: 0.0010,
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);
        insert_period(&mut cache, "EMA", 20, idx, 1.1050);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_3(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.scenario_id, 3);
        assert!((signal.take_profit - 1.1050).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_3_blocks_when_tp_distance_too_small() {
        let params = MrzParams {
            enabled_scenarios: vec![3],
            tp_min_distance: 0.0010,
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);
        insert_period(&mut cache, "EMA", 20, idx, 1.1011);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_3(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_3_short_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![3],
            tp_min_distance: 0.0005,
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, 3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0950,
                middle: 1.0960,
                upper: 1.0975,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);
        insert_period(&mut cache, "EMA", 20, idx, 1.0950);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_3(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Short);
        assert_eq!(signal.scenario_id, 3);
        assert!((signal.take_profit - 1.0950).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_3_short_blocks_when_tp_distance_too_small() {
        let params = MrzParams {
            enabled_scenarios: vec![3],
            tp_min_distance: 0.0010,
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, 3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0950,
                middle: 1.0960,
                upper: 1.0975,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);
        insert_period(&mut cache, "EMA", 20, idx, 1.0998);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_3(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_4_long_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![4],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman_garch(
            &mut cache,
            "KALMAN_GARCH_Z",
            20,
            idx,
            KalmanGarchInsert {
                r: 1.0,
                q: 0.1,
                alpha: 0.1,
                beta: 0.8,
                omega: 0.00001,
                use_log_returns: true,
                scale: 100.0,
                min_periods: 50,
                sigma_floor: 1e-6,
                value: -3.0,
            },
        );
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_4(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.scenario_id, 4);
        assert!((signal.take_profit - 1.1020).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_4_short_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![4],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman_garch(
            &mut cache,
            "KALMAN_GARCH_Z",
            20,
            idx,
            KalmanGarchInsert {
                r: 1.0,
                q: 0.1,
                alpha: 0.1,
                beta: 0.8,
                omega: 0.00001,
                use_log_returns: true,
                scale: 100.0,
                min_periods: 50,
                sigma_floor: 1e-6,
                value: 3.0,
            },
        );
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0980,
                middle: 1.0990,
                upper: 1.1005,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1010);
        let ask = make_candle(1.1012);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_4(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Short);
        assert_eq!(signal.scenario_id, 4);
        assert!((signal.take_profit - 1.0990).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_4_blocks_when_zscore_not_met() {
        let params = MrzParams {
            enabled_scenarios: vec![4],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman_garch(
            &mut cache,
            "KALMAN_GARCH_Z",
            20,
            idx,
            KalmanGarchInsert {
                r: 1.0,
                q: 0.1,
                alpha: 0.1,
                beta: 0.8,
                omega: 0.00001,
                use_log_returns: true,
                scale: 100.0,
                min_periods: 50,
                sigma_floor: 1e-6,
                value: -1.0,
            },
        );
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_4(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_5_blocks_disallowed_vol_cluster() {
        let params = MrzParams {
            enabled_scenarios: vec![5],
            intraday_vol_feature: "garch_forecast".to_string(),
            intraday_vol_cluster_window: 3,
            intraday_vol_cluster_k: 1,
            intraday_vol_min_points: 1,
            intraday_vol_log_transform: false,
            intraday_vol_garch_lookback: 3,
            intraday_vol_allowed: vec!["high".to_string()],
            cluster_hysteresis_bars: 1,
            garch_min_periods: 1,
            garch_scale: 1.0,
            garch_use_log_returns: false,
            atr_length: 2,
            ..Default::default()
        };
        let mut strategy = MeanReversionZScore::new(params);

        let closes = vec![1.1000, 1.1002, 1.1004, 1.1006];
        let idx = closes.len() - 1;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 2, idx, 0.0010);

        let bid = make_candle(closes[idx]);
        let ask = make_candle(closes[idx] + 0.0002);
        let multi_tf = make_multi_tf_cache(&closes);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache).with_multi_tf(&multi_tf);

        let signal = strategy.scenario_5(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_5_allows_vol_cluster() {
        let params = MrzParams {
            enabled_scenarios: vec![5],
            intraday_vol_feature: "garch_forecast".to_string(),
            intraday_vol_cluster_window: 3,
            intraday_vol_cluster_k: 1,
            intraday_vol_min_points: 1,
            intraday_vol_log_transform: false,
            intraday_vol_garch_lookback: 3,
            intraday_vol_allowed: vec!["low".to_string()],
            cluster_hysteresis_bars: 1,
            garch_min_periods: 1,
            garch_scale: 1.0,
            garch_use_log_returns: false,
            atr_length: 2,
            ..Default::default()
        };
        let mut strategy = MeanReversionZScore::new(params);

        let closes = vec![1.1000, 1.1002, 1.1004, 1.1006];
        let idx = closes.len() - 1;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 2, idx, 0.0010);

        let bid = make_candle(closes[idx]);
        let ask = make_candle(closes[idx] + 0.0002);
        let multi_tf = make_multi_tf_cache(&closes);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache).with_multi_tf(&multi_tf);

        let signal = strategy.scenario_5(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.scenario_id, 5);
    }

    #[test]
    fn test_scenario_5_short_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![5],
            intraday_vol_feature: "garch_forecast".to_string(),
            intraday_vol_cluster_window: 3,
            intraday_vol_cluster_k: 1,
            intraday_vol_min_points: 1,
            intraday_vol_log_transform: false,
            intraday_vol_garch_lookback: 3,
            intraday_vol_allowed: vec!["low".to_string()],
            cluster_hysteresis_bars: 1,
            garch_min_periods: 1,
            garch_scale: 1.0,
            garch_use_log_returns: false,
            atr_length: 2,
            ..Default::default()
        };
        let mut strategy = MeanReversionZScore::new(params);

        let closes = vec![1.1000, 1.1002, 1.1004, 1.1006];
        let idx = closes.len() - 1;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, 3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0980,
                middle: 1.0990,
                upper: 1.1005,
            },
        );
        insert_period(&mut cache, "ATR", 2, idx, 0.0010);

        let bid = make_candle(1.1010);
        let ask = make_candle(1.1012);
        let multi_tf = make_multi_tf_cache(&closes);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache).with_multi_tf(&multi_tf);

        let signal = strategy.scenario_5(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Short);
        assert_eq!(signal.scenario_id, 5);
        assert!((signal.take_profit - 1.0990).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_5_blocks_on_hysteresis() {
        let params = MrzParams {
            enabled_scenarios: vec![5],
            intraday_vol_feature: "garch_forecast".to_string(),
            intraday_vol_cluster_window: 3,
            intraday_vol_cluster_k: 1,
            intraday_vol_min_points: 1,
            intraday_vol_log_transform: false,
            intraday_vol_garch_lookback: 3,
            cluster_hysteresis_bars: 5,
            garch_min_periods: 1,
            garch_scale: 1.0,
            garch_use_log_returns: false,
            atr_length: 2,
            ..Default::default()
        };
        let mut strategy = MeanReversionZScore::new(params);

        let closes = vec![1.1000, 1.1002, 1.1004, 1.1006];
        let idx = closes.len() - 1;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 2, idx, 0.0010);

        let bid = make_candle(closes[idx]);
        let ask = make_candle(closes[idx] + 0.0002);
        let multi_tf = make_multi_tf_cache(&closes);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache).with_multi_tf(&multi_tf);

        let signal = strategy.scenario_5(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_vol_cluster_guard_blocks_unsupported_feature() {
        let params = MrzParams {
            intraday_vol_feature: "unsupported_feature".to_string(),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();
        let ctx = BarContext::new(0, 1_000_000_000, &bid, &ask, &cache);

        let meta = strategy
            .vol_cluster_guard(&ctx)
            .expect("expected meta for unsupported feature");
        assert_eq!(
            meta.get("status").and_then(|v| v.as_str()),
            Some("feature_not_supported")
        );
    }

    #[test]
    fn test_scenario_6_multi_tf_all_agreement() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::All,
            scenario6_timeframes: vec!["H1".to_string()],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.1000);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.scenario_id, 6);
    }

    #[test]
    fn test_scenario_6_chain_meta_contains_required_fields() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::All,
            scenario6_timeframes: vec!["H1".to_string()],
            scenario6_params: serde_json::json!({
                "H1.long": {"z_score_long": -3.0}
            }),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -4.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.1000);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx).expect("expected signal");
        let meta = signal.meta;
        let scenario6 = meta
            .get("scenario6")
            .and_then(|v| v.as_object())
            .expect("scenario6 meta");
        let chain = scenario6
            .get("chain")
            .and_then(|v| v.as_array())
            .expect("chain array");
        let first = chain.first().and_then(|v| v.as_object()).expect("chain item");

        for key in [
            "tf",
            "ok",
            "status",
            "kalman_z",
            "threshold",
            "price",
            "upper",
            "lower",
            "params",
        ] {
            assert!(first.contains_key(key), "missing chain key {key}");
        }

        let params_obj = first
            .get("params")
            .and_then(|v| v.as_object())
            .expect("params object");
        let z_long = params_obj
            .get("z_score_long")
            .and_then(serde_json::Value::as_f64)
            .expect("z_score_long");
        assert!((z_long + 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scenario_6_blocks_when_tf_disagrees() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::All,
            scenario6_timeframes: vec!["H1".to_string()],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -1.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.1000);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_6_any_allows_partial_agreement() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::Any,
            scenario6_timeframes: vec!["H1".to_string(), "H4".to_string()],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.1000);

        insert_kalman(&mut cache, "KALMAN_Z_H4", 20, 1.0, 0.1, idx, -1.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H4",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H4", 1, idx, 1.1000);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Long);
        assert_eq!(signal.scenario_id, 6);
        assert!(signal.tags.iter().any(|t| t == "multi_tf_any"));
    }

    #[test]
    fn test_scenario_6_short_signal() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::All,
            scenario6_timeframes: vec!["H1".to_string()],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, 3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0980,
                middle: 1.0990,
                upper: 1.1005,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, 3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.0980,
                middle: 1.0990,
                upper: 1.1005,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.1010);

        let bid = make_candle(1.1010);
        let ask = make_candle(1.1012);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx).expect("expected signal");
        assert_eq!(signal.direction, Direction::Short);
        assert_eq!(signal.scenario_id, 6);
    }

    #[test]
    fn test_scenario_6_any_blocks_when_no_tf_agrees() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::Any,
            scenario6_timeframes: vec!["H1".to_string(), "H4".to_string()],
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        insert_kalman(&mut cache, "KALMAN_Z_H1", 20, 1.0, 0.1, idx, -1.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H1",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H1", 1, idx, 1.1000);

        insert_kalman(&mut cache, "KALMAN_Z_H4", 20, 1.0, 0.1, idx, -1.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER_H4",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "CLOSE_H4", 1, idx, 1.1000);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx);
        assert!(signal.is_none());
    }

    #[test]
    fn test_scenario_6_blocks_when_timeframes_empty() {
        let params = MrzParams {
            enabled_scenarios: vec![6],
            scenario6_mode: Scenario6Mode::All,
            scenario6_timeframes: Vec::new(),
            ..Default::default()
        };
        let strategy = MeanReversionZScore::new(params);

        let idx = 0;
        let mut cache = IndicatorCache::new();
        insert_kalman(&mut cache, "KALMAN_Z", 20, 1.0, 0.1, idx, -3.0);
        insert_bollinger(
            &mut cache,
            "BOLLINGER",
            20,
            2.0,
            idx,
            BollingerValues {
                lower: 1.1010,
                middle: 1.1020,
                upper: 1.1030,
            },
        );
        insert_period(&mut cache, "ATR", 14, idx, 0.0010);

        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let ctx = BarContext::new(idx, 1_000_000_000, &bid, &ask, &cache);

        let signal = strategy.scenario_6(&ctx);
        assert!(signal.is_none());
    }
}
