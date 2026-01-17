//! Bar context for strategy execution
//!
//! Provides a read-only snapshot of market data and indicators
//! for the `Strategy::on_bar()` method.

use omega_indicators::{IndicatorCache, IndicatorParams, IndicatorSpec, MultiTfIndicatorCache};
use omega_types::{Candle, PriceType, Timeframe};
use std::cell::RefCell;
use std::str::FromStr;

/// Read-only context for strategy bar processing.
///
/// Contains all data a strategy needs to make trading decisions:
/// - Current bar data (bid/ask)
/// - Pre-computed indicator values
/// - HTF (higher timeframe) data
/// - Session and news state
///
/// # Example
/// ```ignore
/// fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal> {
///     let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": 20}))?;
///     let htf_ema = ctx.get_htf_indicator("EMA", &json!({"period": 200}))?;
///     // ...
/// }
/// ```
pub struct BarContext<'a> {
    /// Current bar index in the data series
    pub idx: usize,
    /// Timestamp in nanoseconds (bar open time)
    pub timestamp_ns: i64,
    /// Bid candle for current bar
    pub bid: &'a Candle,
    /// Ask candle for current bar
    pub ask: &'a Candle,
    /// Pre-computed indicator cache (primary timeframe)
    pub indicators: &'a IndicatorCache,
    /// Multi-timeframe indicator cache (`price_type` + stepwise aware)
    pub multi_tf: Option<&'a RefCell<MultiTfIndicatorCache>>,
    /// HTF data context (if available)
    pub htf_data: Option<HtfContext<'a>>,
    /// Whether trading session is open
    pub session_open: bool,
    /// Whether news filter is blocking trading
    pub news_blocked: bool,
}

/// Higher timeframe context.
///
/// Contains HTF candle data and indicators for strategies that
/// use multi-timeframe analysis.
pub struct HtfContext<'a> {
    /// HTF bid candle (last completed bar)
    pub bid: &'a Candle,
    /// HTF ask candle (last completed bar)
    pub ask: &'a Candle,
    /// HTF indicator cache
    pub indicators: &'a IndicatorCache,
    /// HTF bar index
    pub idx: usize,
    /// HTF timeframe name (e.g., "H4")
    pub timeframe: &'a str,
}

impl<'a> BarContext<'a> {
    /// Creates a new `BarContext`.
    #[must_use]
    pub fn new(
        idx: usize,
        timestamp_ns: i64,
        bid: &'a Candle,
        ask: &'a Candle,
        indicators: &'a IndicatorCache,
    ) -> Self {
        Self {
            idx,
            timestamp_ns,
            bid,
            ask,
            indicators,
            multi_tf: None,
            htf_data: None,
            session_open: true,
            news_blocked: false,
        }
    }

    /// Sets multi-timeframe cache.
    #[must_use]
    pub fn with_multi_tf(mut self, cache: &'a RefCell<MultiTfIndicatorCache>) -> Self {
        self.multi_tf = Some(cache);
        self
    }

    /// Sets HTF context.
    #[must_use]
    pub fn with_htf(mut self, htf: HtfContext<'a>) -> Self {
        self.htf_data = Some(htf);
        self
    }

    /// Sets session status.
    #[must_use]
    pub fn with_session(mut self, open: bool) -> Self {
        self.session_open = open;
        self
    }

    /// Sets news blocked status.
    #[must_use]
    pub fn with_news_blocked(mut self, blocked: bool) -> Self {
        self.news_blocked = blocked;
        self
    }

    /// Gets an indicator value for the current bar.
    ///
    /// Looks up the indicator in the cache by name and params.
    /// Returns `None` if the indicator is not found or the value is NaN.
    ///
    /// # Arguments
    /// * `name` - Indicator name (e.g., "EMA", "`Z_SCORE`")
    /// * `params` - JSON parameters for the indicator
    ///
    /// # Example
    /// ```ignore
    /// let ema = ctx.get_indicator("EMA", &json!({"period": 20}))?;
    /// ```
    #[must_use]
    pub fn get_indicator(&self, name: &str, params: &serde_json::Value) -> Option<f64> {
        if Self::is_zscore_ema(name, params) {
            return self.get_zscore_ema(PriceType::Bid, params);
        }
        let spec = Self::params_to_spec(name, params)?;
        let value = self.indicators.get_at(&spec, self.idx)?;
        if value.is_nan() { None } else { Some(value) }
    }

    /// Gets an HTF indicator value for the last completed bar.
    ///
    /// Uses the HTF context's bar index to prevent lookahead bias.
    /// Returns `None` if HTF data is not available or value is NaN.
    ///
    /// # Arguments
    /// * `name` - Indicator name
    /// * `params` - JSON parameters
    #[must_use]
    pub fn get_htf_indicator(&self, name: &str, params: &serde_json::Value) -> Option<f64> {
        let htf = self.htf_data.as_ref()?;
        if let Some(cache) = self.multi_tf {
            let tf = Timeframe::from_str(htf.timeframe).ok()?;
            let name_upper = name.to_uppercase();
            if name_upper == "EMA" {
                let period = params
                    .get("period")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let series = cache.borrow_mut().ema_stepwise(tf, PriceType::Bid, period);
                return value_at(&series, self.idx);
            }
        }
        let spec = Self::params_to_spec(name, params)?;
        let value = htf.indicators.get_at(&spec, htf.idx)?;
        if value.is_nan() { None } else { Some(value) }
    }

    /// Gets a multi-output indicator value (e.g., Bollinger upper band).
    ///
    /// # Arguments
    /// * `name` - Base indicator name (e.g., "BOLLINGER")
    /// * `output` - Output name (e.g., "upper", "middle", "lower")
    /// * `params` - JSON parameters
    #[must_use]
    pub fn get_indicator_output(
        &self,
        name: &str,
        output: &str,
        params: &serde_json::Value,
    ) -> Option<f64> {
        let spec = Self::params_to_spec(name, params)?;
        let full_spec = spec.with_output_suffix(output);
        let value = self.indicators.get_at(&full_spec, self.idx)?;
        if value.is_nan() { None } else { Some(value) }
    }

    /// Gets an HTF multi-output indicator value.
    #[must_use]
    pub fn get_htf_indicator_output(
        &self,
        name: &str,
        output: &str,
        params: &serde_json::Value,
    ) -> Option<f64> {
        let htf = self.htf_data.as_ref()?;
        if let Some(cache) = self.multi_tf {
            let tf = Timeframe::from_str(htf.timeframe).ok()?;
            let name_upper = name.to_uppercase();
            if name_upper == "BOLLINGER" {
                let period = params
                    .get("period")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let std_factor = params
                    .get("std_factor")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(2.0);
                let (upper, middle, lower) =
                    cache
                        .borrow_mut()
                        .bollinger_stepwise(tf, PriceType::Bid, period, std_factor);
                return match output {
                    "upper" => value_at(&upper, self.idx),
                    "middle" => value_at(&middle, self.idx),
                    "lower" => value_at(&lower, self.idx),
                    _ => None,
                };
            }
        }
        let spec = Self::params_to_spec(name, params)?;
        let full_spec = spec.with_output_suffix(output);
        let value = htf.indicators.get_at(&full_spec, htf.idx)?;
        if value.is_nan() { None } else { Some(value) }
    }

    /// Converts JSON params to `IndicatorSpec`.
    ///
    /// Maps common JSON parameter patterns to the hashable `IndicatorParams` enum.
    #[allow(clippy::too_many_lines)]
    fn params_to_spec(name: &str, params: &serde_json::Value) -> Option<IndicatorSpec> {
        let name_upper = name.to_uppercase();
        let base_name = name_upper
            .rsplit_once('_')
            .and_then(|(base, suffix)| {
                let bytes = suffix.as_bytes();
                if bytes.len() < 2 {
                    return None;
                }
                let prefix = bytes[0] as char;
                if !matches!(prefix, 'M' | 'H' | 'D' | 'W') {
                    return None;
                }
                if bytes[1..].iter().all(u8::is_ascii_digit) {
                    Some(base)
                } else {
                    None
                }
            })
            .unwrap_or(name_upper.as_str());

        let indicator_params = match base_name {
            "EMA" | "SMA" | "ATR" | "Z_SCORE" => {
                let period = params.get("period").or_else(|| params.get("window"))?;
                let period = period.as_u64().and_then(u64_to_usize)?;
                IndicatorParams::Period(period)
            }
            "BOLLINGER" => {
                let period = params
                    .get("period")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let std_factor = params
                    .get("std_factor")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(2.0);
                IndicatorParams::Bollinger {
                    period,
                    std_factor_x100: scale_to_u32(std_factor, 100.0),
                }
            }
            "KALMAN_Z" | "KALMAN_ZSCORE" => {
                let window = params
                    .get("window")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let r = params
                    .get("r")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1.0);
                let q = params
                    .get("q")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.1);
                IndicatorParams::Kalman {
                    window,
                    r_x1000: scale_to_u32(r, 1000.0),
                    q_x1000: scale_to_u32(q, 1000.0),
                }
            }
            "GARCH_VOL" | "GARCH" => {
                let alpha = params
                    .get("alpha")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.1);
                let beta = params
                    .get("beta")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.8);
                let omega = params
                    .get("omega")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.00001);
                let use_log_returns = params
                    .get("use_log_returns")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true);
                let scale = params
                    .get("scale")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(100.0);
                let min_periods = params
                    .get("min_periods")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)
                    .unwrap_or(20);
                let sigma_floor = params
                    .get("sigma_floor")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0001);
                IndicatorParams::Garch {
                    alpha_x1000: scale_to_u32(alpha, 1000.0),
                    beta_x1000: scale_to_u32(beta, 1000.0),
                    omega_x1000000: scale_to_u32(omega, 1_000_000.0),
                    use_log_returns,
                    scale_x100: scale_to_u32(scale, 100.0),
                    min_periods,
                    sigma_floor_x1e8: scale_to_u32(sigma_floor, 1e8),
                }
            }
            "KALMAN_GARCH" | "KALMAN_GARCH_Z" => {
                let window = params
                    .get("window")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let r = params
                    .get("r")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1.0);
                let q = params
                    .get("q")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.1);
                let alpha = params
                    .get("alpha")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.1);
                let beta = params
                    .get("beta")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.8);
                let omega = params
                    .get("omega")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.00001);
                let use_log_returns = params
                    .get("use_log_returns")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true);
                let scale = params
                    .get("scale")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1.0);
                let min_periods = params
                    .get("min_periods")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)
                    .unwrap_or(20);
                let sigma_floor = params
                    .get("sigma_floor")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1e-8);
                IndicatorParams::KalmanGarch {
                    window,
                    r_x1000: scale_to_u32(r, 1000.0),
                    q_x1000: scale_to_u32(q, 1000.0),
                    alpha_x1000: scale_to_u32(alpha, 1000.0),
                    beta_x1000: scale_to_u32(beta, 1000.0),
                    omega_x1000000: scale_to_u32(omega, 1_000_000.0),
                    use_log_returns,
                    scale_x100: scale_to_u32(scale, 100.0),
                    min_periods,
                    sigma_floor_x1e8: scale_to_u32(sigma_floor, 1e8),
                }
            }
            "VOL_CLUSTER" => {
                let vol_period = params.get("window").or_else(|| params.get("vol_period"))?;
                let high_thresh = params
                    .get("high_vol_threshold")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1.5);
                let low_thresh = params
                    .get("low_vol_threshold")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.5);
                let vol_period = vol_period.as_u64().and_then(u64_to_usize)?;
                IndicatorParams::VolCluster {
                    vol_period,
                    high_vol_threshold_x100: scale_to_u32(high_thresh, 100.0),
                    low_vol_threshold_x100: scale_to_u32(low_thresh, 100.0),
                }
            }
            "CLOSE" => IndicatorParams::Period(1),
            _ => {
                // Fallback: try period-based
                if let Some(period) = params.get("period").or_else(|| params.get("window")) {
                    let period = period.as_u64().and_then(u64_to_usize)?;
                    IndicatorParams::Period(period)
                } else {
                    return None;
                }
            }
        };

        Some(IndicatorSpec::new(name_upper, indicator_params))
    }

    fn parse_timeframe(timeframe: &str) -> Option<Timeframe> {
        Timeframe::from_str(timeframe).ok()
    }

    fn is_zscore_ema(name: &str, params: &serde_json::Value) -> bool {
        name.eq_ignore_ascii_case("Z_SCORE")
            && params
                .get("mean_source")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("ema"))
    }

    fn get_zscore_ema(&self, price_type: PriceType, params: &serde_json::Value) -> Option<f64> {
        let window = params.get("window").or_else(|| params.get("period"))?;
        let window = window.as_u64().and_then(u64_to_usize)?;
        let ema_period = params
            .get("ema_period")
            .and_then(serde_json::Value::as_u64)
            .and_then(u64_to_usize)?;
        if let Some(cache) = self.multi_tf {
            let tf = cache.borrow().primary_timeframe();
            let series = cache
                .borrow_mut()
                .zscore_ema(tf, price_type, window, ema_period);
            return value_at(&series, self.idx);
        }
        if price_type == PriceType::Bid {
            let spec = IndicatorSpec::new("Z_SCORE", IndicatorParams::Period(window));
            let value = self.indicators.get_at(&spec, self.idx)?;
            return if value.is_nan() { None } else { Some(value) };
        }
        None
    }

    /// Gets an indicator value for the specified price type.
    ///
    /// Uses the multi-timeframe cache when available and falls back
    /// to the primary indicator cache.
    #[must_use]
    pub fn get_indicator_with_price_type(
        &self,
        name: &str,
        price_type: PriceType,
        params: &serde_json::Value,
    ) -> Option<f64> {
        if Self::is_zscore_ema(name, params) {
            return self.get_zscore_ema(price_type, params);
        }
        if let Some(cache) = self.multi_tf {
            let tf = cache.borrow().primary_timeframe();
            let name_upper = name.to_uppercase();
            if name_upper == "ATR" {
                let period = params
                    .get("period")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let series = cache.borrow_mut().atr(tf, price_type, period);
                return value_at(&series, self.idx);
            }
            if name_upper == "EMA" {
                let period = params
                    .get("period")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(u64_to_usize)?;
                let series = cache.borrow_mut().ema(tf, price_type, period);
                return value_at(&series, self.idx);
            }
        }
        if price_type == PriceType::Bid {
            return self.get_indicator(name, params);
        }
        None
    }

    /// Gets the stepwise EMA for a given timeframe.
    #[must_use]
    pub fn get_stepwise_ema(
        &self,
        timeframe: &str,
        price_type: PriceType,
        period: usize,
    ) -> Option<f64> {
        if let Some(cache) = self.multi_tf {
            let tf = Self::parse_timeframe(timeframe)?;
            let series = cache.borrow_mut().ema_stepwise(tf, price_type, period);
            return value_at(&series, self.idx);
        }
        if let Some(htf) = self.htf_data.as_ref()
            && htf.timeframe.eq_ignore_ascii_case(timeframe)
        {
            let spec = IndicatorSpec::new("EMA", IndicatorParams::Period(period));
            let value = htf.indicators.get_at(&spec, htf.idx)?;
            return if value.is_nan() { None } else { Some(value) };
        }
        None
    }

    /// Gets the stepwise Kalman-Z score for a given timeframe.
    #[must_use]
    pub fn get_stepwise_kalman_zscore(
        &self,
        timeframe: &str,
        price_type: PriceType,
        window: usize,
        r: f64,
        q: f64,
    ) -> Option<f64> {
        if let Some(cache) = self.multi_tf {
            let tf = Self::parse_timeframe(timeframe)?;
            let series = cache
                .borrow_mut()
                .kalman_zscore_stepwise(tf, price_type, window, r, q);
            return value_at(&series, self.idx);
        }
        let name = format!("KALMAN_Z_{timeframe}");
        let value = self.get_indicator(
            &name,
            &serde_json::json!({
                "window": window,
                "r": r,
                "q": q
            }),
        )?;
        Some(value)
    }

    /// Gets a stepwise Bollinger output (upper/middle/lower) for a timeframe.
    #[must_use]
    pub fn get_stepwise_bollinger_output(
        &self,
        timeframe: &str,
        price_type: PriceType,
        period: usize,
        std_factor: f64,
        output: &str,
    ) -> Option<f64> {
        if let Some(cache) = self.multi_tf {
            let tf = Self::parse_timeframe(timeframe)?;
            let (upper, middle, lower) = cache
                .borrow_mut()
                .bollinger_stepwise(tf, price_type, period, std_factor);
            return match output {
                "upper" => value_at(&upper, self.idx),
                "middle" => value_at(&middle, self.idx),
                "lower" => value_at(&lower, self.idx),
                _ => None,
            };
        }
        let name = format!("BOLLINGER_{timeframe}");
        self.get_indicator_output(
            &name,
            output,
            &serde_json::json!({"period": period, "std_factor": std_factor}),
        )
    }

    /// Gets the aligned close price for a given timeframe.
    #[must_use]
    pub fn get_tf_close(&self, timeframe: &str, price_type: PriceType) -> Option<f64> {
        if let Some(cache) = self.multi_tf {
            let tf = Self::parse_timeframe(timeframe)?;
            let series = cache.borrow_mut().closes_aligned(tf, price_type);
            return value_at(&series, self.idx);
        }
        if timeframe.eq_ignore_ascii_case("PRIMARY") {
            return match price_type {
                PriceType::Bid => Some(self.bid.close),
                PriceType::Ask => Some(self.ask.close),
            };
        }
        if let Some(htf) = self.htf_data.as_ref()
            && htf.timeframe.eq_ignore_ascii_case(timeframe)
        {
            return match price_type {
                PriceType::Bid => Some(htf.bid.close),
                PriceType::Ask => Some(htf.ask.close),
            };
        }
        let name = format!("CLOSE_{timeframe}");
        self.get_indicator(&name, &serde_json::json!({}))
    }
}

impl<'a> HtfContext<'a> {
    /// Creates a new `HtfContext`.
    #[must_use]
    pub fn new(
        bid: &'a Candle,
        ask: &'a Candle,
        indicators: &'a IndicatorCache,
        idx: usize,
        timeframe: &'a str,
    ) -> Self {
        Self {
            bid,
            ask,
            indicators,
            idx,
            timeframe,
        }
    }
}

fn value_at(series: &[f64], idx: usize) -> Option<f64> {
    series.get(idx).copied().filter(|v| v.is_finite())
}

fn u64_to_usize(value: u64) -> Option<usize> {
    usize::try_from(value).ok()
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

#[cfg(test)]
mod tests {
    use super::*;
    use omega_indicators::{EMA, ZScore};

    fn make_candle(close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            close_time_ns: 60_000_000_000 - 1,
            open: close,
            high: close,
            low: close,
            close,
            volume: 0.0,
        }
    }

    #[test]
    fn test_bar_context_new() {
        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();

        let ctx = BarContext::new(10, 1_234_567_890, &bid, &ask, &cache);

        assert_eq!(ctx.idx, 10);
        assert_eq!(ctx.timestamp_ns, 1_234_567_890);
        assert!(ctx.session_open);
        assert!(!ctx.news_blocked);
        assert!(ctx.htf_data.is_none());
    }

    #[test]
    fn test_bar_context_with_session() {
        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();

        let ctx = BarContext::new(10, 1_234_567_890, &bid, &ask, &cache).with_session(false);

        assert!(!ctx.session_open);
    }

    #[test]
    fn test_bar_context_with_news_blocked() {
        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();

        let ctx = BarContext::new(10, 1_234_567_890, &bid, &ask, &cache).with_news_blocked(true);

        assert!(ctx.news_blocked);
    }

    #[test]
    fn test_get_indicator() {
        let candles: Vec<Candle> = (1..=10).map(|i| make_candle(f64::from(i))).collect();
        let bid = &candles[9];
        let ask = make_candle(10.0002);
        let mut cache = IndicatorCache::new();

        // Pre-compute EMA
        let ema = EMA::new(3);
        let spec = IndicatorSpec::new("EMA", IndicatorParams::Period(3));
        cache.get_or_compute(&spec, &candles, &ema);

        let ctx = BarContext::new(9, 1_234_567_890, bid, &ask, &cache);
        let value = ctx.get_indicator("EMA", &serde_json::json!({"period": 3}));

        assert!(value.is_some());
        assert!(value.unwrap().is_finite());
    }

    #[test]
    fn test_get_indicator_nan_returns_none() {
        let candles: Vec<Candle> = (1..=10).map(|i| make_candle(f64::from(i))).collect();
        let bid = &candles[0]; // Index 0 is in warmup period
        let ask = make_candle(1.0002);
        let mut cache = IndicatorCache::new();

        // Pre-compute Z-Score with window 5 (first 4 values are NaN)
        let zscore = ZScore::new(5);
        let spec = IndicatorSpec::new("Z_SCORE", IndicatorParams::Period(5));
        cache.get_or_compute(&spec, &candles, &zscore);

        let ctx = BarContext::new(0, 1_234_567_890, bid, &ask, &cache);
        let value = ctx.get_indicator("Z_SCORE", &serde_json::json!({"period": 5}));

        assert!(value.is_none()); // NaN returns None
    }

    #[test]
    fn test_htf_context_new() {
        let bid = make_candle(1.1000);
        let ask = make_candle(1.1002);
        let cache = IndicatorCache::new();

        let htf = HtfContext::new(&bid, &ask, &cache, 5, "H4");

        assert_eq!(htf.idx, 5);
        assert_eq!(htf.timeframe, "H4");
    }
}
