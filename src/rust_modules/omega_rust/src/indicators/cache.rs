//! IndicatorCache implementation in Rust.
//!
//! This module provides a high-performance cache for technical indicator calculations.
//! It stores OHLCV data and caches computed indicator results to avoid redundant calculations.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    IndicatorCacheRust                         │
//! │  ┌─────────────────────┐    ┌───────────────────────────────┐│
//! │  │   ohlcv_storage     │    │      indicator_cache         ││
//! │  │ HashMap<OhlcvKey,   │    │ HashMap<CacheKey,            ││
//! │  │         OhlcvData>  │    │         IndicatorResult>     ││
//! │  └─────────────────────┘    └───────────────────────────────┘│
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Targets
//!
//! | Indicator | Python Baseline | Rust Target | Speedup |
//! |-----------|-----------------|-------------|---------|
//! | ATR       | 954ms           | ≤19ms       | 50x     |
//! | EMA_stepwise | 45ms         | ≤2.3ms      | 20x     |
//! | Bollinger | 89ms            | ≤4.5ms      | 20x     |
//! | DMI       | 65ms            | ≤3.3ms      | 20x     |
//! | SMA       | 23ms            | ≤2.3ms      | 10x     |
//!
//! ## Usage
//!
//! ```python
//! from omega_rust import IndicatorCacheRust
//!
//! cache = IndicatorCacheRust()
//! cache.register_ohlcv("EUR/USD", "H1", "BID", timestamps, opens, highs, lows, closes, volumes)
//! atr = cache.atr("EUR/USD", "H1", "BID", 14)
//! ```

use crate::error::{OmegaError, Result};
use crate::indicators::atr::atr_impl;
use crate::indicators::bollinger::{bollinger_impl, bollinger_stepwise_impl};
use crate::indicators::choppiness::choppiness_impl;
use crate::indicators::dmi::dmi_impl;
use crate::indicators::ema_extended::{dema_impl, ema_impl, ema_stepwise_impl, tema_impl};
use crate::indicators::garch::{garch_volatility_impl, garch_volatility_local_impl, GarchParams};
use crate::indicators::kalman::{kalman_impl, kalman_zscore_impl, kalman_zscore_stepwise_impl};
use crate::indicators::macd::macd_impl;
use crate::indicators::roc::{momentum_impl, roc_impl};
use crate::indicators::rsi_impl::rsi_impl;
use crate::indicators::sma::{rolling_std_impl, sma_impl};
use crate::indicators::types::{CacheKey, IndicatorResult, OhlcvData, OhlcvKey};
use crate::indicators::zscore::zscore_impl;

use std::collections::HashMap;

/// Indicator Cache for high-performance technical indicator calculations.
///
/// This struct maintains:
/// 1. OHLCV data storage for multiple symbols/timeframes
/// 2. Cached indicator results to avoid redundant calculations
pub struct IndicatorCache {
    /// OHLCV data storage: (symbol, timeframe, price_type) -> OhlcvData
    ohlcv_storage: HashMap<OhlcvKey, OhlcvData>,

    /// Indicator result cache: CacheKey -> IndicatorResult
    indicator_cache: HashMap<CacheKey, IndicatorResult>,
}

impl Default for IndicatorCache {
    fn default() -> Self {
        Self::new()
    }
}

impl IndicatorCache {
    /// Create a new empty IndicatorCache.
    pub fn new() -> Self {
        Self {
            ohlcv_storage: HashMap::new(),
            indicator_cache: HashMap::new(),
        }
    }

    /// Create a new IndicatorCache with estimated capacity.
    pub fn with_capacity(ohlcv_capacity: usize, indicator_capacity: usize) -> Self {
        Self {
            ohlcv_storage: HashMap::with_capacity(ohlcv_capacity),
            indicator_cache: HashMap::with_capacity(indicator_capacity),
        }
    }

    /// Register OHLCV data for a symbol/timeframe/price_type combination.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol (e.g., "EUR/USD")
    /// * `timeframe` - Timeframe (e.g., "H1", "M5")
    /// * `price_type` - Price type (e.g., "BID", "ASK")
    /// * `open` - Open prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `volume` - Volume data
    ///
    /// # Errors
    ///
    /// Returns an error if arrays have different lengths.
    pub fn register_ohlcv(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Result<()> {
        let n = open.len();
        if high.len() != n || low.len() != n || close.len() != n || volume.len() != n {
            return Err(OmegaError::InvalidParameter {
                reason: format!(
                    "Array length mismatch: open={}, high={}, low={}, close={}, volume={}",
                    n,
                    high.len(),
                    low.len(),
                    close.len(),
                    volume.len()
                ),
            });
        }

        // Calculate valid mask (all values finite)
        let valid: Vec<bool> = (0..n)
            .map(|i| {
                open[i].is_finite()
                    && high[i].is_finite()
                    && low[i].is_finite()
                    && close[i].is_finite()
            })
            .collect();

        let key = (
            symbol.to_string(),
            timeframe.to_string(),
            price_type.to_string(),
        );

        let data = OhlcvData {
            open,
            high,
            low,
            close,
            volume,
            valid,
            n_bars: n,
        };

        self.ohlcv_storage.insert(key, data);

        // Clear cached indicators for this symbol/timeframe (data changed)
        self.invalidate_cache(symbol, timeframe, price_type);

        Ok(())
    }

    /// Invalidate cached indicators for a specific symbol/timeframe.
    fn invalidate_cache(&mut self, symbol: &str, timeframe: &str, price_type: &str) {
        self.indicator_cache.retain(|key, _| {
            !(key.symbol == symbol && key.timeframe == timeframe && key.price_type == price_type)
        });
    }

    /// Clear all cached data.
    pub fn clear(&mut self) {
        self.ohlcv_storage.clear();
        self.indicator_cache.clear();
    }

    /// Get number of cached indicators.
    pub fn cache_size(&self) -> usize {
        self.indicator_cache.len()
    }

    /// Get OHLCV data for a key.
    fn get_ohlcv(&self, symbol: &str, timeframe: &str, price_type: &str) -> Result<&OhlcvData> {
        let key = (
            symbol.to_string(),
            timeframe.to_string(),
            price_type.to_string(),
        );
        self.ohlcv_storage.get(&key).ok_or(OmegaError::NotFound {
            item: format!("OHLCV data for {}/{}/{}", symbol, timeframe, price_type),
        })
    }

    // ========================================================================
    // Indicator Methods
    // ========================================================================

    /// Calculate Average True Range (ATR).
    ///
    /// Uses Wilder smoothing (same as Bloomberg/TradingView).
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `timeframe` - Timeframe
    /// * `price_type` - Price type
    /// * `period` - ATR period (typically 14)
    pub fn atr(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("atr", symbol, timeframe, price_type, period);

        // Check cache
        if let Some(result) = self.indicator_cache.get(&cache_key) {
            if let IndicatorResult::Single(values) = result {
                return Ok(values.clone());
            }
        }

        // Calculate
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = atr_impl(&ohlcv.high, &ohlcv.low, &ohlcv.close, period)?;

        // Cache
        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));

        Ok(result)
    }

    /// Calculate Simple Moving Average (SMA).
    pub fn sma(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("sma", symbol, timeframe, price_type, period);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = sma_impl(&ohlcv.close, period)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Exponential Moving Average (EMA).
    pub fn ema(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
        start_idx: Option<usize>,
    ) -> Result<Vec<f64>> {
        let params_str = match start_idx {
            Some(idx) => format!("{},{}", span, idx),
            None => span.to_string(),
        };
        let cache_key = CacheKey::new("ema", symbol, timeframe, price_type, &params_str);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = ema_impl(&ohlcv.close, span, start_idx)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate EMA with stepwise HTF-bar semantics.
    pub fn ema_stepwise(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
        new_bar_indices: &[usize],
    ) -> Result<Vec<f64>> {
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        ema_stepwise_impl(&ohlcv.close, new_bar_indices, span)
    }

    /// Calculate Double EMA (DEMA).
    pub fn dema(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("dema", symbol, timeframe, price_type, span);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = dema_impl(&ohlcv.close, span)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Triple EMA (TEMA).
    pub fn tema(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("tema", symbol, timeframe, price_type, span);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = tema_impl(&ohlcv.close, span)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Bollinger Bands.
    ///
    /// Returns (upper, middle, lower) bands.
    pub fn bollinger(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        std_factor: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let cache_key =
            CacheKey::bollinger(symbol, timeframe, price_type, period, std_factor, false);

        if let Some(IndicatorResult::Triple(upper, middle, lower)) =
            self.indicator_cache.get(&cache_key)
        {
            return Ok((upper.clone(), middle.clone(), lower.clone()));
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = bollinger_impl(&ohlcv.close, period, std_factor)?;

        self.indicator_cache.insert(
            cache_key,
            IndicatorResult::Triple(
                result.upper.clone(),
                result.middle.clone(),
                result.lower.clone(),
            ),
        );
        Ok((result.upper, result.middle, result.lower))
    }

    /// Calculate Bollinger Bands with stepwise semantics.
    pub fn bollinger_stepwise(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        std_factor: f64,
        new_bar_indices: &[usize],
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = bollinger_stepwise_impl(&ohlcv.close, new_bar_indices, period, std_factor)?;
        Ok((result.upper, result.middle, result.lower))
    }

    /// Calculate Directional Movement Index (DMI).
    ///
    /// Returns (+DI, -DI, ADX).
    pub fn dmi(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let cache_key = CacheKey::single_period("dmi", symbol, timeframe, price_type, period);

        if let Some(IndicatorResult::Triple(plus_di, minus_di, adx)) =
            self.indicator_cache.get(&cache_key)
        {
            return Ok((plus_di.clone(), minus_di.clone(), adx.clone()));
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = dmi_impl(&ohlcv.high, &ohlcv.low, &ohlcv.close, period)?;

        self.indicator_cache.insert(
            cache_key,
            IndicatorResult::Triple(
                result.plus_di.clone(),
                result.minus_di.clone(),
                result.adx.clone(),
            ),
        );
        Ok((result.plus_di, result.minus_di, result.adx))
    }

    /// Calculate MACD.
    ///
    /// Returns (macd_line, signal_line, histogram).
    pub fn macd(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        fast_span: usize,
        slow_span: usize,
        signal_span: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let cache_key = CacheKey::macd(
            symbol,
            timeframe,
            price_type,
            fast_span,
            slow_span,
            signal_span,
        );

        if let Some(IndicatorResult::Triple(macd, signal, hist)) =
            self.indicator_cache.get(&cache_key)
        {
            return Ok((macd.clone(), signal.clone(), hist.clone()));
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = macd_impl(&ohlcv.close, fast_span, slow_span, signal_span)?;

        self.indicator_cache.insert(
            cache_key,
            IndicatorResult::Triple(
                result.macd.clone(),
                result.signal.clone(),
                result.histogram.clone(),
            ),
        );
        Ok((result.macd, result.signal, result.histogram))
    }

    /// Calculate Rate of Change (ROC).
    pub fn roc(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("roc", symbol, timeframe, price_type, period);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = roc_impl(&ohlcv.close, period)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Momentum.
    pub fn momentum(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("momentum", symbol, timeframe, price_type, period);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = momentum_impl(&ohlcv.close, period)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Relative Strength Index (RSI).
    ///
    /// Uses Wilder smoothing (same as TradingView/MT5).
    pub fn rsi(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::single_period("rsi", symbol, timeframe, price_type, period);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = rsi_impl(&ohlcv.close, period)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Z-Score.
    pub fn zscore(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        ddof: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::new(
            "zscore",
            symbol,
            timeframe,
            price_type,
            &format!("{},{}", period, ddof),
        );

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = zscore_impl(&ohlcv.close, period, ddof)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Choppiness Index.
    pub fn choppiness(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> Result<Vec<f64>> {
        let cache_key =
            CacheKey::single_period("choppiness", symbol, timeframe, price_type, period);

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = choppiness_impl(&ohlcv.high, &ohlcv.low, &ohlcv.close, period)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Kalman-filtered mean.
    pub fn kalman_mean(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        process_variance: f64,
        measurement_variance: f64,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::new(
            "kalman_mean",
            symbol,
            timeframe,
            price_type,
            &format!("{:.10},{:.10}", process_variance, measurement_variance),
        );

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = kalman_impl(&ohlcv.close, process_variance, measurement_variance)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.mean.clone()));
        Ok(result.mean)
    }

    /// Calculate Kalman Z-Score.
    pub fn kalman_zscore(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        process_variance: f64,
        measurement_variance: f64,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::new(
            "kalman_zscore",
            symbol,
            timeframe,
            price_type,
            &format!("{:.10},{:.10}", process_variance, measurement_variance),
        );

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = kalman_zscore_impl(&ohlcv.close, process_variance, measurement_variance)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    /// Calculate Rolling Standard Deviation.
    pub fn rolling_std(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        ddof: usize,
    ) -> Result<Vec<f64>> {
        let cache_key = CacheKey::new(
            "rolling_std",
            symbol,
            timeframe,
            price_type,
            &format!("{},{}", period, ddof),
        );

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let result = rolling_std_impl(&ohlcv.close, period, ddof)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.clone()));
        Ok(result)
    }

    // ========================================================================
    // GARCH Volatility Methods (Wave 1 - Python-only indicators)
    // ========================================================================

    /// Calculate GARCH(1,1) volatility on returns.
    ///
    /// Returns volatility (σ) of returns, not prices.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `timeframe` - Timeframe
    /// * `price_type` - Price type
    /// * `alpha` - Weight for squared shock (typically 0.05)
    /// * `beta` - Persistence weight (typically 0.90)
    /// * `omega` - Long-run variance component (auto-computed if None)
    /// * `use_log_returns` - Use log returns vs simple returns
    /// * `scale` - Scaling factor for returns (e.g., 100 for percentage)
    /// * `min_periods` - Minimum periods before trusting estimate
    /// * `sigma_floor` - Floor for sigma to prevent numerical issues
    pub fn garch_volatility(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        alpha: f64,
        beta: f64,
        omega: Option<f64>,
        use_log_returns: bool,
        scale: f64,
        min_periods: usize,
        sigma_floor: f64,
    ) -> Result<Vec<f64>> {
        let omega_str = omega.map(|v| v.to_string()).unwrap_or("-1".to_string());
        let cache_key = CacheKey::new(
            "garch_vol",
            symbol,
            timeframe,
            price_type,
            &format!(
                "{:.6},{:.6},{},{},{:.6},{},{}",
                alpha, beta, omega_str, use_log_returns, scale, min_periods, sigma_floor
            ),
        );

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let params = GarchParams {
            alpha,
            beta,
            omega,
            use_log_returns,
            scale,
            min_periods,
            sigma_floor,
        };
        let result = garch_volatility_impl(&ohlcv.close, &params)?;

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(result.sigma.clone()));
        Ok(result.sigma)
    }

    /// Calculate GARCH(1,1) volatility on a local window.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `timeframe` - Timeframe
    /// * `price_type` - Price type
    /// * `idx` - Current index (0-based, inclusive)
    /// * `lookback` - Window size for local estimation
    /// * `alpha` - Weight for squared shock
    /// * `beta` - Persistence weight
    /// * `omega` - Long-run variance component (auto-computed if None)
    /// * `use_log_returns` - Use log returns vs simple returns
    /// * `scale` - Scaling factor for returns
    /// * `min_periods` - Minimum periods before trusting estimate
    /// * `sigma_floor` - Floor for sigma
    #[allow(clippy::too_many_arguments)]
    pub fn garch_volatility_local(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        idx: usize,
        lookback: usize,
        alpha: f64,
        beta: f64,
        omega: Option<f64>,
        use_log_returns: bool,
        scale: f64,
        min_periods: usize,
        sigma_floor: f64,
    ) -> Result<Vec<f64>> {
        // Note: Local methods are typically not cached due to varying idx
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let params = GarchParams {
            alpha,
            beta,
            omega,
            use_log_returns,
            scale,
            min_periods,
            sigma_floor,
        };
        let result = garch_volatility_local_impl(&ohlcv.close, idx, lookback, &params)?;
        Ok(result.sigma)
    }

    /// Get the final sigma value from local GARCH calculation.
    #[allow(clippy::too_many_arguments)]
    pub fn garch_volatility_local_last(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        idx: usize,
        lookback: usize,
        alpha: f64,
        beta: f64,
        omega: Option<f64>,
        use_log_returns: bool,
        scale: f64,
        min_periods: usize,
        sigma_floor: f64,
    ) -> Result<f64> {
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let params = GarchParams {
            alpha,
            beta,
            omega,
            use_log_returns,
            scale,
            min_periods,
            sigma_floor,
        };
        crate::indicators::garch::garch_volatility_local_last(&ohlcv.close, idx, lookback, &params)
    }

    /// Calculate Kalman Z-Score with stepwise (HTF bar) semantics.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `timeframe` - Timeframe
    /// * `price_type` - Price type
    /// * `new_bar_indices` - Indices where new HTF bars occur
    /// * `window` - Rolling window for residual std
    /// * `process_variance` - Kalman process noise (Q)
    /// * `measurement_variance` - Kalman measurement noise (R)
    pub fn kalman_zscore_stepwise(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        new_bar_indices: &[usize],
        window: usize,
        process_variance: f64,
        measurement_variance: f64,
    ) -> Result<Vec<f64>> {
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        kalman_zscore_stepwise_impl(
            &ohlcv.close,
            new_bar_indices,
            window,
            process_variance,
            measurement_variance,
        )
    }

    /// Calculate Kalman+GARCH Z-Score.
    ///
    /// Z = (Close - KalmanMean) / (|Close| × σ_return)
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `timeframe` - Timeframe
    /// * `price_type` - Price type
    /// * `r` - Kalman measurement noise (R)
    /// * `q` - Kalman process noise (Q)
    /// * `alpha` - GARCH alpha
    /// * `beta` - GARCH beta
    /// * `omega` - GARCH omega (auto-computed if None)
    /// * `use_log_returns` - Use log returns
    /// * `scale` - Return scaling factor
    /// * `min_periods` - Minimum periods
    /// * `sigma_floor` - Sigma floor
    #[allow(clippy::too_many_arguments)]
    pub fn kalman_garch_zscore(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        r: f64,
        q: f64,
        alpha: f64,
        beta: f64,
        omega: Option<f64>,
        use_log_returns: bool,
        scale: f64,
        min_periods: usize,
        sigma_floor: f64,
    ) -> Result<Vec<f64>> {
        let omega_str = omega.map(|v| v.to_string()).unwrap_or("-1".to_string());
        let cache_key = CacheKey::new(
            "kalman_garch_z",
            symbol,
            timeframe,
            price_type,
            &format!(
                "{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{},{}",
                r, q, alpha, beta, omega_str, use_log_returns, scale, min_periods, sigma_floor
            ),
        );

        if let Some(IndicatorResult::Single(values)) = self.indicator_cache.get(&cache_key) {
            return Ok(values.clone());
        }

        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let n = ohlcv.close.len();

        // Get Kalman mean
        let kalman = kalman_impl(&ohlcv.close, q, r)?;

        // Get GARCH volatility
        let garch_params = GarchParams {
            alpha,
            beta,
            omega,
            use_log_returns,
            scale,
            min_periods,
            sigma_floor,
        };
        let garch = garch_volatility_impl(&ohlcv.close, &garch_params)?;

        // Calculate Z-Score
        let mut zscore = vec![f64::NAN; n];
        for i in 0..n {
            let close = ohlcv.close[i];
            let km = kalman.mean[i];
            let sigma_ret = garch.sigma[i];

            if !close.is_finite() || !km.is_finite() || !sigma_ret.is_finite() {
                continue;
            }

            // sigma_price = |close| × sigma_return
            let sigma_price = close.abs() * sigma_ret;
            if sigma_price <= 0.0 {
                continue;
            }

            zscore[i] = (close - km) / sigma_price;
        }

        self.indicator_cache
            .insert(cache_key, IndicatorResult::Single(zscore.clone()));
        Ok(zscore)
    }

    /// Calculate local Kalman+GARCH Z-Score at a specific index.
    ///
    /// Returns the Z-Score value at the given index using a local lookback window.
    #[allow(clippy::too_many_arguments)]
    pub fn kalman_garch_zscore_local(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        idx: usize,
        lookback: usize,
        r: f64,
        q: f64,
        alpha: f64,
        beta: f64,
        omega: Option<f64>,
        use_log_returns: bool,
        scale: f64,
        min_periods: usize,
        sigma_floor: f64,
    ) -> Result<Option<f64>> {
        let ohlcv = self.get_ohlcv(symbol, timeframe, price_type)?;
        let n = ohlcv.close.len();

        if idx >= n {
            return Ok(None);
        }

        // Extract local window
        let end_pos = idx + 1;
        let start_pos = end_pos.saturating_sub(lookback);
        let local_closes = &ohlcv.close[start_pos..end_pos];

        if local_closes.len() < 2 {
            return Ok(None);
        }

        // Kalman on local segment
        let kalman = kalman_impl(local_closes, q, r)?;
        let km_last = *kalman.mean.last().unwrap_or(&f64::NAN);

        if !km_last.is_finite() {
            return Ok(None);
        }

        // GARCH on local segment
        let garch_params = GarchParams {
            alpha,
            beta,
            omega,
            use_log_returns,
            scale,
            min_periods,
            sigma_floor,
        };
        let garch = garch_volatility_impl(local_closes, &garch_params)?;
        let sigma_ret_last = *garch.sigma.last().unwrap_or(&f64::NAN);

        if !sigma_ret_last.is_finite() || sigma_ret_last <= 0.0 {
            return Ok(None);
        }

        let close_last = *local_closes.last().unwrap_or(&f64::NAN);
        if !close_last.is_finite() {
            return Ok(None);
        }

        let sigma_price = close_last.abs() * sigma_ret_last;
        if sigma_price <= 0.0 {
            return Ok(None);
        }

        let z = (close_last - km_last) / sigma_price;
        Ok(Some(z))
    }

    /// Volatility cluster series.
    ///
    /// Routes to ATR or local GARCH based on feature type.
    ///
    /// # Arguments
    ///
    /// * `feature` - "atr_points" for ATR, anything else for GARCH
    /// * `atr_period` - Period for ATR calculation
    /// * Other args - GARCH parameters
    #[allow(clippy::too_many_arguments)]
    pub fn vol_cluster_series(
        &mut self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        idx: usize,
        feature: &str,
        atr_period: usize,
        garch_lookback: usize,
        garch_alpha: f64,
        garch_beta: f64,
        garch_omega: Option<f64>,
        garch_use_log_returns: bool,
        garch_scale: f64,
        garch_min_periods: usize,
        garch_sigma_floor: f64,
    ) -> Result<Vec<f64>> {
        let feat_lower = feature.to_lowercase();
        if feat_lower == "atr_points" {
            return self.atr(symbol, timeframe, price_type, atr_period);
        }

        // Default to local GARCH
        self.garch_volatility_local(
            symbol,
            timeframe,
            price_type,
            idx,
            garch_lookback,
            garch_alpha,
            garch_beta,
            garch_omega,
            garch_use_log_returns,
            garch_scale,
            garch_min_periods,
            garch_sigma_floor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> IndicatorCache {
        let mut cache = IndicatorCache::new();

        let open = vec![
            100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0,
        ];
        let high = vec![
            101.0, 102.0, 103.0, 102.5, 104.0, 103.0, 105.0, 104.5, 106.0, 105.0,
        ];
        let low = vec![
            99.0, 100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0,
        ];
        let close = vec![
            100.5, 101.5, 102.5, 102.0, 103.5, 102.5, 104.5, 104.0, 105.5, 104.5,
        ];
        let volume = vec![1000.0; 10];

        cache
            .register_ohlcv("EUR/USD", "H1", "BID", open, high, low, close, volume)
            .unwrap();

        cache
    }

    #[test]
    fn test_cache_creation() {
        let cache = IndicatorCache::new();
        assert_eq!(cache.cache_size(), 0);
    }

    #[test]
    fn test_register_ohlcv() {
        let cache = create_test_cache();
        assert!(cache.get_ohlcv("EUR/USD", "H1", "BID").is_ok());
        assert!(cache.get_ohlcv("GBP/USD", "H1", "BID").is_err());
    }

    #[test]
    fn test_atr_caching() {
        let mut cache = create_test_cache();

        // First call - calculates
        let atr1 = cache.atr("EUR/USD", "H1", "BID", 3).unwrap();
        assert_eq!(cache.cache_size(), 1);

        // Second call - from cache
        let atr2 = cache.atr("EUR/USD", "H1", "BID", 3).unwrap();
        assert_eq!(cache.cache_size(), 1);

        // Results should be identical
        assert_eq!(atr1.len(), atr2.len());
        for i in 0..atr1.len() {
            if atr1[i].is_finite() && atr2[i].is_finite() {
                assert!((atr1[i] - atr2[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_multiple_indicators() {
        let mut cache = create_test_cache();

        let _ = cache.atr("EUR/USD", "H1", "BID", 3).unwrap();
        let _ = cache.sma("EUR/USD", "H1", "BID", 3).unwrap();
        let _ = cache.ema("EUR/USD", "H1", "BID", 3, None).unwrap();

        assert_eq!(cache.cache_size(), 3);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache = create_test_cache();

        let _ = cache.atr("EUR/USD", "H1", "BID", 3).unwrap();
        assert_eq!(cache.cache_size(), 1);

        // Re-register OHLCV data should invalidate cache
        let open = vec![100.0; 10];
        let high = vec![101.0; 10];
        let low = vec![99.0; 10];
        let close = vec![100.5; 10];
        let volume = vec![1000.0; 10];

        cache
            .register_ohlcv("EUR/USD", "H1", "BID", open, high, low, close, volume)
            .unwrap();

        assert_eq!(cache.cache_size(), 0);
    }

    #[test]
    fn test_bollinger() {
        let mut cache = create_test_cache();
        let (upper, middle, lower) = cache.bollinger("EUR/USD", "H1", "BID", 3, 2.0).unwrap();

        assert_eq!(upper.len(), 10);
        assert_eq!(middle.len(), 10);
        assert_eq!(lower.len(), 10);

        // Upper >= Middle >= Lower
        for i in 0..10 {
            if middle[i].is_finite() {
                assert!(upper[i] >= middle[i]);
                assert!(middle[i] >= lower[i]);
            }
        }
    }

    #[test]
    fn test_dmi() {
        let mut cache = create_test_cache();
        let (plus_di, minus_di, adx) = cache.dmi("EUR/USD", "H1", "BID", 3).unwrap();

        assert_eq!(plus_di.len(), 10);
        assert_eq!(minus_di.len(), 10);
        assert_eq!(adx.len(), 10);
    }

    #[test]
    fn test_macd() {
        let mut cache = create_test_cache();
        let (macd, signal, histogram) = cache.macd("EUR/USD", "H1", "BID", 3, 5, 2).unwrap();

        assert_eq!(macd.len(), 10);
        assert_eq!(signal.len(), 10);
        assert_eq!(histogram.len(), 10);
    }
}
