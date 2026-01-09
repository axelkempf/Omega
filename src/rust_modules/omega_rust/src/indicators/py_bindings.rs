//! PyO3 bindings for IndicatorCache.
//!
//! This module exposes the Rust IndicatorCache to Python via PyO3.
//!
//! ## Usage from Python
//!
//! ```python
//! from omega_rust import IndicatorCacheRust
//!
//! cache = IndicatorCacheRust()
//! cache.register_ohlcv(
//!     symbol="EUR/USD",
//!     timeframe="H1",
//!     price_type="BID",
//!     open=opens,      # numpy array
//!     high=highs,
//!     low=lows,
//!     close=closes,
//!     volume=volumes
//! )
//! atr = cache.atr("EUR/USD", "H1", "BID", period=14)
//! ```
//!
//! ## Feature Flag
//!
//! Enable via environment variable: `OMEGA_USE_RUST_INDICATOR_CACHE=1`

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Mutex;

use crate::indicators::cache::IndicatorCache;

/// Python wrapper for IndicatorCache.
///
/// Thread-safe via internal Mutex.
#[pyclass(name = "IndicatorCacheRust")]
pub struct PyIndicatorCache {
    inner: Mutex<IndicatorCache>,
}

#[pymethods]
impl PyIndicatorCache {
    /// Create a new IndicatorCache.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(IndicatorCache::new()),
        }
    }

    /// Register OHLCV data for a symbol/timeframe/price_type.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol (e.g., "EUR/USD")
    /// * `timeframe` - Timeframe (e.g., "H1", "M5")
    /// * `price_type` - Price type ("BID" or "ASK")
    /// * `open` - Open prices as numpy array
    /// * `high` - High prices as numpy array
    /// * `low` - Low prices as numpy array
    /// * `close` - Close prices as numpy array
    /// * `volume` - Volume data as numpy array
    #[pyo3(signature = (symbol, timeframe, price_type, open, high, low, close, volume))]
    pub fn register_ohlcv(
        &self,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        open: PyReadonlyArray1<f64>,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        volume: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        cache
            .register_ohlcv(
                symbol,
                timeframe,
                price_type,
                open.as_slice()?.to_vec(),
                high.as_slice()?.to_vec(),
                low.as_slice()?.to_vec(),
                close.as_slice()?.to_vec(),
                volume.as_slice()?.to_vec(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(())
    }

    /// Clear all cached data.
    pub fn clear(&self) -> PyResult<()> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        cache.clear();
        Ok(())
    }

    /// Get number of cached indicators.
    pub fn cache_size(&self) -> PyResult<usize> {
        let cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        Ok(cache.cache_size())
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
    ///
    /// # Returns
    ///
    /// numpy array of ATR values
    #[pyo3(signature = (symbol, timeframe, price_type, period))]
    pub fn atr<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .atr(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Simple Moving Average (SMA).
    #[pyo3(signature = (symbol, timeframe, price_type, period))]
    pub fn sma<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .sma(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Exponential Moving Average (EMA).
    #[pyo3(signature = (symbol, timeframe, price_type, span, start_idx=None))]
    pub fn ema<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
        start_idx: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .ema(symbol, timeframe, price_type, span, start_idx)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate EMA with stepwise HTF-bar semantics.
    #[pyo3(signature = (symbol, timeframe, price_type, span, new_bar_indices))]
    pub fn ema_stepwise<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
        new_bar_indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .ema_stepwise(symbol, timeframe, price_type, span, &new_bar_indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Double EMA (DEMA).
    #[pyo3(signature = (symbol, timeframe, price_type, span))]
    pub fn dema<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .dema(symbol, timeframe, price_type, span)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Triple EMA (TEMA).
    #[pyo3(signature = (symbol, timeframe, price_type, span))]
    pub fn tema<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        span: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .tema(symbol, timeframe, price_type, span)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Bollinger Bands.
    ///
    /// Returns tuple of (upper, middle, lower) numpy arrays.
    #[pyo3(signature = (symbol, timeframe, price_type, period, std_factor=2.0))]
    pub fn bollinger<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        std_factor: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let (upper, middle, lower) = cache
            .bollinger(symbol, timeframe, price_type, period, std_factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok((
            upper.to_pyarray(py),
            middle.to_pyarray(py),
            lower.to_pyarray(py),
        ))
    }

    /// Calculate Bollinger Bands with stepwise semantics.
    #[pyo3(signature = (symbol, timeframe, price_type, period, std_factor, new_bar_indices))]
    pub fn bollinger_stepwise<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        std_factor: f64,
        new_bar_indices: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let (upper, middle, lower) = cache
            .bollinger_stepwise(
                symbol,
                timeframe,
                price_type,
                period,
                std_factor,
                &new_bar_indices,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok((
            upper.to_pyarray(py),
            middle.to_pyarray(py),
            lower.to_pyarray(py),
        ))
    }

    /// Calculate Directional Movement Index (DMI).
    ///
    /// Returns tuple of (+DI, -DI, ADX) numpy arrays.
    #[pyo3(signature = (symbol, timeframe, price_type, period))]
    pub fn dmi<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let (plus_di, minus_di, adx) = cache
            .dmi(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok((
            plus_di.to_pyarray(py),
            minus_di.to_pyarray(py),
            adx.to_pyarray(py),
        ))
    }

    /// Calculate MACD.
    ///
    /// Returns tuple of (macd_line, signal_line, histogram) numpy arrays.
    #[pyo3(signature = (symbol, timeframe, price_type, fast_span=12, slow_span=26, signal_span=9))]
    pub fn macd<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        fast_span: usize,
        slow_span: usize,
        signal_span: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let (macd, signal, histogram) = cache
            .macd(symbol, timeframe, price_type, fast_span, slow_span, signal_span)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok((
            macd.to_pyarray(py),
            signal.to_pyarray(py),
            histogram.to_pyarray(py),
        ))
    }

    /// Calculate Rate of Change (ROC).
    #[pyo3(signature = (symbol, timeframe, price_type, period))]
    pub fn roc<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .roc(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Momentum.
    #[pyo3(signature = (symbol, timeframe, price_type, period))]
    pub fn momentum<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .momentum(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Relative Strength Index (RSI).
    ///
    /// Uses Wilder smoothing (same as TradingView/MT5).
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `timeframe` - Timeframe
    /// * `price_type` - Price type
    /// * `period` - RSI period (typically 14)
    #[pyo3(signature = (symbol, timeframe, price_type, period=14))]
    pub fn rsi<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .rsi(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Z-Score.
    #[pyo3(signature = (symbol, timeframe, price_type, period, ddof=1))]
    pub fn zscore<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        ddof: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .zscore(symbol, timeframe, price_type, period, ddof)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Choppiness Index.
    #[pyo3(signature = (symbol, timeframe, price_type, period))]
    pub fn choppiness<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .choppiness(symbol, timeframe, price_type, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Kalman-filtered mean.
    #[pyo3(signature = (symbol, timeframe, price_type, process_variance=0.01, measurement_variance=1.0))]
    pub fn kalman_mean<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        process_variance: f64,
        measurement_variance: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .kalman_mean(
                symbol,
                timeframe,
                price_type,
                process_variance,
                measurement_variance,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Kalman Z-Score.
    #[pyo3(signature = (symbol, timeframe, price_type, process_variance=0.01, measurement_variance=1.0))]
    pub fn kalman_zscore<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        process_variance: f64,
        measurement_variance: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .kalman_zscore(
                symbol,
                timeframe,
                price_type,
                process_variance,
                measurement_variance,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Rolling Standard Deviation.
    #[pyo3(signature = (symbol, timeframe, price_type, period, ddof=1))]
    pub fn rolling_std<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        period: usize,
        ddof: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .rolling_std(symbol, timeframe, price_type, period, ddof)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    // ========================================================================
    // GARCH & Combined Indicators (Wave 1 - Python-only indicators in Rust)
    // ========================================================================

    /// Calculate GARCH(1,1) volatility on returns.
    ///
    /// Returns volatility (σ) of returns, not prices.
    #[pyo3(signature = (symbol, timeframe, price_type, alpha=0.05, beta=0.90, omega=None, use_log_returns=true, scale=100.0, min_periods=50, sigma_floor=1e-6))]
    #[allow(clippy::too_many_arguments)]
    pub fn garch_volatility<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .garch_volatility(
                symbol,
                timeframe,
                price_type,
                alpha,
                beta,
                omega,
                use_log_returns,
                scale,
                min_periods,
                sigma_floor,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate GARCH(1,1) volatility on a local window.
    #[pyo3(signature = (symbol, timeframe, price_type, idx, lookback=400, alpha=0.05, beta=0.90, omega=None, use_log_returns=true, scale=100.0, min_periods=50, sigma_floor=1e-6))]
    #[allow(clippy::too_many_arguments)]
    pub fn garch_volatility_local<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .garch_volatility_local(
                symbol,
                timeframe,
                price_type,
                idx,
                lookback,
                alpha,
                beta,
                omega,
                use_log_returns,
                scale,
                min_periods,
                sigma_floor,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Get the final sigma value from local GARCH calculation.
    #[pyo3(signature = (symbol, timeframe, price_type, idx, lookback=400, alpha=0.05, beta=0.90, omega=None, use_log_returns=true, scale=100.0, min_periods=50, sigma_floor=1e-6))]
    #[allow(clippy::too_many_arguments)]
    pub fn garch_volatility_local_last(
        &self,
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
    ) -> PyResult<f64> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .garch_volatility_local_last(
                symbol,
                timeframe,
                price_type,
                idx,
                lookback,
                alpha,
                beta,
                omega,
                use_log_returns,
                scale,
                min_periods,
                sigma_floor,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result)
    }

    /// Calculate Kalman Z-Score with stepwise (HTF bar) semantics.
    #[pyo3(signature = (symbol, timeframe, price_type, new_bar_indices, window=100, process_variance=0.01, measurement_variance=1.0))]
    pub fn kalman_zscore_stepwise<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        timeframe: &str,
        price_type: &str,
        new_bar_indices: Vec<usize>,
        window: usize,
        process_variance: f64,
        measurement_variance: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .kalman_zscore_stepwise(
                symbol,
                timeframe,
                price_type,
                &new_bar_indices,
                window,
                process_variance,
                measurement_variance,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate Kalman+GARCH Z-Score.
    ///
    /// Z = (Close - KalmanMean) / (|Close| × σ_return)
    #[pyo3(signature = (symbol, timeframe, price_type, r=0.01, q=1.0, alpha=0.05, beta=0.90, omega=None, use_log_returns=true, scale=100.0, min_periods=50, sigma_floor=1e-6))]
    #[allow(clippy::too_many_arguments)]
    pub fn kalman_garch_zscore<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .kalman_garch_zscore(
                symbol,
                timeframe,
                price_type,
                r,
                q,
                alpha,
                beta,
                omega,
                use_log_returns,
                scale,
                min_periods,
                sigma_floor,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }

    /// Calculate local Kalman+GARCH Z-Score at a specific index.
    #[pyo3(signature = (symbol, timeframe, price_type, idx, lookback=400, r=0.01, q=1.0, alpha=0.05, beta=0.90, omega=None, use_log_returns=true, scale=100.0, min_periods=50, sigma_floor=1e-6))]
    #[allow(clippy::too_many_arguments)]
    pub fn kalman_garch_zscore_local(
        &self,
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
    ) -> PyResult<Option<f64>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .kalman_garch_zscore_local(
                symbol,
                timeframe,
                price_type,
                idx,
                lookback,
                r,
                q,
                alpha,
                beta,
                omega,
                use_log_returns,
                scale,
                min_periods,
                sigma_floor,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result)
    }

    /// Volatility cluster series - routes to ATR or GARCH based on feature type.
    #[pyo3(signature = (symbol, timeframe, price_type, idx, feature, atr_period=14, garch_lookback=400, garch_alpha=0.05, garch_beta=0.90, garch_omega=None, garch_use_log_returns=true, garch_scale=100.0, garch_min_periods=50, garch_sigma_floor=1e-6))]
    #[allow(clippy::too_many_arguments)]
    pub fn vol_cluster_series<'py>(
        &self,
        py: Python<'py>,
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut cache = self
            .inner
            .lock()
            .map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;

        let result = cache
            .vol_cluster_series(
                symbol,
                timeframe,
                price_type,
                idx,
                feature,
                atr_period,
                garch_lookback,
                garch_alpha,
                garch_beta,
                garch_omega,
                garch_use_log_returns,
                garch_scale,
                garch_min_periods,
                garch_sigma_floor,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.to_pyarray(py))
    }
}

impl Default for PyIndicatorCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Register the IndicatorCache PyO3 class.
pub fn register_indicator_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndicatorCache>()?;
    Ok(())
}
