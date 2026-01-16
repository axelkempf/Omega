use std::collections::HashMap;

use omega_types::{Candle, PriceType, Timeframe};

use crate::impl_::atr::ATR;
use crate::impl_::bollinger::BollingerBands;
use crate::impl_::ema::EMA;
use crate::impl_::garch_volatility_local::{
    garch_volatility_local as garch_volatility_local_fn, GarchLocalParams,
};
use crate::impl_::kalman_garch_zscore_local::{
    kalman_garch_zscore_local as kalman_garch_zscore_local_fn, KalmanGarchLocalParams,
};
use crate::impl_::kalman_zscore::KalmanZScore;
use crate::impl_::vol_cluster_series::{vol_cluster_series, VolFeatureSeries};
use crate::traits::{
    Indicator, IndicatorSpec, MultiOutputIndicator, PriceSeries, TimeframeMapping,
};
use crate::{build_mapping, IndicatorCache, IndicatorParams};

/// Multi-timeframe indicator cache with price_type support.
#[derive(Debug)]
pub struct MultiTfIndicatorCache {
    primary_timeframe: Timeframe,
    primary_timestamps: Vec<i64>,
    stores: HashMap<(Timeframe, PriceType), PriceSeries>,
    candles: HashMap<(Timeframe, PriceType), Vec<Candle>>,
    mappings: HashMap<Timeframe, TimeframeMapping>,
    cache: IndicatorCache,
    vol_feature_cache: HashMap<IndicatorSpec, VolFeatureSeries>,
    local_scalar_cache: HashMap<IndicatorSpec, Option<f64>>,
}

impl MultiTfIndicatorCache {
    /// Creates a new cache from primary candle store and additional timeframe stores.
    pub fn new(
        primary_timeframe: Timeframe,
        primary_bid: Vec<Candle>,
        primary_ask: Vec<Candle>,
        additional: Vec<(Timeframe, Vec<Candle>, Vec<Candle>)>,
    ) -> Self {
        let primary_timestamps: Vec<i64> = primary_bid.iter().map(|c| c.timestamp_ns).collect();
        let mut stores = HashMap::new();
        let mut candles = HashMap::new();

        stores.insert(
            (primary_timeframe, PriceType::Bid),
            build_price_series(primary_timeframe, PriceType::Bid, &primary_bid),
        );
        stores.insert(
            (primary_timeframe, PriceType::Ask),
            build_price_series(primary_timeframe, PriceType::Ask, &primary_ask),
        );
        candles.insert((primary_timeframe, PriceType::Bid), primary_bid);
        candles.insert((primary_timeframe, PriceType::Ask), primary_ask);

        let mut mappings = HashMap::new();

        for (tf, bid, ask) in additional {
            let tf_timestamps: Vec<i64> = bid.iter().map(|c| c.timestamp_ns).collect();
            let mapping = build_mapping(&primary_timestamps, &tf_timestamps, tf);
            mappings.insert(tf, mapping);
            stores.insert((tf, PriceType::Bid), build_price_series(tf, PriceType::Bid, &bid));
            stores.insert((tf, PriceType::Ask), build_price_series(tf, PriceType::Ask, &ask));
            candles.insert((tf, PriceType::Bid), bid);
            candles.insert((tf, PriceType::Ask), ask);
        }

        Self {
            primary_timeframe,
            primary_timestamps,
            stores,
            candles,
            mappings,
            cache: IndicatorCache::new(),
            vol_feature_cache: HashMap::new(),
            local_scalar_cache: HashMap::new(),
        }
    }

    /// Returns primary timeframe.
    pub fn primary_timeframe(&self) -> Timeframe {
        self.primary_timeframe
    }

    /// Returns primary timestamps.
    pub fn primary_timestamps(&self) -> &[i64] {
        &self.primary_timestamps
    }

    /// Returns price series for a timeframe and price type.
    pub fn get_prices(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&PriceSeries> {
        self.stores.get(&(timeframe, price_type))
    }

    /// Returns candle series for a timeframe and price type.
    pub fn get_candles(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&[Candle]> {
        self.candles
            .get(&(timeframe, price_type))
            .map(|c| c.as_slice())
    }

    /// Returns mapping for a target timeframe.
    pub fn mapping(&self, timeframe: Timeframe) -> Option<&TimeframeMapping> {
        self.mappings.get(&timeframe)
    }

    /// Returns aligned close series (vector) for timeframe + price type.
    pub fn get_closes(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&[f64]> {
        self.get_prices(timeframe, price_type).map(|p| p.close.as_slice())
    }

    /// Returns close series aligned to the primary timeframe.
    pub fn closes_aligned(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
    ) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "CLOSE_ALIGNED",
            cache_params(timeframe, price_type, [("period", 1)]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let series = match self.get_series(timeframe, price_type) {
            Some(series) => series,
            None => return vec![f64::NAN; self.primary_timestamps.len()],
        };

        let aligned = map_series(
            self.mapping(timeframe),
            &series.close,
            self.primary_timestamps.len(),
        );
        self.cache.insert(spec, aligned.clone());
        aligned
    }

    /// Returns aligned OHLC close series for timeframe + price type.
    pub fn get_series(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&PriceSeries> {
        self.get_prices(timeframe, price_type)
    }

    /// Computes ATR on native TF and aligns to primary timeframe.
    pub fn atr(&mut self, timeframe: Timeframe, price_type: PriceType, period: usize) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "ATR",
            cache_params(timeframe, price_type, [("period", period as i64)]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let candles = match self.get_candles(timeframe, price_type) {
            Some(candles) => candles,
            None => return vec![f64::NAN; self.primary_timestamps.len()],
        };

        let atr = ATR::new(period).compute(candles);
        let aligned = map_series(self.mapping(timeframe), &atr, self.primary_timestamps.len());
        self.cache.insert(spec, aligned.clone());
        aligned
    }

    /// Computes EMA on native TF and aligns to primary timeframe.
    pub fn ema(&mut self, timeframe: Timeframe, price_type: PriceType, period: usize) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "EMA",
            cache_params(timeframe, price_type, [("period", period as i64)]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let candles = match self.get_candles(timeframe, price_type) {
            Some(candles) => candles,
            None => return vec![f64::NAN; self.primary_timestamps.len()],
        };

        let ema = EMA::new(period).compute(candles);
        let aligned = map_series(self.mapping(timeframe), &ema, self.primary_timestamps.len());
        self.cache.insert(spec, aligned.clone());
        aligned
    }

    /// Computes Z-Score using EMA mean (pandas ewm adjust=false) and aligns to primary.
    pub fn zscore_ema(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        window: usize,
        ema_period: usize,
    ) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "Z_SCORE_EMA",
            cache_params(
                timeframe,
                price_type,
                [
                    ("window", window as i64),
                    ("ema_period", ema_period as i64),
                ],
            ),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let candles = match self.get_candles(timeframe, price_type) {
            Some(candles) => candles,
            None => return vec![f64::NAN; self.primary_timestamps.len()],
        };

        let ema = EMA::new(ema_period).compute(candles);
        let mut residuals = Vec::with_capacity(candles.len());
        for (candle, ema_value) in candles.iter().zip(ema.iter()) {
            residuals.push(candle.close - ema_value);
        }

        let mut z = vec![f64::NAN; residuals.len()];
        if window > 0 && residuals.len() >= window {
            for i in (window - 1)..residuals.len() {
                let start = i + 1 - window;
                let window_vals = &residuals[start..=i];
                let std = sample_std(window_vals);
                if std.is_finite() && std > 0.0 {
                    z[i] = residuals[i] / std;
                }
            }
        }

        let aligned = map_series(self.mapping(timeframe), &z, self.primary_timestamps.len());
        self.cache.insert(spec, aligned.clone());
        aligned
    }

    /// Computes EMA stepwise (update on new TF bar, forward-fill across primary).
    pub fn ema_stepwise(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        period: usize,
    ) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "EMA_STEPWISE",
            cache_params(timeframe, price_type, [
                ("period", period as i64),
            ]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let series = match self.get_series(timeframe, price_type) {
            Some(series) => series,
            None => return vec![f64::NAN; self.primary_timestamps.len()],
        };

        let tf_values = series.close.clone();
        let reduced = reduce_by_mapping(self.mapping(timeframe), &tf_values);

        let ema = EMA::new(period).compute(&candles_from_close(&reduced));
        let expanded = expand_by_mapping(self.mapping(timeframe), &ema, self.primary_timestamps.len());

        self.cache.insert(spec, expanded.clone());
        expanded
    }

    /// Computes Bollinger stepwise (update on new TF bar, forward-fill across primary).
    pub fn bollinger_stepwise(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        period: usize,
        std_factor: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let spec = IndicatorSpec::new(
            "BOLLINGER_STEPWISE",
            cache_params(timeframe, price_type, [
                ("period", period as i64),
                ("std_factor_x100", (std_factor * 100.0).round() as i64),
            ]),
        );

        let upper_spec = spec.with_output_suffix("upper");
        let mid_spec = spec.with_output_suffix("middle");
        let lower_spec = spec.with_output_suffix("lower");

        if let (Some(u), Some(m), Some(l)) = (
            self.cache.get(&upper_spec),
            self.cache.get(&mid_spec),
            self.cache.get(&lower_spec),
        ) {
            return (u.clone(), m.clone(), l.clone());
        }

        let series = match self.get_series(timeframe, price_type) {
            Some(series) => series,
            None => {
                let nan = vec![f64::NAN; self.primary_timestamps.len()];
                return (nan.clone(), nan.clone(), nan);
            }
        };

        let tf_values = series.close.clone();
        let reduced = reduce_by_mapping(self.mapping(timeframe), &tf_values);

        let bb = BollingerBands::new(period, std_factor);
        let result = bb.compute_all(&candles_from_close(&reduced));

        let upper = expand_by_mapping(self.mapping(timeframe), &result.upper, self.primary_timestamps.len());
        let middle = expand_by_mapping(self.mapping(timeframe), &result.middle, self.primary_timestamps.len());
        let lower = expand_by_mapping(self.mapping(timeframe), &result.lower, self.primary_timestamps.len());

        self.cache.insert(upper_spec, upper.clone());
        self.cache.insert(mid_spec, middle.clone());
        self.cache.insert(lower_spec, lower.clone());

        (upper, middle, lower)
    }

    /// Computes Kalman Z-Score stepwise (update on new TF bar, forward-fill across primary).
    pub fn kalman_zscore_stepwise(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        window: usize,
        r: f64,
        q: f64,
    ) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "KALMAN_ZSCORE_STEPWISE",
            cache_params(timeframe, price_type, [
                ("window", window as i64),
                ("r_x1000", (r * 1000.0).round() as i64),
                ("q_x1000", (q * 1000.0).round() as i64),
            ]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let series = match self.get_series(timeframe, price_type) {
            Some(series) => series,
            None => return vec![f64::NAN; self.primary_timestamps.len()],
        };

        let tf_values = series.close.clone();
        let reduced = reduce_by_mapping(self.mapping(timeframe), &tf_values);

        let kz = KalmanZScore::new(window, r, q);
        let z = kz.compute(&candles_from_close(&reduced));
        let expanded = expand_by_mapping(self.mapping(timeframe), &z, self.primary_timestamps.len());

        self.cache.insert(spec, expanded.clone());
        expanded
    }

    /// Computes local GARCH volatility window ending at idx (primary index).
    pub fn garch_volatility_local(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        idx: usize,
        lookback: usize,
        params: GarchLocalParams,
    ) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "GARCH_VOL_LOCAL",
            cache_params(
                timeframe,
                price_type,
                [
                    ("idx", idx as i64),
                    ("lookback", lookback as i64),
                    ("alpha_x1000", (params.alpha * 1000.0).round() as i64),
                    ("beta_x1000", (params.beta * 1000.0).round() as i64),
                    (
                        "omega_x1000000",
                        params
                            .omega
                            .map(|v| (v * 1_000_000.0).round() as i64)
                            .unwrap_or(-1),
                    ),
                    ("use_log_returns", bool_code(params.use_log_returns)),
                    ("scale_x100", (params.scale * 100.0).round() as i64),
                    ("min_periods", params.min_periods as i64),
                    (
                        "sigma_floor_x1e8",
                        (params.sigma_floor * 1e8).round() as i64,
                    ),
                ],
            ),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let series = match self.get_series(timeframe, price_type) {
            Some(series) => series,
            None => return Vec::new(),
        };
        let aligned = map_series(
            self.mapping(timeframe),
            &series.close,
            self.primary_timestamps.len(),
        );
        if aligned.is_empty() || idx >= aligned.len() {
            return Vec::new();
        }

        let values = garch_volatility_local_fn(&aligned, idx, lookback, params);
        self.cache.insert(spec, values.clone());
        values
    }

    /// Computes local Kalman+GARCH Z-Score at idx (primary index).
    pub fn kalman_garch_zscore_local(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        idx: usize,
        lookback: usize,
        params: KalmanGarchLocalParams,
    ) -> Option<f64> {
        let spec = IndicatorSpec::new(
            "KALMAN_GARCH_Z_LOCAL",
            cache_params(
                timeframe,
                price_type,
                [
                    ("idx", idx as i64),
                    ("lookback", lookback as i64),
                    ("r_x1000", (params.r * 1000.0).round() as i64),
                    ("q_x1000", (params.q * 1000.0).round() as i64),
                    ("alpha_x1000", (params.alpha * 1000.0).round() as i64),
                    ("beta_x1000", (params.beta * 1000.0).round() as i64),
                    (
                        "omega_x1000000",
                        params
                            .omega
                            .map(|v| (v * 1_000_000.0).round() as i64)
                            .unwrap_or(-1),
                    ),
                    ("use_log_returns", bool_code(params.use_log_returns)),
                    ("scale_x100", (params.scale * 100.0).round() as i64),
                    ("min_periods", params.min_periods as i64),
                    (
                        "sigma_floor_x1e8",
                        (params.sigma_floor * 1e8).round() as i64,
                    ),
                ],
            ),
        );
        if let Some(value) = self.local_scalar_cache.get(&spec) {
            return *value;
        }

        let series = match self.get_series(timeframe, price_type) {
            Some(series) => series,
            None => {
                self.local_scalar_cache.insert(spec, None);
                return None;
            }
        };
        let aligned = map_series(
            self.mapping(timeframe),
            &series.close,
            self.primary_timestamps.len(),
        );
        let value = kalman_garch_zscore_local_fn(&aligned, idx, lookback, params);
        self.local_scalar_cache.insert(spec, value);
        value
    }

    /// Returns volatility feature series for clustering.
    pub fn vol_cluster_series(
        &mut self,
        timeframe: Timeframe,
        price_type: PriceType,
        idx: usize,
        feature: &str,
        atr_length: usize,
        garch_lookback: usize,
        garch_params: GarchLocalParams,
    ) -> Option<VolFeatureSeries> {
        let spec = IndicatorSpec::new(
            "VOL_CLUSTER_FEATURE",
            cache_params(
                timeframe,
                price_type,
                [
                    ("idx", idx as i64),
                    ("atr_length", atr_length as i64),
                    ("garch_lookback", garch_lookback as i64),
                    ("alpha_x1000", (garch_params.alpha * 1000.0).round() as i64),
                    ("beta_x1000", (garch_params.beta * 1000.0).round() as i64),
                    (
                        "omega_x1000000",
                        garch_params
                            .omega
                            .map(|v| (v * 1_000_000.0).round() as i64)
                            .unwrap_or(-1),
                    ),
                    ("use_log_returns", bool_code(garch_params.use_log_returns)),
                    ("scale_x100", (garch_params.scale * 100.0).round() as i64),
                    ("min_periods", garch_params.min_periods as i64),
                    (
                        "sigma_floor_x1e8",
                        (garch_params.sigma_floor * 1e8).round() as i64,
                    ),
                ],
            ),
        );
        if let Some(series) = self.vol_feature_cache.get(&spec) {
            return Some(series.clone());
        }

        let atr_series = self.atr(timeframe, price_type, atr_length);
        let end_pos = idx + 1;
        let start_pos = end_pos.saturating_sub(garch_lookback.max(1));
        let local_sigma = self.garch_volatility_local(
            timeframe,
            price_type,
            idx,
            garch_lookback,
            garch_params,
        );

        let series = vol_cluster_series(feature, atr_series, local_sigma, start_pos);
        if let Some(series) = series.clone() {
            self.vol_feature_cache.insert(spec, series.clone());
        }
        series
    }
}

fn cache_params(
    timeframe: Timeframe,
    price_type: PriceType,
    params: impl IntoIterator<Item = (&'static str, i64)>,
) -> IndicatorParams {
    let mut custom = Vec::new();
    custom.push(("tf_sec".to_string(), timeframe.to_seconds() as i64));
    custom.push(("price_type".to_string(), price_type_code(price_type)));
    for (key, value) in params {
        custom.push((key.to_string(), value));
    }
    IndicatorParams::Custom(custom)
}

fn price_type_code(price_type: PriceType) -> i64 {
    match price_type {
        PriceType::Bid => 0,
        PriceType::Ask => 1,
    }
}

fn bool_code(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

fn build_price_series(timeframe: Timeframe, price_type: PriceType, candles: &[Candle]) -> PriceSeries {
    let mut open = Vec::with_capacity(candles.len());
    let mut high = Vec::with_capacity(candles.len());
    let mut low = Vec::with_capacity(candles.len());
    let mut close = Vec::with_capacity(candles.len());

    for candle in candles {
        open.push(candle.open);
        high.push(candle.high);
        low.push(candle.low);
        close.push(candle.close);
    }

    PriceSeries {
        timeframe,
        price_type,
        close,
        open,
        high,
        low,
    }
}

fn reduce_by_mapping(mapping: Option<&TimeframeMapping>, series: &[f64]) -> Vec<f64> {
    match mapping {
        None => series.to_vec(),
        Some(map) => {
            let mut reduced = Vec::new();
            let mut last_idx = None;
            for maybe_idx in &map.primary_to_target {
                if let Some(idx) = *maybe_idx {
                    if last_idx != Some(idx) {
                        reduced.push(series[idx]);
                        last_idx = Some(idx);
                    }
                }
            }
            reduced
        }
    }
}

fn map_series(
    mapping: Option<&TimeframeMapping>,
    series: &[f64],
    target_len: usize,
) -> Vec<f64> {
    let reduced = reduce_by_mapping(mapping, series);
    expand_by_mapping(mapping, &reduced, target_len)
}

fn expand_by_mapping(mapping: Option<&TimeframeMapping>, reduced: &[f64], target_len: usize) -> Vec<f64> {
    match mapping {
        None => reduced.to_vec(),
        Some(map) => {
            let mut expanded = vec![f64::NAN; target_len];
            let mut reduced_idx = 0usize;
            let mut last_target = None;

            for (i, maybe_target) in map.primary_to_target.iter().enumerate() {
                if let Some(target_idx) = *maybe_target {
                    if last_target != Some(target_idx) {
                        if reduced_idx < reduced.len() {
                            expanded[i] = reduced[reduced_idx];
                            reduced_idx += 1;
                        }
                        last_target = Some(target_idx);
                    } else if reduced_idx > 0 {
                        expanded[i] = reduced[reduced_idx - 1];
                    }
                }
            }

            // Forward-fill missing values after first finite value
            let mut last = None;
            for v in &mut expanded {
                if v.is_finite() {
                    last = Some(*v);
                } else if let Some(l) = last {
                    *v = l;
                }
            }

            expanded
        }
    }
}

fn candles_from_close(close: &[f64]) -> Vec<Candle> {
    close
        .iter()
        .enumerate()
        .map(|(idx, &value)| Candle {
            timestamp_ns: idx as i64,
            open: value,
            high: value,
            low: value,
            close: value,
            volume: 0.0,
        })
        .collect()
}

fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let denom = values.len() as f64 - 1.0;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let variance = values
        .iter()
        .map(|v| (*v - mean).powi(2))
        .sum::<f64>()
        / denom;
    variance.sqrt()
}
