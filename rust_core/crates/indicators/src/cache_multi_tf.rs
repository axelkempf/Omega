use std::collections::HashMap;

use omega_types::{Candle, PriceType, Timeframe};

use crate::impl_::atr::ATR;
use crate::impl_::bollinger::BollingerBands;
use crate::impl_::ema::EMA;
use crate::impl_::garch_volatility_local::{
    GarchLocalParams, garch_volatility_local as garch_volatility_local_fn,
};
use crate::impl_::kalman_garch_zscore_local::{
    KalmanGarchLocalParams, kalman_garch_zscore_local as kalman_garch_zscore_local_fn,
};
use crate::impl_::kalman_zscore::KalmanZScore;
use crate::impl_::vol_cluster_series::{VolFeatureSeries, vol_cluster_series};
use crate::impl_::z_score::ZScore;
use crate::traits::ZScoreMeanSource;
use crate::traits::{
    Indicator, IndicatorSpec, MultiOutputIndicator, PriceSeries, TimeframeMapping,
};
use crate::{IndicatorCache, IndicatorParams, build_mapping};

/// Multi-timeframe indicator cache with `price_type` support.
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

/// Request parameters for volatility clustering series.
#[derive(Debug, Clone, Copy)]
pub struct VolClusterRequest<'a> {
    /// Target timeframe.
    pub timeframe: Timeframe,
    /// Price type (bid/ask).
    pub price_type: PriceType,
    /// Primary index for local window end.
    pub idx: usize,
    /// Feature name to compute.
    pub feature: &'a str,
    /// ATR length for feature construction.
    pub atr_length: usize,
    /// Lookback window for GARCH.
    pub garch_lookback: usize,
    /// GARCH parameters.
    pub garch_params: GarchLocalParams,
}

impl MultiTfIndicatorCache {
    /// Creates a new cache from primary candle store and additional timeframe stores.
    #[must_use]
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
            stores.insert(
                (tf, PriceType::Bid),
                build_price_series(tf, PriceType::Bid, &bid),
            );
            stores.insert(
                (tf, PriceType::Ask),
                build_price_series(tf, PriceType::Ask, &ask),
            );
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
    #[must_use]
    pub fn primary_timeframe(&self) -> Timeframe {
        self.primary_timeframe
    }

    /// Returns primary timestamps.
    #[must_use]
    pub fn primary_timestamps(&self) -> &[i64] {
        &self.primary_timestamps
    }

    /// Returns price series for a timeframe and price type.
    #[must_use]
    pub fn get_prices(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&PriceSeries> {
        self.stores.get(&(timeframe, price_type))
    }

    /// Returns candle series for a timeframe and price type.
    #[must_use]
    pub fn get_candles(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&[Candle]> {
        self.candles
            .get(&(timeframe, price_type))
            .map(Vec::as_slice)
    }

    /// Returns mapping for a target timeframe.
    #[must_use]
    pub fn mapping(&self, timeframe: Timeframe) -> Option<&TimeframeMapping> {
        self.mappings.get(&timeframe)
    }

    /// Returns aligned close series (vector) for timeframe + price type.
    #[must_use]
    pub fn get_closes(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&[f64]> {
        self.get_prices(timeframe, price_type)
            .map(|p| p.close.as_slice())
    }

    /// Returns close series aligned to the primary timeframe.
    pub fn closes_aligned(&mut self, timeframe: Timeframe, price_type: PriceType) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "CLOSE_ALIGNED",
            cache_params(timeframe, price_type, [("period", 1)]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(series) = self.get_series(timeframe, price_type) else {
            return vec![f64::NAN; self.primary_timestamps.len()];
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
    #[must_use]
    pub fn get_series(&self, timeframe: Timeframe, price_type: PriceType) -> Option<&PriceSeries> {
        self.get_prices(timeframe, price_type)
    }

    /// Computes ATR on native TF and aligns to primary timeframe.
    pub fn atr(&mut self, timeframe: Timeframe, price_type: PriceType, period: usize) -> Vec<f64> {
        let spec = IndicatorSpec::new(
            "ATR",
            cache_params(timeframe, price_type, [("period", usize_to_i64(period))]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(candles) = self.get_candles(timeframe, price_type) else {
            return vec![f64::NAN; self.primary_timestamps.len()];
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
            cache_params(timeframe, price_type, [("period", usize_to_i64(period))]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(candles) = self.get_candles(timeframe, price_type) else {
            return vec![f64::NAN; self.primary_timestamps.len()];
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
                    ("window", usize_to_i64(window)),
                    ("ema_period", usize_to_i64(ema_period)),
                ],
            ),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(candles) = self.get_candles(timeframe, price_type) else {
            return vec![f64::NAN; self.primary_timestamps.len()];
        };

        let z = ZScore::with_mean_source(window, ZScoreMeanSource::Ema, Some(ema_period))
            .compute(candles);

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
            cache_params(timeframe, price_type, [("period", usize_to_i64(period))]),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(series) = self.get_series(timeframe, price_type) else {
            return vec![f64::NAN; self.primary_timestamps.len()];
        };

        let tf_values = series.close.clone();
        let reduced = reduce_by_mapping(self.mapping(timeframe), &tf_values);

        let ema = EMA::new(period).compute(&candles_from_close(&reduced));
        let expanded =
            expand_by_mapping(self.mapping(timeframe), &ema, self.primary_timestamps.len());

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
            cache_params(
                timeframe,
                price_type,
                [
                    ("period", usize_to_i64(period)),
                    ("std_factor_x100", round_to_i64(std_factor * 100.0)),
                ],
            ),
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

        let Some(series) = self.get_series(timeframe, price_type) else {
            let nan = vec![f64::NAN; self.primary_timestamps.len()];
            return (nan.clone(), nan.clone(), nan);
        };

        let tf_values = series.close.clone();
        let reduced = reduce_by_mapping(self.mapping(timeframe), &tf_values);

        let bb = BollingerBands::new(period, std_factor);
        let result = bb.compute_all(&candles_from_close(&reduced));

        let upper = expand_by_mapping(
            self.mapping(timeframe),
            &result.upper,
            self.primary_timestamps.len(),
        );
        let middle = expand_by_mapping(
            self.mapping(timeframe),
            &result.middle,
            self.primary_timestamps.len(),
        );
        let lower = expand_by_mapping(
            self.mapping(timeframe),
            &result.lower,
            self.primary_timestamps.len(),
        );

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
            cache_params(
                timeframe,
                price_type,
                [
                    ("window", usize_to_i64(window)),
                    ("r_x1000", round_to_i64(r * 1000.0)),
                    ("q_x1000", round_to_i64(q * 1000.0)),
                ],
            ),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(series) = self.get_series(timeframe, price_type) else {
            return vec![f64::NAN; self.primary_timestamps.len()];
        };

        let tf_values = series.close.clone();
        let reduced = reduce_by_mapping(self.mapping(timeframe), &tf_values);

        let kz = KalmanZScore::new(window, r, q);
        let z = kz.compute(&candles_from_close(&reduced));
        let expanded =
            expand_by_mapping(self.mapping(timeframe), &z, self.primary_timestamps.len());

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
                    ("idx", usize_to_i64(idx)),
                    ("lookback", usize_to_i64(lookback)),
                    ("alpha_x1000", round_to_i64(params.alpha * 1000.0)),
                    ("beta_x1000", round_to_i64(params.beta * 1000.0)),
                    (
                        "omega_x1000000",
                        params.omega.map_or(-1, |v| round_to_i64(v * 1_000_000.0)),
                    ),
                    ("use_log_returns", bool_code(params.use_log_returns)),
                    ("scale_x100", round_to_i64(params.scale * 100.0)),
                    ("min_periods", usize_to_i64(params.min_periods)),
                    ("sigma_floor_x1e8", round_to_i64(params.sigma_floor * 1e8)),
                ],
            ),
        );
        if let Some(values) = self.cache.get(&spec) {
            return values.clone();
        }

        let Some(series) = self.get_series(timeframe, price_type) else {
            return Vec::new();
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
                    ("idx", usize_to_i64(idx)),
                    ("lookback", usize_to_i64(lookback)),
                    ("r_x1000", round_to_i64(params.r * 1000.0)),
                    ("q_x1000", round_to_i64(params.q * 1000.0)),
                    ("alpha_x1000", round_to_i64(params.alpha * 1000.0)),
                    ("beta_x1000", round_to_i64(params.beta * 1000.0)),
                    (
                        "omega_x1000000",
                        params.omega.map_or(-1, |v| round_to_i64(v * 1_000_000.0)),
                    ),
                    ("use_log_returns", bool_code(params.use_log_returns)),
                    ("scale_x100", round_to_i64(params.scale * 100.0)),
                    ("min_periods", usize_to_i64(params.min_periods)),
                    ("sigma_floor_x1e8", round_to_i64(params.sigma_floor * 1e8)),
                ],
            ),
        );
        if let Some(value) = self.local_scalar_cache.get(&spec) {
            return *value;
        }

        let Some(series) = self.get_series(timeframe, price_type) else {
            self.local_scalar_cache.insert(spec, None);
            return None;
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
        request: VolClusterRequest<'_>,
    ) -> Option<VolFeatureSeries> {
        let garch_params = request.garch_params;
        let spec = IndicatorSpec::new(
            "VOL_CLUSTER_FEATURE",
            cache_params(
                request.timeframe,
                request.price_type,
                [
                    ("idx", usize_to_i64(request.idx)),
                    ("atr_length", usize_to_i64(request.atr_length)),
                    ("garch_lookback", usize_to_i64(request.garch_lookback)),
                    ("alpha_x1000", round_to_i64(garch_params.alpha * 1000.0)),
                    ("beta_x1000", round_to_i64(garch_params.beta * 1000.0)),
                    (
                        "omega_x1000000",
                        garch_params
                            .omega
                            .map_or(-1, |v| round_to_i64(v * 1_000_000.0)),
                    ),
                    ("use_log_returns", bool_code(garch_params.use_log_returns)),
                    ("scale_x100", round_to_i64(garch_params.scale * 100.0)),
                    ("min_periods", usize_to_i64(garch_params.min_periods)),
                    (
                        "sigma_floor_x1e8",
                        round_to_i64(garch_params.sigma_floor * 1e8),
                    ),
                ],
            ),
        );
        if let Some(series) = self.vol_feature_cache.get(&spec) {
            return Some(series.clone());
        }

        let atr_series = self.atr(request.timeframe, request.price_type, request.atr_length);
        let end_pos = request.idx + 1;
        let start_pos = end_pos.saturating_sub(request.garch_lookback.max(1));
        let local_sigma = self.garch_volatility_local(
            request.timeframe,
            request.price_type,
            request.idx,
            request.garch_lookback,
            garch_params,
        );

        let series = vol_cluster_series(request.feature, atr_series, local_sigma, start_pos);
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
    custom.push(("tf_sec".to_string(), u64_to_i64(timeframe.to_seconds())));
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
    i64::from(value)
}

fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn u64_to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[allow(clippy::cast_possible_truncation)]
fn round_to_i64(value: f64) -> i64 {
    if value.is_finite() {
        value.round() as i64
    } else {
        0
    }
}

fn build_price_series(
    timeframe: Timeframe,
    price_type: PriceType,
    candles: &[Candle],
) -> PriceSeries {
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
                if let Some(idx) = *maybe_idx
                    && last_idx != Some(idx)
                {
                    reduced.push(series[idx]);
                    last_idx = Some(idx);
                }
            }
            reduced
        }
    }
}

fn map_series(mapping: Option<&TimeframeMapping>, series: &[f64], target_len: usize) -> Vec<f64> {
    let reduced = reduce_by_mapping(mapping, series);
    expand_by_mapping(mapping, &reduced, target_len)
}

fn expand_by_mapping(
    mapping: Option<&TimeframeMapping>,
    reduced: &[f64],
    target_len: usize,
) -> Vec<f64> {
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
            timestamp_ns: usize_to_i64(idx),
            open: value,
            high: value,
            low: value,
            close: value,
            volume: 0.0,
        })
        .collect()
}
