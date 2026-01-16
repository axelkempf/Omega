//! Omega Indicators
//!
//! Technical indicator engine for the Omega trading system.
//! Provides all indicators needed for Mean Reversion Z-Score and other strategies.
//!
//! # Features
//! - Indicator trait with vectorized computation
//! - Multi-output indicators (e.g., Bollinger Bands)
//! - Caching system for computed indicators
//! - Registry for indicator factories
//!
//! # Available Indicators
//! - SMA: Simple Moving Average
//! - EMA: Exponential Moving Average
//! - ATR: Average True Range (Wilder smoothing)
//! - Bollinger Bands: Upper, Middle, Lower bands
//! - Z-Score: Standard deviation normalized score
//! - Kalman Filter: Level estimation and Z-Score
//! - GARCH(1,1): Volatility estimation
//! - Vol-Cluster: Volatility regime detection

#![deny(clippy::all)]

pub mod cache;
pub mod cache_multi_tf;
pub mod error;
pub mod impl_;
pub mod registry;
pub mod traits;
pub mod timeframe_mapping;

// Re-export main types
pub use cache::IndicatorCache;
pub use cache_multi_tf::{MultiTfIndicatorCache, VolClusterRequest};
pub use error::IndicatorError;
pub use registry::IndicatorRegistry;
pub use traits::{
    Indicator, IndicatorParams, IndicatorSpec, IntoMultiVecs, MultiOutputIndicator, PriceSeries,
    TimeframeMapping,
};
pub use timeframe_mapping::build_mapping;

// Re-export indicator implementations
pub use impl_::{
    atr::ATR,
    bollinger::{BollingerBands, BollingerResult},
    ema::EMA,
    garch_volatility::GarchVolatility,
    garch_volatility_local::{garch_volatility_local, GarchLocalParams},
    kalman_garch_zscore::KalmanGarchZScore,
    kalman_garch_zscore_local::{kalman_garch_zscore_local, KalmanGarchLocalParams},
    kalman_mean::KalmanFilter,
    kalman_zscore::KalmanZScore,
    sma::SMA,
    vol_cluster::{VolCluster, VolClusterState},
    vol_cluster_series::{vol_cluster_series, VolFeatureSeries},
    z_score::ZScore,
};
