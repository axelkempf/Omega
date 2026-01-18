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
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

/// Indicator cache primitives.
pub mod cache;
/// Multi-timeframe indicator cache and mapping helpers.
pub mod cache_multi_tf;
/// Indicator error types.
pub mod error;
/// Indicator implementations.
pub mod impl_;
/// Indicator factory registry.
pub mod registry;
/// Timeframe mapping utilities.
pub mod timeframe_mapping;
/// Indicator traits and specifications.
pub mod traits;

// Re-export main types
/// Re-export of the single-timeframe indicator cache.
pub use cache::IndicatorCache;
/// Re-export of the multi-timeframe indicator cache and volatility cluster request.
pub use cache_multi_tf::{MultiTfIndicatorCache, VolClusterRequest};
/// Re-export of indicator error type.
pub use error::IndicatorError;
/// Re-export of the indicator registry.
pub use registry::IndicatorRegistry;
/// Re-export of timeframe mapping builder.
pub use timeframe_mapping::build_mapping;
/// Re-export of indicator traits and parameter types.
pub use traits::{
    Indicator, IndicatorParams, IndicatorSpec, IntoMultiVecs, MultiOutputIndicator, PriceSeries,
    TimeframeMapping, ZScoreMeanSource,
};

// Re-export indicator implementations
/// Re-export of indicator implementations.
pub use impl_::{
    atr::ATR,
    bollinger::{BollingerBands, BollingerResult},
    ema::EMA,
    garch_volatility::GarchVolatility,
    garch_volatility_local::{GarchLocalParams, garch_volatility_local},
    kalman_garch_zscore::KalmanGarchZScore,
    kalman_garch_zscore_local::{KalmanGarchLocalParams, kalman_garch_zscore_local},
    kalman_mean::KalmanFilter,
    kalman_zscore::KalmanZScore,
    sma::SMA,
    vol_cluster::{VolCluster, VolClusterState},
    vol_cluster_series::{VolFeatureSeries, vol_cluster_series},
    z_score::ZScore,
};
