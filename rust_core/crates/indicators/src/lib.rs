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

pub mod cache;
pub mod error;
pub mod impl_;
pub mod registry;
pub mod traits;

// Re-export main types
pub use cache::IndicatorCache;
pub use error::IndicatorError;
pub use registry::IndicatorRegistry;
pub use traits::{Indicator, IndicatorParams, IndicatorSpec, IntoMultiVecs, MultiOutputIndicator};

// Re-export indicator implementations
pub use impl_::{
    atr::ATR,
    bollinger::{BollingerBands, BollingerResult},
    ema::EMA,
    garch_volatility::GarchVolatility,
    kalman_garch_zscore::KalmanGarchZScore,
    kalman_mean::KalmanFilter,
    kalman_zscore::KalmanZScore,
    sma::SMA,
    vol_cluster::{VolCluster, VolClusterState},
    z_score::ZScore,
};
