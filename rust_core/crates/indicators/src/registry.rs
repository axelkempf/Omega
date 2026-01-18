//! Indicator registry for dynamic indicator creation.

use crate::error::IndicatorError;
use crate::impl_::{
    atr::ATR, ema::EMA, garch_volatility::GarchVolatility, kalman_garch_zscore::KalmanGarchZScore,
    kalman_zscore::KalmanZScore, sma::SMA, vol_cluster::VolCluster, z_score::ZScore,
};
use crate::traits::{Indicator, IndicatorParams, IndicatorSpec, ZScoreMeanSource};
use std::collections::HashMap;
use std::sync::Arc;

/// Factory function type for creating indicators from parameters.
pub type IndicatorFactory =
    Box<dyn Fn(&IndicatorParams) -> Result<Arc<dyn Indicator>, IndicatorError> + Send + Sync>;

/// Registry for indicator factories.
///
/// Allows dynamic creation of indicators by name and parameters.
/// Pre-populated with all MRZ (Mean Reversion Z-Score) indicators.
pub struct IndicatorRegistry {
    /// Indicator factories by name.
    factories: HashMap<String, IndicatorFactory>,
}

impl IndicatorRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Registers an indicator factory.
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&IndicatorParams) -> Result<Arc<dyn Indicator>, IndicatorError>
            + Send
            + Sync
            + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// Creates an indicator from a specification.
    ///
    /// # Errors
    ///
    /// Returns [`IndicatorError::UnknownIndicator`] if the name is not registered
    /// and [`IndicatorError::InvalidParams`] when parameters do not match.
    pub fn create(&self, spec: &IndicatorSpec) -> Result<Arc<dyn Indicator>, IndicatorError> {
        let factory = self
            .factories
            .get(&spec.name)
            .ok_or_else(|| IndicatorError::UnknownIndicator(spec.name.clone()))?;
        factory(&spec.params)
    }

    /// Checks if an indicator is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Returns list of registered indicator names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(String::as_str).collect()
    }

    /// Creates a registry with all default MRZ indicators pre-registered.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // SMA
        registry.register("SMA", |params| match params {
            IndicatorParams::Period(period) => Ok(Arc::new(SMA::new(*period))),
            _ => Err(IndicatorError::invalid_params("SMA requires Period params")),
        });

        // EMA
        registry.register("EMA", |params| match params {
            IndicatorParams::Period(period) => Ok(Arc::new(EMA::new(*period))),
            _ => Err(IndicatorError::invalid_params("EMA requires Period params")),
        });

        // ATR
        registry.register("ATR", |params| match params {
            IndicatorParams::Period(period) => Ok(Arc::new(ATR::new(*period))),
            _ => Err(IndicatorError::invalid_params("ATR requires Period params")),
        });

        // Z-Score
        registry.register("Z_SCORE", |params| match params {
            IndicatorParams::Period(period) => Ok(Arc::new(ZScore::new(*period))),
            IndicatorParams::ZScore {
                window,
                mean_source,
                ema_period,
            } => match mean_source {
                ZScoreMeanSource::Rolling => Ok(Arc::new(ZScore::with_mean_source(
                    *window,
                    *mean_source,
                    None,
                ))),
                ZScoreMeanSource::Ema => {
                    let ema_period = ema_period.ok_or_else(|| {
                        IndicatorError::invalid_params(
                            "Z_SCORE mean_source=Ema requires ema_period",
                        )
                    })?;
                    if ema_period == 0 {
                        return Err(IndicatorError::invalid_params(
                            "Z_SCORE ema_period must be > 0",
                        ));
                    }
                    Ok(Arc::new(ZScore::with_mean_source(
                        *window,
                        *mean_source,
                        Some(ema_period),
                    )))
                }
            },
            _ => Err(IndicatorError::invalid_params(
                "Z_SCORE requires Period or ZScore params",
            )),
        });

        // Kalman Z-Score
        registry.register("KALMAN_Z", |params| match params {
            IndicatorParams::Kalman {
                window,
                r_x1000,
                q_x1000,
            } => Ok(Arc::new(KalmanZScore::from_x1000(
                *window, *r_x1000, *q_x1000,
            ))),
            _ => Err(IndicatorError::invalid_params(
                "KALMAN_Z requires Kalman params",
            )),
        });

        // GARCH Volatility
        registry.register("GARCH_VOL", |params| match params {
            IndicatorParams::Garch {
                alpha_x1000,
                beta_x1000,
                omega_x1000000,
                use_log_returns,
                scale_x100,
                min_periods,
                sigma_floor_x1e8,
            } => Ok(Arc::new(
                GarchVolatility::from_encoded(*alpha_x1000, *beta_x1000, *omega_x1000000)
                    .with_log_returns(*use_log_returns)
                    .with_scale(f64::from(*scale_x100) / 100.0)
                    .with_min_periods(*min_periods)
                    .with_sigma_floor(f64::from(*sigma_floor_x1e8) / 1e8),
            )),
            _ => Err(IndicatorError::invalid_params(
                "GARCH_VOL requires Garch params",
            )),
        });

        // Kalman+GARCH Z-Score
        registry.register("KALMAN_GARCH_Z", |params| match params {
            IndicatorParams::KalmanGarch {
                window: _,
                r_x1000,
                q_x1000,
                alpha_x1000,
                beta_x1000,
                omega_x1000000,
                use_log_returns,
                scale_x100,
                min_periods,
                sigma_floor_x1e8,
            } => Ok(Arc::new(
                KalmanGarchZScore::from_encoded(
                    *r_x1000,
                    *q_x1000,
                    *alpha_x1000,
                    *beta_x1000,
                    *omega_x1000000,
                )
                .with_log_returns(*use_log_returns)
                .with_scale(f64::from(*scale_x100) / 100.0)
                .with_min_periods(*min_periods)
                .with_sigma_floor(f64::from(*sigma_floor_x1e8) / 1e8),
            )),
            _ => Err(IndicatorError::invalid_params(
                "KALMAN_GARCH_Z requires KalmanGarch params",
            )),
        });

        // Vol-Cluster
        registry.register("VOL_CLUSTER", |params| match params {
            IndicatorParams::VolCluster {
                vol_period,
                high_vol_threshold_x100,
                low_vol_threshold_x100,
            } => Ok(Arc::new(VolCluster::from_x100(
                *vol_period,
                *high_vol_threshold_x100,
                *low_vol_threshold_x100,
                50, // Default lookback
            ))),
            _ => Err(IndicatorError::invalid_params(
                "VOL_CLUSTER requires VolCluster params",
            )),
        });

        registry
    }
}

impl Default for IndicatorRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// Note: BollingerBands is a MultiOutputIndicator and doesn't implement Indicator directly,
// so it's handled separately through the cache's get_or_compute_multi method.

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(close: f64) -> omega_types::Candle {
        omega_types::Candle {
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
    fn test_registry_with_defaults() {
        let registry = IndicatorRegistry::with_defaults();

        assert!(registry.contains("SMA"));
        assert!(registry.contains("EMA"));
        assert!(registry.contains("ATR"));
        assert!(registry.contains("Z_SCORE"));
        assert!(registry.contains("KALMAN_Z"));
        assert!(registry.contains("GARCH_VOL"));
        assert!(registry.contains("VOL_CLUSTER"));
        assert!(!registry.contains("UNKNOWN"));
    }

    #[test]
    fn test_registry_create_sma() {
        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new("SMA", IndicatorParams::Period(5));

        let indicator = registry.create(&spec).unwrap();
        assert_eq!(indicator.name(), "SMA");
        assert_eq!(indicator.warmup_periods(), 5);
    }

    #[test]
    fn test_registry_create_ema() {
        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new("EMA", IndicatorParams::Period(10));

        let indicator = registry.create(&spec).unwrap();
        assert_eq!(indicator.name(), "EMA");
        assert_eq!(indicator.warmup_periods(), 10);
    }

    #[test]
    fn test_registry_create_atr() {
        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new("ATR", IndicatorParams::Period(14));

        let indicator = registry.create(&spec).unwrap();
        assert_eq!(indicator.name(), "ATR");
        assert_eq!(indicator.warmup_periods(), 14);
    }

    #[test]
    fn test_registry_create_kalman_z() {
        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new(
            "KALMAN_Z",
            IndicatorParams::Kalman {
                window: 20,
                r_x1000: 500,
                q_x1000: 100,
            },
        );

        let indicator = registry.create(&spec).unwrap();
        assert_eq!(indicator.name(), "KALMAN_Z");
        assert_eq!(indicator.warmup_periods(), 20);
    }

    #[test]
    fn test_registry_create_garch() {
        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new(
            "GARCH_VOL",
            IndicatorParams::Garch {
                alpha_x1000: 100,
                beta_x1000: 850,
                omega_x1000000: 10,
                use_log_returns: true,
                scale_x100: 10_000,
                min_periods: 20,
                sigma_floor_x1e8: 10_000,
            },
        );

        let indicator = registry.create(&spec).unwrap();
        assert_eq!(indicator.name(), "GARCH_VOL");
    }

    #[test]
    fn test_registry_unknown_indicator() {
        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new("UNKNOWN", IndicatorParams::Period(5));

        let result = registry.create(&spec);
        assert!(result.is_err());

        if let Err(IndicatorError::UnknownIndicator(name)) = result {
            assert_eq!(name, "UNKNOWN");
        } else {
            panic!("Expected UnknownIndicator error");
        }
    }

    #[test]
    fn test_registry_invalid_params() {
        let registry = IndicatorRegistry::with_defaults();

        // SMA with wrong params
        let spec = IndicatorSpec::new(
            "SMA",
            IndicatorParams::Kalman {
                window: 20,
                r_x1000: 500,
                q_x1000: 100,
            },
        );

        let result = registry.create(&spec);
        assert!(result.is_err());

        if let Err(IndicatorError::InvalidParams(_)) = result {
            // Expected
        } else {
            panic!("Expected InvalidParams error");
        }
    }

    #[test]
    fn test_registry_compute() {
        let candles: Vec<omega_types::Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let registry = IndicatorRegistry::with_defaults();
        let spec = IndicatorSpec::new("SMA", IndicatorParams::Period(3));

        let indicator = registry.create(&spec).unwrap();
        let result = indicator.compute(&candles);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_registry_custom_indicator() {
        let mut registry = IndicatorRegistry::new();

        // Register a custom indicator
        registry.register("CUSTOM", |params| match params {
            IndicatorParams::Period(period) => Ok(Arc::new(SMA::new(*period * 2))),
            _ => Err(IndicatorError::invalid_params("Custom requires Period")),
        });

        let spec = IndicatorSpec::new("CUSTOM", IndicatorParams::Period(5));
        let indicator = registry.create(&spec).unwrap();

        // Period is doubled
        assert_eq!(indicator.warmup_periods(), 10);
    }
}
