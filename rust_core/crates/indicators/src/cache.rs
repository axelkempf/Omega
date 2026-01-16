//! Indicator cache for avoiding redundant computations.

use crate::traits::{Indicator, IndicatorSpec, IntoMultiVecs, MultiOutputIndicator};
use omega_types::Candle;
use std::collections::{hash_map::Entry, HashMap};

/// Cache for computed indicator values.
///
/// Stores computed indicator series to avoid redundant calculations.
/// Uses `IndicatorSpec` as cache keys.
#[derive(Debug, Default)]
pub struct IndicatorCache {
    cache: HashMap<IndicatorSpec, Vec<f64>>,
}

impl IndicatorCache {
    /// Creates a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Creates a cache with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
        }
    }

    /// Checks if an indicator is already cached.
    #[must_use]
    pub fn contains(&self, spec: &IndicatorSpec) -> bool {
        self.cache.contains_key(spec)
    }

    /// Gets cached values for an indicator, if present.
    #[must_use]
    pub fn get(&self, spec: &IndicatorSpec) -> Option<&Vec<f64>> {
        self.cache.get(spec)
    }

    /// Gets a single value at index from a cached indicator.
    #[must_use]
    pub fn get_at(&self, spec: &IndicatorSpec, idx: usize) -> Option<f64> {
        self.cache.get(spec).and_then(|v| v.get(idx).copied())
    }

    /// Inserts computed values into the cache.
    pub fn insert(&mut self, spec: IndicatorSpec, values: Vec<f64>) {
        self.cache.insert(spec, values);
    }

    /// Gets or computes a single-output indicator.
    ///
    /// If the indicator is already cached, returns the cached values.
    /// Otherwise, computes the indicator, caches it, and returns the values.
    ///
    pub fn get_or_compute(
        &mut self,
        spec: &IndicatorSpec,
        candles: &[Candle],
        indicator: &dyn Indicator,
    ) -> &[f64] {
        let values = match self.cache.entry(spec.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(indicator.compute(candles)),
        };
        values.as_slice()
    }

    /// Gets or computes a multi-output indicator.
    ///
    /// Computes all outputs together (for efficiency) and caches each
    /// output with a composite key (`base_name` + "_" + `output_name`).
    pub fn get_or_compute_multi<T>(
        &mut self,
        base_spec: &IndicatorSpec,
        candles: &[Candle],
        indicator: &T,
    ) -> MultiOutputCacheResult
    where
        T: MultiOutputIndicator,
    {
        let output_names = indicator.output_names();
        let first_key = base_spec.with_output_suffix(output_names[0]);

        // Check if already computed
        if !self.cache.contains_key(&first_key) {
            let result = indicator.compute_all(candles);
            let vecs = result.into_vecs();

            // Cache each output with its composite key
            for (name, vec) in output_names.iter().zip(vecs) {
                let key = base_spec.with_output_suffix(name);
                self.cache.insert(key, vec);
            }
        }

        // Build result with references to cached data
        let mut outputs = HashMap::new();
        for &name in output_names {
            let key = base_spec.with_output_suffix(name);
            if let Some(values) = self.cache.get(&key) {
                outputs.insert(name.to_string(), values.clone());
            }
        }

        MultiOutputCacheResult { outputs }
    }

    /// Clears all cached values.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Returns the number of cached indicators.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Checks if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Returns iterator over cached indicator specs.
    pub fn specs(&self) -> impl Iterator<Item = &IndicatorSpec> {
        self.cache.keys()
    }
}

/// Result container for multi-output indicator cache access.
#[derive(Debug, Clone)]
pub struct MultiOutputCacheResult {
    /// Map from output name to computed values
    pub outputs: HashMap<String, Vec<f64>>,
}

impl MultiOutputCacheResult {
    /// Gets a specific output by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Vec<f64>> {
        self.outputs.get(name)
    }

    /// Gets a value at a specific index from an output.
    #[must_use]
    pub fn get_at(&self, name: &str, idx: usize) -> Option<f64> {
        self.outputs.get(name).and_then(|v| v.get(idx).copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_::bollinger::BollingerBands;
    use crate::impl_::ema::EMA;
    use crate::traits::IndicatorParams;

    fn make_candle(close: f64) -> Candle {
        Candle {
            timestamp_ns: 0,
            open: close,
            high: close,
            low: close,
            close,
            volume: 0.0,
        }
    }

    #[test]
    fn test_cache_get_or_compute() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let mut cache = IndicatorCache::new();
        let ema = EMA::new(3);
        let spec = IndicatorSpec::new("EMA", IndicatorParams::Period(3));

        // First call computes
        assert!(!cache.contains(&spec));
        let ptr1 = {
            let result = cache.get_or_compute(&spec, &candles, &ema);
            result.as_ptr()
        };
        assert!(cache.contains(&spec));

        // Second call uses cache (same pointer)
        let ptr2 = {
            let result = cache.get_or_compute(&spec, &candles, &ema);
            result.as_ptr()
        };
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_cache_multi_output() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let mut cache = IndicatorCache::new();
        let bb = BollingerBands::new(3, 2.0);
        let spec = IndicatorSpec::new(
            "BOLLINGER",
            IndicatorParams::Bollinger {
                period: 3,
                std_factor_x100: 200,
            },
        );

        let result = cache.get_or_compute_multi(&spec, &candles, &bb);

        // Should have all three outputs
        assert!(result.get("upper").is_some());
        assert!(result.get("middle").is_some());
        assert!(result.get("lower").is_some());

        // Cache should have 3 entries (one per output)
        assert_eq!(cache.len(), 3);

        // Individual keys should work
        let upper_spec = spec.with_output_suffix("upper");
        assert!(cache.contains(&upper_spec));
    }

    #[test]
    fn test_cache_get_at() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let mut cache = IndicatorCache::new();
        let ema = EMA::new(3);
        let spec = IndicatorSpec::new("EMA", IndicatorParams::Period(3));

        cache.get_or_compute(&spec, &candles, &ema);

        // Valid index
        let val = cache.get_at(&spec, 4);
        assert!(val.is_some());
        assert!(val.unwrap().is_finite());

        // Invalid index
        assert!(cache.get_at(&spec, 100).is_none());

        // Unknown spec
        let unknown = IndicatorSpec::new("UNKNOWN", IndicatorParams::Period(3));
        assert!(cache.get_at(&unknown, 0).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let candles: Vec<Candle> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(make_candle)
            .collect();

        let mut cache = IndicatorCache::new();
        let ema = EMA::new(3);
        let spec = IndicatorSpec::new("EMA", IndicatorParams::Period(3));

        cache.get_or_compute(&spec, &candles, &ema);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert!(!cache.contains(&spec));
    }
}
