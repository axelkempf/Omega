//! Volatility cluster feature series utilities.

/// Volatility feature series used for clustering.
#[derive(Debug, Clone)]
pub enum VolFeatureSeries {
    /// Full-length series aligned to primary timeframe.
    Full(Vec<f64>),
    /// Local window series with a start index into the full series.
    Local {
        /// Start index of the local window in the primary series.
        start_idx: usize,
        /// Local window values.
        values: Vec<f64>,
    },
}

impl VolFeatureSeries {
    /// Returns the feature value at a primary index if available.
    #[must_use]
    pub fn value_at(&self, idx: usize) -> Option<f64> {
        match self {
            VolFeatureSeries::Full(values) => values.get(idx).copied(),
            VolFeatureSeries::Local { start_idx, values } => {
                let offset = idx.checked_sub(*start_idx)?;
                values.get(offset).copied()
            }
        }
    }

    /// Returns the length of the underlying series.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            VolFeatureSeries::Full(values) | VolFeatureSeries::Local { values, .. } => values.len(),
        }
    }

    /// Returns true if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Builds a volatility feature series based on feature selection.
#[must_use]
pub fn vol_cluster_series(
    feature: &str,
    atr_series: Vec<f64>,
    local_sigma: Vec<f64>,
    local_start_idx: usize,
) -> Option<VolFeatureSeries> {
    let feat = feature.trim().to_lowercase();
    if feat == "atr_points" {
        return Some(VolFeatureSeries::Full(atr_series));
    }
    if local_sigma.is_empty() {
        return None;
    }
    Some(VolFeatureSeries::Local {
        start_idx: local_start_idx,
        values: local_sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vol_feature_series_value_at() {
        let series = VolFeatureSeries::Full(vec![1.0, 2.0, 3.0]);
        assert_eq!(series.value_at(1), Some(2.0));

        let local = VolFeatureSeries::Local {
            start_idx: 2,
            values: vec![10.0, 11.0],
        };
        assert_eq!(local.value_at(2), Some(10.0));
        assert_eq!(local.value_at(3), Some(11.0));
        assert_eq!(local.value_at(1), None);
    }
}
