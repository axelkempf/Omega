//! Timeframe alignment helpers.

use omega_types::Timeframe;

use crate::traits::TimeframeMapping;

/// Build a primary -> target mapping using timestamps.
///
/// For each primary timestamp, the mapped index is the latest target
/// timestamp that is <= primary timestamp (no lookahead).
#[must_use]
pub fn build_mapping(
    primary_timestamps: &[i64],
    target_timestamps: &[i64],
    tf: Timeframe,
) -> TimeframeMapping {
    let mut mapping = Vec::with_capacity(primary_timestamps.len());
    if target_timestamps.is_empty() {
        mapping.resize(primary_timestamps.len(), None);
        return TimeframeMapping::new(tf, mapping);
    }

    let mut j = 0usize;
    let mut last_valid: Option<usize> = None;

    for &ts in primary_timestamps {
        while j < target_timestamps.len() && target_timestamps[j] <= ts {
            last_valid = Some(j);
            j += 1;
        }
        mapping.push(last_valid);
    }

    TimeframeMapping::new(tf, mapping)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_mapping_basic() {
        let primary = vec![10, 20, 30, 40, 50];
        let target = vec![15, 35, 55];
        let map = build_mapping(&primary, &target, Timeframe::H1);

        let expected = vec![None, Some(0), Some(0), Some(1), Some(1)];
        assert_eq!(map.primary_to_target, expected);
    }
}
