//! Helpers for equity-curve derived metrics.

use omega_types::EquityPoint;

/// Computes maximum drawdown (relative, absolute) and duration in bars.
#[must_use]
pub fn compute_drawdown(equity: &[EquityPoint]) -> (f64, f64, u64) {
    if equity.is_empty() {
        return (0.0, 0.0, 0);
    }

    let mut high_water: f64 = equity[0].equity;
    let mut max_dd_rel: f64 = 0.0;
    let mut max_dd_abs: f64 = 0.0;
    let mut current_dd_start = 0usize;
    let mut max_dd_duration = 0u64;
    let mut in_drawdown = false;

    for (idx, point) in equity.iter().enumerate() {
        if point.equity > high_water {
            if in_drawdown {
                let duration = (idx - current_dd_start) as u64;
                max_dd_duration = max_dd_duration.max(duration);
                in_drawdown = false;
            }
            high_water = point.equity;
        } else if high_water > 0.0 {
            if !in_drawdown {
                current_dd_start = idx;
                in_drawdown = true;
            }

            let dd_abs: f64 = high_water - point.equity;
            let dd_rel: f64 = dd_abs / high_water;

            max_dd_abs = max_dd_abs.max(dd_abs);
            max_dd_rel = max_dd_rel.max(dd_rel);
        }
    }

    if in_drawdown {
        let duration = (equity.len() - current_dd_start) as u64;
        max_dd_duration = max_dd_duration.max(duration);
    }

    max_dd_rel = max_dd_rel.clamp(0.0, 1.0);

    (max_dd_rel, max_dd_abs, max_dd_duration)
}
