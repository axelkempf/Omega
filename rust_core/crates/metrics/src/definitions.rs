//! Metric definition catalog.

use std::collections::HashMap;

/// Re-exported metric definition type shared with output contract.
pub use omega_types::MetricDefinition;

/// Provides the default metric definitions for the output contract.
#[derive(Debug, Default)]
pub struct MetricDefinitions;

impl MetricDefinitions {
    /// Returns the default metric definitions keyed by metric name.
    #[must_use]
    #[allow(clippy::too_many_lines)] // Declarative list of metric definitions
    pub fn definitions() -> HashMap<String, MetricDefinition> {
        let mut defs = HashMap::new();

        defs.insert(
            "total_trades".to_string(),
            MetricDefinition {
                unit: "count".to_string(),
                description: "Anzahl abgeschlossener Trades".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "win_rate".to_string(),
            MetricDefinition {
                unit: "ratio".to_string(),
                description: "Anteil der Gewinntrades an allen Trades".to_string(),
                domain: "0..1".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "wins".to_string(),
            MetricDefinition {
                unit: "count".to_string(),
                description: "Anzahl Gewinntrades (trade.result > 0)".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "losses".to_string(),
            MetricDefinition {
                unit: "count".to_string(),
                description: "Anzahl Verlusttrades (trade.result < 0)".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "profit_gross".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Summe aller Trade-Ergebnisse vor Fees".to_string(),
                domain: "any".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "fees_total".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Summe expliziter Fees/Commission (ohne Spread/Slippage)"
                    .to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "profit_net".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "profit_gross - fees_total".to_string(),
                domain: "any".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "max_drawdown".to_string(),
            MetricDefinition {
                unit: "ratio".to_string(),
                description: "Maximaler relativer Drawdown (Peak-to-Trough / Peak)"
                    .to_string(),
                domain: "0..1".to_string(),
                source: "equity".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "max_drawdown_abs".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Maximaler absoluter Drawdown in Waehrung".to_string(),
                domain: ">=0".to_string(),
                source: "equity".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "max_drawdown_duration_bars".to_string(),
            MetricDefinition {
                unit: "bars".to_string(),
                description: "Laengste Drawdown-Periode in Bars".to_string(),
                domain: ">=0".to_string(),
                source: "equity".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "avg_r_multiple".to_string(),
            MetricDefinition {
                unit: "r_multiple".to_string(),
                description: "Durchschnittliches R-Multiple aller Trades".to_string(),
                domain: "any".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "profit_factor".to_string(),
            MetricDefinition {
                unit: "ratio".to_string(),
                description: "sum(positive_pnl) / abs(sum(negative_pnl))".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "avg_trade_pnl".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Durchschnittlicher PnL pro Trade (profit_net / total_trades)"
                    .to_string(),
                domain: "any".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "expectancy".to_string(),
            MetricDefinition {
                unit: "r_multiple".to_string(),
                description: "Erwartungswert pro Trade in R (MVP: avg_r_multiple)"
                    .to_string(),
                domain: "any".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "active_days".to_string(),
            MetricDefinition {
                unit: "days".to_string(),
                description: "Anzahl Tage mit mindestens einem Trade".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "trades_per_day".to_string(),
            MetricDefinition {
                unit: "trades".to_string(),
                description: "total_trades / active_days".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "avg_win".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Durchschnittlicher Gewinn pro Trade".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "avg_loss".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Durchschnittlicher Verlust pro Trade".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "largest_win".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Groesster Gewinntrade".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "largest_loss".to_string(),
            MetricDefinition {
                unit: "account_currency".to_string(),
                description: "Groesster Verlusttrade".to_string(),
                domain: ">=0".to_string(),
                source: "trades".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "sharpe_ratio".to_string(),
            MetricDefinition {
                unit: "ratio".to_string(),
                description: "Sharpe Ratio".to_string(),
                domain: "any".to_string(),
                source: "equity".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "sortino_ratio".to_string(),
            MetricDefinition {
                unit: "ratio".to_string(),
                description: "Sortino Ratio".to_string(),
                domain: "any".to_string(),
                source: "equity".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs.insert(
            "calmar_ratio".to_string(),
            MetricDefinition {
                unit: "ratio".to_string(),
                description: "Calmar Ratio".to_string(),
                domain: "any".to_string(),
                source: "equity".to_string(),
                value_type: "number".to_string(),
            },
        );

        defs
    }
}
