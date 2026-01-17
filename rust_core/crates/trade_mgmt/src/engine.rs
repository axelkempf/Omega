//! Trade manager engine
//!
//! Evaluates trade management rules for open positions.

use crate::actions::Action;
use crate::context::{PositionView, TradeContext};
use crate::rules::{MaxHoldingTimeRule, RuleId, RulePriority, RuleSet};
use serde::{Deserialize, Serialize};

/// Stop update policy for trade management actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StopUpdatePolicy {
    /// Apply stop/TP changes from the next bar (MVP policy).
    #[default]
    ApplyNextBar,
}

/// Trade manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeManagerConfig {
    /// Whether trade management is enabled.
    #[serde(default = "default_trade_manager_enabled")]
    pub enabled: bool,
    /// Policy for stop updates (MVP: `apply_next_bar`).
    #[serde(default)]
    pub stop_update_policy: StopUpdatePolicy,
    /// Rule-specific configuration.
    #[serde(default)]
    pub rules: TradeManagerRulesConfig,
}

impl Default for TradeManagerConfig {
    fn default() -> Self {
        Self {
            enabled: default_trade_manager_enabled(),
            stop_update_policy: StopUpdatePolicy::default(),
            rules: TradeManagerRulesConfig::default(),
        }
    }
}

/// Rule configuration block for trade management.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradeManagerRulesConfig {
    /// Max holding time rule configuration.
    #[serde(default)]
    pub max_holding_time: MaxHoldingTimeConfig,
}

/// Configuration for the max holding time rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxHoldingTimeConfig {
    /// Whether the rule is enabled.
    #[serde(default = "default_max_holding_enabled")]
    pub enabled: bool,
    /// Max holding minutes (0 uses fallback from strategy params).
    #[serde(default = "default_max_holding_minutes")]
    pub max_holding_minutes: u64,
    /// Only apply to these scenarios (empty = all).
    #[serde(default)]
    pub only_scenarios: Vec<u8>,
}

impl Default for MaxHoldingTimeConfig {
    fn default() -> Self {
        Self {
            enabled: default_max_holding_enabled(),
            max_holding_minutes: default_max_holding_minutes(),
            only_scenarios: Vec::new(),
        }
    }
}

fn default_trade_manager_enabled() -> bool {
    false
}

fn default_max_holding_enabled() -> bool {
    true
}

fn default_max_holding_minutes() -> u64 {
    0
}

#[derive(Debug, Clone)]
struct ActionCandidate {
    action: Action,
    rule_id: RuleId,
    priority: RulePriority,
}

/// Trade manager for evaluating rules against positions.
///
/// The trade manager takes a set of rules and evaluates them
/// against open positions to produce actions.
///
/// # Example
/// ```ignore
/// let mut rules = RuleSet::new();
/// rules.add(MaxHoldingTimeRule::new(10, bar_duration_ns));
///
/// let manager = TradeManager::new(rules);
///
/// // In backtest loop:
/// let market = MarketView::from_close(timestamp_ns, bid_close, ask_close);
/// let ctx = TradeContext::new(bar_idx, market, bar_duration_ns);
/// let actions = manager.evaluate(&ctx, &positions);
///
/// for action in actions {
///     // Apply action...
/// }
/// ```
pub struct TradeManager {
    rules: RuleSet,
}

impl TradeManager {
    /// Creates a new trade manager with the given rules.
    #[must_use]
    pub fn new(rules: RuleSet) -> Self {
        Self { rules }
    }

    /// Creates a trade manager from config (optionally using strategy defaults).
    ///
    /// When `max_holding_minutes` is 0 in the config, `fallback_max_holding_minutes`
    /// is used (e.g., mapped from strategy parameters).
    #[must_use]
    pub fn from_config(
        config: &TradeManagerConfig,
        bar_duration_ns: i64,
        fallback_max_holding_minutes: Option<u64>,
    ) -> Self {
        if !config.enabled {
            return TradeManager::empty();
        }

        let mut rules = RuleSet::new();
        let max_cfg = &config.rules.max_holding_time;
        if max_cfg.enabled {
            let max_minutes = if max_cfg.max_holding_minutes > 0 {
                max_cfg.max_holding_minutes
            } else {
                fallback_max_holding_minutes.unwrap_or(0)
            };

            if max_minutes > 0 {
                let mut rule = MaxHoldingTimeRule::from_minutes(max_minutes, bar_duration_ns);
                if !max_cfg.only_scenarios.is_empty() {
                    rule = rule.with_scenarios(max_cfg.only_scenarios.clone());
                }
                rules.add(rule);
            }
        }

        TradeManager::new(rules)
    }

    /// Creates an empty trade manager (no rules).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            rules: RuleSet::new(),
        }
    }

    /// Evaluates rules for all positions.
    ///
    /// Returns a list of actions to apply. Only one action per position
    /// is returned after deterministic conflict resolution.
    ///
    /// # Arguments
    /// * `ctx` - Trade context with market data and session state
    /// * `positions` - Slice of open positions to evaluate
    #[must_use]
    pub fn evaluate(&self, ctx: &TradeContext, positions: &[PositionView]) -> Vec<Action> {
        let mut actions = Vec::new();

        for position in positions {
            let mut candidates = Vec::new();
            for rule in self.rules.iter() {
                // Check if rule applies to this scenario
                if !rule.applies_to_scenario(position.scenario_id) {
                    continue;
                }

                if let Some(action) = rule.evaluate(ctx, position) {
                    if matches!(action, Action::None) {
                        continue;
                    }

                    candidates.push(ActionCandidate {
                        action,
                        rule_id: rule.id(),
                        priority: rule.priority(),
                    });
                }
            }

            if let Some(action) = Self::resolve_candidates(candidates) {
                actions.push(action);
            }
        }

        actions
    }

    fn resolve_candidates(candidates: Vec<ActionCandidate>) -> Option<Action> {
        if candidates.is_empty() {
            return None;
        }

        let mut close_candidates: Vec<_> =
            candidates.iter().filter(|c| c.action.is_close()).collect();
        if !close_candidates.is_empty() {
            close_candidates
                .sort_by(|a, b| (a.priority, &a.rule_id).cmp(&(b.priority, &b.rule_id)));
            return close_candidates.first().map(|c| c.action.clone());
        }

        let mut remaining = candidates;
        remaining.sort_by(|a, b| (a.priority, &a.rule_id).cmp(&(b.priority, &b.rule_id)));
        remaining.first().map(|c| c.action.clone())
    }

    /// Returns the number of rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Checks if the manager has any rules.
    #[must_use]
    pub fn has_rules(&self) -> bool {
        !self.rules.is_empty()
    }
}

impl Default for TradeManager {
    fn default() -> Self {
        Self::empty()
    }
}

/// Builder for creating a `TradeManager` with a fluent API.
#[derive(Default)]
pub struct TradeManagerBuilder {
    rules: RuleSet,
}

impl TradeManagerBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: RuleSet::new(),
        }
    }

    /// Adds a rule to the builder.
    #[must_use]
    pub fn with_rule<R: crate::rules::Rule + 'static>(mut self, rule: R) -> Self {
        self.rules.add(rule);
        self
    }

    /// Builds the trade manager.
    #[must_use]
    pub fn build(self) -> TradeManager {
        TradeManager::new(self.rules)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MarketView;
    use crate::rules::{BreakEvenRule, MaxHoldingTimeRule, Rule};
    use omega_types::{Direction, ExitReason};

    struct DummyRule {
        id: RuleId,
        priority: RulePriority,
        name: &'static str,
        action: Action,
        scenarios: Vec<u8>,
    }

    impl Rule for DummyRule {
        fn id(&self) -> RuleId {
            self.id.clone()
        }

        fn priority(&self) -> RulePriority {
            self.priority
        }

        fn evaluate(&self, _ctx: &TradeContext, _position: &PositionView) -> Option<Action> {
            Some(self.action.clone())
        }

        fn name(&self) -> &'static str {
            self.name
        }

        fn applies_to_scenario(&self, scenario_id: u8) -> bool {
            self.scenarios.is_empty() || self.scenarios.contains(&scenario_id)
        }
    }

    fn make_position_view(
        id: u64,
        direction: Direction,
        entry_time_ns: i64,
        scenario_id: u8,
    ) -> PositionView {
        PositionView::new(id, "EURUSD", direction, entry_time_ns, 1.1000, 0.1)
            .with_stop_loss(1.0950)
            .with_take_profit(1.1100)
            .with_scenario(scenario_id)
    }

    fn make_context(timestamp_ns: i64) -> TradeContext {
        let market = MarketView::from_close(timestamp_ns, 1.1010, 1.1012);
        TradeContext::new(0, market, 60_000_000_000)
    }

    #[test]
    fn test_trade_manager_empty() {
        let manager = TradeManager::empty();

        assert_eq!(manager.rule_count(), 0);
        assert!(!manager.has_rules());

        let positions = vec![make_position_view(1, Direction::Long, 1_000_000_000, 1)];
        let ctx = make_context(2_000_000_000_000);
        let actions = manager.evaluate(&ctx, &positions);

        assert!(actions.is_empty());
    }

    #[test]
    fn test_trade_manager_max_holding_time() {
        let mut rules = RuleSet::new();
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000)); // 10 bars, 1 min bars

        let manager = TradeManager::new(rules);

        let positions = vec![make_position_view(1, Direction::Long, 1_000_000_000, 1)];

        // 11 minutes later
        let timestamp_ns = 1_000_000_000 + (11 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            Action::ClosePosition {
                position_id: 1,
                reason: ExitReason::Timeout,
                ..
            }
        ));
    }

    #[test]
    fn test_trade_manager_one_action_per_position() {
        let mut rules = RuleSet::new();
        rules.add(DummyRule {
            id: RuleId::from("rule_b"),
            priority: RulePriority::new(20),
            name: "rule_b",
            action: Action::close_full(1, ExitReason::Timeout, Some(1.1000), serde_json::json!({})),
            scenarios: Vec::new(),
        });
        rules.add(DummyRule {
            id: RuleId::from("rule_a"),
            priority: RulePriority::new(10),
            name: "rule_a",
            action: Action::close_full(1, ExitReason::Timeout, Some(1.2000), serde_json::json!({})),
            scenarios: Vec::new(),
        });

        let manager = TradeManager::new(rules);

        let positions = vec![make_position_view(1, Direction::Long, 1_000_000_000, 1)];

        // 15 minutes later (both rules would trigger)
        let timestamp_ns = 1_000_000_000 + (15 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        // Highest priority (lowest value) wins, not insertion order
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            Action::ClosePosition {
                exit_price_hint: Some(price),
                ..
            } if (*price - 1.2000).abs() < 1e-10
        ));
    }

    #[test]
    fn test_trade_manager_close_wins_over_modify() {
        let mut rules = RuleSet::new();
        rules.add(DummyRule {
            id: RuleId::from("modify_first"),
            priority: RulePriority::new(0),
            name: "modify_first",
            action: Action::modify_sl(1, 1.0900, crate::actions::StopModifyReason::Manual, 1),
            scenarios: Vec::new(),
        });
        rules.add(DummyRule {
            id: RuleId::from("close_later"),
            priority: RulePriority::new(50),
            name: "close_later",
            action: Action::close_full(1, ExitReason::Timeout, Some(1.1005), serde_json::json!({})),
            scenarios: Vec::new(),
        });

        let manager = TradeManager::new(rules);
        let positions = vec![make_position_view(1, Direction::Long, 1_000_000_000, 1)];

        let timestamp_ns = 1_000_000_000 + (15 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], Action::ClosePosition { .. }));
    }

    #[test]
    fn test_trade_manager_from_config_uses_fallback() {
        let config = TradeManagerConfig {
            enabled: true,
            rules: TradeManagerRulesConfig {
                max_holding_time: MaxHoldingTimeConfig {
                    enabled: true,
                    max_holding_minutes: 0,
                    only_scenarios: vec![1],
                },
            },
            ..Default::default()
        };

        let manager = TradeManager::from_config(&config, 60_000_000_000, Some(30));

        assert!(manager.has_rules());
    }

    #[test]
    fn test_trade_manager_scenario_filter() {
        let mut rules = RuleSet::new();
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000).with_scenarios(vec![2]));

        let manager = TradeManager::new(rules);

        // Position with scenario 1 (rule doesn't apply)
        let positions = vec![make_position_view(1, Direction::Long, 1_000_000_000, 1)];

        let timestamp_ns = 1_000_000_000 + (15 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        // Rule shouldn't trigger because scenario doesn't match
        assert!(actions.is_empty());
    }

    #[test]
    fn test_trade_manager_multiple_positions() {
        let mut rules = RuleSet::new();
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000));

        let manager = TradeManager::new(rules);

        let positions = vec![
            make_position_view(1, Direction::Long, 1_000_000_000, 1), // Will timeout
            make_position_view(2, Direction::Short, 500_000_000_000, 1), // Won't timeout
            make_position_view(3, Direction::Long, 100_000_000_000, 1), // Will timeout
        ];

        let timestamp_ns = 1_000_000_000 + (15 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        // Position 1 and 3 should timeout
        assert_eq!(actions.len(), 2);

        let position_ids: Vec<u64> = actions.iter().filter_map(Action::position_id).collect();
        assert!(position_ids.contains(&1));
        assert!(position_ids.contains(&3));
        assert!(!position_ids.contains(&2));
    }

    #[test]
    fn test_trade_manager_builder() {
        let manager = TradeManagerBuilder::new()
            .with_rule(MaxHoldingTimeRule::new(10, 60_000_000_000))
            .with_rule(BreakEvenRule::enabled(0.0020, 0.0001))
            .build();

        assert_eq!(manager.rule_count(), 2);
        assert!(manager.has_rules());
    }

    #[test]
    fn test_trade_manager_default() {
        let manager = TradeManager::default();
        assert!(!manager.has_rules());
    }
}
