//! Trade manager engine
//!
//! Evaluates trade management rules for open positions.

use crate::actions::Action;
use crate::context::{PositionView, TradeContext};
use crate::rules::RuleSet;

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
    pub fn new(rules: RuleSet) -> Self {
        Self { rules }
    }

    /// Creates an empty trade manager (no rules).
    pub fn empty() -> Self {
        Self {
            rules: RuleSet::new(),
        }
    }

    /// Evaluates rules for all positions.
    ///
    /// Returns a list of actions to apply. Only one action per position
    /// is returned (first rule that matches wins).
    ///
    /// # Arguments
    /// * `ctx` - Trade context with market data and session state
    /// * `positions` - Slice of open positions to evaluate
    pub fn evaluate(&self, ctx: &TradeContext, positions: &[PositionView]) -> Vec<Action> {
        let mut actions = Vec::new();

        for position in positions {
            for rule in self.rules.iter() {
                // Check if rule applies to this scenario
                if !rule.applies_to_scenario(position.scenario_id) {
                    continue;
                }

                if let Some(action) = rule.evaluate(ctx, position) {
                    actions.push(action);
                    break; // One action per position per bar
                }
            }
        }

        actions
    }

    /// Returns the number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Checks if the manager has any rules.
    pub fn has_rules(&self) -> bool {
        !self.rules.is_empty()
    }
}

impl Default for TradeManager {
    fn default() -> Self {
        Self::empty()
    }
}

/// Builder for creating a TradeManager with a fluent API.
#[derive(Default)]
pub struct TradeManagerBuilder {
    rules: RuleSet,
}

impl TradeManagerBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            rules: RuleSet::new(),
        }
    }

    /// Adds a rule to the builder.
    pub fn with_rule<R: crate::rules::Rule + 'static>(mut self, rule: R) -> Self {
        self.rules.add(rule);
        self
    }

    /// Builds the trade manager.
    pub fn build(self) -> TradeManager {
        TradeManager::new(self.rules)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MarketView;
    use crate::rules::{BreakEvenRule, MaxHoldingTimeRule};
    use omega_types::{Direction, ExitReason};

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

        let positions = vec![make_position_view(1, Direction::Long, 1000000000, 1)];
        let ctx = make_context(2000000000000);
        let actions = manager.evaluate(&ctx, &positions);

        assert!(actions.is_empty());
    }

    #[test]
    fn test_trade_manager_max_holding_time() {
        let mut rules = RuleSet::new();
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000)); // 10 bars, 1 min bars

        let manager = TradeManager::new(rules);

        let positions = vec![make_position_view(1, Direction::Long, 1000000000, 1)];

        // 11 minutes later
        let timestamp_ns = 1000000000 + (11 * 60_000_000_000);
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
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000));
        rules.add(MaxHoldingTimeRule::new(5, 60_000_000_000)); // Also would trigger

        let manager = TradeManager::new(rules);

        let positions = vec![make_position_view(1, Direction::Long, 1000000000, 1)];

        // 15 minutes later (both rules would trigger)
        let timestamp_ns = 1000000000 + (15 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        // Only first rule's action should be returned
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn test_trade_manager_scenario_filter() {
        let mut rules = RuleSet::new();
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000).with_scenarios(vec![2]));

        let manager = TradeManager::new(rules);

        // Position with scenario 1 (rule doesn't apply)
        let positions = vec![make_position_view(1, Direction::Long, 1000000000, 1)];

        let timestamp_ns = 1000000000 + (15 * 60_000_000_000);
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
            make_position_view(1, Direction::Long, 1000000000, 1),         // Will timeout
            make_position_view(2, Direction::Short, 500_000_000_000, 1),   // Won't timeout
            make_position_view(3, Direction::Long, 100_000_000_000, 1),    // Will timeout
        ];

        let timestamp_ns = 1000000000 + (15 * 60_000_000_000);
        let ctx = make_context(timestamp_ns);
        let actions = manager.evaluate(&ctx, &positions);

        // Position 1 and 3 should timeout
        assert_eq!(actions.len(), 2);

        let position_ids: Vec<u64> = actions.iter().filter_map(|a| a.position_id()).collect();
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
