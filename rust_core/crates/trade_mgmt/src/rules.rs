//! Trade management rules
//!
//! Defines the Rule trait and rule implementations for trade management.

use crate::actions::Action;
use crate::context::{PositionView, TradeContext};
use omega_types::ExitReason;
use serde::{Deserialize, Serialize};

/// Stable identifier for a trade-management rule.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuleId(String);

impl RuleId {
    /// Creates a new rule identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for RuleId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Rule priority (lower value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RulePriority(u16);

impl RulePriority {
    /// Hard close priority (e.g. `MaxHoldingTime`).
    pub const HARD_CLOSE: Self = Self(10);
    /// Protective stop priority (e.g. Break-even).
    pub const PROTECTIVE_STOP: Self = Self(20);
    /// Trailing stop priority.
    pub const TRAILING: Self = Self(30);

    /// Creates a new priority value.
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    /// Returns the numeric priority value.
    #[must_use]
    pub const fn value(self) -> u16 {
        self.0
    }
}

/// Trait for trade management rules.
///
/// Rules evaluate positions and produce actions like closing
/// or modifying stop losses.
///
/// # Thread Safety
/// Rules must be `Send + Sync` for parallel evaluation.
pub trait Rule: Send + Sync {
    /// Stable rule identifier used for deterministic conflict resolution.
    fn id(&self) -> RuleId;

    /// Rule priority (lower wins) used for conflict resolution.
    fn priority(&self) -> RulePriority;

    /// Evaluates the rule for a position.
    ///
    /// # Arguments
    /// * `ctx` - Trade context with market data and session state
    /// * `position` - Read-only view of the position to evaluate
    ///
    /// # Returns
    /// An action to take, or `None` if no action is needed.
    fn evaluate(&self, ctx: &TradeContext, position: &PositionView) -> Option<Action>;

    /// Name of the rule for logging/debugging.
    fn name(&self) -> &'static str;

    /// Checks if this rule applies to the given scenario.
    fn applies_to_scenario(&self, scenario_id: u8) -> bool {
        // Default: applies to all scenarios
        let _ = scenario_id;
        true
    }
}

/// Collection of rules to apply.
#[derive(Default)]
pub struct RuleSet {
    rules: Vec<Box<dyn Rule>>,
}

impl RuleSet {
    /// Creates a new empty rule set.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Adds a rule to the set.
    pub fn add<R: Rule + 'static>(&mut self, rule: R) {
        self.rules.push(Box::new(rule));
    }

    /// Returns an iterator over the rules.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Rule> {
        self.rules.iter().map(AsRef::as_ref)
    }

    /// Returns the number of rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Checks if the rule set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

// ==================== Rule Implementations ====================

/// Maximum holding time rule.
///
/// Closes positions that have been open for more than the specified
/// number of bars or time duration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxHoldingTimeRule {
    /// Maximum holding time in bars
    pub max_bars: usize,
    /// Bar duration in nanoseconds (for time-based calculation)
    pub bar_duration_ns: i64,
    /// Only apply to specific scenarios (empty = all)
    #[serde(default)]
    pub only_scenarios: Vec<u8>,
}

impl MaxHoldingTimeRule {
    /// Creates a new max holding time rule.
    #[must_use]
    pub fn new(max_bars: usize, bar_duration_ns: i64) -> Self {
        Self {
            max_bars,
            bar_duration_ns,
            only_scenarios: Vec::new(),
        }
    }

    /// Creates a rule that only applies to specific scenarios.
    #[must_use]
    pub fn with_scenarios(mut self, scenarios: Vec<u8>) -> Self {
        self.only_scenarios = scenarios;
        self
    }

    /// Creates a rule from minutes using the bar duration.
    #[must_use]
    pub fn from_minutes(max_minutes: u64, bar_duration_ns: i64) -> Self {
        let safe_bar_ns = bar_duration_ns.max(1);
        let bar_duration_ns_u64 = u64::try_from(safe_bar_ns).unwrap_or(1);
        let max_time_ns = max_minutes.saturating_mul(60).saturating_mul(1_000_000_000);
        let max_bars_u64 = max_time_ns / bar_duration_ns_u64;
        let max_bars = usize::try_from(max_bars_u64).unwrap_or(usize::MAX);
        Self::new(max_bars.max(1), safe_bar_ns)
    }
}

impl Rule for MaxHoldingTimeRule {
    fn id(&self) -> RuleId {
        RuleId::from("max_holding_time")
    }

    fn priority(&self) -> RulePriority {
        RulePriority::HARD_CLOSE
    }

    fn evaluate(&self, ctx: &TradeContext, position: &PositionView) -> Option<Action> {
        if self.bar_duration_ns <= 0 {
            return None;
        }

        let holding_ns = (ctx.market.timestamp_ns - position.entry_time_ns).max(0);
        let holding_bars = holding_ns / self.bar_duration_ns;
        let holding_bars_u64 = u64::try_from(holding_bars).unwrap_or(0);
        let max_bars_u64 = u64::try_from(self.max_bars).unwrap_or(u64::MAX);

        if holding_bars_u64 >= max_bars_u64 {
            // Exit price: bid_close for long, ask_close for short
            let exit_price = ctx.market.exit_price(position.direction);

            return Some(Action::close_with_price(
                position.position_id,
                ExitReason::Timeout,
                exit_price,
            ));
        }

        None
    }

    fn name(&self) -> &'static str {
        "max_holding_time"
    }

    fn applies_to_scenario(&self, scenario_id: u8) -> bool {
        self.only_scenarios.is_empty() || self.only_scenarios.contains(&scenario_id)
    }
}

/// Break-even stop rule (Post-MVP).
///
/// Moves stop loss to entry price (+ buffer) when price moves
/// favorably by a certain amount.
///
/// **Note:** This rule is Post-MVP. Enable via config only when needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakEvenRule {
    /// Trigger distance in price units (move BE when price moves this much)
    pub trigger_distance: f64,
    /// Buffer to add to entry price for the new stop
    pub buffer: f64,
    /// Only apply to specific scenarios (empty = all)
    #[serde(default)]
    pub only_scenarios: Vec<u8>,
    /// Whether this rule is enabled (default: false for Post-MVP)
    #[serde(default)]
    pub enabled: bool,
}

impl BreakEvenRule {
    /// Creates a new break-even rule (disabled by default).
    #[must_use]
    pub fn new(trigger_distance: f64, buffer: f64) -> Self {
        Self {
            trigger_distance,
            buffer,
            only_scenarios: Vec::new(),
            enabled: false,
        }
    }

    /// Creates an enabled break-even rule.
    #[must_use]
    pub fn enabled(trigger_distance: f64, buffer: f64) -> Self {
        Self {
            trigger_distance,
            buffer,
            only_scenarios: Vec::new(),
            enabled: true,
        }
    }

    /// Creates a rule that only applies to specific scenarios.
    #[must_use]
    pub fn with_scenarios(mut self, scenarios: Vec<u8>) -> Self {
        self.only_scenarios = scenarios;
        self
    }
}

impl Rule for BreakEvenRule {
    fn id(&self) -> RuleId {
        RuleId::from("break_even")
    }

    fn priority(&self) -> RulePriority {
        RulePriority::PROTECTIVE_STOP
    }

    fn evaluate(&self, ctx: &TradeContext, position: &PositionView) -> Option<Action> {
        use omega_types::Direction;

        // Post-MVP: only evaluate if explicitly enabled
        if !self.enabled {
            return None;
        }

        let entry = position.entry_price;
        let current_sl = position.stop_loss?;

        // Get current price from MarketView (not position.meta)
        let current_price = ctx.market.current_price(position.direction);

        match position.direction {
            Direction::Long => {
                // Check if price has moved up enough
                if current_price >= entry + self.trigger_distance {
                    // Only modify if new SL is better than current
                    let new_sl = entry + self.buffer;
                    if new_sl > current_sl {
                        return Some(Action::modify_sl(
                            position.position_id,
                            new_sl,
                            crate::actions::StopModifyReason::BreakEven,
                            ctx.idx + 1, // ApplyNextBar
                        ));
                    }
                }
            }
            Direction::Short => {
                // Check if price has moved down enough
                if current_price <= entry - self.trigger_distance {
                    // Only modify if new SL is better than current
                    let new_sl = entry - self.buffer;
                    if new_sl < current_sl {
                        return Some(Action::modify_sl(
                            position.position_id,
                            new_sl,
                            crate::actions::StopModifyReason::BreakEven,
                            ctx.idx + 1, // ApplyNextBar
                        ));
                    }
                }
            }
        }

        None
    }

    fn name(&self) -> &'static str {
        "break_even"
    }

    fn applies_to_scenario(&self, scenario_id: u8) -> bool {
        self.only_scenarios.is_empty() || self.only_scenarios.contains(&scenario_id)
    }
}

/// Trailing stop rule (Post-MVP).
///
/// Moves stop loss to follow price by a fixed distance.
///
/// **Note:** This rule is Post-MVP. Enable via config only when needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopRule {
    /// Trail distance in price units
    pub trail_distance: f64,
    /// Activation distance (start trailing after price moves this much)
    pub activation_distance: f64,
    /// Only apply to specific scenarios (empty = all)
    #[serde(default)]
    pub only_scenarios: Vec<u8>,
    /// Whether this rule is enabled (default: false for Post-MVP)
    #[serde(default)]
    pub enabled: bool,
}

impl TrailingStopRule {
    /// Creates a new trailing stop rule (disabled by default).
    #[must_use]
    pub fn new(trail_distance: f64, activation_distance: f64) -> Self {
        Self {
            trail_distance,
            activation_distance,
            only_scenarios: Vec::new(),
            enabled: false,
        }
    }

    /// Creates an enabled trailing stop rule.
    #[must_use]
    pub fn enabled(trail_distance: f64, activation_distance: f64) -> Self {
        Self {
            trail_distance,
            activation_distance,
            only_scenarios: Vec::new(),
            enabled: true,
        }
    }

    /// Creates a rule that only applies to specific scenarios.
    #[must_use]
    pub fn with_scenarios(mut self, scenarios: Vec<u8>) -> Self {
        self.only_scenarios = scenarios;
        self
    }
}

impl Rule for TrailingStopRule {
    fn id(&self) -> RuleId {
        RuleId::from("trailing_stop")
    }

    fn priority(&self) -> RulePriority {
        RulePriority::TRAILING
    }

    fn evaluate(&self, ctx: &TradeContext, position: &PositionView) -> Option<Action> {
        use omega_types::Direction;

        // Post-MVP: only evaluate if explicitly enabled
        if !self.enabled {
            return None;
        }

        let entry = position.entry_price;
        let current_sl = position.stop_loss?;

        // Get current price from MarketView (not position.meta)
        let current_price = ctx.market.current_price(position.direction);

        match position.direction {
            Direction::Long => {
                // Check if trailing is activated
                if current_price >= entry + self.activation_distance {
                    let new_sl = current_price - self.trail_distance;
                    // Only modify if new SL is better than current
                    if new_sl > current_sl {
                        return Some(Action::modify_sl(
                            position.position_id,
                            new_sl,
                            crate::actions::StopModifyReason::Trailing,
                            ctx.idx + 1, // ApplyNextBar
                        ));
                    }
                }
            }
            Direction::Short => {
                // Check if trailing is activated
                if current_price <= entry - self.activation_distance {
                    let new_sl = current_price + self.trail_distance;
                    // Only modify if new SL is better than current
                    if new_sl < current_sl {
                        return Some(Action::modify_sl(
                            position.position_id,
                            new_sl,
                            crate::actions::StopModifyReason::Trailing,
                            ctx.idx + 1, // ApplyNextBar
                        ));
                    }
                }
            }
        }

        None
    }

    fn name(&self) -> &'static str {
        "trailing_stop"
    }

    fn applies_to_scenario(&self, scenario_id: u8) -> bool {
        self.only_scenarios.is_empty() || self.only_scenarios.contains(&scenario_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MarketView;
    use omega_types::Direction;

    fn make_position_view(
        id: u64,
        direction: Direction,
        entry_price: f64,
        stop_loss: f64,
    ) -> PositionView {
        PositionView::new(id, "EURUSD", direction, 1_000_000_000, entry_price, 0.1)
            .with_stop_loss(stop_loss)
            .with_take_profit(entry_price + 0.01)
    }

    fn make_context(timestamp_ns: i64, bid_close: f64, ask_close: f64) -> TradeContext {
        let market = MarketView::from_close(timestamp_ns, bid_close, ask_close);
        TradeContext::new(0, market, 60_000_000_000)
    }

    #[test]
    fn test_max_holding_time_triggers() {
        let rule = MaxHoldingTimeRule::new(10, 60_000_000_000); // 10 bars, 1 minute bars

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);

        // 11 minutes later (11 bars)
        let timestamp_ns = 1_000_000_000 + (11 * 60_000_000_000);
        let ctx = make_context(timestamp_ns, 1.1010, 1.1012);
        let action = rule.evaluate(&ctx, &position);

        assert!(matches!(
            action,
            Some(Action::ClosePosition {
                reason: ExitReason::Timeout,
                exit_price_hint: Some(price),
                ..
            }) if (price - 1.1010).abs() < 1e-10 // bid_close for long
        ));
    }

    #[test]
    fn test_max_holding_time_triggers_short_uses_ask_close() {
        let rule = MaxHoldingTimeRule::new(10, 60_000_000_000);

        let position = make_position_view(2, Direction::Short, 1.1000, 1.1050);

        let timestamp_ns = 1_000_000_000 + (11 * 60_000_000_000);
        let ctx = make_context(timestamp_ns, 1.0995, 1.0998);
        let action = rule.evaluate(&ctx, &position);

        assert!(matches!(
            action,
            Some(Action::ClosePosition {
                reason: ExitReason::Timeout,
                exit_price_hint: Some(price),
                ..
            }) if (price - 1.0998).abs() < 1e-10 // ask_close for short
        ));
    }

    #[test]
    fn test_max_holding_time_not_yet() {
        let rule = MaxHoldingTimeRule::new(10, 60_000_000_000);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);

        // 5 minutes later (5 bars)
        let timestamp_ns = 1_000_000_000 + (5 * 60_000_000_000);
        let ctx = make_context(timestamp_ns, 1.1010, 1.1012);
        let action = rule.evaluate(&ctx, &position);

        assert!(action.is_none());
    }

    #[test]
    fn test_max_holding_time_from_minutes() {
        let bar_duration_ns = 60_000_000_000i64; // 1 minute
        let rule = MaxHoldingTimeRule::from_minutes(30, bar_duration_ns);

        assert_eq!(rule.max_bars, 30);
    }

    #[test]
    fn test_max_holding_time_scenario_filter() {
        let rule = MaxHoldingTimeRule::new(10, 60_000_000_000).with_scenarios(vec![1, 2]);

        assert!(rule.applies_to_scenario(1));
        assert!(rule.applies_to_scenario(2));
        assert!(!rule.applies_to_scenario(3));
    }

    #[test]
    fn test_break_even_disabled_by_default() {
        // BreakEven is Post-MVP, disabled by default
        let rule = BreakEvenRule::new(0.0020, 0.0001);
        assert!(!rule.enabled);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);
        let ctx = make_context(0, 1.1025, 1.1027); // +25 pips

        // Should return None because disabled
        let action = rule.evaluate(&ctx, &position);
        assert!(action.is_none());
    }

    #[test]
    fn test_break_even_long_triggers_when_enabled() {
        let rule = BreakEvenRule::enabled(0.0020, 0.0001);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);
        let ctx = make_context(0, 1.1025, 1.1027); // +25 pips (bid_close)

        let action = rule.evaluate(&ctx, &position);

        assert!(matches!(
            action,
            Some(Action::ModifyStopLoss {
                new_stop_loss,
                effective_from_idx: 1, // ApplyNextBar
                ..
            }) if (new_stop_loss - 1.1001).abs() < 1e-10
        ));
    }

    #[test]
    fn test_break_even_long_not_triggered() {
        let rule = BreakEvenRule::enabled(0.0020, 0.0001);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);
        let ctx = make_context(0, 1.1015, 1.1017); // Only +15 pips

        let action = rule.evaluate(&ctx, &position);
        assert!(action.is_none());
    }

    #[test]
    fn test_break_even_short_triggers_when_enabled() {
        let rule = BreakEvenRule::enabled(0.0020, 0.0001);

        let position = make_position_view(1, Direction::Short, 1.1000, 1.1050);
        let ctx = make_context(0, 1.0973, 1.0975); // -25 pips (ask_close for short)

        let action = rule.evaluate(&ctx, &position);

        assert!(matches!(
            action,
            Some(Action::ModifyStopLoss {
                new_stop_loss,
                ..
            }) if (new_stop_loss - 1.0999).abs() < 1e-10
        ));
    }

    #[test]
    fn test_trailing_stop_disabled_by_default() {
        // TrailingStop is Post-MVP, disabled by default
        let rule = TrailingStopRule::new(0.0010, 0.0020);
        assert!(!rule.enabled);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);
        let ctx = make_context(0, 1.1030, 1.1032); // +30 pips

        // Should return None because disabled
        let action = rule.evaluate(&ctx, &position);
        assert!(action.is_none());
    }

    #[test]
    fn test_trailing_stop_long_when_enabled() {
        let rule = TrailingStopRule::enabled(0.0010, 0.0020);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);
        let ctx = make_context(0, 1.1030, 1.1032); // +30 pips (bid_close)

        let action = rule.evaluate(&ctx, &position);

        assert!(matches!(
            action,
            Some(Action::ModifyStopLoss {
                new_stop_loss,
                effective_from_idx: 1, // ApplyNextBar
                ..
            }) if (new_stop_loss - 1.1020).abs() < 1e-10 // 1.1030 - 0.0010
        ));
    }

    #[test]
    fn test_trailing_stop_not_activated() {
        let rule = TrailingStopRule::enabled(0.0010, 0.0020);

        let position = make_position_view(1, Direction::Long, 1.1000, 1.0950);
        let ctx = make_context(0, 1.1015, 1.1017); // Only +15 pips

        let action = rule.evaluate(&ctx, &position);
        assert!(action.is_none());
    }

    #[test]
    fn test_rule_set() {
        let mut rules = RuleSet::new();
        rules.add(MaxHoldingTimeRule::new(10, 60_000_000_000));
        rules.add(BreakEvenRule::enabled(0.0020, 0.0001));

        assert_eq!(rules.len(), 2);
        assert!(!rules.is_empty());

        let names: Vec<_> = rules.iter().map(Rule::name).collect();
        assert!(names.contains(&"max_holding_time"));
        assert!(names.contains(&"break_even"));
    }
}
