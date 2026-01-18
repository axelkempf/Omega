//! Trade management actions
//!
//! Defines the actions that trade management rules can produce.

use omega_types::ExitReason;
use serde::{Deserialize, Serialize};

/// Action produced by a trade management rule.
///
/// Actions are evaluated at the end of each bar and applied
/// by the execution engine. Stop modifications use `ApplyNextBar` policy
/// (effective from idx + 1).
///
/// MVP: only `ClosePosition` is guaranteed to be applied by the orchestrator.
/// Post-MVP actions may be emitted but can be ignored unless explicitly enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Close a position immediately
    ClosePosition {
        /// Position ID to close
        position_id: u64,
        /// Reason for closing
        reason: ExitReason,
        /// Suggested exit price (for logging/reporting, actual execution may differ)
        #[serde(skip_serializing_if = "Option::is_none")]
        exit_price_hint: Option<f64>,
        /// Additional metadata for the close action
        #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
        meta: serde_json::Value,
    },

    /// Modify stop loss (applies from next bar)
    ModifyStopLoss {
        /// Position ID to modify
        position_id: u64,
        /// New stop loss price
        new_stop_loss: f64,
        /// Reason for modification
        reason: StopModifyReason,
        /// Bar index from which this modification is effective (typically ctx.idx + 1)
        effective_from_idx: usize,
    },

    /// Modify take profit (applies from next bar)
    ModifyTakeProfit {
        /// Position ID to modify
        position_id: u64,
        /// New take profit price
        new_take_profit: f64,
        /// Bar index from which this modification is effective
        effective_from_idx: usize,
    },

    /// No action needed
    None,
}

/// Reason for stop loss modification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopModifyReason {
    /// Break-even stop
    BreakEven,
    /// Trailing stop
    Trailing,
    /// Manual adjustment
    Manual,
}

impl Action {
    /// Creates a close position action.
    #[must_use]
    pub fn close(position_id: u64, reason: ExitReason) -> Self {
        Action::ClosePosition {
            position_id,
            reason,
            exit_price_hint: None,
            meta: serde_json::Value::Null,
        }
    }

    /// Creates a close position action with exit price hint.
    #[must_use]
    pub fn close_with_price(position_id: u64, reason: ExitReason, exit_price_hint: f64) -> Self {
        Action::ClosePosition {
            position_id,
            reason,
            exit_price_hint: Some(exit_price_hint),
            meta: serde_json::Value::Null,
        }
    }

    /// Creates a close position action with full details.
    #[must_use]
    pub fn close_full(
        position_id: u64,
        reason: ExitReason,
        exit_price_hint: Option<f64>,
        meta: serde_json::Value,
    ) -> Self {
        Action::ClosePosition {
            position_id,
            reason,
            exit_price_hint,
            meta,
        }
    }

    /// Creates a modify stop loss action with effective index.
    #[must_use]
    pub fn modify_sl(
        position_id: u64,
        new_stop_loss: f64,
        reason: StopModifyReason,
        effective_from_idx: usize,
    ) -> Self {
        Action::ModifyStopLoss {
            position_id,
            new_stop_loss,
            reason,
            effective_from_idx,
        }
    }

    /// Creates a modify take profit action with effective index.
    #[must_use]
    pub fn modify_tp(position_id: u64, new_take_profit: f64, effective_from_idx: usize) -> Self {
        Action::ModifyTakeProfit {
            position_id,
            new_take_profit,
            effective_from_idx,
        }
    }

    /// Checks if this is a close action.
    #[must_use]
    pub fn is_close(&self) -> bool {
        matches!(self, Action::ClosePosition { .. })
    }

    /// Checks if this is a modify action.
    #[must_use]
    pub fn is_modify(&self) -> bool {
        matches!(
            self,
            Action::ModifyStopLoss { .. } | Action::ModifyTakeProfit { .. }
        )
    }

    /// Gets the position ID this action targets.
    #[must_use]
    pub fn position_id(&self) -> Option<u64> {
        match self {
            Action::ClosePosition { position_id, .. }
            | Action::ModifyStopLoss { position_id, .. }
            | Action::ModifyTakeProfit { position_id, .. } => Some(*position_id),
            Action::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_close() {
        let action = Action::close(42, ExitReason::Timeout);

        assert!(action.is_close());
        assert!(!action.is_modify());
        assert_eq!(action.position_id(), Some(42));
    }

    #[test]
    fn test_action_close_with_price() {
        let action = Action::close_with_price(42, ExitReason::Timeout, 1.1000);

        assert!(action.is_close());
        match action {
            Action::ClosePosition {
                exit_price_hint, ..
            } => {
                assert_eq!(exit_price_hint, Some(1.1000));
            }
            _ => panic!("Expected ClosePosition"),
        }
    }

    #[test]
    fn test_action_modify_sl() {
        let action = Action::modify_sl(42, 1.0950, StopModifyReason::Trailing, 100);

        assert!(!action.is_close());
        assert!(action.is_modify());
        assert_eq!(action.position_id(), Some(42));

        match action {
            Action::ModifyStopLoss {
                effective_from_idx, ..
            } => {
                assert_eq!(effective_from_idx, 100);
            }
            _ => panic!("Expected ModifyStopLoss"),
        }
    }

    #[test]
    fn test_action_modify_tp() {
        let action = Action::modify_tp(42, 1.1100, 50);

        assert!(!action.is_close());
        assert!(action.is_modify());
        assert_eq!(action.position_id(), Some(42));

        match action {
            Action::ModifyTakeProfit {
                effective_from_idx, ..
            } => {
                assert_eq!(effective_from_idx, 50);
            }
            _ => panic!("Expected ModifyTakeProfit"),
        }
    }

    #[test]
    fn test_action_none() {
        let action = Action::None;

        assert!(!action.is_close());
        assert!(!action.is_modify());
        assert_eq!(action.position_id(), None);
    }

    #[test]
    fn test_action_serde() {
        let action = Action::close(42, ExitReason::Timeout);
        let json = serde_json::to_string(&action).unwrap();
        let deserialized: Action = serde_json::from_str(&json).unwrap();

        assert_eq!(action, deserialized);
    }

    #[test]
    fn test_stop_modify_reason_serde() {
        let reason = StopModifyReason::BreakEven;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"break_even\"");
    }
}
