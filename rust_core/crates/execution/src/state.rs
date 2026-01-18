//! Order and Position state machines.
//!
//! Defines the valid states and transitions for orders and positions.
//! State transitions are deterministic and irreversible (no rollback).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Order state for pending orders (Limit/Stop orders awaiting trigger).
///
/// # State Diagram
/// ```text
/// ┌─────────┐
/// │ Pending │──────┬──────────┬────────────┐
/// └────┬────┘      │          │            │
///      │           ▼          ▼            ▼
///      │      ┌─────────┐ ┌──────────┐ ┌─────────┐
///      │      │Cancelled│ │ Expired  │ │         │
///      │      └─────────┘ └──────────┘ │         │
///      ▼                               │         │
/// ┌───────────┐                        │         │
/// │ Triggered │───────────────────────►│         │
/// └─────┬─────┘                        │         │
///       │                              │         │
///       ├──────────┐                   │         │
///       ▼          ▼                   ▼         ▼
/// ┌──────────┐ ┌──────────┐        (Terminal States)
/// │  Filled  │ │ Rejected │
/// └──────────┘ └──────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderState {
    /// Order created, waiting for trigger condition
    Pending,
    /// Order triggered, waiting for fill (next bar or immediate)
    Triggered,
    /// Order filled → becomes a Position
    Filled,
    /// Order rejected (margin, risk, session, etc.)
    Rejected,
    /// Order cancelled by user/system
    Cancelled,
    /// Order expired (`GoodTillDate` reached)
    Expired,
}

impl OrderState {
    /// Returns the allowed transitions from the current state.
    #[must_use]
    pub fn allowed_transitions(&self) -> &[OrderState] {
        match self {
            OrderState::Pending => &[
                OrderState::Triggered,
                OrderState::Cancelled,
                OrderState::Expired,
            ],
            OrderState::Triggered => &[OrderState::Filled, OrderState::Rejected],
            OrderState::Filled
            | OrderState::Rejected
            | OrderState::Cancelled
            | OrderState::Expired => &[],
        }
    }

    /// Checks if this is a terminal (final) state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderState::Filled | OrderState::Rejected | OrderState::Cancelled | OrderState::Expired
        )
    }

    /// Checks if a transition to the target state is valid.
    #[must_use]
    pub fn can_transition_to(&self, target: OrderState) -> bool {
        self.allowed_transitions().contains(&target)
    }

    /// Checks if the order is still active (not terminal).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }
}

impl fmt::Display for OrderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderState::Pending => write!(f, "Pending"),
            OrderState::Triggered => write!(f, "Triggered"),
            OrderState::Filled => write!(f, "Filled"),
            OrderState::Rejected => write!(f, "Rejected"),
            OrderState::Cancelled => write!(f, "Cancelled"),
            OrderState::Expired => write!(f, "Expired"),
        }
    }
}

/// Position state for open positions.
///
/// # State Diagram
/// ```text
/// ┌──────┐
/// │ Open │────────────┬────────────────┐
/// └──┬───┘            │                │
///    │                ▼                │
///    │           ┌──────────┐          │
///    │           │ Modified │◄─────┐   │
///    │           └────┬─────┘      │   │
///    │                │            │   │
///    │                ├────────────┘   │
///    │                │                │
///    ▼                ▼                ▼
/// ┌────────────────────────────────────────┐
/// │                Closed                  │
/// └────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionState {
    /// Position is open (entry filled)
    Open,
    /// SL/TP was modified (trailing, manual adjustment)
    Modified,
    /// Position was closed (SL/TP/Manual/Timeout)
    Closed,
}

impl PositionState {
    /// Returns the allowed transitions from the current state.
    #[must_use]
    pub fn allowed_transitions(&self) -> &[PositionState] {
        match self {
            PositionState::Open | PositionState::Modified => {
                &[PositionState::Modified, PositionState::Closed]
            }
            PositionState::Closed => &[], // Terminal
        }
    }

    /// Checks if this is a terminal (final) state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, PositionState::Closed)
    }

    /// Checks if a transition to the target state is valid.
    #[must_use]
    pub fn can_transition_to(&self, target: PositionState) -> bool {
        self.allowed_transitions().contains(&target)
    }
}

impl fmt::Display for PositionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PositionState::Open => write!(f, "Open"),
            PositionState::Modified => write!(f, "Modified"),
            PositionState::Closed => write!(f, "Closed"),
        }
    }
}

/// Records a state transition with audit information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition<S: Clone> {
    /// Previous state
    pub from: S,
    /// New state
    pub to: S,
    /// Timestamp of transition in nanoseconds (Unix epoch UTC)
    pub timestamp_ns: i64,
    /// Human-readable reason for the transition
    pub reason: String,
}

impl<S: Clone + fmt::Display> StateTransition<S> {
    /// Creates a new state transition record.
    pub fn new(from: S, to: S, timestamp_ns: i64, reason: impl Into<String>) -> Self {
        Self {
            from,
            to,
            timestamp_ns,
            reason: reason.into(),
        }
    }
}

impl<S: Clone + fmt::Display> fmt::Display for StateTransition<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {} ({})", self.from, self.to, self.reason)
    }
}

/// Deterministic trigger order policy documentation.
///
/// This struct documents the processing order for a single bar.
/// The order is normative and must be followed for reproducibility.
///
/// Pending orders are placed at candle close and therefore cannot trigger in
/// the same candle as placement. If a pending order triggers in a bar, the
/// fill occurs in that same bar, allowing SL/TP to hit in the entry candle.
pub struct TriggerOrderPolicy;

impl TriggerOrderPolicy {
    /// Processing order for a single bar (normative).
    ///
    /// 1. **Pending-Trigger**: Limit/Stop → open (FIFO)
    /// 2. **Exit-Check**: SL/TP checks for open positions
    /// 3. **Trade-Management**: Rule-based actions (timeout, etc.)
    /// 4. **Equity-Update**: Portfolio equity curve update
    /// 5. **New-Signals**: Strategy generates new orders
    pub const PROCESSING_ORDER: &'static [&'static str] = &[
        "1. Pending-Trigger (FIFO by created_at_ns, order_id)",
        "2. Exit-Check (SL/TP checks for open positions)",
        "3. Trade-Management (rule-based actions)",
        "4. Equity-Update (portfolio equity curve)",
        "5. New-Signals (strategy generates new orders)",
    ];

    /// When both SL and TP are hit in the same bar, SL always wins.
    pub const SL_OVER_TP_PRIORITY: bool = true;

    /// Pending orders only trigger starting from the bar after creation.
    pub const PENDING_NEXT_BAR_RULE: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_state_transitions() {
        // From Pending
        assert!(OrderState::Pending.can_transition_to(OrderState::Triggered));
        assert!(OrderState::Pending.can_transition_to(OrderState::Cancelled));
        assert!(OrderState::Pending.can_transition_to(OrderState::Expired));
        assert!(!OrderState::Pending.can_transition_to(OrderState::Filled));
        assert!(!OrderState::Pending.can_transition_to(OrderState::Rejected));

        // From Triggered
        assert!(OrderState::Triggered.can_transition_to(OrderState::Filled));
        assert!(OrderState::Triggered.can_transition_to(OrderState::Rejected));
        assert!(!OrderState::Triggered.can_transition_to(OrderState::Pending));
        assert!(!OrderState::Triggered.can_transition_to(OrderState::Cancelled));

        // Terminal states
        assert!(OrderState::Filled.is_terminal());
        assert!(OrderState::Rejected.is_terminal());
        assert!(OrderState::Cancelled.is_terminal());
        assert!(OrderState::Expired.is_terminal());
        assert!(!OrderState::Pending.is_terminal());
        assert!(!OrderState::Triggered.is_terminal());
    }

    #[test]
    fn test_position_state_transitions() {
        // From Open
        assert!(PositionState::Open.can_transition_to(PositionState::Modified));
        assert!(PositionState::Open.can_transition_to(PositionState::Closed));

        // From Modified
        assert!(PositionState::Modified.can_transition_to(PositionState::Modified));
        assert!(PositionState::Modified.can_transition_to(PositionState::Closed));

        // Terminal state
        assert!(PositionState::Closed.is_terminal());
        assert!(!PositionState::Open.is_terminal());
        assert!(!PositionState::Modified.is_terminal());

        // No transitions from Closed
        assert!(PositionState::Closed.allowed_transitions().is_empty());
    }

    #[test]
    fn test_state_transition_record() {
        let transition = StateTransition::new(
            OrderState::Pending,
            OrderState::Triggered,
            1_000_000_000,
            "Price crossed entry level",
        );

        assert_eq!(transition.from, OrderState::Pending);
        assert_eq!(transition.to, OrderState::Triggered);
        assert_eq!(transition.timestamp_ns, 1_000_000_000);
        assert_eq!(transition.reason, "Price crossed entry level");
    }

    #[test]
    fn test_order_state_display() {
        assert_eq!(format!("{}", OrderState::Pending), "Pending");
        assert_eq!(format!("{}", OrderState::Triggered), "Triggered");
        assert_eq!(format!("{}", OrderState::Filled), "Filled");
    }

    #[test]
    fn test_position_state_display() {
        assert_eq!(format!("{}", PositionState::Open), "Open");
        assert_eq!(format!("{}", PositionState::Modified), "Modified");
        assert_eq!(format!("{}", PositionState::Closed), "Closed");
    }
}
