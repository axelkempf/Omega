//! Position management and tracking.
//!
//! Manages open positions with support for modification (SL/TP adjustment)
//! and tracking of position-level metrics.

use crate::error::PortfolioError;
use omega_types::{Direction, Position, Signal};
use serde_json::Value as JsonValue;

/// Manages a collection of open positions.
#[derive(Debug, Default)]
pub struct PositionManager {
    /// Open positions
    positions: Vec<Position>,
    /// Next position ID to assign
    next_id: u64,
    /// Maximum allowed positions
    max_positions: usize,
}

impl PositionManager {
    /// Creates a new position manager with the given maximum positions.
    #[must_use]
    pub fn new(max_positions: usize) -> Self {
        Self {
            positions: Vec::new(),
            next_id: 1,
            max_positions,
        }
    }

    /// Checks if a new position can be opened.
    #[must_use]
    pub fn can_open(&self) -> bool {
        self.positions.len() < self.max_positions
    }

    /// Returns the number of available position slots.
    #[must_use]
    pub fn available_slots(&self) -> usize {
        self.max_positions.saturating_sub(self.positions.len())
    }

    /// Opens a new position.
    ///
    /// # Arguments
    /// * `signal` - The trading signal
    /// * `fill_price` - Actual fill price (after slippage)
    /// * `size` - Position size in lots
    /// * `entry_time_ns` - Entry timestamp
    ///
    /// # Returns
    /// The assigned position ID.
    ///
    /// # Errors
    /// Returns an error if position limits are exceeded or size is invalid.
    pub fn open_position(
        &mut self,
        signal: &Signal,
        fill_price: f64,
        size: f64,
        entry_time_ns: i64,
    ) -> Result<u64, PortfolioError> {
        if !size.is_finite() || size <= 0.0 {
            return Err(PortfolioError::InvalidSize(
                "position size must be positive and finite".to_string(),
            ));
        }

        if !self.can_open() {
            return Err(PortfolioError::MaxPositionsReached {
                max: self.max_positions,
            });
        }

        let id = self.next_id;
        self.next_id += 1;

        let position = Position {
            id,
            direction: signal.direction,
            entry_time_ns,
            entry_price: fill_price,
            size,
            stop_loss: signal.stop_loss,
            take_profit: signal.take_profit,
            scenario_id: signal.scenario_id,
            meta: signal.meta.clone(),
        };

        self.positions.push(position);
        Ok(id)
    }

    /// Opens a position from raw parameters (not from a Signal).
    ///
    /// # Errors
    /// Returns an error if position limits are exceeded or size is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn open_position_raw(
        &mut self,
        direction: Direction,
        fill_price: f64,
        size: f64,
        stop_loss: f64,
        take_profit: f64,
        entry_time_ns: i64,
        scenario_id: u8,
        meta: JsonValue,
    ) -> Result<u64, PortfolioError> {
        if !size.is_finite() || size <= 0.0 {
            return Err(PortfolioError::InvalidSize(
                "position size must be positive and finite".to_string(),
            ));
        }

        if !self.can_open() {
            return Err(PortfolioError::MaxPositionsReached {
                max: self.max_positions,
            });
        }

        let id = self.next_id;
        self.next_id += 1;

        let position = Position {
            id,
            direction,
            entry_time_ns,
            entry_price: fill_price,
            size,
            stop_loss,
            take_profit,
            scenario_id,
            meta,
        };

        self.positions.push(position);
        Ok(id)
    }

    /// Closes a position and removes it from the manager.
    ///
    /// Returns the closed position if found.
    pub fn close_position(&mut self, position_id: u64) -> Option<Position> {
        let idx = self.positions.iter().position(|p| p.id == position_id)?;
        Some(self.positions.remove(idx))
    }

    /// Gets a reference to a position by ID.
    #[must_use]
    pub fn get_position(&self, position_id: u64) -> Option<&Position> {
        self.positions.iter().find(|p| p.id == position_id)
    }

    /// Gets a mutable reference to a position by ID.
    pub fn get_position_mut(&mut self, position_id: u64) -> Option<&mut Position> {
        self.positions.iter_mut().find(|p| p.id == position_id)
    }

    /// Modifies a position's stop-loss.
    ///
    /// # Errors
    /// Returns an error if the position is missing or the new SL is invalid.
    pub fn modify_stop_loss(
        &mut self,
        position_id: u64,
        new_stop_loss: f64,
    ) -> Result<(), PortfolioError> {
        let position = self
            .get_position_mut(position_id)
            .ok_or(PortfolioError::PositionNotFound(position_id))?;

        // Validate SL direction
        match position.direction {
            Direction::Long => {
                if new_stop_loss >= position.entry_price {
                    return Err(PortfolioError::InvalidStopLoss(
                        "Long SL must be below entry".to_string(),
                    ));
                }
            }
            Direction::Short => {
                if new_stop_loss <= position.entry_price {
                    return Err(PortfolioError::InvalidStopLoss(
                        "Short SL must be above entry".to_string(),
                    ));
                }
            }
        }

        position.stop_loss = new_stop_loss;
        Ok(())
    }

    /// Modifies a position's take-profit.
    ///
    /// # Errors
    /// Returns an error if the position is missing or the new TP is invalid.
    pub fn modify_take_profit(
        &mut self,
        position_id: u64,
        new_take_profit: f64,
    ) -> Result<(), PortfolioError> {
        let position = self
            .get_position_mut(position_id)
            .ok_or(PortfolioError::PositionNotFound(position_id))?;

        // Validate TP direction
        match position.direction {
            Direction::Long => {
                if new_take_profit <= position.entry_price {
                    return Err(PortfolioError::InvalidTakeProfit(
                        "Long TP must be above entry".to_string(),
                    ));
                }
            }
            Direction::Short => {
                if new_take_profit >= position.entry_price {
                    return Err(PortfolioError::InvalidTakeProfit(
                        "Short TP must be below entry".to_string(),
                    ));
                }
            }
        }

        position.take_profit = new_take_profit;
        Ok(())
    }

    /// Moves stop-loss to break-even (entry price).
    ///
    /// # Errors
    /// Returns an error if the position is missing.
    pub fn move_to_break_even(&mut self, position_id: u64) -> Result<(), PortfolioError> {
        let position = self
            .get_position_mut(position_id)
            .ok_or(PortfolioError::PositionNotFound(position_id))?;

        position.stop_loss = position.entry_price;
        Ok(())
    }

    /// Calculates unrealized `PnL` for a position at the given price.
    #[must_use]
    pub fn unrealized_pnl(&self, position_id: u64, current_price: f64) -> Option<f64> {
        let position = self.get_position(position_id)?;
        Some(Self::calculate_pnl(position, current_price))
    }

    /// Calculates total unrealized `PnL` for all positions.
    #[must_use]
    pub fn total_unrealized_pnl(&self, current_price: f64) -> f64 {
        self.positions
            .iter()
            .map(|p| Self::calculate_pnl(p, current_price))
            .sum()
    }

    /// Helper to calculate `PnL` for a position.
    fn calculate_pnl(position: &Position, current_price: f64) -> f64 {
        match position.direction {
            Direction::Long => (current_price - position.entry_price) * position.size,
            Direction::Short => (position.entry_price - current_price) * position.size,
        }
    }

    /// Calculates the risk (distance to SL * size) for a position.
    #[must_use]
    pub fn position_risk(&self, position_id: u64) -> Option<f64> {
        let position = self.get_position(position_id)?;
        Some((position.entry_price - position.stop_loss).abs() * position.size)
    }

    /// Calculates total risk for all positions.
    #[must_use]
    pub fn total_risk(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| (p.entry_price - p.stop_loss).abs() * p.size)
            .sum()
    }

    /// Returns all open positions.
    #[must_use]
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Returns a mutable iterator over positions.
    pub fn positions_mut(&mut self) -> impl Iterator<Item = &mut Position> {
        self.positions.iter_mut()
    }

    /// Returns the number of open positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Checks if there are no open positions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns positions filtered by direction.
    #[must_use]
    pub fn positions_by_direction(&self, direction: &Direction) -> Vec<&Position> {
        self.positions
            .iter()
            .filter(|p| &p.direction == direction)
            .collect()
    }

    /// Returns positions that entered on or after the given timestamp.
    #[must_use]
    pub fn positions_since(&self, timestamp_ns: i64) -> Vec<&Position> {
        self.positions
            .iter()
            .filter(|p| p.entry_time_ns >= timestamp_ns)
            .collect()
    }

    /// Checks if a position was entered in the current bar.
    #[must_use]
    pub fn is_entry_candle(&self, position_id: u64, current_bar_ns: i64) -> bool {
        self.get_position(position_id)
            .is_some_and(|p| p.entry_time_ns == current_bar_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use omega_types::OrderType;
    use serde_json::json;

    fn make_signal(direction: Direction, entry: f64, sl: f64, tp: f64) -> Signal {
        Signal {
            direction,
            order_type: OrderType::Market,
            entry_price: entry,
            stop_loss: sl,
            take_profit: tp,
            size: Some(1.0),
            scenario_id: 0,
            tags: vec![],
            meta: json!({}),
        }
    }

    #[test]
    fn test_open_position() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2001, 1.0, 1_000_000)
            .unwrap();

        assert_eq!(id, 1);
        assert_eq!(manager.len(), 1);

        let position = manager.get_position(id).unwrap();
        assert_relative_eq!(position.entry_price, 1.2001, epsilon = 1e-10);
    }

    #[test]
    fn test_max_positions() {
        let mut manager = PositionManager::new(2);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);

        manager
            .open_position(&signal, 1.2001, 1.0, 1_000_000)
            .unwrap();
        manager
            .open_position(&signal, 1.2002, 1.0, 2_000_000)
            .unwrap();

        // Third should fail
        let result = manager.open_position(&signal, 1.2003, 1.0, 3_000_000);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PortfolioError::MaxPositionsReached { .. }
        ));
    }

    #[test]
    fn test_invalid_size_rejected() {
        let mut manager = PositionManager::new(2);
        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);

        let result = manager.open_position(&signal, 1.2000, 0.0, 1_000_000);
        assert!(matches!(result, Err(PortfolioError::InvalidSize(_))));
    }

    #[test]
    fn test_close_position() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2001, 1.0, 1_000_000)
            .unwrap();

        let closed = manager.close_position(id);
        assert!(closed.is_some());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_modify_stop_loss() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2000, 1.0, 1_000_000)
            .unwrap();

        // Move SL up (trailing)
        manager.modify_stop_loss(id, 1.1980).unwrap();
        assert_relative_eq!(
            manager.get_position(id).unwrap().stop_loss,
            1.1980,
            epsilon = 1e-10
        );

        // Invalid: SL above entry for long
        let result = manager.modify_stop_loss(id, 1.2010);
        assert!(result.is_err());
    }

    #[test]
    fn test_break_even() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2000, 1.0, 1_000_000)
            .unwrap();

        manager.move_to_break_even(id).unwrap();
        assert_relative_eq!(
            manager.get_position(id).unwrap().stop_loss,
            1.2000,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_unrealized_pnl_long() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2000, 1.0, 1_000_000)
            .unwrap();

        // Price went up
        let pnl = manager.unrealized_pnl(id, 1.2050).unwrap();
        assert_relative_eq!(pnl, 0.0050, epsilon = 1e-10);

        // Price went down
        let pnl = manager.unrealized_pnl(id, 1.1980).unwrap();
        assert_relative_eq!(pnl, -0.0020, epsilon = 1e-10);
    }

    #[test]
    fn test_unrealized_pnl_short() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Short, 1.2000, 1.2050, 1.1900);
        let id = manager
            .open_position(&signal, 1.2000, 1.0, 1_000_000)
            .unwrap();

        // Price went down (profit for short)
        let pnl = manager.unrealized_pnl(id, 1.1950).unwrap();
        assert_relative_eq!(pnl, 0.0050, epsilon = 1e-10);
    }

    #[test]
    fn test_position_risk() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2000, 1.0, 1_000_000)
            .unwrap();

        let risk = manager.position_risk(id).unwrap();
        assert_relative_eq!(risk, 0.0050, epsilon = 1e-10);
    }

    #[test]
    fn test_is_entry_candle() {
        let mut manager = PositionManager::new(5);

        let signal = make_signal(Direction::Long, 1.2000, 1.1950, 1.2100);
        let id = manager
            .open_position(&signal, 1.2000, 1.0, 1_000_000)
            .unwrap();

        assert!(manager.is_entry_candle(id, 1_000_000));
        assert!(!manager.is_entry_candle(id, 2_000_000));
    }
}
