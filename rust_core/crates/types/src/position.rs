use crate::signal::Direction;

/// Open position
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Position {
    /// Position ID
    pub id: u64,
    /// Direction of the position
    pub direction: Direction,
    /// Entry timestamp in nanoseconds
    pub entry_time_ns: i64,
    /// Entry price
    pub entry_price: f64,
    /// Position size
    pub size: f64,
    /// Stop loss price
    pub stop_loss: f64,
    /// Take profit price
    pub take_profit: f64,
    /// Scenario ID for categorization
    pub scenario_id: u8,
    /// Additional metadata as JSON
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_serde_roundtrip() {
        let position = Position {
            id: 1,
            direction: Direction::Long,
            entry_time_ns: 1234567890000000000,
            entry_price: 1.1000,
            size: 0.1,
            stop_loss: 1.0950,
            take_profit: 1.1100,
            scenario_id: 1,
            meta: serde_json::json!({"note": "test"}),
        };

        let json = serde_json::to_string(&position).unwrap();
        let deserialized: Position = serde_json::from_str(&json).unwrap();

        assert_eq!(position.id, deserialized.id);
        assert_eq!(position.direction, deserialized.direction);
        assert_eq!(position.entry_time_ns, deserialized.entry_time_ns);
    }
}
