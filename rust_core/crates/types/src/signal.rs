/// Direction of a trade
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Long position
    Long,
    /// Short position
    Short,
}

/// Type of order to execute
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    /// Market order
    Market,
    /// Limit order
    Limit,
    /// Stop order
    Stop,
}

/// Trading signal from strategy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Signal {
    /// Direction of the signal
    pub direction: Direction,
    /// Order type
    pub order_type: OrderType,
    /// Entry price
    pub entry_price: f64,
    /// Stop loss price
    pub stop_loss: f64,
    /// Take profit price
    pub take_profit: f64,
    /// Position size (optional, calculated if None)
    pub size: Option<f64>,
    /// Scenario ID for categorization
    pub scenario_id: u8,
    /// Tags for additional metadata
    pub tags: Vec<String>,
    /// Additional metadata as JSON
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_serde_roundtrip() {
        let signal = Signal {
            direction: Direction::Long,
            order_type: OrderType::Market,
            entry_price: 1.1000,
            stop_loss: 1.0950,
            take_profit: 1.1100,
            size: Some(0.1),
            scenario_id: 1,
            tags: vec!["breakout".to_string()],
            meta: serde_json::json!({"confidence": 0.85}),
        };

        let json = serde_json::to_string(&signal).unwrap();
        let deserialized: Signal = serde_json::from_str(&json).unwrap();

        assert_eq!(signal.direction, deserialized.direction);
        assert_eq!(signal.order_type, deserialized.order_type);
        assert_eq!(signal.scenario_id, deserialized.scenario_id);
    }

    #[test]
    fn test_direction_serialization() {
        assert_eq!(serde_json::to_string(&Direction::Long).unwrap(), "\"long\"");
        assert_eq!(
            serde_json::to_string(&Direction::Short).unwrap(),
            "\"short\""
        );
    }
}
