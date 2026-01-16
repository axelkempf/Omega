/// Repräsentiert eine OHLCV-Kerze
/// `timestamp_ns` ist die **Open-Time** (nicht Close-Time)
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Candle {
    /// Unix epoch nanoseconds UTC (Open-Time)
    pub timestamp_ns: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_serde_roundtrip() {
        let candle = Candle {
            timestamp_ns: 1_234_567_890_000_000_000,
            open: 1.1000,
            high: 1.1020,
            low: 1.0980,
            close: 1.1010,
            volume: 1000.0,
        };

        let json = serde_json::to_string(&candle).unwrap();
        let deserialized: Candle = serde_json::from_str(&json).unwrap();

        assert_eq!(candle, deserialized);
    }
}
