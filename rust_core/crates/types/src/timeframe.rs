/// Trading timeframe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Timeframe {
    /// 1 minute
    M1,
    /// 5 minutes
    M5,
    /// 15 minutes
    M15,
    /// 30 minutes
    M30,
    /// 1 hour
    H1,
    /// 4 hours
    H4,
    /// 1 day
    D1,
    /// 1 week
    W1,
}

/// Error parsing timeframe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseTimeframeError;

impl std::fmt::Display for ParseTimeframeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid timeframe string")
    }
}

impl std::error::Error for ParseTimeframeError {}

impl std::str::FromStr for Timeframe {
    type Err = ParseTimeframeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "M1" => Ok(Timeframe::M1),
            "M5" => Ok(Timeframe::M5),
            "M15" => Ok(Timeframe::M15),
            "M30" => Ok(Timeframe::M30),
            "H1" => Ok(Timeframe::H1),
            "H4" => Ok(Timeframe::H4),
            "D1" => Ok(Timeframe::D1),
            "W1" => Ok(Timeframe::W1),
            _ => Err(ParseTimeframeError),
        }
    }
}

impl Timeframe {
    /// Returns duration in seconds
    #[must_use]
    pub fn to_seconds(&self) -> u64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604_800,
        }
    }

    /// Convert to string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "M1",
            Timeframe::M5 => "M5",
            Timeframe::M15 => "M15",
            Timeframe::M30 => "M30",
            Timeframe::H1 => "H1",
            Timeframe::H4 => "H4",
            Timeframe::D1 => "D1",
            Timeframe::W1 => "W1",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_to_seconds() {
        assert_eq!(Timeframe::M1.to_seconds(), 60);
        assert_eq!(Timeframe::M5.to_seconds(), 300);
        assert_eq!(Timeframe::M15.to_seconds(), 900);
        assert_eq!(Timeframe::M30.to_seconds(), 1800);
        assert_eq!(Timeframe::H1.to_seconds(), 3600);
        assert_eq!(Timeframe::H4.to_seconds(), 14400);
        assert_eq!(Timeframe::D1.to_seconds(), 86400);
        assert_eq!(Timeframe::W1.to_seconds(), 604_800);
    }

    #[test]
    fn test_timeframe_from_str() {
        use std::str::FromStr;
        assert_eq!(Timeframe::from_str("M1"), Ok(Timeframe::M1));
        assert_eq!(Timeframe::from_str("m1"), Ok(Timeframe::M1));
        assert_eq!(Timeframe::from_str("H1"), Ok(Timeframe::H1));
        assert_eq!(Timeframe::from_str("h1"), Ok(Timeframe::H1));
        assert!(Timeframe::from_str("invalid").is_err());
    }

    #[test]
    fn test_timeframe_serde_roundtrip() {
        let tf = Timeframe::H1;
        let json = serde_json::to_string(&tf).unwrap();
        let deserialized: Timeframe = serde_json::from_str(&json).unwrap();
        assert_eq!(tf, deserialized);
    }
}
