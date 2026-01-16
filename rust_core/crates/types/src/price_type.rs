use serde::{Deserialize, Serialize};

/// Price type for bid/ask series selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PriceType {
    /// Bid prices.
    Bid,
    /// Ask prices.
    Ask,
}

impl PriceType {
    /// Returns lowercase string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            PriceType::Bid => "bid",
            PriceType::Ask => "ask",
        }
    }
}

impl std::fmt::Display for PriceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Error parsing price type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsePriceTypeError;

impl std::fmt::Display for ParsePriceTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid price type")
    }
}

impl std::error::Error for ParsePriceTypeError {}

impl std::str::FromStr for PriceType {
    type Err = ParsePriceTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "bid" => Ok(PriceType::Bid),
            "ask" => Ok(PriceType::Ask),
            _ => Err(ParsePriceTypeError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_type_as_str() {
        assert_eq!(PriceType::Bid.as_str(), "bid");
        assert_eq!(PriceType::Ask.as_str(), "ask");
    }

    #[test]
    fn test_price_type_from_str() {
        assert_eq!("bid".parse::<PriceType>(), Ok(PriceType::Bid));
        assert_eq!("ASK".parse::<PriceType>(), Ok(PriceType::Ask));
        assert!("invalid".parse::<PriceType>().is_err());
    }
}
