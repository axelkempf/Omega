use thiserror::Error;

/// Errors that can occur while loading or validating market/news data.
#[derive(Debug, Error)]
pub enum DataError {
    #[error("File not found: {0} ({1})")]
    FileNotFound(String, String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("Invalid column type: {0}")]
    InvalidColumnType(String),

    #[error("Empty data")]
    EmptyData,

    #[error("Corrupt data: {0}")]
    CorruptData(String),

    #[error("Invalid spread: {0}")]
    InvalidSpread(String),

    #[error("Alignment failure: {0}")]
    AlignmentFailure(String),

    #[error("Insufficient data: need {required}, have {available}")]
    InsufficientData { required: usize, available: usize },
}
