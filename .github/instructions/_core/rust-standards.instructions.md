---
description: 'Canonical Rust coding standards for Omega project - Single Source of Truth'
applyTo: '**/*.rs'
---

# Rust Standards

> Kanonische Rust-Standards für das Omega-Projekt.
> Diese Datei ist die Single Source of Truth – alle anderen Instruktionen referenzieren hierher.

---

## Version & Toolchain

- **Edition:** 2024 (spezifiziert in `rust-toolchain.toml`)
- **Pinning:** `rust-toolchain.toml` MUSS versioniert werden
- **Lockfile:** `Cargo.lock` MUSS versioniert werden (Determinismus)

---

## Style Guide

### Allgemein

| Regel | Standard |
|-------|----------|
| Formatter | `rustfmt` |
| Linter | `cargo clippy -- -D warnings` |
| Zeilenlänge | 100 Zeichen |
| Einrückung | 4 Spaces |

### Naming Conventions (RFC 430)

| Element | Konvention | Beispiel |
|---------|-----------|----------|
| Variablen | `snake_case` | `trade_count`, `candle_data` |
| Funktionen | `snake_case` | `calculate_lot_size()` |
| Structs/Enums | `CamelCase` | `TradeManager`, `SignalType` |
| Konstanten | `UPPER_CASE` | `MAX_LOT_SIZE`, `DEFAULT_TIMEOUT` |
| Traits | `CamelCase` | `Strategy`, `Indicator` |
| Modules | `snake_case` | `data_loader`, `execution` |
| Crates | `snake_case` | `omega_core`, `backtest_engine` |

### Anti-Patterns

- ❌ `unwrap()` oder `expect()` ohne Begründung
- ❌ `panic!` in Library-Code
- ❌ Globaler mutable State
- ❌ Unnötiges `clone()` – borrowing bevorzugen
- ❌ `unsafe` ohne Dokumentation
- ❌ Warnings ignorieren

---

## Ownership & Borrowing

### Grundregeln

```rust
// ✅ Borrowing bevorzugen
fn process_candles(candles: &[Candle]) -> Vec<Signal> { ... }

// ✅ &mut nur wenn nötig
fn update_portfolio(portfolio: &mut Portfolio, trade: &Trade) { ... }

// ❌ Unnötiges Cloning
fn process_candles(candles: Vec<Candle>) -> Vec<Signal> { ... }  // Ownership nicht nötig!
```

### Lifetimes

```rust
// ✅ Explizite Lifetime wenn nötig
struct BarContext<'a> {
    candles: &'a [Candle],
    indicators: &'a IndicatorCache,
}

// ✅ Compiler-Inferenz nutzen wenn möglich
fn get_last_candle(candles: &[Candle]) -> Option<&Candle> { ... }
```

### Concurrency

| Typ | Use Case |
|-----|----------|
| `Rc<T>` | Single-threaded Reference Counting |
| `Arc<T>` | Multi-threaded Reference Counting |
| `RefCell<T>` | Interior Mutability (single-threaded) |
| `Mutex<T>` | Interior Mutability (multi-threaded) |
| `RwLock<T>` | Read-heavy multi-threaded access |

---

## Error Handling

### Grundregeln

```rust
// ✅ Result für recoverable Errors
pub fn load_candles(path: &Path) -> Result<Vec<Candle>, DataError> { ... }

// ✅ ? Operator für Propagation
fn process_data(config: &Config) -> Result<Report, ProcessError> {
    let candles = load_candles(&config.data_path)?;
    let signals = generate_signals(&candles)?;
    Ok(build_report(&signals))
}

// ❌ NIEMALS in Library-Code
fn load_candles(path: &Path) -> Vec<Candle> {
    std::fs::read(path).unwrap()  // Panic!
}
```

### Custom Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Data loading failed: {0}")]
    DataError(#[from] DataError),
    
    #[error("Invalid configuration: {field}")]
    ConfigError { field: String },
    
    #[error("Execution failed: {0}")]
    ExecutionError(String),
}

// Result-Alias für Konsistenz
pub type CoreResult<T> = Result<T, CoreError>;
```

### FFI Error Contract (Omega V2)

```rust
// Setup-Fehler → Python Exception via PyO3
#[pyfunction]
fn run_backtest(config_json: &str) -> PyResult<String> {
    let config: Config = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid config: {e}")))?;
    
    // Runtime-Fehler → JSON Error Result
    match engine::run(&config) {
        Ok(result) => Ok(serde_json::to_string(&result).unwrap()),
        Err(e) => Ok(serde_json::to_string(&ErrorResult::from(e)).unwrap()),
    }
}
```

---

## Traits & API Design

### Standard Traits implementieren

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

// Default für optionale Konfiguration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub symbol: String,
    pub timeframe: String,
    #[serde(default = "default_warmup")]
    pub warmup: usize,
}

fn default_warmup() -> usize { 500 }
```

### Trait-basierte Abstraktion

```rust
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn required_indicators(&self) -> Vec<IndicatorSpec>;
    fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal>;
}

pub trait Indicator: Send + Sync {
    fn name(&self) -> &str;
    fn compute(&self, candles: &[Candle]) -> Vec<f64>;
}
```

---

## Performance Guidelines

### Iterators bevorzugen

```rust
// ✅ Iterator-Chain (lazy, effizient)
let valid_trades: Vec<_> = trades
    .iter()
    .filter(|t| t.is_valid())
    .map(|t| t.calculate_pnl())
    .collect();

// ❌ Index-basierte Loops
let mut valid_trades = Vec::new();
for i in 0..trades.len() {
    if trades[i].is_valid() {
        valid_trades.push(trades[i].calculate_pnl());
    }
}
```

### Zero-Copy wo möglich

```rust
// ✅ Borrowing für Parameters
fn process_symbol(symbol: &str) -> Result<Report, Error> { ... }

// ❌ Ownership ohne Grund
fn process_symbol(symbol: String) -> Result<Report, Error> { ... }
```

### Allocations minimieren

```rust
// ✅ Pre-allocate bei bekannter Größe
let mut results = Vec::with_capacity(candles.len());

// ✅ In-place Mutation wenn möglich
candles.sort_by_key(|c| c.timestamp_ns);

// ❌ Unnötige Intermediate Collections
let sorted: Vec<_> = candles.iter().cloned().collect();
```

---

## Testing

### Test-Organisation

```rust
// Unit Tests im selben File
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_pnl_long_profit() {
        let trade = Trade::new(Direction::Long, 1.1000, 1.1050, 0.1);
        assert_eq!(trade.calculate_pnl(), 50.0);
    }
    
    #[test]
    fn test_calculate_pnl_handles_zero_size() {
        let trade = Trade::new(Direction::Long, 1.1000, 1.1050, 0.0);
        assert_eq!(trade.calculate_pnl(), 0.0);
    }
}
```

### Property Tests (proptest)

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn pnl_is_symmetric(
        entry in 0.5f64..2.0,
        exit in 0.5f64..2.0,
        size in 0.01f64..10.0
    ) {
        let long_pnl = calculate_pnl(Direction::Long, entry, exit, size);
        let short_pnl = calculate_pnl(Direction::Short, entry, exit, size);
        
        // Long und Short PnL sind symmetrisch
        prop_assert!((long_pnl + short_pnl).abs() < 1e-10);
    }
}
```

### Integration Tests

```rust
// tests/integration_test.rs
use omega_core::*;

#[test]
fn test_full_backtest_workflow() {
    let config = load_test_config("fixtures/basic_config.json");
    let result = run_backtest(&config).unwrap();
    
    assert!(result.trades.len() > 0);
    assert!(result.metrics.total_trades > 0);
}
```

---

## Documentation

### Rustdoc Format

```rust
/// Calculate the profit/loss for a trade.
///
/// # Arguments
///
/// * `direction` - Trade direction (Long or Short)
/// * `entry_price` - Entry price in quote currency
/// * `exit_price` - Exit price in quote currency
/// * `size` - Position size in lots
///
/// # Returns
///
/// Profit/loss in account currency (positive = profit)
///
/// # Example
///
/// ```
/// use omega_core::calculate_pnl;
///
/// let pnl = calculate_pnl(Direction::Long, 1.1000, 1.1050, 0.1)?;
/// assert_eq!(pnl, 50.0);
/// ```
pub fn calculate_pnl(
    direction: Direction,
    entry_price: f64,
    exit_price: f64,
    size: f64,
) -> Result<f64, CalculationError> { ... }
```

---

## Tools & CI

### Formatierung

```bash
# Formatierung prüfen
cargo fmt --all -- --check

# Formatierung anwenden
cargo fmt --all
```

### Linting

```bash
# Clippy mit Warnings als Errors
cargo clippy --all-targets -- -D warnings

# Alle Checks
cargo clippy --all-targets --all-features -- -D warnings
```

### Tests

```bash
# Alle Tests
cargo test --all

# Mit Output
cargo test --all -- --nocapture
```

---

## Quick Reference

| Aspekt | Standard |
|--------|----------|
| Edition | 2024 |
| Formatter | rustfmt |
| Linter | clippy -D warnings |
| Error Handling | Result<T, E> + thiserror |
| Serialization | serde + serde_json |
| Testing | cargo test + proptest |
| Docs | rustdoc (/// comments) |
| Naming | RFC 430 |
| Panics | NIEMALS in Library-Code |
| unsafe | Nur mit Dokumentation |
