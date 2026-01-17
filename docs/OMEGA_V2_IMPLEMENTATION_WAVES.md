## Omega V2 – Implementierungs-Wellen

> **Status**: Implementierungsphase
> **Erstellt**: 14. Januar 2026
> **Zweck**: Strukturierte Wellen-Implementierung des V2-Systems mit vollständigen Prompts für Codex-Max
> **Referenz**: Basiert auf allen 19 OMEGA_V2_*_PLAN.md Dokumenten

---

## Wellen-Übersicht

| Welle | Fokus | Crates | Geschätzte Komplexität | Abhängigkeiten |
|-------|-------|--------|------------------------|----------------|
| **W0** | Foundation Setup | `types` + Workspace | Mittel | Keine |
| **W1** | Data Layer | `data` | Hoch | W0 |
| **W2** | Indicators | `indicators` | Sehr hoch | W0 |
| **W3** | Execution Core | `execution` + `portfolio` | Hoch | W0 |
| **W4** | Strategy Layer | `strategy` + `trade_mgmt` | Sehr hoch | W0, W2, W3 |
| **W5** | Orchestration | `backtest` | Hoch | W0-W4 |
| **W6** | Output + FFI | `metrics` + `ffi` | Mittel | W0, W5 |
| **W7** | Integration | Python Wrapper + Parity | Mittel | W0-W6 |

---

## W0: Foundation Setup

### Beschreibung
Erstellt das Rust-Workspace-Fundament und das `types`-Crate mit allen gemeinsamen Datenstrukturen.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Foundation Setup (Wave 0)

## Kontext
Du implementierst das Fundament für ein Rust-natives Backtesting-System. Dies ist Wave 0 der Implementierung.

## Ziel
Erstelle den Rust-Workspace und das `types`-Crate mit allen gemeinsamen Datenstrukturen.

## Verzeichnisstruktur (zu erstellen)

```
rust_core/
├── Cargo.toml                    # Workspace-Definition
├── Cargo.lock                    # (wird generiert)
├── rust-toolchain.toml           # Rust 1.75+ stable pinning
├── .cargo/
│   └── config.toml               # Cargo-Konfiguration
└── crates/
    └── types/
        ├── Cargo.toml
        └── src/
            ├── lib.rs            # Re-exports
            ├── candle.rs         # Candle-Struktur
            ├── signal.rs         # Signal-Enum
            ├── trade.rs          # Trade-Struktur
            ├── position.rs       # Position-Struktur
            ├── config.rs         # BacktestConfig
            ├── result.rs         # BacktestResult
            ├── timeframe.rs      # Timeframe-Enum
            └── error.rs          # CoreError
```

## Technische Anforderungen

### Workspace Cargo.toml
```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
authors = ["Omega Team"]
license = "MIT"
rust-version = "1.75"

[workspace.dependencies]
# Interne Crates (werden später hinzugefügt)
omega_types = { path = "crates/types" }

# Externe Crates
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
```

### rust-toolchain.toml
```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

### types/Cargo.toml
```toml
[package]
name = "omega_types"
version.workspace = true
edition.workspace = true

[dependencies]
serde = { workspace = true }
thiserror = { workspace = true }
```

## Datenstrukturen (normativ)

### Candle (candle.rs)
```rust
/// Repräsentiert eine OHLCV-Kerze
/// timestamp_ns ist die **Open-Time** (nicht Close-Time)
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Candle {
    /// Unix epoch nanoseconds UTC (Open-Time)
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
```

### Signal (signal.rs)
```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Direction {
    Long,
    Short,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Signal {
    pub direction: Direction,
    pub order_type: OrderType,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub size: Option<f64>,
    pub scenario_id: u8,
    pub tags: Vec<String>,
    #[serde(default)]
    pub meta: serde_json::Value,
}
```

### Position (position.rs)
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Position {
    pub id: u64,
    pub direction: Direction,
    pub entry_time_ns: i64,
    pub entry_price: f64,
    pub size: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub scenario_id: u8,
    #[serde(default)]
    pub meta: serde_json::Value,
}
```

### Trade (trade.rs)
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Trade {
    pub entry_time_ns: i64,
    pub exit_time_ns: i64,
    pub direction: Direction,
    pub symbol: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub size: f64,
    pub result: f64,           // PnL in account_currency
    pub r_multiple: f64,
    pub reason: ExitReason,
    pub scenario_id: u8,
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    TakeProfit,
    StopLoss,
    Timeout,
    BreakEvenStopLoss,
    TrailingStopLoss,
    Manual,
}
```

### Timeframe (timeframe.rs)
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Timeframe {
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1,
    W1,
}

impl Timeframe {
    /// Returns duration in seconds
    pub fn to_seconds(&self) -> u64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604800,
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "M1" => Some(Timeframe::M1),
            "M5" => Some(Timeframe::M5),
            // ... etc
        }
    }
}
```

### BacktestConfig (config.rs)
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BacktestConfig {
    pub schema_version: String,
    pub strategy_name: String,
    pub symbol: String,
    pub start_date: String,
    pub end_date: String,
    pub run_mode: RunMode,
    pub data_mode: DataMode,
    pub execution_variant: ExecutionVariant,
    pub timeframes: TimeframeConfig,
    #[serde(default = "default_warmup")]
    pub warmup_bars: usize,
    #[serde(default)]
    pub rng_seed: Option<u64>,
    #[serde(default)]
    pub sessions: Option<Vec<SessionConfig>>,
    #[serde(default)]
    pub account: AccountConfig,
    #[serde(default)]
    pub costs: CostsConfig,
    #[serde(default)]
    pub news_filter: Option<NewsFilterConfig>,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub trade_management: Option<TradeManagementConfig>,
    #[serde(default)]
    pub strategy_parameters: serde_json::Value,
}

fn default_warmup() -> usize { 500 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    #[default]
    Dev,
    Prod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataMode {
    #[default]
    Candle,
    Tick,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionVariant {
    #[default]
    V2,
    V1Parity,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimeframeConfig {
    pub primary: String,
    #[serde(default)]
    pub additional: Vec<String>,
    #[serde(default = "default_additional_source")]
    pub additional_source: String,
}

fn default_additional_source() -> String { "separate_parquet".to_string() }

// ============================================
// SUB-CONFIGS (VOLLSTÄNDIG gemäß CONFIG_SCHEMA_PLAN)
// ============================================

/// Session-Fenster für Trading (UTC)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// Start-Zeit (HH:MM, UTC)
    pub start: String,
    /// End-Zeit (HH:MM, UTC) - wenn end <= start, gilt als cross-midnight
    pub end: String,
}

impl SessionConfig {
    /// Prüft ob ein Zeitpunkt (Sekunden seit Mitternacht UTC) in dieser Session liegt
    pub fn contains(&self, seconds_of_day: u32) -> bool {
        let start_secs = parse_hhmm_to_seconds(&self.start);
        let end_secs = parse_hhmm_to_seconds(&self.end);

        if end_secs > start_secs {
            // Normal: 08:00 - 17:00
            seconds_of_day >= start_secs && seconds_of_day < end_secs
        } else {
            // Cross-midnight: 22:00 - 06:00
            seconds_of_day >= start_secs || seconds_of_day < end_secs
        }
    }
}

fn parse_hhmm_to_seconds(hhmm: &str) -> u32 {
    let parts: Vec<&str> = hhmm.split(':').collect();
    let hours: u32 = parts[0].parse().unwrap_or(0);
    let minutes: u32 = parts.get(1).and_then(|m| m.parse().ok()).unwrap_or(0);
    hours * 3600 + minutes * 60
}

/// Account-/Sizing-Konfiguration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccountConfig {
    #[serde(default = "default_initial_balance")]
    pub initial_balance: f64,
    #[serde(default = "default_account_currency")]
    pub account_currency: String,
    #[serde(default = "default_risk_per_trade")]
    pub risk_per_trade: f64,
    #[serde(default = "default_max_positions")]
    pub max_positions: usize,
}

fn default_initial_balance() -> f64 { 10000.0 }
fn default_account_currency() -> String { "EUR".to_string() }
fn default_risk_per_trade() -> f64 { 100.0 }
fn default_max_positions() -> usize { 1 }

impl Default for AccountConfig {
    fn default() -> Self {
        Self {
            initial_balance: default_initial_balance(),
            account_currency: default_account_currency(),
            risk_per_trade: default_risk_per_trade(),
            max_positions: default_max_positions(),
        }
    }
}

/// Kostenmodell-Konfiguration (Toggles + Multipliers)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostsConfig {
    #[serde(default = "default_costs_enabled")]
    pub enabled: bool,
    #[serde(default = "default_multiplier")]
    pub fee_multiplier: f64,
    #[serde(default = "default_multiplier")]
    pub slippage_multiplier: f64,
    #[serde(default = "default_multiplier")]
    pub spread_multiplier: f64,
    /// Symbol-spezifische pip_size (falls nicht aus symbol_specs.yaml)
    #[serde(default)]
    pub pip_size: Option<f64>,
    /// pip_buffer_factor für SL/TP Checks
    #[serde(default = "default_pip_buffer_factor")]
    pub pip_buffer_factor: f64,
}

fn default_costs_enabled() -> bool { true }
fn default_multiplier() -> f64 { 1.0 }
fn default_pip_buffer_factor() -> f64 { 0.5 }

impl Default for CostsConfig {
    fn default() -> Self {
        Self {
            enabled: default_costs_enabled(),
            fee_multiplier: default_multiplier(),
            slippage_multiplier: default_multiplier(),
            spread_multiplier: default_multiplier(),
            pip_size: None,
            pip_buffer_factor: default_pip_buffer_factor(),
        }
    }
}

/// News-Filter-Konfiguration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NewsFilterConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_news_minutes")]
    pub minutes_before: u32,
    #[serde(default = "default_news_minutes")]
    pub minutes_after: u32,
    #[serde(default = "default_min_impact")]
    pub min_impact: NewsImpact,
    /// Currencies to filter - wenn None, aus symbol abgeleitet
    #[serde(default)]
    pub currencies: Option<Vec<String>>,
}

fn default_news_minutes() -> u32 { 30 }
fn default_min_impact() -> NewsImpact { NewsImpact::Medium }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NewsImpact {
    Low,
    #[default]
    Medium,
    High,
}

impl Default for NewsFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            minutes_before: default_news_minutes(),
            minutes_after: default_news_minutes(),
            min_impact: default_min_impact(),
            currencies: None,
        }
    }
}

/// Logging-Konfiguration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoggingConfig {
    #[serde(default)]
    pub enable_entry_logging: bool,
    #[serde(default = "default_logging_mode")]
    pub logging_mode: String,
}

fn default_logging_mode() -> String { "trades_only".to_string() }

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enable_entry_logging: false,
            logging_mode: default_logging_mode(),
        }
    }
}

/// Trade-Management-Konfiguration (MVP: MaxHoldingTime)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TradeManagementConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_stop_update_policy")]
    pub stop_update_policy: StopUpdatePolicy,
    #[serde(default)]
    pub rules: TradeManagementRulesConfig,
}

fn default_stop_update_policy() -> StopUpdatePolicy { StopUpdatePolicy::ApplyNextBar }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopUpdatePolicy {
    #[default]
    ApplyNextBar,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TradeManagementRulesConfig {
    #[serde(default)]
    pub max_holding_time: MaxHoldingTimeConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaxHoldingTimeConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub max_holding_minutes: Option<u64>,
    /// Nur für bestimmte Szenarien (leer = alle)
    #[serde(default)]
    pub only_scenarios: Vec<u8>,
}

fn default_true() -> bool { true }

impl Default for MaxHoldingTimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_holding_minutes: None,
            only_scenarios: Vec::new(),
        }
    }
}
```

### BacktestResult (result.rs)
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BacktestResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trades: Option<Vec<Trade>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equity_curve: Option<Vec<EquityPoint>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ResultMeta>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorResult {
    pub category: String,
    pub message: String,
    #[serde(default)]
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EquityPoint {
    pub timestamp_ns: i64,
    pub equity: f64,
    pub balance: f64,
    pub drawdown: f64,
    pub high_water: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    pub total_trades: u64,
    pub wins: u64,
    pub losses: u64,
    pub win_rate: f64,
    pub profit_gross: f64,
    pub profit_net: f64,
    pub fees_total: f64,
    pub max_drawdown: f64,
    pub max_drawdown_abs: f64,
    pub avg_r_multiple: f64,
    pub profit_factor: f64,
    // ... weitere Metriken
}
```

### CoreError (error.rs)
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Indicator error: {0}")]
    Indicator(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Portfolio error: {0}")]
    Portfolio(String),

    #[error("Strategy error: {0}")]
    Strategy(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
```

## Tests
Erstelle Unit-Tests für:
1. Serde-Roundtrip für alle Structs
2. Timeframe-Parsing und -Konvertierung
3. Default-Werte für Config-Structs

## Qualitätskriterien
- `cargo build` erfolgreich
- `cargo test` alle Tests bestehen
- `cargo clippy -- -D warnings` keine Warnungen
- `cargo fmt --check` formatiert
- Alle `pub` Items mit `///` Dokumentation

## Referenz-Dokumente
- OMEGA_V2_MODULE_STRUCTURE_PLAN.md (Abschnitt 3.1)
- OMEGA_V2_CONFIG_SCHEMA_PLAN.md (vollständig)
- OMEGA_V2_OUTPUT_CONTRACT_PLAN.md (Trade/Result Schema)
- OMEGA_V2_TECH_STACK_PLAN.md (Rust Edition, Dependencies)
```

### Akzeptanzkriterien W0

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Workspace kompiliert | `cargo build` erfolgreich |
| A2 | Alle Structs serialisierbar | Serde-Roundtrip-Tests |
| A3 | Timeframe-Enum vollständig | M1-W1, mit `to_seconds()` |
| A4 | Config-Schema komplett | Alle Pflichtfelder aus Plan |
| A5 | Error-Typen definiert | CoreError mit allen Varianten |
| A6 | Dokumentation | Alle `pub` Items dokumentiert |

### Referenzen
- [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) - Abschnitt 3.1
- [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) - Vollständig
- [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) - Trade/Result Schema
- [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) - Rust Edition, Dependencies

---

## W1: Data Layer

### Beschreibung
Implementiert das `data`-Crate für Parquet-Loading, Bid/Ask-Alignment und Datenvalidierung.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Data Layer (Wave 1)

## Kontext
Du implementierst das Data-Loading-Layer für ein Rust-natives Backtesting-System. Wave 0 (types) ist bereits implementiert.

## Ziel
Erstelle das `data`-Crate mit Parquet-Loading, Bid/Ask-Alignment und Datenvalidierung.

## Verzeichnisstruktur

```
rust_core/crates/data/
├── Cargo.toml
└── src/
    ├── lib.rs            # Re-exports
    ├── loader.rs         # Parquet-Laden
    ├── alignment.rs      # Bid/Ask Alignment (Inner Join)
    ├── store.rs          # CandleStore
    ├── validation.rs     # Datenqualitäts-Checks
    ├── market_hours.rs   # Session-Filter
    ├── news.rs           # News Calendar Loading (optional)
    └── error.rs          # DataError
```

## Cargo.toml

```toml
[package]
name = "omega_data"
version.workspace = true
edition.workspace = true

[dependencies]
omega_types = { workspace = true }
arrow = "51"
parquet = "51"
thiserror = { workspace = true }
tracing = "0.1"
```

## Implementierung

### loader.rs - Parquet Loading
```rust
use arrow::array::{Float64Array, TimestampNanosecondArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::Path;
use crate::error::DataError;
use omega_types::Candle;

/// Lädt Candles aus einer Parquet-Datei
/// Schema erwartet: UTC time (Timestamp ns), Open, High, Low, Close, Volume
pub fn load_candles(path: &Path) -> Result<Vec<Candle>, DataError> {
    let file = std::fs::File::open(path)
        .map_err(|e| DataError::FileNotFound(path.display().to_string(), e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| DataError::ParseError(e.to_string()))?;

    let reader = builder.build()
        .map_err(|e| DataError::ParseError(e.to_string()))?;

    let mut candles = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| DataError::ParseError(e.to_string()))?;

        // Extract columns by name
        let timestamp_col = batch.column_by_name("UTC time")
            .ok_or_else(|| DataError::MissingColumn("UTC time".to_string()))?;
        let timestamps = timestamp_col
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .ok_or_else(|| DataError::InvalidColumnType("UTC time".to_string()))?;

        // Similar for Open, High, Low, Close, Volume...

        for i in 0..batch.num_rows() {
            candles.push(Candle {
                timestamp_ns: timestamps.value(i),
                open: open_arr.value(i),
                high: high_arr.value(i),
                low: low_arr.value(i),
                close: close_arr.value(i),
                volume: volume_arr.value(i),
            });
        }
    }

    Ok(candles)
}
```

### alignment.rs - Bid/Ask Alignment
```rust
use omega_types::Candle;
use std::collections::HashSet;

/// Ergebnis des Bid/Ask Alignments
#[derive(Debug)]
pub struct AlignedData {
    pub bid: Vec<Candle>,
    pub ask: Vec<Candle>,
    pub timestamps: Vec<i64>,
    pub alignment_stats: AlignmentStats,
}

#[derive(Debug)]
pub struct AlignmentStats {
    pub bid_count_before: usize,
    pub ask_count_before: usize,
    pub aligned_count: usize,
    pub discarded_count: usize,
}

/// Führt Inner Join auf Timestamps durch
/// KRITISCH: Keine Interpolation! Nur gemeinsame Timestamps behalten.
pub fn align_bid_ask(bid: Vec<Candle>, ask: Vec<Candle>) -> Result<AlignedData, DataError> {
    let bid_timestamps: HashSet<i64> = bid.iter().map(|c| c.timestamp_ns).collect();
    let ask_timestamps: HashSet<i64> = ask.iter().map(|c| c.timestamp_ns).collect();

    // Inner Join: nur Timestamps die in BEIDEN vorhanden sind
    let common: HashSet<i64> = bid_timestamps.intersection(&ask_timestamps).copied().collect();

    if common.is_empty() {
        return Err(DataError::AlignmentFailure(
            "No common timestamps between bid and ask".to_string()
        ));
    }

    // Filter und sortiere
    let mut aligned_bid: Vec<Candle> = bid.into_iter()
        .filter(|c| common.contains(&c.timestamp_ns))
        .collect();
    let mut aligned_ask: Vec<Candle> = ask.into_iter()
        .filter(|c| common.contains(&c.timestamp_ns))
        .collect();

    aligned_bid.sort_by_key(|c| c.timestamp_ns);
    aligned_ask.sort_by_key(|c| c.timestamp_ns);

    let timestamps: Vec<i64> = aligned_bid.iter().map(|c| c.timestamp_ns).collect();

    let stats = AlignmentStats {
        bid_count_before: bid_timestamps.len(),
        ask_count_before: ask_timestamps.len(),
        aligned_count: aligned_bid.len(),
        discarded_count: bid_timestamps.len() + ask_timestamps.len() - 2 * aligned_bid.len(),
    };

    // Warning loggen wenn > 1% verloren
    if stats.discarded_count as f64 / stats.bid_count_before as f64 > 0.01 {
        tracing::warn!(
            "Alignment discarded {} bars ({:.2}%)",
            stats.discarded_count,
            100.0 * stats.discarded_count as f64 / stats.bid_count_before as f64
        );
    }

    Ok(AlignedData {
        bid: aligned_bid,
        ask: aligned_ask,
        timestamps,
        alignment_stats: stats,
    })
}
```

### validation.rs - Datenqualitäts-Checks
```rust
use omega_types::Candle;
use crate::error::DataError;

/// Validiert Candle-Daten gemäß Data Governance Contract
pub fn validate_candles(candles: &[Candle]) -> Result<(), DataError> {
    if candles.is_empty() {
        return Err(DataError::EmptyData);
    }

    for (i, candle) in candles.iter().enumerate() {
        // CHECK 1: Keine NaN/Inf
        if !candle.open.is_finite() || !candle.high.is_finite()
           || !candle.low.is_finite() || !candle.close.is_finite() {
            return Err(DataError::CorruptData(format!(
                "NaN/Inf at index {}: {:?}", i, candle
            )));
        }

        // CHECK 2: OHLC Konsistenz
        if candle.low > candle.open || candle.low > candle.close
           || candle.high < candle.open || candle.high < candle.close
           || candle.low > candle.high {
            return Err(DataError::CorruptData(format!(
                "Invalid OHLC at index {}: low={}, high={}, open={}, close={}",
                i, candle.low, candle.high, candle.open, candle.close
            )));
        }

        // CHECK 3: Timestamps monoton steigend
        if i > 0 && candle.timestamp_ns <= candles[i-1].timestamp_ns {
            return Err(DataError::CorruptData(format!(
                "Non-monotonic timestamp at index {}: {} <= {}",
                i, candle.timestamp_ns, candles[i-1].timestamp_ns
            )));
        }
    }

    Ok(())
}

/// Validiert Bid/Ask Spread (Open/Close müssen Bid <= Ask erfüllen)
pub fn validate_spread(bid: &[Candle], ask: &[Candle]) -> Result<(), DataError> {
    for (i, (b, a)) in bid.iter().zip(ask.iter()).enumerate() {
        if b.open > a.open || b.close > a.close {
            return Err(DataError::InvalidSpread(format!(
                "Invalid spread at index {}: bid_close={} > ask_close={}",
                i, b.close, a.close
            )));
        }
    }
    Ok(())
}
```

### store.rs - CandleStore
```rust
use omega_types::{Candle, Timeframe};

/// Hauptdatenstruktur für aligned Bid/Ask Candles
#[derive(Debug)]
pub struct CandleStore {
    pub bid: Vec<Candle>,
    pub ask: Vec<Candle>,
    pub timestamps: Vec<i64>,
    pub timeframe: Timeframe,
    pub symbol: String,
    pub warmup_bars: usize,
}

impl CandleStore {
    pub fn len(&self) -> usize {
        self.bid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bid.is_empty()
    }

    pub fn trading_bars(&self) -> usize {
        self.len().saturating_sub(self.warmup_bars)
    }

    /// Gibt (bid, ask) für Index zurück
    pub fn get(&self, idx: usize) -> Option<(&Candle, &Candle)> {
        if idx < self.len() {
            Some((&self.bid[idx], &self.ask[idx]))
        } else {
            None
        }
    }
}

/// Multi-Timeframe Store für Primary + HTF
#[derive(Debug)]
pub struct MultiTfStore {
    pub primary: CandleStore,
    pub htf: Option<CandleStore>,
    /// Mapping: Primary-Index → HTF-Index (letzte abgeschlossene HTF-Bar)
    pub htf_index_map: Vec<Option<usize>>,
}
```

### error.rs - DataError
```rust
use thiserror::Error;

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
```

## Path Resolution
Implementiere Environment-basierte Pfadauflösung:
```rust
/// Löst Datenpfad auf basierend auf Config
pub fn resolve_data_path(symbol: &str, timeframe: &str, side: &str) -> PathBuf {
    let root = std::env::var("OMEGA_DATA_PARQUET_ROOT")
        .unwrap_or_else(|_| "data/parquet".to_string());

    PathBuf::from(root)
        .join(symbol)
        .join(format!("{}_{}_{}.parquet", symbol, timeframe, side))
}
```

## Tests

### Positive Tests
1. Parquet-Loading mit Fixture-Dateien (BID/ASK separat)
2. Alignment-Invarianten (gleiche Länge, keine Duplikate)
3. Validierungs-Checks (NaN, OHLC-Konsistenz, Monotonie)
4. Path-Resolution für verschiedene Env-Varianten

### Negative Test-Fixtures (Data Governance - Hard Fail)
Referenz: [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md)

Implementiere dedizierte Test-Fixtures für jede Fail-Kondition:

```rust
// tests/fixtures/mod.rs
pub mod invalid_fixtures {
    use std::path::PathBuf;
    
    /// Fixture mit nicht-monotonen Timestamps
    pub fn non_monotonic_timestamps() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/non_monotonic_timestamps.parquet")
    }
    
    /// Fixture mit Duplikat-Timestamps (gleicher Timestamp, unterschiedliche OHLCV)
    pub fn duplicate_timestamps_divergent() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/duplicate_timestamps_divergent.parquet")
    }
    
    /// Fixture mit fehlenden Pflicht-Spalten
    pub fn missing_column_close() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/missing_column_close.parquet")
    }
    
    /// Fixture mit falschem Spalten-Typ (String statt f64)
    pub fn wrong_column_type() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/wrong_column_type.parquet")
    }
    
    /// Fixture mit NaN in OHLCV
    pub fn nan_in_ohlcv() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/nan_in_ohlcv.parquet")
    }
    
    /// Fixture mit OHLC-Inkonsistenz (High < Low)
    pub fn ohlc_inconsistent() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/ohlc_inconsistent.parquet")
    }
    
    /// Fixture mit Bid > Ask (Invalid Spread)
    pub fn bid_exceeds_ask() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/bid_exceeds_ask.parquet")
    }
    
    /// Leere Parquet-Datei
    pub fn empty_parquet() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/empty.parquet")
    }
    
    /// Korrupte Parquet-Datei (binary garbage)
    pub fn corrupt_parquet() -> PathBuf {
        PathBuf::from("tests/fixtures/invalid/corrupt.parquet")
    }
}
```

### Fixture-Generator (proptest)
```rust
// tests/generators.rs
use proptest::prelude::*;
use omega_types::Candle;

/// Generiert valide Candle-Sequenzen für Property-Tests
pub fn valid_candle_sequence(len: usize) -> impl Strategy<Value = Vec<Candle>> {
    prop::collection::vec(valid_candle(), len..=len)
        .prop_map(|mut candles| {
            // Sortiere nach Timestamp und mache monoton
            candles.sort_by_key(|c| c.timestamp_ns);
            let mut ts = 1704067200_000_000_000i64; // 2024-01-01 00:00:00 UTC
            for candle in &mut candles {
                candle.timestamp_ns = ts;
                ts += 60_000_000_000; // +1 Minute
            }
            candles
        })
}

fn valid_candle() -> impl Strategy<Value = Candle> {
    (
        1.0f64..2.0,  // base price (z.B. EURUSD range)
        0.0001..0.01, // spread range
    ).prop_map(|(base, spread)| {
        let low = base - spread;
        let high = base + spread;
        Candle {
            timestamp_ns: 0, // wird in valid_candle_sequence überschrieben
            open: base,
            high,
            low,
            close: base + (spread * 0.5),
            volume: 100.0,
        }
    })
}

/// Generiert invalide Sequenzen für Negative-Tests
pub fn non_monotonic_sequence() -> impl Strategy<Value = Vec<Candle>> {
    valid_candle_sequence(10).prop_map(|mut candles| {
        // Vertausche zwei Timestamps für Non-Monotonie
        if candles.len() >= 2 {
            let tmp = candles[0].timestamp_ns;
            candles[0].timestamp_ns = candles[1].timestamp_ns;
            candles[1].timestamp_ns = tmp;
        }
        candles
    })
}
```

### Governance-Check Tests
```rust
#[cfg(test)]
mod governance_tests {
    use super::*;
    use crate::fixtures::invalid_fixtures::*;
    
    #[test]
    fn test_reject_non_monotonic_timestamps() {
        let result = load_and_validate(non_monotonic_timestamps());
        assert!(matches!(result, Err(DataError::CorruptData(_))));
    }
    
    #[test]
    fn test_reject_duplicate_timestamps_divergent() {
        let result = load_and_validate(duplicate_timestamps_divergent());
        assert!(matches!(result, Err(DataError::CorruptData(_))));
    }
    
    #[test]
    fn test_reject_missing_column() {
        let result = load_and_validate(missing_column_close());
        assert!(matches!(result, Err(DataError::MissingColumn(_))));
    }
    
    #[test]
    fn test_reject_wrong_column_type() {
        let result = load_and_validate(wrong_column_type());
        assert!(matches!(result, Err(DataError::InvalidColumnType(_))));
    }
    
    #[test]
    fn test_reject_nan_in_ohlcv() {
        let result = load_and_validate(nan_in_ohlcv());
        assert!(matches!(result, Err(DataError::CorruptData(_))));
    }
    
    #[test]
    fn test_reject_ohlc_inconsistent() {
        let result = load_and_validate(ohlc_inconsistent());
        assert!(matches!(result, Err(DataError::CorruptData(_))));
    }
    
    #[test]
    fn test_reject_invalid_spread() {
        let result = load_and_validate_bid_ask(
            bid_exceeds_ask(),
            valid_ask_fixture()
        );
        assert!(matches!(result, Err(DataError::InvalidSpread(_))));
    }
    
    #[test]
    fn test_reject_empty_parquet() {
        let result = load_and_validate(empty_parquet());
        assert!(matches!(result, Err(DataError::EmptyData)));
    }
    
    #[test]
    fn test_reject_corrupt_parquet() {
        let result = load_and_validate(corrupt_parquet());
        assert!(matches!(result, Err(DataError::ParseError(_))));
    }
}
```

## Referenz-Dokumente
- OMEGA_V2_DATA_FLOW_PLAN.md (Phase 2)
- OMEGA_V2_DATA_GOVERNANCE_PLAN.md (Alignment, Validation)
- OMEGA_V2_CONFIG_SCHEMA_PLAN.md (Path Resolution)
```

### Akzeptanzkriterien W1

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Parquet-Loading funktioniert | Test mit Fixture-Parquet |
| A2 | Alignment korrekt | Inner-Join, keine Interpolation |
| A3 | Validierung streng | Hard-fail bei Corrupt Data |
| A4 | Timestamps monoton | Validation-Check implementiert |
| A5 | Bid <= Ask geprüft | Spread-Validierung |
| A6 | Path Resolution | Environment-aware |

### Referenzen
- [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) - Phase 2 vollständig
- [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) - Alignment, Validation
- [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) - Path Resolution

---

## W2: Indicators

### Beschreibung
Implementiert das `indicators`-Crate mit allen für MRZ benötigten Indikatoren.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Indicators (Wave 2)

## Kontext
Du implementierst die Indikator-Engine für ein Rust-natives Backtesting-System. Wave 0 (types) ist bereits implementiert.

## Ziel
Erstelle das `indicators`-Crate mit Indicator-Trait, Cache und allen für Mean Reversion Z-Score benötigten Indikatoren.

## Verzeichnisstruktur

```
rust_core/crates/indicators/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── traits.rs         # Indicator Trait
    ├── cache.rs          # IndicatorCache
    ├── registry.rs       # IndicatorRegistry
    ├── error.rs          # IndicatorError
    └── impl/
        ├── mod.rs
        ├── ema.rs        # Exponential Moving Average
        ├── sma.rs        # Simple Moving Average
        ├── atr.rs        # Average True Range (Wilder)
        ├── bollinger.rs  # Bollinger Bands
        ├── z_score.rs    # Standard Z-Score
        ├── kalman_mean.rs         # Kalman Filter Level
        ├── kalman_zscore.rs       # Kalman Z-Score
        ├── garch_volatility.rs    # GARCH(1,1) Volatility
        ├── kalman_garch_zscore.rs # Kalman+GARCH Z-Score
        └── vol_cluster.rs         # Vol-Cluster State
```

## Indicator Trait (traits.rs)
```rust
use omega_types::Candle;

/// Konfiguration für einen Indikator
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndicatorSpec {
    pub name: String,
    pub params: IndicatorParams,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndicatorParams {
    Period(usize),
    Bollinger { period: usize, std_factor_x100: u32 },
    Kalman { window: usize, r_x1000: u32, q_x1000: u32 },
    Garch { alpha_x1000: u32, beta_x1000: u32, omega_x1000000: u32 },
    // ... weitere Parameter-Varianten
}

/// Trait für alle Indikatoren
pub trait Indicator: Send + Sync {
    /// Berechnet den Indikator für alle Candles
    /// Gibt Vec<f64> zurück mit gleicher Länge wie candles
    /// NaN für Bars wo Warmup nicht erfüllt ist
    fn compute(&self, candles: &[Candle]) -> Vec<f64>;

    /// Name des Indikators
    fn name(&self) -> &str;

    /// Minimale Bars für validen Output
    fn warmup_periods(&self) -> usize;
}
```

## Core-Indikatoren

### EMA (ema.rs)
```rust
pub struct EMA {
    pub period: usize,
}

impl Indicator for EMA {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let mut result = vec![f64::NAN; candles.len()];
        if candles.len() < self.period {
            return result;
        }

        let multiplier = 2.0 / (self.period as f64 + 1.0);

        // Initial SMA
        let initial_sma: f64 = candles[..self.period]
            .iter()
            .map(|c| c.close)
            .sum::<f64>() / self.period as f64;

        result[self.period - 1] = initial_sma;

        // EMA calculation
        for i in self.period..candles.len() {
            let prev_ema = result[i - 1];
            result[i] = (candles[i].close - prev_ema) * multiplier + prev_ema;
        }

        result
    }

    fn name(&self) -> &str { "EMA" }
    fn warmup_periods(&self) -> usize { self.period }
}
```

### ATR - Wilder (atr.rs)
```rust
pub struct ATR {
    pub period: usize,
}

impl ATR {
    fn true_range(candle: &Candle, prev_close: f64) -> f64 {
        let hl = candle.high - candle.low;
        let hc = (candle.high - prev_close).abs();
        let lc = (candle.low - prev_close).abs();
        hl.max(hc).max(lc)
    }
}

impl Indicator for ATR {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let mut result = vec![f64::NAN; candles.len()];
        if candles.len() < self.period + 1 {
            return result;
        }

        // Calculate TR series
        let mut tr = vec![0.0; candles.len()];
        tr[0] = candles[0].high - candles[0].low;
        for i in 1..candles.len() {
            tr[i] = Self::true_range(&candles[i], candles[i-1].close);
        }

        // Initial ATR (simple average)
        let initial: f64 = tr[1..=self.period].iter().sum::<f64>() / self.period as f64;
        result[self.period] = initial;

        // Wilder smoothing: ATR = (prev_ATR * (n-1) + TR) / n
        for i in (self.period + 1)..candles.len() {
            result[i] = (result[i-1] * (self.period - 1) as f64 + tr[i]) / self.period as f64;
        }

        result
    }

    fn name(&self) -> &str { "ATR" }
    fn warmup_periods(&self) -> usize { self.period + 1 }
}
```

### Bollinger Bands (bollinger.rs)
```rust
pub struct BollingerBands {
    pub period: usize,
    pub std_factor: f64,
}

pub struct BollingerResult {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
}

impl BollingerBands {
    pub fn compute_all(&self, candles: &[Candle]) -> BollingerResult {
        let len = candles.len();
        let mut upper = vec![f64::NAN; len];
        let mut middle = vec![f64::NAN; len];
        let mut lower = vec![f64::NAN; len];

        if len < self.period {
            return BollingerResult { upper, middle, lower };
        }

        for i in (self.period - 1)..len {
            let window: Vec<f64> = candles[(i + 1 - self.period)..=i]
                .iter()
                .map(|c| c.close)
                .collect();

            let sma = window.iter().sum::<f64>() / self.period as f64;
            let variance = window.iter()
                .map(|x| (x - sma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std = variance.sqrt();

            middle[i] = sma;
            upper[i] = sma + self.std_factor * std;
            lower[i] = sma - self.std_factor * std;
        }

        BollingerResult { upper, middle, lower }
    }
}
```

### Z-Score (z_score.rs)
```rust
pub struct ZScore {
    pub window: usize,
}

impl Indicator for ZScore {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let mut result = vec![f64::NAN; candles.len()];

        for i in (self.window - 1)..candles.len() {
            let window: Vec<f64> = candles[(i + 1 - self.window)..=i]
                .iter()
                .map(|c| c.close)
                .collect();

            let mean = window.iter().sum::<f64>() / self.window as f64;
            let variance = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.window as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                result[i] = (candles[i].close - mean) / std;
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    fn name(&self) -> &str { "Z_SCORE" }
    fn warmup_periods(&self) -> usize { self.window }
}
```

### Kalman Filter (kalman_mean.rs, kalman_zscore.rs)
```rust
pub struct KalmanFilter {
    pub r: f64,  // Measurement noise
    pub q: f64,  // Process noise
}

impl KalmanFilter {
    /// Berechnet Kalman-geglättete Preisreihe
    pub fn compute_level(&self, prices: &[f64]) -> Vec<f64> {
        let mut result = vec![f64::NAN; prices.len()];
        if prices.is_empty() {
            return result;
        }

        let mut x = prices[0];  // State estimate
        let mut p = 1.0;        // Error covariance

        result[0] = x;

        for i in 1..prices.len() {
            // Predict
            let x_pred = x;
            let p_pred = p + self.q;

            // Update
            let k = p_pred / (p_pred + self.r);
            x = x_pred + k * (prices[i] - x_pred);
            p = (1.0 - k) * p_pred;

            result[i] = x;
        }

        result
    }
}

pub struct KalmanZScore {
    pub window: usize,
    pub r: f64,
    pub q: f64,
}

impl Indicator for KalmanZScore {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let kalman = KalmanFilter { r: self.r, q: self.q };
        let kalman_level = kalman.compute_level(&prices);

        let mut result = vec![f64::NAN; candles.len()];

        for i in (self.window - 1)..candles.len() {
            let residuals: Vec<f64> = ((i + 1 - self.window)..=i)
                .map(|j| prices[j] - kalman_level[j])
                .collect();

            let mean = residuals.iter().sum::<f64>() / self.window as f64;
            let std = (residuals.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / self.window as f64).sqrt();

            let current_residual = prices[i] - kalman_level[i];

            if std > 1e-10 {
                result[i] = (current_residual - mean) / std;
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    fn name(&self) -> &str { "KALMAN_Z" }
    fn warmup_periods(&self) -> usize { self.window }
}
```

### GARCH(1,1) (garch_volatility.rs)
```rust
pub struct GarchVolatility {
    pub alpha: f64,
    pub beta: f64,
    pub omega: f64,
    pub use_log_returns: bool,
    pub scale: f64,
    pub min_periods: usize,
    pub sigma_floor: f64,
}

impl Indicator for GarchVolatility {
    fn compute(&self, candles: &[Candle]) -> Vec<f64> {
        let mut result = vec![f64::NAN; candles.len()];
        if candles.len() < self.min_periods + 1 {
            return result;
        }

        // Calculate returns
        let returns: Vec<f64> = (1..candles.len())
            .map(|i| {
                if self.use_log_returns {
                    (candles[i].close / candles[i-1].close).ln()
                } else {
                    (candles[i].close - candles[i-1].close) / candles[i-1].close
                }
            })
            .collect();

        // Initial variance (sample variance of first min_periods returns)
        let init_returns = &returns[..self.min_periods];
        let mean = init_returns.iter().sum::<f64>() / self.min_periods as f64;
        let init_var = init_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / self.min_periods as f64;

        let mut sigma2 = init_var.max(self.sigma_floor.powi(2));
        result[self.min_periods] = sigma2.sqrt() * self.scale;

        // GARCH recursion: sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
        for i in self.min_periods..returns.len() {
            let r_prev = returns[i - 1];
            sigma2 = self.omega + self.alpha * r_prev.powi(2) + self.beta * sigma2;
            sigma2 = sigma2.max(self.sigma_floor.powi(2));
            result[i + 1] = sigma2.sqrt() * self.scale;
        }

        result
    }

    fn name(&self) -> &str { "GARCH_VOL" }
    fn warmup_periods(&self) -> usize { self.min_periods + 1 }
}
```

## Indicator Cache (cache.rs)
```rust
use std::collections::HashMap;
use omega_types::Candle;
use crate::traits::{Indicator, IndicatorSpec};

pub struct IndicatorCache {
    cache: HashMap<IndicatorSpec, Vec<f64>>,
}

impl IndicatorCache {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    /// Holt oder berechnet einen Indikator
    pub fn get_or_compute(
        &mut self,
        spec: &IndicatorSpec,
        candles: &[Candle],
        indicator: &dyn Indicator,
    ) -> &[f64] {
        if !self.cache.contains_key(spec) {
            let values = indicator.compute(candles);
            self.cache.insert(spec.clone(), values);
        }
        self.cache.get(spec).unwrap()
    }

    /// Gibt gecachten Wert für Index zurück
    pub fn get(&self, spec: &IndicatorSpec, idx: usize) -> Option<f64> {
        self.cache.get(spec).and_then(|v| v.get(idx).copied())
    }
}
```

## Indicator Registry (registry.rs)
```rust
use std::collections::HashMap;
use std::sync::Arc;
use omega_types::Candle;
use crate::traits::{Indicator, IndicatorSpec, IndicatorParams};
use crate::error::IndicatorError;

/// Factory-Funktion für Indikator-Erstellung
pub type IndicatorFactory = Box<dyn Fn(&IndicatorParams) -> Result<Arc<dyn Indicator>, IndicatorError> + Send + Sync>;

/// Registry für alle verfügbaren Indikatoren
pub struct IndicatorRegistry {
    factories: HashMap<String, IndicatorFactory>,
}

impl IndicatorRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Registriert eine Indikator-Factory
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&IndicatorParams) -> Result<Arc<dyn Indicator>, IndicatorError> + Send + Sync + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// Erstellt einen Indikator aus IndicatorSpec
    pub fn create(&self, spec: &IndicatorSpec) -> Result<Arc<dyn Indicator>, IndicatorError> {
        let factory = self.factories.get(&spec.name)
            .ok_or_else(|| IndicatorError::UnknownIndicator(spec.name.clone()))?;
        factory(&spec.params)
    }

    /// Erstellt Standard-Registry mit allen MRZ-Indikatoren
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        
        // EMA
        registry.register("EMA", |params| {
            match params {
                IndicatorParams::Period(period) => Ok(Arc::new(crate::impl_::ema::EMA { period: *period })),
                _ => Err(IndicatorError::InvalidParams("EMA requires Period".into())),
            }
        });
        
        // SMA
        registry.register("SMA", |params| {
            match params {
                IndicatorParams::Period(period) => Ok(Arc::new(crate::impl_::sma::SMA { period: *period })),
                _ => Err(IndicatorError::InvalidParams("SMA requires Period".into())),
            }
        });
        
        // ATR
        registry.register("ATR", |params| {
            match params {
                IndicatorParams::Period(period) => Ok(Arc::new(crate::impl_::atr::ATR { period: *period })),
                _ => Err(IndicatorError::InvalidParams("ATR requires Period".into())),
            }
        });
        
        // Z-Score
        registry.register("Z_SCORE", |params| {
            match params {
                IndicatorParams::Period(period) => Ok(Arc::new(crate::impl_::z_score::ZScore { window: *period })),
                _ => Err(IndicatorError::InvalidParams("Z_SCORE requires Period".into())),
            }
        });
        
        // Kalman Z-Score
        registry.register("KALMAN_Z", |params| {
            match params {
                IndicatorParams::Kalman { window, r_x1000, q_x1000 } => {
                    Ok(Arc::new(crate::impl_::kalman_zscore::KalmanZScore {
                        window: *window,
                        r: *r_x1000 as f64 / 1000.0,
                        q: *q_x1000 as f64 / 1000.0,
                    }))
                }
                _ => Err(IndicatorError::InvalidParams("KALMAN_Z requires Kalman params".into())),
            }
        });
        
        // GARCH Volatility
        registry.register("GARCH_VOL", |params| {
            match params {
                IndicatorParams::Garch { alpha_x1000, beta_x1000, omega_x1000000 } => {
                    Ok(Arc::new(crate::impl_::garch_volatility::GarchVolatility {
                        alpha: *alpha_x1000 as f64 / 1000.0,
                        beta: *beta_x1000 as f64 / 1000.0,
                        omega: *omega_x1000000 as f64 / 1_000_000.0,
                        use_log_returns: true,
                        scale: 100.0,
                        min_periods: 20,
                        sigma_floor: 0.0001,
                    }))
                }
                _ => Err(IndicatorError::InvalidParams("GARCH_VOL requires Garch params".into())),
            }
        });
        
        registry
    }
}
```

## Multi-Output Policy (Multi-Output-Indikatoren)

Einige Indikatoren (z.B. Bollinger Bands) liefern mehrere Outputs. Diese werden über dedizierte Structs und Cache-Keys behandelt:

### Multi-Output Trait Extension
```rust
/// Trait für Multi-Output-Indikatoren wie Bollinger Bands
pub trait MultiOutputIndicator: Send + Sync {
    /// Typ der Output-Struktur
    type Output;
    
    /// Berechnet alle Outputs
    fn compute_all(&self, candles: &[Candle]) -> Self::Output;
    
    /// Name des Indikators
    fn name(&self) -> &str;
    
    /// Minimale Bars für validen Output
    fn warmup_periods(&self) -> usize;
    
    /// Liste der Output-Namen (für Cache-Keys)
    fn output_names(&self) -> &[&str];
}
```

### Multi-Output Cache Extension
```rust
impl IndicatorCache {
    /// Cached Multi-Output Indikator mit separaten Keys pro Output
    pub fn get_or_compute_multi<T: MultiOutputIndicator>(
        &mut self,
        base_spec: &IndicatorSpec,
        candles: &[Candle],
        indicator: &T,
    ) -> MultiOutputResult
    where
        T::Output: IntoMultiVecs,
    {
        // Generiere Cache-Keys für jeden Output
        let output_names = indicator.output_names();
        let first_key = make_multi_key(base_spec, output_names[0]);
        
        // Prüfe ob bereits cached
        if !self.cache.contains_key(&first_key) {
            let result = indicator.compute_all(candles);
            let vecs = result.into_vecs();
            
            // Cache jeden Output separat
            for (name, vec) in output_names.iter().zip(vecs.into_iter()) {
                let key = make_multi_key(base_spec, name);
                self.cache.insert(key, vec);
            }
        }
        
        // Sammle alle Outputs
        MultiOutputResult {
            outputs: output_names.iter()
                .map(|name| (name.to_string(), self.cache.get(&make_multi_key(base_spec, name)).unwrap().clone()))
                .collect()
        }
    }
}

fn make_multi_key(base: &IndicatorSpec, output_name: &str) -> IndicatorSpec {
    IndicatorSpec {
        name: format!("{}_{}", base.name, output_name),
        params: base.params.clone(),
    }
}

pub struct MultiOutputResult {
    pub outputs: HashMap<String, Vec<f64>>,
}

pub trait IntoMultiVecs {
    fn into_vecs(self) -> Vec<Vec<f64>>;
}
```

### Bollinger Bands als Multi-Output
```rust
impl MultiOutputIndicator for BollingerBands {
    type Output = BollingerResult;
    
    fn compute_all(&self, candles: &[Candle]) -> Self::Output {
        // Existierende Implementierung (siehe oben)
        self.compute_bands(candles)
    }
    
    fn name(&self) -> &str { "BOLLINGER" }
    fn warmup_periods(&self) -> usize { self.period }
    fn output_names(&self) -> &[&str] { &["upper", "middle", "lower"] }
}

impl IntoMultiVecs for BollingerResult {
    fn into_vecs(self) -> Vec<Vec<f64>> {
        vec![self.upper, self.middle, self.lower]
    }
}
```

### Usage Pattern in Strategy
```rust
// Im BarContext oder Strategy:
let bb_spec = IndicatorSpec {
    name: "BOLLINGER".to_string(),
    params: IndicatorParams::Bollinger { period: 20, std_factor_x100: 200 },
};

// Bollinger Bands abrufen (alle drei auf einmal)
let bb = cache.get_or_compute_multi(&bb_spec, candles, &bollinger);
let upper = bb.outputs.get("upper").unwrap();
let middle = bb.outputs.get("middle").unwrap();
let lower = bb.outputs.get("lower").unwrap();

// Oder einzeln per Composite-Key
let bb_upper = cache.get(&make_multi_key(&bb_spec, "upper"), idx);
```

## Tests
1. Jeder Indikator gegen bekannte Referenzwerte
2. EMA/SMA Konvergenz bei konstanten Inputs
3. ATR True Range Berechnung
4. Bollinger Bands Symmetrie
5. Z-Score bei normalverteilten Daten
6. Kalman Filter Glättungseigenschaften
7. GARCH Varianz-Persistenz

## V1-Python-Parität
KRITISCH: Implementierung muss numerisch identisch zu V1-Python sein:
- ATR: Wilder-Smoothing, nicht SMA
- Bollinger: Sample Std (n-1) (pandas default)
- Kalman: Identische Initialisierung
- GARCH: Identische Return-Definition

## Referenz-Dokumente
- OMEGA_V2_INDICATOR_CACHE__PLAN.md (Vollständig)
- OMEGA_V2_STRATEGIES_PLAN.md (Abschnitt 6 - Indikatoren)
```

### Akzeptanzkriterien W2

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Alle MRZ-Indikatoren | EMA, ATR, BB, Z, Kalman, GARCH |
| A2 | Numerische V1-Python-Parität | Vergleich gegen Python-Referenz |
| A3 | Cache funktioniert | Kein doppeltes Berechnen |
| A4 | NaN für Warmup | Erste N Werte sind NaN |
| A5 | Vektorisiert | Keine Per-Bar Calls |

### Referenzen
- [OMEGA_V2_INDICATOR_CACHE__PLAN.md](OMEGA_V2_INDICATOR_CACHE__PLAN.md) - Vollständig
- [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) - Abschnitt 6

---

## W3: Execution Core

### Beschreibung
Implementiert `execution` und `portfolio` Crates für Order-Ausführung und Portfolio-Management.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Execution Core (Wave 3)

## Kontext
Du implementierst das Execution-Layer für ein Rust-natives Backtesting-System. Wave 0 (types) ist bereits implementiert.

## Ziel
Erstelle die `execution` und `portfolio` Crates für Order-Fill-Simulation, Slippage/Fees und Portfolio-State-Management.

## Verzeichnisstruktur

```
rust_core/crates/
├── execution/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── engine.rs      # ExecutionEngine
│       ├── slippage.rs    # SlippageModel Trait + Impl
│       ├── fees.rs        # FeeModel Trait + Impl
│       ├── fill.rs        # Fill-Logik
│       ├── costs.rs       # YAML Costs Loading
│       └── error.rs
│
└── portfolio/
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── portfolio.rs   # Portfolio-Hauptstruktur
        ├── position_manager.rs
        ├── equity.rs
        ├── stops.rs       # SL/TP Prüfung
        └── error.rs
```

## execution/Cargo.toml
```toml
[package]
name = "omega_execution"
version.workspace = true
edition.workspace = true

[dependencies]
omega_types = { workspace = true }
serde = { workspace = true }
serde_yaml = "0.9"
rand = "0.8"
rand_chacha = "0.3"
thiserror = { workspace = true }
```

## Slippage Model (slippage.rs)
```rust
use omega_types::Direction;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

pub trait SlippageModel: Send + Sync {
    /// Berechnet Slippage für einen Fill
    /// Returns: Slippage in Preiseinheiten (positiv = adverse)
    fn calculate(&self, price: f64, direction: Direction, rng: &mut ChaCha8Rng) -> f64;
}

/// Fixer Slippage in Pips
pub struct FixedSlippage {
    pub pips: f64,
    pub pip_size: f64,
}

impl SlippageModel for FixedSlippage {
    fn calculate(&self, _price: f64, direction: Direction, _rng: &mut ChaCha8Rng) -> f64 {
        let base = self.pips * self.pip_size;
        match direction {
            Direction::Long => base,   // Teurer kaufen
            Direction::Short => -base, // Günstiger verkaufen
        }
    }
}

/// Volatilitätsbasierter Slippage mit Randomness
pub struct VolatilitySlippage {
    pub base_pips: f64,
    pub pip_size: f64,
    pub jitter_factor: f64,  // 0.0 - 1.0
}

impl SlippageModel for VolatilitySlippage {
    fn calculate(&self, _price: f64, direction: Direction, rng: &mut ChaCha8Rng) -> f64 {
        let jitter: f64 = rng.gen_range(-self.jitter_factor..self.jitter_factor);
        let pips = self.base_pips * (1.0 + jitter);
        let base = pips * self.pip_size;
        match direction {
            Direction::Long => base,
            Direction::Short => -base,
        }
    }
}
```

## Fee Model (fees.rs)
```rust
pub trait FeeModel: Send + Sync {
    /// Berechnet Fee für eine Order
    fn calculate(&self, size: f64, price: f64) -> f64;
}

pub struct PercentageFee {
    pub percent: f64,  // z.B. 0.001 für 0.1%
}

impl FeeModel for PercentageFee {
    fn calculate(&self, size: f64, price: f64) -> f64 {
        size * price * self.percent
    }
}

pub struct FixedFee {
    pub fee_per_lot: f64,
}

impl FeeModel for FixedFee {
    fn calculate(&self, size: f64, _price: f64) -> f64 {
        size * self.fee_per_lot
    }
}
```

## Fill Logic (fill.rs)
```rust
use omega_types::{Candle, Direction, OrderType};

/// Ergebnis eines Fill-Versuchs
#[derive(Debug)]
pub struct FillResult {
    pub filled: bool,
    pub fill_price: f64,
    pub slippage_applied: f64,
}

/// Berechnet Fill-Preis für Market Order
pub fn market_fill(
    signal_price: f64,
    direction: Direction,
    slippage: f64,
) -> FillResult {
    let fill_price = match direction {
        Direction::Long => signal_price + slippage,
        Direction::Short => signal_price - slippage,
    };
    FillResult {
        filled: true,
        fill_price,
        slippage_applied: slippage,
    }
}

/// Prüft ob Limit/Stop Order triggered und berechnet Gap-aware Fill
pub fn pending_fill(
    order_type: OrderType,
    entry_price: f64,
    direction: Direction,
    bid: &Candle,
    ask: &Candle,
    slippage: f64,
) -> Option<FillResult> {
    let triggered = match (order_type, direction) {
        (OrderType::Limit, Direction::Long) => ask.low <= entry_price,
        (OrderType::Limit, Direction::Short) => bid.high >= entry_price,
        (OrderType::Stop, Direction::Long) => ask.high >= entry_price,
        (OrderType::Stop, Direction::Short) => bid.low <= entry_price,
        _ => return None,
    };

    if !triggered {
        return None;
    }

    // Gap-aware Fill: schlechterer von entry_price und Open
    let base_fill = match direction {
        Direction::Long => entry_price.max(ask.open),
        Direction::Short => entry_price.min(bid.open),
    };

    let fill_price = match direction {
        Direction::Long => base_fill + slippage,
        Direction::Short => base_fill - slippage,
    };

    Some(FillResult {
        filled: true,
        fill_price,
        slippage_applied: slippage,
    })
}
```

## Order/Position State Machine (execution/state.rs)

> **Normativ:** Jede Order und Position durchläuft definierte Zustände.
> State-Übergänge sind deterministisch und rückwärts-inkompatibel (kein Rollback).

```rust
use serde::{Deserialize, Serialize};

/// Order-Zustand: Pending Orders (Limit/Stop Orders awaiting trigger)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderState {
    /// Order wurde erstellt, wartet auf Trigger-Bedingung
    Pending,
    /// Order wurde getriggert, wartet auf Fill (next bar oder sofort)
    Triggered,
    /// Order wurde gefüllt → wird zu Position
    Filled,
    /// Order wurde abgelehnt (Margin, Risk, Session, etc.)
    Rejected,
    /// Order wurde vom User/System storniert
    Cancelled,
    /// Order ist abgelaufen (GoodTillDate erreicht)
    Expired,
}

impl OrderState {
    /// Erlaubte Übergänge von aktuellem Zustand
    pub fn allowed_transitions(&self) -> &[OrderState] {
        match self {
            OrderState::Pending => &[
                OrderState::Triggered,
                OrderState::Cancelled,
                OrderState::Expired,
            ],
            OrderState::Triggered => &[
                OrderState::Filled,
                OrderState::Rejected,
            ],
            OrderState::Filled => &[],     // Terminal
            OrderState::Rejected => &[],   // Terminal
            OrderState::Cancelled => &[],  // Terminal
            OrderState::Expired => &[],    // Terminal
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, 
            OrderState::Filled | 
            OrderState::Rejected | 
            OrderState::Cancelled | 
            OrderState::Expired
        )
    }
}

/// Position-Zustand: Offene Positionen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionState {
    /// Position ist offen (Entry gefüllt)
    Open,
    /// SL/TP wurde modifiziert (Trailing, Manual)
    Modified,
    /// Position wurde geschlossen (SL/TP/Manual/Timeout)
    Closed,
}

impl PositionState {
    pub fn allowed_transitions(&self) -> &[PositionState] {
        match self {
            PositionState::Open => &[PositionState::Modified, PositionState::Closed],
            PositionState::Modified => &[PositionState::Modified, PositionState::Closed],
            PositionState::Closed => &[],  // Terminal
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, PositionState::Closed)
    }
}

/// State Transition mit Audit-Trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition<S: Clone> {
    pub from: S,
    pub to: S,
    pub timestamp_ns: i64,
    pub reason: String,
}
```

## PendingBook (execution/pending_book.rs)

> **Normativ:** PendingBook verwaltet alle Pending Orders (Limit/Stop).
> Trigger-Reihenfolge ist deterministisch: Creation-Time → Order-ID.

```rust
use omega_types::{Candle, Direction, OrderType};
use crate::state::OrderState;
use std::collections::BTreeMap;

/// Pending Order im Book
#[derive(Debug, Clone)]
pub struct PendingOrder {
    pub id: u64,
    pub order_type: OrderType,
    pub direction: Direction,
    pub entry_price: f64,
    pub size: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub state: OrderState,
    pub created_at_ns: i64,
    pub good_till_ns: Option<i64>,  // GTC = None
    pub scenario_id: u8,
    pub meta: serde_json::Value,
}

/// Trigger-Event für eine Order
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    pub order_id: u64,
    pub triggered_at_ns: i64,
    pub trigger_price: f64,
}

/// PendingBook: Verwaltet alle Pending Orders
/// 
/// DETERMINISMUS-REGELN (normativ):
/// 1. Orders werden nach (created_at_ns, id) sortiert → FIFO + Tie-Break
/// 2. Trigger-Prüfung erfolgt in dieser Reihenfolge
/// 3. Orders werden am Candle-Close platziert → Trigger erst ab next_bar
/// 4. Wenn Trigger in Bar t: Fill in derselben Bar t (Same-Bar Entry möglich)
///    → SL/TP können in der Entry-Candle ausgelöst werden
#[derive(Debug, Default)]
pub struct PendingBook {
    /// Orders sortiert nach (created_at_ns, order_id) für deterministische Iteration
    orders: BTreeMap<(i64, u64), PendingOrder>,
    next_id: u64,
}

impl PendingBook {
    pub fn new() -> Self {
        Self {
            orders: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Fügt neue Pending Order hinzu
    pub fn add_order(&mut self, mut order: PendingOrder) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        order.id = id;
        order.state = OrderState::Pending;
        
        let key = (order.created_at_ns, id);
        self.orders.insert(key, order);
        id
    }

    /// Prüft alle Orders auf Trigger und gibt Events zurück (deterministisch sortiert)
    /// 
    /// WICHTIG: Pending Orders werden am Candle-Close platziert.
    /// Trigger daher NICHT im selben Bar wie Erstellung (next_bar Regel).
    pub fn check_triggers(
        &mut self,
        bid: &Candle,
        ask: &Candle,
        current_bar_ns: i64,
    ) -> Vec<TriggerEvent> {
        let mut events = Vec::new();

        for ((created_ns, order_id), order) in self.orders.iter_mut() {
            // Skip: bereits getriggert oder terminal
            if order.state != OrderState::Pending {
                continue;
            }

            // next_bar Regel: Order muss VOR diesem Bar erstellt sein
            if *created_ns >= current_bar_ns {
                continue;
            }

            // Expiration Check
            if let Some(gtd) = order.good_till_ns {
                if current_bar_ns > gtd {
                    order.state = OrderState::Expired;
                    continue;
                }
            }

            // Trigger-Bedingung prüfen
            let triggered = match (order.order_type, order.direction) {
                // Limit Long: Ask fällt auf/unter Entry
                (OrderType::Limit, Direction::Long) => ask.low <= order.entry_price,
                // Limit Short: Bid steigt auf/über Entry
                (OrderType::Limit, Direction::Short) => bid.high >= order.entry_price,
                // Stop Long: Ask steigt auf/über Entry
                (OrderType::Stop, Direction::Long) => ask.high >= order.entry_price,
                // Stop Short: Bid fällt auf/unter Entry
                (OrderType::Stop, Direction::Short) => bid.low <= order.entry_price,
                // Market: sofort (sollte nicht im PendingBook sein)
                (OrderType::Market, _) => true,
            };

            if triggered {
                order.state = OrderState::Triggered;
                events.push(TriggerEvent {
                    order_id: *order_id,
                    triggered_at_ns: current_bar_ns,
                    trigger_price: order.entry_price,
                });
            }
        }

        events
    }

    /// Markiert Order als gefüllt und entfernt aus Book
    pub fn mark_filled(&mut self, order_id: u64) -> Option<PendingOrder> {
        let key = self.orders.iter()
            .find(|((_, id), _)| *id == order_id)
            .map(|(k, _)| *k)?;
        
        let mut order = self.orders.remove(&key)?;
        order.state = OrderState::Filled;
        Some(order)
    }

    /// Storniert eine Order
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        for ((_, id), order) in self.orders.iter_mut() {
            if *id == order_id && order.state == OrderState::Pending {
                order.state = OrderState::Cancelled;
                return true;
            }
        }
        false
    }

    /// Alle getriggerten Orders (für Fill-Processing)
    pub fn get_triggered(&self) -> Vec<&PendingOrder> {
        self.orders.values()
            .filter(|o| o.state == OrderState::Triggered)
            .collect()
    }

    /// Bereinigt terminal Orders
    pub fn cleanup_terminal(&mut self) {
        self.orders.retain(|_, o| !o.state.is_terminal());
    }

    pub fn pending_count(&self) -> usize {
        self.orders.values()
            .filter(|o| o.state == OrderState::Pending)
            .count()
    }
}
```

## Deterministic Trigger Order Policy

> **Normativ:** Die Reihenfolge der Verarbeitung ist determiniert und dokumentiert.

```rust
/// Deterministische Verarbeitungsreihenfolge für einen Bar
///
/// PHASE 1: Pending Order Triggers
///   - Sortiert nach: (created_at_ns, order_id) → FIFO + Tie-Break
///   - Alle Orders die Trigger-Bedingung erfüllen werden markiert
///   - Fill erfolgt mit pending_fill() (Gap-aware) in derselben Trigger-Candle
///
/// PHASE 2: Exit-Checks (SL/TP)
///   - SL hat IMMER Priorität über TP
///   - Begründung: Risikomanagement > Profit-Taking
///   - Bei gleichzeitigem SL+TP Hit: SL gewinnt
///   - in_entry_candle Speziallogik: TP nur wenn Close beyond TP
///
/// PHASE 3: Trade-Management
///   - Rule-basierte Actions (z.B. Timeout)
///   - Stop/TP-Updates gelten ab next_bar
///
/// PHASE 4: Equity-Update
///   - Equity/Balance pro Bar aktualisieren
///
/// PHASE 5: Neue Signale
///   - Strategy.on_bar() wird aufgerufen
///   - Neue Market Orders: sofort Fill
///   - Neue Pending Orders: in PendingBook (Placement am Candle-Close, Trigger ab next_bar)
pub struct TriggerOrderPolicy;

impl TriggerOrderPolicy {
    /// Dokumentiert die Verarbeitungsreihenfolge für Audit
    pub const PROCESSING_ORDER: &'static [&'static str] = &[
        "1. Pending-Triggers (FIFO nach created_at_ns, order_id)",
        "2. Exit-Checks (SL/TP, SL > TP Priorität, in_entry_candle Regel)",
        "3. Trade-Management (Rule-Actions, Stop/TP-Updates ab next_bar)",
        "4. Equity-Update (Equity/Balance pro Bar)",
        "5. Strategy-Signale (neue Orders erstellen)",
    ];

    /// Bei gleichzeitigem SL und TP Hit: SL gewinnt
    pub const SL_OVER_TP_PRIORITY: bool = true;

    /// Pending Orders triggern erst ab next_bar nach Erstellung
    pub const PENDING_NEXT_BAR_RULE: bool = true;

    /// Deterministische Sortierung für Order-Verarbeitung
    pub fn sort_orders_deterministic(orders: &mut [&PendingOrder]) {
        orders.sort_by_key(|o| (o.created_at_ns, o.id));
    }
}
```

## SL/TP Check (portfolio/stops.rs)
```rust
use omega_types::{Candle, Direction, Position, ExitReason};

/// pip_buffer für SL/TP Checks
pub const DEFAULT_PIP_BUFFER_FACTOR: f64 = 0.5;

#[derive(Debug)]
pub struct StopCheckResult {
    pub triggered: bool,
    pub reason: ExitReason,
    pub exit_price: f64,
}

/// Prüft SL/TP für eine Position
/// WICHTIG: SL hat Priorität über TP im selben Candle
pub fn check_stops(
    position: &Position,
    bid: &Candle,
    ask: &Candle,
    pip_size: f64,
    pip_buffer_factor: f64,
    in_entry_candle: bool,
) -> Option<StopCheckResult> {
    let pip_buffer = pip_size * pip_buffer_factor;

    let (sl_hit, tp_hit) = match position.direction {
        Direction::Long => {
            let sl = bid.low <= position.stop_loss + pip_buffer;
            let tp = bid.high >= position.take_profit - pip_buffer;
            (sl, tp)
        }
        Direction::Short => {
            let sl = ask.high >= position.stop_loss - pip_buffer;
            let tp = ask.low <= position.take_profit + pip_buffer;
            (sl, tp)
        }
    };

    // SL hat Priorität
    if sl_hit {
        let exit_price = match position.direction {
            Direction::Long => position.stop_loss,
            Direction::Short => position.stop_loss,
        };
        return Some(StopCheckResult {
            triggered: true,
            reason: ExitReason::StopLoss,
            exit_price,
        });
    }

    // TP Check (mit in_entry_candle Speziallogik)
    if tp_hit {
        // In Entry-Candle: TP nur wenn Close "jenseits" des TP
        if in_entry_candle {
            let tp_valid = match position.direction {
                Direction::Long => bid.close > position.take_profit,
                Direction::Short => ask.close < position.take_profit,
            };
            if !tp_valid {
                return None;
            }
        }

        return Some(StopCheckResult {
            triggered: true,
            reason: ExitReason::TakeProfit,
            exit_price: position.take_profit,
        });
    }

    None
}
```

## Portfolio (portfolio/portfolio.rs)
```rust
use omega_types::{Position, Trade, Direction, Signal, ExitReason};
use crate::equity::EquityTracker;
use crate::error::PortfolioError;

pub struct Portfolio {
    pub cash: f64,
    pub positions: Vec<Position>,
    pub closed_trades: Vec<Trade>,
    pub equity_tracker: EquityTracker,
    next_position_id: u64,
    pub max_positions: usize,
}

impl Portfolio {
    pub fn new(initial_balance: f64, max_positions: usize) -> Self {
        Self {
            cash: initial_balance,
            positions: Vec::new(),
            closed_trades: Vec::new(),
            equity_tracker: EquityTracker::new(initial_balance),
            next_position_id: 1,
            max_positions,
        }
    }

    pub fn can_open_position(&self) -> bool {
        self.positions.len() < self.max_positions
    }

    pub fn open_position(
        &mut self,
        signal: &Signal,
        fill_price: f64,
        size: f64,
        entry_time_ns: i64,
        entry_fee: f64,
    ) -> Result<u64, PortfolioError> {
        if !self.can_open_position() {
            return Err(PortfolioError::MaxPositionsReached);
        }

        let id = self.next_position_id;
        self.next_position_id += 1;

        let position = Position {
            id,
            direction: signal.direction.clone(),
            entry_time_ns,
            entry_price: fill_price,
            size,
            stop_loss: signal.stop_loss,
            take_profit: signal.take_profit,
            scenario_id: signal.scenario_id,
            meta: signal.meta.clone(),
        };

        self.positions.push(position);
        self.cash -= entry_fee;

        Ok(id)
    }

    pub fn close_position(
        &mut self,
        position_id: u64,
        exit_price: f64,
        exit_time_ns: i64,
        reason: ExitReason,
        exit_fee: f64,
        symbol: &str,
    ) -> Option<Trade> {
        let pos_idx = self.positions.iter().position(|p| p.id == position_id)?;
        let position = self.positions.remove(pos_idx);

        let pnl = match position.direction {
            Direction::Long => (exit_price - position.entry_price) * position.size,
            Direction::Short => (position.entry_price - exit_price) * position.size,
        };

        let risk = (position.entry_price - position.stop_loss).abs() * position.size;
        let r_multiple = if risk > 0.0 { pnl / risk } else { 0.0 };

        let trade = Trade {
            entry_time_ns: position.entry_time_ns,
            exit_time_ns,
            direction: position.direction,
            symbol: symbol.to_string(),
            entry_price: position.entry_price,
            exit_price,
            stop_loss: position.stop_loss,
            take_profit: position.take_profit,
            size: position.size,
            result: pnl,
            r_multiple,
            reason,
            scenario_id: position.scenario_id,
            meta: position.meta,
        };

        self.cash += pnl - exit_fee;
        self.closed_trades.push(trade.clone());

        Some(trade)
    }

    pub fn update_equity(&mut self, timestamp_ns: i64, current_price: f64) {
        let unrealized_pnl: f64 = self.positions.iter()
            .map(|p| match p.direction {
                Direction::Long => (current_price - p.entry_price) * p.size,
                Direction::Short => (p.entry_price - current_price) * p.size,
            })
            .sum();

        let equity = self.cash + unrealized_pnl;
        self.equity_tracker.update(timestamp_ns, equity, self.cash);
    }
}
```

## Tests
1. Market Fill mit Slippage
2. Limit/Stop Trigger-Bedingungen
3. Gap-aware Fill (Entry schlechter als Order)
4. SL-Priorität über TP
5. in_entry_candle TP-Speziallogik
6. Portfolio-Balance-Konsistenz
7. R-Multiple Berechnung

## Referenz-Dokumente
- OMEGA_V2_EXECUTION_MODEL_PLAN.md (Vollständig)
- OMEGA_V2_MODULE_STRUCTURE_PLAN.md (Abschnitt 3.4, 3.5)
```

### Akzeptanzkriterien W3

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Slippage-Modelle | Fixed + Volatility |
| A2 | Gap-aware Fills | Entry <= Open (adverse) |
| A3 | SL > TP Priorität | Test mit both-hit |
| A4 | in_entry_candle | TP nur wenn Close beyond |
| A5 | Portfolio konsistent | cash + positions = equity |
| A6 | Deterministische RNG | ChaCha8 mit Seed |

### Referenzen
- [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) - Vollständig
- [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) - Abschnitt 3.4, 3.5

---

## W4: Strategy Layer

### Beschreibung
Implementiert `strategy` und `trade_mgmt` Crates mit Strategy-Trait und Mean Reversion Z-Score.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Strategy Layer (Wave 4)

## Kontext
Du implementierst das Strategy-Layer für ein Rust-natives Backtesting-System. Waves 0-3 sind bereits implementiert.

## Ziel
Erstelle die `strategy` und `trade_mgmt` Crates mit Strategy-Trait, BarContext und vollständiger Mean Reversion Z-Score Implementierung (Szenarien 1-6).

## Verzeichnisstruktur

```
rust_core/crates/
├── strategy/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── traits.rs      # Strategy Trait
│       ├── context.rs     # BarContext
│       ├── registry.rs    # StrategyRegistry
│       ├── error.rs
│       └── impl/
│           ├── mod.rs
│           └── mean_reversion_z_score.rs
│
└── trade_mgmt/
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── engine.rs      # TradeManager
        ├── rules.rs       # Rule Traits
        ├── actions.rs     # Action Types
        └── error.rs
```

## Strategy Trait (strategy/traits.rs)
```rust
use omega_types::Signal;
use crate::context::BarContext;

pub trait Strategy: Send + Sync {
    /// Verarbeitet einen Bar und gibt optional ein Signal zurück
    fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal>;

    /// Name der Strategie für Registry
    fn name(&self) -> &str;

    /// Benötigte Indikatoren (für Pre-Computation)
    fn required_indicators(&self) -> Vec<IndicatorRequirement>;

    /// Benötigte HTF Timeframes
    fn required_htf_timeframes(&self) -> Vec<String> {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct IndicatorRequirement {
    pub name: String,
    pub timeframe: Option<String>,  // None = primary
    pub params: serde_json::Value,
}
```

## BarContext (strategy/context.rs)
```rust
use omega_types::Candle;
use omega_indicators::IndicatorCache;

/// Read-only Snapshot für Strategy.on_bar()
pub struct BarContext<'a> {
    pub idx: usize,
    pub timestamp_ns: i64,
    pub bid: &'a Candle,
    pub ask: &'a Candle,
    pub indicators: &'a IndicatorCache,
    pub htf_data: Option<&'a HtfContext<'a>>,
    pub session_open: bool,
    pub news_blocked: bool,
}

pub struct HtfContext<'a> {
    pub bid: &'a Candle,
    pub ask: &'a Candle,
    pub indicators: &'a IndicatorCache,
    pub idx: usize,
}

impl<'a> BarContext<'a> {
    /// Holt Indikator-Wert für aktuellen Index
    pub fn get_indicator(&self, name: &str, params: &serde_json::Value) -> Option<f64> {
        // Implementation...
    }

    /// Holt HTF Indikator (letzte abgeschlossene Bar)
    pub fn get_htf_indicator(&self, name: &str, params: &serde_json::Value) -> Option<f64> {
        // Implementation mit Lookahead-Prevention...
    }
}
```

## Mean Reversion Z-Score (strategy/impl/mean_reversion_z_score.rs)
```rust
use omega_types::{Signal, Direction, OrderType};
use crate::{Strategy, BarContext, IndicatorRequirement};

/// Mean Reversion Z-Score Strategy (MVP)
/// Implementiert alle 6 Szenarien
pub struct MeanReversionZScore {
    pub params: MrzParams,
    state: MrzState,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MrzParams {
    // Core
    pub ema_length: usize,
    pub atr_length: usize,
    pub atr_mult: f64,
    pub b_b_length: usize,
    pub std_factor: f64,
    pub window_length: usize,
    pub z_score_long: f64,   // negative
    pub z_score_short: f64,  // positive
    pub kalman_r: f64,
    pub kalman_q: f64,

    // HTF
    pub htf_tf: String,
    pub htf_ema: usize,
    pub htf_filter: HtfFilter,

    // GARCH (Scenario 4/5)
    pub garch_alpha: f64,
    pub garch_beta: f64,
    pub garch_omega: f64,

    // Scenario 3
    pub tp_min_distance: f64,  // Price distance, not pips!

    // Scenario 5 (Vol Cluster)
    pub intraday_vol_cluster_window: usize,
    pub intraday_vol_cluster_k: usize,
    pub intraday_vol_allowed: Vec<String>,
    pub cluster_hysteresis_bars: usize,

    // Scenario 6 (Multi-TF)
    pub scenario6_mode: Scenario6Mode,
    pub scenario6_timeframes: Vec<String>,
    pub scenario6_params: serde_json::Value,

    // Gating
    pub direction_filter: DirectionFilter,
    pub enabled_scenarios: Vec<u8>,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HtfFilter {
    #[default]
    Both,
    Above,
    Below,
    None,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DirectionFilter {
    #[default]
    Both,
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Scenario6Mode {
    #[default]
    All,
    Any,
}

impl Strategy for MeanReversionZScore {
    fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal> {
        // Entry-Gates
        if !ctx.session_open || ctx.news_blocked {
            return None;
        }

        // Try each enabled scenario
        for scenario_id in &self.params.enabled_scenarios {
            if let Some(signal) = self.try_scenario(*scenario_id, ctx) {
                return Some(signal);
            }
        }

        None
    }

    fn name(&self) -> &str {
        "mean_reversion_z_score"
    }

    fn required_indicators(&self) -> Vec<IndicatorRequirement> {
        vec![
            IndicatorRequirement {
                name: "EMA".to_string(),
                timeframe: None,
                params: json!({"period": self.params.ema_length}),
            },
            IndicatorRequirement {
                name: "ATR".to_string(),
                timeframe: None,
                params: json!({"period": self.params.atr_length}),
            },
            // ... weitere Indikatoren
        ]
    }
}

impl MeanReversionZScore {
    fn try_scenario(&mut self, scenario_id: u8, ctx: &BarContext) -> Option<Signal> {
        match scenario_id {
            1 => self.scenario_1(ctx),
            2 => self.scenario_2(ctx),
            3 => self.scenario_3(ctx),
            4 => self.scenario_4(ctx),
            5 => self.scenario_5(ctx),
            6 => self.scenario_6(ctx),
            _ => None,
        }
    }

    /// Scenario 1: Z-Score + EMA Take Profit
    fn scenario_1(&self, ctx: &BarContext) -> Option<Signal> {
        let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": self.params.window_length}))?;
        let ema = ctx.get_indicator("EMA", &json!({"period": self.params.ema_length}))?;
        let atr = ctx.get_indicator("ATR", &json!({"period": self.params.atr_length}))?;

        // Long Entry
        if self.params.direction_filter != DirectionFilter::Short
           && zscore <= self.params.z_score_long {
            let sl = ctx.bid.low - self.params.atr_mult * atr;
            let tp = ema;
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 1,
                tags: vec!["scenario1".to_string()],
                meta: json!({"zscore": zscore}),
            });
        }

        // Short Entry
        if self.params.direction_filter != DirectionFilter::Long
           && zscore >= self.params.z_score_short {
            let sl = ctx.ask.high + self.params.atr_mult * atr;
            let tp = ema;
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 1,
                tags: vec!["scenario1".to_string()],
                meta: json!({"zscore": zscore}),
            });
        }

        None
    }

    /// Scenario 2: Kalman-Z + Bollinger, TP = BB-Mid
    fn scenario_2(&self, ctx: &BarContext) -> Option<Signal> {
        // HTF Filter Check
        if !self.check_htf_bias(ctx) {
            return None;
        }

        let kalman_z = ctx.get_indicator("KALMAN_Z", &json!({
            "window": self.params.window_length,
            "r": self.params.kalman_r,
            "q": self.params.kalman_q
        }))?;
        let bb_lower = ctx.get_indicator("BB_LOWER", &json!({
            "period": self.params.b_b_length,
            "std_factor": self.params.std_factor
        }))?;
        let bb_upper = ctx.get_indicator("BB_UPPER", &json!({
            "period": self.params.b_b_length,
            "std_factor": self.params.std_factor
        }))?;
        let bb_mid = ctx.get_indicator("BB_MID", &json!({
            "period": self.params.b_b_length
        }))?;
        let atr = ctx.get_indicator("ATR", &json!({"period": self.params.atr_length}))?;

        let close = ctx.bid.close;

        // Long
        if self.params.direction_filter != DirectionFilter::Short
           && kalman_z <= self.params.z_score_long
           && close <= bb_lower {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 2,
                tags: vec!["scenario2".to_string(), "kalman".to_string()],
                meta: json!({"kalman_z": kalman_z, "bb_lower": bb_lower}),
            });
        }

        // Short
        if self.params.direction_filter != DirectionFilter::Long
           && kalman_z >= self.params.z_score_short
           && close >= bb_upper {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.ask.high + self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 2,
                tags: vec!["scenario2".to_string(), "kalman".to_string()],
                meta: json!({"kalman_z": kalman_z, "bb_upper": bb_upper}),
            });
        }

        None
    }

    /// Scenario 3: Limit Entry + Minimum TP Distance
    /// Pending-Order die erst ab next_bar getriggert werden kann
    fn scenario_3(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": self.params.window_length}))?;
        let atr = ctx.get_indicator("ATR", &json!({"period": self.params.atr_length}))?;
        let ema = ctx.get_indicator("EMA", &json!({"period": self.params.ema_length}))?;

        // Long mit Limit-Entry
        if self.params.direction_filter != DirectionFilter::Short
           && zscore <= self.params.z_score_long {
            let entry_price = ctx.bid.low;  // Limit unterhalb aktueller Preis
            let sl = entry_price - self.params.atr_mult * atr;
            let raw_tp = ema;
            
            // Minimum TP Distance Check (tp_min_distance ist Preis-Distanz!)
            let tp = if (raw_tp - entry_price).abs() < self.params.tp_min_distance {
                entry_price + self.params.tp_min_distance
            } else {
                raw_tp
            };

            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Limit { price: entry_price },
                entry_price,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 3,
                tags: vec!["scenario3".to_string(), "limit".to_string()],
                meta: json!({
                    "zscore": zscore,
                    "tp_adjusted": tp != raw_tp,
                    "raw_tp": raw_tp
                }),
            });
        }

        // Short mit Limit-Entry
        if self.params.direction_filter != DirectionFilter::Long
           && zscore >= self.params.z_score_short {
            let entry_price = ctx.ask.high;  // Limit oberhalb aktueller Preis
            let sl = entry_price + self.params.atr_mult * atr;
            let raw_tp = ema;
            
            // Minimum TP Distance Check
            let tp = if (entry_price - raw_tp).abs() < self.params.tp_min_distance {
                entry_price - self.params.tp_min_distance
            } else {
                raw_tp
            };

            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Limit { price: entry_price },
                entry_price,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 3,
                tags: vec!["scenario3".to_string(), "limit".to_string()],
                meta: json!({
                    "zscore": zscore,
                    "tp_adjusted": tp != raw_tp,
                    "raw_tp": raw_tp
                }),
            });
        }

        None
    }

    /// Scenario 4: Same-Bar SL/TP Tie → SL-Priorität
    /// GARCH-basierte Volatilität für dynamische SL/TP
    fn scenario_4(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": self.params.window_length}))?;
        let garch_vol = ctx.get_indicator("GARCH_VOL", &json!({
            "alpha": self.params.garch_alpha,
            "beta": self.params.garch_beta,
            "omega": self.params.garch_omega
        }))?;
        let atr = ctx.get_indicator("ATR", &json!({"period": self.params.atr_length}))?;
        let ema = ctx.get_indicator("EMA", &json!({"period": self.params.ema_length}))?;

        // Volatility-adjusted multiplier (GARCH scaling)
        let vol_mult = garch_vol.max(0.5).min(2.0);  // Clamp for safety
        let adjusted_atr = atr * vol_mult;

        // Long
        if self.params.direction_filter != DirectionFilter::Short
           && zscore <= self.params.z_score_long {
            let sl = ctx.bid.low - self.params.atr_mult * adjusted_atr;
            let tp = ema;
            
            // Bei Same-Bar Tie: SL hat Priorität (implizit durch ExecutionEngine)
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 4,
                tags: vec!["scenario4".to_string(), "garch".to_string()],
                meta: json!({
                    "zscore": zscore,
                    "garch_vol": garch_vol,
                    "vol_mult": vol_mult,
                    "adjusted_atr": adjusted_atr
                }),
            });
        }

        // Short
        if self.params.direction_filter != DirectionFilter::Long
           && zscore >= self.params.z_score_short {
            let sl = ctx.ask.high + self.params.atr_mult * adjusted_atr;
            let tp = ema;

            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: sl,
                take_profit: tp,
                size: None,
                scenario_id: 4,
                tags: vec!["scenario4".to_string(), "garch".to_string()],
                meta: json!({
                    "zscore": zscore,
                    "garch_vol": garch_vol,
                    "vol_mult": vol_mult,
                    "adjusted_atr": adjusted_atr
                }),
            });
        }

        None
    }

    /// Scenario 5: in_entry_candle Spezialfall + Intraday Vol Clustering
    /// Inkludiert Limit-TP Regel (kein TP-Fill in Entry-Bar)
    fn scenario_5(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        // Intraday Volatility Cluster Check
        if !self.check_vol_cluster(ctx) {
            return None;
        }

        let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": self.params.window_length}))?;
        let kalman_z = ctx.get_indicator("KALMAN_Z", &json!({
            "window": self.params.window_length,
            "r": self.params.kalman_r,
            "q": self.params.kalman_q
        }))?;
        let atr = ctx.get_indicator("ATR", &json!({"period": self.params.atr_length}))?;
        let bb_mid = ctx.get_indicator("BB_MID", &json!({
            "period": self.params.b_b_length
        }))?;

        // Combined Z-Score (regular + kalman)
        let combined_z = (zscore + kalman_z) / 2.0;

        // Long
        if self.params.direction_filter != DirectionFilter::Short
           && combined_z <= self.params.z_score_long {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 5,
                // in_entry_candle-Flag wird von ExecutionEngine basierend auf
                // Position-Entry-Bar gesetzt, nicht hier im Signal
                tags: vec![
                    "scenario5".to_string(),
                    "vol_cluster".to_string(),
                    "combined_z".to_string()
                ],
                meta: json!({
                    "zscore": zscore,
                    "kalman_z": kalman_z,
                    "combined_z": combined_z,
                    "vol_cluster": self.get_current_vol_cluster(ctx)
                }),
            });
        }

        // Short
        if self.params.direction_filter != DirectionFilter::Long
           && combined_z >= self.params.z_score_short {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.ask.high + self.params.atr_mult * atr,
                take_profit: bb_mid,
                size: None,
                scenario_id: 5,
                tags: vec![
                    "scenario5".to_string(),
                    "vol_cluster".to_string(),
                    "combined_z".to_string()
                ],
                meta: json!({
                    "zscore": zscore,
                    "kalman_z": kalman_z,
                    "combined_z": combined_z,
                    "vol_cluster": self.get_current_vol_cluster(ctx)
                }),
            });
        }

        None
    }

    /// Scenario 6: Multi-TF + Sessions/Warmup Mix
    /// All: Alle TFs müssen Signal geben
    /// Any: Mindestens ein TF gibt Signal
    fn scenario_6(&self, ctx: &BarContext) -> Option<Signal> {
        if !self.check_htf_bias(ctx) {
            return None;
        }

        // Multi-TF Agreement Check
        let tf_signals = self.check_multi_tf_signals(ctx);
        
        let has_agreement = match self.params.scenario6_mode {
            Scenario6Mode::All => tf_signals.iter().all(|s| s.is_some()),
            Scenario6Mode::Any => tf_signals.iter().any(|s| s.is_some()),
        };

        if !has_agreement {
            return None;
        }

        // Primary TF Signal
        let zscore = ctx.get_indicator("Z_SCORE", &json!({"window": self.params.window_length}))?;
        let atr = ctx.get_indicator("ATR", &json!({"period": self.params.atr_length}))?;
        let ema = ctx.get_indicator("EMA", &json!({"period": self.params.ema_length}))?;

        // Extra HTF Check (zusätzlich zum Standard-HTF)
        let extra_htf_ok = self.check_extra_htf_filter(ctx);
        if !extra_htf_ok {
            return None;
        }

        // Long
        if self.params.direction_filter != DirectionFilter::Short
           && zscore <= self.params.z_score_long {
            return Some(Signal {
                direction: Direction::Long,
                order_type: OrderType::Market,
                entry_price: ctx.ask.close,
                stop_loss: ctx.bid.low - self.params.atr_mult * atr,
                take_profit: ema,
                size: None,
                scenario_id: 6,
                tags: vec![
                    "scenario6".to_string(),
                    format!("multi_tf_{}", self.params.scenario6_mode.as_str()),
                ],
                meta: json!({
                    "zscore": zscore,
                    "mode": self.params.scenario6_mode.as_str(),
                    "tf_agreement": tf_signals.len(),
                    "extra_htf_ok": extra_htf_ok
                }),
            });
        }

        // Short
        if self.params.direction_filter != DirectionFilter::Long
           && zscore >= self.params.z_score_short {
            return Some(Signal {
                direction: Direction::Short,
                order_type: OrderType::Market,
                entry_price: ctx.bid.close,
                stop_loss: ctx.ask.high + self.params.atr_mult * atr,
                take_profit: ema,
                size: None,
                scenario_id: 6,
                tags: vec![
                    "scenario6".to_string(),
                    format!("multi_tf_{}", self.params.scenario6_mode.as_str()),
                ],
                meta: json!({
                    "zscore": zscore,
                    "mode": self.params.scenario6_mode.as_str(),
                    "tf_agreement": tf_signals.len(),
                    "extra_htf_ok": extra_htf_ok
                }),
            });
        }

        None
    }

    // ==================== Helper Methods ====================

    /// Prüft Intraday Volatility Cluster
    fn check_vol_cluster(&self, ctx: &BarContext) -> bool {
        let cluster = self.get_current_vol_cluster(ctx);
        match cluster {
            Some(c) => self.params.intraday_vol_allowed.contains(&c),
            None => true,  // Wenn kein Cluster bestimmt werden kann, passieren lassen
        }
    }

    fn get_current_vol_cluster(&self, ctx: &BarContext) -> Option<String> {
        // Vol-Cluster wird als Indikator berechnet
        // Gibt String zurück: "low", "medium", "high"
        ctx.get_indicator("VOL_CLUSTER", &json!({
            "window": self.params.intraday_vol_cluster_window,
            "k": self.params.intraday_vol_cluster_k
        })).map(|v| {
            match v as i32 {
                0 => "low".to_string(),
                1 => "medium".to_string(),
                _ => "high".to_string(),
            }
        })
    }

    /// Multi-TF Signal Check für Scenario 6
    fn check_multi_tf_signals(&self, ctx: &BarContext) -> Vec<Option<Direction>> {
        self.params.scenario6_timeframes.iter().map(|tf| {
            // Hole TF-spezifischen Z-Score
            let zscore = ctx.get_indicator(&format!("Z_SCORE_{}", tf), &self.params.scenario6_params)?;
            
            if zscore <= self.params.z_score_long {
                Some(Direction::Long)
            } else if zscore >= self.params.z_score_short {
                Some(Direction::Short)
            } else {
                None
            }
        }).collect()
    }

    /// Extra HTF Filter für Scenario 6
    fn check_extra_htf_filter(&self, ctx: &BarContext) -> bool {
        // Zusätzlicher HTF-Filter wenn konfiguriert
        // Wird über scenario6_params gesteuert
        if let Some(extra_tf) = self.params.scenario6_params.get("extra_htf_tf").and_then(|v| v.as_str()) {
            let extra_ema = self.params.scenario6_params.get("extra_htf_ema")
                .and_then(|v| v.as_u64())
                .unwrap_or(200) as usize;
            
            let htf_ema = ctx.get_htf_indicator(&format!("EMA_{}", extra_tf), &json!({"period": extra_ema}));
            let htf_price = ctx.htf_data.map(|h| h.bid.close);
            
            match (htf_ema, htf_price) {
                (Some(ema), Some(price)) => price > ema, // Vereinfacht: nur Above-Check
                _ => true,
            }
        } else {
            true
        }
    }

    fn check_htf_bias(&self, ctx: &BarContext) -> bool {
        if self.params.htf_filter == HtfFilter::None {
            return true;
        }

        let htf_ema = ctx.get_htf_indicator("EMA", &json!({"period": self.params.htf_ema}));
        let htf_price = ctx.htf_data.map(|h| h.bid.close);

        match (htf_ema, htf_price) {
            (Some(ema), Some(price)) => match self.params.htf_filter {
                HtfFilter::Above => price > ema,
                HtfFilter::Below => price < ema,
                HtfFilter::Both => true,
                HtfFilter::None => true,
            },
            _ => true,  // Wenn HTF-Daten fehlen, passieren lassen
        }
    }
}
```

## Trade Manager (trade_mgmt/engine.rs)
```rust
use omega_types::{Position, ExitReason};
use crate::rules::{Rule, RuleSet};
use crate::actions::Action;

pub struct TradeManager {
    rules: RuleSet,
}

impl TradeManager {
    pub fn new(rules: RuleSet) -> Self {
        Self { rules }
    }

    /// Evaluiert Rules für alle Positionen
    /// Gibt Actions zurück, die im nächsten Bar angewendet werden
    pub fn evaluate(&self, positions: &[Position], timestamp_ns: i64) -> Vec<Action> {
        let mut actions = Vec::new();

        for position in positions {
            for rule in self.rules.iter() {
                if let Some(action) = rule.evaluate(position, timestamp_ns) {
                    actions.push(action);
                    break;  // Eine Action pro Position pro Bar
                }
            }
        }

        actions
    }
}
```

## MaxHoldingTime Rule (trade_mgmt/rules.rs)
```rust
use omega_types::{Position, ExitReason};
use crate::actions::Action;

pub trait Rule: Send + Sync {
    fn evaluate(&self, position: &Position, timestamp_ns: i64) -> Option<Action>;
}

pub struct MaxHoldingTimeRule {
    pub max_bars: usize,
    pub bar_duration_ns: i64,
}

impl Rule for MaxHoldingTimeRule {
    fn evaluate(&self, position: &Position, timestamp_ns: i64) -> Option<Action> {
        let holding_ns = timestamp_ns - position.entry_time_ns;
        let holding_bars = holding_ns / self.bar_duration_ns;

        if holding_bars as usize >= self.max_bars {
            return Some(Action::ClosePosition {
                position_id: position.id,
                reason: ExitReason::Timeout,
            });
        }

        None
    }
}
```

## Tests
1. Jedes Szenario 1-6 separat
2. HTF-Filter Logik
3. Direction-Filter
4. Session/News Gates
5. MaxHoldingTime Rule
6. Parameter-Overrides

## V1-Python-Parität
KRITISCH: Signal-Generierung muss identisch zu V1 Python:
- Gleiche Bedingungsreihenfolge
- Gleiche Preisberechnungen
- Gleiche Meta-Daten

## Referenz-Dokumente
- OMEGA_V2_STRATEGIES_PLAN.md (Vollständig)
- OMEGA_V2_TRADE_MANAGER_PLAN.md (Vollständig)
```

### Akzeptanzkriterien W4

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Alle 6 Szenarien | Unit-Tests pro Szenario |
| A2 | HTF-Filter | Above/Below/Both/None |
| A3 | Session/News Gates | Blocking funktioniert |
| A4 | MaxHoldingTime | Timeout-Exits |
| A5 | V1-Python-Signal-Parität | Golden-File Vergleich |
| A6 | BarContext korrekt | Read-only, keine Mutation |

### Referenzen
- [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) - Vollständig
- [OMEGA_V2_TRADE_MANAGER_PLAN.md](OMEGA_V2_TRADE_MANAGER_PLAN.md) - Vollständig

---

## W5: Orchestration

### Beschreibung
Implementiert das `backtest`-Crate mit dem Event-Loop, der alle anderen Crates orchestriert.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Orchestration (Wave 5)

## Kontext
Du implementierst den Backtest-Orchestrator für ein Rust-natives Backtesting-System. Waves 0-4 sind bereits implementiert.

## Ziel
Erstelle das `backtest`-Crate mit dem Event-Loop, Warmup-Handling und Orchestrierung aller anderen Crates.

## Verzeichnisstruktur

```
rust_core/crates/backtest/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── engine.rs        # BacktestEngine
    ├── runner.rs        # High-level run_backtest()
    ├── context.rs       # Run-Context
    ├── warmup.rs        # Warmup-Handling
    ├── event_loop.rs    # Main Loop
    ├── result_builder.rs # Result-Assembly
    └── error.rs
```

## Cargo.toml

```toml
[package]
name = "omega_backtest"
version.workspace = true
edition.workspace = true

[dependencies]
omega_types = { workspace = true }
omega_data = { workspace = true }
omega_indicators = { workspace = true }
omega_execution = { workspace = true }
omega_portfolio = { workspace = true }
omega_strategy = { workspace = true }
omega_trade_mgmt = { workspace = true }

serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
tracing = "0.1"
rand_chacha = "0.3"
```

## BacktestEngine (engine.rs)

```rust
use omega_types::{BacktestConfig, BacktestResult, Trade, Candle};
use omega_data::{CandleStore, MultiTfStore};
use omega_indicators::IndicatorCache;
use omega_execution::{ExecutionEngine, SlippageModel, FeeModel};
use omega_portfolio::Portfolio;
use omega_strategy::{Strategy, BarContext};
use omega_trade_mgmt::TradeManager;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

pub struct BacktestEngine {
    config: BacktestConfig,
    data: MultiTfStore,
    indicators: IndicatorCache,
    execution: ExecutionEngine,
    portfolio: Portfolio,
    strategy: Box<dyn Strategy>,
    trade_manager: TradeManager,
    rng: ChaCha8Rng,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Result<Self, BacktestError> {
        // 1. Load Data
        let data = load_data(&config)?;

        // 2. Validate Warmup
        validate_warmup(&data, config.warmup_bars)?;

        // 3. Create Strategy
        let strategy = create_strategy(&config.strategy_name, &config.strategy_parameters)?;

        // 4. Pre-compute Indicators
        let indicators = compute_indicators(&data, strategy.required_indicators())?;

        // 5. Create Execution Engine
        let execution = create_execution_engine(&config)?;

        // 6. Create Portfolio
        let portfolio = Portfolio::new(
            config.account.initial_balance,
            config.account.max_positions.unwrap_or(1),
        );

        // 7. Create Trade Manager
        let trade_manager = create_trade_manager(&config)?;

        // 8. Create RNG
        let rng = match config.run_mode {
            RunMode::Dev => {
                let seed = config.rng_seed.unwrap_or(42);
                ChaCha8Rng::seed_from_u64(seed)
            }
            RunMode::Prod => {
                match config.rng_seed {
                    Some(seed) => ChaCha8Rng::seed_from_u64(seed),
                    None => ChaCha8Rng::from_entropy(),
                }
            }
        };

        Ok(Self {
            config,
            data,
            indicators,
            execution,
            portfolio,
            strategy,
            trade_manager,
            rng,
        })
    }

    pub fn run(mut self) -> BacktestResult {
        let warmup = self.config.warmup_bars;
        let len = self.data.primary.len();

        tracing::info!(
            "Starting backtest: {} bars ({} warmup, {} trading)",
            len, warmup, len - warmup
        );

        // Event Loop
        for idx in warmup..len {
            self.process_bar(idx);
        }

        // Build Result
        self.build_result()
    }

    fn process_bar(&mut self, idx: usize) {
        let timestamp_ns = self.data.primary.timestamps[idx];
        let (bid, ask) = self.data.primary.get(idx).unwrap();

        // 1. Build BarContext
        let ctx = self.build_context(idx, timestamp_ns, bid, ask);

        // 2. Check Pending Order Triggers (Limit/Stop)
        self.check_pending_triggers(idx, bid, ask, timestamp_ns);

        // 3. Check Stops (SL/TP) for open positions
        self.check_stops(idx, bid, ask, timestamp_ns);

        // 4. Trade Management Rules
        self.apply_trade_management(timestamp_ns);

        // 5. Strategy Signal (if entry allowed)
        if self.can_enter_position(&ctx) {
            if let Some(signal) = self.strategy.on_bar(&ctx) {
                self.process_signal(signal, idx, timestamp_ns);
            }
        }

        // 6. Update Equity
        let mid_price = (bid.close + ask.close) / 2.0;
        self.portfolio.update_equity(timestamp_ns, mid_price);
    }

    fn check_stops(&mut self, idx: usize, bid: &Candle, ask: &Candle, timestamp_ns: i64) {
        let positions_to_close: Vec<(u64, StopCheckResult)> = self.portfolio.positions
            .iter()
            .filter_map(|pos| {
                let in_entry_candle = pos.entry_time_ns == timestamp_ns;
                check_stops(
                    pos,
                    bid,
                    ask,
                    self.config.costs.pip_size,
                    self.config.costs.pip_buffer_factor.unwrap_or(0.5),
                    in_entry_candle,
                ).map(|result| (pos.id, result))
            })
            .collect();

        for (pos_id, result) in positions_to_close {
            // Apply exit slippage (inverted direction)
            let exit_slippage = self.calculate_exit_slippage(&result);
            let exit_price = result.exit_price + exit_slippage;

            // Calculate exit fee
            let position = self.portfolio.positions.iter().find(|p| p.id == pos_id).unwrap();
            let exit_fee = self.execution.fee_model.calculate(position.size, exit_price);

            // Close position
            self.portfolio.close_position(
                pos_id,
                exit_price,
                timestamp_ns,
                result.reason,
                exit_fee,
                &self.config.symbol,
            );
        }
    }

    fn can_enter_position(&self, ctx: &BarContext) -> bool {
        ctx.session_open
            && !ctx.news_blocked
            && self.portfolio.can_open_position()
    }

    fn build_context<'a>(
        &'a self,
        idx: usize,
        timestamp_ns: i64,
        bid: &'a Candle,
        ask: &'a Candle,
    ) -> BarContext<'a> {
        let session_open = self.check_session(timestamp_ns);
        let news_blocked = self.check_news_blocked(idx);
        let htf_data = self.get_htf_context(idx);

        BarContext {
            idx,
            timestamp_ns,
            bid,
            ask,
            indicators: &self.indicators,
            htf_data,
            session_open,
            news_blocked,
        }
    }

    fn build_result(self) -> BacktestResult {
        let trades = self.portfolio.closed_trades;
        let equity_curve = self.portfolio.equity_tracker.to_equity_points();

        BacktestResult {
            ok: true,
            error: None,
            trades: Some(trades),
            metrics: None,  // Computed by metrics crate
            equity_curve: Some(equity_curve),
            meta: Some(self.build_meta()),
        }
    }
}
```

## Runner (runner.rs)

```rust
use omega_types::{BacktestConfig, BacktestResult};
use crate::engine::BacktestEngine;

/// Main entry point - receives config JSON, returns result JSON
pub fn run_backtest_from_json(config_json: &str) -> Result<String, BacktestError> {
    // 1. Parse Config
    let config: BacktestConfig = serde_json::from_str(config_json)
        .map_err(|e| BacktestError::ConfigParse(e.to_string()))?;

    // 2. Validate Config
    validate_config(&config)?;

    // 3. Create Engine
    let engine = BacktestEngine::new(config)?;

    // 4. Run Backtest
    let result = engine.run();

    // 5. Serialize Result
    let result_json = serde_json::to_string(&result)
        .map_err(|e| BacktestError::ResultSerialize(e.to_string()))?;

    Ok(result_json)
}

fn validate_config(config: &BacktestConfig) -> Result<(), BacktestError> {
    // Symbol validation
    if config.symbol.is_empty() {
        return Err(BacktestError::ConfigValidation("symbol is empty".to_string()));
    }
    
    // Date range validation
    if config.start_date >= config.end_date {
        return Err(BacktestError::ConfigValidation(
            "start_date must be before end_date".to_string()
        ));
    }
    
    // Warmup validation (default: 500)
    if config.warmup_bars == 0 {
        return Err(BacktestError::ConfigValidation(
            "warmup_bars must be > 0".to_string()
        ));
    }
    
    // Timeframe validation (must be uppercase: M1, M5, M15, M30, H1, H4, D1)
    let valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];
    if !valid_timeframes.contains(&config.timeframes.primary.as_str()) {
        return Err(BacktestError::ConfigValidation(format!(
            "invalid primary timeframe '{}', must be one of: {:?}",
            config.timeframes.primary, valid_timeframes
        )));
    }
    for tf in &config.timeframes.additional {
        if !valid_timeframes.contains(&tf.as_str()) {
            return Err(BacktestError::ConfigValidation(format!(
                "invalid additional timeframe '{}', must be one of: {:?}",
                tf, valid_timeframes
            )));
        }
    }
    
    // Execution variant validation
    match config.execution_variant.as_str() {
        "v2" | "v1_parity" => {}
        _ => {
            return Err(BacktestError::ConfigValidation(format!(
                "invalid execution_variant '{}', must be 'v2' or 'v1_parity'",
                config.execution_variant
            )));
        }
    }
    
    // Account validation
    if config.account.initial_balance <= 0.0 {
        return Err(BacktestError::ConfigValidation(
            "account.initial_balance must be > 0".to_string()
        ));
    }
    if config.account.risk_per_trade <= 0.0 {
        return Err(BacktestError::ConfigValidation(
            "account.risk_per_trade must be > 0".to_string()
        ));
    }
    
    // Costs validation (multipliers must be >= 0)
    if config.costs.fee_multiplier < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.fee_multiplier must be >= 0".to_string()
        ));
    }
    if config.costs.slippage_multiplier < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.slippage_multiplier must be >= 0".to_string()
        ));
    }
    if config.costs.spread_multiplier < 0.0 {
        return Err(BacktestError::ConfigValidation(
            "costs.spread_multiplier must be >= 0".to_string()
        ));
    }
    
    Ok(())
}
```

## Warmup Handling (warmup.rs)

```rust
use omega_data::CandleStore;
use crate::error::BacktestError;

/// Validates that sufficient data exists for warmup
pub fn validate_warmup(data: &CandleStore, warmup_bars: usize) -> Result<(), BacktestError> {
    if data.len() <= warmup_bars {
        return Err(BacktestError::InsufficientData {
            required: warmup_bars + 1,
            available: data.len(),
        });
    }

    tracing::info!(
        "Warmup validated: {} bars required, {} available ({} trading bars)",
        warmup_bars,
        data.len(),
        data.len() - warmup_bars
    );

    Ok(())
}

/// Validates HTF warmup (if HTF enabled)
pub fn validate_htf_warmup(
    htf_data: Option<&CandleStore>,
    warmup_bars: usize,
) -> Result<(), BacktestError> {
    if let Some(htf) = htf_data {
        if htf.len() < warmup_bars {
            return Err(BacktestError::InsufficientHtfData {
                required: warmup_bars,
                available: htf.len(),
            });
        }
    }
    Ok(())
}
```

## Session/News Checks

```rust
impl BacktestEngine {
    fn check_session(&self, timestamp_ns: i64) -> bool {
        match &self.config.sessions {
            None => true,  // No sessions = always open
            Some(sessions) => {
                // Convert timestamp to UTC time-of-day
                let time_of_day = extract_time_of_day(timestamp_ns);
                sessions.iter().any(|s| s.contains(time_of_day))
            }
        }
    }

    fn check_news_blocked(&self, idx: usize) -> bool {
        match &self.news_mask {
            None => false,  // No news filter = never blocked
            Some(mask) => mask.get(idx).copied().unwrap_or(false),
        }
    }
}
```

## Tests

1. Event-Loop Sequenz (Trigger → Stops → Management → Signal)
2. Warmup-Validierung (Fail bei insufficient data)
3. Session-Filter (Trades nur in Sessions)
4. News-Filter (Trades nicht in Blackout)
5. Determinismus (DEV-Mode)
6. End-to-End mit kleiner Fixture

## Referenz-Dokumente
- OMEGA_V2_DATA_FLOW_PLAN.md (Phase 4 - Event Loop)
- OMEGA_V2_EXECUTION_MODEL_PLAN.md (Event-Loop Reihenfolge)
- OMEGA_V2_CONFIG_SCHEMA_PLAN.md (Sessions, News Filter)
```

### Akzeptanzkriterien W5

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Event-Loop korrekt | Sequenz: Trigger → Stops → Mgmt → Signal |
| A2 | Warmup-Skip | Keine Signale in Warmup-Phase |
| A3 | Session-Filter | Trades nur in konfigurierten Sessions |
| A4 | News-Blocking | Kein Entry während News-Window |
| A5 | Determinismus | Identische Ergebnisse bei gleichem Seed |
| A6 | Equity-Tracking | Per-Bar Equity in Result |

### Referenzen
- [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) - Phase 4
- [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) - Event-Loop

---

## W6: Output + FFI

### Beschreibung
Implementiert `metrics` und `ffi` Crates für Metriken-Berechnung und Python-Binding.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Output + FFI (Wave 6)

## Kontext
Du implementierst das Output-Layer für ein Rust-natives Backtesting-System. Waves 0-5 sind bereits implementiert.

## Ziel
Erstelle die `metrics` und `ffi` Crates für Metriken-Berechnung und PyO3 Python-Binding.

## Verzeichnisstruktur

```
rust_core/crates/
├── metrics/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── compute.rs       # Main compute_metrics()
│       ├── definitions.rs   # MetricDefinition struct
│       ├── trade_metrics.rs # Trade-basierte Metriken
│       ├── equity_metrics.rs # Equity-basierte Metriken
│       └── output.rs        # MetricsOutput Serialisierung
│
└── ffi/
    ├── Cargo.toml
    └── src/
        └── lib.rs           # PyO3 Entry Point
```

## metrics/Cargo.toml

```toml
[package]
name = "omega_metrics"
version.workspace = true
edition.workspace = true

[dependencies]
omega_types = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

## Metrics Compute (metrics/compute.rs)

```rust
use omega_types::{Trade, EquityPoint, Metrics};
use crate::definitions::MetricDefinitions;
use std::collections::HashSet;

/// Hauptfunktion zur Metriken-Berechnung
pub fn compute_metrics(
    trades: &[Trade],
    equity_curve: &[EquityPoint],
    fees_total: f64,
    risk_per_trade: f64,
) -> MetricsOutput {
    let mut metrics = Metrics::default();

    // Trade-basierte Metriken
    metrics.total_trades = trades.len() as u64;
    metrics.wins = trades.iter().filter(|t| t.result > 0.0).count() as u64;
    metrics.losses = trades.iter().filter(|t| t.result < 0.0).count() as u64;

    // Win Rate
    metrics.win_rate = if metrics.total_trades > 0 {
        metrics.wins as f64 / metrics.total_trades as f64
    } else {
        0.0
    };

    // Profit/Fees (explizite Fees kommen aus Execution)
    metrics.profit_gross = trades.iter().map(|t| t.result).sum();
    metrics.fees_total = fees_total;
    metrics.profit_net = metrics.profit_gross - metrics.fees_total;

    // Drawdown (equity-basiert)
    let (max_dd, max_dd_abs, max_dd_duration) = compute_drawdown(equity_curve);
    metrics.max_drawdown = max_dd;
    metrics.max_drawdown_abs = max_dd_abs;
    metrics.max_drawdown_duration_bars = max_dd_duration;

    // R-Multiple (normativ: trade_pnl / risk_per_trade)
    metrics.avg_r_multiple = if metrics.total_trades > 0 && risk_per_trade > 0.0 {
        trades
            .iter()
            .map(|t| t.result / risk_per_trade)
            .sum::<f64>()
            / metrics.total_trades as f64
    } else {
        0.0
    };

    // MVP+: avg_trade_pnl + expectancy (MVP: == avg_r_multiple)
    metrics.avg_trade_pnl = if metrics.total_trades > 0 {
        metrics.profit_net / metrics.total_trades as f64
    } else {
        0.0
    };
    metrics.expectancy = metrics.avg_r_multiple;

    // MVP+: active_days + trades_per_day (UTC-Tage aus entry_time_ns)
    let unique_days: HashSet<i64> = trades
        .iter()
        .map(|t| t.entry_time_ns / 86_400_000_000_000) // ns pro Tag
        .collect();
    metrics.active_days = unique_days.len() as u64;
    metrics.trades_per_day = if metrics.active_days > 0 {
        metrics.total_trades as f64 / metrics.active_days as f64
    } else {
        0.0
    };

    // Profit Factor
    let gross_profit: f64 = trades.iter().filter(|t| t.result > 0.0).map(|t| t.result).sum();
    let gross_loss: f64 = trades.iter().filter(|t| t.result < 0.0).map(|t| t.result.abs()).sum();
    metrics.profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        0.0
    };

    // Round and create output
    let definitions = MetricDefinitions::default();
    MetricsOutput::new(round_metrics(metrics), definitions)
}
```

## Drawdown Calculation (metrics/equity_metrics.rs)

```rust
use omega_types::EquityPoint;

/// Berechnet Max Drawdown (relativ, absolut, Dauer)
pub fn compute_drawdown(equity: &[EquityPoint]) -> (f64, f64, u64) {
    if equity.is_empty() {
        return (0.0, 0.0, 0);
    }

    let mut high_water = equity[0].equity;
    let mut max_dd_rel = 0.0;
    let mut max_dd_abs = 0.0;
    let mut current_dd_start = 0usize;
    let mut max_dd_duration = 0u64;
    let mut in_drawdown = false;

    for (i, point) in equity.iter().enumerate() {
        if point.equity > high_water {
            // New high water mark
            if in_drawdown {
                let duration = (i - current_dd_start) as u64;
                max_dd_duration = max_dd_duration.max(duration);
                in_drawdown = false;
            }
            high_water = point.equity;
        } else if high_water > 0.0 {
            // In drawdown
            if !in_drawdown {
                current_dd_start = i;
                in_drawdown = true;
            }

            let dd_abs = high_water - point.equity;
            let dd_rel = dd_abs / high_water;

            max_dd_abs = max_dd_abs.max(dd_abs);
            max_dd_rel = max_dd_rel.max(dd_rel);
        }
    }

    // Check if still in drawdown at end
    if in_drawdown {
        let duration = (equity.len() - current_dd_start) as u64;
        max_dd_duration = max_dd_duration.max(duration);
    }

    // Clamp to [0, 1]
    max_dd_rel = max_dd_rel.clamp(0.0, 1.0);

    (max_dd_rel, max_dd_abs, max_dd_duration)
}
```

## Rounding (metrics/output.rs)

```rust
use omega_types::Metrics;

/// Rundet Metriken gemäß Output-Contract
pub fn round_metrics(mut metrics: Metrics) -> Metrics {
    // account_currency: 2 Dezimalstellen
    metrics.profit_gross = round_to_decimals(metrics.profit_gross, 2);
    metrics.profit_net = round_to_decimals(metrics.profit_net, 2);
    metrics.fees_total = round_to_decimals(metrics.fees_total, 2);
    metrics.max_drawdown_abs = round_to_decimals(metrics.max_drawdown_abs, 2);
    metrics.avg_trade_pnl = round_to_decimals(metrics.avg_trade_pnl, 2);

    // ratios: 6 Dezimalstellen
    metrics.win_rate = round_to_decimals(metrics.win_rate, 6);
    metrics.max_drawdown = round_to_decimals(metrics.max_drawdown, 6);
    metrics.avg_r_multiple = round_to_decimals(metrics.avg_r_multiple, 6);
    metrics.profit_factor = round_to_decimals(metrics.profit_factor, 6);
    metrics.expectancy = round_to_decimals(metrics.expectancy, 6);
    metrics.trades_per_day = round_to_decimals(metrics.trades_per_day, 6);

    metrics
}

fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    (value * factor).round() / factor
}
```

## MetricsOutput mit Definitions

```rust
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use omega_types::Metrics;

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsOutput {
    pub metrics: Metrics,
    pub definitions: HashMap<String, MetricDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    pub unit: String,
    pub description: String,
    pub domain: String,
    pub source: String,
    #[serde(rename = "type")]
    pub value_type: String,
}

impl MetricDefinitions {
    pub fn default() -> HashMap<String, MetricDefinition> {
        let mut defs = HashMap::new();

        defs.insert("total_trades".to_string(), MetricDefinition {
            unit: "count".to_string(),
            description: "Anzahl abgeschlossener Trades".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("win_rate".to_string(), MetricDefinition {
            unit: "ratio".to_string(),
            description: "Anteil der Gewinntrades an allen Trades".to_string(),
            domain: "0..1".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("wins".to_string(), MetricDefinition {
            unit: "count".to_string(),
            description: "Anzahl Gewinntrades (trade.result > 0)".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("losses".to_string(), MetricDefinition {
            unit: "count".to_string(),
            description: "Anzahl Verlusttrades (trade.result < 0)".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("profit_gross".to_string(), MetricDefinition {
            unit: "account_currency".to_string(),
            description: "Summe aller Trade-Ergebnisse vor Fees".to_string(),
            domain: "any".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("fees_total".to_string(), MetricDefinition {
            unit: "account_currency".to_string(),
            description: "Summe expliziter Fees/Commission (ohne Spread/Slippage)".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("profit_net".to_string(), MetricDefinition {
            unit: "account_currency".to_string(),
            description: "profit_gross - fees_total".to_string(),
            domain: "any".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("max_drawdown".to_string(), MetricDefinition {
            unit: "ratio".to_string(),
            description: "Maximaler relativer Drawdown (Peak-to-Trough / Peak)".to_string(),
            domain: "0..1".to_string(),
            source: "equity".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("max_drawdown_abs".to_string(), MetricDefinition {
            unit: "account_currency".to_string(),
            description: "Maximaler absoluter Drawdown in Währung".to_string(),
            domain: ">=0".to_string(),
            source: "equity".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("max_drawdown_duration_bars".to_string(), MetricDefinition {
            unit: "bars".to_string(),
            description: "Längste Drawdown-Periode in Bars".to_string(),
            domain: ">=0".to_string(),
            source: "equity".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("avg_r_multiple".to_string(), MetricDefinition {
            unit: "r_multiple".to_string(),
            description: "Durchschnittliches R-Multiple aller Trades".to_string(),
            domain: "any".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("profit_factor".to_string(), MetricDefinition {
            unit: "ratio".to_string(),
            description: "sum(positive_pnl) / abs(sum(negative_pnl))".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        // MVP+ Metrics (SOLL)
        defs.insert("avg_trade_pnl".to_string(), MetricDefinition {
            unit: "account_currency".to_string(),
            description: "Durchschnittlicher PnL pro Trade (profit_net / total_trades)".to_string(),
            domain: "any".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("expectancy".to_string(), MetricDefinition {
            unit: "r_multiple".to_string(),
            description: "Erwartungswert pro Trade in R (MVP: avg_r_multiple)".to_string(),
            domain: "any".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("active_days".to_string(), MetricDefinition {
            unit: "days".to_string(),
            description: "Anzahl Tage mit mindestens einem Trade".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs.insert("trades_per_day".to_string(), MetricDefinition {
            unit: "trades".to_string(),
            description: "total_trades / active_days".to_string(),
            domain: ">=0".to_string(),
            source: "trades".to_string(),
            value_type: "number".to_string(),
        });

        defs
    }
}
```

## Result-Assembly Integration (BacktestResult)

**Wichtig:** `BacktestResult` erhält ein zusätzliches Feld `metric_definitions`, damit der Output-Contract `metrics.json` (metrics + definitions) erfüllt werden kann.

```rust
use omega_metrics::compute::compute_metrics;

fn build_result(self) -> BacktestResult {
    let trades = self.portfolio.closed_trades;
    let equity_curve = self.portfolio.equity_tracker.to_equity_points();
    let fees_total = self.execution.fees_total();

    let metrics_output = compute_metrics(
        &trades,
        &equity_curve,
        fees_total,
        self.config.account.risk_per_trade,
    );

    BacktestResult {
        ok: true,
        error: None,
        trades: Some(trades),
        metrics: Some(metrics_output.metrics),
        metric_definitions: Some(metrics_output.definitions),
        equity_curve: Some(equity_curve),
        meta: Some(self.build_meta()),
    }
}
```

## FFI Crate (ffi/lib.rs)

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use omega_backtest::runner::run_backtest_from_json;
use omega_types::{BacktestResult, ErrorResult};

/// Python-Modul für Omega V2 Backtest
#[pymodule]
fn omega_bt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}

/// Main entry point - receives config JSON, returns result JSON
#[pyfunction]
fn run_backtest(config_json: &str) -> PyResult<String> {
    match run_backtest_from_json(config_json) {
        Ok(result_json) => Ok(result_json),
        Err(e) if e.is_config_error() => Err(PyValueError::new_err(e.to_string())),
        Err(e) => {
            let error_result = BacktestResult {
                ok: false,
                error: Some(ErrorResult::from(e)),
                trades: None,
                metrics: None,
                metric_definitions: None,
                equity_curve: None,
                meta: None,
            };
            let error_json = serde_json::to_string(&error_result)
                .unwrap_or_else(|_| "{\"ok\":false,\"error\":{\"category\":\"runtime\",\"message\":\"serialization_failed\"}}".to_string());
            Ok(error_json)
        }
    }
}
```

```rust
impl BacktestError {
    pub fn is_config_error(&self) -> bool {
        matches!(self, BacktestError::ConfigParse(_)|BacktestError::ConfigValidation(_))
    }
}
```

## ffi/Cargo.toml

```toml
[package]
name = "omega_bt"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib"]

[dependencies]
omega_types = { workspace = true }
omega_backtest = { workspace = true }
omega_metrics = { workspace = true }
pyo3 = { version = "0.20", features = ["extension-module"] }
serde_json = { workspace = true }
```

## Build Configuration (pyproject.toml Addition)

```toml
[tool.maturin]
module-name = "omega_bt"
python-source = "python"
features = ["extension-module"]
```

## Tests

1. Alle Metriken-Formeln gegen bekannte Werte
2. Drawdown Edge-Cases (konstante Equity, nur Verluste)
3. Rundung (2dp für Currency, 6dp für Ratios)
4. FFI Roundtrip (JSON → Rust → JSON)
5. Error-Contract: Config-Fehler → Python Exception, Runtime → JSON Error
6. Fees-Semantik: fees_total enthält nur explizite Fees/Commission

## Referenz-Dokumente
- OMEGA_V2_METRICS_DEFINITION_PLAN.md (Vollständig)
- OMEGA_V2_OUTPUT_CONTRACT_PLAN.md (Vollständig)
- OMEGA_V2_TECH_STACK_PLAN.md (PyO3, Maturin)
```

### Akzeptanzkriterien W6

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Alle MVP-Metriken | total_trades, win_rate, profit_*, drawdown |
| A2 | Korrekte Rundung | 2dp Currency, 6dp Ratios |
| A3 | Definitions vollständig | Jeder Key hat Definition |
| A4 | FFI funktioniert | Python kann `run_backtest()` aufrufen |
| A5 | Error-Handling | Errors als JSON-Result |
| A6 | Wheel baut | maturin build erfolgreich |

### Referenzen
- [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) - Vollständig
- [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) - Vollständig
- [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) - PyO3, Maturin

---

## W7: Integration

### Beschreibung
Erstellt den Python-Wrapper, Output-Artefakte und V1-Python-Parity-Tests.

### Codex-Max Prompt

```markdown
# Task: Omega V2 Integration (Wave 7)

## Kontext
Du implementierst die Python-Integration für ein Rust-natives Backtesting-System. Waves 0-6 sind bereits implementiert.

## Ziel
Erstelle das Python-Package `bt`, Output-Artefakt-Writer und V1-Python-Parity-Tests.

## Verzeichnisstruktur

```
python/
├── bt/
│   ├── __init__.py
│   ├── runner.py        # High-level run_backtest()
│   ├── config.py        # Config-Loading/Validation
│   ├── output.py        # Artefakt-Writer
│   └── reporting.py     # Optional: Visualisierung
│
└── tests/
    ├── conftest.py
    ├── test_integration.py
    ├── test_golden.py
    ├── test_parity.py   # V1 vs V2 Vergleich
    └── fixtures/
        ├── golden/      # Golden-File Fixtures
        └── data/        # Parquet/CSV Testdaten
```

## Python Package (bt/__init__.py)

```python
"""Omega V2 Backtest Python Interface"""

from .runner import run_backtest
from .config import load_config, validate_config
from .output import write_artifacts

__all__ = ["run_backtest", "load_config", "validate_config", "write_artifacts"]
```

## Runner (bt/runner.py)

```python
"""High-level backtest runner"""

import json
from pathlib import Path
from typing import Any
import omega_bt  # Rust FFI Module

from .config import load_config, validate_config
from .output import write_artifacts


def run_backtest(
    config_path: str | Path | None = None,
    config_dict: dict | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run a V2 backtest.

    Args:
        config_path: Path to config JSON file
        config_dict: Config as dictionary (alternative to config_path)
        output_dir: Directory for output artifacts (default: var/results/backtests/<run_id>/)

    Returns:
        BacktestResult as dictionary
    """
    # 1. Load Config
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        config = load_config(config_path)
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    # 2. Validate Config
    validate_config(config)

    # 3. Serialize and call Rust
    config_json = json.dumps(config)
    result_json = omega_bt.run_backtest(config_json)

    # 4. Parse Result
    result = json.loads(result_json)

    # 5. Write Artifacts (if output_dir specified or default)
    if output_dir is not None or result.get("ok", False):
        run_id = result.get("meta", {}).get("run_id", "unknown")
        if output_dir is None:
            output_dir = Path("var/results/backtests") / run_id
        write_artifacts(result, output_dir)

    return result
```

## Config Loading (bt/config.py)

```python
"""Config loading and validation"""

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config from JSON file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = json.load(f)

    # Apply defaults
    config.setdefault("schema_version", "2.0")
    config.setdefault("run_mode", "dev")
    config.setdefault("data_mode", "candle")
    config.setdefault("execution_variant", "v2")
    config.setdefault("warmup_bars", 500)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate config structure"""
    required = ["strategy_name", "symbol", "start_date", "end_date"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    if config["start_date"] >= config["end_date"]:
        raise ValueError("start_date must be before end_date")
```

## Output Writer (bt/output.py)

```python
"""Artifact output writer"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_artifacts(result: dict[str, Any], output_dir: str | Path) -> None:
    """
    Write all output artifacts to directory.

    Artifacts:
        - meta.json
        - trades.json
        - equity.csv
        - metrics.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. meta.json
    meta = result.get("meta", {})
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    meta["generated_at_ns"] = int(datetime.now(timezone.utc).timestamp() * 1e9)
    write_json(output_dir / "meta.json", meta)

    # 2. trades.json (Root is Array)
    trades = result.get("trades", [])
    write_json(output_dir / "trades.json", trades)

    # 3. metrics.json
    metrics_output = {
        "metrics": result.get("metrics", {}),
        "definitions": result.get("metric_definitions", {}),
    }
    write_json(output_dir / "metrics.json", metrics_output)

    # 4. equity.csv
    equity = result.get("equity_curve", [])
    write_equity_csv(output_dir / "equity.csv", equity)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with stable formatting"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_equity_csv(path: Path, equity: list[dict]) -> None:
    """Write equity curve as CSV"""
    if not equity:
        # Write empty file with header
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("timestamp,timestamp_ns,equity,balance,drawdown,high_water\n")
        return

    fieldnames = ["timestamp", "timestamp_ns", "equity", "balance", "drawdown", "high_water"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in equity:
            # Convert timestamp_ns to ISO
            ts_ns = point.get("timestamp_ns", 0)
            ts_iso = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()
            row = {
                "timestamp": ts_iso,
                "timestamp_ns": ts_ns,
                "equity": point.get("equity", 0),
                "balance": point.get("balance", 0),
                "drawdown": point.get("drawdown", 0),
                "high_water": point.get("high_water", 0),
            }
            writer.writerow(row)
```

## Parity Tests (tests/test_parity.py)

```python
"""V1 vs V2 Parity Tests"""

import json
import pytest
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# Import V1 runner (existing Python implementation)
from src.backtest_engine.runner import run_backtest as run_v1_backtest

# Import V2 runner
from bt import run_backtest as run_v2_backtest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"


class TestV1V2Parity:
    """Tests for V1 Python ↔ V2 parity"""

    @pytest.fixture
    def parity_config(self):
        """Config for parity testing"""
        return {
            "schema_version": "1.0.0",
            "strategy_name": "mean_reversion_z_score",
            "symbol": "EURUSD",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "run_mode": "dev",
            "data_mode": "parquet",
            "execution_variant": "v1_parity",  # V1-Python-Paritätsmodus
            "rng_seed": 42,
            "timeframes": {
                "primary": "M1",
                "htf": ["M5", "H1", "D1"]
            },
            "warmup_bars": 500,
            "account": {
                "initial_balance": 10000.0,
                "account_currency": "EUR",
                "risk_per_trade": 0.01,
                "max_positions": 1
            },
            "costs": {
                "spread_multiplier": 1.0,
                "commission_multiplier": 1.0,
                "slippage_multiplier": 1.0
            },
            "strategy_params": {
                "atr_length": 14,
                "atr_mult": 1.5,
                "bb_length": 20,
                "std_factor": 2.0,
                "z_score_long": -2.0,
                "z_score_short": 2.0,
                "use_htf_filter": True,
                "htf_ema_length": 50
            }
        }

    def test_trade_count_parity(self, parity_config):
        """Trade count must match between V1 and V2"""
        v1_result = run_v1_backtest(parity_config)
        v2_result = run_v2_backtest(config_dict=parity_config)

        v1_trades = v1_result.get("trades", [])
        v2_trades = v2_result.get("trades", [])

        assert len(v1_trades) == len(v2_trades), (
            f"Trade count mismatch: V1={len(v1_trades)}, V2={len(v2_trades)}"
        )

    def test_trade_events_parity(self, parity_config):
        """Entry/Exit events must match"""
        v1_result = run_v1_backtest(parity_config)
        v2_result = run_v2_backtest(config_dict=parity_config)

        v1_trades = v1_result.get("trades", [])
        v2_trades = v2_result.get("trades", [])

        for i, (v1, v2) in enumerate(zip(v1_trades, v2_trades)):
            # Entry time must match exactly
            assert v1["entry_time_ns"] == v2["entry_time_ns"], (
                f"Trade {i}: entry_time mismatch"
            )
            # Exit time must match exactly
            assert v1["exit_time_ns"] == v2["exit_time_ns"], (
                f"Trade {i}: exit_time mismatch"
            )
            # Direction must match
            assert v1["direction"] == v2["direction"], (
                f"Trade {i}: direction mismatch"
            )
            # Exit reason must match
            assert v1["reason"] == v2["reason"], (
                f"Trade {i}: reason mismatch"
            )

    def test_price_parity_quantized(self, parity_config):
        """Prices must match after tick-size quantization"""
        tick_size = Decimal("0.00001")  # EURUSD tick

        v1_result = run_v1_backtest(parity_config)
        v2_result = run_v2_backtest(config_dict=parity_config)

        for i, (v1, v2) in enumerate(zip(v1_result["trades"], v2_result["trades"])):
            v1_entry = quantize_price(v1["entry_price"], tick_size)
            v2_entry = quantize_price(v2["entry_price"], tick_size)
            assert v1_entry == v2_entry, f"Trade {i}: entry_price mismatch"

            v1_exit = quantize_price(v1["exit_price"], tick_size)
            v2_exit = quantize_price(v2["exit_price"], tick_size)
            assert v1_exit == v2_exit, f"Trade {i}: exit_price mismatch"

    def test_pnl_tolerance(self, parity_config):
        """PnL must be within tolerance after rounding"""
        v1_result = run_v1_backtest(parity_config)
        v2_result = run_v2_backtest(config_dict=parity_config)

        # Per-trade tolerance: ±0.05 after 2dp rounding
        for i, (v1, v2) in enumerate(zip(v1_result["trades"], v2_result["trades"])):
            v1_pnl = round(v1["result"], 2)
            v2_pnl = round(v2["result"], 2)
            assert abs(v1_pnl - v2_pnl) <= 0.05, (
                f"Trade {i}: PnL mismatch V1={v1_pnl}, V2={v2_pnl}"
            )

        # Aggregate tolerance: ±0.01 after 2dp rounding
        v1_total = round(sum(t["result"] for t in v1_result["trades"]), 2)
        v2_total = round(sum(t["result"] for t in v2_result["trades"]), 2)
        assert abs(v1_total - v2_total) <= 0.01, (
            f"Aggregate PnL mismatch: V1={v1_total}, V2={v2_total}"
        )


def quantize_price(price: float, tick_size: Decimal) -> Decimal:
    """Quantize price to tick size"""
    return (Decimal(str(price)) / tick_size).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) * tick_size


class TestDeterminism:
    """Tests for DEV-mode determinism"""

    def test_identical_runs(self, parity_config):
        """Two identical DEV runs must produce identical results"""
        result1 = run_v2_backtest(config_dict=parity_config)
        result2 = run_v2_backtest(config_dict=parity_config)

        # Normalize (remove timestamps)
        normalize_result(result1)
        normalize_result(result2)

        assert result1 == result2, "DEV-mode runs are not deterministic"


def normalize_result(result: dict) -> None:
    """Remove non-deterministic fields for comparison"""
    if "meta" in result:
        result["meta"].pop("generated_at", None)
        result["meta"].pop("generated_at_ns", None)
```

## Golden File Tests (tests/test_integration.py)

```python
"""Integration and Golden File Tests"""

import json
import pytest
from pathlib import Path

from bt import run_backtest


GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


class TestGoldenFiles:
    """Golden file regression tests"""

    @pytest.mark.parametrize("scenario", [
        "scenario_1_market_tp",
        "scenario_2_market_sl",
        "scenario_3_pending_trigger",
        "scenario_4_sl_tp_tiebreak",
        "scenario_5_in_entry_candle",
        "scenario_6_full_mix",
    ])
    def test_golden_scenario(self, scenario):
        """Test against golden file for scenario"""
        config_path = GOLDEN_DIR / f"{scenario}_config.json"
        golden_path = GOLDEN_DIR / f"{scenario}_expected.json"

        # Run backtest
        result = run_backtest(config_path=config_path)

        # Load golden
        with open(golden_path) as f:
            expected = json.load(f)

        # Compare (normalized)
        assert_results_equal(result, expected)


def assert_results_equal(actual: dict, expected: dict) -> None:
    """Compare results with normalization"""
    # Normalize timestamps
    normalize_result(actual)
    normalize_result(expected)

    # Compare trades
    assert len(actual.get("trades", [])) == len(expected.get("trades", []))

    for i, (a, e) in enumerate(zip(actual["trades"], expected["trades"])):
        assert a["entry_time_ns"] == e["entry_time_ns"], f"Trade {i} entry mismatch"
        assert a["exit_time_ns"] == e["exit_time_ns"], f"Trade {i} exit mismatch"
        assert a["direction"] == e["direction"], f"Trade {i} direction mismatch"
        assert a["reason"] == e["reason"], f"Trade {i} reason mismatch"

    # Compare metrics (with tolerance)
    for key in expected.get("metrics", {}):
        if key in actual.get("metrics", {}):
            if isinstance(expected["metrics"][key], float):
                assert abs(actual["metrics"][key] - expected["metrics"][key]) < 1e-6
            else:
                assert actual["metrics"][key] == expected["metrics"][key]
```

## Tests (Python-seitig)

1. Config Loading und Validation
2. Output-Artefakt-Writing (alle 4 Files)
3. Golden-File Regression (6 Szenarien)
4. V1-Python–V2 Parity (Trade-Events, Preise, PnL)
5. Determinismus (identische DEV-Runs)
6. Error-Handling (ungültige Config)

## Referenz-Dokumente
- OMEGA_V2_TESTING_VALIDATION_PLAN.md (Vollständig)
- OMEGA_V2_OUTPUT_CONTRACT_PLAN.md (Artefakt-Formate)
```

### Akzeptanzkriterien W7

| # | Kriterium | Validierung |
|---|-----------|-------------|
| A1 | Python-Package funktioniert | `from bt import run_backtest` |
| A2 | Alle 4 Artefakte | meta.json, trades.json, equity.csv, metrics.json |
| A3 | Golden-Tests bestehen | 6 Szenarien |
| A4 | V1-Python-Parity | Events exakt, PnL ±0.05 |
| A5 | Determinismus | Identische DEV-Runs |
| A6 | pip install funktioniert | maturin develop --manifest-path rust_core/crates/ffi/Cargo.toml |

### Referenzen
- [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) - Vollständig
- [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) - Artefakt-Formate

---

## Checkliste für jede Welle

Vor dem Start einer Welle:
- [ ] Vorherige Welle abgeschlossen und getestet
- [ ] Alle referenzierten Plan-Dokumente gelesen
- [ ] Codex-Max Prompt vorbereitet

Nach Abschluss einer Welle:
- [ ] `cargo build` erfolgreich
- [ ] `cargo test` alle Tests bestehen
- [ ] `cargo clippy -- -D warnings` clean
- [ ] `cargo fmt --check` formatiert
- [ ] Dokumentation vollständig

---

## Anhang: Vollständige Referenzliste

| Dokument | Relevante Wellen |
|----------|------------------|
| OMEGA_V2_VISION_PLAN.md | Alle |
| OMEGA_V2_ARCHITECTURE_PLAN.md | W0, W5 |
| OMEGA_V2_MODULE_STRUCTURE_PLAN.md | W0-W6 |
| OMEGA_V2_DATA_FLOW_PLAN.md | W1, W5 |
| OMEGA_V2_DATA_GOVERNANCE_PLAN.md | W1 |
| OMEGA_V2_CONFIG_SCHEMA_PLAN.md | W0, W6 |
| OMEGA_V2_EXECUTION_MODEL_PLAN.md | W3 |
| OMEGA_V2_STRATEGIES_PLAN.md | W4 |
| OMEGA_V2_TRADE_MANAGER_PLAN.md | W4 |
| OMEGA_V2_INDICATOR_CACHE__PLAN.md | W2 |
| OMEGA_V2_METRICS_DEFINITION_PLAN.md | W6 |
| OMEGA_V2_OUTPUT_CONTRACT_PLAN.md | W6 |
| OMEGA_V2_TECH_STACK_PLAN.md | W0, W6 |
| OMEGA_V2_TESTING_VALIDATION_PLAN.md | W7 |
| OMEGA_V2_CI_WORKFLOW_PLAN.md | W7 |

---

*Dieser Implementierungsplan ist die verbindliche Referenz für die wellenweise Umsetzung des Omega V2 Systems.*
