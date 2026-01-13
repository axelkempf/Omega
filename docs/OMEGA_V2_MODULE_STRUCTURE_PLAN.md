# Omega V2 – Modul-Struktur-Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Detaillierte Spezifikation der Ordner-, Datei- und Modul-Struktur für das Omega V2 Backtesting-System  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Vision, strategische Ziele, Erfolgskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Übergeordneter Blueprint, Module, Regeln |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Detaillierter Datenfluss, Phasen, Validierung |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives JSON-Config-Schema (Felder, Defaults, Validierung, Migration) |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell: Bid/Ask, Fills, SL/TP, Slippage/Fees |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Metrik-Modul (`metrics` Crate), Keys/Definitionen, File-per-Metric |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (Artefakte, Schema, Zeit/Units, Pfade) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Packaging/Build-Matrix |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI/CD Workflow, Quality Gates, Build-Matrix, Security, Release-Assets |

---

## 1. Übersicht

Dieser Plan definiert die vollständige Verzeichnis- und Dateistruktur des neuen Rust-basierten Backtesting-Kerns sowie des Python-Wrappers. Jede Datei ist mit ihrer Verantwortlichkeit dokumentiert, und die Abhängigkeiten zwischen Modulen sind explizit dargestellt.

### 1.1 Design-Prinzipien

| Prinzip | Beschreibung |
|---------|--------------|
| **Separation of Concerns** | Jedes Crate hat genau eine Verantwortlichkeit |
| **Einweg-Abhängigkeiten** | Abhängigkeiten fließen nur nach unten (keine Zyklen) |
| **Explizite Schnittstellen** | Nur klar definierte `pub` Exports pro Modul |
| **Testbarkeit** | Jedes Crate ist isoliert testbar |
| **Erweiterbarkeit** | Neue Strategien/Indikatoren ohne Kern-Änderungen |

---

## 2. Vollständige Verzeichnisstruktur

```
rust_core/                              ← Workspace Root
│
├── Cargo.toml                          ← Workspace-Definition
├── Cargo.lock                          ← Lock-File (versioniert)
├── rust-toolchain.toml                 ← Rust-Version Pinning
├── .cargo/
│   └── config.toml                     ← Cargo-Konfiguration (opt-level, etc.)
│
├── crates/
│   │
│   ├── types/                          ← [CRATE] Gemeinsame Datentypen
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports aller Module
│   │       ├── candle.rs               ← Candle-Struktur
│   │       ├── signal.rs               ← Signal-Enum (Long/Short/Exit)
│   │       ├── trade.rs                ← Trade-Struktur
│   │       ├── position.rs             ← Position-Struktur
│   │       ├── config.rs               ← BacktestConfig
│   │       ├── result.rs               ← BacktestResult
│   │       ├── timeframe.rs            ← Timeframe-Enum
│   │       └── error.rs                ← CoreError
│   │
│   ├── data/                           ← [CRATE] Data Loading
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── loader.rs               ← Parquet-Laden
│   │       ├── alignment.rs            ← Bid/Ask Alignment
│   │       ├── store.rs                ← CandleStore
│   │       ├── validation.rs           ← Datenqualitäts-Checks
│   │       ├── market_hours.rs         ← Market Hours Filter
│   │       └── error.rs                ← DataError
│   │
│   ├── indicators/                     ← [CRATE] Indikator-Engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── traits.rs               ← Indicator Trait
│   │       ├── cache.rs                ← IndicatorCache
│   │       ├── registry.rs             ← IndicatorRegistry
│   │       ├── error.rs                ← IndicatorError
│   │       └── impl/
│   │           ├── mod.rs              ← Implementierungs-Modul
│   │           ├── ema.rs              ← EMA-Indikator
│   │           ├── sma.rs              ← SMA-Indikator
│   │           ├── atr.rs              ← ATR-Indikator
│   │           ├── bollinger.rs        ← Bollinger Bands
│   │           ├── z_score.rs          ← Z-Score
│   │           ├── kalman.rs           ← Kalman Filter
│   │           └── garch.rs            ← GARCH Volatility
│   │
│   ├── execution/                      ← [CRATE] Order-Ausführung
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── engine.rs               ← ExecutionEngine
│   │       ├── slippage.rs             ← SlippageModel Trait + Implementierungen
│   │       ├── fees.rs                 ← FeeModel Trait + Implementierungen
│   │       ├── fill.rs                 ← Order Fill Logic
│   │       └── error.rs                ← ExecutionError
│   │
│   ├── portfolio/                      ← [CRATE] Portfolio Management
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── portfolio.rs            ← Portfolio-Hauptstruktur
│   │       ├── position_manager.rs     ← Position-Lifecycle
│   │       ├── equity.rs               ← Equity Tracking
│   │       ├── stops.rs                ← Stop-Loss/Take-Profit
│   │       └── error.rs                ← PortfolioError
│   │
│   ├── strategy/                       ← [CRATE] Strategie-Interface
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── traits.rs               ← Strategy Trait
│   │       ├── context.rs              ← BarContext
│   │       ├── registry.rs             ← StrategyRegistry
│   │       ├── error.rs                ← StrategyError
│   │       └── impl/
│   │           ├── mod.rs              ← Implementierungs-Modul
│   │           └── mean_reversion_z_score.rs  ← Mean Reversion Strategie
│   │
│   ├── backtest/                       ← [CRATE] Event Loop + Orchestrierung
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── engine.rs               ← BacktestEngine
│   │       ├── runner.rs               ← Backtest Runner
│   │       ├── warmup.rs               ← Warmup-Handling
│   │       ├── event_loop.rs           ← Haupt-Event-Loop
│   │       └── error.rs                ← BacktestError
│   │   └── tests/
│   │       ├── integration_test.rs     ← End-to-End Tests
│   │       └── fixtures/               ← Test-Daten
│   │
│   ├── metrics/                        ← [CRATE] Performance-Metriken
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← Re-exports
│   │       ├── compute.rs              ← compute_metrics()
│   │       ├── sharpe.rs               ← Sharpe Ratio
│   │       ├── sortino.rs              ← Sortino Ratio
│   │       ├── drawdown.rs             ← Max Drawdown
│   │       ├── profit_factor.rs        ← Profit Factor
│   │       ├── win_rate.rs             ← Win Rate
│   │       └── advanced.rs             ← Weitere Metriken
│   │
│   └── ffi/                            ← [CRATE] Python-Binding (PyO3)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs                  ← PyO3 Module Definition
│           ├── entry.rs                ← run_backtest() FFI Entry Point
│           ├── config.rs               ← Config JSON Parsing
│           ├── result.rs               ← Result JSON Serialization
│           └── error.rs                ← FFI Error Handling
│
└── python/
    └── bt/                             ← Python Package (dünner Wrapper)
        ├── __init__.py                 ← Package Init + FFI Import
        ├── runner.py                   ← Python Backtest Runner
        ├── config.py                   ← Config Loading + Validation
        ├── report.py                   ← Visualisierung + Reporting
        └── py.typed                    ← Type Stub Marker
```

---

## 3. Detaillierte Datei-Beschreibungen

### 3.1 Crate: `types` (Fundament)

> **Verantwortung**: Definiert alle gemeinsam genutzten Datenstrukturen. Hat KEINE Abhängigkeiten zu anderen Omega-Crates.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports aller Module, Crate-Dokumentation | `pub use` für alle öffentlichen Typen |
| `candle.rs` | Repräsentation einer Kurs-Kerze (OHLCV) | `Candle { timestamp_ns, open, high, low, close, volume }` |
| `signal.rs` | Trading-Signale von Strategien | `enum Signal { Long, Short, Exit, None }` |
| `trade.rs` | Abgeschlossener Trade mit allen Details | `Trade { entry_time, exit_time, pnl, fees, ... }` |
| `position.rs` | Offene Position im Portfolio | `Position { direction, entry_price, size, sl, tp, ... }` |
| `config.rs` | Backtest-Konfiguration (aus JSON) | `BacktestConfig { symbol, timeframe, dates, params, ... }` |
| `result.rs` | Backtest-Ergebnis (zu JSON) | `BacktestResult { trades, metrics, equity_curve, ... }` |
| `timeframe.rs` | Timeframe-Definitionen | `enum Timeframe { M1, M5, M15, M30, H1, H4, D1, W1 }` |
| `error.rs` | Gemeinsamer Error-Typ | `enum CoreError { Config, Data, Indicator, ... }` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
```

**Zeit-/Timestamp-Contract (Vorschlag A):**

- `timestamp_ns` ist ein `i64` in **epoch-nanoseconds (UTC)**.
- Für Candles ist `timestamp_ns` die **Open-Time** (Beginn der Kerze).
- Zeitreihen sind **strictly monotonic increasing** und **unique** (keine Duplikate) pro Datei/Stream.

---

### 3.2 Crate: `data` (Daten-Layer)

> **Verantwortung**: Lädt Marktdaten aus Parquet-Dateien, führt Bid/Ask Alignment durch und validiert Datenqualität.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports, Modul-Koordination | `pub use loader::*, store::*, alignment::*` |
| `loader.rs` | Parquet-Dateien laden (`arrow-rs`) | `load_candles(path: &Path) -> Result<Vec<Candle>>` |
| `alignment.rs` | Bid/Ask Timestamp-Alignment (Inner Join) | `align_bid_ask(bid: &[Candle], ask: &[Candle]) -> AlignedData` |
| `store.rs` | Speichert geladene Daten, Multi-TF Zugriff | `CandleStore { bid: Vec<Candle>, ask: Vec<Candle>, ... }` |
| `validation.rs` | Datenqualitäts-Checks (monoton, keine NaN, etc.) | `validate_candles(candles: &[Candle]) -> ValidationResult` |
| `market_hours.rs` | Filtert Candles außerhalb Handelszeiten | `filter_market_hours(candles: &[Candle], hours: &MarketHours)` |
| `alt_data.rs` | Alternative Data Store (z.B. News) | `AltDataStore { news: Option<NewsCalendarIndex>, ... }` |
| `news.rs` | News Calendar laden + Index/Mask erzeugen | `load_news(path) -> NewsCalendarIndex` |
| `error.rs` | Datenladen-spezifische Fehler | `enum DataError { FileNotFound, ParseError, AlignmentError, ... }` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
arrow = "51"
parquet = "51"
```

---

### 3.3 Crate: `indicators` (Indikator-Engine)

> **Verantwortung**: Berechnet technische Indikatoren vektorisiert, cached Ergebnisse, stellt Registry für dynamische Registrierung bereit.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports, Modul-Koordination | `pub use traits::*, cache::*, registry::*` |
| `traits.rs` | Indicator Trait Definition | `trait Indicator { fn compute(&self, candles: &[Candle]) -> Vec<f64>; }` |
| `cache.rs` | Caching berechneter Indikatoren | `IndicatorCache { cache: HashMap<IndicatorKey, Vec<f64>> }` |
| `registry.rs` | Dynamische Indikator-Registrierung | `IndicatorRegistry::register(name, factory_fn)` |
| `error.rs` | Indikator-spezifische Fehler | `enum IndicatorError { InsufficientData, InvalidParams, ... }` |
| `impl/mod.rs` | Sub-Modul für Implementierungen | Re-exports aller Indikatoren |
| `impl/ema.rs` | Exponential Moving Average | `struct EMA { period: usize }` |
| `impl/sma.rs` | Simple Moving Average | `struct SMA { period: usize }` |
| `impl/atr.rs` | Average True Range | `struct ATR { period: usize }` |
| `impl/bollinger.rs` | Bollinger Bands | `struct BollingerBands { period, std_dev }` |
| `impl/z_score.rs` | Z-Score Berechnung | `struct ZScore { window: usize }` |
| `impl/kalman.rs` | Kalman Filter | `struct KalmanFilter { r, q }` |
| `impl/garch.rs` | GARCH Volatilitätsmodell | `struct Garch { alpha, beta, omega }` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
```

---

### 3.4 Crate: `execution` (Order-Ausführung)

> **Verantwortung**: Simuliert realistische Order-Ausführung inkl. Slippage und Gebühren.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports | `pub use engine::*, slippage::*, fees::*, costs::*` |
| `engine.rs` | Execution Engine Orchestrierung | `ExecutionEngine::execute(signal, portfolio)` |
| `slippage.rs` | Slippage-Modelle | `trait SlippageModel`, `FixedSlippage`, `VolatilitySlippage` |
| `fees.rs` | Gebühren-Modelle | `trait FeeModel`, `PercentageFee`, `FixedFee`, `TieredFee` |
| `costs.rs` | YAML→Kostenmodell (Configs) | `load_costs(execution_costs, symbol_specs) -> ExecutionCosts` |
| `fill.rs` | Order Fill Logik | `compute_fill_price(order, bid, ask, slippage)` |
| `error.rs` | Execution-spezifische Fehler | `enum ExecutionError { InsufficientBalance, ... }` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
```

---

### 3.5 Crate: `portfolio` (Portfolio Management)

> **Verantwortung**: Verwaltet Cash, Positionen, Equity-Kurve und Stop-Management.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports | `pub use portfolio::Portfolio` |
| `portfolio.rs` | Haupt-Portfolio-Struktur | `Portfolio { cash, positions, equity_history }` |
| `position_manager.rs` | Position-Lifecycle (Open/Close/Modify) | `open_position()`, `close_position()`, `modify_stops()` |
| `equity.rs` | Equity Tracking über Zeit | `update_equity(price)`, `get_equity_curve()` |
| `stops.rs` | Stop-Loss / Take-Profit Prüfung | `check_stops(bid, ask) -> Vec<ClosedPosition>` |
| `error.rs` | Portfolio-spezifische Fehler | `enum PortfolioError { MaxPositionsReached, ... }` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
omega_execution = { path = "../execution" }
```

---

### 3.6 Crate: `strategy` (Strategie-Interface)

> **Verantwortung**: Definiert Strategy Trait, BarContext für Datenzugriff, Registry für Strategien und konkrete Strategie-Implementierungen.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports | `pub use traits::Strategy, context::BarContext` |
| `traits.rs` | Strategy Trait Definition | `trait Strategy { fn on_bar(&mut self, ctx: &BarContext) -> Option<Signal>; }` |
| `context.rs` | Bar Context (read-only Snapshot) | `BarContext { idx, candles, indicators, htf_data, session_open, news_blocked }` |
| `registry.rs` | Strategie-Registrierung | `StrategyRegistry::register(name, factory_fn)` |
| `error.rs` | Strategie-spezifische Fehler | `enum StrategyError { ConfigError, ... }` |
| `impl/mod.rs` | Sub-Modul für Implementierungen | Re-exports |
| `impl/mean_reversion_z_score.rs` | Mean Reversion Z-Score Strategie | `struct MeanReversionZScore { params, state }` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
omega_indicators = { path = "../indicators" }
omega_portfolio = { path = "../portfolio" }
```

---

### 3.7 Crate: `backtest` (Event Loop)

> **Verantwortung**: Orchestriert den gesamten Backtest – lädt Daten, berechnet Indikatoren, führt Event Loop aus, sammelt Ergebnisse.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports | `pub use engine::BacktestEngine` |
| `engine.rs` | Backtest Engine Hauptstruktur | `BacktestEngine::new(config)`, `run() -> BacktestResult` |
| `runner.rs` | High-Level Runner | `run_backtest(config) -> Result<BacktestResult>` |
| `warmup.rs` | Warmup-Handling | `calculate_warmup_bars(config)` |
| `event_loop.rs` | Haupt-Event-Loop | `for idx in warmup..len { ... }` |
| `error.rs` | Backtest-spezifische Fehler | `enum BacktestError { DataError, StrategyError, ... }` |
| `tests/integration_test.rs` | End-to-End Integration Tests | Full Backtest Tests |
| `tests/fixtures/` | Test-Daten (kleine Parquets) | Fixture-Dateien |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
omega_data = { path = "../data" }
omega_indicators = { path = "../indicators" }
omega_execution = { path = "../execution" }
omega_portfolio = { path = "../portfolio" }
omega_strategy = { path = "../strategy" }
omega_metrics = { path = "../metrics" }
```

---

### 3.8 Crate: `metrics` (Performance-Metriken)

> **Verantwortung**: Berechnet alle Performance-Metriken aus Trade-Liste und Equity-Kurve.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | Re-exports | `pub use compute::compute_metrics` |
| `compute.rs` | Zentrale Metrik-Berechnung | `compute_metrics(trades, equity) -> Metrics` |
| `sharpe.rs` | Sharpe Ratio Berechnung | `compute_sharpe(returns, risk_free_rate)` |
| `sortino.rs` | Sortino Ratio Berechnung | `compute_sortino(returns, risk_free_rate)` |
| `drawdown.rs` | Max Drawdown Berechnung | `compute_max_drawdown(equity_curve)` |
| `profit_factor.rs` | Profit Factor | `compute_profit_factor(trades)` |
| `win_rate.rs` | Win Rate | `compute_win_rate(trades)` |
| `advanced.rs` | Weitere Metriken (Calmar, MAR, etc.) | Zusätzliche Metriken |

**Abhängigkeiten (Cargo.toml)**:
```toml
[dependencies]
omega_types = { path = "../types" }
```

---

### 3.9 Crate: `ffi` (Python-Binding)

> **Verantwortung**: Stellt die PyO3-basierte FFI-Schnittstelle bereit. Einziger Entry Point für Python.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `lib.rs` | PyO3 Module Definition | `#[pymodule] fn omega_bt(...)` |
| `entry.rs` | FFI Entry Point | `#[pyfunction] fn run_backtest(config_json: &str) -> PyResult<String>` |
| `config.rs` | JSON → BacktestConfig | `parse_config(json: &str) -> Result<BacktestConfig>` |
| `result.rs` | BacktestResult → JSON | `serialize_result(result: BacktestResult) -> String` |
| `error.rs` | FFI Error Handling | `impl From<CoreError> for PyErr` |

**Abhängigkeiten (Cargo.toml)**:
```toml
[lib]
name = "omega_bt"
crate-type = ["cdylib"]

[dependencies]
omega_types = { path = "../types" }
omega_backtest = { path = "../backtest" }
pyo3 = { version = "0.20", features = ["extension-module"] }
serde_json = "1.0"
```

---

### 3.10 Python Package: `bt`

> **Verantwortung**: Dünner Python-Wrapper für den Rust-Kern. Lädt Configs, ruft FFI auf, generiert Reports.

| Datei | Verantwortlichkeit | Wichtigste Exports |
|-------|-------------------|-------------------|
| `__init__.py` | Package Init, importiert FFI-Modul | `from .runner import run_backtest` |
| `runner.py` | Python Backtest Runner | `run_backtest(config_path: str) -> BacktestResult` |
| `config.py` | Config Loading + Validation | `load_config(path)`, `validate_config(config)` |
| `report.py` | Visualisierung + Reporting | `generate_report(result)`, `plot_equity(result)` |
| `py.typed` | Type Stub Marker | (leer, für mypy/pyright) |

---

## 4. Modul-Abhängigkeits-Matrix

### 4.1 Crate-zu-Crate Abhängigkeiten

```
              ┌─────────────────────────────────────────────────────────────────────┐
              │                        ABHÄNGIGKEITS-MATRIX                          │
              │                                                                       │
              │    Zeile nutzt Spalte (→ = abhängig von)                             │
              └─────────────────────────────────────────────────────────────────────┘

                  types   data   indicators  execution  portfolio  strategy  backtest  metrics  ffi
              ┌────────┬───────┬────────────┬──────────┬──────────┬─────────┬─────────┬────────┬─────┐
    types     │   -    │       │            │          │          │         │         │        │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    data      │   ●    │   -   │            │          │          │         │         │        │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    indicators│   ●    │       │     -      │          │          │         │         │        │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    execution │   ●    │       │            │    -     │          │         │         │        │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    portfolio │   ●    │       │            │    ●     │    -     │         │         │        │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    strategy  │   ●    │       │     ●      │          │    ●     │    -    │         │        │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    backtest  │   ●    │   ●   │     ●      │    ●     │    ●     │    ●    │    -    │   ●    │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    metrics   │   ●    │       │            │          │          │         │         │   -    │     │
              ├────────┼───────┼────────────┼──────────┼──────────┼─────────┼─────────┼────────┼─────┤
    ffi       │   ●    │       │            │          │          │         │    ●    │        │  -  │
              └────────┴───────┴────────────┴──────────┴──────────┴─────────┴─────────┴────────┴─────┘

              ● = direkte Abhängigkeit
```

### 4.2 Abhängigkeits-Hierarchie (Visualisierung)

```
                                    LAYER 0 (Fundament)
                    ┌─────────────────────────────────────────┐
                    │                 types                    │
                    │    Candle, Signal, Trade, Position,      │
                    │    BacktestConfig, BacktestResult        │
                    └────────────────────┬────────────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
            ▼                            ▼                            ▼
                                    LAYER 1 (Basis-Services)
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │     data      │        │  indicators   │        │   execution   │
    │               │        │               │        │               │
    │  Parquet Load │        │  EMA, ATR,    │        │  Slippage,    │
    │  Alignment    │        │  Bollinger    │        │  Fees         │
    └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
            │                        │                        │
            │                        │                        │
            │                        ▼                        │
            │                                                 │
            │                   LAYER 2 (State Management)    │
            │            ┌───────────────────────┐            │
            │            │      portfolio        │◀───────────┘
            │            │                       │
            │            │  Cash, Positions,     │
            │            │  Equity, Stops        │
            │            └───────────┬───────────┘
            │                        │
            │                        ▼
            │
            │                   LAYER 3 (Business Logic)
            │            ┌───────────────────────┐
            │            │       strategy        │
            │            │                       │
            │            │  Strategy Trait,      │
            │            │  MeanReversionZScore  │
            │            └───────────┬───────────┘
            │                        │
            └────────────────────────┼────────────────────────┐
                                     │                        │
                                     ▼                        │
                                LAYER 4 (Orchestrierung)      │
                         ┌───────────────────────┐            │
                         │       backtest        │◀───────────┘
                         │                       │
                         │  BacktestEngine,      │
                         │  Event Loop           │
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
             ▼                                               ▼
                                LAYER 5 (Output/Interface)
     ┌───────────────┐                               ┌───────────────┐
     │    metrics    │                               │      ffi      │
     │               │                               │               │
     │  Sharpe,      │                               │  PyO3,        │
     │  Sortino,     │                               │  run_backtest │
     │  Drawdown     │                               │  JSON I/O     │
     └───────────────┘                               └───────────────┘
```

---

## 5. Datei-interne Abhängigkeiten

### 5.1 Crate `types` (Interne Struktur)

```
lib.rs ──────────────┬──────────────────────────────────────────┐
                     │                                          │
      ┌──────────────┼──────────────┬──────────────┐           │
      ▼              ▼              ▼              ▼           ▼
  candle.rs     signal.rs      trade.rs     position.rs   timeframe.rs
      │              │              │              │           │
      └──────────────┴──────┬───────┴──────────────┘           │
                            ▼                                   │
                       config.rs ◀──────────────────────────────┘
                            │
                            ▼
                       result.rs
                            │
                            ▼
                        error.rs (nutzt alle für Error-Varianten)
```

### 5.2 Crate `indicators` (Interne Struktur)

```
lib.rs
   │
   ├──► traits.rs          (Indicator Trait)
   │        │
   │        ▼
   ├──► cache.rs           (nutzt traits::Indicator)
   │        │
   │        ▼
   ├──► registry.rs        (nutzt traits::Indicator, cache)
   │
   └──► impl/
            │
            └──► mod.rs
                   │
                   ├──► ema.rs        (impl Indicator)
                   ├──► sma.rs        (impl Indicator)
                   ├──► atr.rs        (impl Indicator)
                   ├──► bollinger.rs  (impl Indicator, nutzt sma)
                   ├──► z_score.rs    (impl Indicator, nutzt sma)
                   ├──► kalman.rs     (impl Indicator)
                   └──► garch.rs      (impl Indicator)
```

### 5.3 Crate `backtest` (Interne Struktur)

```
lib.rs
   │
   ├──► engine.rs ─────────────────────┐
   │        │                          │
   │        ▼                          │
   ├──► runner.rs                      │
   │        │                          │
   │        └──────────────────────────┤
   │                                   │
   ├──► warmup.rs ◀────────────────────┤
   │                                   │
   ├──► event_loop.rs ◀────────────────┘
   │
   └──► error.rs (aggregiert alle Fehler)
```

---

## 6. Externe Dependencies pro Crate

### 6.1 Dependency-Übersicht

| Crate | Externe Dependencies | Zweck |
|-------|---------------------|-------|
| **types** | `serde`, `chrono` | Serialisierung, Zeitstempel |
| **data** | `arrow`, `parquet` | Parquet I/O |
| **indicators** | (keine) | Pure Rust Berechnungen |
| **execution** | (keine) | Pure Rust Logik |
| **portfolio** | (keine) | Pure Rust State |
| **strategy** | (keine) | Pure Rust Logik |
| **backtest** | `tracing` (optional) | Logging |
| **metrics** | (keine) | Pure Rust Berechnungen |
| **ffi** | `pyo3`, `serde_json` | Python-Binding |

### 6.2 Workspace Cargo.toml

```toml
[workspace]
members = [
    "crates/types",
    "crates/data",
    "crates/indicators",
    "crates/execution",
    "crates/portfolio",
    "crates/strategy",
    "crates/backtest",
    "crates/metrics",
    "crates/ffi",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Omega Team"]
license = "MIT"

[workspace.dependencies]
# Interne Crates
omega_types = { path = "crates/types" }
omega_data = { path = "crates/data" }
omega_indicators = { path = "crates/indicators" }
omega_execution = { path = "crates/execution" }
omega_portfolio = { path = "crates/portfolio" }
omega_strategy = { path = "crates/strategy" }
omega_backtest = { path = "crates/backtest" }
omega_metrics = { path = "crates/metrics" }

# Externe Crates
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
arrow = "51"
parquet = "51"
pyo3 = { version = "0.20", features = ["extension-module"] }
tracing = "0.1"
thiserror = "1.0"
```

---

## 7. Test-Strategie pro Crate

### 7.1 Test-Typen

| Crate | Unit Tests | Integration Tests | Property Tests |
|-------|-----------|-------------------|----------------|
| **types** | ✓ Serialisierung | - | ✓ Serde roundtrip |
| **data** | ✓ Loader, Alignment | ✓ Mit Fixtures | - |
| **indicators** | ✓ Jeder Indikator | - | ✓ Numerische Stabilität |
| **execution** | ✓ Fill, Slippage, Fees | - | - |
| **portfolio** | ✓ Position Lifecycle | - | - |
| **strategy** | ✓ Signal Generation | - | - |
| **backtest** | - | ✓ Full Backtest | - |
| **metrics** | ✓ Jede Metrik | - | ✓ Edge Cases |
| **ffi** | ✓ JSON Parsing | ✓ Python-Aufruf | - |

### 7.2 Test-Verzeichnisse

```
crates/
├── types/
│   └── src/
│       └── tests/          ← Unit Tests inline
│
├── data/
│   ├── src/
│   │   └── tests/          ← Unit Tests inline
│   └── tests/
│       └── fixtures/       ← Test-Parquet-Dateien
│
├── backtest/
│   └── tests/
│       ├── integration_test.rs
│       └── fixtures/       ← Full Backtest Test-Daten
│
└── ffi/
    └── tests/
        └── python_test.py  ← Python Integration Test
```

---

## 8. Build- und Release-Prozess

### 8.1 Build-Befehle

| Befehl | Zweck |
|--------|-------|
| `cargo build` | Debug Build aller Crates |
| `cargo build --release` | Optimierter Release Build |
| `cargo build -p omega_ffi --release` | Nur FFI-Crate bauen |
| `cargo test` | Alle Tests ausführen |
| `cargo test -p omega_indicators` | Tests nur für Indicators |
| `cargo clippy -- -D warnings` | Linting |
| `cargo fmt --check` | Format-Check |

### 8.2 Python-Package Build

```bash
# In rust_core/
cd crates/ffi
maturin develop --release    # Für Entwicklung
maturin build --release      # Für Distribution
```

---

## 9. Nächste Schritte

### Phase 1: Grundgerüst
- [ ] Workspace erstellen (`rust_core/Cargo.toml`)
- [ ] `types` Crate mit allen Structs
- [ ] `data` Crate mit Parquet-Loader
- [ ] Erste Tests mit Fixture-Daten

### Phase 2: Indikatoren
- [ ] `indicators` Crate mit Basis-Indikatoren
- [ ] Tests gegen Python-Referenzwerte
- [ ] Cache-Implementierung

### Phase 3: Execution + Portfolio
- [ ] `execution` Crate
- [ ] `portfolio` Crate
- [ ] Slippage/Fee-Modelle

### Phase 4: Strategy + Backtest
- [ ] `strategy` Crate mit Trait
- [ ] Mean Reversion Z-Score portieren
- [ ] `backtest` Crate mit Event Loop

### Phase 5: Metrics + FFI
- [ ] `metrics` Crate
- [ ] `ffi` Crate mit PyO3
- [ ] Python-Wrapper

### Phase 6: Validierung
- [ ] Vergleich Alt vs. Neu (identische Ergebnisse)
- [ ] Performance-Benchmark
- [ ] Dokumentation abschließen

---

## 10. Referenzen

- [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) – Übergeordneter Architekturplan
- [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) – Detaillierter Datenfluss
- [Rust Book](https://doc.rust-lang.org/book/) – Rust Dokumentation
- [PyO3 Guide](https://pyo3.rs/) – Python-Binding Dokumentation

---

*Dieser Plan dient als verbindliche Referenz für die Implementierung der Omega V2 Modul-Struktur.*
