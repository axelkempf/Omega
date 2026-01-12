---
title: "ADR-0006: Wave 4 - Pure Rust Strategy Execution"
status: Proposed
date: 2026-01-13
deciders:
  - Axel Kempf
consulted:
  - Omega Maintainers
---

# ADR-0006: Wave 4 - Pure Rust Strategy Execution

## Status

**Proposed** – Bereit für Review und Implementierung

## Kontext

### Problem: FFI-Overhead als Performance-Bottleneck

Wave 3 (Event Engine Rust Migration) hat gezeigt, dass der FFI-Overhead bei ~150.000 Aufrufen pro Backtest zum dominanten Performance-Faktor wird. Trotz erfolgreicher Migration der Event-Loop nach Rust wurden **keine signifikanten Speedups erreicht**, da die Strategie-Evaluierung weiterhin in Python läuft und pro Bar mehrere FFI-Crossings erzeugt.

### Performance-Baseline (Wave 3 Event Engine)

| Metrik | Python | Rust (Wave 3) | Diff |
|--------|--------|---------------|------|
| Avg Time | 34.0s | 36.3s | **0.94x (langsamer!)** |
| Peak RAM | 118.5 MB | 118.5 MB | 0 MB |
| FFI Calls/Backtest | 0 | ~150.000 | Bottleneck |

### Root Cause Analysis

Pro Bar (~30.000 Bars) werden folgende FFI-Crossings ausgeführt:

| Callback | Beschreibung | Calls/Bar |
|----------|--------------|-----------|
| `strategy.evaluate()` | Strategie-Evaluierung | 1 |
| `indicator_cache.*` | Indikator-Abfragen | 3-5 |
| `executor.process_signal()` | Signal-Verarbeitung | 0-1 |
| `portfolio.update_equity()` | Portfolio-Update | 1 |
| **Total** | | **~5/Bar** |

**Gesamt: 30.000 × 5 = ~150.000 FFI-Calls pro Backtest**

### Erfolgreiche Referenz: Wave 1 (IndicatorCache)

Wave 1 hat gezeigt, dass Pure-Rust-Implementierungen massive Speedups erreichen:

| Indikator | Python Baseline | Rust | Speedup |
|-----------|-----------------|------|---------|
| ATR | 954ms | 0.13ms | **7299x** |
| SMA | 212ms | 0.40ms | **528x** |
| EMA | 132ms | 0.39ms | **337x** |
| MACD | 171ms | 0.60ms | **285x** |
| **Gesamt** | 1,893ms | 4ms | **474x** |

## Entscheidung

### Pure Rust Strategy Execution

Wir implementieren einen **RustStrategy Trait**, der es ermöglicht, Strategien vollständig in Rust auszuführen. Die FFI-Crossings werden von ~150.000 auf **exakt 2** reduziert:

1. **Init**: Python → Rust (Strategie-Konfiguration)
2. **Run**: Rust führt alle 30.000 Bars aus
3. **Result**: Rust → Python (Trade-Ergebnisse)

### Architektur

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BACKTEST ENGINE                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │     Python API Layer (src/backtest_engine/core/event_engine.py)            │ │
│  │                                                                            │ │
│  │  class EventEngine:                                                        │ │
│  │      def run(self):                                                        │ │
│  │          if self._can_use_rust_strategy():                                 │ │
│  │              # ONLY 2 FFI CALLS                                            │ │
│  │              results = omega_rust.run_strategy(                            │ │
│  │                  strategy_config,    ◄── 1. Init                           │ │
│  │                  candle_data,                                              │ │
│  │              )                                                             │ │
│  │              self._process_results(results)  ◄── 2. Result                 │ │
│  │          else:                                                             │ │
│  │              self._run_python()                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                              │                                                   │
│                              │ FFI Boundary (PyO3) - nur 2 Crossings!           │
│                              ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │        Rust Layer (src/rust_modules/omega_rust/src/strategy/)              │ │
│  │                                                                            │ │
│  │  pub trait RustStrategy: Send + Sync {                                     │ │
│  │      fn evaluate(&self, slice: &DataSlice, cache: &IndicatorCache)         │ │
│  │          -> Option<TradeSignal>;                                           │ │
│  │      fn manage_position(&self, pos: &Position, slice: &DataSlice)          │ │
│  │          -> PositionAction;                                                │ │
│  │  }                                                                         │ │
│  │                                                                            │ │
│  │  // Vollständige Loop-Ausführung in Rust                                   │ │
│  │  for bar_index in 0..total_bars {                                          │ │
│  │      let slice = data.slice_at(bar_index);                                 │ │
│  │      if let Some(signal) = strategy.evaluate(&slice, &cache) {             │ │
│  │          executor.process_signal(signal);                                  │ │
│  │      }                                                                     │ │
│  │      executor.evaluate_exits(&slice);                                      │ │
│  │      portfolio.update_equity(slice.timestamp);                             │ │
│  │  }                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Integration mit bestehenden Rust-Modulen:                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ PortfolioRust│  │IndicatorCache│  │  Executor    │  │   Strategy   │        │
│  │   (Wave 2)   │  │   (Wave 1)   │  │  (Wave 4)    │  │   (Wave 4)   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### RustStrategy Trait Definition

```rust
/// Core strategy trait - alle Strategien implementieren diesen Trait
pub trait RustStrategy: Send + Sync {
    /// Evaluiert einen Bar und gibt optional ein Trade-Signal zurück
    fn evaluate(&self, slice: &DataSlice, cache: &IndicatorCache) -> Option<TradeSignal>;
    
    /// Verwaltet eine offene Position (Trailing Stop, Timeout, etc.)
    fn manage_position(&self, position: &Position, slice: &DataSlice) -> PositionAction;
    
    /// Gibt den primären Timeframe zurück
    fn primary_timeframe(&self) -> Timeframe;
    
    /// Optional: Custom Initialisierung
    fn on_init(&mut self, _config: &StrategyConfig) -> Result<(), StrategyError> {
        Ok(())
    }
}
```

### Datentypen

```rust
/// Repräsentiert einen Zeitpunkt im Backtest
pub struct DataSlice {
    pub symbol: String,
    pub timeframe: Timeframe,
    pub index: usize,
    pub timestamp: i64,
    pub bid: CandleData,
    pub ask: CandleData,
}

/// Trade-Signal vom Strategy
pub enum TradeSignal {
    Long {
        entry_price: f64,
        stop_loss: f64,
        take_profit: Option<f64>,
        reason: String,
        tags: Vec<String>,
    },
    Short {
        entry_price: f64,
        stop_loss: f64,
        take_profit: Option<f64>,
        reason: String,
        tags: Vec<String>,
    },
}

/// Position-Management-Aktion
pub enum PositionAction {
    Hold,
    ModifyStopLoss { new_sl: f64 },
    ModifyTakeProfit { new_tp: f64 },
    Close { reason: String },
}
```

### Feature-Flag-Steuerung

```python
# Environment Variable
OMEGA_USE_RUST_STRATEGY = "auto" | "true" | "false"

# auto (default): Rust wenn Strategie implementiert ist
# true: Rust erzwingen (Fehler wenn nicht verfügbar)
# false: Python erzwingen (für Debugging)
```

## Konsequenzen

### Positive Konsequenzen

| Metrik | Python Baseline | Rust Target | Erwarteter Speedup |
|--------|-----------------|-------------|-------------------|
| Full Backtest | 34.0s | 3.4s | **≥10x** |
| FFI Calls | 150.000 | 2 | **75.000x weniger** |
| Peak RAM | 118.5 MB | ~80 MB | **~30% weniger** |
| Durchsatz | ~1k bars/s | ~10k bars/s | **10x** |

### Negative Konsequenzen

1. **Entwicklungsaufwand**: Neue Strategien müssen in Rust implementiert werden
2. **Debugging-Komplexität**: Rust-Strategien schwerer zu debuggen
3. **Doppelte Codebasis**: Python-Referenz + Rust-Implementierung
4. **Lernkurve**: Team muss Rust lernen

### Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Trade-Parität verletzt | Mittel | Kritisch | Parity-Tests mit Golden-Files |
| Determinismus-Bruch | Niedrig | Kritisch | Seed-Propagation, Property-Tests |
| Komplexe Strategien schwer portierbar | Mittel | Mittel | Modulares Design, Scenario-by-Scenario |
| Python-Fallback langsamer als zuvor | Niedrig | Niedrig | Feature-Flag für Fallback |

## Alternativen

### Alternative 1: JIT-Kompilierung (Numba/PyPy)

- **Beschreibung**: Python-Strategien mit JIT kompilieren
- **Warum nicht gewählt**: 
  - Maximal 10-30x Speedup (nicht 75.000x FFI-Reduktion)
  - Nicht alle Python-Features unterstützt
  - Instabile Type-Inference bei komplexen Strategien

### Alternative 2: Batch-FFI (Arrow-basiert)

- **Beschreibung**: Alle Bars als Arrow-Batch an Rust, Python verarbeitet Batch
- **Warum nicht gewählt**:
  - Immer noch Python-Overhead in der Loop
  - Nur ~2-3x Speedup erreichbar
  - Komplexe Memory-Management-Probleme

### Alternative 3: Code-Generation (Python → Rust)

- **Beschreibung**: Automatische Transpilierung von Python-Strategien
- **Warum nicht gewählt**:
  - Hohe Komplexität bei dynamischen Python-Features
  - Debugging praktisch unmöglich
  - Fehlerhafte Transpilierung schwer erkennbar

### Alternative 4: WebAssembly (WASM)

- **Beschreibung**: Strategien in WASM kompilieren
- **Warum nicht gewählt**:
  - Zusätzliche Abstraktionsschicht
  - Geringerer Speedup als natives Rust
  - Debugging noch schwieriger

## Implementierte Dateien

### Rust-Module (src/rust_modules/omega_rust/src/strategy/)

| Datei | LOC | Beschreibung |
|-------|-----|--------------|
| `mod.rs` | ~50 | Modul-Exports |
| `traits.rs` | ~100 | RustStrategy Trait Definition |
| `types.rs` | ~200 | DataSlice, TradeSignal, PositionAction |
| `executor.rs` | ~300 | RustExecutor für Signal-Verarbeitung |
| `mean_reversion_zscore.rs` | ~800 | MeanReversionZScore Implementierung |
| `registry.rs` | ~100 | Strategy-Registry für Python-Lookup |
| **Gesamt** | ~1.550 | |

### Python-Integration (src/backtest_engine/)

| Datei | Änderung |
|-------|----------|
| `core/event_engine.py` | `_can_use_rust_strategy()`, `_run_rust_strategy()` |
| `core/rust_strategy_bridge.py` | NEU: Python↔Rust Bridge |
| `strategy/strategy_wrapper.py` | Rust-Strategy-Detection |

### Tests

| Datei | Beschreibung |
|-------|--------------|
| `tests/test_rust_strategy_parity.py` | Trade-Parity: Python vs Rust |
| `tests/benchmarks/test_bench_rust_strategy.py` | Performance-Benchmarks |
| `tests/golden/test_golden_rust_strategy.py` | Determinismus-Validierung |

## Usage

### Python-Konfiguration für Rust-Strategie

```python
# configs/backtest/mean_reversion_z_score.json
{
    "strategy": {
        "module": "strategies.mean_reversion_z_score.backtest.backtest_strategy",
        "class": "MeanReversionZScoreStrategy",
        "rust_enabled": true,  # NEU: Aktiviert Rust-Backend
        "init_args": {
            "symbol": "EURUSD",
            "timeframe": "M5",
            "enabled_scenarios": [1, 2],
            "direction_filter": "both"
        }
    }
}
```

### Rust-Strategie registrieren

```rust
// src/rust_modules/omega_rust/src/strategy/registry.rs

pub fn register_strategies(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MeanReversionZScoreRust>()?;
    // Weitere Strategien hier registrieren
    Ok(())
}
```

### Python-API aufrufen

```python
from omega_rust import MeanReversionZScoreRust, run_backtest_rust

# Strategie-Konfiguration
config = MeanReversionZScoreRust.from_config({
    "symbol": "EURUSD",
    "timeframe": "M5",
    "atr_length": 14,
    "z_score_long": -2.5,
    "z_score_short": 2.5,
    "enabled_scenarios": [1, 2],
})

# Backtest ausführen (nur 2 FFI-Calls!)
results = run_backtest_rust(
    strategy=config,
    candle_data=arrow_data,  # Arrow IPC Format
)

# Ergebnisse verarbeiten
for trade in results.trades:
    print(f"Trade: {trade.direction} @ {trade.entry_price}")
```

## Referenzen

- [ADR-0005: Wave 1 IndicatorCache Migration](./ADR-0005-wave1-indicator-cache-rust-migration.md)
- [FFI Specification: rust_strategy.md](../ffi/rust_strategy.md)
- [Performance Baseline: Event Engine](../../reports/performance_baselines/p0-01_event_engine.json)
- [Wave 3 Learnings](../WAVE_3_MIGRATION_LEARNINGS.md)
- [Migration Runbook: Pure Rust Strategies](../runbooks/pure_rust_strategy_migration.md)

## Änderungshistorie

| Datum | Autor | Änderung |
|-------|-------|----------|
| 2026-01-13 | AI Agent (Claude Opus 4.5) | Initiale Version |
