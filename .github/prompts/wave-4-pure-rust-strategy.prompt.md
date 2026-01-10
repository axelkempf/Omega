# Wave 4: Pure Rust Strategy Implementation Plan

## Objective

Entwickle einen detaillierten Implementierungsplan für **Pure Rust Strategien**, um den FFI-Overhead zu eliminieren und maximale Performance zu erreichen.

---

## Context

### Projekt-Übersicht

Dieses Repository ist ein Python-basierter Trading-Stack mit:
- **Live-Engine**: MT5-Adapter, Risk-Layer, Execution (`src/hf_engine/`)
- **Backtest-Engine**: Event-driven Backtests, Optimizer (`src/backtest_engine/`)
- **UI-Engine**: FastAPI für Prozesssteuerung (`src/ui_engine/`)
- **Strategien**: Trading-Strategien in Python (`src/strategies/`)

### Bisherige Rust-Migrationen (Wave 0-3)

| Wave | Komponente | Status | Performance |
|------|------------|--------|-------------|
| Wave 0 | Slippage & Fee Models | ✅ Complete | ~5x schneller |
| Wave 1 | IndicatorCache | ✅ Complete | **148.6x schneller**, 70% weniger RAM |
| Wave 2 | Portfolio State | ✅ Complete | ~10x schneller |
| Wave 3 | Event Engine Loop | ✅ Complete | **Kein Speedup** (FFI-Overhead dominiert) |

### Das Problem: FFI-Overhead

Wave 3 zeigte, dass der Rust Event Loop **keinen Performance-Gewinn** bringt, wenn Strategien in Python bleiben:

```
Rust Loop → Python: strategy.evaluate()      # FFI-Call + GIL
Rust Loop → Python: simulator.process_signal() # FFI-Call + GIL
Rust Loop → Python: simulator.evaluate_exits() # FFI-Call + GIL
Rust Loop → Python: simulator.manage_positions() # FFI-Call + GIL
Rust Loop → Python: portfolio.update_equity()  # FFI-Call + GIL
```

Bei ~30,000 Bars = **~150,000 FFI-Aufrufe** pro Backtest.

### Die Lösung: Pure Rust Strategien

Wenn die Strategie-Logik in Rust läuft, reduziert sich der FFI-Overhead auf ~2 Aufrufe (Init + Result):

```
Python: config → Rust
Rust: Event Loop + Strategy + Indicators + Portfolio (alles in Rust)
Rust: results → Python
```

**Erwarteter Speedup: 10-50x** (basierend auf IndicatorCache-Benchmarks).

---

## Anforderungen

### Funktionale Anforderungen

1. **RustStrategy Trait**: Generische Schnittstelle für Rust-Strategien
2. **Indikator-Integration**: Zugriff auf IndicatorCache (bereits in Rust)
3. **Signal-Generierung**: TradeSignal-Struct für Entry/Exit-Signale
4. **Multi-Timeframe Support**: Zugriff auf M1, M5, H1, etc.
5. **Position Management**: SL/TP, Trailing, Break-Even
6. **Konfigurierbarkeit**: Parameter via JSON/YAML wie bei Python-Strategien

### Nicht-funktionale Anforderungen

1. **Backward Compatibility**: Python-Strategien müssen weiterhin funktionieren
2. **Determinismus**: Identische Ergebnisse wie Python-Implementierung
3. **Testbarkeit**: Golden-Tests gegen Python-Referenz
4. **Dokumentation**: Klare API-Doku für Strategie-Entwicklung

### Constraints

1. **PyO3 0.27**: Bestehende FFI-Infrastruktur nutzen
2. **Bestehende Rust-Module**: `omega_rust` Crate erweitern
3. **Feature Flags**: `OMEGA_USE_RUST_STRATEGY` Umgebungsvariable
4. **Kein Breaking Change**: Bestehende Backtests dürfen nicht brechen

---

## Referenz-Strategie: Mean Reversion Z-Score

Die erste Pure-Rust-Strategie soll `mean_reversion_z_score` sein:

### Python-Implementierung (Referenz)

Pfad: `src/strategies//mean_reversion_z_score_strategy.py`

Kernlogik:
- Kalman-Filter für geglätteten Preis
- Z-Score Berechnung über Lookback-Periode
- Entry bei Z-Score Threshold-Überschreitung
- Exit bei Mean-Reversion oder SL/TP

### Rust-Ziel

```rust
pub struct MeanReversionZScore {
    config: StrategyConfig,
    // ... Parameter
}

impl RustStrategy for MeanReversionZScore {
    fn evaluate(&self, slice: &DataSlice, cache: &IndicatorCache) -> Option<TradeSignal>;
    fn manage_position(&self, position: &Position, slice: &DataSlice) -> Option<PositionAction>;
}
```

---

## Bestehende Rust-Infrastruktur

### Relevante Dateien

- `src/rust_modules/omega_rust/src/lib.rs` - PyO3 Exports
- `src/rust_modules/omega_rust/src/indicators/` - IndicatorCache (148x schneller)
- `src/rust_modules/omega_rust/src/event/` - Event Engine Loop
- `src/rust_modules/omega_rust/src/portfolio/` - Portfolio State
- `src/rust_modules/omega_rust/src/slippage.rs` - Slippage Model
- `src/rust_modules/omega_rust/src/fees.rs` - Fee Model

### Indikator-Funktionen (bereits verfügbar)

```rust
// Alle bereits in Rust implementiert:
fn ema(&self, tf: &str, side: &str, period: usize) -> Vec<f64>;
fn sma(&self, tf: &str, side: &str, period: usize) -> Vec<f64>;
fn rsi(&self, tf: &str, side: &str, period: usize) -> Vec<f64>;
fn atr(&self, tf: &str, side: &str, period: usize) -> Vec<f64>;
fn bollinger(&self, tf: &str, side: &str, period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>);
fn zscore(&self, tf: &str, side: &str, period: usize) -> Vec<f64>;
fn kalman_zscore(&self, tf: &str, side: &str, lookback: usize, q: f64, r: f64) -> Vec<f64>;
// ... und mehr
```

---

## Erwartete Deliverables

### Phase 1: Design (1-2 Tage)

1. **ADR**: `docs/adr/ADR-XXXX-pure-rust-strategies.md`
2. **FFI-Spezifikation**: `docs/ffi/rust_strategy.md`
3. **API-Design**: `RustStrategy` Trait + Helper-Structs

### Phase 2: Core Implementation (3-5 Tage)

1. **Strategy Trait**: `src/rust_modules/omega_rust/src/strategy/mod.rs`
2. **DataSlice Struct**: Candle-Daten für Strategie-Auswertung
3. **TradeSignal Struct**: Entry/Exit-Signal mit Parametern
4. **PositionAction Enum**: SL/TP/Trailing/Close Aktionen
5. **Python Bindings**: PyO3 Wrapper für Rust-Strategien

### Phase 3: Reference Strategy (2-3 Tage)

1. **Mean Reversion Z-Score in Rust**: `src/rust_modules/omega_rust/src/strategy/mean_reversion_zscore.rs`
2. **Parity Tests**: Vergleich Python vs Rust Ergebnisse
3. **Golden Tests**: Determinismus-Validierung

### Phase 4: Integration (2-3 Tage)

1. **Runner Integration**: `src/backtest_engine/runner.py` Erweiterung
2. **Config Mapping**: JSON → Rust Strategy Parameter
3. **Feature Flag**: `OMEGA_USE_RUST_STRATEGY`
4. **Fallback**: Python-Strategy bei Rust-Fehler

### Phase 5: Benchmarks & Validation (1-2 Tage)

1. **Performance Benchmark**: `tools/benchmark_rust_strategy.py`
2. **Memory Tracking**: Peak RAM, Allocation-Pattern
3. **Documentation Update**: README, architecture.md

---

## Erfolgs-Kriterien

| Kriterium | Ziel | Messung |
|-----------|------|---------|
| **Performance** | ≥10x schneller | Benchmark-Tool |
| **Memory** | ≥30% weniger RAM | tracemalloc |
| **Parity** | 100% Trade-Match | Parity-Tests |
| **Determinismus** | Reproduzierbar | Golden-Tests |
| **Test Coverage** | ≥90% | pytest-cov |

---

## Risiken & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Indikator-Diskrepanz | Mittel | Hoch | Extensive Parity-Tests |
| Floating-Point-Differenzen | Hoch | Niedrig | Toleranz-basierte Vergleiche |
| Komplexe Exit-Logik | Mittel | Mittel | Schrittweise Migration |
| Config-Parsing Fehler | Niedrig | Mittel | Schema-Validierung |

---

## Fragen an den Plan-Ersteller

1. Soll die Rust-Strategy-API auch für **Live-Trading** nutzbar sein?
2. Welche weiteren Strategien sollen nach Mean Reversion migriert werden?
3. Soll ein **Strategy-Generator** (Python → Rust Transpiler) entwickelt werden?
4. Wie soll **Logging/Debugging** in Rust-Strategien funktionieren?
5. Sollen **Custom Indicators** in Rust unterstützt werden?

---

## Referenzen

- [Wave 3 Implementation Plan](../../docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md)
- [IndicatorCache Benchmark](../../tools/benchmark_indicator_cache.py)
- [Event Engine Benchmark](../../tools/benchmark_event_engine.py)
- [FFI Lessons Learned](../../docs/WAVE_3_MIGRATION_LEARNINGS.md)
- [Rust/Julia Migration Prep](../../docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md)
