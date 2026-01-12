# Omega V2 – Architektur-Zielbericht

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Übergeordneter Architekturplan für die Neugestaltung des Backtesting-Kerns  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | **Vision, strategische Ziele, Erfolgskriterien** |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Detaillierter Datenfluss, Phasen, Validierung |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Datei- und Verzeichnisstruktur, Interfaces |

---

## 1. Zielbild

### 1.1 Warum dieser Bericht existiert

Das aktuelle Omega-System leidet unter folgenden strukturellen Problemen:

| Problem | Auswirkung |
|---------|------------|
| **Unübersichtliche Rust-Integration** | FFI-Grenzen überall verstreut, jede Komponente hat eigene Feature-Flags |
| **Hoher FFI-Overhead** | Tausende Python↔Rust Calls pro Backtest (pro Bar, pro Indikator) |
| **Monolithischer Code** | `runner.py` mit 3200+ Zeilen, schwer zu testen und erweitern |
| **Ineffizienter Datenfluss** | Mehrfache Konvertierung: Python List → numpy → Rust |
| **Vermischte Zuständigkeiten** | Keine klaren Modul-Grenzen |

### 1.2 Ziel der Neugestaltung

**Vision**: Ein hochperformantes, modulares Backtesting-System mit:

- **Rust als Kern**: Alle performance-kritischen Komponenten in Rust
- **Python als Orchestrator**: Dünne Schicht für Config-Management und Reporting
- **Single FFI Boundary**: Nur EIN Aufruf pro Backtest (Config rein → Result raus)
- **Zero-Copy Datenfluss**: Daten einmal nach Rust, dort bleiben sie
- **Klare Modularität**: Eigenständige Crates mit definierten Schnittstellen

### 1.3 Scope-Abgrenzung

| In Scope | Out of Scope (vorerst) |
|----------|------------------------|
| Backtesting-Kern neu schreiben | Live-Trading-Pfad (bleibt unverändert) |
| Single-Symbol Backtests | Multi-Symbol Backtests (später) |
| Mean Reversion Z-Score Strategie | Weitere Strategien (später) |
| Rust-native Strategien | Python-Strategien |

---

## 2. Angestrebter Datenfluss

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PYTHON LAYER                                    │
│                         (Dünner Orchestrator)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. Config laden (JSON)                                                     │
│   2. Parquet-Pfade bestimmen                                                 │
│   3. Strategie-Parameter extrahieren                                         │
│   4. EINMALIGER FFI-CALL → Rust-Engine starten                              │
│   5. Ergebnis empfangen (JSON)                                               │
│   6. Report generieren / Speichern                                           │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │  FFI: run_backtest(config_json) → result_json
                                   │  (EIN Call, EINE Grenze)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RUST LAYER                                      │
│                         (Gesamter Kern)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     DATA LOADER                                      │    │
│  │  • Parquet direkt lesen (arrow/polars)                              │    │
│  │  • Bid/Ask in Rust-Structs (Vec<Candle>)                            │    │
│  │  • Multi-TF Alignment (M1 → M5 → H1 → D1)                           │    │
│  │  • Market Hours Filter                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     INDICATOR ENGINE                                 │    │
│  │  • Pre-compute ALLE Indikatoren vor Loop                            │    │
│  │  • Vektorisiert (SIMD wo möglich)                                   │    │
│  │  • Indicator Registry (trait-basiert, erweiterbar)                  │    │
│  │  • Cache: HashMap<(Name, TF, Params), Vec<f64>>                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     EVENT LOOP                                       │    │
│  │  for idx in warmup..candles.len() {                                 │    │
│  │      ctx = BarContext::new(idx, &candles, &indicators);             │    │
│  │      signal = strategy.on_bar(&ctx);                                │    │
│  │      if let Some(sig) = signal {                                    │    │
│  │          executor.process(sig, &mut portfolio);                     │    │
│  │      }                                                               │    │
│  │      portfolio.check_stops(bid, ask);                               │    │
│  │      portfolio.update_equity(close_price);                          │    │
│  │  }                                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     RESULT BUILDER                                   │    │
│  │  • Metrics berechnen (Sharpe, Sortino, Drawdown, etc.)              │    │
│  │  • Trade-Liste serialisieren                                         │    │
│  │  • Equity Curve exportieren                                          │    │
│  │  • JSON-Result erstellen                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               │  Zurück zu Python: BacktestResult (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PYTHON LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   • Result deserialisieren                                                   │
│   • JSON speichern (Trades, Metriken)                                        │
│   • Equity Curve speichern                                                   │
│   • Visualisierung (matplotlib, optional)                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Vergleich: Alt vs. Neu

| Aspekt | Aktuell | Neu (Ziel) |
|--------|---------|------------|
| FFI-Calls pro Backtest | Tausende (pro Bar, pro Indikator) | **1** |
| Daten-Konvertierung | Python List → numpy → Rust (mehrfach) | **Einmal (Parquet → Rust)** |
| Candle-Repräsentation | Python-Klasse (teuer) | **Rust struct (günstig)** |
| Indikator-Berechnung | Pro Abruf, mit FFI | **Pre-computed, Rust-nativ** |
| Strategy-Aufruf | Python-Funktion pro Bar | **Rust-Methode pro Bar** |
| Event Loop | Python for-loop | **Rust for-loop** |

---

## 3. Modulare Architektur

### 3.1 Crate-Struktur

```
rust_core/                     ← Workspace Root
├── Cargo.toml                 ← Workspace Definition
│
├── crates/
│   ├── types/                 ← Gemeinsame Datentypen (keine Abhängigkeiten)
│   ├── data/                  ← Data Loading (Parquet → Structs)
│   ├── indicators/            ← Indikator-Engine
│   ├── strategy/              ← Strategy Trait + Implementierungen
│   ├── execution/             ← Order-Ausführung, Slippage, Fees
│   ├── portfolio/             ← Portfolio State Management
│   ├── backtest/              ← Event Loop + Orchestrierung
│   ├── metrics/               ← Performance-Metriken Berechnung
│   └── ffi/                   ← Python-Binding (PyO3)
│
└── python/
    └── bt/                    ← Python Package (dünner Wrapper)
        ├── __init__.py
        ├── runner.py          ← Config → FFI → Result
        └── report.py          ← Visualisierung
```

### 3.2 Modul-Abhängigkeiten

```
                    ┌─────────────────┐
                    │      types      │  ← Keine Abhängigkeiten
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│      data       │ │   indicators    │ │    execution    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
              ┌─────────────────────────┐
              │       portfolio         │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │        strategy         │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │        backtest         │
              └───────────┬─────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│     metrics     │             │       ffi       │
└─────────────────┘             └─────────────────┘
```

### 3.3 Modul-Verantwortlichkeiten

| Modul | Verantwortung | Wichtigste Exports |
|-------|---------------|-------------------|
| **types** | Gemeinsame Datenstrukturen | `Candle`, `Signal`, `Trade`, `Position`, `BacktestConfig`, `BacktestResult` |
| **data** | Parquet lesen, TF-Alignment | `load_candles()`, `CandleStore`, `align_timeframes()` |
| **indicators** | Indikator-Berechnungen | `trait Indicator`, `IndicatorCache`, `IndicatorRegistry` |
| **execution** | Slippage, Fees, Order-Fill | `trait SlippageModel`, `trait FeeModel`, `ExecutionEngine` |
| **portfolio** | Cash, Positions, Equity | `Portfolio`, `open_position()`, `close_position()` |
| **strategy** | Strategy-Interface | `trait Strategy`, `BarContext`, `MeanReversionZScore` |
| **backtest** | Event Loop Orchestrierung | `BacktestEngine::run()` |
| **metrics** | Performance-Metriken | `compute_metrics()`, `Metrics` |
| **ffi** | Python-Binding | `run_backtest(config_json)` |

---

## 4. Systemregeln

### 4.1 Architektur-Regeln

| # | Regel | Begründung |
|---|-------|------------|
| R1 | **Einweg-Abhängigkeiten** | Modul A darf B nutzen ⟺ B ist tiefer in der Hierarchie. Keine Zyklen! |
| R2 | **Klare Schnittstellen** | Nur `pub` für Structs, Traits, Factory-Funktionen. Interna bleiben `pub(crate)` |
| R3 | **Kein Cross-Cutting** | `data` kennt nicht `Portfolio`, `indicators` kennt nicht `Strategy` |
| R4 | **Eigene Error-Typen** | Jedes Modul definiert eigene Fehler, `types` hat gemeinsamen `CoreError` |
| R5 | **Keine Globals** | Kein `lazy_static!`, kein `thread_local!`, kein `static mut` |
| R6 | **Explizite Abhängigkeiten** | Alles wird durchgereicht (Dependency Injection) |

### 4.2 Code-Qualitäts-Regeln

| # | Regel | Umsetzung |
|---|-------|-----------|
| Q1 | **Jedes Modul testbar** | Eigenes `tests/` Verzeichnis pro Crate |
| Q2 | **Unit-Tests isoliert** | Testen nur eigenes Modul, keine externen Abhängigkeiten |
| Q3 | **Integration-Tests zentral** | Nur in `backtest/tests/` |
| Q4 | **Dokumentation pflicht** | `///` für alle `pub` Items |
| Q5 | **Clippy clean** | `cargo clippy -- -D warnings` |
| Q6 | **Formatierung einheitlich** | `cargo fmt` |

### 4.3 FFI-Regeln

| # | Regel | Begründung |
|---|-------|------------|
| F1 | **Single Entry Point** | Nur `run_backtest()` als öffentliche FFI-Funktion |
| F2 | **JSON für Config/Result** | Einfache Serialisierung, human-readable |
| F3 | **Keine Python-Objekte in Rust** | Alles über JSON/Bytes, keine PyO3-Objekte halten |
| F4 | **Fehler als JSON** | Rust-Fehler werden als JSON-Error zurückgegeben |

### 4.4 Strategie-Regeln

| # | Regel | Begründung |
|---|-------|------------|
| S1 | **Strategie deklariert Indikatoren** | `required_indicators()` gibt alle benötigten Indikatoren zurück |
| S2 | **Strategie ist stateful** | Darf eigenen State halten (Position Manager, Filter, etc.) |
| S3 | **Strategie bekommt nur Snapshot** | `BarContext` ist read-only, keine Mutation von außen |
| S4 | **Registrierung via Factory** | `StrategyRegistry::register("name", factory_fn)` |

---

## 5. Offene Punkte / Zu Klären

### 5.1 Technische Entscheidungen

| # | Frage | Optionen | Status |
|---|-------|----------|--------|
| T1 | Parquet-Library | `arrow-rs` vs. `polars` | Offen |
| T2 | JSON-Library | `serde_json` (Standard) | Vorläufig entschieden |
| T3 | Parallelisierung | `rayon` für Indikator-Berechnung? | Offen |
| T4 | Warmup-Handling | Explizit in Config oder automatisch? | Offen |
| T5 | HTF-Daten | Separate Parquets oder aus M1 aggregieren? | Offen |

### 5.2 Strategie-Design

| # | Frage | Kontext |
|---|-------|---------|
| S1 | Wie viele Szenarien hat Mean Reversion Z-Score? | Aktuell 6 Szenarien, alle migrieren? |
| S2 | News-Filter in Rust? | Aktuell Python-basiert mit CSV |
| S3 | Position Manager Logik? | Max Holding Time, Trailing Stop, etc. |
| S4 | HTF-EMA Filter Details? | 2 Ebenen (D1 + auto HTF) |

### 5.3 Output-Format

| # | Frage | Aktuell | Ziel |
|---|-------|---------|------|
| O1 | Trade-Liste Format | JSON mit allen Details | Beibehalten |
| O2 | Equity Curve | Liste von Floats | Beibehalten |
| O3 | Metriken | ~30 verschiedene | Alle übernehmen? |
| O4 | Logging während Backtest | Python logging | Rust tracing? |

### 5.4 Migration

| # | Frage | Überlegung |
|---|-------|------------|
| M1 | Reihenfolge der Module | Bottom-up (types → data → ...) oder Top-down? |
| M2 | Parallel zum alten System? | Neues System in separatem Ordner entwickeln? |
| M3 | Validierung | Wie sicherstellen, dass Ergebnisse identisch sind? |
| M4 | Performance-Baseline | Aktuelles System benchmarken vor Migration? |

---

## 6. Nächste Schritte (Vorschlag)

### Phase 0: Vorbereitung
- [ ] Aktuelles System benchmarken (Baseline)
- [ ] Alle Szenarien der Mean Reversion Strategie dokumentieren
- [ ] JSON-Config Schema definieren
- [ ] JSON-Result Schema definieren

### Phase 1: Fundament
- [ ] `types` Crate erstellen
- [ ] `data` Crate erstellen (Parquet laden)
- [ ] Erste Tests: Daten laden und verifizieren

### Phase 2: Indikatoren
- [ ] `indicators` Crate erstellen
- [ ] Basis-Indikatoren implementieren (EMA, ATR, Bollinger)
- [ ] Tests gegen Python-Referenz

### Phase 3: Execution
- [ ] `execution` Crate erstellen
- [ ] `portfolio` Crate erstellen
- [ ] Slippage/Fee-Modelle implementieren

### Phase 4: Strategy
- [ ] `strategy` Crate erstellen
- [ ] Strategy Trait definieren
- [ ] Mean Reversion Z-Score portieren

### Phase 5: Integration
- [ ] `backtest` Crate erstellen
- [ ] `metrics` Crate erstellen
- [ ] End-to-End Test

### Phase 6: FFI
- [ ] `ffi` Crate erstellen
- [ ] Python-Wrapper
- [ ] Vergleich Alt vs. Neu

---

## 7. Anhang

### 7.1 Referenz: Aktuelle Datenquellen

| Quelle | Format | Zeitraum | Granularität |
|--------|--------|----------|--------------|
| Dukascopy | CSV → Parquet | 17-18 Jahre | Tick bis Monthly |
| Bid/Ask separat | Ja | - | - |

### 7.2 Referenz: Aktuelle Strategie-Parameter (Mean Reversion Z-Score)

```
atr_length, atr_mult, b_b_length, std_factor, window_length,
z_score_long, z_score_short, ema_length, kalman_r, kalman_q,
tp_min_distance, direction_filter, enabled_scenarios,
use_position_manager, max_holding_minutes,
htf_tf, htf_ema, htf_filter, extra_htf_tf, extra_htf_ema, extra_htf_filter,
garch_alpha, garch_beta, garch_omega, garch_use_log_returns, garch_scale,
garch_min_periods, garch_sigma_floor, local_z_lookback, intraday_vol_feature
```

### 7.3 Glossar

| Begriff | Definition |
|---------|------------|
| **FFI** | Foreign Function Interface – Schnittstelle zwischen Python und Rust |
| **Crate** | Rust-Paket/Modul |
| **Trait** | Rust-Interface (ähnlich wie Interface in anderen Sprachen) |
| **PyO3** | Rust-Library für Python-Bindings |
| **Zero-Copy** | Daten werden nicht kopiert, nur Referenzen übergeben |
| **HTF** | Higher Timeframe (z.B. H1, D1 relativ zu M5) |

---

*Dieser Bericht dient als Grundlage für detaillierte Umsetzungspläne der einzelnen Module.*
