# Omega V2 – Vision & Zielbild

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Übergeordnete Vision, strategische Ziele und Erfolgskriterien für die Omega V2 Architektur  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Übergeordneter Blueprint, Module, Regeln |
| [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) | Strategie-Spezifikation (MVP: Mean Reversion Z-Score), Szenarien 1–6, Guards/Filter, Indikatoren |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Detaillierter Datenfluss, Phasen, Validierung |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Data-Quality-Policies, Snapshots/Manifests, Fail-Fast Regeln |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell: Bid/Ask, Fills, SL/TP, Slippage/Fees |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Datei- und Verzeichnisstruktur, Interfaces |
| [OMEGA_V2_INDICATOR_CACHE__PLAN.md](OMEGA_V2_INDICATOR_CACHE__PLAN.md) | Indikator-Cache: Multi-TF, Stepwise-Semantik, V1-Parität |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Normative Metrik-Keys, Definitionen/Units, Scores, Rundung |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives JSON-Config-Schema (Felder, Defaults, Validierung, Migration) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (Artefakte, Schema, Zeit/Units, Pfade) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Packaging/Build-Matrix |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Logging/Tracing (tracing), Profiling (flamegraph/pprof), Performance-Counter, Determinismus |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Qualitätsstrategie, V1↔V2 Parität (DEV), Determinismus |
| [OMEGA_V2_FORMATTING_PLAN.md](OMEGA_V2_FORMATTING_PLAN.md) | Format-/Lint-/Docstring-Regeln (Code/Doku/Kommentare), Durchsetzung via pre-commit + CI |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI/CD Workflow, Quality Gates, Build-Matrix, Security, Release-Assets |

## 1. Zusammenfassung

Dieses Dokument definiert das **strategische Zielbild** für Omega V2. Es beantwortet die Fragen:
- **Warum** brauchen wir eine Neugestaltung?
- **Was** soll erreicht werden?
- **Wie** messen wir Erfolg?

Die technischen Details zur Umsetzung finden sich in den verlinkten Dokumenten im Abschnitt **Verwandte Dokumente**.

---

## 2. Problemanalyse: Status Quo

### 2.1 Aktueller Systemzustand

Das bestehende Omega-System ist organisch gewachsen und weist fundamentale strukturelle Schwächen auf:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AKTUELLER ZUSTAND (Omega V1)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    PYTHON LAYER (DOMINANT)                            │   │
│  │                                                                       │   │
│  │   runner.py (3200+ Zeilen)                                           │   │
│  │      │                                                                │   │
│  │      ├── Config laden                                                 │   │
│  │      ├── Daten laden (pandas)                                         │   │
│  │      ├── Event Loop (Python for-loop)  ◄──── BOTTLENECK              │   │
│  │      │      │                                                         │   │
│  │      │      ├── FFI: get_indicator()    ◄──── Tausende Calls         │   │
│  │      │      ├── FFI: calculate_z_score() ◄──── Tausende Calls        │   │
│  │      │      ├── Python: Strategy Logic                                │   │
│  │      │      └── FFI: execute_order()    ◄──── Tausende Calls         │   │
│  │      │                                                                │   │
│  │      └── Report generieren                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              │ Tausende FFI-Calls                           │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    RUST LAYER (FRAGMENTIERT)                          │   │
│  │                                                                       │   │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │   │
│  │   │indi_ffi│  │ exec_ffi│  │slipp_ffi│  │metrics  │                 │   │
│  │   │        │  │         │  │         │  │   _ffi  │                 │   │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘                 │   │
│  │       ▲             ▲            ▲            ▲                       │   │
│  │       │             │            │            │                       │   │
│  │   Eigene Feature-Flags, keine gemeinsame Architektur                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Identifizierte Kernprobleme

| # | Problem | Symptome | Business-Impact |
|---|---------|----------|-----------------|
| **P1** | **FFI-Overhead** | Tausende Python↔Rust Calls pro Backtest | ~60% der Laufzeit ist FFI-Crossing |
| **P2** | **Monolithischer Code** | `runner.py` mit 3200+ Zeilen | Änderungen riskant, Tests unmöglich |
| **P3** | **Daten-Kopien** | Python List → numpy → Rust (mehrfach) | Memory-Spikes, GC-Pressure |
| **P4** | **Verstreute FFI-Grenzen** | Jede Komponente eigene Bindings | Inkonsistente Fehlerbehandlung |
| **P5** | **Vermischte Zuständigkeiten** | Strategy + Execution + Metrics in einem File | Keine Separation of Concerns |
| **P6** | **Python Event Loop** | `for candle in candles` in Python | ~100x langsamer als Rust-Loop |

### 2.3 Quantifizierte Performance-Baseline

| Metrik | Aktueller Wert | Ziel (Omega V2) |
|--------|----------------|-----------------|
| FFI-Calls pro Backtest (10k Candles) | ~50.000 | **1** |
| Backtest-Laufzeit (10k Candles) | ~8-12 Sekunden | **< 500ms** |
| Memory Peak | ~500MB | **< 100MB** |
| Code-Zeilen in `runner.py` | 3.200+ | **< 100** (nur Orchestration) |

---

## 3. Vision: Omega V2 Zielbild

### 3.1 Leitprinzipien

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OMEGA V2 VISION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     SINGLE FFI BOUNDARY                              │   │
│   │                                                                      │   │
│   │   Python: "Führe Backtest mit dieser Config aus"                    │   │
│   │      │                                                               │   │
│   │      │  run_backtest(config_json) ──────────────────────────────▶   │   │
│   │      │                                                               │   │
│   │      │  ◀─────────────────────────────────── result_json            │   │
│   │      │                                                               │   │
│   │   Python: "Hier ist das Ergebnis"                                   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     RUST-NATIVE EXECUTION                            │   │
│   │                                                                      │   │
│   │   • Daten laden (Parquet → Vec<Candle>)                             │   │
│   │   • Indikatoren vorberechnen (vektorisiert, SIMD)                   │   │
│   │   • Event Loop (tight Rust for-loop)                                │   │
│   │   • Strategy Execution (trait-basiert)                              │   │
│   │   • Portfolio Management (keine GC-Pause)                           │   │
│   │   • Metriken berechnen                                              │   │
│   │   • Ergebnis serialisieren                                          │   │
│   │                                                                      │   │
│   │   ALLES in Rust, KEINE Rückrufe nach Python                         │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     MODULARE ARCHITEKTUR                             │   │
│   │                                                                      │   │
│   │   types ── data ── indicators ── execution                          │   │
│   │              │                       │                               │   │
│   │              └─────── portfolio ─────┘                               │   │
│   │                          │                                           │   │
│   │                      strategy                                        │   │
│   │                          │                                           │   │
│   │                      backtest ─── metrics                            │   │
│   │                          │                                           │   │
│   │                        ffi                                           │   │
│   │                                                                      │   │
│   │   Klare Grenzen, testbar, erweiterbar                               │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Strategische Ziele

| Ziel | Beschreibung | Erfolgskriterium |
|------|--------------|------------------|
| **Z1: Performance** | 20x schnellere Backtests | < 500ms für 10k Candles |
| **Z2: Modularität** | Unabhängige, testbare Module | Jedes Crate isoliert testbar |
| **Z3: Wartbarkeit** | Klare Code-Struktur | Neue Strategie < 1 Tag zu implementieren |
| **Z4: Erweiterbarkeit** | Plugin-artige Strategien/Indikatoren | Kein Kern-Code für neue Strategie nötig |
| **Z5: Reproduzierbarkeit** | Regressionssichere Ergebnisse | **DEV-Mode**: identische Trades wie V1 (gleiche Entry/Exit-Events) |
| **Z6: Ressourcen-Effizienz** | Minimaler Memory-Footprint | < 100MB Peak, keine GC-Spikes |

### 3.3 Nicht-Ziele (Explizit Out of Scope)

| Nicht-Ziel | Begründung |
|------------|------------|
| Live-Trading in Rust | Live-Engine (MT5) bleibt Python-basiert |
| Multi-Symbol Backtests | Phase 2 (nach erfolgreicher Single-Symbol Migration) |
| Python-Strategien in V2 | V2 unterstützt nur Rust-native Strategien |
| Rückwärtskompatibilität zur V1 Python-API | Clean Break (neue API). **Output-Artefakte** bleiben jedoch V1-freundlich (Dateinamen + Kernfelder) und sind normativ spezifiziert (siehe Output-Contract). |
| GPU-Beschleunigung | Premature Optimization, CPU reicht für Zielsetzung |
| Parallelisierung im Core-Loop (rayon) | Erst Korrektheit/Determinismus und 1:1 Parität, dann Parallelität |

### 3.4 MVP Scope (Phase 1)

Der MVP ist bewusst eng geschnitten: **1 Symbol, 1 Strategie, 1:1 Parität**.

- **Strategie**: Mean Reversion Z-Score (inkl. aller HTF-Features wie in V1)
- **Order-Typen**: Market + Limit + Stop (MVP), Direction (long/short/both), max. Positionen konfigurierbar
- **Kostenmodell**: Spread aus Bid/Ask; Fees + Slippage aus bestehenden YAML-Configs (`configs/`)
- **News Filter**: Rust-native Alternative-Data (News Calendar) aus Parquet (wird in Phase 2 geladen)
- **Sessions**: Indikatoren über alle Bars; Entry-Signale nur innerhalb konfigurierter Trading-Sessions
- **Artefakte**: `trades.json`, `equity.csv`, `metrics.json`, `meta.json` (gemäß Output-Contract)

### 3.5 Kanonische 6 Szenarien (Mean Reversion Z-Score, MVP)

**Zweck:** Diese Szenarien sind die fachliche Minimal-Suite, um Feature-Parität und Regressionen eindeutig zu messen.

**Kanonische Test-Baseline (für alle 6 Szenarien, sofern nicht anders begründet):**

- Symbol: `EURUSD`
- Primary TF: `M5`
- Additional TFs: `D1`

**Szenarien (exakt 6):**

1. Market-Entry Long → Take-Profit
2. Market-Entry Long → Stop-Loss
3. Pending Entry (Limit/Stop) → Trigger ab `next_bar` → Exit
4. Same-Bar SL/TP Tie → SL-Priorität
5. `in_entry_candle` Spezialfall inkl. Limit-TP Regel
6. Mix aus Sessions/Warmup/HTF-Einflüssen, der die Strategie-Signalbildung deterministisch abdeckt

**Hinweis:** Zusätzlich zu Szenario 6 sind isolierte Session-Contract-Tests zulässig (z.B. „keine Trades außerhalb Session-Fenster“), aber Sessions sind damit nicht aus der kanonischen Suite ausgeklammert.

---

## 4. Architektur-Prinzipien

### 4.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PYTHON LAYER                                    │
│                         (Orchestration & Reporting)                          │
│                                                                              │
│   Verantwortung:                                                            │
│   • Config laden und validieren                                             │
│   • FFI-Call absetzen                                                       │
│   • Ergebnisse speichern                                                    │
│   • Visualisierung generieren                                               │
│                                                                              │
│   KEINE Geschäftslogik, KEINE Berechnungen                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ EIN FFI-Call (run_backtest)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RUST FFI LAYER                                  │
│                         (Entry Point & Serialization)                        │
│                                                                              │
│   Verantwortung:                                                            │
│   • JSON Config deserialisieren                                             │
│   • Rust-Engine koordinieren                                                │
│   • Ergebnis serialisieren                                                  │
│   • Fehler in Python-freundliches Format umwandeln                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RUST CORE LAYER                                 │
│                         (Business Logic & Engine)                            │
│                                                                              │
│   Verantwortung:                                                            │
│   • Daten laden (Parquet → Structs)                                         │
│   • Indikatoren berechnen                                                   │
│   • Event Loop ausführen                                                    │
│   • Portfolio verwalten                                                     │
│   • Metriken berechnen                                                      │
│                                                                              │
│   Alle performance-kritischen Operationen                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Design-Prinzipien

| Prinzip | Beschreibung | Auswirkung |
|---------|--------------|------------|
| **Single Responsibility** | Jedes Modul hat genau eine Aufgabe | Hohe Kohäsion, lose Kopplung |
| **Dependency Inversion** | High-Level Module abhängig von Abstraktionen | `Strategy` kennt nicht `Portfolio`-Implementierung |
| **Open/Closed** | Offen für Erweiterung, geschlossen für Modifikation | Neue Indikatoren via Trait, kein Kern-Change |
| **Single Source of Truth** | Daten existieren nur an einem Ort | Keine Duplikation zwischen Python/Rust |
| **Fail Fast** | Fehler früh erkennen und klar kommunizieren | Validierung beim Laden, nicht im Loop |
| **Validate Once** | Validierungen sind zentralisiert und passieren genau einmal | Keine „teilweise“ Checks in Hot-Paths, reproduzierbares Debugging |
| **Deterministic Dev Mode** | DEV-Mode nutzt stabile Seeds/Ordering | Identische Trades für Regressionstests |
| **Schema Contracts** | Input/Output haben explizite Contracts | `trades.json`/`equity.csv` bleiben kompatibel und validierbar |
| **Zero-Copy wo möglich** | Daten nicht kopieren, sondern referenzieren | Slices statt Vektorkopien |

### 4.3 Technologie-Stack

| Komponente | Technologie | Begründung |
|------------|-------------|------------|
| **Kern-Engine** | Rust 1.75+ | Performance, Sicherheit, keine GC |
| **Python-Binding** | PyO3 | Beste Rust↔Python Integration |
| **Daten-Format** | Parquet (Arrow) | Kompakt, spaltenbasiert, Rust-nativ |
| **Parquet-Reader** | `arrow-rs` | Rust-native Parquet/Arrow Implementierung (kein Pandas/Polars im Core) |
| **Serialisierung** | serde + serde_json | Standard für Rust, exzellente Performance |
| **Config (YAML)** | serde_yaml | 1:1 Nutzung der bestehenden `configs/*.yaml` (Fees/Specs) |
| **Determinismus** | RNG mit fixierbarem Seed | DEV-Mode reproduzierbar, PROD-Mode erlaubt Stochastik |
| **Orchestration** | Python 3.12+ | Bestehende Infrastruktur, einfache Scripting |
| **Build** | maturin | Standard für PyO3-Projekte |

## 5. Vergleich: V1 vs. V2

### 5.1 Architektur-Vergleich

| Aspekt | Omega V1 | Omega V2 |
|--------|----------|----------|
| **FFI-Grenzen** | Viele (pro Funktion) | **Eine (pro Backtest)** |
| **Event Loop** | Python | **Rust** |
| **Datenfluss** | Bidirektional | **Unidirektional** |
| **Daten-Format (intern)** | Python List / numpy | **Rust Vec / Arrow** |
| **Strategy-Sprache** | Python | **Rust** |
| **Modularität** | Monolithisch | **Crate-basiert** |
| **Testbarkeit** | Schwierig | **Isolierte Unit-Tests** |
| **Erweiterbarkeit** | Code ändern | **Trait implementieren** |

### 5.2 Performance-Projektion

| Metrik | V1 (aktuell) | V2 (Ziel) | Verbesserung |
|--------|--------------|-----------|--------------|
| FFI-Overhead | ~60% Laufzeit | ~0.1% Laufzeit | **~600x** |
| Event Loop | ~30% Laufzeit | ~5% Laufzeit | **~6x** |
| Memory Allocs | Tausende/Backtest | Hunderte/Backtest | **~10x** |
| **Gesamt-Laufzeit** | **8-12s** | **< 500ms** | **~20x** |

### 5.3 Entwickler-Erfahrung

| Aspekt | V1 | V2 |
|--------|-----|-----|
| Neue Strategie hinzufügen | 2-3 Tage | **< 1 Tag** |
| Neuen Indikator hinzufügen | 1 Tag | **2-4 Stunden** |
| Bug lokalisieren | Schwierig (Monolith) | **Einfach (Module)** |
| Test schreiben | Komplex (Dependencies) | **Einfach (Isolation)** |
| Deployment | Python + Rust Chaos | **Single Wheel** |

---

## 6. Erfolgskriterien & Metriken

### 6.1 Definition of Done (DoD)

Omega V2 gilt als erfolgreich abgeschlossen, wenn:

| # | Kriterium | Messbar durch |
|---|-----------|---------------|
| **E1** | 10k-Candle Backtest in < 500ms | Benchmark-Suite |
| **E2** | **DEV-Mode (Paritäts-Variante)**: Events matchen V1 exakt (`entry_time_ns`, `exit_time_ns`, `direction`, `reason`, Anzahl/Sortierung); Preise matchen nach Tick-Quantisierung; PnL/Fees innerhalb definierter Toleranzen | Regressionstest (Trade-Event/Preis/PnL-Vergleich) |
| **E3** | Jedes Crate hat > 80% Test-Coverage | `cargo tarpaulin` |
| **E4** | Keine Panics in Production | Fuzzing + Integration Tests |
| **E5** | Mean Reversion Z-Score vollständig portiert | Alle 6 Szenarien funktional |
| **E6** | Dokumentation vollständig | Alle `pub` Items dokumentiert |
| **E7** | Output-Artefakte sind stabil und validierbar | Schema-/Golden-File-Tests für `trades.json`/`equity.csv`/`metrics.json`/`meta.json` |

#### 6.1.1 Paritäts- und Vergleichsregeln (DEV)

Omega V2 unterstützt zwei Ausführungs-Varianten:

- `execution_variant = "v2"` (**Default**): kanonische V2-Execution (siehe Execution Model Plan).
- `execution_variant = "v1_parity"`: Paritäts-Variante für V1↔V2 Vergleich (Regression/CI), um die V1-Event-Parität zuverlässig messbar zu machen.

**Vergleichsregeln (Paritäts-Variante, normativ):**

- **Events:** exakt, basierend auf `entry_time_ns`, `exit_time_ns`, `direction`, `reason` sowie Anzahl und Sortierung.
- **Preise:** vor Vergleich auf `tick_size` quantisieren (aus `configs/symbol_specs.yaml`), danach **exakt** vergleichen.
- **PnL/Fees (account currency):**
    - pro Trade `result` (gerundet auf 2 Dezimalstellen) darf um **±0.05** abweichen.
    - Aggregat (`metrics.json`): `profit_net` und `fees_total` (2 Dezimalstellen) dürfen jeweils um **±0.01** abweichen.

### 6.2 Meilensteine

| Phase | Meilenstein | Erfolgskriterium |
|-------|-------------|------------------|
| **M0** | Baseline etabliert | V1 Performance dokumentiert |
| **M1** | `types` + `data` funktional | Parquet laden, Daten validieren |
| **M2** | `indicators` funktional | EMA, ATR, Bollinger gegen V1 verifiziert |
| **M3** | `portfolio` + `execution` funktional | Order-Fill mit Slippage/Fees korrekt |
| **M4** | `strategy` funktional | Mean Reversion portiert |
| **M5** | End-to-End Backtest | Kompletter Durchlauf, DEV-Mode: Trades = V1, Artefakte gemäß Output-Contract |
| **M6** | Performance-Ziel erreicht | < 500ms für 10k Candles |

### 6.3 Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Ergebnisse weichen von V1 ab | Mittel | Hoch | DEV-Mode Trade-Event-Vergleich + toleranzbasierte Metrik-Checks |
| Performance-Ziel nicht erreicht | Niedrig | Mittel | Profiling ab Phase 1 |
| Rust-Lernkurve zu steil | Mittel | Mittel | Schrittweise Migration |
| PyO3-Inkompatibilität | Niedrig | Hoch | Early Spike für FFI |
| Scope Creep | Hoch | Mittel | Strikte Phase-Gates |

### 6.4 Output-Artefakte (Contract)

Omega V2 liefert für jeden Backtest **stabile, validierbare Artefakte** (auch für CI/Regression):

- `trades.json`: Trade-/Order-Events (Entry/Exit) inkl. Szenario-Label
- `equity.csv`: Equity Curve / Balance über Zeit (pro Bar)
- `metrics.json`: Kernmetriken für Vergleiche und Optimizer
- `meta.json`: Run-Metadaten (Provenance/Config/Versionen)

**Metriken (MVP, mindestens):**

- Profit (raw)
- Profit (after fees)
- Max Drawdown (absolut)
- Max Drawdown (vom Initial Balance)
- Winrate
- Average R-Multiple
- Traded lots/contracts/shares
- Total fees
- Wins / Losses

---

## 7. Migration-Strategie

### 7.1 Grundsätze

| Prinzip | Beschreibung |
|---------|--------------|
| **Parallel Development** | V2 in separatem Verzeichnis (`rust_core/`), V1 bleibt unberührt |
| **Incremental Validation** | Nach jedem Modul: Vergleich mit V1-Referenzwerten |
| **Feature Parity First** | Erst alle Features von V1, dann Optimierungen |
| **No Big Bang** | Schrittweise Ablösung, nicht alles auf einmal |

### 7.2 Phasen-Übersicht

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION TIMELINE                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 0        Phase 1         Phase 2         Phase 3         Phase 4    │
│  Vorbereitung   Fundament       Core            Integration     Cutover    │
│                                                                             │
│  ┌─────────┐   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌────────┐ │
│  │Baseline │   │  types  │     │indicators│    │ backtest│     │  FFI   │ │
│  │Benchmark│   │  data   │     │execution │    │ metrics │     │ Python │ │
│  │Schema   │   │         │     │portfolio │    │         │     │Wrapper │ │
│  │         │   │         │     │strategy  │    │         │     │        │ │
│  └─────────┘   └─────────┘     └─────────┘     └─────────┘     └────────┘ │
│       │              │               │               │               │      │
│       ▼              ▼               ▼               ▼               ▼      │
│  V1 Reference   Load & Parse   Business Logic  End-to-End      Production  │
│  Values         Verified       Verified        Verified        Ready       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Rollback-Strategie

Falls V2 nicht die Erwartungen erfüllt:

1. V1 bleibt während gesamter Entwicklung funktionsfähig
2. Keine V1-Code-Änderungen für V2-Entwicklung
3. V2 kann jederzeit verworfen werden ohne V1-Impact
4. Feature-Flags erlauben A/B-Vergleich vor finalem Cutover

### 7.4 Optimizer: DEV/PROD Policy (normativ)

**Grundsatz:** Optimizer-Runs folgen der `run_mode` Policy der V2-Config.

- `run_mode = dev`: deterministisch (Sampling/Seeds/Ordering reproduzierbar).
- `run_mode = prod`: stochastisch (Seed aus OS-RNG), **aber** Seeds pro Trial werden in Artefakten/Meta mitgeschrieben, sodass Runs replaybar bleiben.

**Tie-Break für "Best Params" (wenn Score gleich):**

1. `score desc`
2. `profit_net desc`
3. `max_drawdown asc`
4. `config.hash asc`

---

## 8. Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Übergeordneter Blueprint, Module, Regeln |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Detaillierter Datenfluss, Phasen, Validierung |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Datei- und Verzeichnisstruktur, Interfaces |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives JSON-Config-Schema (Felder, Defaults, Validierung, Migration) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (Artefakte, Schema, Zeit/Units, Pfade) |

---

## 9. Glossar

| Begriff | Definition |
|---------|------------|
| **FFI** | Foreign Function Interface – Schnittstelle zwischen Python und Rust |
| **Crate** | Rust-Paket/Modul, eigenständig kompilierbar |
| **Trait** | Rust-Interface, definiert Verhalten ohne Implementierung |
| **Zero-Copy** | Daten werden nicht kopiert, nur Referenzen übergeben |
| **Event Loop** | Hauptschleife, die Candle für Candle durchiteriert |
| **Warmup** | Initiale Bars, die für Indikator-Berechnung benötigt werden |
| **HTF** | Higher Timeframe (z.B. D1 relativ zu M5) |
| **SIMD** | Single Instruction Multiple Data – CPU-Vektorisierung |

---

## 10. Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 12.01.2026 | Initiale Version, extrahiert aus OMEGA_V2_ARCHITECTURE_PLAN |

---

*Dieses Dokument definiert das "Warum" und "Was" der Omega V2 Migration. Die Details zum "Wie" finden sich in den spezialisierten Plänen.*
