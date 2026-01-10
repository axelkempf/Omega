# Wave 4: Execution Simulator Full Rust Migration - Implementation Plan Prompt

## Kontext

Du bist ein Migrations-Architekt, der den Implementierungsplan für **Wave 4** der Rust-Migration des Omega Trading-Systems erstellt. Das Ziel ist die **vollständige Migration des ExecutionSimulators zu Rust**, sodass keine Python-Fallback-Logik mehr benötigt wird.

Wave 4 baut auf den **erfolgreich abgeschlossenen Waves 0-3** auf:
- ✅ **Wave 0:** Slippage & Fee (Rust Integration, FFI Pattern etabliert)
- ✅ **Wave 1:** IndicatorCache (474x Speedup, Rust-Backend für Indikatoren)
- ✅ **Wave 2:** Portfolio (State-Management in Rust)
- ✅ **Wave 3:** Event Engine (Hybrid Rust Loop + Python Callbacks)

---

## Zielsetzung Wave 4

**Primärziel:** Full Rust ExecutionSimulator ohne Python-Fallback

| Aspekt | Beschreibung |
|--------|--------------|
| **Modul** | `src/backtest_engine/core/execution_simulator.py` (655 LOC) |
| **Target** | `src/rust_modules/omega_rust/src/execution/` |
| **Feature Flag** | `OMEGA_USE_RUST_EXECUTION_SIMULATOR=always` (kein "auto"/"python" mehr) |
| **Performance Target** | ≥8x Speedup (baseline: ~45ms/1K signals → target: <5ms) |
| **Determinismus** | Bit-genaue Reproduzierbarkeit (Golden-Tests) |

---

## Vorhandene Artefakte (Input für den Plan)

### FFI-Spezifikation
- **Datei:** `docs/ffi/execution_simulator.md` (727 Zeilen)
- **Inhalt:**
  - Arrow Schemas für `PortfolioPosition`, `TradeSignal`, `SymbolSpec`
  - API-Contracts für alle Methoden
  - Error Codes und Nullability-Dokumentation
  - Performance-Targets

### Migration Runbook
- **Datei:** `docs/runbooks/execution_simulator_migration.md` (306 Zeilen)
- **Status:** Template vorhanden, noch nicht ausgeführt
- **Struktur:** 7-Phasen-Struktur mit Rollback-Plan

### Performance Baseline
- **Datei:** `reports/performance_baselines/p0-01_execution_simulator.json`
- **Metrics:**
  - Signal Processing (1K): 45ms baseline → <5ms target
  - Exit Evaluation (1K): 32ms baseline → <4ms target
  - Full Backtest Loop: 85s baseline → <15s target

### Bestehende Rust-Infrastruktur (aus Wave 0-3)
- `src/rust_modules/omega_rust/Cargo.toml`
- `src/rust_modules/omega_rust/src/error.rs` (Error Handling Pattern)
- `src/rust_modules/omega_rust/src/slippage.rs` (Wave 0)
- `src/rust_modules/omega_rust/src/portfolio/` (Wave 2)
- `src/rust_modules/omega_rust/src/event/` (Wave 3)

### Arrow Schema Registry
- **Datei:** `src/shared/arrow_schemas.py`
- **Relevante Schemas:** `POSITION_SCHEMA`, `TRADE_SIGNAL_SCHEMA`

---

## ExecutionSimulator Kernfunktionalität

Das Modul implementiert folgende kritische Trading-Logik:

### 1. Signal-Verarbeitung (`process_signal`)
```
TradeSignal → Entry-Validation → Position-Sizing → Position-Eröffnung
```
- Market/Limit/Stop Order-Typen
- Risk-Based Position Sizing (via LotSizer)
- Slippage- und Fee-Anwendung

### 2. Exit-Evaluation (`evaluate_exit`)
```
Position + Candle → SL/TP Hit Detection → PnL Calculation → Position Close
```
- Stop-Loss Hit Detection (mit Gap-Handling)
- Take-Profit Hit Detection
- Trailing Stop Updates
- R-Multiple Berechnung

### 3. Entry-Trigger (`entry_trigger`)
```
Pending Position + Tick/Candle → Activation Check → Market Entry
```
- Limit Order Activation (Ask ≤ Limit für Long, Bid ≥ Limit für Short)
- Stop Order Activation (Bid ≥ Stop für Long, Ask ≤ Stop für Short)

### 4. Position-Sizing (`calculate_lot_size`)
```
Risk + SL Distance + SymbolSpec → Lot Size (Risk-Based)
```
- Point Value Berechnung
- Lot Size Rounding (volume_step)
- Min/Max Lot Constraints

---

## Lessons Learned aus Wave 0-3 (zu beachten)

### Aus Wave 0 (Slippage & Fee)
| Learning | Anwendung für Wave 4 |
|----------|---------------------|
| Namespace Conflicts | `execution` vs Python stdlib prüfen |
| RNG Determinismus | Slippage-RNG Seeds in Rust konsistent |
| FFI-Overhead | Batch-Processing für Signal-Arrays |

### Aus Wave 1 (IndicatorCache)
| Learning | Anwendung für Wave 4 |
|----------|---------------------|
| Integration ≠ Implementierung | End-to-End Backtest muss Rust nutzen |
| API-Drift | Protocol für Python/Rust Parity |
| Methoden-Parität | Alle `process_signal` Varianten abdecken |

### Aus Wave 2 (Portfolio)
| Learning | Anwendung für Wave 4 |
|----------|---------------------|
| State-Management | Positions-State in Rust (nicht extern) |
| Batch-API | `process_signals_batch()` implementieren |
| Type-Conversion | Position↔Arrow Konverter |

### Aus Wave 3 (Event Engine)
| Learning | Anwendung für Wave 4 |
|----------|---------------------|
| Callback Elimination | **KEIN Python-Callback** in ExecutionSimulator |
| Full Rust | Alle Branches in Rust (kein Hybrid) |
| Determinismus | Golden-Test mit exakter Reproduzierbarkeit |

---

## Zu erstellender Implementierungsplan

Erstelle einen detaillierten Implementierungsplan im Format der bestehenden Wave-Pläne (`docs/WAVE_X_*_IMPLEMENTATION_PLAN.md`) mit folgenden Abschnitten:

### 1. Executive Summary
- Warum ExecutionSimulator als Wave 4?
- Prerequisites (aus Wave 0-3)
- Go/No-Go Kriterien

### 2. Voraussetzungen & Status
- Infrastructure Readiness (was ist schon da)
- Python-Modul Baseline (LOC, Methoden, Komplexität)
- Performance-Baseline mit konkreten Zahlen
- Bottleneck-Analyse

### 3. Lessons Learned Integration
- Konkrete Anwendung der Learnings aus Wave 0-3
- Design-Entscheidungen basierend auf Learnings
- **Entscheidung: Full Rust (kein Hybrid)** begründen

### 4. Architektur-Übersicht
- Ziel-Architektur Diagramm (ASCII)
- Rust-Modul-Struktur (`src/rust_modules/omega_rust/src/execution/`)
- Core Types (Rust structs/enums)
- PyO3 Interface (öffentliche API)

### 5. Implementierungs-Phasen (10-14 Tage)
Definiere konkrete Phasen mit:
- **Phase 1:** Rust Skeleton & Types (2 Tage)
- **Phase 2:** Signal Processing (2-3 Tage)
- **Phase 3:** Exit Evaluation (2-3 Tage)
- **Phase 4:** Entry Trigger & Pending Orders (2 Tage)
- **Phase 5:** Position Sizing Integration (1-2 Tage)
- **Phase 6:** Python API Wrapper (1 Tag)
- **Phase 7:** Testing & Validation (2-3 Tage)

Pro Phase:
- Konkrete Rust-Dateien und Funktionen
- Python-Änderungen
- Test-Anforderungen
- Akzeptanzkriterien

### 6. Rust-Implementation Details
- Alle Rust Structs mit vollständigen Feldern
- Alle Trait-Implementierungen
- PyO3 #[pyclass] und #[pymethods] Definitionen
- Arrow IPC Integration (Deserialize/Serialize)

### 7. Python-Integration
- `execution_simulator.py` Änderungen
- Feature-Flag Handling (`OMEGA_USE_RUST_EXECUTION_SIMULATOR`)
- Deprecation der Python-Logik (keine Dual-Maintenance)
- Imports und Re-exports

### 8. Test-Strategie
- Unit Tests (Rust-native): `execution/tests/`
- Integration Tests (Python): `tests/test_execution_simulator_rust.py`
- Golden-Tests: `tests/golden/test_golden_execution.py`
- Property-Based Tests: `tests/property/test_execution_properties.py`
- Performance-Tests: `tests/benchmarks/test_bench_execution_simulator.py`
- Determinismus-Validierung

### 9. Validierung & Akzeptanzkriterien
- Performance Target: ≥8x Speedup
- Determinismus: Bit-genaue Reproduzierbarkeit
- API Parity: Alle Python-Methoden in Rust verfügbar
- CI/CD Gates: Alle Tests grün

### 10. Rollback-Plan
- Feature-Flag Deaktivierung (falls Probleme)
- Schritte zur Rückkehr zu Python-Implementation
- Monitoring während Rollout

### 11. Checklisten
- [ ] Pre-Implementation Checklist
- [ ] Daily Progress Checklist
- [ ] Post-Implementation Validation Checklist
- [ ] Sign-off Matrix

---

## Spezifische Anforderungen für Full Rust

### Warum kein Hybrid (im Gegensatz zu Wave 3)?

Die Event Engine benötigte Python-Callbacks für Strategy-Evaluation, weil Strategien Python-Klassen sind. Der ExecutionSimulator hat **keine externen Python-Callbacks** - alle Logik ist self-contained:

| Funktion | Python-Dependency | Full Rust möglich? |
|----------|-------------------|-------------------|
| `process_signal()` | Keine | ✅ Ja |
| `evaluate_exit()` | Keine | ✅ Ja |
| `entry_trigger()` | Keine | ✅ Ja |
| `calculate_lot_size()` | LotSizer (kann in Rust) | ✅ Ja |
| Slippage/Fee | ✅ Bereits Rust (Wave 0) | ✅ Ja |

### Full Rust Architektur

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Python API Layer (Thin Wrapper)                      │
│  class ExecutionSimulator:                                              │
│      def __init__(self, ...):                                           │
│          self._rust = ExecutionSimulatorRust(...)  # ALWAYS Rust       │
│                                                                         │
│      def process_signal(self, signal) -> Optional[Position]:           │
│          return self._rust.process_signal(signal)  # Rust call         │
│                                                                         │
│      def evaluate_exit(...) -> Optional[CloseResult]:                   │
│          return self._rust.evaluate_exit(...)      # Rust call         │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              │ PyO3 FFI (Arrow IPC)
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Rust ExecutionSimulator (Full Logic)               │
│  src/rust_modules/omega_rust/src/execution/                            │
│  ├── mod.rs                                                            │
│  ├── simulator.rs      # ExecutionSimulatorRust struct + methods      │
│  ├── position.rs       # PortfolioPosition state machine              │
│  ├── signal.rs         # TradeSignal processing                       │
│  ├── trigger.rs        # Entry/Exit trigger logic                     │
│  ├── sizing.rs         # Position sizing (risk-based)                 │
│  └── tests/            # Rust unit tests                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Referenz-Dokumente (lies diese vor Erstellung)

1. **FFI Spec:** `docs/ffi/execution_simulator.md`
2. **Runbook:** `docs/runbooks/execution_simulator_migration.md`
3. **Wave 3 Plan:** `docs/WAVE_3_EVENT_ENGINE_IMPLEMENTATION_PLAN.md` (als Template)
4. **Migration Readiness:** `docs/MIGRATION_READINESS_VALIDATION.md`
5. **Rust Migration Prep:** `docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`
6. **Error Handling:** `src/shared/error_codes.py`, `src/rust_modules/omega_rust/src/error.rs`
7. **Arrow Schemas:** `src/shared/arrow_schemas.py`

---

## Output-Format

Erstelle das Dokument als `docs/WAVE_4_EXECUTION_SIMULATOR_IMPLEMENTATION_PLAN.md` mit:

1. **Markdown-Header** mit Version, Datum, Status, Branch-Name
2. **Executive Summary** (max 1 Seite)
3. **Detaillierte Phasen** mit Code-Snippets (Rust und Python)
4. **ASCII-Diagramme** für Architektur
5. **Tabellen** für Status-Tracking
6. **Checklisten** am Ende
7. **Konsistente Formatierung** wie die bestehenden Wave-Pläne

---

## Erfolgskriterien für den Plan

Der Plan gilt als vollständig, wenn:

- [ ] Alle 11 Abschnitte sind ausgearbeitet
- [ ] Rust-Code-Snippets sind syntaktisch korrekt
- [ ] Python-Integration ist vollständig spezifiziert
- [ ] Performance-Targets sind quantifiziert (≥8x)
- [ ] Determinismus-Strategie ist dokumentiert
- [ ] Rollback-Plan ist praktikabel
- [ ] Geschätzter Aufwand ist realistisch (10-14 Tage)
- [ ] Plan referenziert alle vorhandenen Artefakte
- [ ] Lessons Learned aus Wave 0-3 sind integriert
- [ ] Full Rust (kein Hybrid) ist begründet

---

## Zusätzliche Hinweise

### Kritische Invarianten (dürfen nicht brechen)

1. **Determinismus:** Gleicher Input → Gleicher Output (Seeds, keine Floating-Point-Drift)
2. **Resume-Semantik:** `magic_number` Matching muss funktionieren
3. **R-Multiple Berechnung:** `result / risk_per_trade` exakt wie Python
4. **Gap-Handling:** SL/TP bei Gap-Opens korrekt (Worst-Case für Short, Best-Case für Long)

### Performance-Kritische Pfade

1. **Hot Path:** `evaluate_exit()` wird pro Bar für alle offenen Positions aufgerufen
2. **Batch-Optimierung:** Array-Operationen statt Einzelaufrufe
3. **Memory:** Position-State in Rust halten (kein FFI-Overhead pro Bar)

### Testabdeckung-Anforderungen

- Unit Tests: ≥90% Line Coverage in Rust
- Integration Tests: Alle Python-API-Methoden getestet
- Golden Tests: Exakte Byte-für-Byte Reproduzierbarkeit
- Property Tests: Edge Cases (NaN, Inf, extreme Lot Sizes)
- Benchmark Tests: 8x Speedup validiert

---

*Erstellt: 2026-01-10*
*Branch: `migration/wave-4-execution-engine`*
