# ADR-0001: Rust und Julia Migrations-Strategie

## Status

**Accepted** (finalisiert am 2026-01-05)

## Kontext

Das Omega Trading System ist ein Python-basierter Trading-Stack mit:
- **Event-driven Backtesting** (`src/backtest_engine/`) - 99.8% Python
- **Live-Trading Engine** (`src/hf_engine/`) - MT5-Adapter (Windows-only)
- **FastAPI UI** (`src/ui_engine/`)
- **Strategien** (`src/strategies/`) - modulare Trading-Strategien

### Code-Metriken (Baseline Stand 2026-01-04)

| Komponente | Python-Dateien | LOC | Type Coverage | Test Coverage |
|------------|----------------|-----|---------------|---------------|
| `backtest_engine` | 45 | ~8,500 | 95% (strict) | 87% |
| `hf_engine` | 28 | ~4,200 | 72% (relaxed) | 65% |
| `ui_engine` | 12 | ~1,800 | 90% (strict) | 78% |
| `strategies` | 15 | ~2,100 | 85% (strict) | 72% |

### Herausforderungen

1. **Performance**: Numerische Berechnungen in Python sind CPU-bound; Backtests mit vielen Symbolen/Zeiträumen sind langsam (typische Laufzeit: 45-120 Sekunden für 1-Jahres-Backtest)
2. **Type Safety**: ~~Große Teile der Codebase haben `ignore_errors=true` in Mypy~~ → **Phase 1 abgeschlossen**: Migrations-Kandidaten sind jetzt strict-compliant
3. **Skalierbarkeit**: Multi-Core-Parallelisierung ist durch Python GIL limitiert (max 1.8x Speedup bei joblib trotz 8 Cores)
4. **Determinismus**: Reproduzierbare Backtests erfordern strikte Seed-Kontrolle und deterministische Berechnungen

### Kräfte

- **Stabilität**: Live-Trading darf nicht durch Migration beeinträchtigt werden
- **Inkrementalität**: Migration muss schrittweise erfolgen, ohne Breaking Changes
- **Performance**: Backtest-Durchsatz soll um Faktor 5-10x verbessert werden
- **Team-Skills**: Rust/Julia-Expertise muss aufgebaut werden

### Constraints

- Python ≥3.12 als Basis bleibt erhalten
- `hf_engine/` bleibt pure Python (MT5-Integration, Stabilität)
- Keine Breaking Changes für bestehende Nutzer während Vorbereitung

## Entscheidung

### Migrations-Ansatz

Wir wählen einen **Hybrid-Ansatz** mit schrittweiser Migration ausgewählter Module:

1. **Rust** für Performance-kritische, numerische Hot-Paths
   - `indicator_cache.py` - Cache und Indikator-Berechnungen
   - `event_engine.py` - Event-Loop und Dispatch
   - Rating-Module - Scoring-Funktionen

2. **Julia** für Research und Analysis
   - Monte-Carlo-Simulationen
   - Statistische Analysen
   - Flexible Forschungs-Pipelines

3. **Python bleibt** für:
   - Live-Trading (`hf_engine/`)
   - UI (`ui_engine/`)
   - Strategien (`strategies/`)
   - I/O-bound Operations (`data/`)

### Vorbereitungs-Phasen

Vor der eigentlichen Migration erfolgt eine **18-wöchige Vorbereitungsphase**:

1. **Phase 0**: Foundation (Baselines, ADRs)
2. **Phase 1**: Type Safety Hardening (Mypy-Strict)
3. **Phase 2**: Interface-Definition (FFI-Typen)
4. **Phase 3**: Test-Infrastruktur (Benchmarks, Property-Tests)
5. **Phase 4**: Build-System (CI/CD für Rust/Julia)
6. **Phase 5**: Dokumentation (Runbooks, Validation)

### Technologie-Stack

| Komponente | Technologie | Begründung |
|------------|-------------|------------|
| Rust-Python-Bindings | PyO3 + Maturin | Standard; aktiv maintained; pip-kompatibel |
| Julia-Python-Bindings | PythonCall.jl | Moderner als PyJulia; besseres GIL-Handling |
| Serialisierung | Apache Arrow IPC | Zero-Copy; Schema-Evolution; Multi-Language |
| Build-System | Makefile + just | Standard; Cross-Platform; CI-Integration |

## Konsequenzen

### Positive Konsequenzen

- **Performance**: 5-10x Speedup für numerische Berechnungen erwartet
- **Type Safety**: Striktes Typing an FFI-Grenzen erzwingt Korrektheit
- **Parallelisierung**: Rust umgeht GIL; echte Multi-Core-Nutzung
- **Wartbarkeit**: Klare Interface-Definitionen verbessern Code-Qualität
- **Skalierbarkeit**: Große Backtests werden praktikabel

### Negative Konsequenzen

- **Komplexität**: Hybrid-Stack erfordert mehr Build-Tooling
- **Lernkurve**: Team muss Rust/Julia lernen
- **Debugging**: Cross-Language Debugging ist schwieriger
- **CI-Zeit**: Build-Zeiten steigen durch zusätzliche Kompilierung

### Risiken

| Risiko | Mitigation |
|--------|------------|
| FFI-Overhead überwiegt Gains | Batch-APIs; Arrow Zero-Copy; Benchmarks vor Migration |
| Determinismus-Verlust | Golden-File Tests; Seed-Propagation; Property-Tests |
| Live-Trading-Regression | Strikt isolierter `hf_engine/`; Trading-Safety-Tests |
| Team-Resistance | Einfache Module zuerst; Pair-Programming; gute Doku |

## Migrations-Reihenfolge (Priorisiert)

Die Module werden in folgender Reihenfolge migriert, basierend auf Performance-Impact und Komplexität:

### Priorität 1: High Performance Impact (Rust)

| Modul | Sprache | Erwarteter Speedup | Komplexität | Status |
|-------|---------|-------------------|-------------|--------|
| `indicator_cache.py` | Rust | 8-15x | Mittel | Ready for Migration |
| `rating/score_calculator.py` | Rust | 5-10x | Niedrig | Ready for Migration |
| `rating/metric_adjustments.py` | Rust | 5-8x | Niedrig | Ready for Migration |

### Priorität 2: Medium Performance Impact (Rust/Julia)

| Modul | Sprache | Erwarteter Speedup | Komplexität | Status |
|-------|---------|-------------------|-------------|--------|
| `core/event_engine.py` | Rust | 3-5x | Hoch | Ready for Migration |
| `core/execution_simulator.py` | Rust | 4-7x | Mittel | Ready for Migration |
| `optimizer/monte_carlo.py` | Julia | 10-20x | Mittel | Ready for Migration |

### Priorität 3: Research/Analysis (Julia)

| Modul | Sprache | Erwarteter Speedup | Komplexität | Status |
|-------|---------|-------------------|-------------|--------|
| `analysis/statistical_tests.py` | Julia | 5-10x | Niedrig | Vorbereitung |
| `analysis/regime_detection.py` | Julia | 8-15x | Mittel | Vorbereitung |

## Performance-Targets (Benchmark-Basiert)

Basierend auf den Performance-Baselines aus Phase 3:

| Operation | Python Baseline | Rust Target | Speedup |
|-----------|-----------------|-------------|---------|
| EMA(14) über 1M Candles | 45ms | <5ms | >9x |
| RSI(14) über 1M Candles | 62ms | <7ms | >8x |
| Full Backtest (1 Jahr, M1) | 85s | <15s | >5x |
| Monte Carlo (10k Sims) | 120s | <10s | >12x |
| Score Calculation (Batch) | 250ms | <30ms | >8x |

## Alternativen

### Alternative 1: Pure Python + Numba/Cython

- **Beschreibung**: JIT-Kompilierung mit Numba oder Cython-Extensions
- **Warum nicht gewählt**: 
  - Numba hat eingeschränkte Unterstützung für komplexe Datenstrukturen
  - Cython erfordert ähnlichen Aufwand wie Rust, aber weniger Performance-Gewinn
  - Keine echte GIL-Umgehung

### Alternative 2: Rewrite in Rust/Julia only

- **Beschreibung**: Kompletter Rewrite ohne Python
- **Warum nicht gewählt**:
  - Zu hohes Risiko und Aufwand
  - Python-Ecosystem (Pandas, NumPy, FastAPI) ist wertvoll
  - MT5-Integration erfordert Python

### Alternative 3: Status Quo (Pure Python)

- **Beschreibung**: Keine Migration; Performance-Optimierung in Python
- **Warum nicht gewählt**:
  - GIL bleibt Bottleneck
  - Fundamentale Performance-Limits
  - Type Safety bleibt schwach

### Alternative 4: Go statt Rust

- **Beschreibung**: Go für Performance-kritische Module
- **Warum nicht gewählt**:
  - Go's FFI (cgo) ist langsamer als Rust's PyO3
  - Rust bietet bessere Memory Safety Garantien
  - Rust hat besseres NumPy/Arrow Ecosystem (arrow-rs)

## Implementierungs-Checkliste (Vorbereitungsphase)

- [x] Phase 0: Foundation (Baselines, ADRs)
- [x] Phase 1: Type Safety Hardening (Mypy-Strict für Kandidaten)
- [x] Phase 2: Interface-Definition (FFI-Typen, Arrow-Schemas)
- [x] Phase 3: Test-Infrastruktur (Benchmarks, Property-Tests, Golden-Files)
- [x] Phase 4: Build-System (CI/CD für Rust/Julia)
- [ ] Phase 5: Dokumentation (Runbooks, Validation)

## Referenzen

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin User Guide](https://www.maturin.rs/)
- [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl)
- [Apache Arrow](https://arrow.apache.org/)
- [ADR-0002: Serialisierungsformat](ADR-0002-serialization-format.md)
- [ADR-0003: Error Handling](ADR-0003-error-handling.md)
- [ADR-0004: Build-System Architecture](ADR-0004-build-system.md)
- [Omega PYTHON_312_MIGRATION_PLAN.md](../PYTHON_312_MIGRATION_PLAN.md)
- [Omega RUST_JULIA_MIGRATION_PREPARATION_PLAN.md](../RUST_JULIA_MIGRATION_PREPARATION_PLAN.md)

## Änderungshistorie

| Datum | Autor | Änderung |
|-------|-------|----------|
| 2026-01-03 | GitHub Copilot | Initiale Version (Proposed) |
| 2026-01-05 | AI Agent | Finalisiert (Accepted); Migrations-Reihenfolge, Performance-Targets hinzugefügt |
