# ADR-0001: Rust und Julia Migrations-Strategie

## Status

Proposed

## Kontext

Das Omega Trading System ist ein Python-basierter Trading-Stack mit:
- **Event-driven Backtesting** (`src/backtest_engine/`) - 99.8% Python
- **Live-Trading Engine** (`src/hf_engine/`) - MT5-Adapter
- **FastAPI UI** (`src/ui_engine/`)

### Herausforderungen

1. **Performance**: Numerische Berechnungen in Python sind CPU-bound; Backtests mit vielen Symbolen/Zeiträumen sind langsam
2. **Type Safety**: Große Teile der Codebase haben `ignore_errors=true` in Mypy; FFI-Grenzen sind nicht typsicher definiert
3. **Skalierbarkeit**: Multi-Core-Parallelisierung ist durch Python GIL limitiert

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

## Referenzen

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin User Guide](https://www.maturin.rs/)
- [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl)
- [Apache Arrow](https://arrow.apache.org/)
- [Omega PYTHON_312_MIGRATION_PLAN.md](../PYTHON_312_MIGRATION_PLAN.md)

## Änderungshistorie

| Datum | Autor | Änderung |
|-------|-------|----------|
| 2026-01-03 | GitHub Copilot | Initiale Version |
