# Migration Runbook Template

**Version:** 1.0  
**Erstellt:** 2026-01-05  
**Zuletzt aktualisiert:** 2026-01-05

---

## Übersicht

Dieses Template dient als Vorlage für Migrations-Runbooks einzelner Module.
Jedes Runbook dokumentiert den vollständigen Prozess zur Migration eines Python-Moduls zu Rust oder Julia.

---

## Template-Struktur

### 1. Modul-Identifikation

````markdown
# Migration Runbook: [MODUL_NAME]

**Python-Pfad:** `src/[pfad]/[modul].py`  
**Zielsprache:** Rust | Julia  
**FFI-Integration:** PyO3/Maturin | PythonCall.jl  
**Priorität:** High | Medium | Low  
**Geschätzter Aufwand:** XS | S | M | L | XL  
**Status:** 🔴 Nicht begonnen | 🟡 In Arbeit | 🟢 Abgeschlossen

---

## Executive Summary

[2-3 Sätze: Was macht das Modul? Warum wird es migriert? Erwarteter Benefit.]

---

## Vorbedingungen

### Typ-Sicherheit
- [ ] Modul ist mypy --strict compliant
- [ ] Alle öffentlichen Funktionen haben vollständige Type Hints
- [ ] TypedDict/Protocol-Definitionen in `src/backtest_engine/core/types.py`

### Interface-Dokumentation
- [ ] FFI-Spezifikation in `docs/ffi/[modul].md`
- [ ] Arrow-Schemas definiert in `src/shared/arrow_schemas.py`
- [ ] Nullability-Konvention dokumentiert

### Test-Infrastruktur
- [ ] Benchmark-Suite in `tests/benchmarks/test_bench_[modul].py`
- [ ] Property-Based Tests in `tests/property/test_prop_[modul].py`
- [ ] Golden-File Tests in `tests/golden/test_golden_[modul].py`
- [ ] Test-Coverage ≥ 85%

### Performance-Baselines
- [ ] Baseline in `reports/performance_baselines/p0-01_[modul].json`
- [ ] Improvement-Target definiert (z.B. 5x Speedup)

---

## Migration Steps

### Step 1: Rust/Julia Modul Setup

**Rust:**
```bash
# Neues Modul in src/rust_modules/omega_rust/src/
mkdir -p src/rust_modules/omega_rust/src/[modul]
touch src/rust_modules/omega_rust/src/[modul]/mod.rs
```

**Julia:**
```bash
# Neues Modul in src/julia_modules/omega_julia/src/
touch src/julia_modules/omega_julia/src/[modul].jl
```

### Step 2: Interface Implementation

- [ ] Input-Typen von Python-TypedDict zu Rust-Structs / Julia-Types übersetzen
- [ ] Output-Typen definieren
- [ ] Arrow-Serialisierung implementieren
- [ ] Error-Handling nach ADR-0003 implementieren

### Step 3: Core-Logik portieren

- [ ] Python-Algorithmus in Rust/Julia neu implementieren
- [ ] Numerische Korrektheit validieren (Property-Tests)
- [ ] Edge-Cases behandeln (NaN, Inf, leere Arrays)

### Step 4: FFI-Bindings

**Rust (PyO3):**
```rust
#[pyfunction]
fn function_name(input: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    // Implementation
}

#[pymodule]
fn omega_rust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(function_name, m)?)?;
    Ok(())
}
```

**Julia (PythonCall):**
```julia
function function_name(input::Vector{Float64})::Vector{Float64}
    # Implementation
end
```

### Step 5: Python-Wrapper

```python
# src/[pfad]/[modul].py

def function_name(input: np.ndarray) -> np.ndarray:
    """Python-Wrapper mit Fallback."""
    try:
        from omega_rust import function_name as _rust_impl
        return _rust_impl(input)
    except ImportError:
        # Pure Python Fallback
        return _python_impl(input)
```

### Step 6: Testing

- [ ] Unit-Tests passieren
- [ ] Property-Based Tests passieren
- [ ] Golden-File Tests passieren (Determinismus)
- [ ] Benchmark zeigt erwarteten Speedup
- [ ] Integration-Tests mit Backtest-Engine

### Step 7: Documentation

- [ ] Docstrings aktualisiert
- [ ] FFI-Dokumentation aktualisiert
- [ ] CHANGELOG.md Eintrag
- [ ] architecture.md aktualisiert

---

## Rollback-Plan

### Bei Fehler in Produktion

1. **Sofortmaßnahme:** Feature-Flag deaktivieren
   ```python
   # settings.py
   USE_RUST_[MODUL] = False
   ```

2. **Fallback:** Python-Implementation wird automatisch verwendet

3. **Analyse:**
   - Logs prüfen
   - Edge-Case identifizieren
   - Issue erstellen

4. **Fix:**
   - Bugfix in Rust/Julia
   - Property-Test erweitern
   - Golden-File updaten

### Bei Performance-Regression

1. Benchmark-History prüfen: `python tools/benchmark_history.py compare`
2. Profiling: `cargo flamegraph` (Rust) oder `@profile` (Julia)
3. Bei > 10% Regression: Rollback zu Python

---

## Akzeptanzkriterien

### Funktional
- [ ] Alle bestehenden Tests passieren
- [ ] Keine Regression in Backtest-Determinismus
- [ ] Output-Format kompatibel mit bestehenden Consumern

### Performance
- [ ] Speedup ≥ [X]x gegenüber Python-Baseline
- [ ] Memory-Usage ≤ Python-Baseline
- [ ] Keine Memory-Leaks (Valgrind/miri clean)

### Qualität
- [ ] Code Review bestanden
- [ ] mypy --strict für Python-Wrapper
- [ ] clippy --pedantic für Rust (0 Warnings)
- [ ] Dokumentation vollständig

---

## Referenzen

- FFI-Spezifikation: `docs/ffi/[modul].md`
- Performance-Baseline: `reports/performance_baselines/p0-01_[modul].json`
- Arrow-Schemas: `src/shared/arrow_schemas.py`
- ADR-0001: Migration Strategy
- ADR-0002: Serialization Format
- ADR-0003: Error Handling
- ADR-0004: Build System

---

## Changelog

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| YYYY-MM-DD | 1.0 | Initiale Version | [Autor] |

````

---

## Verwendung

1. Kopiere dieses Template: `cp docs/runbooks/MIGRATION_RUNBOOK_TEMPLATE.md docs/runbooks/[modul]_migration.md`
2. Ersetze alle `[PLATZHALTER]` mit modulspezifischen Werten
3. Arbeite die Checklisten Schritt für Schritt ab
4. Dokumentiere Abweichungen und Learnings

---

## Best Practices

### Do's

- ✅ Runbook VOR der Migration erstellen
- ✅ Jeden Schritt explizit abhaken
- ✅ Blockers/Issues sofort dokumentieren
- ✅ Rollback-Plan testen bevor Go-Live
- ✅ Changelog pflegen

### Don'ts

- ❌ Steps überspringen
- ❌ Tests erst am Ende schreiben
- ❌ Ohne Benchmark-Baseline migrieren
- ❌ Ohne Feature-Flag deployen
- ❌ Mehrere Module gleichzeitig migrieren

---

## Automatisierung (optional, geplant)

Aktuell existieren keine Repository-Skripte wie `tools/create_runbook.sh` oder
`tools/migration_progress.py`. Wenn Runbook-Automatisierung ergänzt wird,
referenziert dieser Abschnitt die dann real vorhandenen Tools.
