---
module: walkforward
phase: 3
prerequisites:
  - Type-Hints vollständig (mypy --strict)
  - ≥70% Test-Coverage
  - Performance-Baseline dokumentiert
  - FFI-Spec finalisiert
  - Optimizer Migration abgeschlossen
rollback_procedure: docs/runbooks/rollback_generic.md
---

# Migration Runbook: Walkforward

**Status:** 🔴 Nicht begonnen (Readiness/Go-No-Go: `docs/MIGRATION_READINESS_VALIDATION.md`)

## 1. Modul-Übersicht

| Attribut | Wert |
| -------- | ---- |
| Quell-Modul | `src/backtest_engine/optimizer/walkforward.py` |
| Ziel-Sprache | Julia (Orchestration), Python (Fallback) |
| Priorität | P3 - Research-Beschleunigung |
| Geschätzter Aufwand | 4-5 Tage |
| Abhängigkeit | Optimizer Migration |

---

## 2. Architektur

```
Julia Orchestrator
├── Window Generation
├── Parallel Window Execution
│   ├── Window 1: Optimizer (Rust/Python)
│   ├── Window 2: Optimizer (Rust/Python)
│   └── ...
└── Result Aggregation
```

---

## 3. Migration Steps

### Phase 1: Julia Implementation (Day 1-3)

```julia
# src/julia_modules/omega_julia/src/walkforward.jl
module OmegaWalkforward

using Arrow
using Dates
using Base.Threads

export WalkforwardConfig, walkforward_validate, rolling_windows

struct WalkforwardConfig
    train_months::Int
    test_months::Int
    step_months::Int
    min_trades::Int
end

struct WindowSpec
    window_id::Int
    train_start::DateTime
    train_end::DateTime
    test_start::DateTime
    test_end::DateTime
end

"""
Generate rolling windows for walkforward validation.
"""
function rolling_windows(
    data_start::DateTime,
    data_end::DateTime,
    config::WalkforwardConfig,
)::Vector{WindowSpec}
    windows = WindowSpec[]
    window_id = 1
    
    current = data_start
    while true
        train_end = current + Month(config.train_months)
        test_end = train_end + Month(config.test_months)
        
        if test_end > data_end
            break
        end
        
        push!(windows, WindowSpec(
            window_id,
            current,
            train_end,
            train_end,
            test_end,
        ))
        
        window_id += 1
        current += Month(config.step_months)
    end
    
    return windows
end

"""
Run walkforward validation with parallel window execution.
"""
function walkforward_validate(
    data::Arrow.Table,
    optimizer_fn::Function,
    config::WalkforwardConfig;
    n_workers::Int = Threads.nthreads(),
)::Arrow.Table
    # Generate windows
    windows = rolling_windows(
        minimum(data.timestamp),
        maximum(data.timestamp),
        config,
    )
    
    # Parallel execution
    results = Vector{Any}(undef, length(windows))
    
    @threads for i in eachindex(windows)
        window = windows[i]
        
        # Filter data for window
        train_data = filter_timerange(data, window.train_start, window.train_end)
        test_data = filter_timerange(data, window.test_start, window.test_end)
        
        # Run optimizer on train data
        best_params = optimizer_fn(train_data)
        
        # Evaluate on test data
        test_result = evaluate(test_data, best_params)
        
        results[i] = merge_results(window, best_params, test_result)
    end
    
    # Aggregate to Arrow Table
    return create_result_table(results)
end

end # module
```

### Phase 2: Python Integration (Day 3-4)

```python
# src/backtest_engine/optimizer/walkforward_julia.py
from juliacall import Main as jl

class WalkforwardValidatorJulia:
    """Walkforward Validator mit Julia Backend."""
    
    def __init__(self, config: WalkforwardConfig):
        self.config = config
        self._julia = jl
        self._julia.eval('using OmegaWalkforward')
    
    def run(self, data: pd.DataFrame, strategy_class, param_space):
        # Convert to Arrow
        data_arrow = pa.Table.from_pandas(data)
        data_ipc = data_arrow.serialize().to_pybytes()
        
        # Call Julia
        result_ipc = self._julia.OmegaWalkforward.walkforward_validate(
            data_ipc,
            self._create_optimizer_fn(strategy_class, param_space),
            self._to_julia_config(),
        )
        
        # Convert result back
        return pa.ipc.read_table(result_ipc).to_pandas()
```

### Phase 3: Testing (Day 4-5)

```bash
# Julia Tests
julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.test()'

# Integration Tests
pytest tests/test_walkforward_full_preload_slicing.py -v
pytest tests/test_walkforward_analyzer_refined_num_samples_column.py -v
pytest tests/test_discover_walkforward_groups_snapshot_priority.py -v

# Performance
# Status: PLANNED (derzeit keine dedizierte pytest-benchmark Suite für Walkforward)
```

---

## 4. Validierung

### 4.1 Window Generation Check

```python
def test_window_generation_equivalence():
    """Julia and Python must generate identical windows."""
    config = WalkforwardConfig(train_months=6, test_months=2, step_months=1)
    
    py_windows = walkforward_python.get_windows(start, end, config)
    jl_windows = walkforward_julia.get_windows(start, end, config)
    
    assert len(py_windows) == len(jl_windows)
    for py, jl in zip(py_windows, jl_windows):
        assert py.train_start == jl.train_start
        assert py.test_end == jl.test_end
```

### 4.2 Performance Check

| Operation | Python | Julia | Target | Status |
| --------- | ------ | ----- | ------ | ------ |
| Window Gen | 0.5s | - | 0.05s | ⏳ |
| Single Window | 60s | - | 10s | ⏳ |
| Full WF (12 win) | 720s | - | 120s | ⏳ |

---

## 5. Rollback-Trigger

| Kriterium | Schwellwert | Aktion |
| --------- | ----------- | ------ |
| Window Mismatch | >0 | Rollback |
| Result Divergence | >1% | Rollback |
| Julia FFI Fehler | Jeder | Python Fallback |

---

## 6. Abnahme-Checkliste

- [ ] Window-Generation identisch
- [ ] Parallel-Execution funktioniert
- [ ] Results match Python Reference
- [ ] Performance-Target erreicht
- [ ] macOS + Linux Tests grün
- [ ] Code-Review abgeschlossen

---

## 7. Sign-off Matrix

| Phase | Reviewer | Datum | Status |
|-------|----------|-------|--------|
| FFI-Spec Review | - | - | ⏳ Pending |
| Code Review (Julia) | - | - | ⏳ Pending |
| Code Review (Python Integration) | - | - | ⏳ Pending |
| Performance Validation | - | - | ⏳ Pending |
| Cross-Platform Validation | - | - | ⏳ Pending |
| Final Approval | - | - | ⏳ Pending |

### Sign-off Kriterien

1. **FFI-Spec Review**: FFI-Spezifikation ist vollständig und abgenommen
2. **Code Review (Julia)**: Julia-Code folgt Style Guide, alle Tests grün
3. **Code Review (Python Integration)**: Python-Wrapper ist mypy --strict compliant
4. **Performance Validation**: ≥6x Speedup für Full Walkforward erreicht
5. **Cross-Platform Validation**: Tests auf macOS und Linux erfolgreich
6. **Final Approval**: Alle vorherigen Sign-offs erteilt, Go-Live freigegeben

### Abhängigkeiten Sign-off

| Abhängigkeit | Status | Blocker für |
|--------------|--------|-------------|
| Optimizer Migration | ⏳ Pending | Dieses Modul |

---

## 8. Referenzen

- FFI-Spezifikation: `docs/ffi/walkforward.md`
- Julia-Modul: `src/julia_modules/omega_julia/src/walkforward.jl` (docs-lint:planned)
- Arrow-Schemas: `src/shared/arrow_schemas.py`
- ADR-0001: Migration Strategy
- ADR-0002: Serialization Format

---

## Changelog

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2026-01-05 | 1.0 | Initiale Version | Omega Team |
| 2026-01-08 | 1.1 | Sign-off Matrix, Referenzen, Abhängigkeiten hinzugefügt | Omega Team |

