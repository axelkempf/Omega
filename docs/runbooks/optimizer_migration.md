---
module: optimizer
phase: 3
prerequisites:
  - Type-Hints vollständig (mypy --strict)
  - ≥70% Test-Coverage
  - Performance-Baseline dokumentiert
  - FFI-Spec finalisiert
  - Julia Environment konfiguriert
rollback_procedure: docs/runbooks/rollback_generic.md
---

# Migration Runbook: Optimizer

**Status:** 🔴 Nicht begonnen (Readiness/Go-No-Go: `docs/MIGRATION_READINESS_VALIDATION.md`)

## 1. Modul-Übersicht

| Attribut | Wert |
| -------- | ---- |
| Quell-Module | `src/backtest_engine/optimizer/*.py` |
| Ziel-Sprachen | Julia (Monte-Carlo), Rust (Orchestration) |
| Priorität | P3 - Research-Beschleunigung |
| Geschätzter Aufwand | 6-8 Tage |

---

## 2. Hybrid-Architektur

```
Python (Optuna) ─────────────┐
                             │
Rust (Orchestration) ◄───────┤───► Result Aggregation
                             │
Julia (Monte-Carlo) ◄────────┘
```

- **Python**: Optuna-Integration bleibt (TPE Sampler, DB Storage)
- **Julia**: Monte-Carlo Simulations (parallel, numerisch)
- **Rust**: Trial-Orchestrierung, Arrow IPC, Result-Handling

---

## 3. Migration Steps

### Phase 1: Julia Setup (Day 1-2)

```bash
# Julia Project initialisieren
cd src/julia_modules/omega_julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Dependencies
# Project.toml sollte enthalten:
# Arrow = "2.7"
# PythonCall = "0.9"
```

```julia
# src/julia_modules/omega_julia/src/monte_carlo.jl
module OmegaMonteCarlo

using Arrow
using Base.Threads

export monte_carlo_robustness

function monte_carlo_robustness(
    base_metrics::Dict,
    perturbations::Matrix{Float64},
    evaluator::Function;
    n_sims::Int = 1000,
)::Dict
    results = Vector{Float64}(undef, n_sims)
    
    @threads for i in 1:n_sims
        perturbed = apply_perturbation(base_metrics, perturbations[i, :])
        results[i] = evaluator(perturbed)
    end
    
    return Dict(
        "mean" => mean(results),
        "std" => std(results),
        "ci_lower" => quantile(results, 0.025),
        "ci_upper" => quantile(results, 0.975),
    )
end

end # module
```

### Phase 2: Rust Orchestration (Day 3-5)

```rust
// src/rust_modules/omega_rust/src/optimizer/orchestrator.rs

#[pyclass]
pub struct OptimizerOrchestrator {
    config: OptimizerConfig,
}

#[pymethods]
impl OptimizerOrchestrator {
    /// Run single trial, returns score
    pub fn run_trial(
        &self,
        py: Python<'_>,
        trial_id: i32,
        params: HashMap<String, f64>,
        objective: PyObject,
    ) -> PyResult<f64> {
        // Call Python objective with params
        let result = objective.call1(py, (params,))?;
        result.extract::<f64>(py)
    }
    
    /// Aggregate results from Arrow IPC
    pub fn aggregate_results(&self, results_ipc: Vec<&[u8]>) -> PyResult<Vec<u8>> {
        // Combine all trial results
        // Return as Arrow IPC
    }
}
```

### Phase 3: Integration (Day 6-8)

```python
# Python Integration Layer
class HybridOptimizer:
    """Optimizer mit Rust/Julia Backend."""
    
    def __init__(self, config: OptimizerConfig):
        self._optuna_study = optuna.create_study(...)
        self._rust_orchestrator = OptimizerOrchestrator(config)
        self._julia_mc = None  # Lazy load
    
    def optimize(self, objective, direction="maximize"):
        # Use Rust for orchestration
        # Optuna for sampling
        # Julia for Monte-Carlo
        pass
```

---

## 4. Validierung

### 4.1 Monte-Carlo Validation

```bash
# Julia Monte-Carlo testen
julia --project=src/julia_modules/omega_julia -e '
    using OmegaMonteCarlo
    # Test cases
'

# Performance vergleichen
pytest tests -k monte_carlo -v
```

### 4.2 Performance Check

| Operation | Python | Hybrid | Target | Status |
| --------- | ------ | ------ | ------ | ------ |
| Monte-Carlo (10K) | 120s | - | <10s | ⏳ |
| Trial Execution | 85s | - | <15s | ⏳ |
| Result Aggregation | 2s | - | <0.2s | ⏳ |

---

## 5. Rollback-Trigger

| Kriterium | Schwellwert | Aktion |
| --------- | ----------- | ------ |
| Monte-Carlo Divergenz | >1% | Rollback |
| Performance-Regression | >10% | Rollback |
| Julia FFI Fehler | Jeder | Python Fallback |
| Rust Panic | Jeder | Rollback |

---

## 6. Abnahme-Checkliste

- [ ] Julia Monte-Carlo korrekt
- [ ] Rust Orchestration funktioniert
- [ ] Optuna-Integration erhalten
- [ ] Performance-Targets erreicht
- [ ] Cross-Platform Tests (macOS, Linux)
- [ ] Code-Review abgeschlossen

---

## 7. Sign-off Matrix

| Rolle | Name | Datum | Signatur |
|-------|------|-------|----------|
| Tech Lead | | | ⏳ |
| QA Lead | | | ⏳ |
| DevOps | | | ⏳ |

