# FFI-Spezifikation: Optimizer

**Modul:** `src/backtest_engine/optimizer/*.py`  
**Task-ID:** P2-XX  
**Target-Sprache:** Julia (Research), Rust (Orchestration)  
**Status:** Dokumentiert (2026-01-06)

---

## 1. Übersicht

Der Optimizer orchestriert Hyperparameter-Suche über Optuna und Monte-Carlo-Simulation. Julia ist für numerische Simulationen vorgesehen, Rust für Performance-kritische Orchestrierung.

### 1.1 Verantwortlichkeiten

- Grid-Search und Bayesian Optimization
- Monte-Carlo Robustness-Testing
- Parallel-Execution von Backtests
- Result-Aggregation und Ranking

---

## 2. Python Interface

```python
class OptimizerConfig:
    """Optimizer-Konfiguration."""
    
    n_trials: int
    n_jobs: int
    sampler: Literal["tpe", "cmaes", "random", "grid"]
    pruner: Literal["median", "hyperband", "none"]
    seed: int | None
    param_space: dict[str, ParamRange]

class Optimizer:
    """Hyperparameter-Optimizer."""
    
    def __init__(self, config: OptimizerConfig) -> None: ...
    
    def optimize(
        self,
        objective: Callable[[Trial], float],
        direction: Literal["maximize", "minimize"] = "maximize",
    ) -> OptimizationResult: ...
    
    def get_best_params(self) -> dict[str, Any]: ...
    
    def get_trials_dataframe(self) -> pd.DataFrame: ...
```

---

## 3. Arrow Schema

```python
OPTIMIZATION_RESULT_SCHEMA = pa.schema([
    pa.field("trial_id", pa.int32(), nullable=False),
    pa.field("params", pa.map_(pa.utf8(), pa.float64()), nullable=False),
    pa.field("value", pa.float64(), nullable=False),
    pa.field("state", pa.utf8(), nullable=False),  # COMPLETE, PRUNED, FAIL
    pa.field("duration_seconds", pa.float64(), nullable=False),
    pa.field("user_attrs", pa.map_(pa.utf8(), pa.utf8()), nullable=True),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "optimization_result",
})
```

---

## 4. Julia Interface (Monte-Carlo)

```julia
module OmegaMonteCarlo

export monte_carlo_robustness, bootstrap_ci

"""
Monte-Carlo Robustness Test für Parameter-Stability.

# Arguments
- `base_metrics::MetricsDict`: Baseline-Metriken
- `param_perturbations::Matrix{Float64}`: n_sims × n_params
- `evaluator::Function`: (params) -> score
- `n_simulations::Int`: Anzahl Simulationen

# Returns
- `RobustnessResult`: mean, std, ci_lower, ci_upper
"""
function monte_carlo_robustness(
    base_metrics::MetricsDict,
    param_perturbations::Matrix{Float64},
    evaluator::Function;
    n_simulations::Int = 1000,
    confidence::Float64 = 0.95,
)::RobustnessResult
    # Parallel execution with @threads
end

"""
Bootstrap Confidence Intervals für Metriken.
"""
function bootstrap_ci(
    data::Vector{Float64};
    n_bootstrap::Int = 10000,
    confidence::Float64 = 0.95,
)::Tuple{Float64, Float64}
end

end # module
```

---

## 5. Rust Interface (Orchestration)

```rust
#[pyclass]
pub struct OptimizerRust {
    config: OptimizerConfig,
    study: Option<Study>,
}

#[pymethods]
impl OptimizerRust {
    #[new]
    pub fn new(config: OptimizerConfigPy) -> PyResult<Self>;
    
    /// Run optimization, returns results as Arrow IPC
    pub fn optimize(
        &mut self,
        py: Python<'_>,
        objective: PyObject,  // Python callable
        direction: &str,
    ) -> PyResult<Vec<u8>>;
    
    /// Get best parameters
    pub fn get_best_params(&self) -> PyResult<HashMap<String, f64>>;
    
    /// Get all trials as Arrow IPC
    pub fn get_trials_arrow(&self) -> PyResult<Vec<u8>>;
}
```

---

## 6. Error Handling

| Code | Name | Beschreibung |
| ---- | ---- | ------------ |
| `6001` | `OPTIMIZATION_FAILED` | Alle Trials fehlgeschlagen |
| `6002` | `INVALID_PARAM_SPACE` | Ungültige Parameter-Definition |
| `6003` | `OBJECTIVE_ERROR` | Fehler in Objective-Function |
| `6004` | `PRUNER_ERROR` | Pruner-Konfiguration ungültig |

---

## 7. Performance-Targets

| Operation | Python Baseline | Target | Speedup |
| --------- | --------------- | ------ | ------- |
| Monte-Carlo (10K sims) | 120s | <10s (Julia) | 12x |
| Trial Execution (avg) | 85s | <15s (Rust) | 5.5x |
| Result Aggregation | 2s | <0.2s (Rust) | 10x |

---

## 8. Referenzen

- Optuna: `https://optuna.org/`
- Julia Monte-Carlo: `src/julia_modules/omega_julia/src/monte_carlo.jl`
- Arrow-Schemas: `src/shared/arrow_schemas.py`
