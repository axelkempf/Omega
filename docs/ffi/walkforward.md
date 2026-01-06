# FFI-Spezifikation: Walkforward

**Modul:** `src/backtest_engine/optimizer/walkforward.py`  
**Task-ID:** P2-XX  
**Target-Sprache:** Julia (Orchestration), Python (Fallback)  
**Status:** Dokumentiert (2026-01-06)

---

## 1. Übersicht

Walkforward-Validation orchestriert rolling-window Optimierung und Out-of-Sample-Validierung. Die Orchestrierung kann in Julia erfolgen, während einzelne Backtests in Rust/Python laufen.

### 1.1 Verantwortlichkeiten

- Rolling-Window Splitting (Train/Test)
- Koordination von Optimizer-Läufen pro Window
- Aggregation von Out-of-Sample Ergebnissen
- Robustness-Metriken über alle Windows

---

## 2. Python Interface

```python
@dataclass
class WalkforwardConfig:
    """Walkforward-Konfiguration."""
    
    train_months: int  # Training-Window Länge
    test_months: int   # Test-Window Länge
    step_months: int   # Schritt zwischen Windows
    min_trades_per_window: int = 30
    optimization_config: OptimizerConfig | None = None

@dataclass
class WalkforwardResult:
    """Walkforward-Ergebnis."""
    
    windows: list[WindowResult]
    aggregated_metrics: AggregatedMetrics
    stability_score: float
    out_of_sample_pnl: float

class WalkforwardValidator:
    """Walkforward-Validation Orchestrator."""
    
    def __init__(self, config: WalkforwardConfig) -> None: ...
    
    def run(
        self,
        data: SymbolData,
        strategy_class: type[Strategy],
        param_space: dict[str, ParamRange],
    ) -> WalkforwardResult: ...
    
    def get_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[DateRange, DateRange]]: ...
```

---

## 3. Arrow Schema

```python
WALKFORWARD_WINDOW_SCHEMA = pa.schema([
    pa.field("window_id", pa.int32(), nullable=False),
    pa.field("train_start", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("train_end", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("test_start", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("test_end", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("best_params", pa.map_(pa.utf8(), pa.float64()), nullable=False),
    pa.field("train_score", pa.float64(), nullable=False),
    pa.field("test_score", pa.float64(), nullable=False),
    pa.field("test_trades", pa.int32(), nullable=False),
    pa.field("test_pnl", pa.float64(), nullable=False),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "walkforward_window",
})

WALKFORWARD_RESULT_SCHEMA = pa.schema([
    pa.field("strategy_name", pa.utf8(), nullable=False),
    pa.field("symbol", pa.utf8(), nullable=False),
    pa.field("n_windows", pa.int32(), nullable=False),
    pa.field("stability_score", pa.float64(), nullable=False),
    pa.field("avg_test_score", pa.float64(), nullable=False),
    pa.field("total_oos_pnl", pa.float64(), nullable=False),
    pa.field("param_consistency", pa.float64(), nullable=False),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "walkforward_result",
})
```

---

## 4. Julia Interface

```julia
module OmegaWalkforward

export walkforward_validate, rolling_windows

"""
Walkforward Validation mit paralleler Window-Execution.

# Arguments
- `data::Arrow.Table`: Symbol-Daten
- `strategy::Function`: Strategy-Factory
- `config::WalkforwardConfig`: Konfiguration

# Returns
- `WalkforwardResult`: Aggregierte Ergebnisse
"""
function walkforward_validate(
    data::Arrow.Table,
    strategy::Function,
    config::WalkforwardConfig;
    n_workers::Int = Threads.nthreads(),
)::WalkforwardResult
    windows = rolling_windows(data, config)
    
    # Parallel execution
    results = Threads.@threads for window in windows
        optimize_window(data, strategy, window, config)
    end
    
    aggregate_results(results)
end

"""
Generiere Rolling-Windows für Walkforward.
"""
function rolling_windows(
    data::Arrow.Table,
    config::WalkforwardConfig,
)::Vector{WindowSpec}
end

end # module
```

---

## 5. Rust Interface (Optional)

```rust
#[pyclass]
pub struct WalkforwardValidatorRust {
    config: WalkforwardConfig,
}

#[pymethods]
impl WalkforwardValidatorRust {
    #[new]
    pub fn new(config: WalkforwardConfigPy) -> Self;
    
    /// Generate rolling windows
    pub fn get_windows(&self, start: i64, end: i64) -> PyResult<Vec<WindowSpec>>;
    
    /// Run validation, returns results as Arrow IPC
    pub fn run(
        &self,
        py: Python<'_>,
        data_ipc: &[u8],
        strategy_factory: PyObject,
        param_space: HashMap<String, ParamRangePy>,
    ) -> PyResult<Vec<u8>>;
}
```

---

## 6. Error Handling

| Code | Name | Beschreibung |
| ---- | ---- | ------------ |
| `7001` | `INSUFFICIENT_DATA` | Nicht genug Daten für Window |
| `7002` | `NO_TRADES_IN_WINDOW` | Window hat keine Trades |
| `7003` | `OPTIMIZATION_FAILED` | Optimizer-Fehler in Window |
| `7004` | `WINDOW_OVERLAP_ERROR` | Ungültige Window-Konfiguration |

---

## 7. Performance-Targets

| Operation | Python Baseline | Target | Speedup |
| --------- | --------------- | ------ | ------- |
| Window Generation | 0.5s | 0.05s (Rust) | 10x |
| Single Window Opt | 60s | 10s (Rust+Julia) | 6x |
| Full Walkforward (12 win) | 720s | 120s | 6x |
| Result Aggregation | 1s | 0.1s | 10x |

---

## 8. Referenzen

- Optimizer: `docs/ffi/optimizer.md`
- Julia Walkforward: `src/julia_modules/omega_julia/src/walkforward.jl`
- Arrow-Schemas: `src/shared/arrow_schemas.py`
