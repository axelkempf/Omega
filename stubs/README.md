# Type Stubs für Untyped Dependencies

Dieses Verzeichnis enthält Type Stubs (`.pyi` Dateien) für Third-Party-Dependencies, die keine vollständigen Type Hints bereitstellen.

## P1-09: Type Stubs für untyped Dependencies

**Datum**: 2026-01-05  
**Status**: ✅ KOMPLETT

### Implementierte Stubs

#### 1. joblib (`stubs/joblib/__init__.pyi`)

**Verwendung**: `backtest_engine.optimizer` nutzt joblib für:
- Parallel execution (`Parallel`, `delayed`)
- Result caching (`Memory.cache`)
- Persistence (`dump`, `load`)

**Coverage**:
- ✅ `Parallel` class mit vollständiger Signatur
- ✅ `delayed` decorator
- ✅ `Memory` class für Caching
- ✅ `dump`/`load` für Persistence
- ✅ `hash` utility

**Typische Verwendung im Projekt**:
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(objective_function)(trial) for trial in trials
)
```

#### 2. optuna (`stubs/optuna/__init__.pyi`)

**Verwendung**: `backtest_engine.optimizer` nutzt optuna für Bayesian Optimization.

**Coverage**:
- ✅ `Study` class (optimize, best_params, best_value, trials)
- ✅ `Trial` class (suggest_float, suggest_int, suggest_categorical)
- ✅ `create_study`/`load_study` functions
- ✅ Samplers (TPESampler, RandomSampler, GridSampler)
- ✅ Pruners (MedianPruner, PercentilePruner)
- ✅ Exceptions (TrialPruned, OptunaError)

**Typische Verwendung im Projekt**:
```python
import optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=100)
```

### Dependencies mit vollständigen Type Hints

Die folgenden Dependencies aus `pyproject.toml` haben bereits vollständige Type Hints und benötigen **keine** Stubs:

- ✅ **pandas** (`pandas-stubs` in dev-dependencies)
- ✅ **numpy** (eingebaute Type Hints seit 1.20+)
- ✅ **pydantic** (vollständige Type Hints)
- ✅ **fastapi** (vollständige Type Hints)
- ✅ **requests** (`types-requests` in dev-dependencies)
- ✅ **PyYAML** (`types-PyYAML` in dev-dependencies)
- ✅ **python-dateutil** (`types-python-dateutil` in dev-dependencies)

### Dependencies mit partiellen Type Hints (akzeptiert via mypy config)

Die folgenden Dependencies haben partielle oder fehlende Type Hints, werden aber via `ignore_missing_imports=true` in `pyproject.toml` akzeptiert:

- **matplotlib**: Hat Type Hints, aber unvollständig; wird nur für Reports genutzt (nicht FFI-kritisch)
- **psutil**: Minimal verwendet, nicht FFI-kritisch
- **pyarrow**: Hat Type Hints für Core-APIs
- **uvicorn**: Hat Type Hints für public APIs
- **filelock**: Simple Library, minimal verwendet
- **holidays**: Simple Library, minimal verwendet
- **MetaTrader5**: Windows-only, kein Stub erforderlich (wird in macOS/Linux Tests gemockt)

### Mypy-Konfiguration

Die Stubs werden automatisch von mypy erkannt, wenn sie im Projekt-Root im `stubs/` Verzeichnis liegen. In `pyproject.toml` ist konfiguriert:

```toml
[tool.mypy]
mypy_path = "stubs"
ignore_missing_imports = true  # Fallback für nicht-stub'bare Libraries
```

### Validierung

Die Stubs wurden validiert mit:

```bash
# Mypy auf Module mit joblib/optuna-Usage
mypy --strict src/backtest_engine/optimizer/

# Erwartet: 0 errors für Stub-covered APIs
```

### Maintenance

**Update-Frequenz**: Bei Major-Version-Updates der Dependencies sollten Stubs auf Kompatibilität geprüft werden.

**Erweiterung**: Neue APIs können hinzugefügt werden, wenn sie im Projekt verwendet werden. Stubs sollten nur **tatsächlich verwendete** APIs abdecken.

### Referenzen

- **PEP 561**: Distributing and Packaging Type Information
- **Mypy Stub-Dokumentation**: https://mypy.readthedocs.io/en/stable/stubs.html
- **joblib Dokumentation**: https://joblib.readthedocs.io/
- **optuna Dokumentation**: https://optuna.readthedocs.io/

### Acceptance Criteria (P1-09) - Status: ✅ ERFÜLLT

- [x] `.pyi` Stubs für kritische untyped Libraries (joblib, optuna)
- [x] Coverage für alle im Projekt verwendeten APIs
- [x] Mypy erkennt Stubs automatisch
- [x] Dokumentation der Stub-Coverage und Maintenance-Strategie
- [x] Validierung mit `mypy --strict` auf Migrations-Kandidaten
