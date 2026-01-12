# Wave 3 Event Engine Migration - Learnings & Best Practices

**Dokument:** Learnings für künftige Migrations-Waves  
**Erstellt:** 2025-01-XX  
**Migration Branch:** `migration/wave-3-event-engine-migration`  
**PR:** #24

---

## Executive Summary

Die Wave 3 Event Engine Migration wurde erfolgreich abgeschlossen. Dieses Dokument konsolidiert alle Learnings aus der Implementierung für künftige Rust-Migrationen.

---

## 1. PyO3 0.27 API-Änderungen (KRITISCH)

### Problem: `PyObject` Typ deprecated

In PyO3 0.27 ist `PyObject` als Alias für `Py<PyAny>` deprecated. Der Compiler zeigt dies jedoch **nicht** immer als Warning an - teilweise schlägt der Build schlicht fehl.

**Lösung:**
```rust
// ❌ ALT - funktioniert nicht mehr zuverlässig
fn callback(py: Python, callback: PyObject) -> PyResult<()>

// ✅ NEU - expliziter Typ
fn callback(py: Python, callback: Py<PyAny>) -> PyResult<()>
```

**Anwendung:**
- Alle Funktions-Signaturen prüfen
- Import ändern: `use pyo3::Py;` statt `PyObject`
- Return Types ebenfalls anpassen: `-> Py<PyAny>`

### Problem: `downcast` Method deprecated

```rust
// ⚠️ Warning - funktioniert noch
result.bind(py).downcast::<PyList>()?

// ✅ Bevorzugt (PyO3 0.27+)
result.bind(py).cast::<PyList>()?
```

**Empfehlung:** Warnungen am Ende der Migration beheben, da sie nicht blocking sind.

---

## 2. Ownership-Patterns in Rust-Loops

### Problem: Move in Loop

Wenn eine Variable vor einem Loop erstellt und innerhalb genutzt wird, kann es zu Ownership-Problemen kommen:

```rust
// ❌ FEHLER - slice_map moved in first iteration
let slice_map: HashMap<String, Py<PyAny>> = ...;
for i in 0..total {
    // slice_map wird hier gemoved
    callback.call(py, (slice_map,))  // MOVE!
}
```

**Lösung:** Dict innerhalb der Loop erstellen:

```rust
// ✅ KORREKT - dict wird jedes mal neu erstellt
for i in 0..total {
    let py_dict = PyDict::new(py);
    py_dict.set_item(symbol_key.clone(), slice_py)?;
    callback.call(py, (i, py_dict.unbind()))?;
}
```

**Regel:** Alles was in einen Callback übergeben wird, muss entweder:
- `.clone()` werden
- Neu erstellt werden pro Iteration
- Als Reference übergeben werden (`&`)

---

## 3. Python-Callback-Integration Pattern

### Empfohlene Struktur für Rust mit Python Callbacks:

```rust
#[pyclass]
pub struct RustEngine {
    // Interne State
}

#[pymethods]
impl RustEngine {
    #[pyo3(signature = (
        required_callback,
        optional_callback = None
    ))]
    fn run(
        &mut self,
        py: Python<'_>,
        required_callback: Py<PyAny>,
        optional_callback: Option<Py<PyAny>>,
    ) -> PyResult<Stats> {
        // Callback aufrufen
        let result = required_callback.call1(py, (arg1, arg2))?;
        
        // Optional callback
        if let Some(ref cb) = optional_callback {
            cb.call1(py, (arg1,))?;
        }
        
        Ok(stats)
    }
}
```

**Wichtig:**
- `call1` für Argumente als Tuple
- `call0` für keine Argumente
- `extract::<T>()` für Rückgabewerte
- Option<Py<PyAny>> für optionale Callbacks

---

## 4. Timestamp-Handling (DateTime ↔ i64)

### Rust verwendet i64 Microseconds:

```rust
pub struct CandleData {
    pub timestamp_us: i64,  // Microseconds since epoch
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
```

### Python Konversion:

```python
# Python datetime → Rust i64
timestamp_us = int(candle.timestamp.timestamp() * 1_000_000)

# Rust i64 → Python datetime
from datetime import datetime, timezone
dt = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc)
```

**Empfehlung:** UTC explizit verwenden, nie naive datetimes.

---

## 5. Feature-Flag-System Pattern

### Empfohlene Implementation:

```python
# 1. Modul-Level Cache
_RUST_AVAILABLE: Optional[bool] = None
_RUST_MODULE = None

def _check_rust_available() -> bool:
    """Lazy check for Rust backend."""
    global _RUST_AVAILABLE, _RUST_MODULE
    if _RUST_AVAILABLE is None:
        try:
            import omega_rust
            _RUST_MODULE = omega_rust
            _RUST_AVAILABLE = True
        except ImportError:
            _RUST_AVAILABLE = False
    return _RUST_AVAILABLE

# 2. Feature Flag Reading
def _should_use_rust() -> bool:
    """Check if Rust backend should be used."""
    if not _check_rust_available():
        return False
    flag = os.environ.get("OMEGA_USE_RUST_<MODULE>", "auto").lower()
    return flag in ("auto", "true", "1", "yes")

# 3. CI Verification Helper
def get_active_backend() -> str:
    """Return 'rust' or 'python' for CI verification."""
    return "rust" if _should_use_rust() else "python"
```

### Environment Variables:

| Variable | Wert | Effekt |
|----------|------|--------|
| `auto` (default) | Rust wenn verfügbar |
| `true` | Rust erzwingen |
| `false` | Python erzwingen |

---

## 6. Test-Strategie für Migrationen

### Minimal Required Tests:

```python
# 1. Backend Verification Test
class TestBackendVerify:
    def test_rust_available(self):
        assert _check_rust_available() is True
    
    def test_active_backend_default(self):
        assert get_active_backend() == "rust"
    
    def test_feature_flag_false(self):
        with patch.dict(os.environ, {"OMEGA_USE_RUST_X": "false"}):
            assert _should_use_rust() is False

# 2. Parity Test
class TestParity:
    def test_rust_python_match(self):
        result_rust = run_with_rust()
        result_python = run_with_python()
        assert result_rust == result_python  # oder np.allclose für floats
```

---

## 7. Build & Release Checklist

### Pre-Build:
- [ ] `cargo check` ohne Errors
- [ ] Alle Warnings dokumentiert (nicht-blocking)
- [ ] `Cargo.toml` Dependencies aktuell

### Build:
```bash
cd src/rust_modules/omega_rust
maturin develop --release
```

### Post-Build Validation:
```python
import omega_rust
print(omega_rust.get_<module>_backend())  # → "rust"
```

### CI Integration:
```yaml
- name: Build Rust
  run: |
    cd src/rust_modules/omega_rust
    maturin build --release
    pip install target/wheels/*.whl

- name: Verify Rust Backend
  run: |
    python -c "from src.x import get_active_backend; assert get_active_backend() == 'rust'"
```

---

## 8. Häufige Fehler & Fixes

### Fehler 1: "type annotation needed"

**Ursache:** Rust kann Typ nicht inferieren.

```rust
// ❌ 
let result = callback.call1(py, (arg,))?;

// ✅ 
let result: Py<PyAny> = callback.call1(py, (arg,))?;
```

### Fehler 2: "borrowed value does not live long enough"

**Ursache:** Temporäre Referenz überlebt Scope nicht.

```rust
// ❌ 
let s = String::from("test");
dict.set_item(&s[..], value)?;  // s wird nach diesem Statement dropped

// ✅ 
let s = String::from("test");
let key = s.clone();
dict.set_item(key, value)?;
```

### Fehler 3: "`Py<PyAny>` cannot be sent between threads safely"

**Ursache:** GIL-gebundene Objekte in Thread-Boundary.

**Lösung:** Alle Python-Objekte nur im GIL-Block (`Python::with_gil()`) verwenden.

---

## 9. Performance-Optimierung Guidelines

### Do:
- Batch-Callbacks statt einzelne Calls
- `Vec::with_capacity()` für bekannte Größen
- `&str` statt `String` wo möglich

### Don't:
- Keine Python-Calls in inneren Loops
- Keine Clone-Chain (`clone().clone()`)
- Keine unbeschränkten `Vec::push()` ohne capacity

### Messung:
```rust
use std::time::Instant;

let start = Instant::now();
// ... operation ...
stats.loop_time_ms = start.elapsed().as_secs_f64() * 1000.0;
```

---

## 10. Checkliste für neue Migrations-Waves

### Phase 1: Vorbereitung
- [ ] FFI-Spezifikation in `docs/ffi/` erstellt
- [ ] Migration Runbook in `docs/runbooks/` erstellt
- [ ] Performance Baseline gemessen
- [ ] Abhängigkeiten (vorherige Waves) abgeschlossen

### Phase 2: Rust Implementation
- [ ] Modul-Struktur unter `src/rust_modules/omega_rust/src/<module>/`
- [ ] `types.rs` mit allen Structs
- [ ] `engine.rs` oder `<name>.rs` mit Hauptlogik
- [ ] `mod.rs` mit Exports
- [ ] `lib.rs` Registration

### Phase 3: Integration
- [ ] Python-Wrapper mit Feature-Flag
- [ ] `_run_rust()` und `_run_python()` Methoden
- [ ] `get_active_backend()` CI-Helper

### Phase 4: Testing
- [ ] Backend Verification Tests
- [ ] Parity Tests (Rust == Python)
- [ ] Performance Tests (Speedup validiert)
- [ ] Regression Tests (bestehende Tests grün)

### Phase 5: Documentation
- [ ] Learnings dokumentiert
- [ ] CHANGELOG.md aktualisiert
- [ ] architecture.md aktualisiert (falls nötig)

---

## Appendix: Wave 3 Spezifische Erkenntnisse

### Position Management Callback

Das Event Engine braucht optionale Callbacks für Position Management:

```rust
#[pyo3(signature = (
    strategy_callback,
    executor,
    portfolio,
    slice_obj,
    progress_callback = None,
    position_mgmt_callback = None
))]
fn run(
    &mut self,
    py: Python<'_>,
    strategy_callback: Py<PyAny>,
    executor: Py<PyAny>,
    portfolio: Py<PyAny>,
    slice_obj: Py<PyAny>,
    progress_callback: Option<Py<PyAny>>,
    position_mgmt_callback: Option<Py<PyAny>>,
) -> PyResult<EventEngineStats> {
    // ...
}
```

Python-Seite muss Wrapper erstellen:

```python
def _create_position_mgmt_callback(self) -> Optional[Callable[..., None]]:
    pm = getattr(strategy_instance, "position_manager", None)
    if pm is None:
        return None
    
    def callback(symbol_slice, bid_candle, ask_candle):
        pm.manage_positions(...)
    
    return callback
```

---

## Änderungshistorie

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 2025-01-XX | Initial Release nach Wave 3 |
