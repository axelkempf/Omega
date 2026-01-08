# Nullability-Konvention für FFI-Grenzen

**Phase 2 Task:** P2-09  
**Status:** ✅ Dokumentiert (2026-01-05)

---

## Übersicht

Dieses Dokument definiert die Konventionen für nullable Werte an FFI-Grenzen zwischen Python, Rust und Julia. Eine konsistente Nullability-Behandlung ist kritisch für:

1. **Type Safety**: Vermeidung von Null-Pointer-Exceptions
2. **API-Klarheit**: Explizite Dokumentation optionaler Werte
3. **Performance**: Optimale Speicherrepräsentation
4. **Interoperabilität**: Konsistentes Verhalten über Sprachen hinweg

---

## Nullability Mapping

### Grundlegende Typ-Korrespondenz

| Python | Arrow | Rust | Julia | Semantik |
|--------|-------|------|-------|----------|
| `None` | `null` | `None` (Option) | `nothing` | Fehlender Wert |
| `Optional[T]` | nullable field | `Option<T>` | `Union{T, Nothing}` | Optionaler Wert |
| `T \| None` | nullable field | `Option<T>` | `Union{T, Nothing}` | Optionaler Wert |
| `float` mit `NaN` | `float64` | `f64::NAN` | `NaN` | Invalider numerischer Wert |

### Unterscheidung: None vs NaN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       None vs NaN Semantik                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  None/null/nothing:                                                          │
│  ├─ Bedeutung: Wert existiert nicht / wurde nicht gesetzt                   │
│  ├─ Beispiel: exit_time einer offenen Position                              │
│  └─ Verwendung: Optional fields, nullable columns                           │
│                                                                              │
│  NaN (Not a Number):                                                         │
│  ├─ Bedeutung: Numerischer Wert ist undefiniert (z.B. 0/0)                  │
│  ├─ Beispiel: Indikator-Wert während Warmup-Phase                           │
│  └─ Verwendung: Numerische Arrays, kontinuierliche Daten                    │
│                                                                              │
│  WICHTIG: NaN ≠ None                                                        │
│  - NaN ist ein gültiger IEEE 754 Float-Wert                                 │
│  - None ist ein separater Zustand (kein Wert)                               │
│  - Arrow nullable + NaN sind orthogonal                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FFI Nullability Patterns

### Pattern 1: Optional Scalar (None → Option<T>)

**Verwendung**: Einzelne optionale Werte

```python
# Python
def get_exit_price(position_id: str) -> float | None:
    """Returns exit price or None if position is still open."""
    ...
```

```rust
// Rust (PyO3)
#[pyfunction]
fn get_exit_price(position_id: &str) -> Option<f64> {
    // Returns Some(price) or None
}
```

**Arrow Schema**: `pa.field("exit_price", pa.float64(), nullable=True)`

### Pattern 2: Validity Mask (NaN + valid flag)

**Verwendung**: Numerische Arrays mit fehlenden Werten

```python
# Python
@dataclass
class IndicatorSeries:
    values: np.ndarray  # float64, may contain NaN
    valid: np.ndarray   # bool, True where value is valid

# Oder: Arrow RecordBatch mit validity mask
batch = pa.RecordBatch.from_arrays(
    [values_array, valid_array],
    schema=pa.schema([
        pa.field("value", pa.float64(), nullable=False),  # NaN statt null
        pa.field("valid", pa.bool_(), nullable=False),
    ])
)
```

**Warum Validity Mask statt nullable**:
- Zero-Copy Transfer (keine Null-Prüfung pro Element)
- Explizite Kontrolle über Gültigkeit
- Kompatibel mit NumPy (NaN ist Float, null nicht)

### Pattern 3: Arrow Nullable Column

**Verwendung**: Strukturierte Daten mit optionalen Feldern

```python
# Arrow Schema
position_schema = pa.schema([
    pa.field("entry_time", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("exit_time", pa.timestamp("us", tz="UTC"), nullable=True),  # None wenn offen
    pa.field("result", pa.float64(), nullable=True),  # None wenn offen
])
```

**Rust-Seite**:
```rust
// Arrow nullable columns werden automatisch zu Option<T>
let exit_time: Option<DateTime<Utc>> = batch.column("exit_time").get(i);
```

### Pattern 4: Default Values (Vermeidung von None)

**Verwendung**: Wenn sinnvoller Default existiert

```python
# Python
def calculate_score(
    trades: list[Trade],
    min_trades: int = 10,  # Default statt Optional
) -> float:
    ...
```

**Konvention**: Verwende Defaults wenn:
- Sinnvoller Default-Wert existiert
- None-Handling unnötige Komplexität hinzufügt
- API-Konsistenz verbessert wird

---

## Nullability nach Datentyp

### OHLCV-Daten (Candles)

| Feld | Nullable | Behandlung |
|------|----------|------------|
| timestamp | Nein | Pflichtfeld |
| open | Nein | NaN für fehlende Bars |
| high | Nein | NaN für fehlende Bars |
| low | Nein | NaN für fehlende Bars |
| close | Nein | NaN für fehlende Bars |
| volume | Nein | NaN oder 0.0 |
| valid | Nein | Boolean Mask |

**Begründung**: Numerische Felder verwenden NaN + valid mask für Zero-Copy.

### Trade Signals

| Feld | Nullable | Behandlung |
|------|----------|------------|
| timestamp | Nein | Pflichtfeld |
| direction | Nein | "long" \| "short" |
| entry | Nein | Pflichtfeld |
| sl | Nein | Pflichtfeld |
| tp | Nein | Pflichtfeld |
| size | Nein | Pflichtfeld |
| symbol | Nein | Pflichtfeld |
| reason | Ja | None wenn nicht gesetzt |
| scenario | Ja | None wenn nicht gesetzt |
| meta | Ja | Leeres Dict wenn nicht gesetzt |

**Begründung**: Kern-Felder sind Pflicht, Metadata ist optional.

### Positions

| Feld | Nullable | Behandlung |
|------|----------|------------|
| entry_time | Nein | Pflichtfeld |
| exit_time | Ja | None wenn offen |
| entry_price | Nein | Pflichtfeld |
| exit_price | Ja | None wenn offen |
| result | Ja | None wenn offen |
| status | Nein | "open" \| "pending" \| "closed" |

**Begründung**: Exit-bezogene Felder sind null bis Position geschlossen.

### Indicator Series

| Feld | Nullable | Behandlung |
|------|----------|------------|
| timestamp | Nein | Aligned mit Candles |
| value | Nein | NaN während Warmup |
| valid | Nein | False während Warmup |

**Begründung**: Validity Mask statt nullable für Performance.

---

## Konventionen nach Sprache

### Python

```python
# Typisierung
from typing import Optional

# Pattern 1: Optional[T] für nullable
def get_exit_price(pos_id: str) -> Optional[float]:
    ...

# Pattern 2: NaN für numerische Arrays
def get_indicator(tf: str) -> np.ndarray:  # may contain NaN
    ...

# Pattern 3: Default statt Optional
def calculate(data: np.ndarray, period: int = 20) -> np.ndarray:
    ...

# Validation
def process_signal(signal: TradeSignalDict) -> None:
    if signal.get("entry") is None:
        raise ValidationError("entry is required", field="entry")
```

### Rust

```rust
// Pattern 1: Option<T> für nullable
fn get_exit_price(pos_id: &str) -> Option<f64> {
    ...
}

// Pattern 2: f64::NAN für numerische Daten
fn get_indicator(tf: &str) -> Vec<f64> {
    // May contain NAN
}

// Pattern 3: Default
fn calculate(data: &[f64], period: Option<usize>) -> Vec<f64> {
    let period = period.unwrap_or(20);
    ...
}

// Null-Safety
fn process_signal(signal: &TradeSignal) -> Result<(), OmegaError> {
    // Rust's type system prevents null dereference
}
```

### Julia

```julia
# Pattern 1: Union{T, Nothing} für nullable
function get_exit_price(pos_id::String)::Union{Float64, Nothing}
    ...
end

# Pattern 2: NaN für numerische Daten
function get_indicator(tf::String)::Vector{Float64}
    # May contain NaN
end

# Pattern 3: Default
function calculate(data::Vector{Float64}; period::Int=20)::Vector{Float64}
    ...
end
```

---

## Arrow Nullable Handling

### Schema-Definition

```python
import pyarrow as pa

# Nicht-nullable Felder
pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False)
pa.field("price", pa.float64(), nullable=False)

# Nullable Felder
pa.field("exit_time", pa.timestamp("us", tz="UTC"), nullable=True)
pa.field("reason", pa.utf8(), nullable=True)
```

### Null-Check Patterns

```python
# Python: Prüfung auf null in Arrow Column
column = batch.column("exit_time")
for i in range(len(column)):
    if column[i].is_valid:
        value = column[i].as_py()
    else:
        value = None  # Null
```

```rust
// Rust: Prüfung auf null
let column = batch.column("exit_time");
for i in 0..column.len() {
    if column.is_valid(i) {
        let value: DateTime<Utc> = column.value(i);
    } else {
        // Null handling
    }
}
```

---

## Nullability Validation

### Input Validation

```python
# src/shared/validation.py

from typing import Any, TypeVar
from .exceptions import ValidationError
from .error_codes import ErrorCode

T = TypeVar("T")

def require_not_none(value: T | None, field: str) -> T:
    """Validiert dass Wert nicht None ist.
    
    Args:
        value: Zu prüfender Wert
        field: Feldname für Fehlermeldung
        
    Returns:
        Wert wenn nicht None
        
    Raises:
        ValidationError: Wenn value None ist
    """
    if value is None:
        raise ValidationError(
            f"Required field '{field}' is None",
            field=field,
            error_code=ErrorCode.NULL_POINTER,
        )
    return value


def validate_no_nan(array: np.ndarray, field: str) -> None:
    """Validiert dass Array keine NaN-Werte enthält.
    
    Args:
        array: NumPy Array
        field: Feldname für Fehlermeldung
        
    Raises:
        ValidationError: Wenn NaN vorhanden
    """
    if np.any(np.isnan(array)):
        nan_count = np.sum(np.isnan(array))
        raise ValidationError(
            f"Field '{field}' contains {nan_count} NaN values",
            field=field,
            error_code=ErrorCode.NAN_RESULT,
        )
```

### Output Guarantees

```python
# Dokumentation von Nullability in Docstrings

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Berechnet EMA.
    
    Args:
        prices: Close-Preise (keine NaN erlaubt)
        period: EMA-Periode
        
    Returns:
        EMA-Werte. Die ersten (period-1) Werte sind NaN (Warmup).
        
    Nullability:
        - Input: prices darf keine NaN enthalten
        - Output: Erste (period-1) Werte sind NaN, Rest garantiert nicht-NaN
    """
```

---

## Checkliste für Nullability

### Bei neuen Funktionen/APIs

- [ ] Ist jeder Parameter als nullable/nicht-nullable dokumentiert?
- [ ] Sind Defaults für optionale Parameter definiert?
- [ ] Wird Input auf unerwartete None/NaN validiert?
- [ ] Ist Output-Nullability dokumentiert?
- [ ] Sind Arrow-Schema-Felder korrekt als nullable markiert?

### Bei FFI-Grenzen

- [ ] Ist None → Option<T> Mapping korrekt?
- [ ] Wird NaN vs null konsistent behandelt?
- [ ] Sind Validity Masks für numerische Arrays definiert?
- [ ] Ist Null-Handling in Rust panic-safe?

---

## Referenzen

- [Apache Arrow Null Handling](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps)
- [Rust Option<T>](https://doc.rust-lang.org/std/option/)
- [Julia Nothing vs Missing](https://docs.julialang.org/en/v1/manual/missing/)
- [ADR-0002: Serialisierungsformat](../adr/ADR-0002-serialization-format.md)
- [ADR-0003: Error Handling](../adr/ADR-0003-error-handling.md)
