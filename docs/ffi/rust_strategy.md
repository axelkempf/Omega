# FFI Specification: Rust Strategy

**Version:** 1.0  
**Status:** Proposed  
**Module:** `src/rust_modules/omega_rust/src/strategy/`  
**Related ADR:** [ADR-0006: Pure Rust Strategies](../adr/ADR-0006-wave4-pure-rust-strategies.md)

---

## Overview

Diese Spezifikation definiert das FFI-Interface für Pure Rust Strategies. Das Ziel ist die Reduktion von ~150.000 FFI-Calls pro Backtest auf **exakt 2 FFI-Calls** (Init + Result).

### FFI-Reduktionsstrategie

| Phase | Aktuell (Wave 3) | Wave 4 (Pure Rust) |
|-------|------------------|-------------------|
| Strategy Init | 1 Call | 1 Call |
| Per-Bar Evaluation | 150.000 Calls | 0 Calls |
| Result Collection | - | 1 Call |
| **Total** | ~150.001 | **2** |

---

## Type Definitions

### Core Types

#### DataSlice

Repräsentiert einen einzelnen Zeitpunkt im Backtest mit allen verfügbaren Daten.

```rust
/// Arrow Schema: DATA_SLICE_SCHEMA
#[derive(Clone, Debug)]
pub struct DataSlice {
    /// Symbol (z.B. "EURUSD")
    pub symbol: String,
    
    /// Primärer Timeframe
    pub timeframe: Timeframe,
    
    /// Index im Candle-Array (0-indexed)
    pub index: usize,
    
    /// Unix Timestamp in Microseconds
    pub timestamp_us: i64,
    
    /// Bid Candle
    pub bid: CandleData,
    
    /// Ask Candle
    pub ask: CandleData,
}

impl DataSlice {
    /// Spread in Pips
    pub fn spread_pips(&self) -> f64 {
        (self.ask.close - self.bid.close) * 10_000.0
    }
}
```

**Arrow Schema:**

```
DATA_SLICE_SCHEMA = pa.schema([
    ("symbol", pa.utf8()),
    ("timeframe", pa.utf8()),
    ("index", pa.uint64()),
    ("timestamp_us", pa.int64()),
    ("bid_open", pa.float64()),
    ("bid_high", pa.float64()),
    ("bid_low", pa.float64()),
    ("bid_close", pa.float64()),
    ("bid_volume", pa.float64()),
    ("ask_open", pa.float64()),
    ("ask_high", pa.float64()),
    ("ask_low", pa.float64()),
    ("ask_close", pa.float64()),
    ("ask_volume", pa.float64()),
])
```

#### CandleData

```rust
#[derive(Clone, Debug, Copy)]
pub struct CandleData {
    pub timestamp_us: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
```

#### Timeframe

```rust
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1,
}

impl Timeframe {
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::M1 => "M1",
            Timeframe::M5 => "M5",
            Timeframe::M15 => "M15",
            Timeframe::M30 => "M30",
            Timeframe::H1 => "H1",
            Timeframe::H4 => "H4",
            Timeframe::D1 => "D1",
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "M1" => Some(Timeframe::M1),
            "M5" => Some(Timeframe::M5),
            "M15" => Some(Timeframe::M15),
            "M30" => Some(Timeframe::M30),
            "H1" => Some(Timeframe::H1),
            "H4" => Some(Timeframe::H4),
            "D1" => Some(Timeframe::D1),
            _ => None,
        }
    }
}
```

#### TradeSignal

```rust
/// Generiert von RustStrategy::evaluate()
#[derive(Clone, Debug)]
pub enum TradeSignal {
    Long {
        entry_price: f64,
        stop_loss: f64,
        take_profit: Option<f64>,
        reason: String,
        tags: Vec<String>,
        scenario: Option<u32>,
        meta: Option<HashMap<String, f64>>,
    },
    Short {
        entry_price: f64,
        stop_loss: f64,
        take_profit: Option<f64>,
        reason: String,
        tags: Vec<String>,
        scenario: Option<u32>,
        meta: Option<HashMap<String, f64>>,
    },
}

impl TradeSignal {
    pub fn direction(&self) -> &'static str {
        match self {
            TradeSignal::Long { .. } => "long",
            TradeSignal::Short { .. } => "short",
        }
    }
    
    pub fn entry_price(&self) -> f64 {
        match self {
            TradeSignal::Long { entry_price, .. } => *entry_price,
            TradeSignal::Short { entry_price, .. } => *entry_price,
        }
    }
}
```

**Arrow Schema:**

```
TRADE_SIGNAL_SCHEMA = pa.schema([
    ("direction", pa.utf8()),          # "long" | "short"
    ("entry_price", pa.float64()),
    ("stop_loss", pa.float64()),
    ("take_profit", pa.float64()),     # nullable
    ("reason", pa.utf8()),
    ("tags", pa.list_(pa.utf8())),
    ("scenario", pa.uint32()),         # nullable
    ("timestamp_us", pa.int64()),
])
```

#### Position

```rust
#[derive(Clone, Debug)]
pub struct Position {
    pub id: u64,
    pub symbol: String,
    pub direction: Direction,
    pub entry_price: f64,
    pub entry_time_us: i64,
    pub stop_loss: f64,
    pub take_profit: Option<f64>,
    pub lot_size: f64,
    pub pnl_pips: f64,
    pub pnl_usd: f64,
    pub bars_held: usize,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum Direction {
    Long,
    Short,
}
```

#### PositionAction

```rust
/// Rückgabe von RustStrategy::manage_position()
#[derive(Clone, Debug)]
pub enum PositionAction {
    /// Position halten, keine Änderung
    Hold,
    
    /// Stop-Loss anpassen (Trailing Stop)
    ModifyStopLoss { new_sl: f64 },
    
    /// Take-Profit anpassen
    ModifyTakeProfit { new_tp: f64 },
    
    /// Position schließen
    Close { reason: String },
}
```

---

## Trait Definition

### RustStrategy Trait

```rust
/// Core Trait für alle Rust-Strategien
/// 
/// Strategien MÜSSEN diesen Trait implementieren um im Backtest-Loop
/// ausgeführt zu werden.
pub trait RustStrategy: Send + Sync {
    /// Evaluiert einen einzelnen Bar und generiert optional ein Trade-Signal
    ///
    /// # Arguments
    /// * `slice` - Der aktuelle Daten-Slice mit Bid/Ask Candles
    /// * `cache` - IndicatorCache für Indikator-Abfragen
    ///
    /// # Returns
    /// * `Some(TradeSignal)` wenn ein Trade geöffnet werden soll
    /// * `None` wenn kein Trade
    ///
    /// # Garantien
    /// * Diese Methode wird für JEDEN Bar aufgerufen (kein Skip)
    /// * Die Methode ist side-effect-frei (pure function)
    /// * Der IndicatorCache enthält bereits alle pre-computed Werte
    fn evaluate(&self, slice: &DataSlice, cache: &IndicatorCache) -> Option<TradeSignal>;
    
    /// Verwaltet eine offene Position
    ///
    /// # Arguments
    /// * `position` - Die aktuelle Position
    /// * `slice` - Der aktuelle Daten-Slice
    ///
    /// # Returns
    /// * `PositionAction::Hold` - Keine Änderung
    /// * `PositionAction::ModifyStopLoss` - Stop anpassen
    /// * `PositionAction::Close` - Position schließen
    fn manage_position(&self, position: &Position, slice: &DataSlice) -> PositionAction;
    
    /// Gibt den primären Timeframe der Strategie zurück
    fn primary_timeframe(&self) -> Timeframe;
    
    /// Optional: Custom Initialisierung nach Config-Load
    fn on_init(&mut self, _config: &StrategyConfig) -> Result<(), StrategyError> {
        Ok(())
    }
    
    /// Optional: Cleanup bei Strategy-Deallokation
    fn on_deinit(&mut self) {
        // Default: no-op
    }
    
    /// Gibt den Namen der Strategie zurück (für Logging)
    fn name(&self) -> &str;
    
    /// Gibt die Version der Strategie zurück
    fn version(&self) -> &str {
        "1.0.0"
    }
}
```

---

## PyO3 Bindings

### Strategy Configuration

```rust
#[pyclass]
#[derive(Clone, Debug)]
pub struct StrategyConfig {
    #[pyo3(get, set)]
    pub symbol: String,
    
    #[pyo3(get, set)]
    pub timeframe: String,
    
    #[pyo3(get, set)]
    pub enabled_scenarios: Vec<u32>,
    
    #[pyo3(get, set)]
    pub direction_filter: String,  // "long" | "short" | "both"
    
    // Strategy-specific parameters als JSON
    #[pyo3(get, set)]
    pub params: HashMap<String, PyObject>,
}

#[pymethods]
impl StrategyConfig {
    #[new]
    pub fn new(symbol: String, timeframe: String) -> Self {
        Self {
            symbol,
            timeframe,
            enabled_scenarios: vec![1, 2, 3, 4, 5, 6],
            direction_filter: "both".to_string(),
            params: HashMap::new(),
        }
    }
    
    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        serde_json::from_str(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}
```

### Backtest Runner

```rust
/// Haupteinstiegspunkt für Python
#[pyfunction]
#[pyo3(signature = (strategy_name, config, candle_data))]
pub fn run_backtest_rust(
    py: Python<'_>,
    strategy_name: &str,
    config: StrategyConfig,
    candle_data: &PyAny,  // Arrow IPC RecordBatch
) -> PyResult<BacktestResult> {
    // 1. Strategy aus Registry laden
    let strategy = STRATEGY_REGISTRY
        .get(strategy_name)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown strategy: {}", strategy_name)
        ))?;
    
    // 2. Candle-Daten parsen (Zero-Copy Arrow)
    let candles = parse_arrow_candles(candle_data)?;
    
    // 3. IndicatorCache initialisieren
    let mut cache = IndicatorCache::new(&candles);
    
    // 4. Backtest-Loop (komplett in Rust)
    let results = execute_backtest(strategy, &candles, &mut cache, &config)?;
    
    // 5. Ergebnisse zurückgeben
    Ok(results)
}

#[pyclass]
pub struct BacktestResult {
    #[pyo3(get)]
    pub trades: Vec<TradeResult>,
    
    #[pyo3(get)]
    pub equity_curve: Vec<f64>,
    
    #[pyo3(get)]
    pub metrics: HashMap<String, f64>,
    
    #[pyo3(get)]
    pub bars_processed: usize,
    
    #[pyo3(get)]
    pub execution_time_ms: f64,
}
```

### Strategy Registry

```rust
lazy_static::lazy_static! {
    pub static ref STRATEGY_REGISTRY: HashMap<String, Box<dyn RustStrategy>> = {
        let mut m = HashMap::new();
        
        // Mean Reversion Z-Score
        m.insert(
            "mean_reversion_z_score".to_string(),
            Box::new(MeanReversionZScoreRust::default()) as Box<dyn RustStrategy>
        );
        
        // Weitere Strategien hier registrieren
        
        m
    };
}
```

---

## Python Bridge API

### Python-seitige Integration

```python
# src/backtest_engine/core/rust_strategy_bridge.py

from typing import Any, Dict, List, Optional
import os

# Lazy import
_RUST_MODULE = None

def _get_rust_module():
    global _RUST_MODULE
    if _RUST_MODULE is None:
        try:
            import omega_rust
            _RUST_MODULE = omega_rust
        except ImportError:
            return None
    return _RUST_MODULE


def is_rust_strategy_available(strategy_name: str) -> bool:
    """Prüft ob eine Rust-Implementierung für die Strategie existiert."""
    rust = _get_rust_module()
    if rust is None:
        return False
    return hasattr(rust, "STRATEGY_REGISTRY") and strategy_name in rust.STRATEGY_REGISTRY


def should_use_rust_strategy() -> bool:
    """Prüft ob Rust-Strategien verwendet werden sollen."""
    flag = os.environ.get("OMEGA_USE_RUST_STRATEGY", "auto").lower()
    if flag == "false" or flag == "0":
        return False
    if flag == "true" or flag == "1":
        return True
    # auto: nur wenn verfügbar
    return _get_rust_module() is not None


def run_rust_backtest(
    strategy_name: str,
    config: Dict[str, Any],
    candle_data: Any,  # Arrow RecordBatch
) -> "BacktestResult":
    """
    Führt einen vollständigen Backtest in Rust aus.
    
    Args:
        strategy_name: Name der registrierten Rust-Strategie
        config: Strategy-Konfiguration
        candle_data: Arrow IPC RecordBatch mit OHLCV Daten
        
    Returns:
        BacktestResult mit trades, equity_curve, metrics
        
    Raises:
        ValueError: Wenn Strategie nicht gefunden
        RuntimeError: Bei Backtest-Fehler
    """
    rust = _get_rust_module()
    if rust is None:
        raise RuntimeError("omega_rust module not available")
    
    rust_config = rust.StrategyConfig.from_json(json.dumps(config))
    return rust.run_backtest_rust(strategy_name, rust_config, candle_data)


def get_active_backend() -> str:
    """Gibt 'rust' oder 'python' zurück für CI-Verifikation."""
    if should_use_rust_strategy():
        return "rust"
    return "python"
```

---

## Error Handling

### Rust Error Types

```rust
#[derive(Debug, Clone)]
pub enum StrategyError {
    /// Konfigurationsfehler
    ConfigError { reason: String },
    
    /// Ungültiger Parameter
    InvalidParameter { name: String, reason: String },
    
    /// Indikator-Fehler
    IndicatorError { indicator: String, reason: String },
    
    /// Daten-Fehler
    DataError { reason: String },
    
    /// Interner Fehler
    InternalError { reason: String },
}

impl From<StrategyError> for PyErr {
    fn from(e: StrategyError) -> PyErr {
        match e {
            StrategyError::ConfigError { reason } => 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(reason),
            StrategyError::InvalidParameter { name, reason } => 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid parameter '{}': {}", name, reason)
                ),
            StrategyError::IndicatorError { indicator, reason } => 
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Indicator '{}' error: {}", indicator, reason)
                ),
            StrategyError::DataError { reason } => 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(reason),
            StrategyError::InternalError { reason } => 
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(reason),
        }
    }
}
```

### Error Codes

| Code | Name | Beschreibung |
|------|------|--------------|
| E4001 | STRATEGY_NOT_FOUND | Strategie nicht in Registry |
| E4002 | INVALID_CONFIG | Ungültige Konfiguration |
| E4003 | INDICATOR_NOT_AVAILABLE | Indikator nicht berechnet |
| E4004 | INSUFFICIENT_DATA | Nicht genug Warmup-Daten |
| E4005 | POSITION_LIMIT_EXCEEDED | Zu viele offene Positionen |
| E4006 | INVALID_SIGNAL | Ungültiges Trade-Signal |

---

## Performance Constraints

### Timing Requirements

| Operation | Max Latency | Beschreibung |
|-----------|-------------|--------------|
| `evaluate()` | 1μs | Pro-Bar Strategy Evaluation |
| `manage_position()` | 500ns | Position Management |
| `on_init()` | 100ms | Strategy Initialisierung |
| Full Backtest (30k bars) | 3.4s | Gesamt-Backtest |

### Memory Requirements

| Component | Max Memory | Beschreibung |
|-----------|------------|--------------|
| IndicatorCache | 50 MB | Alle pre-computed Indikatoren |
| Position State | 10 KB | Alle offenen Positionen |
| Trade History | 5 MB | Alle geschlossenen Trades |
| **Total** | ~80 MB | Peak RAM (vs 118 MB Python) |

### Optimization Guidelines

1. **Keine Heap-Allokation in `evaluate()`**: Alle Strings pre-allokiert
2. **SIMD für Batch-Operationen**: nutze `std::simd` wo möglich
3. **Cache-friendly Memory Layout**: Struct of Arrays für Candle-Daten
4. **Inline häufig aufgerufene Funktionen**: `#[inline(always)]`

---

## Testing Requirements

### Parity Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_parity_with_python() {
        // Lade Golden-File mit Python-Ergebnissen
        let python_signals = load_golden_signals("mean_reversion_zscore_v1.json");
        
        // Führe Rust-Strategie aus
        let rust_signals = run_test_backtest();
        
        // Vergleiche Signal-für-Signal
        assert_eq!(rust_signals.len(), python_signals.len());
        for (rust, python) in rust_signals.iter().zip(python_signals.iter()) {
            assert_eq!(rust.direction(), python.direction);
            assert!((rust.entry_price() - python.entry_price).abs() < 1e-6);
            assert!((rust.stop_loss() - python.stop_loss).abs() < 1e-6);
        }
    }
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_evaluate_never_panics(
        close in 1.0..2.0f64,
        atr in 0.0001..0.01f64,
        zscore in -5.0..5.0f64,
    ) {
        let slice = DataSlice::mock(close);
        let cache = IndicatorCache::mock(atr, zscore);
        let strategy = MeanReversionZScoreRust::default();
        
        // Should never panic
        let _ = strategy.evaluate(&slice, &cache);
    }
}
```

---

## Nullability Convention

| Field | Nullable | Default |
|-------|----------|---------|
| `take_profit` | ✅ | None |
| `scenario` | ✅ | None |
| `meta` | ✅ | None |
| `tags` | ❌ | `vec![]` |
| `reason` | ❌ | `""` |
| `entry_price` | ❌ | Required |
| `stop_loss` | ❌ | Required |

---

## Migration Path

### Phase 1: Foundation (Week 1-2)
- [ ] `strategy/mod.rs` mit Trait-Definition
- [ ] `strategy/types.rs` mit DataSlice, TradeSignal
- [ ] `strategy/registry.rs` mit Strategy-Lookup

### Phase 2: MeanReversionZScore (Week 3-4)
- [ ] Scenario 1 (Long/Short) implementieren
- [ ] Scenario 2 implementieren
- [ ] Parity-Tests für Scenario 1+2

### Phase 3: Full Strategy (Week 5-6)
- [ ] Scenarios 3-6 implementieren
- [ ] Position Management
- [ ] Full Parity-Tests

### Phase 4: Integration (Week 7-8)
- [ ] Python-Bridge
- [ ] Feature-Flag
- [ ] Performance-Benchmarks

### Phase 5: Documentation (Week 9)
- [ ] Runbook aktualisieren
- [ ] Beispiele dokumentieren
- [ ] CI/CD Pipeline

---

## References

- [ADR-0006: Pure Rust Strategies](../adr/ADR-0006-wave4-pure-rust-strategies.md)
- [Indicator Cache FFI](./indicator_cache.md)
- [Event Engine FFI](./event_engine.md)
- [Arrow Schema Registry](../../src/shared/arrow_schemas.py)
- [PyO3 0.27 Documentation](https://pyo3.rs/)
