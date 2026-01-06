# FFI-Spezifikation: Slippage & Fee Models

**Modul:** `src/backtest_engine/core/slippage.py`, `src/backtest_engine/core/fee.py`  
**Task-ID:** P2-XX  
**Status:** Dokumentiert (2026-01-06)

---

## 1. Übersicht

Slippage- und Fee-Module berechnen realistische Ausführungskosten für Trades. Sie sind ideale Pilot-Kandidaten: reine Mathematik, keine Abhängigkeiten, gut testbar.

### 1.1 Verantwortlichkeiten

- Slippage-Berechnung basierend auf Spread, Volatilität, Volume
- Fee-Berechnung (Commission, Swap, Spread-Markup)
- Unterstützung verschiedener Kostenmodelle pro Broker/Symbol
- Batch-Berechnung für Optimizer-Szenarien

---

## 2. Python Interface

```python
class SlippageModel(Protocol):
    """Protocol für Slippage-Modelle."""
    
    def calculate_slippage(
        self,
        price: float,
        size: float,
        direction: Direction,
        spread: float,
        volatility: float | None = None,
    ) -> float:
        """Berechne Slippage in Preis-Units."""
        ...

class FeeModel(Protocol):
    """Protocol für Fee-Modelle."""
    
    def calculate_fee(
        self,
        price: float,
        size: float,
        symbol: str,
        holding_days: float = 0.0,
    ) -> float:
        """Berechne Gesamtkosten (Commission + Swap)."""
        ...

@dataclass
class ExecutionCosts:
    """Aggregierte Ausführungskosten."""
    
    slippage: float
    commission: float
    swap: float
    spread_cost: float
    total: float
```

---

## 3. Arrow Schema

```python
EXECUTION_COSTS_SCHEMA = pa.schema([
    pa.field("trade_id", pa.utf8(), nullable=False),
    pa.field("slippage", pa.float64(), nullable=False),
    pa.field("commission", pa.float64(), nullable=False),
    pa.field("swap", pa.float64(), nullable=False),
    pa.field("spread_cost", pa.float64(), nullable=False),
    pa.field("total", pa.float64(), nullable=False),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "execution_costs",
})
```

---

## 4. Rust Interface

```rust
/// Slippage calculation result
#[derive(Clone, Debug)]
pub struct SlippageResult {
    pub slippage_price: f64,  // Price after slippage
    pub slippage_amount: f64, // Absolute slippage
}

/// Fee calculation result
#[derive(Clone, Debug)]
pub struct FeeResult {
    pub commission: f64,
    pub swap: f64,
    pub total: f64,
}

#[pyfunction]
/// Calculate slippage for a batch of trades
/// 
/// Uses SIMD for parallel calculation
pub fn calculate_slippage_batch(
    prices: Vec<f64>,
    sizes: Vec<f64>,
    directions: Vec<i8>,  // 1=Long, -1=Short
    spreads: Vec<f64>,
    model: &str,  // "fixed", "proportional", "volatility"
    params: HashMap<String, f64>,
) -> PyResult<Vec<f64>>;  // Slippage amounts

#[pyfunction]
/// Calculate fees for a batch of trades
pub fn calculate_fee_batch(
    prices: Vec<f64>,
    sizes: Vec<f64>,
    symbols: Vec<String>,
    holding_days: Vec<f64>,
    fee_config_ipc: &[u8],  // Arrow IPC with fee configuration
) -> PyResult<Vec<u8>>;  // Arrow IPC with FeeResults
```

---

## 5. Slippage-Modelle

### 5.1 Fixed Slippage

```rust
pub fn fixed_slippage(spread: f64, direction: Direction) -> f64 {
    match direction {
        Direction::Long => spread / 2.0,
        Direction::Short => -spread / 2.0,
    }
}
```

### 5.2 Proportional Slippage

```rust
pub fn proportional_slippage(
    price: f64,
    size: f64,
    base_slippage: f64,
    size_factor: f64,
) -> f64 {
    base_slippage * (1.0 + size * size_factor)
}
```

### 5.3 Volatility-Based Slippage

```rust
pub fn volatility_slippage(
    spread: f64,
    atr: f64,
    volatility_factor: f64,
) -> f64 {
    spread + atr * volatility_factor
}
```

---

## 6. Error Handling

| Code | Name | Beschreibung |
| ---- | ---- | ------------ |
| `5001` | `INVALID_SLIPPAGE_MODEL` | Unbekanntes Slippage-Modell |
| `5002` | `INVALID_FEE_CONFIG` | Fee-Konfiguration ungültig |
| `5003` | `SYMBOL_NOT_IN_CONFIG` | Symbol nicht in Fee-Config |
| `5004` | `NEGATIVE_SIZE` | Trade-Size darf nicht negativ sein |

---

## 7. Performance-Targets

| Operation | Python Baseline | Rust Target | Speedup |
| --------- | --------------- | ----------- | ------- |
| Slippage (single) | 0.02ms | 0.001ms | 20x |
| Slippage (batch 1K) | 15ms | 0.5ms | 30x |
| Fee (single) | 0.03ms | 0.002ms | 15x |
| Fee (batch 1K) | 25ms | 1ms | 25x |

---

## 8. Referenzen

- Config: `configs/execution_costs.yaml`
- Symbol-Specs: `configs/symbol_specs.yaml`
- Arrow-Schemas: `src/shared/arrow_schemas.py`
