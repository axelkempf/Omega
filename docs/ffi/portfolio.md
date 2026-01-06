# FFI-Spezifikation: Portfolio

**Modul:** `src/backtest_engine/core/portfolio.py`  
**Task-ID:** P2-XX  
**Status:** Dokumentiert (2026-01-06)

---

## 1. Übersicht

Portfolio verwaltet den Zustand aller offenen Positionen und die Equity-Kurve während des Backtests. Es ist die zentrale State-Machine für Kapitalverwaltung.

### 1.1 Verantwortlichkeiten

- Positionsverwaltung (Open, Close, Modify)
- Equity-Tracking und Drawdown-Berechnung
- Margin-Berechnung und Risiko-Management
- Trade-History und Performance-Metrics

---

## 2. Python Interface

```python
@dataclass
class PortfolioState:
    """Snapshot des Portfolio-Zustands."""
    
    balance: Decimal
    equity: Decimal
    margin_used: Decimal
    free_margin: Decimal
    positions: dict[str, Position]
    unrealized_pnl: Decimal
    realized_pnl: Decimal

class Portfolio:
    """Portfolio State Manager."""
    
    def __init__(self, initial_balance: Decimal, leverage: float = 100.0) -> None: ...
    
    def open_position(self, signal: TradeSignal, entry_price: float) -> Position: ...
    
    def close_position(self, position_id: str, exit_price: float, exit_time: datetime) -> Trade: ...
    
    def get_state(self) -> PortfolioState: ...
    
    def get_equity_curve(self) -> pd.DataFrame: ...
    
    def calculate_metrics(self) -> PerformanceMetrics: ...
```

---

## 3. Arrow Schema

```python
PORTFOLIO_STATE_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("balance", pa.float64(), nullable=False),
    pa.field("equity", pa.float64(), nullable=False),
    pa.field("margin_used", pa.float64(), nullable=False),
    pa.field("unrealized_pnl", pa.float64(), nullable=False),
    pa.field("realized_pnl", pa.float64(), nullable=False),
    pa.field("position_count", pa.int32(), nullable=False),
    pa.field("drawdown", pa.float64(), nullable=False),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "portfolio_state",
})

POSITION_SCHEMA = pa.schema([
    pa.field("position_id", pa.utf8(), nullable=False),
    pa.field("symbol", pa.utf8(), nullable=False),
    pa.field("magic_number", pa.int64(), nullable=False),
    pa.field("direction", pa.int8(), nullable=False),  # 1=Long, -1=Short
    pa.field("size", pa.float64(), nullable=False),
    pa.field("entry_price", pa.float64(), nullable=False),
    pa.field("entry_time", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("current_price", pa.float64(), nullable=True),
    pa.field("unrealized_pnl", pa.float64(), nullable=True),
    pa.field("take_profit", pa.float64(), nullable=True),
    pa.field("stop_loss", pa.float64(), nullable=True),
], metadata={
    "schema_version": "1.0.0",
    "schema_name": "position",
})
```

---

## 4. Rust Interface

```rust
#[pyclass]
pub struct PortfolioRust {
    state: PortfolioState,
    positions: HashMap<String, Position>,
    equity_history: Vec<EquityPoint>,
    trade_history: Vec<Trade>,
}

#[pymethods]
impl PortfolioRust {
    #[new]
    pub fn new(initial_balance: f64, leverage: f64) -> Self;
    
    /// Open new position, returns position ID
    pub fn open_position(&mut self, signal_ipc: &[u8], entry_price: f64) -> PyResult<String>;
    
    /// Close position, returns Trade as Arrow IPC
    pub fn close_position(&mut self, position_id: &str, exit_price: f64, exit_time: i64) -> PyResult<Vec<u8>>;
    
    /// Get current state as Arrow IPC
    pub fn get_state_arrow(&self) -> PyResult<Vec<u8>>;
    
    /// Get all positions as Arrow IPC
    pub fn get_positions_arrow(&self) -> PyResult<Vec<u8>>;
    
    /// Calculate performance metrics
    pub fn calculate_metrics(&self) -> PyResult<MetricsPy>;
}
```

---

## 5. Error Handling

| Code | Name | Beschreibung |
| ---- | ---- | ------------ |
| `4001` | `POSITION_NOT_FOUND` | Position ID existiert nicht |
| `4002` | `INSUFFICIENT_MARGIN` | Nicht genug freie Margin |
| `4003` | `INVALID_POSITION_SIZE` | Size <= 0 oder zu groß |
| `4004` | `DUPLICATE_POSITION_ID` | Position ID bereits vergeben |

---

## 6. Performance-Targets

| Operation | Python Baseline | Rust Target | Speedup |
| --------- | --------------- | ----------- | ------- |
| open_position | 0.5ms | 0.05ms | 10x |
| close_position | 0.4ms | 0.04ms | 10x |
| get_state | 0.2ms | 0.02ms | 10x |
| calculate_metrics | 15ms | 2ms | 7x |

---

## 7. Referenzen

- Arrow-Schemas: `src/shared/arrow_schemas.py`
- Execution: `docs/ffi/execution_simulator.md`
