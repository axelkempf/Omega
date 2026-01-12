# FFI Interface Specification: ExecutionSimulator

**Modul:** `src/backtest_engine/core/execution_simulator.py`  
**Migrations-Ziel:** Rust (via PyO3/maturin)  
**Phase 2 Task:** P2-03  
**Status:** ✅ Spezifiziert (2026-01-05)

---

## Executive Summary

`ExecutionSimulator` simuliert die **Orderausführung** (Market, Limit, Stop) im Backtest.
Verantwortlich für:
- Signal-Verarbeitung → Position-Eröffnung
- Entry-Trigger für Pending Orders (Limit/Stop)
- Exit-Evaluation (SL/TP Hit Detection)
- Positionsgrößen-Berechnung (Risk-Based Sizing)
- Slippage und Fee-Anwendung

Migrations-Kandidat aufgrund:
- Numerisch intensive Berechnungen (Sizing, PnL)
- Hohe Aufruffrequenz (pro Bar für alle offenen Positionen)
- Klare State-Machine-Semantik

---

## Data Structures

### Position States

```python
# @ffi_boundary: Input/Output

PositionStatus: TypeAlias = Literal["open", "pending", "closed"]
Direction: TypeAlias = Literal["long", "short"]
OrderType: TypeAlias = Literal["market", "limit", "stop"]
```

### PortfolioPosition (Core State Object)

```python
# @ffi_boundary: Input/Output

@dataclass
class PortfolioPosition:
    """Position-Repräsentation im Backtest."""
    
    # Entry-Daten
    entry_time: datetime
    direction: Direction           # "long" | "short"
    symbol: str
    entry_price: float
    
    # Risk Levels
    stop_loss: float
    take_profit: float
    initial_stop_loss: float       # Ursprünglicher SL (für BE-Tracking)
    initial_take_profit: float     # Ursprünglicher TP
    
    # Sizing
    size: float                    # Lots (0.01 bis max)
    risk_per_trade: float          # Risiko in Konto-Währung
    
    # Order-Info
    order_type: OrderType          # "market" | "limit" | "stop"
    status: PositionStatus         # "open" | "pending" | "closed"
    
    # Exit-Daten (nach Close)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    result: Optional[float] = None  # PnL in Konto-Währung
    reason: Optional[str] = None    # "stop_loss" | "take_profit" | "manual" | ...
    
    # Trigger-Zeit (für Limit/Stop Orders)
    trigger_time: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_closed(self) -> bool:
        return self.status == "closed"
    
    @property
    def r_multiple(self) -> Optional[float]:
        """R-Multiple = PnL / Risiko."""
        if self.result is None or self.risk_per_trade <= 0:
            return None
        return self.result / self.risk_per_trade
```

### Arrow Schema für PortfolioPosition

```
PortfolioPosition {
  entry_time: timestamp[us, tz=UTC]
  direction: utf8 (enum: "long" | "short")
  symbol: utf8
  entry_price: float64
  stop_loss: float64
  take_profit: float64
  initial_stop_loss: float64
  initial_take_profit: float64
  size: float64
  risk_per_trade: float64
  order_type: utf8 (enum: "market" | "limit" | "stop")
  status: utf8 (enum: "open" | "pending" | "closed")
  exit_time: timestamp[us, tz=UTC] (nullable)
  exit_price: float64 (nullable)
  result: float64 (nullable)
  reason: utf8 (nullable)
  trigger_time: timestamp[us, tz=UTC] (nullable)
  metadata: utf8 (JSON-serialized)
}
```

### Symbol Specification

```python
# @ffi_boundary: Input (via Constructor)

@dataclass
class SymbolSpec:
    """Broker-spezifische Symbol-Eigenschaften."""
    
    symbol: str
    pip_size: float              # z.B. 0.0001 für EURUSD
    tick_size: float             # Minimale Preisbewegung
    tick_value: float            # Wert eines Ticks in Konto-Währung
    contract_size: float         # z.B. 100000 für FX
    volume_min: float            # Min Lot Size (z.B. 0.01)
    volume_max: float            # Max Lot Size
    volume_step: float           # Lot-Schrittgröße (z.B. 0.01)
    quote_currency: str          # z.B. "USD" für EURUSD
    base_currency: str           # z.B. "EUR" für EURUSD
```

---

## ExecutionSimulator API

### Constructor

```python
def __init__(
    self,
    portfolio: Portfolio,
    risk_per_trade: float = 100.0,
    slippage_model: Optional[SlippageModel] = None,
    fee_model: Optional[FeeModel] = None,
    symbol_specs: Optional[Union[Dict[str, SymbolSpec], SymbolSpecsRegistry]] = None,
    lot_sizer: Optional[LotSizer] = None,
    commission_model: Optional[CommissionModel] = None,
    rate_provider: Optional[RateProvider] = None,
) -> None:
    """
    Initialisiert Execution Simulator.
    
    @ffi_boundary: Input
    
    Args:
        portfolio: Portfolio-Objekt für Position-Tracking
        risk_per_trade: Fixes Risiko pro Trade (Konto-Währung)
        slippage_model: Optional Slippage-Simulation
        fee_model: Optional Fee-Berechnung (Legacy)
        symbol_specs: Symbol-Spezifikationen (Dict oder Registry)
        lot_sizer: Optional Risk-Based Lot Sizer
        commission_model: Optional Commission Model (bevorzugt)
        rate_provider: Optional FX-Rate-Provider für Währungsumrechnung
    """
```

### State Properties

```python
# @ffi_boundary: Output

active_positions: List[PortfolioPosition]
# Alle nicht-geschlossenen Positionen (open + pending)
```

---

## Signal Processing

### process_signal()

```python
def process_signal(
    self,
    signal: TradeSignal,
) -> None:
    """
    Verarbeitet ein neues Trading-Signal.
    
    @ffi_boundary: Input
    
    Args:
        signal: TradeSignal mit direction, entry, sl, tp, etc.
        
    Behavior nach order_type:
    
    1. Market Order (type="market"):
       - Sofortige Ausführung
       - Slippage wird angewendet
       - Position wird als "open" erstellt
       - Lot-Size wird berechnet (risk-based)
       - Entry-Fee wird registriert
       
    2. Limit/Stop Order (type="limit" | "stop"):
       - Position wird als "pending" erstellt
       - Kein Sizing (size=0)
       - Entry erst bei Trigger (check_if_entry_triggered)
       
    Side Effects:
        - Fügt Position zu active_positions hinzu
        - Registriert Entry bei Portfolio (Market nur)
        - Registriert Fee bei Portfolio (Market nur)
    """
```

### Market Order Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Signal     │────▶│   Slippage   │────▶│   Sizing     │
│   (Market)   │     │   Apply      │     │   Calculate  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Portfolio   │◀────│  Fee Calc    │◀────│  Create      │
│  Register    │     │  & Register  │     │  Position    │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Pending Order Flow

```
┌──────────────┐     ┌──────────────┐
│   Signal     │────▶│   Create     │
│ (Limit/Stop) │     │  Pending     │
└──────────────┘     │  Position    │
                     │  (size=0)    │
                     └──────────────┘
                            │
                            │ (spätere Bar)
                            ▼
                     ┌──────────────┐
                     │   Entry      │
                     │  Triggered?  │
                     └──────┬───────┘
                            │ yes
                            ▼
                     ┌──────────────┐
                     │  trigger_    │
                     │  entry()     │
                     └──────────────┘
```

---

## Entry Trigger Logic

### check_if_entry_triggered() - Candle Mode

```python
def check_if_entry_triggered(
    self,
    pos: PortfolioPosition,
    bid_candle: Candle,
    ask_candle: Optional[Candle] = None,
) -> bool:
    """
    Prüft ob Pending-Entry ausgelöst wird (Candle-Modus).
    
    @ffi_boundary: Input/Output
    
    Args:
        pos: Pending Position
        bid_candle: Aktuelle Bid-Candle
        ask_candle: Aktuelle Ask-Candle (optional)
        
    Returns:
        True wenn Entry getriggert werden soll
        
    Logic:
        - status muss "pending" sein
        - Candle-Timestamp muss > entry_time sein
        
        Limit Order:
          - Long:  ask_candle.low <= entry_price
          - Short: bid_candle.high >= entry_price
          
        Stop Order:
          - Long:  ask_candle.high >= entry_price
          - Short: bid_candle.low <= entry_price
    """
```

### trigger_entry() - Candle Mode

```python
def trigger_entry(
    self,
    pos: PortfolioPosition,
    candle: Candle,
) -> None:
    """
    Setzt Pending-Position auf OPEN.
    
    @ffi_boundary: Input
    
    Args:
        pos: Pending Position
        candle: Trigger-Candle
        
    Side Effects:
        - Berechnet pos.size (risk-based)
        - Setzt pos.status = "open"
        - Setzt pos.trigger_time = candle.timestamp
        - Registriert Entry bei Portfolio
        - Registriert Fee bei Portfolio
    """
```

### Tick-Mode Variants

```python
def check_if_entry_triggered_tick(
    self,
    pos: PortfolioPosition,
    tick: Tick,
) -> bool:
    """Tick-basierte Entry-Prüfung."""

def trigger_entry_tick(
    self,
    pos: PortfolioPosition,
    tick: Tick,
) -> None:
    """Tick-basierter Entry-Trigger."""

def process_signal_tick(
    self,
    signal: TradeSignal,
    tick: Tick,
) -> None:
    """Tick-basierte Signal-Verarbeitung."""
```

---

## Exit Evaluation

### evaluate_exits() - Candle Mode

```python
def evaluate_exits(
    self,
    bid_candle: Candle,
    ask_candle: Optional[Candle] = None,
) -> None:
    """
    Prüft ob offene Positionen geschlossen werden müssen.
    
    @ffi_boundary: Input
    
    Args:
        bid_candle: Aktuelle Bid-Candle
        ask_candle: Aktuelle Ask-Candle (optional)
        
    Exit-Logik pro Position:
    
        1. Skip wenn:
           - Position ist bereits closed
           - Candle-Timestamp <= entry_time
           
        2. Pending Entry Check:
           - Wenn status="pending": check_if_entry_triggered()
           - Bei True: trigger_entry()
           
        3. SL/TP Hit Detection:
           
           Long Position:
             - SL hit: bid_candle.low <= stop_loss + pip_buffer
             - TP hit: bid_candle.high >= take_profit - pip_buffer
             
           Short Position:
             - SL hit: ask_candle.high >= stop_loss - pip_buffer
             - TP hit: ask_candle.low <= take_profit + pip_buffer
             
        4. Entry-Candle Special Case:
           - In der Entry-Candle: nur schließen wenn SL/TP definitiv erreicht
           - Für Limit-Orders: zusätzliche Close-Price-Validierung
           
        5. Exit Execution:
           - Slippage anwenden (wenn model)
           - pos.close(timestamp, exit_price, reason)
           - Fee berechnen und registrieren
           - Portfolio exit registrieren
           
    Side Effects:
        - Modifiziert active_positions (entfernt geschlossene)
        - Registriert Exits bei Portfolio
        - Registriert Fees bei Portfolio
    """
```

### evaluate_exits_tick() - Tick Mode

```python
def evaluate_exits_tick(
    self,
    tick: Tick,
) -> None:
    """
    Tick-basierte Exit-Evaluation.
    
    @ffi_boundary: Input
    
    Exit-Logik:
        Long Position:
          - SL hit: tick.bid <= stop_loss
          - TP hit: tick.bid >= take_profit
          
        Short Position:
          - SL hit: tick.ask >= stop_loss
          - TP hit: tick.ask <= take_profit
    """
```

---

## Position Sizing

### _unit_value_per_price()

```python
def _unit_value_per_price(
    self,
    symbol: str,
) -> float:
    """
    Geldwert pro 1.0 Preis-Einheit für 1 Lot.
    
    @ffi_boundary: Internal
    
    Formel: tick_value / tick_size
    
    Fallback: FX-Konvertierung via RateProvider
    
    Returns:
        float: Unit value in Konto-Währung
        
    Cache: _unit_value_cache[symbol]
    """
```

### _quantize_volume()

```python
def _quantize_volume(
    self,
    symbol: str,
    raw_lots: float,
) -> float:
    """
    Quantisiert Volumen auf Broker-konforme Werte.
    
    @ffi_boundary: Internal
    
    Args:
        symbol: Trading-Symbol
        raw_lots: Berechnete Lot-Size (unquantisiert)
        
    Returns:
        Quantisierte Lot-Size:
        - Gerundet nach unten auf volume_step
        - Mindestens volume_min
        - Maximal volume_max
        
    Formel:
        step = volume_step (z.B. 0.01)
        vmin = volume_min (z.B. 0.01)
        vmax = volume_max (z.B. 100.0)
        
        n_steps = floor((raw_lots - vmin) / step)
        lots = vmin + n_steps * step
        lots = clamp(lots, vmin, vmax)
    """
```

### Risk-Based Sizing Logic

```python
# Formel für Market-Order Sizing:

sl_distance = abs(entry_price - stop_loss)
unit_val = _unit_value_per_price(symbol)
risk_per_lot = sl_distance * unit_val

size_lots = risk_per_trade / risk_per_lot
size_lots = _quantize_volume(symbol, size_lots)

# Alternativ mit LotSizer:
size_lots = lot_sizer.size_risk_based(
    symbol=symbol,
    price=entry_price,
    stop_pips=stop_pips,
    risk_amount_acct=risk_per_trade,
    t=timestamp,
)
```

---

## Slippage and Fees

### SlippageModel Interface

```python
# @ffi_boundary: Input (optional)

class SlippageModel(Protocol):
    def apply(
        self,
        price: float,
        direction: Direction,
        pip_size: float,
    ) -> float:
        """
        Wendet Slippage auf Preis an.
        
        Long Entry: price + slippage (schlechter)
        Short Entry: price - slippage (schlechter)
        """
```

### FeeModel Interface

```python
# @ffi_boundary: Input (optional)

class FeeModel(Protocol):
    def calculate(
        self,
        size: float,
        price: float,
        contract_size: Optional[float] = None,
    ) -> float:
        """Berechnet Fee in Konto-Währung."""

class CommissionModel(Protocol):
    def fee_for_order(
        self,
        symbol: str,
        size: float,
        price: float,
        t: datetime,
        side: Side,  # ENTRY | EXIT
    ) -> float:
        """Berechnet Commission in Konto-Währung."""
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| process_signal() | O(1) | Konstant pro Signal |
| evaluate_exits() | O(p) | p = active_positions |
| check_if_entry_triggered() | O(1) | Pro Position |
| _quantize_volume() | O(1) | Simple math |
| _unit_value_per_price() | O(1) | Cached |

### Memory Profile

```
ExecutionSimulator Instance:
  - active_positions: List[PortfolioPosition]  ~100 bytes/position
  - _pip_cache: Dict[str, tuple]               ~50 bytes/symbol
  - _unit_value_cache: Dict[str, float]        ~20 bytes/symbol
  - symbol_specs: Dict[str, SymbolSpec]        ~200 bytes/symbol
```

---

## State Machine: Position Lifecycle

```
        ┌───────────────┐
        │    Signal     │
        │   Received    │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │  Order Type?  │
        └───────┬───────┘
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│   Market    │   │ Limit/Stop  │
│ (Immediate) │   │ (Pending)   │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│   OPEN      │   │   PENDING   │
│   (active)  │   │  (waiting)  │
└──────┬──────┘   └──────┬──────┘
       │                 │
       │          ┌──────▼──────┐
       │          │  Triggered? │
       │          └──────┬──────┘
       │                 │ yes
       │          ┌──────▼──────┐
       │          │    OPEN     │
       │          │   (active)  │
       │          └──────┬──────┘
       │                 │
       └────────┬────────┘
                │
        ┌───────▼───────┐
        │  SL/TP Hit?   │
        │  Manual Exit? │
        └───────┬───────┘
                │ yes
        ┌───────▼───────┐
        │    CLOSED     │
        │  (finalized)  │
        └───────────────┘
```

---

## FFI Migration Strategy

### Rust Struct Definition

```rust
#[derive(Clone, Debug)]
pub struct PortfolioPosition {
    pub entry_time: i64,  // epoch micros
    pub direction: Direction,
    pub symbol: String,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub size: f64,
    pub status: PositionStatus,
    // ... weitere Felder
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Long,
    Short,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionStatus {
    Open,
    Pending,
    Closed,
}
```

### PyO3 Bindings

```rust
#[pyclass]
pub struct ExecutionSimulator {
    active_positions: Vec<PortfolioPosition>,
    risk_per_trade: f64,
    symbol_specs: HashMap<String, SymbolSpec>,
    // ...
}

#[pymethods]
impl ExecutionSimulator {
    #[new]
    pub fn new(
        portfolio: &Portfolio,
        risk_per_trade: f64,
        // ...
    ) -> PyResult<Self> { ... }
    
    pub fn process_signal(&mut self, signal: PyTradeSignal) -> PyResult<()> { ... }
    
    pub fn evaluate_exits(&mut self, bid: PyCandle, ask: Option<PyCandle>) -> PyResult<()> { ... }
}
```

### Benchmark Targets

| Operation | Python Baseline | Rust Target |
|-----------|-----------------|-------------|
| process_signal() | ~50µs | <5µs |
| evaluate_exits() (10 pos) | ~500µs | <50µs |
| Full backtest (100k bars) | ~30s | <3s |

---

## Critical Invariants

1. **Chronologische Reihenfolge:** entry_time < exit_time (immer)
2. **Pending before Open:** Limit/Stop Orders müssen durch trigger_entry() gehen
3. **Size > 0 für Open:** Nur offene Positionen haben size > 0
4. **Risk Preservation:** size * sl_distance * unit_value ≈ risk_per_trade
5. **Fee Completeness:** Jeder Entry/Exit hat genau eine Fee-Registrierung

---

## Related Modules

- `src/backtest_engine/core/portfolio.py` - Portfolio, PortfolioPosition
- `src/backtest_engine/core/slippage_and_fee.py` - SlippageModel, FeeModel
- `src/backtest_engine/sizing/lot_sizer.py` - LotSizer
- `src/backtest_engine/sizing/commission.py` - CommissionModel
- `src/backtest_engine/sizing/symbol_specs_registry.py` - SymbolSpec, Registry
- `src/backtest_engine/data/candle.py` - Candle
- `src/backtest_engine/data/tick.py` - Tick
