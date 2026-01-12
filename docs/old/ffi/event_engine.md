# FFI Interface Specification: EventEngine

**Modul:** `src/backtest_engine/core/event_engine.py`  
**Migrations-Ziel:** Rust (via PyO3/maturin)  
**Phase 2 Task:** P2-02  
**Status:** ✅ Spezifiziert (2026-01-05)

---

## Executive Summary

`EventEngine` und `CrossSymbolEventEngine` sind die **Hauptschleifen** des Backtest-Systems.
Sie koordinieren:
- Bar-by-Bar Iteration über Candle-Daten
- Strategy Evaluation (Signalgenerierung)
- Order Execution
- Position Management
- Portfolio Updates

Migrations-Kandidat aufgrund:
- Hoher Aufruffrequenz (n Iterationen für n Bars)
- Klarer Event-Loop-Semantik
- Dependency Injection Pattern (Strategy, Executor, Portfolio)

---

## Data Structures

### Event Loop Context

```python
# @ffi_boundary: Input

# Single-Symbol Engine
bid_candles: List[Candle]           # Chronologisch sortiert
ask_candles: List[Candle]           # Parallel zu bid_candles (gleiche Länge)
multi_candle_data: MultiCandleData  # Aligned Multi-TF Daten (siehe indicator_cache.md)
symbol: str                         # z.B. "EURUSD"
original_start_dt: datetime         # Warmup-Ende, erstes gültiges Signal-Datum

# Cross-Symbol Engine  
candle_lookups: CandleLookups       # Dict[symbol][side][timestamp] -> Candle
common_timestamps: CommonTimestamps # Synchronisierte Zeitachse
primary_tf: str                     # Haupt-Timeframe, z.B. "M1"
```

### TypedDict Definitionen (aus types.py)

```python
# @ffi_boundary: Input

CandleLookups: TypeAlias = Mapping[
    Symbol,                          # str, z.B. "EURUSD"
    Mapping[
        PriceType,                   # "bid" | "ask"
        Mapping[
            TimestampKey,            # datetime | str | int | float
            Candle
        ]
    ]
]

CommonTimestamps: TypeAlias = Sequence[TimestampKey]
```

### Injected Dependencies

```python
# @ffi_boundary: Input (via Constructor)

strategy: StrategyWrapper           # evaluate(idx, slice_map) -> List[TradeSignal] | None
executor: ExecutionSimulator        # process_signal(), evaluate_exits()
portfolio: Portfolio                # update(), register_entry/exit()
on_progress: Optional[Callable[[int, int], None]]  # Progress callback (current, total)
```

---

## EventEngine (Single-Symbol)

### Constructor Signature

```python
def __init__(
    self,
    bid_candles: List[Candle],
    ask_candles: List[Candle],
    strategy: StrategyWrapper,
    executor: ExecutionSimulator,
    portfolio: Portfolio,
    multi_candle_data: Dict[str, Dict[str, List[Candle]]],
    symbol: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    original_start_dt: Optional[datetime] = None,
) -> None:
    """
    Initialisiert Single-Symbol Event Engine.
    
    @ffi_boundary: Input
    
    Args:
        bid_candles: Bid-Kerzen für Primary TF (chronologisch)
        ask_candles: Ask-Kerzen (parallel zu bid_candles)
        strategy: Strategy-Wrapper mit evaluate() Methode
        executor: Execution Simulator für Order-Processing
        portfolio: Portfolio für Tracking
        multi_candle_data: Aligned Multi-TF Candle-Daten
        symbol: Trading-Symbol
        on_progress: Optional Progress-Callback
        original_start_dt: Warmup-Ende Timestamp
        
    Constraints:
        - len(bid_candles) == len(ask_candles)
        - original_start_dt muss gesetzt sein (ValueError sonst)
        - bid_candles müssen chronologisch sortiert sein
    """
```

### Main Loop: run()

```python
def run(self) -> None:
    """
    Hauptschleife der Event Engine.
    
    @ffi_boundary: Internal (koordiniert externe Komponenten)
    
    Loop Semantik (pro Bar i):
        1. ENTRY: strategy.evaluate(i, slice_map) -> signals
           - Für jedes Signal: executor.process_signal(signal)
        
        2. EXITS: executor.evaluate_exits(bid_candle, ask_candle)
           - Prüft SL/TP für offene Positionen
        
        3. POSITION MANAGEMENT: pm.manage_positions(...)
           - Trailing Stop, Break-Even, etc.
        
        4. PORTFOLIO UPDATE: portfolio.update(timestamp)
           - Equity Tracking, Drawdown-Berechnung
        
        5. PROGRESS: on_progress(current, total)
    
    Raises:
        ValueError: wenn original_start_dt nicht gesetzt
        ValueError: wenn kein Startindex gefunden (Timestamp-Mismatch)
    
    Side Effects:
        - Modifiziert executor.active_positions
        - Modifiziert portfolio (via register_entry/exit/fee)
        - Modifiziert strategy state (falls stateful)
    """
```

### Event Loop State Machine

```
┌──────────────┐
│  START       │
│  (i=0)       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  LOOP: for i in range(start_index, total)                   │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ 1. ENTRY    │───▶│ 2. EXITS    │───▶│ 3. POS MGMT │      │
│  │ evaluate()  │    │ eval_exits()│    │ manage()    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                                      │             │
│         ▼                                      ▼             │
│  ┌─────────────┐                      ┌─────────────┐       │
│  │ process_    │                      │ 4. PORTFOLIO│       │
│  │ signal()    │                      │ update()    │       │
│  └─────────────┘                      └─────────────┘       │
│                                               │              │
│                                               ▼              │
│                                       ┌─────────────┐       │
│                                       │ 5. PROGRESS │       │
│                                       │ callback    │       │
│                                       └─────────────┘       │
│                                               │              │
│  ◄────────────────────────────────────────────┘ (next i)    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  END         │
└──────────────┘
```

---

## CrossSymbolEventEngine (Multi-Symbol)

### Constructor Signature

```python
def __init__(
    self,
    candle_lookups: Dict[str, Dict[str, Dict[Any, Candle]]],
    common_timestamps: List[Any],
    strategy: Any,
    executor: ExecutionSimulator,
    portfolio: Portfolio,
    primary_tf: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    original_start_dt: Optional[Any] = None,
) -> None:
    """
    Initialisiert Multi-Symbol Event Engine.
    
    @ffi_boundary: Input
    
    Args:
        candle_lookups: Dict[symbol][side][timestamp] -> Candle
        common_timestamps: Synchronisierte Timestamps (alle Symbols)
        strategy: Strategy (muss Multi-Symbol evaluate() haben)
        executor: Execution Simulator
        portfolio: Portfolio
        primary_tf: Primary Timeframe String
        on_progress: Optional Progress-Callback
        original_start_dt: Warmup-Ende
        
    Constraints:
        - Alle Symbols in candle_lookups müssen für common_timestamps abgedeckt sein
    """
```

### Multi-Symbol Loop

```python
def run(self) -> None:
    """
    Hauptschleife für Multi-Symbol Backtests.
    
    @ffi_boundary: Internal
    
    Loop Semantik (pro Timestamp ts):
        1. multi_slice.set_timestamp(ts)
        2. strategy.evaluate(idx, multi_slice) -> signals
        3. Für jedes Symbol: executor.evaluate_exits(bid, ask)
        4. Position Management pro Symbol
        5. portfolio.update(ts)
        6. on_progress(current, total)
        
    Unterschied zu Single-Symbol:
        - MultiSymbolSlice statt SymbolDataSlice
        - Exit-Prüfung pro Symbol in Loop
        - Position Management pro Symbol
    """
```

---

## Callback Signatures

### Strategy Evaluate

```python
# @ffi_boundary: Input/Output

def evaluate(
    self,
    index: int,                                    # Bar-Index
    slice_map: Dict[str, SymbolDataSlice] | MultiSymbolSlice,
) -> List[TradeSignal] | TradeSignal | None:
    """
    Strategy Evaluation Callback.
    
    Input:
        index: Aktueller Bar-Index (0-basiert)
        slice_map: Zugriff auf Candle-Daten und Indikatoren
        
    Output:
        - None: Kein Signal
        - TradeSignal: Einzelnes Signal
        - List[TradeSignal]: Mehrere Signale (Multi-Entry)
    """
```

### Progress Callback

```python
# @ffi_boundary: Output

on_progress: Callable[[int, int], None]
# Args: (current_step, total_steps)
# current_step: 1-basiert, startet bei 1
# total_steps: Gesamtanzahl Bars nach Warmup
```

---

## TradeSignal Type (aus types.py)

```python
# @ffi_boundary: Input (von Strategy) / Output (an Executor)

class TradeSignalDict(TypedDict, total=False):
    """Signal-Shape wie von Strategien geliefert."""
    
    direction: RawSignalDirection  # "buy" | "sell" | "long" | "short"
    entry: float                   # Entry-Preis (optional bei Market)
    sl: float                      # Stop-Loss
    tp: float                      # Take-Profit
    
    symbol: Symbol                 # Trading-Symbol
    type: OrderType                # "market" | "limit" | "stop"
    
    reason: str                    # Signal-Begründung
    tags: Sequence[str]            # Klassifizierungs-Tags
    scenario: str                  # Scenario-Name
    
    meta: Mapping[str, Any]        # Zusätzliche Metadaten
    metadata: Mapping[str, Any]    # Backcompat Alias für meta

# Runtime-Objekt (nach Normalisierung durch StrategyWrapper)
class TradeSignal:
    direction: Direction           # "long" | "short" (normalisiert)
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    timestamp: datetime
    type: str                      # "market" | "limit" | "stop"
    reason: str | None
    tags: Sequence[str]
    scenario: str | None
    meta: Mapping[str, Any]
```

### Arrow Schema für TradeSignal

```
TradeSignal {
  direction: utf8 (enum: "long" | "short")
  entry_price: float64
  stop_loss: float64
  take_profit: float64
  symbol: utf8
  timestamp: timestamp[us, tz=UTC]
  type: utf8 (enum: "market" | "limit" | "stop")
  reason: utf8 (nullable)
  scenario: utf8 (nullable)
  meta: utf8 (JSON-serialized)
}
```

---

## SymbolDataSlice Protocol

```python
# @ffi_boundary: Input (an Strategy)

@runtime_checkable
class SymbolDataSliceProtocol(Protocol):
    index: int
    indicators: IndicatorCacheProtocol | None
    
    def set_index(self, index: int) -> None:
        """Setzt aktuellen Bar-Index."""
    
    def latest(
        self,
        timeframe: str,
        price_type: str = "bid",
    ) -> CandleLike | None:
        """Gibt aktuelle Candle für TF/Side zurück."""
    
    def history(
        self,
        timeframe: str,
        price_type: str = "bid",
        length: int = 20,
    ) -> list[CandleLike]:
        """Gibt letzte n Candles zurück (älteste zuerst)."""
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Loop Iteration | O(n) | n = Anzahl Bars |
| strategy.evaluate() | O(1)* | *Strategy-abhängig |
| executor.process_signal() | O(1) | Per Signal |
| executor.evaluate_exits() | O(p) | p = offene Positionen |
| portfolio.update() | O(1) | Amortisiert |

### Bottleneck Analysis

```
┌──────────────────────────────────────────────────────────────┐
│  Typische Backtest-Laufzeit (100k Bars, 1 Symbol)           │
│                                                              │
│  Event Loop Overhead:    ~5%   (Python Loop, Index Update)   │
│  Strategy Evaluate:     ~40%   (Indicator Access)            │
│  Execution Simulator:   ~30%   (Exit Checks, Sizing)         │
│  Portfolio Update:      ~15%   (Equity Tracking)             │
│  Progress Callback:     ~10%   (wenn enabled)                │
│                                                              │
│  → Strategy + Execution sind primäre Optimierungsziele       │
└──────────────────────────────────────────────────────────────┘
```

---

## FFI Migration Strategy

### Option A: Full Rust Event Loop

```rust
// Rust übernimmt komplette Loop-Kontrolle
pub fn run_backtest(
    candles: ArrowBatch,
    strategy: &dyn Strategy,    // Trait object
    executor: &mut Executor,
    portfolio: &mut Portfolio,
) -> BacktestResult { ... }
```

**Vorteile:** Maximale Performance
**Nachteile:** Strategy muss auch in Rust sein

### Option B: Hybrid mit Python Callbacks

```rust
// Rust Loop, aber Python Strategy-Callback
#[pyfunction]
pub fn run_backtest_hybrid(
    py: Python,
    candles: PyArrowArray,
    strategy_callback: PyObject,  // Python callable
    executor: PyRef<Executor>,
) -> PyResult<BacktestResult> {
    for i in 0..candles.len() {
        // Call Python strategy
        let signals = strategy_callback.call1(py, (i, slice))?;
        // Process in Rust
        executor.process_signals(signals)?;
    }
}
```

**Vorteile:** Bestandene Python-Strategien funktionieren weiter
**Nachteile:** GIL-Overhead bei jedem Callback

### Option C: Batch Processing

```rust
// Rust berechnet Indikatoren, Python entscheidet
pub fn precompute_indicators(candles: ArrowBatch) -> IndicatorBatch { ... }

// Python Loop mit Rust-Indikatoren
// for i in range(n):
//     indicators = rust_indicators[i]
//     signal = strategy.evaluate(i, indicators)
```

**Empfehlung:** Option C für initiale Migration, dann schrittweise zu Option A

---

## Critical Invariants

1. **Bar-Reihenfolge:** Chronologisch aufsteigend
2. **Warmup-Respektierung:** Keine Signale vor `original_start_dt`
3. **Entry vor Exit:** Signale werden vor Exit-Checks verarbeitet
4. **Position Manager nach Exits:** PM läuft nach evaluate_exits()
5. **Portfolio immer aktuell:** update() am Ende jeder Iteration

---

## Related Modules

- `src/backtest_engine/core/execution_simulator.py` - ExecutionSimulator
- `src/backtest_engine/core/portfolio.py` - Portfolio
- `src/backtest_engine/core/symbol_data_slicer.py` - SymbolDataSlice
- `src/backtest_engine/core/multi_symbol_slice.py` - MultiSymbolSlice
- `src/backtest_engine/strategy/strategy_wrapper.py` - StrategyWrapper
- `src/shared/protocols.py` - Protocol-Definitionen
