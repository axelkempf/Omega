# Data-Flow Diagramme für Migrations-Kandidaten

**Phase 2 Task:** P2-10  
**Status:** ✅ Dokumentiert (2026-01-05)

---

## Übersicht

Dieses Dokument visualisiert die Datenflüsse durch die identifizierten Migrations-Kandidaten:

1. **indicator_cache** - Indikator-Berechnung und Caching
2. **event_engine** - Event-getriebene Backtesting-Engine
3. **execution_simulator** - Trade-Ausführungssimulation
4. **rating_modules** - Strategie-Bewertung und Scoring

---

## 1. Indicator Cache Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             INDICATOR CACHE DATA FLOW                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌──────────────────────────────────────────────┐
│  Raw OHLCV   │────▶│  DataHandler     │────▶│              IndicatorCache                   │
│  (Parquet)   │     │  (Load/Validate) │     │                                              │
└──────────────┘     └──────────────────┘     │  ┌─────────────┐    ┌─────────────────────┐  │
                                              │  │ Cache Store │◀───│ Indicator Compute   │  │
                                              │  │ (Dict)      │    │ (NumPy/TA-Lib)      │  │
                                              │  └──────┬──────┘    └─────────────────────┘  │
                                              │         │                      ▲             │
                                              │         ▼                      │             │
                                              │  ┌─────────────┐    ┌──────────┴──────────┐  │
                                              │  │ get()       │    │ compute_indicator() │  │
                                              │  └─────────────┘    └─────────────────────┘  │
                                              └──────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │  Strategy Logic  │
                                              │  (Signal Gen)    │
                                              └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA TYPES                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  Input:                                                                                      │
│  ├─ OHLCV DataFrames (pd.DataFrame)                                                         │
│  │   └─ Columns: UTC time, Open, High, Low, Close, Volume                                   │
│  │   └─ Index: DatetimeIndex (UTC)                                                          │
│  │                                                                                          │
│  Cache Storage:                                                                              │
│  ├─ Dict[str, np.ndarray]                                                                   │
│  │   └─ Key: "EMA_20_M15" (indicator_period_timeframe)                                      │
│  │   └─ Value: 1D float64 array (aligned with OHLCV index)                                  │
│  │                                                                                          │
│  Output:                                                                                     │
│  ├─ np.ndarray (float64)                                                                    │
│  │   └─ First N values may be NaN (warmup period)                                           │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            FFI BOUNDARY (Rust Migration)                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌──────────────────────┐         Arrow IPC          ┌──────────────────────┐               │
│  │     Python Side      │◀──────────────────────────▶│      Rust Side       │               │
│  │                      │                            │                      │               │
│  │  IndicatorCache      │         Schema:            │  IndicatorEngine     │               │
│  │  (Orchestrator)      │    ┌────────────────┐      │  (Computation)       │               │
│  │                      │    │ OHLCV_SCHEMA   │      │                      │               │
│  │  - Cache management  │    │ INDICATOR_SCHEMA     │  - Fast EMA/SMA/RSI  │               │
│  │  - Lazy loading      │    └────────────────┘      │  - SIMD optimized    │               │
│  │                      │                            │  - Zero-copy return  │               │
│  └──────────────────────┘                            └──────────────────────┘               │
│                                                                                              │
│  Transfer Pattern:                                                                           │
│  ├─ Python → Rust: OHLCV RecordBatch (Arrow IPC)                                            │
│  ├─ Rust → Python: Indicator RecordBatch (zero-copy)                                        │
│  └─ Caching: Python-side (Rust stateless)                                                   │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Event Engine Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EVENT ENGINE DATA FLOW                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

                     ┌─────────────────────────────────────────────────────────────────┐
                     │                       EventEngine                               │
                     │                                                                 │
┌──────────────┐     │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  OHLCV Data  │────▶│  │ Event Queue │───▶│ Dispatcher  │───▶│ Event Handlers      │ │
│  (Aligned)   │     │  │             │    │             │    │                     │ │
└──────────────┘     │  │ - BAR       │    │ - Route by  │    │ - StrategyHandler   │ │
                     │  │ - SIGNAL    │    │   type      │    │ - PositionHandler   │ │
┌──────────────┐     │  │ - FILL      │    │ - Priority  │    │ - RiskHandler       │ │
│  Signals     │────▶│  │ - ORDER     │    │   queue     │    │ - LoggingHandler    │ │
│  (Strategy)  │     │  │ - POSITION  │    │             │    │                     │ │
└──────────────┘     │  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
                     │                                                   │            │
                     │                     ┌─────────────────────────────┘            │
                     │                     ▼                                          │
                     │  ┌─────────────────────────────────────────────────────────┐   │
                     │  │                    State Manager                         │   │
                     │  │                                                         │   │
                     │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
                     │  │  │ Positions   │  │ Orders      │  │ Equity Curve    │  │   │
                     │  │  │ (Open)      │  │ (Pending)   │  │ (History)       │  │   │
                     │  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
                     │  └─────────────────────────────────────────────────────────┘   │
                     │                           │                                    │
                     └───────────────────────────┼────────────────────────────────────┘
                                                 │
                                                 ▼
                                    ┌───────────────────────┐
                                    │  BacktestResult       │
                                    │  (Trades, Metrics)    │
                                    └───────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    EVENT TYPES                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  Event          Fields                           Priority   Direction                        │
│  ────────────   ──────────────────────────────   ────────   ─────────                        │
│  BAR            timestamp, ohlcv, symbol          10        Data → Strategy                  │
│  SIGNAL         timestamp, direction, entry...    20        Strategy → Execution             │
│  ORDER          order_id, signal, status          30        Execution → Fill                 │
│  FILL           order_id, fill_price, slippage    40        Fill → Position                  │
│  POSITION       position_id, status, pnl          50        Position → State                 │
│  STOP           position_id, reason               60        Risk → Position                  │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            FFI BOUNDARY (Rust Migration)                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌──────────────────────┐         Arrow IPC          ┌──────────────────────┐               │
│  │     Python Side      │◀──────────────────────────▶│      Rust Side       │               │
│  │                      │                            │                      │               │
│  │  EventEngine         │    Schemas:                │  EventProcessor      │               │
│  │  (Orchestrator)      │    ┌────────────────┐      │  (Hot Path)          │               │
│  │                      │    │ TRADE_SIGNAL   │      │                      │               │
│  │  - Handler registry  │    │ POSITION       │      │  - Fast event loop   │               │
│  │  - State persistence │    │ EQUITY_CURVE   │      │  - Parallel handlers │               │
│  │  - Results export    │    └────────────────┘      │  - Lock-free queue   │               │
│  └──────────────────────┘                            └──────────────────────┘               │
│                                                                                              │
│  Transfer Pattern:                                                                           │
│  ├─ Batch Mode: Full backtest data → Rust → Results                                         │
│  ├─ Streaming Mode: Event batches (configurable size)                                       │
│  └─ State Sync: Positions/Equity nur bei Checkpoints                                        │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Execution Simulator Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION SIMULATOR DATA FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────┐      ┌─────────────────────────────────────────────────────────────────────┐
│ Trade Signal  │─────▶│                      ExecutionSimulator                             │
│               │      │                                                                     │
│ - entry       │      │  ┌─────────────┐    ┌─────────────────┐    ┌───────────────────┐   │
│ - sl          │      │  │ Cost Model  │───▶│ Slippage Model  │───▶│ Fill Generator    │   │
│ - tp          │      │  │             │    │                 │    │                   │   │
│ - size        │      │  │ - spread    │    │ - market impact │    │ - fill_price      │   │
│ - direction   │      │  │ - commission│    │ - latency       │    │ - fill_time       │   │
└───────────────┘      │  │ - swap      │    │ - partial fills │    │ - actual_slippage │   │
                       │  └─────────────┘    └─────────────────┘    └─────────┬─────────┘   │
┌───────────────┐      │                                                      │             │
│ Market Data   │─────▶│  ┌──────────────────────────────────────────────────┘             │
│               │      │  │                                                                 │
│ - bid         │      │  ▼                                                                 │
│ - ask         │      │  ┌─────────────────────────────────────────────────────────────┐   │
│ - spread      │      │  │                    Fill Result                               │   │
│ - liquidity   │      │  │                                                             │   │
└───────────────┘      │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐│   │
                       │  │  │ Filled Order  │  │ Position      │  │ Execution Stats   ││   │
                       │  │  │               │  │               │  │                   ││   │
                       │  │  │ - order_id    │  │ - position_id │  │ - total_slippage  ││   │
                       │  │  │ - fill_price  │  │ - entry_price │  │ - total_spread    ││   │
                       │  │  │ - fill_time   │  │ - size        │  │ - total_commission││   │
                       │  │  │ - slippage    │  │ - direction   │  │                   ││   │
                       │  │  └───────────────┘  └───────────────┘  └───────────────────┘│   │
                       │  └─────────────────────────────────────────────────────────────┘   │
                       └─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                EXECUTION COST MODEL                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  Total Cost = Spread + Slippage + Commission + Swap                                          │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                           Cost Components                                            │    │
│  │                                                                                      │    │
│  │  Spread (Bid-Ask):                                                                   │    │
│  │  ├─ Static: Fixed pip value from symbol_specs.yaml                                  │    │
│  │  └─ Dynamic: Time-of-day dependent (news, session)                                  │    │
│  │                                                                                      │    │
│  │  Slippage:                                                                           │    │
│  │  ├─ Market Order: Proportional to size / liquidity                                  │    │
│  │  ├─ Limit Order: Zero (price guaranteed)                                            │    │
│  │  └─ Stop Order: Variable (gap risk)                                                 │    │
│  │                                                                                      │    │
│  │  Commission:                                                                         │    │
│  │  ├─ Per-Trade: Fixed amount per round-trip                                          │    │
│  │  └─ Per-Lot: Proportional to volume                                                 │    │
│  │                                                                                      │    │
│  │  Swap:                                                                               │    │
│  │  ├─ Long Swap: Rate for holding long position overnight                             │    │
│  │  └─ Short Swap: Rate for holding short position overnight                           │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            FFI BOUNDARY (Rust Migration)                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌──────────────────────┐         Arrow IPC          ┌──────────────────────┐               │
│  │     Python Side      │◀──────────────────────────▶│      Rust Side       │               │
│  │                      │                            │                      │               │
│  │  ExecutionSimulator  │    Schemas:                │  ExecutionCore       │               │
│  │  (Orchestrator)      │    ┌────────────────┐      │  (Hot Path)          │               │
│  │                      │    │ TRADE_SIGNAL   │      │                      │               │
│  │  - Config loading    │    │ POSITION       │      │  - Batch execution   │               │
│  │  - Results export    │    │ FILL_RESULT    │      │  - Parallel cost calc│               │
│  │  - Cost config       │    └────────────────┘      │  - SIMD slippage     │               │
│  └──────────────────────┘                            └──────────────────────┘               │
│                                                                                              │
│  Transfer Pattern:                                                                           │
│  ├─ Batch: All signals → Rust → All fills (optimal)                                         │
│  ├─ Streaming: Signal batches with market data                                              │
│  └─ Config: YAML → Rust struct (one-time)                                                   │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Rating Modules Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              RATING MODULES DATA FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────┐
│   BacktestResult      │
│                       │
│   - trades[]          │
│   - equity_curve      │
│   - parameters        │
└──────────┬────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Rating Pipeline                                              │
│                                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Stage 1: Base Metrics                                      │ │
│  │                                                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │ Total PnL   │  │ Win Rate    │  │ Sharpe      │  │ Max DD      │  │ Profit      │  │ │
│  │  │             │  │             │  │ Ratio       │  │             │  │ Factor      │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                                   │
│                                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Stage 2: Robustness Tests                                  │ │
│  │                                                                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  │ Trade Dropout   │  │ Cost Shock      │  │ Timing Jitter   │  │ TP/SL Stress    │   │ │
│  │  │ Score           │  │ Score           │  │ Score           │  │ Score           │   │ │
│  │  │                 │  │                 │  │                 │  │                 │   │ │
│  │  │ Monte Carlo:    │  │ Cost +50%:      │  │ Entry ±3 bars:  │  │ TP/SL ±10%:     │   │ │
│  │  │ Remove 10-30%   │  │ Still profit?   │  │ Still profit?   │  │ Still profit?   │   │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                                   │
│                                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Stage 3: Statistical Tests                                 │ │
│  │                                                                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                         │ │
│  │  │ Monte Carlo     │  │ Bayesian        │  │ Stability       │                         │ │
│  │  │ Confidence      │  │ Alpha/Beta      │  │ Index           │                         │ │
│  │  │                 │  │                 │  │                 │                         │ │
│  │  │ P(Profit > 0)   │  │ Prior: Uniform  │  │ Variance of     │                         │ │
│  │  │ from N samples  │  │ Posterior prob  │  │ rolling metrics │                         │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘                         │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                                   │
│                                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Stage 4: Final Score                                       │ │
│  │                                                                                         │ │
│  │                        ┌─────────────────────────────────────┐                         │ │
│  │                        │         Weighted Aggregation         │                         │ │
│  │                        │                                     │                         │ │
│  │                        │  final_score = Σ (weight_i × score_i)                        │ │
│  │                        │                                     │                         │ │
│  │                        │  Output: 0.0 - 10.0 Rating           │                         │ │
│  │                        └─────────────────────────────────────┘                         │ │
│  └────────────────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                             ┌───────────────────────┐
                             │   RatingResult        │
                             │                       │
                             │   - final_score       │
                             │   - component_scores  │
                             │   - confidence        │
                             │   - recommendation    │
                             └───────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            FFI BOUNDARY (Rust Migration)                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌──────────────────────┐         Arrow IPC          ┌──────────────────────┐               │
│  │     Python Side      │◀──────────────────────────▶│      Rust Side       │               │
│  │                      │                            │                      │               │
│  │  RatingPipeline      │    Schemas:                │  RatingCore          │               │
│  │  (Orchestrator)      │    ┌────────────────┐      │  (Computation)       │               │
│  │                      │    │ TRADE_SCHEMA   │      │                      │               │
│  │  - Pipeline config   │    │ EQUITY_CURVE   │      │  - Parallel Monte    │               │
│  │  - Result export     │    │ RATING_SCORE   │      │    Carlo simulation  │               │
│  │  - Weight tuning     │    └────────────────┘      │  - SIMD statistics   │               │
│  └──────────────────────┘                            └──────────────────────┘               │
│                                                                                              │
│  Transfer Pattern:                                                                           │
│  ├─ Batch: All trades + equity → Rust → All scores                                          │
│  ├─ Monte Carlo: Configurable sample count (1000-10000)                                     │
│  └─ Result: Component scores + aggregated score                                             │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. End-to-End Backtest Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE BACKTEST DATA FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────────────────────────────────────────────────────────────────┐
│             │    │                                                                         │
│   Config    │───▶│                         BacktestRunner                                  │
│   (JSON)    │    │                                                                         │
│             │    │  1. Load Data                                                           │
└─────────────┘    │     ┌─────────────┐                                                     │
                   │     │ DataHandler │◀── Parquet/CSV                                      │
┌─────────────┐    │     └──────┬──────┘                                                     │
│             │    │            │                                                             │
│   Market    │───▶│            ▼                                                             │
│   Data      │    │  2. Initialize Indicators                                               │
│  (Parquet)  │    │     ┌────────────────┐                                                  │
│             │    │     │ IndicatorCache │◀── [FFI: Rust Compute]                           │
└─────────────┘    │     └───────┬────────┘                                                  │
                   │             │                                                            │
                   │             ▼                                                            │
                   │  3. Run Event Loop                                                       │
                   │     ┌─────────────────────────────────────────────────────────────────┐ │
                   │     │                    EventEngine                                  │ │
                   │     │                                                                 │ │
                   │     │  for each bar:                                                  │ │
                   │     │    ├─ BAR event → Strategy                                      │ │
                   │     │    ├─ SIGNAL event ← Strategy ◀── [FFI: Rust Signal Gen?]       │ │
                   │     │    ├─ ORDER event → ExecutionSimulator                          │ │
                   │     │    │                  ◀── [FFI: Rust Execution]                 │ │
                   │     │    ├─ FILL event ← ExecutionSimulator                           │ │
                   │     │    ├─ POSITION event → PositionManager                          │ │
                   │     │    └─ Update equity curve                                       │ │
                   │     │                                                                 │ │
                   │     └─────────────────────────────────────────────────────────────────┘ │
                   │                        │                                                │
                   │                        ▼                                                │
                   │  4. Collect Results                                                     │
                   │     ┌─────────────────────┐                                             │
                   │     │   BacktestResult    │                                             │
                   │     │   - trades[]        │                                             │
                   │     │   - equity_curve    │                                             │
                   │     │   - parameters      │                                             │
                   │     └──────────┬──────────┘                                             │
                   │                │                                                        │
                   │                ▼                                                        │
                   │  5. Rate Strategy                                                       │
                   │     ┌─────────────────────┐                                             │
                   │     │   RatingPipeline    │◀── [FFI: Rust Rating]                       │
                   │     └──────────┬──────────┘                                             │
                   │                │                                                        │
                   │                ▼                                                        │
                   │  6. Export Results                                                      │
                   │     ┌─────────────────────┐                                             │
                   │     │   var/results/      │                                             │
                   │     │   - trades.csv      │                                             │
                   │     │   - metrics.json    │                                             │
                   │     │   - rating.json     │                                             │
                   │     └─────────────────────┘                                             │
                   │                                                                         │
                   └─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FFI HOT PATHS                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  Priority   Module              Reason                       Transfer Pattern               │
│  ────────   ──────────────────  ─────────────────────────    ──────────────────────────     │
│     1       IndicatorCache      Called every bar (N×M)       OHLCV batch → Indicator batch  │
│     2       RatingPipeline      Monte Carlo (1000+ samples)  Trades batch → Scores batch    │
│     3       ExecutionSimulator  Called per signal            Signal batch → Fill batch      │
│     4       EventEngine         Orchestration overhead       Event batch (if streaming)     │
│                                                                                              │
│  Legend:                                                                                     │
│  ├─ N: Number of bars (100k-1M typical)                                                     │
│  ├─ M: Number of indicators per bar (10-50 typical)                                         │
│  └─ Hot Path: Called in inner loop, performance critical                                    │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Zusammenfassung der Transfer-Patterns

| Modul | Input Schema | Output Schema | Batch Size | Frequenz |
|-------|--------------|---------------|------------|----------|
| IndicatorCache | OHLCV_SCHEMA | INDICATOR_SCHEMA | Full history | Per indicator |
| EventEngine | Multiple events | Multiple events | Configurable | Per bar |
| ExecutionSimulator | TRADE_SIGNAL_SCHEMA | POSITION_SCHEMA | Per signal batch | Per signal |
| RatingPipeline | TRADE_SCHEMA, EQUITY_CURVE_SCHEMA | RATING_SCORE_SCHEMA | Full backtest | Once per backtest |

---

## Referenzen

- [ADR-0001: Migrationsstrategie](../adr/ADR-0001-migration-strategy.md)
- [ADR-0002: Serialisierungsformat](../adr/ADR-0002-serialization-format.md)
- [Arrow Schema Definitions](../../src/shared/arrow_schemas.py)
- [Indicator Cache FFI Spec](./indicator_cache.md)
- [Event Engine FFI Spec](./event_engine.md)
- [Execution Simulator FFI Spec](./execution_simulator.md)
- [Rating Modules FFI Spec](./rating_modules.md)
