# Omega V2 – Data Flow Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Vollständige Spezifikation des Datenflusses im Omega V2 Backtesting-System

---

## 1. Übersicht: Data Flow Philosophie

### 1.1 Kernprinzipien

| Prinzip | Beschreibung |
|---------|--------------|
| **Single FFI Boundary** | Nur EIN Python↔Rust Aufruf pro Backtest |
| **Zero-Copy** | Daten einmal nach Rust laden, dort behalten |
| **Unidirektional** | Daten fließen immer in eine Richtung (keine Rückflüsse) |
| **Immutabilität** | Daten werden transformiert, nicht mutiert |
| **Lazy Loading** | Daten nur laden, wenn benötigt |

### 1.2 Data Flow auf höchster Ebene

```
┌────────────────┐      JSON Config       ┌────────────────┐      JSON Result      ┌────────────────┐
│                │ ──────────────────────▶│                │ ──────────────────────▶│                │
│  Python Layer  │                        │   Rust Engine  │                        │  Python Layer  │
│  (Orchestrator)│                        │    (Kern)      │                        │   (Reporter)   │
│                │◀ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│                │                        │                │
└────────────────┘   (Keine Rückflüsse    └────────────────┘                        └────────────────┘
                      während Backtest)
```

---

## 2. Detaillierter Data Flow

### 2.1 Phase 1: Initialisierung (Python → Rust)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PYTHON: INITIALISIERUNG                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Config File    │
│  (JSON)         │
│                 │
│ • strategy_name │
│ • symbol        │
│ • start_date    │
│ • end_date      │
│ • timeframe     │
│ • parameters    │
│ • data_paths    │
└────────┬────────┘
         │
         │ Laden & Validieren
         ▼
┌─────────────────┐
│  BacktestConfig │
│  (Python Dict)  │
│                 │
│ Anreichern mit: │
│ • Absolute Paths│
│ • Defaults      │
│ • Validierung   │
└────────┬────────┘
         │
         │ json.dumps()
         ▼
┌─────────────────┐
│  Config JSON    │
│  (String)       │
│                 │
│ Serialisiert    │
│ für FFI-Call    │
└────────┬────────┘
         │
         │ ══════════════════════════════════════
         │            FFI BOUNDARY
         │ ══════════════════════════════════════
         │
         │ run_backtest(config_json: &str)
         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: INITIALISIERUNG                                     │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Config JSON    │
│  (String)       │
└────────┬────────┘
         │
         │ serde_json::from_str()
         ▼
┌─────────────────┐
│  BacktestConfig │
│  (Rust Struct)  │
│                 │
│ Typisiert:      │
│ • symbol: String│
│ • tf: Timeframe │
│ • dates: Range  │
│ • params: Params│
└─────────────────┘
```

### 2.2 Phase 2: Daten laden (Rust-intern)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: DATA LOADING                                        │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  BID Parquet    │     │  ASK Parquet    │
│  (Disk)         │     │  (Disk)         │
│                 │     │                 │
│ EURUSD_M5_BID   │     │ EURUSD_M5_ASK   │
│ .parquet        │     │ .parquet        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │  arrow-rs/polars      │
         │  read_parquet()       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           RAW CANDLE DATA               │
│                                         │
│  Vec<RawCandle> {                       │
│      timestamp: i64,                    │
│      open: f64,                         │
│      high: f64,                         │
│      low: f64,                          │
│      close: f64,                        │
│      volume: f64,                       │
│  }                                      │
└────────────────────┬────────────────────┘
                     │
                     │ Filterung:
                     │ • Date Range
                     │ • Market Hours
                     │ • Lücken-Handling
                     ▼
┌─────────────────────────────────────────┐
│           CANDLE STORE                   │
│                                         │
│  CandleStore {                          │
│      bid: Vec<Candle>,                  │
│      ask: Vec<Candle>,                  │
│      timestamps: Vec<i64>,              │
│      len: usize,                        │
│  }                                      │
│                                         │
│  Candle {                               │
│      o: f64, h: f64, l: f64, c: f64,   │
│  }                                      │
└────────────────────┬────────────────────┘
                     │
                     │ Falls HTF benötigt:
                     │ Aggregation M5 → H1 → D1
                     ▼
┌─────────────────────────────────────────┐
│        MULTI-TIMEFRAME STORE            │
│                                         │
│  MultiTfStore {                         │
│      m5: CandleStore,                   │
│      h1: CandleStore,     // optional   │
│      d1: CandleStore,     // optional   │
│      htf_index_map: HashMap<i64, usize>,│
│  }                                      │
└─────────────────────────────────────────┘
```

### 2.3 Phase 3: Indikator-Berechnung (Rust-intern)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: INDICATOR COMPUTATION                               │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│        MULTI-TIMEFRAME STORE            │
│                                         │
│  (Read-only Referenz)                   │
└────────────────────┬────────────────────┘
                     │
                     │ Strategy.required_indicators()
                     │ liefert Liste der benötigten
                     │ Indikatoren
                     ▼
┌─────────────────────────────────────────┐
│        INDICATOR REQUIREMENTS           │
│                                         │
│  Vec<IndicatorSpec> {                   │
│      IndicatorSpec {                    │
│          name: "EMA",                   │
│          timeframe: M5,                 │
│          params: {length: 20},          │
│      },                                 │
│      IndicatorSpec {                    │
│          name: "ATR",                   │
│          timeframe: M5,                 │
│          params: {length: 14},          │
│      },                                 │
│      IndicatorSpec {                    │
│          name: "EMA",                   │
│          timeframe: D1,                 │
│          params: {length: 50},          │
│      },                                 │
│      ...                                │
│  }                                      │
└────────────────────┬────────────────────┘
                     │
                     │ Für jeden Indikator:
                     │ compute_indicator()
                     │ (Vektorisiert, SIMD)
                     ▼
┌─────────────────────────────────────────┐
│         INDICATOR CACHE                  │
│                                         │
│  IndicatorCache {                       │
│      cache: HashMap<                    │
│          (Name, Timeframe, Params),     │
│          Vec<f64>                       │
│      >                                  │
│  }                                      │
│                                         │
│  Beispiel-Einträge:                     │
│  ─────────────────────────────────────  │
│  ("EMA", M5, {20}) → [1.0821, 1.0823,..]│
│  ("ATR", M5, {14}) → [0.0012, 0.0013,..]│
│  ("BB_UPPER", M5, {20,2}) → [...]       │
│  ("BB_LOWER", M5, {20,2}) → [...]       │
│  ("Z_SCORE", M5, {50}) → [...]          │
│  ("EMA", D1, {50}) → [...]              │
│  ("EMA", H1, {100}) → [...]             │
│                                         │
│  Alle Arrays haben Länge == candles.len │
│  (NaN für Warmup-Periode)               │
└─────────────────────────────────────────┘
```

### 2.4 Phase 4: Event Loop (Rust-intern)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: EVENT LOOP                                          │
└──────────────────────────────────────────────────────────────────────────────────┘

                     ┌─────────────────────────────────────────┐
                     │           EINGABE-DATEN                 │
                     │                                         │
                     │  • CandleStore (bid, ask)               │
                     │  • IndicatorCache                       │
                     │  • Strategy Instance                    │
                     │  • ExecutionEngine                      │
                     │  • Portfolio (initial)                  │
                     └────────────────────┬────────────────────┘
                                          │
                                          │ Warmup überspringen
                                          │ idx = warmup_periods
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║                         FOR idx IN warmup..len                              ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1: BarContext erstellen                                              │   │
│  │                                                                            │   │
│  │  BarContext {                                                              │   │
│  │      idx: usize,                        // Aktueller Index                 │   │
│  │      timestamp: i64,                    // candles.timestamps[idx]         │   │
│  │      bid: &Candle,                      // candles.bid[idx]                │   │
│  │      ask: &Candle,                      // candles.ask[idx]                │   │
│  │      indicators: &IndicatorCache,       // Read-only Zugriff               │   │
│  │      htf_idx: Option<usize>,            // Index in HTF-Store              │   │
│  │  }                                                                         │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                        │
│                                          ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 2: Strategy.on_bar(&ctx) aufrufen                                    │   │
│  │                                                                            │   │
│  │  Strategie liest:                                                          │   │
│  │  • ctx.get_indicator("EMA", M5, {20})[idx]                                 │   │
│  │  • ctx.get_indicator("ATR", M5, {14})[idx]                                 │   │
│  │  • ctx.bid.close, ctx.ask.close                                            │   │
│  │  • ctx.get_htf_indicator("EMA", D1, {50})                                  │   │
│  │                                                                            │   │
│  │  Strategie gibt zurück:                                                    │   │
│  │  Option<Signal>                                                            │   │
│  │                                                                            │   │
│  │  Signal {                                                                  │   │
│  │      direction: Long | Short,                                              │   │
│  │      entry_price: f64,                                                     │   │
│  │      stop_loss: f64,                                                       │   │
│  │      take_profit: f64,                                                     │   │
│  │      size: f64,                                                            │   │
│  │      scenario: String,                                                     │   │
│  │  }                                                                         │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                        │
│                                          ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 3: ExecutionEngine.process(signal)                                   │   │
│  │                                                                            │   │
│  │  Falls Signal vorhanden:                                                   │   │
│  │                                                                            │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  3a. Slippage anwenden                                               │  │   │
│  │  │      fill_price = entry_price + slippage_model.calc(...)             │  │   │
│  │  └─────────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                             │   │
│  │                              ▼                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  3b. Fees berechnen                                                  │  │   │
│  │  │      fee = fee_model.calc(size, price, ...)                          │  │   │
│  │  └─────────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                             │   │
│  │                              ▼                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  3c. Order erstellen                                                 │  │   │
│  │  │      Order { fill_price, size, fee, sl, tp, ... }                    │  │   │
│  │  └─────────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                             │   │
│  │                              ▼                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  3d. Portfolio.open_position(order)                                  │  │   │
│  │  │      • Position zur Liste hinzufügen                                 │  │   │
│  │  │      • Cash reduzieren um Margin/Fee                                 │  │   │
│  │  └─────────────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                        │
│                                          ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 4: Portfolio.check_stops(bid, ask)                                   │   │
│  │                                                                            │   │
│  │  Für jede offene Position:                                                 │   │
│  │                                                                            │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Stop Loss getroffen?                                                │  │   │
│  │  │  Take Profit getroffen?                                              │  │   │
│  │  │  Max Holding Time erreicht?                                          │  │   │
│  │  │  Trailing Stop triggered?                                            │  │   │
│  │  └─────────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                             │   │
│  │                              ▼                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Falls Close-Bedingung erfüllt:                                      │  │   │
│  │  │  • Position schließen                                                │  │   │
│  │  │  • Trade-Record erstellen                                            │  │   │
│  │  │  • PnL berechnen (inkl. Slippage/Fee)                                │  │   │
│  │  │  • Cash aktualisieren                                                │  │   │
│  │  └─────────────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                        │
│                                          ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 5: Portfolio.update_equity(close_price)                              │   │
│  │                                                                            │   │
│  │  equity = cash + sum(open_positions.unrealized_pnl)                        │   │
│  │  equity_curve.push(equity)                                                 │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                        │
│                                          │ Loop weiter: idx += 1                  │
│                                          ▼                                        │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║                         END FOR                                             ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Phase 5: Result Building (Rust → Python)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: RESULT BUILDING                                     │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐     ┌─────────────────────────────────┐
│           PORTFOLIO (Final)             │     │         TRADE HISTORY           │
│                                         │     │                                 │
│  • final_equity: f64                    │     │  Vec<Trade> {                   │
│  • final_cash: f64                      │     │      Trade {                    │
│  • equity_curve: Vec<f64>               │     │          entry_time,            │
│  • open_positions: Vec<Position>        │     │          exit_time,             │
│                                         │     │          direction,             │
│                                         │     │          entry_price,           │
│                                         │     │          exit_price,            │
│                                         │     │          size,                  │
│                                         │     │          pnl,                   │
│                                         │     │          pnl_pips,              │
│                                         │     │          scenario,              │
│                                         │     │          exit_reason,           │
│                                         │     │          fees,                  │
│                                         │     │          slippage,              │
│                                         │     │      },                         │
│                                         │     │      ...                        │
│                                         │     │  }                              │
└────────────────────┬────────────────────┘     └────────────────┬────────────────┘
                     │                                           │
                     └─────────────────┬─────────────────────────┘
                                       │
                                       │ Metrics berechnen
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              METRICS COMPUTATION                                 │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Returns berechnen                                                        │   │
│  │  returns = equity_curve.pct_change()                                      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                          │
│                                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Performance Metrics                                                      │   │
│  │  • total_return: (final_equity - initial) / initial                       │   │
│  │  • cagr: compound_annual_growth_rate(returns)                             │   │
│  │  • sharpe_ratio: mean(returns) / std(returns) * sqrt(252)                 │   │
│  │  • sortino_ratio: mean(returns) / downside_std(returns) * sqrt(252)       │   │
│  │  • calmar_ratio: cagr / max_drawdown                                      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                          │
│                                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Risk Metrics                                                             │   │
│  │  • max_drawdown: max_peak_to_trough(equity_curve)                         │   │
│  │  • max_drawdown_duration: longest_underwater_period(equity_curve)         │   │
│  │  • volatility: std(returns) * sqrt(252)                                   │   │
│  │  • var_95: percentile(returns, 5)                                         │   │
│  │  • cvar_95: mean(returns < var_95)                                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                          │
│                                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Trade Metrics                                                            │   │
│  │  • total_trades: trades.len()                                             │   │
│  │  • win_rate: winning_trades / total_trades                                │   │
│  │  • profit_factor: gross_profit / gross_loss                               │   │
│  │  • avg_trade: mean(trades.pnl)                                            │   │
│  │  • avg_win: mean(winning_trades.pnl)                                      │   │
│  │  • avg_loss: mean(losing_trades.pnl)                                      │   │
│  │  • expectancy: win_rate * avg_win - (1 - win_rate) * |avg_loss|           │   │
│  │  • max_consecutive_wins: ...                                              │   │
│  │  • max_consecutive_losses: ...                                            │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                          │
│                                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Scenario Metrics (pro Szenario)                                          │   │
│  │  • scenario_trades: filter(trades, scenario == X)                         │   │
│  │  • scenario_win_rate, scenario_pnl, ...                                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ Alles zusammenführen
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKTEST RESULT                                     │
│                                                                                  │
│  BacktestResult {                                                                │
│      config: BacktestConfig,          // Echo der Eingabe                        │
│      metrics: Metrics,                // Alle berechneten Metriken               │
│      trades: Vec<Trade>,              // Vollständige Trade-Liste                │
│      equity_curve: Vec<f64>,          // Equity pro Bar                          │
│      timestamps: Vec<i64>,            // Timestamps für Equity                   │
│      execution_time_ms: u64,          // Laufzeit des Backtests                  │
│      candle_count: usize,             // Anzahl verarbeiteter Candles            │
│  }                                                                               │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         │ serde_json::to_string()
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RESULT JSON                                         │
│                              (String)                                            │
│                                                                                  │
│  Serialisiert für FFI-Rückgabe                                                   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         │ ══════════════════════════════════════
                                         │            FFI BOUNDARY
                                         │ ══════════════════════════════════════
                                         │
                                         │ return result_json
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PYTHON: RESULT PROCESSING                                 │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Result JSON (String)                   │
└────────────────────┬────────────────────┘
                     │
                     │ json.loads()
                     ▼
┌─────────────────────────────────────────┐
│  BacktestResult (Python Dict)           │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  JSON speichern │     │  Visualisierung │
│                 │     │                 │
│ • Trades JSON   │     │ • Equity Curve  │
│ • Metrics JSON  │     │ • Drawdown Plot │
│ • Config Echo   │     │ • Trade Chart   │
└─────────────────┘     └─────────────────┘
```

---

## 3. Datenstrukturen im Detail

### 3.1 Config JSON Schema (Python → Rust)

```json
{
  "strategy": {
    "name": "MeanReversionZScore",
    "parameters": {
      "atr_length": 14,
      "atr_mult": 1.5,
      "bb_length": 20,
      "std_factor": 2.0,
      "z_score_long": -2.0,
      "z_score_short": 2.0,
      "ema_length": 20,
      "enabled_scenarios": ["A", "B", "C"],
      "htf_filter": {
        "enabled": true,
        "timeframe": "D1",
        "ema_length": 50
      }
    }
  },
  "backtest": {
    "symbol": "EURUSD",
    "timeframe": "M5",
    "start_date": "2020-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:59:59Z",
    "initial_capital": 100000.0,
    "position_size": 0.01
  },
  "data": {
    "bid_path": "/data/parquet/EURUSD/EURUSD_M5_BID.parquet",
    "ask_path": "/data/parquet/EURUSD/EURUSD_M5_ASK.parquet",
    "htf_bid_path": "/data/parquet/EURUSD/EURUSD_D1_BID.parquet",
    "htf_ask_path": "/data/parquet/EURUSD/EURUSD_D1_ASK.parquet"
  },
  "execution": {
    "slippage_model": "fixed",
    "slippage_pips": 0.5,
    "fee_model": "spread",
    "spread_pips": 0.8
  }
}
```

### 3.2 Result JSON Schema (Rust → Python)

```json
{
  "status": "success",
  "config_echo": { ... },
  "execution_time_ms": 1234,
  "candle_count": 524160,
  "metrics": {
    "total_return": 0.2534,
    "cagr": 0.0821,
    "sharpe_ratio": 1.42,
    "sortino_ratio": 2.15,
    "calmar_ratio": 1.89,
    "max_drawdown": -0.0435,
    "max_drawdown_duration_days": 45,
    "volatility": 0.12,
    "var_95": -0.0123,
    "cvar_95": -0.0189,
    "total_trades": 1847,
    "win_rate": 0.523,
    "profit_factor": 1.67,
    "avg_trade_pnl": 13.72,
    "avg_win_pnl": 45.23,
    "avg_loss_pnl": -32.18,
    "expectancy": 8.45,
    "max_consecutive_wins": 12,
    "max_consecutive_losses": 7,
    "trades_per_day": 0.46,
    "scenario_metrics": {
      "A": { "trades": 523, "win_rate": 0.54, "pnl": 8934.23 },
      "B": { "trades": 412, "win_rate": 0.51, "pnl": 6721.12 }
    }
  },
  "trades": [
    {
      "id": 1,
      "entry_time": "2020-01-15T10:35:00Z",
      "exit_time": "2020-01-15T14:20:00Z",
      "direction": "long",
      "entry_price": 1.11234,
      "exit_price": 1.11389,
      "size": 0.01,
      "pnl": 15.50,
      "pnl_pips": 15.5,
      "scenario": "A",
      "exit_reason": "take_profit",
      "fees": 0.80,
      "slippage": 0.50
    }
  ],
  "equity_curve": [100000.0, 100015.50, 99985.20, ...],
  "timestamps": [1579084500, 1579085400, ...]
}
```

---

## 4. Data Flow Regeln

### 4.1 Richtung und Ownership

| Daten | Richtung | Ownership | Lebensdauer |
|-------|----------|-----------|-------------|
| Config JSON | Python → Rust | Rust übernimmt (Kopie) | Backtest-Dauer |
| Parquet Files | Disk → Rust | Rust (exklusiv) | Backtest-Dauer |
| Candles | Rust-intern | CandleStore | Backtest-Dauer |
| Indicators | Rust-intern | IndicatorCache | Backtest-Dauer |
| BarContext | Rust-intern | Per-Bar (temporär) | Eine Iteration |
| Signals | Rust-intern | Per-Bar (temporär) | Eine Iteration |
| Portfolio State | Rust-intern | Portfolio Struct | Backtest-Dauer |
| Trades | Rust-intern | Vec<Trade> | Backtest-Dauer |
| Result JSON | Rust → Python | Python übernimmt | Nach Backtest |

### 4.2 Zero-Copy Prinzipien

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          ZERO-COPY FLOW                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   Parquet File                                                                 ║
║        │                                                                       ║
║        │  Memory-mapped read (arrow-rs)                                        ║
║        ▼                                                                       ║
║   Arrow RecordBatch ────────────────────────────────────────┐                  ║
║        │                                                    │                  ║
║        │  Zeiger-Konvertierung (kein Kopieren)              │                  ║
║        ▼                                                    │                  ║
║   Vec<Candle> ──────────────────────────────────────────────┤                  ║
║        │                                                    │                  ║
║        │  Slice-Referenz (&[Candle])                        │ Alle Daten       ║
║        ▼                                                    │ bleiben im       ║
║   Indicator Computation ────────────────────────────────────┤ Speicher an      ║
║        │                                                    │ EINEM Ort        ║
║        │  Index-basierter Zugriff                           │                  ║
║        ▼                                                    │                  ║
║   BarContext (nur Referenzen) ──────────────────────────────┤                  ║
║        │                                                    │                  ║
║        │  ctx.bid[idx] → &Candle                            │                  ║
║        │  ctx.indicators["EMA"][idx] → &f64                 │                  ║
║        ▼                                                    │                  ║
║   Strategy.on_bar(&ctx) ────────────────────────────────────┘                  ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 4.3 Keine Rückflüsse während Backtest

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          UNIDIREKTIONALER FLOW                                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   Python ─────────────────────────────────────────────────────────▶ Rust       ║
║      │                                                                │        ║
║      │  Config JSON (einmalig)                                        │        ║
║      │                                                                │        ║
║      X ◀───────────────────────────────────────────────────────────── X        ║
║      │     KEINE Callbacks                                            │        ║
║      │     KEINE Progress-Updates                                     │        ║
║      │     KEINE Logging-Calls                                        │        ║
║      │     KEINE Mid-Backtest Queries                                 │        ║
║      │                                                                │        ║
║      │ ◀──────────────────────────────────────────────────────────────│        ║
║      │  Result JSON (einmalig, am Ende)                               │        ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 5. Multi-Timeframe Data Flow

### 5.1 HTF-Daten Alignment

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-TIMEFRAME ALIGNMENT                                   │
└────────────────────────────────────────────────────────────────────────────────┘

M5 Timeline:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ 0   │ 1   │ 2   │ 3   │ 4   │ 5   │ 6   │ 7   │ 8   │ 9   │ 10  │ 11  │ 12  │ M5
│00:00│00:05│00:10│00:15│00:20│00:25│00:30│00:35│00:40│00:45│00:50│00:55│01:00│
└──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
   │     │     │     │     │     │     │     │     │     │     │     │     │
   └─────┴─────┴─────┴─────┴─────┘     └─────┴─────┴─────┴─────┴─────┴─────┘
              │                                      │
              ▼                                      ▼
         ┌─────────┐                            ┌─────────┐
H1:      │    0    │                            │    1    │
         │  00:00  │                            │  01:00  │
         └────┬────┘                            └────┬────┘
              │                                      │
              │  htf_index_map[m5_idx] → h1_idx      │
              │                                      │
              ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  HTF Index Mapping:                                                              │
│                                                                                  │
│  m5_idx: 0  1  2  3  4  5  6  7  8  9  10 11 12                                  │
│  h1_idx: 0  0  0  0  0  0  0  0  0  0  0  0  1                                   │
│                                                                                  │
│  Bei M5 idx=5 (00:25): HTF-Indikator für H1 bar 0 (00:00-00:59) verwenden       │
│  → Kein Lookahead! H1 bar 0 ist erst um 01:00 "fertig"                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 HTF Indicator Access Pattern

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     HTF INDICATOR ACCESS                                        │
└────────────────────────────────────────────────────────────────────────────────┘

BarContext.get_htf_indicator(name, tf, params):

    1. Aktuellen M5 idx kennen
    2. HTF idx aus Mapping holen: htf_idx = htf_index_map[m5_idx]
    3. WICHTIG: htf_idx - 1 verwenden (letzte ABGESCHLOSSENE Bar!)
    4. Indikatorwert zurückgeben: indicators[(name, tf, params)][htf_idx - 1]

┌─────────────────────────────────────────────────────────────────────────────────┐
│  LOOKAHEAD PREVENTION:                                                           │
│                                                                                  │
│  Um 00:25 (M5 idx=5):                                                            │
│  • H1 Bar 0 (00:00-00:59) ist NICHT abgeschlossen                                │
│  • Wir dürfen NUR H1 Bar -1 (vorheriger Tag) verwenden                          │
│  • Oder: "H1 EMA bei Tagesanfang" (Close von gestern)                            │
│                                                                                  │
│  Um 01:05 (M5 idx=13):                                                           │
│  • H1 Bar 0 (00:00-00:59) ist jetzt abgeschlossen                                │
│  • H1 Bar 1 (01:00-01:59) läuft                                                  │
│  • Wir dürfen H1 Bar 0 verwenden                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Flow Validierung

### 6.1 Invarianten die geprüft werden müssen

| # | Invariante | Prüfung |
|---|------------|---------|
| I1 | Keine Lookahead-Bias | HTF-Daten nur von abgeschlossenen Bars |
| I2 | Timestamps monoton steigend | candles[i].timestamp < candles[i+1].timestamp |
| I3 | Bid ≤ Ask | Für jede Bar: bid.close ≤ ask.close |
| I4 | Indikatoren aligned | indicators.len() == candles.len() |
| I5 | NaN nur in Warmup | indicators[warmup:] enthält keine NaN |
| I6 | Portfolio Balance konsistent | cash + margin + unrealized_pnl == equity |
| I7 | Trades vollständig | Jeder geschlossene Trade hat entry + exit |

### 6.2 Checkpoints im Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           VALIDATION CHECKPOINTS                                │
└────────────────────────────────────────────────────────────────────────────────┘

CHECKPOINT 1: Nach Config Parse
├── Alle Pflichtfelder vorhanden?
├── Parameter in gültigen Ranges?
└── Pfade existieren?

CHECKPOINT 2: Nach Data Load
├── Parquet gelesen ohne Fehler?
├── Timestamps monoton?
├── Bid/Ask konsistent?
├── Date Range korrekt gefiltert?
└── Ausreichend Daten für Warmup?

CHECKPOINT 3: Nach Indicator Computation
├── Alle angeforderten Indikatoren berechnet?
├── Länge == candles.len()?
├── NaN nur in Warmup-Bereich?
└── HTF-Mapping korrekt?

CHECKPOINT 4: Nach Event Loop
├── Portfolio Balance konsistent?
├── Alle Positionen geschlossen (falls end_of_backtest)?
├── Equity Curve vollständig?
└── Trade-Liste vollständig?

CHECKPOINT 5: Vor Result Serialization
├── Alle Metriken berechnet?
├── Keine NaN/Inf in Metriken?
└── JSON serialisierbar?
```

---

## 7. Zusammenfassung: Data Flow auf einen Blick

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OMEGA V2 DATA FLOW - ZUSAMMENFASSUNG                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐                                                               │
│  │   Python     │                                                               │
│  │ Orchestrator │                                                               │
│  └──────┬───────┘                                                               │
│         │                                                                       │
│         │ 1. Config JSON ──────────────────────────────────────────────────┐    │
│         │                                                                  │    │
│         │ ════════════════════ FFI (EINMAL) ═══════════════════════════   │    │
│         │                                                                  │    │
│         │                      ┌───────────────────────────────────────────┘    │
│         │                      │                                                │
│         │                      ▼                                                │
│         │         ┌────────────────────────────────────────────────────┐        │
│         │         │                   RUST ENGINE                       │        │
│         │         │                                                     │        │
│         │         │   2. Parquet ───▶ CandleStore                       │        │
│         │         │                         │                           │        │
│         │         │   3. Candles ───▶ IndicatorCache                    │        │
│         │         │                         │                           │        │
│         │         │   4. for bar in candles:                            │        │
│         │         │        │                                            │        │
│         │         │        ├─▶ BarContext (Refs)                        │        │
│         │         │        ├─▶ Strategy.on_bar() → Signal               │        │
│         │         │        ├─▶ Execution.process(Signal)                │        │
│         │         │        ├─▶ Portfolio.check_stops()                  │        │
│         │         │        └─▶ Portfolio.update_equity()                │        │
│         │         │                         │                           │        │
│         │         │   5. Trades ───▶ Metrics                            │        │
│         │         │                         │                           │        │
│         │         │   6. BacktestResult ────┘                           │        │
│         │         │                                                     │        │
│         │         └────────────────────────────────────────────────────┘        │
│         │                                     │                                 │
│         │ ════════════════════ FFI (EINMAL) ═══════════════════════════        │
│         │                                     │                                 │
│         │◀──────── 7. Result JSON ────────────┘                                 │
│         │                                                                       │
│  ┌──────┴───────┐                                                               │
│  │   Python     │                                                               │
│  │  Reporter    │                                                               │
│  └──────────────┘                                                               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KERNPRINZIPIEN:                                                                │
│  ✓ Single FFI Boundary (1 Call rein, 1 Call raus)                               │
│  ✓ Zero-Copy innerhalb Rust (Referenzen, kein Kopieren)                         │
│  ✓ Unidirektional (keine Callbacks während Backtest)                            │
│  ✓ Alle Daten pre-computed (Indikatoren vor Loop)                               │
│  ✓ Immutable Input (Candles/Indicators ändern sich nicht)                       │
│  ✓ Lookahead-Prevention (HTF nur abgeschlossene Bars)                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*Dieser Data Flow Plan ist die verbindliche Spezifikation für die Implementierung des Omega V2 Backtesting-Systems.*
