# Omega V2 – Data Flow Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Vollständige Spezifikation des Datenflusses im Omega V2 Backtesting-System  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Vision, strategische Ziele, Erfolgskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Übergeordneter Blueprint, Module, Regeln |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives JSON-Config-Schema (Felder, Defaults, Validierung, Migration) |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell (Bid/Ask-Regeln, Fills, SL/TP, Slippage/Fees) |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Datei- und Verzeichnisstruktur, Interfaces |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Normative Metrik-Keys, Definitionen/Units, Edge-Cases |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (Artefakte, Schema, Zeit/Units, Pfade) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Packaging/Build-Matrix |

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

### 1.3 UTC/Timestamp-Contract (Vorschlag A – festgezogen)

Dieser Contract gilt **für alle Zeitachsen** im V2-Core (Candles, News-Events, Equity, Trades).

| Ebene | Repräsentation | Semantik |
|------:|----------------|----------|
| Parquet (Disk) | Arrow `Timestamp(Nanosecond, "UTC")` in Spalte **`UTC time`** | **Candle Open-Time** (Beginn der Kerze) bzw. bei Events die Event-Zeit als UTC-Instant |
| Rust (Core) | `i64` **epoch-nanoseconds** in UTC (`timestamp_ns`) | Unix epoch in Nanosekunden; **strictly monotonic increasing** und **unique** pro Zeitreihe |
| Python (Orchestrator/Reporter) | timezone-aware UTC Datetimes (z.B. `datetime64[ns, UTC]`) | Nur Serialisierung/Reporting; keine Zeitlogik im Hot-Path |

**Invarianten:**

- Candle-`UTC time` ist **Open-Time**, nicht Close-Time.
- Zeitzone ist immer **UTC** (keine lokale Zeitzone, kein DST).
- `timestamp_ns` wird im Core als `i64` geführt; Vergleiche/Filter (Start/Ende) erfolgen in derselben Einheit.

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
│ • timeframes    │
│   - primary     │
│ • run_mode       │  ← dev|prod (Determinismus/Regression)
│ • data_mode      │  ← candle|tick (Daten-Granularität)
│ • rng_seed       │  ← optional (v.a. für dev)
│ • parameters    │
│ • warmup_bars   │  ← Default: 500
│ • sessions       │  ← optional: Trading Sessions (UTC)
│ • execution      │  ← optional: Order-Typen/Direction/Max-Positions
│ • costs          │  ← keine Pfade (fixed / environment driven)
│ • news_filter    │  ← optional: enabled + Window/Impact (keine Pfade)
└────────┬────────┘
         │
         │ Laden & Validieren
         ▼
┌─────────────────┐
│  BacktestConfig │
│  (Python Dict)  │
│                 │
│ Anreichern mit: │
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

Phase 2 ist die **zentrale Stelle für alle Datenqualitäts-Operationen**. Hier werden:
- Pfade automatisch aus Config-Parametern abgeleitet
- Bid/Ask Alignment sichergestellt
- Datenqualität validiert
- Warmup-Verfügbarkeit geprüft
- Optionale Alternative-Data (News) geladen und indiziert
- Kosten-/Symbol-Configs (YAML) als Input für Execution vorbereitet

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: DATA LOADING                                        │
│                    (Zentrale Datenqualitäts-Phase)                                │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.1: AUTOMATISCHE PATH RESOLUTION                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input aus Config (siehe docs/OMEGA_V2_CONFIG_SCHEMA_PLAN.md):                    │
│  • symbol: "EURUSD"                                                              │
│  • timeframes.primary: "M5"                                                      │
│  • timeframes.htf_filter.timeframe: "D1" (optional)                              │
│                                                                                  │
│  Input aus Environment/Defaults (nicht in Config):                                │
│  • data_root: "/data/parquet"  (Environment oder Default)                        │
│  • execution_costs_file: (fixed / Env)                                            │
│  • symbol_specs_file: (fixed / Env)                                               │
│  • news_calendar_file: (fixed / Env)                                              │
│                                                                                  │
│  Automatisch generierte Pfade:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PRIMARY TIMEFRAME (timeframes.primary = M5):                            │    │
│  │  • bid_path = {data_root}/{symbol}/{symbol}_{timeframes.primary}_BID.parquet │    │
│  │  • ask_path = {data_root}/{symbol}/{symbol}_{timeframes.primary}_ASK.parquet │    │
│  │                                                                          │    │
│  │  Beispiel:                                                               │    │
│  │  • /data/parquet/EURUSD/EURUSD_M5_BID.parquet                           │    │
│  │  • /data/parquet/EURUSD/EURUSD_M5_ASK.parquet                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  HTF TIMEFRAMES (falls timeframes.htf_filter.enabled):                   │    │
│  │  • htf_bid_path = {data_root}/{symbol}/{symbol}_{htf_tf}_BID.parquet    │    │
│  │  • htf_ask_path = {data_root}/{symbol}/{symbol}_{htf_tf}_ASK.parquet    │    │
│  │                                                                          │    │
│  │  Beispiel (D1):                                                          │    │
│  │  • /data/parquet/EURUSD/EURUSD_D1_BID.parquet                           │    │
│  │  • /data/parquet/EURUSD/EURUSD_D1_ASK.parquet                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ❌ FEHLER wenn Datei nicht existiert → Backtest ABBRUCH                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.2: PARQUET LADEN (Pro Timeframe)                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐                                     │
│  │  BID Parquet    │     │  ASK Parquet    │                                     │
│  │  (Disk)         │     │  (Disk)         │                                     │
│  │                 │     │                 │                                     │
│  │ EURUSD_M5_BID   │     │ EURUSD_M5_ASK   │                                     │
│  │ .parquet        │     │ .parquet        │                                     │
│  └────────┬────────┘     └────────┬────────┘                                     │
│           │                       │                                              │
│           │  arrow-rs             │                                              │
│           │  read_parquet()       │                                              │
│           ▼                       ▼                                              │
│  ┌─────────────────────────────────────────┐                                     │
│  │           RAW CANDLE DATA               │                                     │
│  │                                         │                                     │
│  │  Vec<RawCandle> {                       │                                     │
│  │        timestamp_ns: i64,               │                                     │
│  │        high: f64,                       │                                     │
│  │        low: f64,                        │                                     │
│  │        close: f64,                      │                                     │
│  │        volume: f64,                     │                                     │
│  │  }                                      │                                     │
│  └─────────────────────────────────────────┘                                     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.3: BID/ASK ALIGNMENT (KRITISCH!)                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Problem: Bid und Ask Parquets können unterschiedliche Timestamps haben!         │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  Alignment-Strategie: INNER JOIN auf Timestamps                           │  │
│  │                                                                           │  │
│  │  BID Timestamps:  [00:00, 00:05, 00:10, 00:20, 00:25, 00:30]              │  │
│  │  ASK Timestamps:  [00:00, 00:05, 00:15, 00:20, 00:25, 00:30]              │  │
│  │                          ↓                                                │  │
│  │  ALIGNED:         [00:00, 00:05, 00:20, 00:25, 00:30]                     │  │
│  │                                                                           │  │
│  │  → Nur Timestamps behalten, die in BEIDEN vorhanden sind                  │  │
│  │  → Fehlende Bars werden NICHT interpoliert (Datenintegrität!)             │  │
│                                                                                  │
│                                                                                  │
│  Logging:                                                                        │
│  • Anzahl Bid-Bars vor Alignment                                                 │
│  • Anzahl Ask-Bars vor Alignment                                                 │
│  • Anzahl Bars nach Alignment                                                    │
│  • Anzahl verworfener Bars (Warning wenn > 1%)                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.4: DATENQUALITÄTS-VALIDIERUNG                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  CHECK 1: Timestamps monoton steigend                                    │    │
│  │  for i in 1..len {                                                       │    │
│  │      assert!(timestamps[i] > timestamps[i-1]);                           │    │
│  │  }                                                                       │    │
│  │  ❌ Fehler → Backtest ABBRUCH (korrupte Daten)                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  CHECK 2: Bid ≤ Ask für alle Bars                                        │    │
│  │  for i in 0..len {                                                       │    │
│  │      assert!(bid[i].close <= ask[i].close);                              │    │
│  │      assert!(bid[i].high <= ask[i].high);                                │    │
│  │  }                                                                       │    │
│  │  ❌ Fehler → Backtest ABBRUCH (ungültige Spread-Daten)                   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  CHECK 3: Keine NaN/Inf in OHLC                                          │    │
│  │  for candle in candles {                                                 │    │
│  │      assert!(candle.o.is_finite() && candle.h.is_finite() ...);          │    │
│  │  }                                                                       │    │
│  │  ❌ Fehler → Backtest ABBRUCH (korrupte Preisdaten)                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  CHECK 4: OHLC Konsistenz                                                │    │
│  │  for candle in candles {                                                 │    │
│  │      assert!(candle.l <= candle.o && candle.l <= candle.c);              │    │
│  │      assert!(candle.h >= candle.o && candle.h >= candle.c);              │    │
│  │      assert!(candle.l <= candle.h);                                      │    │
│  │  }                                                                       │    │
│  │  ❌ Fehler → Backtest ABBRUCH (inkonsistente OHLC)                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  CHECK 5: Lücken-Analyse (Warning, kein Abbruch)                         │    │
│  │  Gap = timestamp[i] - timestamp[i-1] > expected_interval * 2             │    │
│  │  → Logging: Anzahl und Positionen der Gaps                               │    │
│  │  → Warning wenn > 5% der erwarteten Bars fehlen                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.5: DATE RANGE FILTER                                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Filterung auf Config-Zeitraum:                                                  │
│  • start_date aus Config (inklusiv)                                              │
│  • end_date aus Config (inklusiv)                                                │
│                                                                                  │
│  candles = candles.filter(|c| c.timestamp >= start && c.timestamp <= end);       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.5b: NEWS PARQUET LADEN & INDEX (optional)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Nur wenn news_filter.enabled == true:                                            │
│                                                                                  │
│  1. News Parquet laden (arrow-rs)                                                │
│     • Erwartet Arrow `Timestamp(Nanosecond, "UTC")` (tz-aware)                  │
│       – im Core als `i64` epoch-nanoseconds UTC (`timestamp_ns`)                 │
│     • Fail Fast bei nicht-monotonen oder ungültigen Zeiten                       │
│                                                                                  │
│  2. Date Range Filter (optional erweitert um Window pre/post)                    │
│                                                                                  │
│  3. News Index erzeugen (für O(1) Zugriff im Event-Loop)                         │
│                                                                                  │
│     Beispiel-Policy (vereinfachte Darstellung):                                  │
│     ┌───────────────────────────────────────────────────────────────────────┐    │
│     │  NewsEvent {                                                          │    │
│     │      timestamp_ns: i64,   // UTC epoch-ns (Event-Time)                 │    │
│     │      impact: u8,       // z.B. 1=low, 2=medium, 3=high                 │    │
│     │      currency: String, // optional                                     │    │
│     │  }                                                                     │    │
│     │                                                                       │    │
│     │  Für jeden Candle Timestamp ts:                                       │    │
│     │  is_blocked = exists event e mit                                     │    │
│     │      e.impact >= config.news_filter.min_impact                        │    │
│     │  und ts in [e.ts - pre_window, e.ts + post_window]                    │    │
│     └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Output: NewsCalendarIndex / NewsMask (Länge == candles.len())                   │
│                                                                                  │
│  ❌ FEHLER wenn enabled aber Datei fehlt/korrupt → Backtest ABBRUCH              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.6: WARMUP-VALIDIERUNG (KRITISCH!)                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Warmup-Konfiguration:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  WICHTIG: Warmup ist KONFIGURIERBAR, nicht indikator-abhängig!           │    │
│  │                                                                          │    │
│  │  Grund: Konsistenz zwischen Live-Trading und Backtest.                   │    │
│  │  Im Live-Trading werden immer N Kerzen geladen, unabhängig               │    │
│  │  von den verwendeten Indikatoren.                                        │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  CONFIG WARMUP SETTINGS:                                         │    │    │
│  │  │                                                                  │    │    │
│  │  │  warmup_bars: 500,   // Warmup-Bars (global, pro TF angewendet)  │    │    │
│  │  │                                                                  │    │    │
│  │  │  DEFAULT = 500 Bars (wie im Live-Trading)                        │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  Warmup-Auflösung:                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  warmup = config.warmup_bars.unwrap_or(500)                      │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  Beispiele mit Default (500):                                            │    │
│  │  • M5 Primary TF: 500 M5-Bars = ~42 Stunden                              │    │
│  │  • H1 Primary TF: 500 H1-Bars = ~21 Tage                                 │    │
│  │  • D1 HTF: 500 D1-Bars = ~2 Jahre                                        │    │
│  │                                                                          │    │
│  │  Custom Beispiele:                                                       │    │
│  │  • warmup_bars = 1000 → Doppelte Warmup-Phase                            │    │
│  │  • warmup_bars = 200 → Reduzierter Warmup                                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Validierung:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  available_bars = candles.len()                                          │    │
│  │  required_warmup = config.warmup_bars.unwrap_or(500)                     │    │
│  │                                                                          │    │
│  │  if available_bars <= required_warmup {                                  │    │
│  │      return Err(BacktestError::InsufficientData {                        │    │
│  │          required: required_warmup + 1,                                  │    │
│  │          available: available_bars,                                      │    │
│  │          configured_warmup: required_warmup,                             │    │
│  │          symbol: config.symbol.clone(),                                  │    │
│  │          timeframe: config.timeframes.primary.clone(),                   │    │
│  │      });                                                                 │    │
│  │  }                                                                       │    │
│  │                                                                          │    │
│  │  ❌ Fehler → Backtest ABBRUCH (nicht genug Daten für Warmup)             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  HTF Warmup-Validierung (falls zusätzliche TFs aktiv sind):                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  htf_available = htf_candles.len()                                       │    │
│  │  htf_required = config.warmup_bars.unwrap_or(500)                        │    │
│  │                                                                          │    │
│  │  if htf_available < htf_required {                                       │    │
│  │      return Err(BacktestError::InsufficientHTFData { ... });             │    │
│  │  }                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Erfolgs-Logging:                                                                │
│  • "Primary Warmup: {primary_warmup} Bars (configured), Trading: {n} Bars"       │
│  • "HTF Warmup: {htf_warmup} Bars (configured)" (falls HTF enabled)              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.7: CANDLE STORE ERSTELLEN                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────┐                                     │
│  │           CANDLE STORE                   │                                     │
│  │                                         │                                     │
│  │  CandleStore {                          │                                     │
│  │      bid: Vec<Candle>,                  │                                     │
│  │      ask: Vec<Candle>,                  │                                     │
│  │      timestamps: Vec<i64>,              │                                     │
│  │      len: usize,                        │                                     │
│  │      warmup_periods: usize,             │  // NEU: Warmup aus Validierung     │
│  │  }                                      │                                     │
│  │                                         │                                     │
│  │  Candle {                               │                                     │
│  │      o: f64, h: f64, l: f64, c: f64,   │                                     │
│  │  }                                      │                                     │
│  └─────────────────────────────────────────┘                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2.8: HTF LADEN & ALIGNMENT (falls konfiguriert)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Falls htf_filter.enabled == true:                                               │
│                                                                                  │
│  1. HTF Parquets laden (automatisch abgeleitete Pfade)                           │
│  2. HTF Bid/Ask Alignment (wie bei Primary TF)                                   │
│  3. HTF Datenqualitäts-Validierung                                               │
│  4. HTF Date Range Filter                                                        │
│  5. HTF Warmup-Validierung                                                       │
│                                                                                  │
│  HTF Index Mapping erstellen:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  htf_index_map: HashMap<i64, usize>                                      │    │
│  │                                                                          │    │
│  │  Für jeden Primary-TF Timestamp:                                         │    │
│  │  → Finde zugehörigen HTF-Bar-Index                                       │    │
│  │  → Speichere Mapping: primary_ts → htf_idx                               │    │
│  │                                                                          │    │
│  │  htf_index_map[m5_ts] = find_htf_bar_index(m5_ts, htf_timestamps)        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: MULTI-TIMEFRAME STORE                                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────┐                                     │
│  │        MULTI-TIMEFRAME STORE            │                                     │
│  │                                         │                                     │
│  │  MultiTfStore {                         │                                     │
│  │      primary: CandleStore,              │  // M5 (Bid+Ask aligned)            │
│  │      htf: Option<CandleStore>,          │  // D1 (Bid+Ask aligned)            │
│  │      htf_index_map: HashMap<i64, usize>,│  // M5_ts → D1_idx                  │
│  │      data_quality_report: DataReport,   │  // Statistiken                     │
│  │  }                                      │                                     │
│  │                                         │                                     │
│  │  DataReport {                           │                                     │
│  │      primary_bars_raw: usize,           │                                     │
│  │      primary_bars_aligned: usize,       │                                     │
│  │      htf_bars_raw: Option<usize>,       │                                     │
│  │      htf_bars_aligned: Option<usize>,   │                                     │
│  │      gaps_detected: usize,              │                                     │
│  │      warmup_periods: usize,             │                                     │
│  │      trading_periods: usize,            │                                     │
│  │  }                                      │                                     │
│  └─────────────────────────────────────────┘                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2.1 Data Loading Error Handling

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA LOADING FEHLERBEHANDLUNG                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ABBRUCH-FEHLER (Backtest wird nicht gestartet):                                 │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  ❌ FileNotFound                                                                 │
│     → Parquet-Datei existiert nicht                                              │
│     → Error: "Data file not found: {path}"                                       │
│                                                                                  │
│  ❌ InsufficientData                                                             │
│     → Nicht genug Bars für Warmup + Trading                                      │
│     → Error: "Insufficient data: need {n}, have {m}"                             │
│                                                                                  │
│  ❌ CorruptData                                                                  │
│     → NaN/Inf in Preisen, nicht-monotone Timestamps                              │
│     → Error: "Corrupt data at index {i}: {details}"                              │
│                                                                                  │
│  ❌ InvalidSpread                                                                │
│     → Bid > Ask (unmöglicher Zustand)                                            │
│     → Error: "Invalid spread at {timestamp}: bid={bid}, ask={ask}"               │
│                                                                                  │
│  ❌ AlignmentFailure                                                             │
│     → Keine gemeinsamen Timestamps zwischen Bid und Ask                          │
│     → Error: "Bid/Ask alignment failed: no common timestamps"                    │
│                                                                                  │
│  WARNINGS (Backtest läuft weiter, aber loggt):                                   │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  ⚠️ LargeGaps                                                                    │
│     → > 5% der erwarteten Bars fehlen                                            │
│     → Warning: "Large gaps detected: {n} missing bars ({pct}%)"                  │
│                                                                                  │
│  ⚠️ AlignmentLoss                                                                │
│     → > 1% der Bars durch Alignment verloren                                     │
│     → Warning: "Alignment discarded {n} bars ({pct}%)"                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
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

Vor Start des Loops wird die **ExecutionEngine** einmalig initialisiert:

- **Fees/Costs**: Laden aus `configs/execution_costs.yaml` + `configs/symbol_specs.yaml` (YAML → Rust Structs)
- **Slippage RNG**:
    - `mode=dev`: deterministisch via `rng_seed`
    - `mode=prod`: nicht-deterministisch (Stochastik erlaubt)


```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUST: EVENT LOOP                                          │
└──────────────────────────────────────────────────────────────────────────────────┘

                     ┌─────────────────────────────────────────┐
                     │           EINGABE-DATEN                 │
                     │                                         │
                     │  • CandleStore (bid, ask)               │
                     │  • IndicatorCache                       │
                     │  • SessionSchedule (UTC)                │
                     │  • NewsCalendarIndex (optional)          │
                     │  • ExecutionCosts (Fees/Slippage, YAML)  │
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
│  │      session_open: bool,                // nur Entries innerhalb Sessions    │   │
│  │      news_blocked: bool,                // optional: News Window aktiv       │   │
│  │  }                                                                         │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                        │
│                                          ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1b: Entry-Gates (Session/News)                                       │   │
│  │                                                                            │   │
│  │  entry_allowed = ctx.session_open && !ctx.news_blocked;                    │   │
│  │                                                                            │   │
│  │  // Indikatoren sind bereits berechnet. Entry-Signale werden nur           │   │
│  │  // bei entry_allowed evaluiert. Positions-/Exit-Logik (z.B. SL/TP)        │   │
│  │  // bleibt unabhängig davon.                                               │   │
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
│  │  • ctx.session_open / ctx.news_blocked (Entry-Gates)                       │   │
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
│ • Outputs (SoT) │
│ • Config (SoT)  │
└─────────────────┘     └─────────────────┘
```

Normativ (Outputs): `docs/OMEGA_V2_OUTPUT_CONTRACT_PLAN.md`

Normativ (Config): `docs/OMEGA_V2_CONFIG_SCHEMA_PLAN.md`


---

## 3. Data Flow Regeln

### 3.1 Richtung und Ownership

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

### 3.2 Zero-Copy Prinzipien

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

### 3.3 Keine Rückflüsse während Backtest

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

## 4. Multi-Timeframe Data Flow

### 4.1 HTF-Daten Alignment

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

### 4.2 HTF Indicator Access Pattern

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

## 5. Data Flow Validierung

### 5.1 Invarianten die geprüft werden müssen

| # | Invariante | Prüfung | Phase | Bei Verletzung |
|---|------------|---------|-------|----------------|
| I1 | Keine Lookahead-Bias | HTF-Daten nur von abgeschlossenen Bars | Phase 4 | ABBRUCH |
| I2 | Timestamps monoton steigend | candles[i].timestamp < candles[i+1].timestamp | Phase 2 | ABBRUCH |
| I3 | Bid ≤ Ask | Für jede Bar: bid.close ≤ ask.close | Phase 2 | ABBRUCH |
| I4 | Indikatoren aligned | indicators.len() == candles.len() | Phase 3 | ABBRUCH |
| I5 | NaN nur in Warmup | indicators[warmup:] enthält keine NaN | Phase 3 | ABBRUCH |
| I6 | Portfolio Balance konsistent | cash + margin + unrealized_pnl == equity | Phase 4 | ABBRUCH |
| I7 | Trades vollständig | Jeder geschlossene Trade hat entry + exit | Phase 5 | WARNING |
| **I8** | **Bid/Ask Alignment** | **Gleiche Timestamps für Bid und Ask auf allen TFs** | **Phase 2** | **ABBRUCH** |
| **I9** | **Warmup verfügbar** | **available_bars > required_warmup** | **Phase 2** | **ABBRUCH** |
| **I10** | **HTF Alignment** | **Jeder Primary-TF Timestamp hat gültigen HTF-Index** | **Phase 2** | **ABBRUCH** |
| **I11** | **Pfade existieren** | **Alle automatisch generierten Datenpfade existieren** | **Phase 2** | **ABBRUCH** |

### 5.2 Checkpoints im Flow

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

## 6. Zusammenfassung: Data Flow auf einen Blick

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
