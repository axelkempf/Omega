# Omega V2 – Execution Model Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Normative Spezifikation des Ausführungsmodells (Bid/Ask-Regeln, Fill-Algorithmen, Intrabar-Tie-Breaks, SL/TP-Prioritäten, Slippage/Fees, Sizing/Quantisierung)
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite; fachliche Referenz für V1-Python-Parität: `src/old/backtest_engine/core/execution_simulator.py`

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, FFI-Grenze, Verantwortlichkeiten |
| [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) | Strategie-Spezifikation (MVP: Mean Reversion Z-Score), Szenarien 1–6, Guards/Filter, Indikatoren |
| [OMEGA_V2_TRADE_MANAGER_PLAN.md](OMEGA_V2_TRADE_MANAGER_PLAN.md) | Trade-/Position-Management: Rule-Prioritäten, MaxHoldingTime, Stop-Update-Policy |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Datenfluss, Bid/Ask-Alignment, Timestamp-Contract |
| [OMEGA_V2_DATA_GOVERNANCE_PLAN.md](OMEGA_V2_DATA_GOVERNANCE_PLAN.md) | Data-Quality-Policies (Alignment/Gaps/Duplicates), News=Parquet, Snapshots/Manifests |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Module/Crates (Execution, Costs, Portfolio), Schnittstellen |
| [OMEGA_V2_INDICATOR_CACHE__PLAN.md](OMEGA_V2_INDICATOR_CACHE__PLAN.md) | Indikator-Cache: Multi-TF, Stepwise-Semantik, V1-Python-Parität |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Config-Schema (run_mode, data_mode, rng_seed, costs) |
| [OMEGA_V2_METRICS_DEFINITION_PLAN.md](OMEGA_V2_METRICS_DEFINITION_PLAN.md) | Normative Semantik für Profit/Fees/Drawdown-Metriken |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (`trades.json`, Exit-`reason`, Zeit/Units) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Packaging/Build-Matrix |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Logging/Tracing (tracing), Profiling (flamegraph/pprof), Performance-Counter, Determinismus |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Execution-Validierung (Tie-Breaks, Same-Bar), Golden-Files, Parität/Determinismus |
| [OMEGA_V2_FORMATTING_PLAN.md](OMEGA_V2_FORMATTING_PLAN.md) | Format-/Lint-Regeln (Code/Doku/Kommentare), Durchsetzung via pre-commit + CI |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | CI/CD Workflow, Quality Gates, Build-Matrix, Security, Release-Assets |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für das Omega V2 Backtest-Ausführungsmodell.

**Primäres Ziel:**

- **Kanonische V2-Semantik** (Default) mit explizit dokumentierten Abweichungen zu V1.
- Zusätzlich existiert ein **V1-Python-Paritätsmodus** (`execution_variant = "v1_parity"`) für Regression/CI, der die V1-Semantik so nah wie möglich repliziert.

**Normativ (MVP):**

- Order-Typen: `market`, `limit`, `stop`
- Daten-Modi: `candle` (MVP); `tick` ist Post-MVP
- Exits: `stop_loss`, `take_profit`, `timeout` (+ Break-Even/Trailing als Meta-Label)
- Kosten: Spread ist implizit über Bid/Ask im Data-Contract enthalten; zusätzlich Slippage + Fees/Commission

---

## 2. Geltungsbereich

### 2.1 In Scope

- Fill- und Trigger-Regeln für `market/limit/stop`.
- Bid/Ask-Seitenwahl je Richtung (long/short) für Entry/Exit.
- Candle-Mode Tie-Breaks inkl. `in_entry_candle` und `pip_buffer`.
- Slippage- und Fee/Commission-Applikation (inkl. Determinismus).
- Sizing/Quantisierung und Guardrails (min. SL-Distanz).

### 2.2 Out of Scope (vorerst)

- Margin-/Leverage-Modell, Liquidation, Zins-/Swap-Modell.
- Netting/Hedging-Accounting über mehrere Positionen (V2-MVP ist Single-Symbol; Positionen sind unabhängig).
- Tick-Mode Ausführungsmodell (V2-MVP fokussiert Candle-Mode).
- Partial-Fills / Order-Book Simulation.

---

## 3. Begriffe und Datenmodelle

### 3.1 Zeit- und Candle-Definition

- Candles sind Bar-Daten mit **Open-Time** als `timestamp_ns` (siehe Data-Flow Plan).
- Alle Zeiten sind UTC.

### 3.2 Position/Order State

- `pending`: Limit/Stop ist platziert, aber noch nicht ausgelöst.
- `open`: Position ist aktiv.
- `closed`: Position ist geschlossen.

**Normativ:**

- Pending Orders dürfen **nicht** in derselben Candle auslösen, in der sie platziert wurden (Trigger erst ab `next_bar`).

---

## 4. Preconditions (harte Contracts)

### 4.1 Bid/Ask Alignment

- Im Candle-Mode MUSS pro `timestamp_ns` eine Bid- und Ask-Candle verfügbar sein (Aligned Series).
- Fehlt eine Seite, ist das ein **hard fail** (Run abbrechen).

### 4.2 Datenintegrität

- `timestamp_ns` ist strictly monotonic increasing und unique pro Serie.

---

## 5. Event-Loop & Reihenfolge (Candle-Mode)

Pro Candle-Step (pro `timestamp_ns`) gilt die Reihenfolge:

1. **Pending-Trigger prüfen** (Limit/Stop → open)
2. **Exit-Check** für alle `open` Positionen (SL/TP)
3. **Trade-/Position-Management** evaluieren (Rules → Actions; z.B. `timeout` Close)
4. **Portfolio/Equity** updaten (siehe Output-Contract `equity.csv`)

**Normativ:**

- Market-Orders werden zum Signal-Zeitpunkt ausgeführt.
- Pending Orders triggern erst ab der nächsten Candle (`bid.timestamp > pos.entry_time`).
- Entry und Exit können in derselben Candle passieren (Same-Bar), wenn `trigger_time == candle.timestamp`.

**Trade-Management (MVP-Contract):**

- Trade-Management läuft **nach** SL/TP-Checks, damit harte, marktgetriebene Exits (Stops) Priorität behalten.
- Stop/TP-Updates aus Trade-Management gelten (Determinismus) ab **next bar** (siehe `OMEGA_V2_TRADE_MANAGER_PLAN.md`).
- `timeout`-Exits werden als deterministische Close-Actions modelliert und in `trades.json.reason = "timeout"` geschrieben.

---

## 6. Entry-Modell

### 6.1 Market Entry (Candle-Mode)

**Fill-Preis (baseline):** `signal.entry_price`.

**Slippage:**

- Slippage MUSS angewendet werden.
- Slippage ist **adverse** für den Entry (long teurer, short günstiger).

**Sizing:**

- Sizing ist risk-basiert (Stop-Distanz und `risk_per_trade`).
- Volumen MUSS broker-konform quantisiert werden (siehe Abschnitt 10).

### 6.2 Pending Entry Trigger (Limit/Stop) – Candle-Mode

**Trigger-Checks (V1-Python-Parität):**

- Limit:
  - long: Trigger über **ASK.low <= entry_price**
  - short: Trigger über **BID.high >= entry_price**
- Stop:
  - long: Trigger über **ASK.high >= entry_price**
  - short: Trigger über **BID.low <= entry_price**

### 6.3 Fill-Preis bei Limit/Stop (Gap-aware)

Wenn eine Pending Order in Candle $t$ triggert, gilt:

- Ausgangspunkt ist `entry_price`.
- Zusätzlich wird ein **Gap-aware Fill** angewendet: Fill ist der **schlechtere** von
  - `entry_price`
  - und dem Candle-Open der relevanten Seite (Bid/Ask) in $t$.

**Normativ (Gap-aware Fill):**

- long Entries (Buy): `fill_price = max(entry_price, ask_open_t)`
- short Entries (Sell): `fill_price = min(entry_price, bid_open_t)`

### 6.4 Slippage auf Limit/Stop-Entries

- Slippage MUSS auch auf Limit/Stop-Entries angewendet werden.
- Slippage-Richtung ist beim Entry **nicht invertiert** (adverse Entry).

---

## 7. Exit-Modell

### 7.1 Candle-Mode SL/TP-Checks (inkl. pip_buffer)

`pip_buffer = pip_size * pip_buffer_factor` mit Default `pip_buffer_factor = 0.5`.

**Side-Auswahl:**

- long Exits prüfen auf **BID**
- short Exits prüfen auf **ASK** (Fallback auf BID ist im V2-Contract nicht zulässig; fehlende ASK ist hard fail)

**Hit-Definition (V1-Python-Parität):**

- long:
  - SL hit wenn `bid.low <= stop_loss + pip_buffer`
  - TP hit wenn `bid.high >= take_profit - pip_buffer`
- short:
  - SL hit wenn `ask.high >= stop_loss - pip_buffer`
  - TP hit wenn `ask.low <= take_profit + pip_buffer`

### 7.2 Tie-Break: SL vs. TP im selben Candle

- Wenn SL und TP im selben Candle hit sind, hat **SL Priorität**.

### 7.3 `in_entry_candle` Speziallogik

`in_entry_candle = (candle.timestamp_ns == trigger_time_ns)`.

Wenn `in_entry_candle`:

- SL hat Priorität.
- TP ist grundsätzlich erlaubt.
- **Spezialfall Limit-TP:** TP darf in Entry-Candle nur realisiert werden, wenn der Close “jenseits” des TP liegt:
  - long: `bid.close > take_profit`
  - short: `ask.close < take_profit`
  - sonst: kein Exit in dieser Candle.

### 7.4 Exit-Slippage (adverse, Richtung invertiert)

- Slippage MUSS auf **allen Exits** angewendet werden.
- Slippage-Richtung MUSS beim Exit **invertiert** werden (adverse Fill):
  - long Exit (Sell) nutzt Slippage-Direction `short`
  - short Exit (Buy-to-cover) nutzt Slippage-Direction `long`

---

## 8. Fees/Commission (einheitliches Kostenkonzept)

Omega V2 behandelt `FeeModel` und `CommissionModel` als **gleiches Konzept**: Transaktionskosten pro Order/Seite.

**Normativ:**

- Pro Entry/Exit wird **genau einmal** eine Gebühr gebucht.
- Die konkrete Implementierung (per_lot, per_million_notional, percent_of_notional, min_fee) ist Konfigurationsdetail (siehe `configs/execution_costs.yaml`).

**Abgrenzung Spread/Slippage vs. `fees_total` (normativ):**

| Kostenart | Wirkung | Erfassung in `fees_total` |
|-----------|---------|---------------------------|
| **Spread** | Implizit über Bid/Ask-Differenz im Fill-Preis | ❌ Nein (im Trade-PnL enthalten) |
| **Slippage** | Implizit über adverses Fill (Entry/Exit) | ❌ Nein (im Trade-PnL enthalten) |
| **Commission/Fee** | Explizite Gebühr pro Order/Seite | ✅ Ja (in `fees_total` summiert) |

**Wichtig:** Die Metrik `fees_total` (siehe `OMEGA_V2_METRICS_DEFINITION_PLAN.md`) enthält **nur explizite** Broker-Gebühren/Commission. Spread und Slippage wirken **implizit** über die Fill-Preise und sind damit bereits im Trade-PnL (`result`) enthalten.

---

## 9. Determinismus & RNG

Gemäß Config-Schema Plan:

- `run_mode = dev`: deterministisch reproduzierbar.
- `run_mode = prod`: stochastisch (OS-RNG), außer `rng_seed` ist explizit gesetzt.

**Normativ:**

- Slippage-Randomness MUSS aus `rng_seed` deterministisch ableitbar sein (z.B. `seed_i = rng_seed + trade_index`).
- Der Backtest-Core MUSS `trade_index` stabil definieren (z.B. fortlaufend pro Fill-Ereignis).

---

## 10. Sizing, Guardrails, Quantisierung

### 10.1 Min. SL-Distanz (Trade-Guard)

- Für **alle** Order-Typen (market/limit/stop) MUSS vor Sizing geprüft werden, ob die SL-Distanz ausreichend ist.

**Normativ:**

- `min_sl_distance_pips` ist symbol-spezifisch.
- Quelle: `configs/symbol_specs.yaml` (zusätzlicher optionaler Key pro Symbol).
- Fallback, falls nicht gesetzt: `min_sl_distance_pips_default = 0.1`.

Trade wird verworfen, wenn:

- `abs(entry_fill_price - stop_loss) < (min_sl_distance_pips * pip_size)`

### 10.2 Volumen-Quantisierung

- Volumen MUSS auf broker-konforme Steps **nach unten** quantisiert werden (Floor), damit das Risiko nicht überschritten wird.
- Min/Max/Step kommen aus `configs/symbol_specs.yaml` (`volume_min`, `volume_step`, `volume_max`).

---

## 11. Mapping auf Output-Contract (`trades.json`)

### 11.1 `reason`

- `take_profit`
- `stop_loss`
- `break_even_stop_loss` (nur als `reason`, wenn explizit gewünscht; ansonsten via `meta`)

### 11.2 Break-Even/Trailing Label (Meta)

Da `meta` im Output-Contract frei ist, SOLL der Execution-Layer zusätzliche Klassifizierungen dort ablegen:

- `meta.stop_loss_kind`: `initial | break_even | trailing`
- `meta.in_entry_candle`: `true|false`

---

## 12. Abweichungen zu V1 (explizit, normativ)

V2 weicht an folgenden Punkten **bewusst** von V1 ab:

1. **Gap-aware Fill bei Limit/Stop** (V1 füllt “ideal” zu entry_price).
2. **Slippage auch auf Limit/Stop-Entries**.
3. **Exit-Slippage mit invertierter Richtung** (adverse Exit-Fills; V1 nutzt `pos.direction`).

Diese Abweichungen sind Teil der universalen Wahrheit von V2 und müssen bei Paritäts-Tests separat bewertet werden (z.B. über Feature-Flags oder golden-file Varianten).

**Normativ (Hybrid-Parität):** Für V1-Python↔V2 Paritätsläufe in DEV/CI wird die V2-Engine im `execution_variant = "v1_parity"` betrieben. Kanonische V2-Golden/Determinismus-Läufe verwenden `execution_variant = "v2"`.

---

## 13. Validierung & Tests (MUSS für Implementierung)

- Unit-Tests für Trigger-/Fill-Funktionen (Bid/Ask-Seitenwahl, Gap-fill, pip_buffer, in_entry_candle).
- Golden-File Regression gegen V1 mit klarer Markierung der erwarteten Abweichungen (Abschnitt 12).
- Property-Tests: monotone Timestamps, keine NaNs, keine negativen Volumina, deterministische Slippage in dev.
