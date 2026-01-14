# Omega V2 – Trade Manager / Position Management Plan

> **Status**: Planungsphase  
> **Erstellt**: 14. Januar 2026  
> **Zweck**: Normative Spezifikation eines robusten, modularen Trade-/Position-Managers (institutionelles Trade Management) für Omega V2 mit **V1-Parität (MVP)** und klaren Erweiterungspunkten.

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Crate-Boundaries, Systemregeln, Einweg-Abhängigkeiten |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Workspace-/Crate-Struktur, Abhängigkeiten |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell, Reihenfolge im Event-Loop, Determinismus |
| [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) | Strategie-Parameter (MRZ), `use_position_manager`, Parität-Fallen |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Config-Schema + Defaults |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | `trades.json` Felder inkl. `reason` und `meta` |
| [OMEGA_V2_TESTING_VALIDATION_PLAN.md](OMEGA_V2_TESTING_VALIDATION_PLAN.md) | Golden/Parity, Determinismus-Gates |
| [OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md](OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md) | Logging/Tracing, Profiling, Determinismus-Guards |

---

## 1. Zielbild

Omega V2 bekommt eine dedizierte Trade-Management-Schicht, die **über** dem reinen SL/TP-Stop-Handling (Portfolio/Stops) liegt.

### 1.1 MVP-Ziel (V1-Parität)

- **MaxHoldingTime / Time-Stop** für offene Positionen.
- **Saubere, deterministische Close-Reasons** (mindestens: `timeout`).
- Keine stillen Semantik-Änderungen: **Determinismus** und **V1-Parität** sind vorrangig.

### 1.2 Post-MVP (institutionelle Upgrades – vorgemerkt)

Diese Features sind explizit vorgesehen, aber nicht Teil des MVP:

- Break-Even (BE) Moves (inkl. Buffer/Rules, niemals SL widening)
- Trailing Stop (ATR/Price Action/HTF/Volatility)
- Session-End / Market-Hours Close
- News Freeze / News Exit
- Partial Close / Scale-Out (wenn Modellierung/Costs entschieden)
- Kill-Switch / Circuit Breaker / Risk-Off
- Multi-Position Accounting (Netting/Hedging) – aktuell out-of-scope im Execution Model MVP

---

## 2. Architektur-Integration (kompatibel zu V2)

### 2.1 Schichten-Trennung (Separation of Concerns)

- `omega_portfolio` bleibt **Owner** der Positionen (Lifecycle: open/close/modify).
- `omega_execution` bleibt **Owner** der Fill-Semantik (inkl. Slippage/Fees, deterministisch).
- `omega_trade_mgmt` ist **Policy/Decision Layer**:
  - liest Snapshot-Views (Market/Positions/Context)
  - hält eigenen, strategy-spezifischen State
  - erzeugt **Actions/Intents**, die von Backtest/Live appliziert werden

Diese Trennung verhindert Cross-Cutting und bleibt kompatibel zu R1–R7 (Architekturregeln).

### 2.2 Event-Loop Position (Candle-Mode)

Der Trade Manager wird im Backtest-Loop an einer deterministischen Stelle aufgerufen:

1. Pending-Trigger prüfen (Limit/Stop → open)
2. Exit-Check für alle `open` Positionen (SL/TP) – Portfolio/Stops
3. **Trade Management**: Regeln evaluieren → Actions erzeugen
4. Portfolio/Equity updaten

**Wichtig:** Trade-Management darf keine Lookahead-Artefakte erzeugen.

### 2.3 Stop-Update-Policy (Entscheidung)

Für Candle-Mode gilt verbindlich:

- **Stop/TP-Änderungen werden erst ab der nächsten Bar aktiv** (`ApplyNextBar`).

Damit kann eine Regel nicht „im Nachhinein“ innerhalb derselben Candle einen Stop so verschieben, dass ein Stop-Out in derselben Candle retroaktiv entsteht.

**Ausnahme:** Ein expliziter Close-Intent (z.B. `timeout`) ist ein eigener Exit und wird zum definierten Exit-Preis der aktuellen Bar ausgeführt.

---

## 3. Crate-Design: `omega_trade_mgmt`

### 3.1 Ort im Workspace

Neues Crate im Rust-Workspace (gemäß Module Structure Plan):

- `rust_core/crates/trade_mgmt/`

### 3.2 Abhängigkeiten (Einweg-Abhängigkeiten)

**MVP:**
- `omega_types`

**Optional (Post-MVP, nur falls notwendig):**
- keine zwingenden weiteren Abhängigkeiten vorgesehen

**Begründung:** `trade_mgmt` soll ein reiner Decision-Layer bleiben und **keine** Portfolio-/Execution-Interna importieren.

### 3.3 Öffentliche API (Exports)

`omega_trade_mgmt` exportiert:

- `TradeManager`
- `TradeManagerConfig`
- `TradeContext`
- `PositionView` und `MarketView`
- `Rule` Trait + Standard-Regeln
- `Action`, `ActionKind`, `CloseReason`
- `RuleId`, `RulePriority`

---

## 4. Datenmodelle und Interfaces (normativ)

### 4.1 Read-only Views (keine Cross-Cutting Abhängigkeiten)

**PositionView** (Snapshot):

- `position_id: i64`
- `symbol: String`
- `direction: Direction` (`long|short` in Output; intern kann ein Enum genutzt werden)
- `status: PositionStatus` (`open|pending|closed`)
- `entry_time_ns: i64`
- `entry_price: f64`
- `stop_loss: Option<f64>`
- `take_profit: Option<f64>`
- `size: f64`
- `meta: serde_json::Value` oder `HashMap<String, String>` (nur serialisierbar; MVP kann minimal sein)

**MarketView** (Snapshot):

- `timestamp_ns: i64` (Open-Time der Candle)
- `bid_open/high/low/close: f64`
- `ask_open/high/low/close: f64`

**TradeContext**:

- `idx: usize`
- `market: MarketView`
- `session_open: bool`
- `news_blocked: bool`
- `mode: TradeMgmtMode` (MVP: `Candle`)

### 4.2 Actions (Intents)

Der Trade Manager gibt eine Liste von Actions zurück. Die Anwendung (Backtest Engine) ist verantwortlich für:

- deterministische Reihenfolge
- Konfliktauflösung
- Applikation an Portfolio/Execution

**Action (Enum):**

- `ClosePosition { position_id, reason, exit_price_hint, meta }`
- `ModifyStops { position_id, new_sl, new_tp, effective_from_idx }`
- `CancelPending { order_id, reason }` (Post-MVP)
- `ScaleOut { position_id, fraction_or_size, reason }` (Post-MVP)

**MVP:** ausschließlich `ClosePosition`.

### 4.3 CloseReason (Output-kompatibel)

Für `trades.json.reason` wird ein superset-kompatibles Enum definiert.

**MVP:**
- `stop_loss`
- `take_profit`
- `timeout`

**Post-MVP (vorgemerkt):**
- `breakeven`
- `trailing_stop`
- `session_end`
- `news`
- `manual`
- `pending_expired`

Wenn ein Reason nicht im MVP-Contract des Output-Plans als `reason` vorgesehen ist, MUSS er zumindest in `meta.labels` auftauchen. Für V1-Parität wird `timeout` als echtes `reason` benötigt.

---

## 5. Rule Engine

### 5.1 Rule Trait

Eine Regel ist ein deterministischer Transformer von `(Context, Positions, State)` zu `Actions`.

**Normativer Trait (Konzept):**

- `fn id(&self) -> RuleId`
- `fn priority(&self) -> RulePriority`
- `fn on_bar(&mut self, ctx: &TradeContext, positions: &[PositionView], state: &mut TradeStateStore) -> Vec<Action>`

### 5.2 Rule-Prioritäten (Konfliktauflösung)

Konflikte sind erwartbar (mehrere Regeln wollen schließen/ändern). Es gilt:

1. `ClosePosition` gewinnt immer gegen `ModifyStops`.
2. Bei mehreren `ClosePosition` für dieselbe Position gilt deterministisch:
   - niedrigste `priority` (höchste Wichtigkeit) gewinnt
   - bei Gleichstand: lexikografische `RuleId`
3. `ModifyStops` werden nur übernommen, wenn sie **monoton** sind:
   - SL darf nie „wider“ werden (kein SL widening)
   - TP darf optional angepasst werden (Policy per Config)
4. `ModifyStops` werden mit `effective_from_idx = ctx.idx + 1` geplant.

**Empfohlene Prioritäten (institutionell, superset):**

| Priority (0=höchste) | Regelklasse | Beispiele |
|---:|---|---|
| 0 | Hard Safety / Kill Switch | Risk-Off, Data invalid |
| 10 | Hard Close | MaxHoldingTime, SessionEnd |
| 20 | Protective Stops | Break-Even |
| 30 | Trailing | ATR trailing |
| 40 | Profit Mgmt | Scale-out |

**MVP:** Nur `MaxHoldingTimeRule` mit Priority 10.

---

## 6. MVP-Regel: MaxHoldingTimeRule (V1-Parität)

### 6.1 Semantik

- Gilt nur, wenn `use_position_manager=true`.
- Optionales Filtering nach `scenario` (wie V1: nur Szenario 3). Das Filtering ist ein Config-Detail.
- Eine Position wird geschlossen, wenn:

$$\Delta t = (ctx.market.timestamp\_ns - position.entry\_time\_ns) \ge max\_holding\_minutes \cdot 60 \cdot 10^9$$

### 6.2 Exit-Preis

Damit V1-Backtest-Parität möglich bleibt:

- Exit erfolgt zum **Bar-Close** der relevanten Seite:
  - `long`: `bid_close`
  - `short`: `ask_close`

CloseReason = `timeout`.

### 6.3 Determinismus

- Regel darf nur Informationen aus `ctx` und `PositionView` verwenden.
- Keine Systemzeit, keine Randomness.

---

## 7. Config Schema (Erweiterung, kompatibel)

### 7.1 Strategie-Parameter (bestehende Felder)

Bestehende MRZ-Parameter bleiben gültig:

- `use_position_manager: bool`
- `max_holding_minutes: int`

### 7.2 Neues, optionales Config-Objekt (Trade Management)

Ergänzend (und zukünftssicher) wird ein Block vorgeschlagen:

- `trade_management.enabled: bool` (Default: `false`)
- `trade_management.stop_update_policy: "apply_next_bar"` (Default: `apply_next_bar`)
- `trade_management.rules.max_holding_time.enabled: bool` (Default: `true`, wenn TM enabled)
- `trade_management.rules.max_holding_time.max_holding_minutes: int` (Default: aus Strategy-Params)
- `trade_management.rules.max_holding_time.only_scenarios: array<string>` (Default: `[]` = alle)

**Compat Mapping (normativ):**

- Wenn `use_position_manager=true`, dann MUSS `trade_management.enabled=true` angenommen werden.
- Wenn beide gesetzt sind und widersprüchlich, gewinnt `trade_management.enabled` (explizit > implizit).

---

## 8. Output Contract (Implikationen)

Für V1-Parität MUSS `trades.json.reason` mindestens die Werte `stop_loss`, `take_profit`, `timeout` zulassen.

Zusätzliche Labels (BE/Trailing/etc.) werden über `trades.json.meta` transportiert.

---

## 9. Tests (Pflicht für MVP)

Referenz: Testing/Validation Plan.

- Golden-Parity Test: V1 vs V2 (identische Inputs) ⇒ identische Trades (normalisiert) inkl. `reason="timeout"`.
- Unit Test: `MaxHoldingTimeRule` mit synthetischen Candles/Positions.
- Determinismus Test: mehrfacher Run ⇒ bit-identisches `trades.json`.

---

## 10. Live-Integration (Ausblick)

Auch wenn dieses Dokument primär den Backtest adressiert, ist die Architektur absichtlich so gewählt, dass derselbe Rule-Layer später live genutzt werden kann:

- Live-Runner baut `PositionView` aus Broker-Positions
- `Action`s werden in Broker-Calls übersetzt
- Resume bleibt über `magic_number`/Position-ID möglich

MVP-Implementierung muss jedoch keine Live-TM-Regeln umfassen.
