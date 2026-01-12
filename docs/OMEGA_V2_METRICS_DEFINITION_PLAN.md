# Omega V2 – Metrics Definition Plan

> **Status**: Planungsphase  
> **Erstellt**: 12. Januar 2026  
> **Zweck**: Normative Spezifikation aller Performance-Metriken und Score-Keys (Definitionen, Units, Domains, Edge-Cases, Rundung) für Omega V2  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, FFI-Grenze, Verantwortlichkeiten |
| [OMEGA_V2_DATA_FLOW_PLAN.md](OMEGA_V2_DATA_FLOW_PLAN.md) | Datenfluss, Timestamp-/Bar-Contract, Determinismus |
| [OMEGA_V2_EXECUTION_MODEL_PLAN.md](OMEGA_V2_EXECUTION_MODEL_PLAN.md) | Ausführungsmodell, Kosten-Semantik, Bid/Ask-Regeln |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Module/Crates, `metrics` Crate, Serialisierung |
| [OMEGA_V2_CONFIG_SCHEMA_PLAN.md](OMEGA_V2_CONFIG_SCHEMA_PLAN.md) | Normatives Config-Schema (Input, Defaults, Validierung) |
| [OMEGA_V2_OUTPUT_CONTRACT_PLAN.md](OMEGA_V2_OUTPUT_CONTRACT_PLAN.md) | Normativer Output-Contract (`metrics.json` Shape, Zeit/Units) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Packaging/Build-Matrix |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für:

- **Metrik-Keys** (Naming + Stabilität)
- **Units/Definitionen** (machine-readable in `metrics.json:definitions`)
- **Semantik/Formeln** pro Key (inkl. Edge-Cases)
- **Rundung/Normalisierung** für stabile Artefakte (Golden-Files)

**Abgrenzung:**

- Der **Output-Contract** definiert *Form* und *Pflichtartefakte* (`metrics.json` ist MUSS).  
- Dieses Dokument definiert *Inhalt* und *Bedeutung* der Keys.

**Wichtig:** Robustheits-/Stress-/Stabilitäts-Scores (V1 `rating/`) sind in V2 **first-class definiert**, werden aber gemäß Entscheidung **nicht** in der `metrics.json` des Base-Runs ausgegeben. Sie gehören in einen späteren Optimizer-/Rating-Aggregate-Contract.

---

## 2. Harte Leitplanken (Normativ)

### 2.1 `metrics.json` Shape

- Root ist ein JSON-Objekt.
- Enthält:
  - `metrics`: flaches Objekt (Key → Value)
  - `definitions`: Objekt (Key → Definition)
- `metrics` Values sind nur: `number | boolean | string`.
- **Keine Arrays** in `metrics` (Aggregator-/Optimizer-freundlich).

### 2.2 Naming-Policy

- Keys sind **`snake_case`**.
- Scores tragen Suffix **`_score`**.
- p-Values tragen Präfix **`p_value_`**.

### 2.3 Domain-Policy

- Alle Scores werden hart auf $[0,1]$ geclamped.
- Drawdown-Ratio ist $[0,1]$.

### 2.4 Rundung beim Schreiben (Artefaktstabilität)

Beim Serialisieren von `metrics.json` gilt:

- `account_currency`: **2** Dezimalstellen
- Ratios (z.B. `win_rate`, `max_drawdown`): **6** Dezimalstellen
- Scores (z.B. `*_score`): **6** Dezimalstellen

Die Rundung ist Teil des Contracts (Golden-File Stabilität).

---

## 3. `definitions`-Format (Normativ)

### 3.1 Felder

Für jeden Key `k` in `metrics` MUSS ein Eintrag `definitions[k]` existieren.

Normatives Minimum (V2):

- `unit`: string
- `description`: string
- `domain`: string (Range/Constraints, z.B. `0..1`, `>=0`)
- `source`: string (`trades|equity|meta|optimizer`)
- `type`: string (z.B. `number`, `boolean`, `string`, `number|string`)

Beispiel:

```json
{
  "metrics": {
    "win_rate": 0.52
  },
  "definitions": {
    "win_rate": {
      "unit": "ratio",
      "description": "Anteil der Gewinntrades an allen Trades.",
      "domain": "0..1",
      "source": "trades",
      "type": "number"
    }
  }
}
```

### 3.2 Units-Vokabular (MVP)

- `count`
- `ratio`
- `account_currency`
- `bars`
- `r_multiple`
- `days`
- `trades`

---

## 4. Datenquellen pro Metrik (Normativ)

### 4.1 Primäre Inputs

- `trades.json`: abgeschlossene Trades (Entry/Exit) inkl. `result` und optional `r_multiple` (siehe Output-Contract).
- `equity.csv`: Equity pro Bar (für Drawdown- und Equity-basierte Metriken).
- `meta.json`: Run-Infos (z.B. `account_currency`).

### 4.2 Single-Run vs. Optimizer-Aggregate

- **Single-Run-Metriken**: werden aus `trades.json` und `equity.csv` berechnet und in `metrics.json` geschrieben.
- **Optimizer/Rating-Metriken** (Robustness/Stress/Confidence): werden aus *mehreren Runs* abgeleitet und sind daher `source: optimizer`.

---

## 5. Metrik-Katalog (V2)

### 5.1 MVP-Pflichtmetriken (MUSS)

Diese Keys sind das V2-MVP-Minimum und entsprechen dem Output-Contract.

| Key | Unit | Type | Domain | Source | Kurzdefinition |
|-----|------|------|--------|--------|----------------|
| `total_trades` | `count` | `number` | `>=0` | `trades` | Anzahl abgeschlossener Trades |
| `wins` | `count` | `number` | `>=0` | `trades` | Anzahl Trades mit Gewinn (PnL>0) |
| `losses` | `count` | `number` | `>=0` | `trades` | Anzahl Trades mit Verlust (PnL<0) |
| `win_rate` | `ratio` | `number` | `0..1` | `trades` | `wins / total_trades` |
| `profit_gross` | `account_currency` | `number` | `any` | `trades` | Summe PnL **vor** Fees |
| `fees_total` | `account_currency` | `number` | `>=0` | `trades` | Summe expliziter Gebühren/Commission |
| `profit_net` | `account_currency` | `number` | `any` | `trades` | `profit_gross - fees_total` |
| `max_drawdown` | `ratio` | `number` | `0..1` | `equity` | Max. relativer Drawdown |
| `max_drawdown_abs` | `account_currency` | `number` | `>=0` | `equity` | Max. absoluter Drawdown |
| `max_drawdown_duration_bars` | `bars` | `number` | `>=0` | `equity` | Dauer des Max-Drawdowns in Bars (Primary TF) |
| `avg_r_multiple` | `r_multiple` | `number` | `any` | `trades` | Durchschnittliches $R$ pro Trade |
| `profit_factor` | `ratio` | `number` | `>=0` | `trades` | `gross_profit / gross_loss_abs` |

### 5.2 MVP+ (SOLL, sofort verfügbar)

Diese Keys sind bewusst günstig zu berechnen und werden im MVP+ direkt ausgegeben.

| Key | Unit | Type | Domain | Source | Kurzdefinition |
|-----|------|------|--------|--------|----------------|
| `avg_trade_pnl` | `account_currency` | `number` | `any` | `trades` | `profit_net / total_trades` (falls `total_trades>0`) |
| `expectancy` | `r_multiple` | `number` | `any` | `trades` | Erwartungswert pro Trade in $R$ (im MVP identisch zu `avg_r_multiple`) |
| `active_days` | `days` | `number` | `>=0` | `trades` | Anzahl UTC-Tage mit mindestens einem Trade-Entry |
| `trades_per_day` | `trades` | `number` | `>=0` | `trades` | `total_trades / active_days` (falls `active_days>0`) |

---

## 6. Detaildefinitionen & Edge-Cases (Normativ)

### 6.1 Zählmetriken (`total_trades`, `wins`, `losses`, `win_rate`)

- `total_trades`: Länge der Trade-Liste.
- `wins`: Anzahl Trades mit `trades.json.result > 0`.
- `losses`: Anzahl Trades mit `trades.json.result < 0`.
- `win_rate`:
  - Wenn `total_trades == 0`: `win_rate = 0.0`.

### 6.2 Profit/Fees (`profit_gross`, `fees_total`, `profit_net`, `avg_trade_pnl`)

**Wichtige Semantik:**

- Spread/Slippage wirken im Execution Model über Bid/Ask und Fill-Preis und sind damit **implizit** im Trade-PnL enthalten.
- `fees_total` sind **explizite** Gebühren/Commission.

Definitionen:

- `profit_gross`: Summe aller `trades.json.result` (PnL vor expliziten Fees).
- `fees_total`: Summe expliziter Gebühren/Commission (>=0), die durch das Kostenmodell im Run angefallen sind.
- `profit_net = profit_gross - fees_total`.
- `avg_trade_pnl`:
  - Wenn `total_trades == 0`: `avg_trade_pnl = 0.0`.
  - Sonst: `profit_net / total_trades`.

### 6.3 Drawdown (`max_drawdown`, `max_drawdown_abs`, `max_drawdown_duration_bars`)

- Equity-Serie ist `equity.csv:equity` (siehe Output-Contract).
- High-Water-Mark $H(t) = \max_{\tau \le t} equity(\tau)$.

Relativer Drawdown:

$$
DD_{rel}(t) = \begin{cases}
0, & H(t) = 0 \\
\frac{H(t) - equity(t)}{H(t)}, & \text{sonst}
\end{cases}
$$

- `max_drawdown = max_t DD_rel(t)`.
- `max_drawdown_abs = max_t (H(t) - equity(t))`.

Dauer:

- `max_drawdown_duration_bars`: Länge (in Bars auf Primary TF) zwischen High-Water-Moment und dem Zeitpunkt der vollständigen Recovery.
- Wenn keine Recovery stattfindet: Dauer bis Run-Ende.

### 6.4 R-Multiples (`avg_r_multiple`, `expectancy`)

R-Definition (normativ):

$$
R = \frac{trade\_pnl\_net}{risk\_per\_trade}
$$

- `risk_per_trade` stammt aus der V2-Config (`account.risk_per_trade`).
- `trade_pnl_net` ist die pro Trade exportierte PnL aus `trades.json.result` (netto bzgl. Bid/Ask/Slippage, **vor** expliziten Fees).
- `avg_r_multiple` ist der Durchschnitt über alle Trades (0 bei `total_trades==0`).
- `expectancy` ist im MVP ein Kompatibilitäts-Key und entspricht `avg_r_multiple`.

### 6.5 Profit Factor (`profit_factor`)

- `gross_profit`: Summe aller positiven Trade-PnL (vor Fees).
- `gross_loss_abs`: Betrag der Summe aller negativen Trade-PnL (vor Fees).

$$
profit\_factor = \begin{cases}
0, & gross\_loss\_abs = 0 \\
\frac{gross\_profit}{gross\_loss\_abs}, & \text{sonst}
\end{cases}
$$

### 6.6 Aktivität (`active_days`, `trades_per_day`)

- `active_days`: Anzahl unterschiedlicher UTC-Dates (YYYY-MM-DD) aus `entry_time` der Trades.
- `trades_per_day`:
  - Wenn `active_days == 0`: `0.0`.
  - Sonst: `total_trades / active_days`.

---

## 7. Erweiterungen (Phase 2, definiert aber nicht MVP)

### 7.1 Sharpe/Sortino (zwei Varianten)

V2 definiert beide Varianten als separate Keys:

- Trade-basiert (R-Multiples):
  - `sharpe_trade_r`
  - `sortino_trade_r`
- Equity-basiert (Daily):
  - `sharpe_equity_daily`
  - `sortino_equity_daily`

**Risk-free Rate:** im MVP nicht unterstützt (implizit 0; kein Config-Feld).

**Insufficient-Samples Policy (normativ):**

- Wenn die Berechnungsgrundlage weniger als 2 Samples hat, wird der Wert als String geschrieben:
  - `"n/a"`

Damit sind diese Keys effektiv `type: "number|string"`.

---

## 8. Robustness/Stress/Confidence (Optimizer-Layer, V1 `rating/`)

### 8.1 Status & Output-Ort

Diese Metriken sind **definiert** (Keys + Semantik), werden aber gemäß Entscheidung **nicht** in `metrics.json` des Base-Runs ausgegeben.

- `source`: `optimizer`
- Erwarteter Zielort: Optimizer-/Walkforward-Aggregate (separater Contract, nicht MVP).

### 8.2 Definierte Score-Keys (V2)

Alle Scores werden auf $[0,1]$ geclamped (`domain: 0..1`, `unit: ratio`).

| Key | Domain | Kurzdefinition | V1-Referenz |
|-----|--------|----------------|-------------|
| `robustness_1_score` | 0..1 | Parameter-Jitter Robustheit (Penalty-Modell) | `src/old/backtest_engine/rating/robustness_score_1.py` |
| `data_jitter_score` | 0..1 | ATR-skaliertes OHLC-Jitter Robustheit | `src/old/backtest_engine/rating/data_jitter_score.py` |
| `cost_shock_score` | 0..1 | Kosten-Schock Robustheit (Faktoren 1.25/1.50/2.00) | `src/old/backtest_engine/rating/cost_shock_score.py` |
| `timing_jitter_score` | 0..1 | Timing-Jitter Robustheit (Backshift in Monaten) | `src/old/backtest_engine/rating/timing_jitter_score.py` |
| `trade_dropout_score` | 0..1 | Robustheit ggü. zufälligem Trade-Entfall | `src/old/backtest_engine/rating/trade_dropout_score.py` |
| `tp_sl_stress_score` | 0..1 | TP/SL-Exit Robustheit ggü. Candle-Extremes | `src/old/backtest_engine/rating/tp_sl_stress_score.py` |
| `stability_score` | 0..1 | Stabilität über Subperioden (Yearly Re-runs) | `src/old/backtest_engine/rating/stability_score.py` |
| `ulcer_index` | >=0 | Ulcer Index (weekly, drawdown in %) | `src/old/backtest_engine/rating/ulcer_index_score.py` |
| `ulcer_index_score` | 0..1 | `1 - ulcer_index/ulcer_cap` | `src/old/backtest_engine/rating/ulcer_index_score.py` |

### 8.3 p-Values (V2)

- `p_value_mean_r_gt_0` (bootstrap p-value für Hypothese $E[R] > 0$)
- `p_value_net_profit_gt_0` (bootstrap p-value für Hypothese $E[profit] > 0$)

Domain: `0..1`, Unit: `ratio`, Source: `optimizer`.

V1-Referenz:

- `src/old/backtest_engine/rating/p_values.py`

---

## 9. Konfigurierbarkeit (V2-Config: `metrics.*`)

Damit Optimizer/Rating die robusten Metriken deterministisch und reproduzierbar orchestrieren kann, werden Parameter in der V2-Config geführt (siehe Config-Schema-Plan).

Normative Defaults (V1-kompatibel):

- `metrics.robustness.enabled`: `false`
- `metrics.robustness.mode`: `"full"`
- `metrics.robustness.jitter_frac`: `0.05`
- `metrics.robustness.jitter_repeats`: `5`
- `metrics.robustness.dropout_frac`: `0.10`
- `metrics.robustness.dropout_runs`: `1`
- `metrics.robustness.cost_shock_factors`: `[1.25, 1.50, 2.00]`
- `metrics.data_jitter.atr_period`: `14`
- `metrics.data_jitter.sigma_atr`: `0.10`
- `metrics.data_jitter.repeats`: `5`
- `metrics.data_jitter.penalty_cap`: `0.5`
- `metrics.data_jitter.min_price`: `1e-9`
- `metrics.data_jitter.fraq`: `0.0`
- `metrics.ulcer.ulcer_cap`: `10.0`
- `metrics.timing_jitter.divisors`: `[10, 5, 20]`
- `metrics.timing_jitter.min_months`: `1`

---

## 10. Implementierungs-Mapping (Rust, File-per-Metric)

### 10.1 `metrics` Crate

Das Rust `metrics`-Crate folgt dem Indikator-Pattern: **eine Datei pro Metrik**.

- `compute.rs`: ruft alle Metrik-Funktionen auf und erzeugt ein strukturiertes `Metrics`-Objekt.
- Pro Key eine Datei, z.B.:
  - `total_trades.rs`
  - `win_rate.rs`
  - `profit.rs` (oder getrennt `profit_gross.rs`, `profit_net.rs`, `fees_total.rs`)
  - `drawdown.rs`
  - `avg_r_multiple.rs`
  - `profit_factor.rs`
  - `activity.rs` (oder `active_days.rs`, `trades_per_day.rs`)

### 10.2 Tests

- Unit-Tests pro Metrik (Edge-Cases: 0 Trades, 0 Losses, konstante Equity).
- Golden-File Tests (siehe Output-Contract): Rundung + deterministische Seeds.

---

## 11. V1-Mapping (Referenz)

- V1 Core-Stats: `src/old/backtest_engine/report/metrics.py`
- V1 Robustness/Rating: `src/old/backtest_engine/rating/*`

V2 übernimmt die robusten Metriken inhaltlich, trennt aber strikt:

- Single-Run `metrics.json` (vergleichbare Kernmetriken)
- Optimizer-/Rating-Aggregate (robuste Scores)
