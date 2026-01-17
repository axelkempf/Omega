---
title: "OMEGA V2 W6 Fix Prompt – Metrics, FFI, Determinismus"
status: "Proposed"
date: "2026-01-17"
deciders:
  - "Axel Kempf"
consulted:
  - "OMEGA_V2_METRICS_DEFINITION_PLAN.md"
  - "OMEGA_V2_OUTPUT_CONTRACT_PLAN.md"
  - "OMEGA_V2_TECH_STACK_PLAN.md"
---

## Ziel

Korrigiere die Abweichungen in Wave 6 (Metrics + FFI), ergänze fehlende Tests und stelle deterministische, plan-konforme Output‑Artefakte sicher. Zusätzliche Metrik‑Keys bleiben erhalten, müssen aber korrekt berechnet und dokumentiert werden. Sharpe/Sortino müssen an die V2‑Plan‑Namen und -Semantik angepasst werden.

## Kontext

- Relevante Implementierung: `rust_core/crates/metrics/*`, `rust_core/crates/ffi/*`, `rust_core/crates/backtest/src/result_builder.rs`, `rust_core/crates/types/src/result.rs`
- Relevante Pläne: `OMEGA_V2_METRICS_DEFINITION_PLAN.md`, `OMEGA_V2_OUTPUT_CONTRACT_PLAN.md`, `OMEGA_V2_TECH_STACK_PLAN.md`
- Vorgaben: Rust 2024, kein `panic!` über FFI, Single FFI Boundary, Output‑Contract für `metrics.json`

## Scope

### In Scope

1. **Tests ergänzen** (Metriken, Rundung, Drawdown‑Edge‑Cases, FFI‑Error‑Contract).
2. **Metrik‑Berechnungen vervollständigen** für alle vorhandenen Keys.
3. **Sharpe/Sortino** an V2‑Plan anpassen (Key‑Namen + Semantik + `"n/a"`‑Policy).
4. **Deterministische JSON‑Ausgabe** für `definitions` sicherstellen.
5. **Fehlerkategorien** granular nach Plan mappen (statt nur `config|runtime`).
6. **Plan‑Update**: „eine Datei pro Metrik“ als optional/locker dokumentieren.
7. **CI‑Workflow anpassen**: `paths-ignore` für Doku‑Änderungen.
8. **Wheel‑Build verifizieren**: manuelle Prüfung dokumentieren.

### Out of Scope

- Neue Metriken außerhalb der bereits vorhandenen Keys.
- Änderungen am Execution‑Model oder Strategy‑Logik.
- Python‑Wrapper (Wave 7) – nur FFI/Result‑Contract.

## Aufgabenpakete (detailliert)

### 1) Tests ergänzen

**Ort:** `rust_core/crates/metrics/` und `rust_core/crates/ffi/`.

**Pflichttests (min.):**
- `compute_metrics`:
  - 0 Trades → `win_rate=0`, `avg_trade_pnl=0`, `profit_factor=0`.
  - Nur Gewinne / nur Verluste → `profit_factor` korrekt, `avg_win/avg_loss` korrekt.
- `compute_drawdown`:
  - Konstante Equity → Drawdown 0, Duration 0.
  - Drawdown ohne Recovery → Duration bis Run‑Ende.
- `round_metrics`:
  - Currency‑Keys auf 2dp, Ratio‑Keys auf 6dp (exakt, keine Toleranz).
- FFI Error‑Contract:
  - Config‑Fehler → Python Exception.
  - Runtime‑Fehler → JSON mit `ok:false` und **korrekter Kategorie**.

### 2) Zusätzliche Keys korrekt berechnen

Aktuell vorhandene Keys **müssen berechnet** werden, nicht nur `0.0` bleiben:
- `avg_win`, `avg_loss`, `largest_win`, `largest_loss` basierend auf Trades.
- `calmar_ratio` korrekt definieren (z. B. annualisierte Rendite / max_drawdown, mit sauberen Edge‑Cases).

**Hinweis:** Falls die Berechnung einen stabilen Zeitbezug benötigt, nutze `equity_curve` und die verfügbaren Zeitstempel. Bei fehlender Basis → `n/a` als String und `definitions[*].type = number|string`.

### 3) Sharpe/Sortino an Plan anpassen

**Plan‑Konformität:**
- Keys: `sharpe_trade_r`, `sortino_trade_r`, `sharpe_equity_daily`, `sortino_equity_daily`.
- Trade‑basiert: R‑Multiples pro Trade, keine Annualisierung.
- Equity‑basiert: Daily Returns, Annualisierung mit $\sqrt{252}$.
- <2 Samples → `n/a` (String, `number|string`).

**Legacy‑Keys:**
- `sharpe_ratio` und `sortino_ratio` entweder entfernen **oder** als deprecated behalten, aber korrekt berechnen und in `definitions` markieren.

### 4) Determinismus der JSON‑Ausgabe

**Ziel:** stabile Key‑Order für `definitions` (Golden‑Files).

**Umsetzung:**
- `HashMap` → `BTreeMap` in `MetricDefinitions` und `BacktestResult.metric_definitions`.
- Alternativ: kanonische Serialisierung vor `serde_json::to_string`.

### 5) Fehlerkategorien granularer mappen

**Minimal‑Mapping:**
- `Data` → `market_data`
- `Execution`, `Portfolio`, `TradeManagement` → `execution`
- `Strategy` → `strategy`
- alles übrige → `runtime`

**Ort:** `rust_core/crates/backtest/src/error.rs` in `impl From<BacktestError> for ErrorResult`.

### 6) Plan‑Update (Dokumentation)

**Ziel:** Abweichung bei „eine Datei pro Metrik“ formal akzeptieren.

**Update:** `OMEGA_V2_METRICS_DEFINITION_PLAN.md` Abschnitt 10.1:
- Gruppierte Dateien (z. B. `trade_metrics.rs`, `equity_metrics.rs`) sind erlaubt.
- Begründung: bessere Kohäsion und Lesbarkeit.

### 7) CI‑Workflow und Wheel‑Verifikation

- **paths-ignore ergänzen:** `.github/workflows/omega-v2-ci.yml` um `docs/**` und `**/*.md` erweitern, um Doku‑Only Änderungen zu überspringen.
- **Wheel‑Build manuell verifizieren:** lokale Prüfung dokumentieren (z. B. `maturin build --release`, dann Wheel‑Artefakte prüfen und Import‑Smoke‑Test). Ergebnis als manuelle Checkliste festhalten.

## Dateien / Touchpoints

- `rust_core/crates/metrics/src/compute.rs`
- `rust_core/crates/metrics/src/definitions.rs`
- `rust_core/crates/metrics/src/output.rs`
- `rust_core/crates/metrics/src/trade_metrics.rs`
- `rust_core/crates/metrics/src/equity_metrics.rs`
- `rust_core/crates/types/src/result.rs`
- `rust_core/crates/backtest/src/result_builder.rs`
- `rust_core/crates/backtest/src/error.rs`
- `rust_core/crates/ffi/src/lib.rs`
- `.github/workflows/omega-v2-ci.yml`
- `docs/OMEGA_V2_METRICS_DEFINITION_PLAN.md`

## Akzeptanzkriterien

- Tests grün (`cargo test`) inkl. Metriken, Rundung, Drawdown und FFI‑Contract.
- `metrics.json` enthält korrekte Werte **und** Definitions für alle Keys.
- Sharpe/Sortino folgen dem Plan (Keys + Semantik + `n/a`).
- JSON‑Output deterministisch (stabile Key‑Order).
- Error‑Kategorien entsprechen Plan (`market_data|execution|strategy|runtime`).
- CI‑Workflow überspringt Doku‑Only Änderungen via `paths-ignore`.
- Wheel‑Build manuell verifiziert und dokumentiert.

## Prompt für offene Punkte (Stand nach Review)

Bitte setze die **noch offenen Punkte** aus dem W6‑Fix um. Fokus nur auf die unten genannten Lücken, ohne zusätzliche Refactors oder neue Features.

### Offene Aufgaben

1. **`n/a`‑Policy für Sharpe/Sortino umsetzen**
  - Wenn <2 Samples: Wert als String `"n/a"` ausgeben.
  - `definitions[*].value_type` auf `"number|string"` setzen für:
    - `sharpe_trade_r`, `sortino_trade_r`, `sharpe_equity_daily`, `sortino_equity_daily`.
  - Output‑Contract strikt einhalten, keine neuen Dependencies.

2. **FFI‑Error‑Contract Tests ergänzen**
  - Config‑Fehler → Python Exception.
  - Runtime‑Fehler → JSON mit `ok:false` und korrekter Kategorie (`market_data|execution|strategy|runtime`).
  - Tests deterministisch, keine Zeit-/Netz‑Abhängigkeiten.

3. **CI‑Workflow anpassen**
  - `.github/workflows/omega-v2-ci.yml` um `paths-ignore` erweitern:
    - `docs/**`
    - `**/*.md`
  - Ziel: Docs‑only Änderungen sollen den Workflow nicht triggern.

4. **Wheel‑Build Verifikation dokumentieren**
  - Manuelle Checkliste dokumentieren (z. B. `maturin build --release`, Artefakte prüfen, Import‑Smoke‑Test).
  - Ergebnis in einer passenden Doku‑Sektion festhalten.

## Hinweise & Constraints

- Keine neuen externen Dependencies, außer zwingend nötig.
- Kein `panic!` über FFI.
- Output‑Contract (`metrics` + `definitions`) strikt einhalten.
- Tests deterministisch (keine Zeit‑/Netz‑Abhängigkeiten).