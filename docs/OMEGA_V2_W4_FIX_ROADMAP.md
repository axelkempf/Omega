---
title: "OMEGA V2 – W4 Fix Roadmap"
status: Proposed
date: 2026-01-15
deciders:
  - Axel Kempf
consulted:
  - GitHub Copilot
---

## Überblick

Diese Roadmap beschreibt die Fix‑Wellen für die W4‑Implementierung
(Strategy + Trade Management), basierend auf den V2‑Plänen. Die Wellen
folgen dem Stil der V2 Implementation Plan Prompts (Codex‑Max‑Prompt).

Annahmen:

- W0–W3 sind implementiert.
- W5–W7 sind noch nicht implementiert.

Ziel:

- V1‑Parity‑kritische Abweichungen schließen.
- W4‑Akzeptanzkriterien erfüllen.
- Standards (Rust, Testing, FFI‑Boundary) einhalten.

## Fix‑Wellen‑Übersicht

| Fix‑Welle | Fokus | Priorität | Abhängigkeiten |
|---|---|---|---|
| **FX4‑A** | Strategy Parity (Szenarien 3–6) | Parity‑kritisch | W0–W3 |
| **FX4‑B** | Trade‑Mgmt API Alignment | Parity‑kritisch | FX4‑A |
| **FX4‑C** | Tests + Parity Gates | Parity‑kritisch | FX4‑A, FX4‑B |
| **FX4‑D** | Post‑MVP Regeln (BE/Trailing) | Später | FX4‑C |

---

## FX4‑A: Strategy Parity (Szenarien 3–6)

### Codex‑Max Prompt

```markdown
# Task: Omega V2 W4 Fix – Strategy Parity (FX4‑A)

## Kontext
Die W4‑Implementierung weicht in Szenario 3–6 vom Strategies‑Plan ab.
Wave 0–3 sind implementiert, Wave 5–7 noch nicht.

## Ziel
Bringe die MRZ‑Strategie auf Plan‑Konformität:
- Szenario 3–6 gemäß OMEGA_V2_STRATEGIES_PLAN.md
- Indikator‑Requirements vollständig
- Multi‑TF Overlay korrekt
- Vol‑Cluster Guard gemäß Spezifikation

## Betroffene Dateien
- rust_core/crates/strategy/src/impl_/mean_reversion_z_score.rs
- rust_core/crates/strategy/src/context.rs
- rust_core/crates/strategy/src/traits.rs (falls nötig)

## Anforderungen
1. Szenario 3: Kalman‑Z + Bollinger Entry, TP=EMA mit tp_min_distance
2. Szenario 4: Kalman+GARCH‑Z + Bollinger Entry, TP=BB‑Mid
3. Szenario 5: Szenario‑2 Basis + Vol‑Cluster Guard (GARCH‑Forecast)
4. Szenario 6: Multi‑TF Overlay (Kalman‑Z + Bollinger pro TF)
5. required_indicators() ergänzt:
   - Kalman+GARCH‑Z
   - Vol‑Cluster (garch_forecast)
   - Scenario‑6 TF‑Indikatoren
6. required_htf_timeframes() ergänzt:
   - htf_tf + scenario6_timeframes

## Akzeptanzkriterien
- Szenario‑Logik entspricht Plan exakt
- Keine Lookahead‑Artefakte
- Strategy‑Output deterministisch
- Tags/Meta enthalten scenario_id und relevante Marker

## Referenzen
- OMEGA_V2_STRATEGIES_PLAN.md
- OMEGA_V2_INDICATOR_CACHE__PLAN.md
- OMEGA_V2_EXECUTION_MODEL_PLAN.md
```

---

## FX4‑B: Trade‑Mgmt API Alignment

### Codex‑Max Prompt

```markdown
# Task: Omega V2 W4 Fix – Trade Management API (FX4‑B)

## Kontext
Das aktuelle trade_mgmt nutzt Position direkt und hängt von meta‑Feldern
ab. Das widerspricht dem Trade‑Manager‑Plan.

## Ziel
Passe die Trade‑Mgmt‑API an die V2‑Spezifikation an:
- Read‑only Views (PositionView, MarketView, TradeContext)
- Action‑Shape mit exit_price_hint und effective_from_idx
- Deterministische Prioritäten (MVP‑Scope: MaxHoldingTime)

## Betroffene Dateien
- rust_core/crates/trade_mgmt/src/engine.rs
- rust_core/crates/trade_mgmt/src/rules.rs
- rust_core/crates/trade_mgmt/src/actions.rs
- rust_core/crates/trade_mgmt/src/lib.rs

## Anforderungen
1. Neue Context‑Typen (PositionView, MarketView, TradeContext)
2. Action‑Enum erweitern:
   - ClosePosition { position_id, reason, exit_price_hint, meta }
   - ModifyStopLoss { effective_from_idx }
3. MaxHoldingTimeRule nutzt ctx.market.timestamp_ns und Close‑Preis
   (long: bid_close, short: ask_close)
4. BreakEven/Trailing als Post‑MVP markieren oder Feature‑Flag
5. Stop‑Update‑Policy: ApplyNextBar (effective_from_idx = idx + 1)

## Akzeptanzkriterien
- MVP‑Scope bleibt MaxHoldingTime
- Keine Abhängigkeit von position.meta["current_price"]
- Deterministische Actions

## Referenzen
- OMEGA_V2_TRADE_MANAGER_PLAN.md
- OMEGA_V2_EXECUTION_MODEL_PLAN.md
```

---

## FX4‑C: Tests + Parity Gates

### Codex‑Max Prompt

```markdown
# Task: Omega V2 W4 Fix – Tests & Parity (FX4‑C)

## Kontext
W4‑Akzeptanzkriterien verlangen Szenario‑Tests und Parity‑Gates.
Diese fehlen aktuell.

## Ziel
Ergänze Tests für MRZ‑Szenarien und Trade‑Mgmt‑MVP:
- Unit‑Tests pro Szenario 1–6
- Session/News Gates
- MaxHoldingTime Rule Tests
- Vorbereitung für Golden Parity Tests (W7)

## Betroffene Dateien
- rust_core/crates/strategy/src/impl_/mean_reversion_z_score.rs
- rust_core/crates/strategy/tests/* (neu)
- rust_core/crates/trade_mgmt/src/rules.rs

## Anforderungen
1. Szenario‑Tests: positive & negative Fälle
2. Guards: session_open/news_blocked blocken
3. Trade‑Mgmt: Timeout‑Exit korrekt und deterministisch
4. Tests deterministisch (keine Randomness)

## Akzeptanzkriterien
- Alle Szenario‑Tests grün
- Trade‑Mgmt Tests grün
- Kein Lookahead‑Bias

## Referenzen
- OMEGA_V2_TESTING_VALIDATION_PLAN.md
- OMEGA_V2_STRATEGIES_PLAN.md
```

---

## FX4‑D: Post‑MVP Regeln (Break‑Even, Trailing)

### Codex‑Max Prompt

```markdown
# Task: Omega V2 W4 Fix – Post‑MVP Trade Rules (FX4‑D)

## Kontext
Break‑Even und Trailing sind im Plan Post‑MVP. Sie sind optional,
aber müssen sauber isoliert werden.

## Ziel
Isoliere Post‑MVP Regeln hinter Feature‑Flags oder Config‑Gate.

## Betroffene Dateien
- rust_core/crates/trade_mgmt/src/rules.rs
- rust_core/crates/trade_mgmt/src/lib.rs

## Anforderungen
1. BreakEven/Trailing nur aktiv, wenn explizit konfiguriert
2. Regeln nutzen ausschließlich TradeContext/MarketView
3. Keine Abhängigkeit von position.meta

## Akzeptanzkriterien
- Post‑MVP Rules deaktiviert per Default
- Deterministische Berechnungen

## Referenzen
- OMEGA_V2_TRADE_MANAGER_PLAN.md
```

---

## Abschlusskriterien

- FX4‑A bis FX4‑C sind abgeschlossen und grün in CI
- W4‑Akzeptanzkriterien aus OMEGA_V2_IMPLEMENTATION_WAVES.md erfüllt
- Parity‑kritische Abweichungen geschlossen
