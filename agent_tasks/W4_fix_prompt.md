## Prompt für den KI‑Agenten

Du bist ein Implementierungs‑Agent für das Omega‑Repo (Branch: `architektur/umsetzung-v2-plan`).  
Aufgabe: Behebe **W4/V2 Abweichungen 1,2,3,7** sowie **beide Standard‑Check‑Abweichungen**.  
**Nicht anfassen**: Abweichung 4 & 5 (bewusst ignorieren), Abweichung 6 (Open/Close‑Timestamps) bleibt unverändert. Long/Short‑Verhalten wird separat behandelt.

### Kontext / Scope
- Ziel: Konformität mit `OMEGA_V2_STRATEGIES_PLAN.md` und `OMEGA_V2_TRADE_MANAGER_PLAN.md`.
- Du arbeitest nur an Wave‑4 Code (`rust_core/crates/strategy`, `rust_core/crates/trade_mgmt`) und ergänzenden Tests.
- Halte FFI‑/Architektur‑Grenzen ein (keine Cross‑Cutting‑Deps).

### Zu erledigen (W4/V2 Abweichungen)
1) **MrzParams ergänzen**  
   - In `mean_reversion_z_score.rs`: `use_position_manager: bool` und `max_holding_minutes: int` (Defaults gemäß Plan).  
   - Parameter müssen via `from_params` akzeptiert werden und in `MrzParams` sichtbar sein (für Config‑Mapping).

2) **Trade‑Management‑Plan‑Gaps schließen**  
   - Implementiere `RuleId`/`RulePriority` und deterministische Konfliktauflösung gemäß Plan.  
   - Füge `TradeManagerConfig` hinzu (mind. `enabled`, `stop_update_policy`, rule‑config für MaxHoldingTime).  
   - `PositionView` um `status` und `meta` erweitern (serde‑fähig).  
   - Action‑API: MVP‑konform bleiben, aber Post‑MVP Actions dürfen existieren; dokumentiere Verhalten klar.  
   - Konfliktauflösung: ClosePosition gewinnt, danach Priority/RuleId.

3) **Scenario‑6 Timeframe‑Normalisierung**  
   - `scenario6_timeframes` in `from_params` konsequent normalisieren (uppercase + trim).  
   - Ebenso Schlüssel in `scenario6_params` konsistent behandeln (mind. tolerant match).

7) **Tests ergänzen**  
   - HTF‑Filtertests (Bias A/B) für Above/Below/Both/None.  
   - Tests für Parameter‑Override‑Hierarchie (falls Override‑Logik im Strategy‑Layer liegt; sonst gezielte Unit‑Tests am entsprechenden Modul).

### Standards‑Abweichungen beheben
- **Docstrings**: Alle `pub` Items ohne `///` dokumentieren (Strategy‑/Trade‑Mgmt‑APIs).  
- **Lint‑Regel**: `#![deny(missing_docs)]` und `#![warn(clippy::pedantic)]` passend hinzufügen (mindestens in den betroffenen Crates), ohne Build zu brechen.

### Akzeptanzkriterien
- Alle neuen Felder serialisierbar, Default‑Werte stabil.  
- Tests grün; neue Tests decken HTF‑Filter und Override‑Hierarchie ab.  
- Doc‑Coverage für public API vollständig.

### Hinweise
- Keine Änderungen an Timestamp‑Semantik (Open/Close).  
- Keine Anpassung von Szenario‑Logik für Long/Short.  
- Fokussiere auf minimale, plan‑konforme Änderungen.
