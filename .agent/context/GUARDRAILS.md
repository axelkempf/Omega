# Omega Guardrails (Shared Context)

> Nicht verhandelbare Invarianten für alle Agenten und alle Tasks.

## 1. Determinismus / Reproduzierbarkeit

- **Backtests sind deterministisch**: Gleiche Inputs → bitgleiche Outputs
- **Keine Netz-Calls** in Backtest-Logik
- **Keine Systemzeit** in Berechnungen (nur Config/Logs)
- **Seeds fixieren** für alle Random-Komponenten
- **Keine Lookahead-Bias**: Nur Daten bis `idx` verwenden

## 2. Trading Safety First

- **Keine stillen Semantik-Änderungen** in Execution/Stops/Fees
- **Verhaltensänderungen brauchen**:
  - Config-Flag ODER
  - Migration + Tests + Doku
- **SL darf nie „wider" werden** (No SL Widening Policy)
- **Exit-Reasons müssen explizit sein** (stop_loss, take_profit, timeout, ...)

## 3. Runtime-State

- **`var/` ist operational kritisch**:
  - `var/tmp/heartbeat_<id>.txt`
  - `var/tmp/stop_<id>.signal`
  - `var/logs/`, `var/results/`
- **Pfad-Änderungen brauchen Migration**

## 4. MT5 / OS-Kompatibilität

- **MT5 ist Windows-only**
- **Backtests müssen auf macOS/Linux laufen** (ohne MT5)
- **Defensive Imports**: `try/except ImportError`

## 5. Dependency Policy

- **Single Source of Truth**: `pyproject.toml` (Python), `Cargo.toml` (Rust)
- **Neuer Import → Dependency hinzufügen**
- **Optionale Deps defensiv importieren**

## 6. Security

- **Keine Secrets in Code/Logs/Docs**
- **Allowlist-only Redaction** (nicht Blacklist)
- **Keine API-Keys hardcoded**
- **ENV-Vars für Secrets**

## 7. Output-Contract

- **Schema ist normativ**: `docs/OMEGA_V2_OUTPUT_CONTRACT_PLAN.md`
- **Felder, Types, Units, Rundung** einhalten
- **Breaking Changes brauchen Migration + Golden-Update**

## 8. Tests

- **Deterministisch**: Seeds fix, keine Netz-Calls
- **MT5 nicht voraussetzen**: Mocken
- **Golden/Parity für Contract-Änderungen**

---

## Checklist (vor jedem Merge)

- [ ] Determinismus gewährleistet
- [ ] Keine stillen Semantik-Änderungen
- [ ] Output-Contract eingehalten
- [ ] Keine Secrets geleakt
- [ ] Tests vorhanden und grün
- [ ] Doku aktualisiert (wenn Interface/Config betroffen)
