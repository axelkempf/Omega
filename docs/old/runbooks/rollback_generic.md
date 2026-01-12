# Generic Rollback Procedure (FFI / Migration)

Dieses Dokument beschreibt eine **generische Rollback-Prozedur** für Module, die im Rahmen der Rust/Julia-Migration via Feature-Flag (Python-Fallback) ausgerollt werden.

## Wann verwenden?

- Invarianten schlagen fehl (Divergenz Python vs. Rust/Julia).
- Runtime/FFI-Fehler (Panic, Segfault, unexpected Exception).
- Determinismus bricht (Golden-File Tests, Snapshot-Tests).
- Performance-Regression (typisch >10%).

## Rollback-Schritte

1. **Feature-Flag deaktivieren**

   Deaktiviere das modul-spezifische Flag, damit der **Python-Fallback** greift.

   Beispiele (je nach Runbook):

   - `export USE_RUST_PORTFOLIO=false`
   - `export USE_RUST_SLICER=false`
   - `export OMEGA_USE_RUST_MULTI_SYMBOL=false`

   Wenn das Flag aus einer JSON/YAML-Konfiguration gelesen wird, committe keinen Hotfix in `configs/` ohne Review – nutze für akute Incidents zuerst eine Runtime-Override-Option oder eine separate „hotfix config“.

2. **Prozess sauber stoppen und neu starten**

   - Live/UI gesteuert: Stop-Signal in `var/tmp/stop_<account_id>.signal` setzen bzw. UI-Endpoint nutzen.
   - Backtests: Prozess neu starten.

   Erwartung: nach Restart werden keine Rust/Julia Backends geladen.

3. **Logs prüfen**

   - Logs: `var/logs/`
   - Ergebnisse/Artefakte: `var/results/`

   Sammle mindestens:

   - Exception-Trace / Error-Code
   - Konfiguration (die verwendet wurde)
   - Repro-Schritte (Command + Config-Pfad)

4. **Verifikation (Smoke / Regression)**

   Führe einen minimalen, deterministischen Check aus:

   - betroffene Unit-/Integration-Tests aus dem Runbook
   - mindestens 1 Golden-/Snapshot-Test, falls vorhanden
   - optional: relevanter Benchmark zum Regression-Nachweis

5. **Issue erstellen**

   Erstelle ein Issue mit:

   - Modulname
   - erwartetes vs. tatsächliches Verhalten
   - Logs/Artefakte (sanitisiert; keine Secrets)
   - betroffene Plattform (macOS/Linux/Windows)

## Wieder-Freigabe (Re-Enable)

Re-Enable erst, wenn:

- Root Cause fixiert ist (inkl. Test, der den Bug reproduziert und verhindert).
- Determinismus/Schema-Invarianten nachweislich intakt sind.
- Performance-Regression ausgeschlossen ist (Baseline-Vergleich dokumentiert).
