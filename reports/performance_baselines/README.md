## Zweck

Dieses Verzeichnis enthält **Performance-Snapshots** (z. B. `p0-01_*.json`) für die Migrationsvorbereitung (Rust/Julia). Diese Dateien sind primär **diagnostisch** und dienen als nachvollziehbare, menschenlesbare Referenz für die Performance in Python.

## Was ist in `p0-01_*.json` enthalten?

- Custom JSON-Format (kein `pytest-benchmark` Format)
- Typischer Inhalt:
  - Metadaten (Hardware/OS/Datum)
  - Laufzeiten (z. B. `init_seconds`)
  - Per-Operation Timing-Snapshots
  - Optional: Profil-/Top-Stacks

## CI-Regression-Gate (pytest-benchmark)

Die harte Regressionserkennung in CI läuft über `pytest-benchmark`:

- Tests: `tests/benchmarks/`
- Workflow: `.github/workflows/benchmarks.yml`
- Output: `.benchmark_results/benchmark_results.json`
- Artifact-Name: `benchmark-results-3.12`

### Baseline-Quelle in CI

- **Pull Requests:** Baseline-Vergleich ist **aktiv** und **blocking**. Die Baseline wird aus dem **letzten erfolgreichen `main` Run** dieses Workflows als Artifact geladen.
- **Push auf `main`:** Baseline-Vergleich ist **aus** (Bootstrap). Der Run erzeugt das Artifact, das danach als Baseline für PRs dient.

Optional kann eine repo-gepinnte Baseline verwendet werden, wenn `reports/performance_baselines/benchmark_baseline.json` existiert (hat Vorrang vor dem Artifact).

## Baseline „bootstrapen“ (wenn noch keine existiert)

1. Einmal den Workflow **auf `main`** laufen lassen (z. B. per `workflow_dispatch`).
2. Danach existiert ein erfolgreiches Baseline-Artifact, und PRs können gegen `main` vergleichen.

## Hinweise

- Die `p0-01_*.json` Snapshots sind **nicht** die Baseline-Datei für das CI-Regressions-Gate.
- Wenn PRs ohne Baseline laufen, ist das ein **harte Fehlerbedingung** (Operational Truth: kein stilles „grün“ ohne Vergleich).