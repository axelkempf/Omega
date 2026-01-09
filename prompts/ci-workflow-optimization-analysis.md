# CI/CD Workflow Optimierungs-Analyse Prompt

## Kontext

Du bist ein CI/CD-Experte, der die GitHub Actions Workflows eines Python-basierten Trading-Stack-Projekts mit Rust/Julia-FFI-Integration analysiert. Das Projekt folgt strengen Qualitätsstandards und hat ein kritisches Ziel: **Auf dem `main` Branch darf niemals fehlerhafter Code landen.**

## Aktuelles Problem

- **Laufzeit pro Commit**: ~15 Minuten für alle Workflows
- **Kumulative Wartezeit**: Bei Fehlern und Korrekturen häufen sich Stunden an Wartezeit
- **Kritische Anforderung**: Null-Toleranz für Bugs auf `main` Branch
- **Trade-off**: Balance zwischen Geschwindigkeit und Sicherheit

## Vorhandene Workflows

Die folgenden Workflows existieren im Repository:

1. **ci.yml** - Haupt-CI-Pipeline
   - Lint (black, isort, flake8)
   - Docs-Lint
   - Security (bandit, pip-audit)
   - Type-Check (mypy - mehrere Stufen)
   - Tests (pytest mit umfangreichen Tests)
   - Directory Structure Validation
   - FFI Contract Tests
   - Migration Tests

2. **cross-platform-ci.yml** - Cross-Platform Matrix
   - Linux, macOS, Windows Tests
   - Path-basierte Conditional Execution
   - MT5-spezifische Windows-Tests
   - Rust/Julia-spezifische Tests

3. **benchmarks.yml** - Performance Benchmarks
   - Baseline-Vergleiche
   - Performance-Regression-Detection
   - Path-Filter für Core-Module

4. **julia-tests.yml** - Julia-spezifische Tests
   - Julia Package Tests
   - Python-Julia FFI Validation
   - Path-Filter für Julia-Module

5. **rust-build.yml** - Rust Build & Tests
   - Cargo Build
   - Clippy Linting
   - Rust Tests & Benchmarks
   - Path-Filter für Rust-Module

6. **release.yml** - Release Automation
   - Wahrscheinlich nur bei Tags

## Deine Aufgabe

Analysiere die bestehenden Workflows und erstelle eine **strategische Optimierungs-Empfehlung** mit folgenden Komponenten:

### 1. Workflow-Kategorisierung nach Kritikalität

Kategorisiere jeden Job/Workflow in eine dieser Stufen:

- **🔴 CRITICAL GATE** (Muss bei jedem PR/Push auf main laufen, blockiert Merge)
  - Beispiele: Unit Tests, Lint, Security Critical Issues, Type Safety für kritische Module
  - Ziel: < 5 Minuten

- **🟡 IMPORTANT GATE** (Sollte laufen, aber kann parallelisiert oder optimiert werden)
  - Beispiele: Integration Tests, Cross-Platform Tests (selektiv), Performance-Tests
  - Ziel: < 10 Minuten parallel

- **🟢 OPTIONAL/SCHEDULED** (Kann auf Nightly/Weekly oder manuell laufen)
  - Beispiele: Full Cross-Platform Matrix, Umfangreiche Benchmarks, Deep Security Scans
  - Ziel: Laufen regelmäßig, aber nicht bei jedem Commit

- **🔵 CONDITIONAL** (Nur bei Änderungen an spezifischen Pfaden)
  - Beispiele: Rust-Build nur bei Rust-Änderungen, Julia-Tests nur bei Julia-Änderungen
  - Ziel: Maximale Effizienz durch Selektivität

### 2. Trigger-Strategie Matrix

Erstelle eine Tabelle mit Empfehlungen:

| Workflow/Job | PR (Feature Branch) | PR (zu main) | Push zu main | Schedule/Manual | Begründung |
|--------------|---------------------|--------------|--------------|-----------------|------------|
| ... | ... | ... | ... | ... | ... |

### 3. Parallelisierungs-Möglichkeiten

Identifiziere:
- Jobs, die parallel laufen können (keine Dependencies)
- Matrix-Strategien, die optimiert werden können
- Caching-Potenziale (Dependencies, Build-Artefakte)

### 4. Branch-Protection-Rules Empfehlungen

Definiere, welche Checks **required** sein müssen für:
- PRs zu `main`
- Direkte Pushes zu `main` (falls erlaubt)

### 5. Fast-Fail vs. Complete-Run Strategie

Empfehle:
- Welche Jobs sollten bei erstem Fehler abbrechen (`fail-fast: true`)
- Welche sollten komplett durchlaufen für vollständigen Feedback

### 6. Konkrete Optimierungs-Vorschläge

Für jeden Workflow:

#### Schnelligkeit-Gewinne:
- Caching-Strategien (Dependencies, Build-Artefakte, Test-Caches)
- Parallelisierung (Matrix-Splits, Job-Dependencies optimieren)
- Selektive Ausführung (Path-Filter, Conditional Steps)
- Shallow Clones (`fetch-depth: 1`)

#### Qualitäts-Sicherung:
- Welche Tests/Checks sind unverzichtbar für main-Branch-Qualität
- Welche Checks können "weicher" sein (Warnings statt Failures)
- Post-Merge-Validierung vs. Pre-Merge-Gating

#### Kosten-Nutzen:
- Welche Jobs liefern den höchsten ROI (Return on Investment) für Fehler-Prävention
- Welche sind "Nice-to-have" aber nicht kritisch

### 7. Konkrete Workflow-Änderungen

Schlage vor:

```yaml
# Beispiel: Optimierter Trigger mit Path-Filtern
on:
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
  push:
    branches: [main]
    paths-ignore:
      - 'docs/**'
      - '*.md'
```

### 8. Monitoring & Metriken

Empfehle:
- Welche Metriken sollten getrackt werden (Workflow-Dauer, Fehlerrate, etc.)
- Wie man Regressions-Detektion für CI-Performance implementiert

## Erfolgskriterien der Optimierung

Die Optimierung ist erfolgreich, wenn:

1. ✅ **Null Bugs auf main**: Alle kritischen Checks laufen vor Merge
2. ✅ **< 8 Min für Critical Path**: Schnelles Feedback für Entwickler
3. ✅ **< 15 Min für Complete Run**: Alle wichtigen Checks abgeschlossen
4. ✅ **Intelligente Selektivität**: Unnötige Jobs werden übersprungen
5. ✅ **Klare Feedback-Priorisierung**: Entwickler sehen zuerst die wichtigsten Fehler

## Zusätzliche Überlegungen

### Development Workflow:
- Feature Branches: Schnelles Feedback, leichtgewichtige Checks
- PRs zu main: Vollständige Validierung vor Merge
- Post-Merge: Umfangreiche Tests (Cross-Platform, Benchmarks)

### Hybrid Stack Besonderheiten:
- Python ≥3.12 (Haupt-Stack)
- Rust-Module (nur bei Rust-Änderungen testen)
- Julia-Module (nur bei Julia-Änderungen testen)
- MT5-Integration (nur auf Windows, nur wenn relevant)

### Path-basierte Intelligenz:
```
src/rust_modules/** → Rust-Build
src/julia_modules/** → Julia-Tests
src/backtest_engine/** → Backtest-Tests + Benchmarks
src/hf_engine/** → Live-Engine Tests (ggf. MT5)
src/strategies/** → Strategy-Tests
tests/** → Relevante Test-Suites
```

## Output-Format

Erstelle einen strukturierten Report mit:

1. **Executive Summary** (2-3 Sätze: Hauptprobleme + Hauptlösungen)
2. **Workflow-Kategorisierung** (Tabelle)
3. **Trigger-Strategie Matrix** (Tabelle)
4. **Top 5 Quick Wins** (Sofort umsetzbare Optimierungen)
5. **Mittelfristige Optimierungen** (Erfordern mehr Arbeit)
6. **Konkrete Workflow-YAML-Snippets** (Ready-to-use)
7. **Branch-Protection-Regeln** (GitHub Settings)
8. **Monitoring-Plan** (Wie messe ich den Erfolg?)

## Beispiel-Analyse (als Inspiration)

### Beispiel: ci.yml Lint Job

**Aktuell:**
- Läuft bei jedem Push/PR
- Dauer: ~2 Min
- Kritikalität: 🔴 CRITICAL GATE

**Optimierung:**
```yaml
lint:
  runs-on: ubuntu-latest
  # Optimization: Skip if only docs changed
  if: |
    github.event_name == 'push' ||
    (github.event_name == 'pull_request' &&
     !contains(github.event.pull_request.labels.*.name, 'skip-lint'))
  steps:
    - uses: actions/checkout@v6
      with:
        fetch-depth: 1  # Shallow clone
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
        cache: 'pip'  # Cache pip dependencies
    # ... rest bleibt gleich
```

**Gewinn:**
- ~30s durch Pip-Cache
- Übersprungen bei Docs-only Changes
- Bleibt kritischer Gate

---

## Starte die Analyse

Analysiere nun die vorhandenen Workflows im Repository `/Users/axelkempf/Omega/.github/workflows/` und erstelle die oben beschriebene Optimierungs-Empfehlung.

**Fokus dabei auf:**
1. Minimierung der Wartezeit für Entwickler
2. Maximale Sicherheit gegen Bugs auf main
3. Praktische, sofort umsetzbare Vorschläge
4. Berücksichtigung der Hybrid-Stack-Architektur (Python/Rust/Julia)
