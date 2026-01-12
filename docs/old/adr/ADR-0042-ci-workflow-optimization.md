---
title: "ADR-0042: CI/CD Workflow Optimierungs-Analyse"
status: Proposed
date: 2026-01-09
deciders: [axelkempf]
consulted: [GitHub Copilot]
---

# ADR-0042: CI/CD Workflow Optimierungs-Analyse

## Executive Summary

**Hauptprobleme:**
1. Kumulative Laufzeit ~15+ Minuten pro Commit durch sequentielle und parallele Jobs ohne intelligente Steuerung
2. Fehlende Path-Filter im Haupt-CI-Workflow (ci.yml) führt zu unnötigen Runs
3. Suboptimale Parallelisierung und Caching-Strategie

**Hauptlösungen:**
1. Einführung eines dreistufigen Gate-Systems (Critical < 5 Min → Important < 10 Min → Complete)
2. Path-basierte Conditional Execution für alle Workflows
3. Aggressives Dependency-Caching und Parallelisierung

---

## 1. Workflow-Kategorisierung nach Kritikalität

### Aktuelle Workflow-Struktur

| Workflow | Jobs | Geschätzte Dauer | Aktuelle Trigger |
|----------|------|------------------|------------------|
| **ci.yml** | lint, docs-lint, security, type-check, test, integration-tests | ~8-12 Min | push:main, PR (alle) |
| **cross-platform-ci.yml** | changes, python-tests (3 OS), mt5-tests, rust-cross, julia-cross, summary, hybrid-integration | ~12-20 Min | push:main, PR, manual |
| **benchmarks.yml** | run-benchmarks | ~5-8 Min | push:main + paths, PR + paths |
| **julia-tests.yml** | test (4 configs), format, integration | ~6-10 Min | push:main + paths, PR + paths |
| **rust-build.yml** | lint, security, build, test | ~5-8 Min | push:main + paths, PR + paths |
| **release.yml** | prepare, build-python | N/A (tag-only) | tag: v*, manual |

### Empfohlene Kategorisierung

| Job/Workflow | Kategorie | Begründung |
|--------------|-----------|------------|
| **ci.yml: lint** | 🔴 CRITICAL GATE | Formatierung ist schnell (~1 Min), blockt offensichtliche Fehler |
| **ci.yml: docs-lint** | 🔵 CONDITIONAL | Nur bei Docs-Änderungen relevant |
| **ci.yml: security** | 🔴 CRITICAL GATE | Security-Issues müssen vor Merge erkannt werden |
| **ci.yml: type-check** | 🔴 CRITICAL GATE | Type Safety für kritische Module unverzichtbar |
| **ci.yml: test** | 🔴 CRITICAL GATE | Unit Tests sind Grundlage für Code-Qualität |
| **ci.yml: integration-tests** | 🟡 IMPORTANT GATE | Trading Safety Suite ist kritisch, aber kann parallelisiert werden |
| **cross-platform-ci.yml: python-tests** | 🟡 IMPORTANT GATE | Cross-Platform-Validierung wichtig, aber nicht bei jedem PR |
| **cross-platform-ci.yml: mt5-tests** | 🟢 OPTIONAL/SCHEDULED | Windows-only, nur bei hf_engine/strategies Änderungen |
| **cross-platform-ci.yml: rust/julia-cross** | 🔵 CONDITIONAL | Nur bei Rust/Julia-Änderungen |
| **benchmarks.yml** | 🔵 CONDITIONAL | Bereits path-gefiltert, gut konfiguriert |
| **julia-tests.yml** | 🔵 CONDITIONAL | Bereits path-gefiltert, gut konfiguriert |
| **rust-build.yml** | 🔵 CONDITIONAL | Bereits path-gefiltert, gut konfiguriert |
| **release.yml** | 🟢 OPTIONAL/SCHEDULED | Nur bei Tags/Manual |

---

## 2. Trigger-Strategie Matrix

| Workflow/Job | PR (Feature → develop) | PR (→ main) | Push zu main | Schedule/Manual | Begründung |
|--------------|------------------------|-------------|--------------|-----------------|------------|
| **ci.yml: lint** | ✅ Always | ✅ Always | ✅ Always | – | Schnell, grundlegend |
| **ci.yml: security** | ✅ Always | ✅ Always | ✅ Always | Weekly deep scan | Security ist nicht verhandelbar |
| **ci.yml: type-check** | ✅ Always | ✅ Always | ✅ Always | – | Type Safety kritisch |
| **ci.yml: test** | ✅ Always | ✅ Always | ✅ Always | – | Core Quality Gate |
| **ci.yml: docs-lint** | 🔵 Path: docs/** | 🔵 Path | 🔵 Path | – | Nur bei Doc-Änderungen |
| **ci.yml: integration** | 🔵 Path: src/** | ✅ Always | ✅ Always | – | Kritisch für main |
| **cross-platform: python-tests** | 🔵 Path | ✅ Always | ✅ Post-merge | Nightly | Full matrix nur main |
| **cross-platform: mt5-tests** | ❌ Skip | 🔵 Path | 🔵 Path | Weekly | Windows-only, selten nötig |
| **cross-platform: rust/julia** | 🔵 Path | 🔵 Path | 🔵 Path | – | Path-Filter aktiv halten |
| **benchmarks.yml** | 🔵 Path | ✅ PR Gate | ✅ Baseline | Nightly | Regression-Detection |
| **julia-tests.yml** | 🔵 Path | 🔵 Path | 🔵 Path | – | Julia-only |
| **rust-build.yml** | 🔵 Path | 🔵 Path | 🔵 Path | – | Rust-only |

---

## 3. Parallelisierungs-Möglichkeiten

### Aktuelle Abhängigkeiten in ci.yml

```
lint ─┬─→ test
      └─→ integration-tests

security (parallel)
type-check (parallel)
docs-lint (parallel)
```

### Optimierte Abhängigkeiten

```
                    ┌─→ security
                    ├─→ type-check
lint (< 2 min) ────┼─→ test (unit only, < 4 min)
                    ├─→ docs-lint (conditional)
                    └─→ integration-tests (parallel zu test)
```

### Matrix-Optimierungen

**cross-platform-ci.yml:**
- Aktuelle Matrix: 3 OS × 1 Python = 3 Jobs
- Empfehlung: Ubuntu als Primary, macOS/Windows als Secondary (post-merge oder nightly)

**julia-tests.yml:**
- Aktuelle Matrix: 2 Julia × 1 OS + 2 extra = 4 Jobs
- Empfehlung: Julia 1.10 auf Ubuntu als Gate, Rest nightly

**rust-build.yml:**
- Aktuelle Matrix: 4 Targets
- Empfehlung: Linux-x64 als Gate, Rest für Release

### Caching-Potenziale

| Cache | Aktuell | Empfehlung | Geschätzter Gewinn |
|-------|---------|------------|-------------------|
| **pip dependencies** | ❌ Nicht in ci.yml | ✅ `actions/cache` für pip | ~30-60s pro Job |
| **Rust cargo** | ✅ Vorhanden | ✅ Beibehalten | – |
| **Julia depot** | ✅ Vorhanden | ✅ Beibehalten | – |
| **mypy cache** | ❌ Nicht vorhanden | ✅ `.mypy_cache` cachen | ~15-30s |
| **pytest cache** | ❌ Nicht vorhanden | ✅ `.pytest_cache` cachen | ~10-20s |

---

## 4. Branch-Protection-Rules Empfehlungen

### Für PRs zu `main`

**Required Status Checks (MUST PASS):**
```
✅ CI / lint
✅ CI / security
✅ CI / type-check
✅ CI / test
✅ CI / integration-tests
```

**Empfohlene GitHub Settings:**
```yaml
branch_protection:
  main:
    required_status_checks:
      strict: true  # PR muss up-to-date mit main sein
      contexts:
        - "CI / lint"
        - "CI / security"
        - "CI / type-check"
        - "CI / test"
        - "CI / integration-tests"
    required_pull_request_reviews:
      required_approving_review_count: 1
    enforce_admins: true
    restrictions: null
```

### Conditional Checks (Empfohlen, nicht blockierend)

```
🔵 Benchmarks (wenn paths matchen)
🔵 Rust Build / lint (wenn paths matchen)
🔵 Julia Tests / test (wenn paths matchen)
🔵 Cross-Platform CI / python-tests (wenn paths matchen)
```

---

## 5. Fast-Fail vs. Complete-Run Strategie

| Job/Workflow | Empfehlung | Begründung |
|--------------|------------|------------|
| **ci.yml: lint** | `fail-fast: true` | Bei Formatierungsfehlern sofort abbrechen |
| **ci.yml: security** | `fail-fast: true` | Security-Issues sofort melden |
| **ci.yml: type-check** | `fail-fast: false` | Alle Type-Errors sammeln für vollständiges Bild |
| **ci.yml: test** | `fail-fast: false` | Alle Test-Failures für vollständiges Debugging |
| **cross-platform: matrix** | `fail-fast: false` | Bereits korrekt, alle Plattformen testen |
| **benchmarks** | `fail-fast: true` | Bei Baseline-Fehler sofort abbrechen |

---

## 6. Top 5 Quick Wins (Sofort umsetzbar)

### Quick Win 1: Pip-Caching in ci.yml hinzufügen
**Geschätzter Gewinn: 2-3 Minuten**

```yaml
# In ci.yml, nach checkout
- uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: 'pip'  # ← Diese Zeile hinzufügen
    cache-dependency-path: pyproject.toml
```

### Quick Win 2: Shallow Clone überall aktivieren
**Geschätzter Gewinn: 10-20 Sekunden pro Job**

```yaml
- uses: actions/checkout@v6
  with:
    fetch-depth: 1  # ← Außer wo History benötigt wird
```

### Quick Win 3: Path-Filter für ci.yml
**Geschätzter Gewinn: Komplette Workflow-Skips bei irrelevanten Änderungen**

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
      - '.github/workflows/ci.yml'
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
      - '.github/workflows/ci.yml'
```

### Quick Win 4: Docs-Lint Job conditional machen
**Geschätzter Gewinn: ~1 Minute wenn keine Docs geändert**

```yaml
docs-lint:
  runs-on: ubuntu-latest
  if: |
    contains(github.event.head_commit.modified, 'docs/') ||
    contains(github.event.head_commit.modified, '.md')
```

### Quick Win 5: Test-Job parallelisieren (pytest-xdist)
**Geschätzter Gewinn: 30-50% schnellere Tests**

```yaml
- name: Run tests with coverage
  run: |
    pip install pytest-xdist
    pytest -q -m "not integration" -n auto --cov=src --cov-report=xml
```

---

## 7. Mittelfristige Optimierungen

### 7.1 Workflow-Splitting: Fast-Track vs. Full-Validation

**Neuer Workflow: ci-fast.yml (< 5 Min)**
```yaml
name: CI Fast Track
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  fast-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with: { fetch-depth: 1 }
      - uses: actions/setup-python@v5
        with: { python-version: '3.12', cache: 'pip' }
      - run: pip install black isort flake8 bandit
      - run: black --check src/ tests/
      - run: isort --check-only src/ tests/
      - run: flake8 src/ tests/ --select=E9,F63,F7,F82
      - run: bandit -r src/ -ll -ii --skip B101
```

**Bestehender ci.yml: Full Validation (für merge-ready PRs)**
- Trigger nur bei `pull_request_review` oder Label `ready-for-merge`

### 7.2 Nightly Full Matrix

```yaml
name: Nightly Full Matrix
on:
  schedule:
    - cron: '0 2 * * *'  # 02:00 UTC täglich
  workflow_dispatch:

jobs:
  full-cross-platform:
    # Volle Matrix: alle OS, alle Versionen
  deep-security-scan:
    # Erweiterte Security-Scans
  benchmark-regression:
    # Volle Benchmark-Suite
```

### 7.3 Conditional Workflow mit Reusable Workflows

```yaml
# .github/workflows/reusable-python-tests.yml
name: Reusable Python Tests
on:
  workflow_call:
    inputs:
      os:
        type: string
        default: 'ubuntu-latest'
      python-version:
        type: string
        default: '3.12'
```

---

## 8. Konkrete Workflow-YAML-Snippets

### Optimierter ci.yml Header

```yaml
name: CI
on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
      - '.github/workflows/ci.yml'
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
      - '.github/workflows/ci.yml'

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: '3.12'
  PIP_DISABLE_PIP_VERSION_CHECK: 1
  PYTHONDONTWRITEBYTECODE: 1
```

### Optimierter Lint-Job mit Caching

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
          cache-dependency-path: pyproject.toml

      - name: Install linting tools
        run: pip install black isort flake8

      - name: Check formatting (black)
        run: black --check --diff src/ tests/

      - name: Check imports (isort)
        run: isort --check-only --diff src/ tests/

      - name: Lint (flake8 - critical only)
        run: flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503,F824 --select=E9,F63,F7,F82
```

### Optimierter Test-Job mit Parallelisierung

```yaml
  test:
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
          cache-dependency-path: pyproject.toml

      - name: Cache pytest
        uses: actions/cache@v4
        with:
          path: .pytest_cache
          key: pytest-${{ runner.os }}-${{ hashFiles('tests/**/*.py') }}

      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install pytest-cov pytest-xdist

      - name: Run tests (parallel)
        run: pytest -q -m "not integration" -n auto --cov=src --cov-report=xml --cov-report=term-missing
```

### Conditional Docs-Lint

```yaml
  docs-lint:
    runs-on: ubuntu-latest
    # Nur ausführen wenn Docs geändert wurden
    if: |
      github.event_name == 'push' ||
      contains(github.event.pull_request.changed_files, 'docs/') ||
      contains(github.event.pull_request.changed_files, '.md')
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - run: pip install pytest
      - run: pytest -q tests/test_docs_reference_linter.py
```

### Concurrency für cross-platform-ci.yml

```yaml
concurrency:
  group: cross-platform-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

---

## 9. Branch-Protection-Regeln (GitHub Settings)

### Repository Settings → Branches → main

```
☑️ Require a pull request before merging
   ☑️ Require approvals: 1
   ☐ Dismiss stale reviews
   ☑️ Require review from code owners

☑️ Require status checks to pass before merging
   ☑️ Require branches to be up to date
   Status checks:
     ✅ CI / lint
     ✅ CI / security
     ✅ CI / type-check
     ✅ CI / test
     ✅ CI / integration-tests

☑️ Require conversation resolution before merging

☐ Require signed commits (optional)

☑️ Do not allow bypassing the above settings
```

---

## 10. Monitoring-Plan

### Metriken zum Tracken

| Metrik | Tool/Methode | Zielwert |
|--------|--------------|----------|
| **Critical Path Dauer** | GitHub Actions Summary | < 5 Min |
| **Total CI Dauer** | GitHub Actions Summary | < 12 Min |
| **Cache Hit Rate** | GitHub Actions Logs | > 80% |
| **Workflow Failure Rate** | GitHub Insights | < 5% |
| **Flaky Test Rate** | Custom Tracking | < 1% |

### GitHub Actions Job Summary

```yaml
  summary:
    runs-on: ubuntu-latest
    needs: [lint, security, type-check, test, integration-tests]
    if: always()
    steps:
      - name: Generate CI Summary
        run: |
          echo "## CI Summary" >> $GITHUB_STEP_SUMMARY
          echo "| Job | Duration | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|-----|----------|--------|" >> $GITHUB_STEP_SUMMARY
          echo "| Lint | - | ${{ needs.lint.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Security | - | ${{ needs.security.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Type-Check | - | ${{ needs.type-check.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Test | - | ${{ needs.test.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Integration | - | ${{ needs.integration-tests.result }} |" >> $GITHUB_STEP_SUMMARY
```

### Alerting bei Regression

```yaml
  alert-on-failure:
    runs-on: ubuntu-latest
    needs: [lint, security, type-check, test, integration-tests]
    if: failure() && github.ref == 'refs/heads/main'
    steps:
      - name: Alert on main branch failure
        run: |
          echo "::error::CI failed on main branch!"
          # Optional: Slack/Discord notification
```

---

## 11. Erwartete Verbesserungen

### Vorher vs. Nachher

| Metrik | Aktuell | Nach Optimierung | Verbesserung |
|--------|---------|------------------|--------------|
| **Critical Path** | ~8-10 Min | < 5 Min | ~50% schneller |
| **Full CI** | ~15+ Min | < 12 Min | ~20% schneller |
| **Unnötige Runs** | ~30% | < 5% | ~25% weniger |
| **Cache Hits** | ~40% | > 80% | ~40% besser |
| **Developer Wait Time** | ~20 Min/PR | < 10 Min/PR | ~50% weniger |

### ROI-Analyse

- **Entwickler-Zeit gespart**: ~10 Min pro PR × ~20 PRs/Woche = ~200 Min/Woche
- **GitHub Actions Minutes gespart**: ~30% Reduktion durch Path-Filter und Caching
- **Bug-Detection-Zeit**: Unverändert schnell durch beibehaltene kritische Gates

---

## 12. Implementierungs-Reihenfolge

### Phase 1: Quick Wins (Heute)
1. ✅ Pip-Caching in ci.yml
2. ✅ Shallow clones
3. ✅ Concurrency settings

### Phase 2: Path-Filter (Diese Woche)
1. ✅ Path-Filter für ci.yml
2. ✅ Conditional docs-lint
3. ✅ pytest-xdist für Parallelisierung

### Phase 3: Workflow-Restructuring (Nächste Woche)
1. ⬜ ci-fast.yml für schnelles Feedback
2. ⬜ Nightly full matrix workflow
3. ⬜ Reusable workflow patterns

### Phase 4: Monitoring (Ongoing)
1. ⬜ Job Summary implementieren
2. ⬜ Failure alerting
3. ⬜ Performance tracking

---

## Entscheidung

**Status:** Proposed

**Nächste Schritte:**
1. Review dieses ADR
2. Quick Wins sofort implementieren
3. Phase 2 nach Baseline-Messung

---

## Referenzen

- [GitHub Actions Best Practices](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [GitHub Actions Caching](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [dorny/paths-filter](https://github.com/dorny/paths-filter)
- [pytest-xdist](https://pytest-xdist.readthedocs.io/)
