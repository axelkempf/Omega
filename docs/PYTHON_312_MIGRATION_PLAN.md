# Konvertierungsplan: Python 3.10 → Python 3.12

## Executive Summary

Dieses Dokument beschreibt den vollständigen Migrationsplan für die Omega Codebase von Python 3.10 auf Python 3.12. Die Migration bietet Performance-Verbesserungen, neue Sprachfeatures und verbesserte Typ-Unterstützung.

**Aktueller Stand:**
- Python Version: ≥3.10 (pyproject.toml: `requires-python = ">=3.10"`)
- CI/CD läuft auf Python 3.11
- pre-commit black verwendet `python3.11`
- mypy konfiguriert für `python_version = "3.10"`

**Ziel:**
- Minimum Python Version: ≥3.12
- Nutzung neuer Python 3.11/3.12 Features
- Verbesserte Performance und Typ-Sicherheit

---

## Inhaltsverzeichnis

1. [Kompatibilitätsanalyse](#1-kompatibilitätsanalyse)
2. [Neue Python 3.12 Features](#2-neue-python-312-features)
3. [Dependency-Analyse](#3-dependency-analyse)
4. [Konfigurationsänderungen](#4-konfigurationsänderungen)
5. [Code-Migrationsschritte](#5-code-migrationsschritte)
6. [CI/CD-Anpassungen](#6-cicd-anpassungen)
7. [Testplan](#7-testplan)
8. [Rollout-Strategie](#8-rollout-strategie)
9. [Risikobewertung](#9-risikobewertung)
10. [Checkliste](#10-checkliste)

---

## 1. Kompatibilitätsanalyse

### 1.1 Aktuelle Codebase-Charakteristiken

| Aspekt | Status | Anmerkungen |
|--------|--------|-------------|
| `from __future__ import annotations` | ✅ Verwendet | In ~55 Dateien, ermöglicht moderne Type Hints |
| Union-Syntax (`X \| Y`) | ✅ Teilweise | Moderne Syntax bereits in Basis-Modulen |
| `Optional[X]` | ⚠️ Gemischt | ~50 Dateien verwenden noch `Optional[X]` |
| `Dict[K,V]` / `List[T]` | ⚠️ Gemischt | ~70 Dateien verwenden noch `typing`-Generics |
| `match` Statement | ✅ Bereits verwendet | In 3 Dateien (runner.py, template, final_param_selector) |
| `Self` Type | ✅ Minimal | 1 Datei (news_filter.py) |
| `zoneinfo` | ✅ Verwendet | Native Python 3.9+ Zeitzonenbibliothek |

### 1.2 Breaking Changes in Python 3.11/3.12

#### Python 3.11 Breaking Changes
- `asynchat` und `asyncore` Module entfernt → **Nicht betroffen** (nicht verwendet)
- `smtpd` Modul entfernt → **Nicht betroffen** (nicht verwendet)
- `locale.getdefaultlocale()` deprecated → **Prüfen** (kann in Dependencies vorkommen)

#### Python 3.12 Breaking Changes
- `distutils` vollständig entfernt → **Nicht betroffen** (setuptools in pyproject.toml)
- `imp` Modul entfernt → **Nicht betroffen** (nicht verwendet)
- `sqlite3` requires SQLite 3.15.2+ → **OK** (moderne Systeme haben höhere Versionen)
- `typing.TypedDict` erbt nicht mehr von `dict` → **Prüfen** (falls TypedDict verwendet)
- `asyncio.get_event_loop()` Deprecation → **Prüfen** (in async Code)
- `pkg_resources` deprecated → **Nicht betroffen** (importlib.metadata verwendet)

### 1.3 Potenzielle Probleme im Code

```
Dateien mit altem Typing-Style (zu migrieren):
├── src/backtest_engine/core/event_engine.py   (Dict, List, Optional, Callable)
├── src/hf_engine/infra/logging/*.py           (Optional, List, Dict)
├── src/backtest_engine/optimizer/*.py         (Dict, List, Optional, Tuple)
└── ~70 weitere Dateien mit typing-Imports
```

---

## 2. Neue Python 3.12 Features

### 2.1 Empfohlene Features zur Adoption

#### 2.1.1 Type Parameter Syntax (PEP 695)
```python
# Alt (Python 3.10)
from typing import TypeVar, Generic
T = TypeVar('T')
class Stack(Generic[T]):
    def push(self, item: T) -> None: ...

# Neu (Python 3.12)
class Stack[T]:
    def push(self, item: T) -> None: ...
```

**Anwendung:** Basis-Klassen in `src/strategies/_base/`, `src/backtest_engine/core/`

#### 2.1.2 F-String Improvements (PEP 701)
```python
# Neu in Python 3.12: Verschachtelte Quotes und mehrzeilige F-Strings
# Quotes können jetzt innerhalb von F-Strings wiederverwendet werden
name = "world"
message = f"Hello {f"dear {name}"}"  # Erlaubt in Python 3.12
multiline = f"""
    Result: {
        compute_value()
    }
"""
```

#### 2.1.3 `@override` Decorator (PEP 698)
```python
from typing import override

class BacktestStrategy(Strategy):
    @override
    def generate_signal(self, symbol: str, date: datetime) -> list[TradeSetup]:
        ...
```

**Anwendung:** Alle Strategy-Implementierungen, PositionManager-Klassen

#### 2.1.4 Improved Error Messages
Python 3.12 bietet deutlich bessere Fehlermeldungen – automatisch verfügbar nach Upgrade.

### 2.2 Performance-Verbesserungen

| Feature | Verbesserung | Relevanz für Projekt |
|---------|-------------|---------------------|
| Faster CPython | ~5-15% schneller | ✅ Backtests, Optimizer |
| Comprehension Inlining | Schnellere Comprehensions | ✅ Datenverarbeitung |
| Asyncio Improvements | Bessere async Performance | ⚠️ Nur UI/Datafeed |
| Memory Optimizations | Geringerer Speicherverbrauch | ✅ Große Backtest-Runs |

---

## 3. Dependency-Analyse

### 3.1 Core Dependencies

| Dependency | Min. Version | Python 3.12 Support | Status |
|------------|-------------|---------------------|--------|
| pandas | ≥1.5 | ✅ Ab pandas 2.0 | **Empfehlung:** Update auf ≥2.1 |
| numpy | ≥1.23 | ✅ Ab numpy 1.24 | **Empfehlung:** Update auf ≥1.26 |
| fastapi | ≥0.109 | ✅ Vollständig | OK |
| pydantic | ≥2.5 | ✅ Vollständig | OK |
| optuna | ≥3.4 | ✅ Vollständig | OK |
| matplotlib | ≥3.7 | ✅ Ab 3.8 | **Empfehlung:** Update auf ≥3.8 |
| uvicorn | ≥0.23 | ✅ Vollständig | OK |
| MetaTrader5 | ≥5.0 | ⚠️ Zu prüfen | Windows-only, testen |
| joblib | ≥1.3 | ✅ Vollständig | OK |

### 3.2 Dev Dependencies

| Dependency | Min. Version | Python 3.12 Support | Status |
|------------|-------------|---------------------|--------|
| pytest | ≥7.4 | ✅ Vollständig | OK |
| black | ≥24.8.0 | ✅ Vollständig | OK |
| mypy | ≥1.13 | ✅ Vollständig | OK |
| isort | ≥5.13.2 | ✅ Vollständig | OK |
| flake8 | ≥7.1 | ✅ Vollständig | OK |

### 3.3 Analysis/ML Dependencies

| Dependency | Min. Version | Python 3.12 Support | Status |
|------------|-------------|---------------------|--------|
| scipy | ≥1.11 | ✅ Ab scipy 1.12 | **Empfehlung:** Update auf ≥1.12 |
| scikit-learn | ≥1.2 | ✅ Ab scikit-learn 1.3 | **Empfehlung:** Update auf ≥1.4 |
| hdbscan | ≥0.8 | ⚠️ Prüfen | Kann Probleme haben |
| torch | ≥2.1 | ✅ Ab torch 2.1 | OK |

### 3.4 Empfohlene Dependency-Updates

```toml
# pyproject.toml - Neue Minimum Versionen
dependencies = [
    "pandas>=2.1",          # war: >=1.5
    "numpy>=1.26",          # war: >=1.23
    "matplotlib>=3.8",      # war: >=3.7
    # ... Rest bleibt
]

[project.optional-dependencies]
analysis = [
    "scipy>=1.12",          # war: >=1.11
    "scikit-learn>=1.4",    # war: >=1.2
    "hdbscan>=0.8.33",      # Version mit Python 3.12 Support
]
```

---

## 4. Konfigurationsänderungen

### 4.1 pyproject.toml

```diff
# pyproject.toml

[project]
name = "omega"
version = "1.2.0"  # Version Bump für Breaking Change
- requires-python = ">=3.10"
+ requires-python = ">=3.11"

dependencies = [
-   "pandas>=1.5",
+   "pandas>=2.1",
-   "numpy>=1.23",
+   "numpy>=1.26",
-   "matplotlib>=3.7",
+   "matplotlib>=3.8",
    # ... Rest unverändert
]

[project.optional-dependencies]
analysis = [
-   "scipy>=1.11",
+   "scipy>=1.12",
-   "scikit-learn>=1.2",
+   "scikit-learn>=1.4",
    "hdbscan>=0.8.33",
    "tqdm>=4.65",
]

[tool.mypy]
- python_version = "3.10"
+ python_version = "3.12"
```

### 4.2 .pre-commit-config.yaml

```diff
# .pre-commit-config.yaml

  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
-       language_version: python3.11
+       language_version: python3.12
```

### 4.3 README.md

```diff
# README.md

## Voraussetzungen

- - Python **>= 3.10**
+ - Python **>= 3.11**
```

---

## 5. Code-Migrationsschritte

### 5.1 Phase 1: Type Hints Modernisierung (Optional, aber empfohlen)

#### Schritt 1: Entferne veraltete typing-Imports
```python
# Alt
from typing import Dict, List, Optional, Tuple, Union

# Neu - verwende built-in generics
# Dict -> dict, List -> list, Tuple -> tuple
# Optional[X] -> X | None
# Union[X, Y] -> X | Y
```

**Betroffene Dateien (Priorität):**
1. `src/strategies/_base/*.py` - Basis-Module (strikt typisiert)
2. `src/ui_engine/models.py` - Pydantic-Modelle
3. `src/backtest_engine/core/*.py` - Kern-Engine

#### Schritt 2: Nutze `@override` Decorator
```python
from typing import override  # Python 3.12+

class MyStrategy(Strategy):
    @override
    def name(self) -> str:
        return "my_strategy"
```

### 5.2 Phase 2: Neue Type Parameter Syntax (Optional)

```python
# Kandidaten für Type Parameter Syntax
# src/backtest_engine/core/indicator_cache.py
# src/strategies/_base/base_strategy.py
```

### 5.3 Phase 3: Performance-Optimierungen

Nach Migration können Python 3.12 spezifische Optimierungen genutzt werden:
- `itertools.batched()` für Batch-Verarbeitung
- Verbesserte `asyncio` Patterns

---

## 6. CI/CD-Anpassungen

### 6.1 .github/workflows/ci.yml

```diff
# .github/workflows/ci.yml

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
-       with: {python-version: '3.11'}
+       with: {python-version: '3.12'}

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
-       with: {python-version: '3.11'}
+       with: {python-version: '3.12'}

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
-       with: {python-version: '3.11'}
+       with: {python-version: '3.12'}

  test:
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
-       with: {python-version: '3.11'}
+       with: {python-version: '3.12'}

  integration-tests:
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
-       with: {python-version: '3.11'}
+       with: {python-version: '3.12'}
```

### 6.2 Optional: Multi-Version Matrix (Übergangsphase)

```yaml
# Während der Übergangsphase: Test auf beiden Versionen
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
```

---

## 7. Testplan

### 7.1 Pre-Migration Tests

- [ ] Vollständiger Test-Suite auf Python 3.11 (Baseline)
- [ ] Dokumentation aller bestehenden Test-Ergebnisse
- [ ] Identifikation kritischer Pfade

### 7.2 Migration Tests

| Testphase | Beschreibung | Erfolgskriterium |
|-----------|-------------|------------------|
| Unit Tests | `pytest -q -m "not integration"` | Alle Tests bestehen |
| Integration Tests | `pytest -q -m "integration"` | Alle Tests bestehen |
| Trading Safety | `pytest -q -m "trading_safety"` | Alle Tests bestehen |
| Type Checking | `mypy src/` | Keine neuen Fehler |
| Linting | `flake8`, `black --check`, `isort --check` | Keine Fehler |
| Security | `bandit -r src/`, `pip-audit` | Keine kritischen Issues |

### 7.3 Post-Migration Validation

- [ ] Backtest-Run mit Referenz-Konfiguration
- [ ] Vergleich Backtest-Ergebnisse (deterministische Reproduzierbarkeit)
- [ ] Performance-Benchmark (Laufzeit-Vergleich)
- [ ] UI-Engine Start und Funktionstest
- [ ] Walkforward-Optimizer Testlauf

### 7.4 Regressions-Checkliste

```bash
# Kritische Tests nach Migration
pytest tests/test_rating_robustness_and_stability.py -v
pytest tests/test_monte_carlo_fast_eval_consistency.py -v
pytest tests/test_deterministic_dev_mode_scores.py -v
pytest tests/integration/ -m "trading_safety" -v
```

---

## 8. Rollout-Strategie

### 8.1 Phasenplan

```
Phase 1: Vorbereitung (1-2 Tage)
├── Dependencies auf Python 3.12 Kompatibilität prüfen
├── Lokale Entwicklungsumgebung mit Python 3.12 aufsetzen
└── Vollständige Test-Suite auf Python 3.12 ausführen

Phase 2: Konfiguration (1 Tag)
├── pyproject.toml aktualisieren
├── CI/CD Workflows anpassen
├── pre-commit-config aktualisieren
└── README.md aktualisieren

Phase 3: Code-Migration (2-3 Tage, optional)
├── Type Hints modernisieren (schrittweise)
├── @override Decorator einführen
└── Neue Features nutzen wo sinnvoll

Phase 4: Testing & Validation (1-2 Tage)
├── Vollständige Test-Suite
├── Backtest-Vergleichstest
├── Performance-Benchmark
└── Live-Trading Dry-Run (wenn möglich)

Phase 5: Deployment (1 Tag)
├── PR erstellen und Review
├── Merge in main
├── Dokumentation aktualisieren
└── CHANGELOG.md ergänzen
```

### 8.2 Rollback-Plan

Falls kritische Probleme auftreten:
1. Revert des Merge-Commits
2. CI/CD auf Python 3.11 zurücksetzen
3. Issue erstellen mit Problem-Dokumentation
4. Root-Cause-Analyse durchführen

---

## 9. Risikobewertung

### 9.1 Risiko-Matrix

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Dependency-Inkompatibilität | Mittel | Hoch | Vorab-Tests, graduelle Updates |
| MetaTrader5 Python 3.12 Issues | Niedrig | Hoch | Windows-Test isoliert, Fallback |
| hdbscan Kompatibilität | Mittel | Mittel | Alternative: scikit-learn HDBSCAN |
| Performance-Regression | Niedrig | Mittel | Benchmark vor/nach Migration |
| Type Hint Fehler nach Update | Niedrig | Niedrig | mypy --strict Tests |

### 9.2 Bekannte Risiken

1. **MetaTrader5**: Windows-only Bibliothek, Python 3.12 Support nicht garantiert
   - **Mitigation:** Separate Test auf Windows-Umgebung vor Migration

2. **hdbscan**: Kann Kompilierungsprobleme haben
   - **Mitigation:** `hdbscan>=0.8.33` oder Alternative `sklearn.cluster.HDBSCAN`

3. **Pandas 2.x Breaking Changes**: `append()` entfernt, Copy-on-Write default
   - **Mitigation:** Code-Review für DataFrame-Operationen

---

## 10. Checkliste

### Pre-Migration
- [ ] Python 3.12 lokal installiert und getestet
- [ ] Alle Dependencies auf Kompatibilität geprüft
- [ ] Backup/Snapshot der aktuellen Konfiguration
- [ ] Baseline Test-Results dokumentiert

### Konfiguration
- [ ] `pyproject.toml` aktualisiert
- [ ] `.pre-commit-config.yaml` aktualisiert
- [ ] `.github/workflows/ci.yml` aktualisiert
- [ ] `README.md` aktualisiert
- [ ] `architecture.md` bei Bedarf aktualisiert

### Code-Änderungen (Optional)
- [ ] `typing` Imports modernisiert (Priorität: _base/, models.py)
- [ ] `@override` Decorator eingeführt
- [ ] Type Parameter Syntax wo sinnvoll

### Testing
- [ ] Unit Tests bestanden
- [ ] Integration Tests bestanden
- [ ] Trading Safety Tests bestanden
- [ ] Type Checking (mypy) bestanden
- [ ] Linting bestanden
- [ ] Security Scans bestanden
- [ ] Backtest-Reproduzierbarkeit validiert

### Deployment
- [ ] PR erstellt mit vollständiger Beschreibung
- [ ] Code Review durchgeführt
- [ ] CI/CD Pipeline grün
- [ ] Merge in main
- [ ] CHANGELOG.md aktualisiert
- [ ] Version auf 1.2.0 erhöht

---

## Anhang

### A. Nützliche Befehle

```bash
# Python 3.12 Installation (pyenv)
pyenv install 3.12.0
pyenv local 3.12.0

# Virtuelle Umgebung neu aufsetzen
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,analysis]

# Tests ausführen
pytest -q --tb=short

# Type Coverage Report
python tools/type_coverage.py

# Dependency Check
pip check
pip-audit
```

### B. Python 3.12 Release Notes

- [Python 3.12 What's New](https://docs.python.org/3.12/whatsnew/3.12.html)
- [PEP 695 – Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [PEP 698 – Override Decorator](https://peps.python.org/pep-0698/)
- [PEP 701 – F-String Syntax](https://peps.python.org/pep-0701/)

### C. Referenzen

- [Python 3.11 → 3.12 Porting Guide](https://docs.python.org/3.12/whatsnew/3.12.html#porting-to-python-3-12)
- [mypy Python 3.12 Support](https://mypy.readthedocs.io/)
- [NumPy Python Support](https://numpy.org/neps/nep-0029-deprecation_policy.html)

---

*Dokument erstellt: Januar 2025*
*Autor: GitHub Copilot*
*Version: 1.0*
