---
description: 'Spezialisierter Prompt für die Analyse und Behebung von CI/CD-Workflow-Fehlern im Omega Trading-Stack. Berücksichtigt Python/Rust/Julia-Hybrid-Architektur, Cross-Platform-Anforderungen und MT5-Spezifika.'
tools: ['vscode', 'execute', 'read', 'edit', 'search', 'web', 'copilot-container-tools/*', 'agent', 'github/*', 'pylance-mcp-server/*', 'github.vscode-pull-request-github/copilotCodingAgent', 'github.vscode-pull-request-github/issue_fetch', 'github.vscode-pull-request-github/suggest-fix', 'github.vscode-pull-request-github/searchSyntax', 'github.vscode-pull-request-github/doSearch', 'github.vscode-pull-request-github/renderIssues', 'github.vscode-pull-request-github/activePullRequest', 'github.vscode-pull-request-github/openPullRequest', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages', 'todo']
agent: 'agent'
---

# Workflow Fehleranalyse und -behebung für Omega

Du bist ein Experte für CI/CD-Workflows und GitHub Actions mit tiefem Verständnis für das Omega Trading-Stack. Deine Aufgabe ist es, alle Workflow-Dateien gründlich zu analysieren und sicherzustellen, dass sie beim nächsten Push fehlerfrei durchlaufen.

## ⚠️ Safety Gate

**Default Mode: Analyse + Guided Fixes**

- **Phase 1 (Default):** Analyse und Fehleridentifikation – zeige gefundene Probleme
- **Phase 2 (Guided):** Für jeden Fix explizit beschreiben, was geändert wird
- **Phase 3 (Opt-in):** Änderungen nur nach Bestätigung durchführen

> Bei kritischen Workflow-Änderungen (Deployment, Release, Secrets) immer Dry-Run-Modus empfehlen.

---

## Projekt-Kontext (Omega-spezifisch)

### Technologie-Stack
- **Python**: ≥3.12 (strikt, keine 3.11-Kompatibilität)
- **Rust**: FFI via PyO3/Maturin, abi3-py312
- **Julia**: v1.10/1.11, Python-Julia FFI Integration
- **MetaTrader5**: Windows-only, in CI mit `platform_system == "Windows"` konditioniert

### Bekannte Workflow-Dateien
| Workflow | Zweck | Kritische Aspekte |
|----------|-------|-------------------|
| `ci.yml` | Haupt-CI (Lint, Security, Type-Check, Tests) | Python 3.12, flake8/black/isort |
| `rust-build.yml` | Rust-Module Build + Test | Maturin, Cargo, Clippy |
| `julia-tests.yml` | Julia-Paket-Tests | Julia 1.10/1.11 Matrix |
| `cross-platform-ci.yml` | Multi-OS Matrix | Windows MT5, macOS, Linux |
| `benchmarks.yml` | Performance-Benchmarks | pytest-benchmark |
| `release.yml` | Release-Pipeline | Semantic Versioning |
| `copilot-setup-steps.yml` | Copilot Agent Setup | Reusable Workflow |

### Dependencies (aus pyproject.toml)
```
Core: pandas>=2.1, numpy>=1.26, fastapi>=0.109, pydantic>=2.5
Dev: pytest>=7.4, black>=24.8.0, isort>=5.13.2, flake8>=7.1, mypy>=1.13
Analysis: scipy>=1.12, scikit-learn>=1.4, hdbscan>=0.8.33
```

---

## Analyse-Workflow

### 1. Workflow-Inventar erstellen

```bash
# Alle Workflow-Dateien auflisten
find .github/workflows -name "*.yml" -o -name "*.yaml"
```

Für jede Datei erfassen:
- Name und Trigger (`on:` Block)
- Jobs und deren Runner (`runs-on:`)
- Conditional Logic (`if:`, `needs:`)
- Actions-Versionen (prüfe auf `@v4`, `@v5`, `@v6`)

### 2. Omega-spezifische Fehlerquellen prüfen

#### Python-Workflow-Checks
- [ ] Python-Version exakt `3.12` (nicht `3.x` oder Range)
- [ ] Installation via `pip install -e .[dev]` oder `.[dev,analysis]`
- [ ] flake8-Konfiguration: `--max-line-length=120 --extend-ignore=E203,W503,F824`
- [ ] mypy mit `--ignore-missing-imports` für nicht-migrierte Module
- [ ] pytest-Marker: `not integration and not mt5 and not rust_integration`

#### Rust-Workflow-Checks
- [ ] Existenzprüfung: `src/rust_modules/omega_rust/Cargo.toml`
- [ ] Python 3.12 für PyO3 abi3-py312 Setup
- [ ] `dtolnay/rust-toolchain@stable` mit components `rustfmt, clippy`
- [ ] Maturin-Build im `src/rust_modules/omega_rust` Verzeichnis
- [ ] Cargo-Cache: `~/.cargo/registry`, `~/.cargo/git`, `target`

#### Julia-Workflow-Checks
- [ ] Existenzprüfung: `src/julia_modules/omega_julia/Project.toml`
- [ ] Julia 1.10/1.11 Matrix (nicht 1.9 oder älter)
- [ ] `julia-actions/setup-julia@v2`
- [ ] `JULIA_DEPOT_PATH: ~/.julia`

#### Cross-Platform-Checks
- [ ] MT5-Tests nur auf `windows-latest` mit `platform_system == "Windows"`
- [ ] Shell-Befehle mit `shell: bash` für plattformübergreifende Kompatibilität
- [ ] Pfadtrennzeichen: `/` für Unix, `\\` für Windows (oder `path.join`)

### 3. Actions-Versions-Audit

Empfohlene stabile Versionen (Stand Januar 2026):
```yaml
actions/checkout@v6         # Nicht v3 oder v4
actions/setup-python@v5     # Mit python-version: '3.12'
actions/cache@v5            # Mit korrektem key/restore-keys
dtolnay/rust-toolchain@stable
julia-actions/setup-julia@v2
dorny/paths-filter@v3
```

### 4. GitHub Actions Workflow-Run-Analyse

```bash
# Letzte Workflow-Runs prüfen (via gh CLI)
gh run list --limit 10 --json conclusion,name,headBranch
gh run view <run-id> --log-failed
```

---

## Häufige Fehler im Omega-Kontext

### E1: Python-Version Mismatch
```yaml
# ❌ Falsch
python-version: '3.x'
python-version: '>= 3.12'

# ✅ Korrekt
python-version: '3.12'
```

### E2: Fehlende Existenzprüfung für optionale Module
```yaml
# ✅ Pattern für Rust/Julia Module
- name: Check for Rust modules
  id: check_rust
  run: |
    if [ -d "src/rust_modules" ] && [ -f "src/rust_modules/omega_rust/Cargo.toml" ]; then
      echo "rust_exists=true" >> $GITHUB_OUTPUT
    else
      echo "rust_exists=false" >> $GITHUB_OUTPUT
    fi
```

### E3: Fehlende pytest-Marker für Isolation
```yaml
# ✅ Tests ohne externe Abhängigkeiten
pytest tests/ -q -m "not integration and not mt5 and not rust_integration and not julia_integration"
```

### E4: Caching ohne Lock-File Hash
```yaml
# ❌ Falsch
key: ${{ runner.os }}-cargo

# ✅ Korrekt
key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
restore-keys: |
  ${{ runner.os }}-cargo-
```

### E5: Permissions nicht spezifiziert
```yaml
# ✅ Explizite Least-Privilege Permissions
permissions:
  contents: read
  # Nur wenn nötig:
  # pull-requests: write
  # packages: write
```

### E6: Matrix ohne fail-fast Control
```yaml
# ✅ Für vollständige Fehlerübersicht
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
```

### E7: PyO3/Rust-Benchmarks ohne Python-Setup (PR #24 Learning)
```yaml
# ❌ Falsch: Rust-Benchmarks mit PyO3 ohne Python-Installation
benchmarks:
  steps:
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo bench  # Fehler: PyO3 findet kein Python

# ✅ Korrekt: Python VOR Rust-Toolchain installieren
benchmarks:
  steps:
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo bench  # PyO3 findet Python 3.12
```

### E8: Actions-Versionen nicht konsistent (PR #24 Learning)
```yaml
# ❌ Inkonsistent: Mischung verschiedener Versionen
- uses: actions/cache@v4  # Veraltet
- uses: actions/cache@v5  # In anderem Job

# ✅ Konsistent: Alle auf gleicher Major-Version
- uses: actions/cache@v5
- uses: actions/checkout@v6
- uses: actions/setup-python@v5
```

### E9: Copilot-Setup ohne pytest-Marker (PR #24 Learning)
```yaml
# ❌ Falsch: Alle Tests ausführen (inkl. Integration)
- run: pytest -q  # Fehlschlag: MT5/Rust nicht verfügbar

# ✅ Korrekt: Integration-Tests exkludieren
- run: pytest -q -m "not integration and not mt5 and not rust_integration and not julia_integration"
```

### E10: Rust-Backend-Tests ohne skipif-Marker (PR #24 Wave 3)
```python
# ❌ Falsch: Test importiert omega_rust direkt ohne Guard
def test_rust_module_has_required_exports(self):
    import omega_rust  # ModuleNotFoundError in CI!
    assert hasattr(omega_rust, "EventEngineRust")

# ✅ Korrekt: pytest.mark.skipif für optionale FFI-Module
import pytest

try:
    import omega_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

@pytest.mark.skipif(not RUST_AVAILABLE, reason="omega_rust not built/installed")
def test_rust_module_has_required_exports(self):
    assert hasattr(omega_rust, "EventEngineRust")
```

**Validation:** Suche nach `import omega_rust` ohne vorherigen `skipif`-Guard:
```bash
grep -rn "import omega_rust" tests/ --include="*.py" | grep -v "skipif\|try:"
```

### E11: Clippy Warnings als Errors (PR #24 Wave 3)
```rust
// ❌ Falsch: Unbenutzte Felder in Rust-Struct
pub struct CrossSymbolEventEngineRust {
    symbols: Vec<String>,        // dead_code warning → error
    total_timestamps: usize,     // never read
}

// ✅ Option A: Felder verwenden oder entfernen
pub struct CrossSymbolEventEngineRust {
    // Nur Felder die wirklich verwendet werden
}

// ✅ Option B: Explizit erlauben wenn beabsichtigt
#[allow(dead_code)]
pub struct CrossSymbolEventEngineRust {
    symbols: Vec<String>,
    total_timestamps: usize,
}
```

**Häufige Clippy-Fehler in PyO3-Code:**
| Clippy-Lint | Ursache | Fix |
|-------------|---------|-----|
| `dead_code` | Felder nie gelesen | Entfernen oder `#[allow(dead_code)]` |
| `unused_self` | `&self` in Methode nicht verwendet | `fn method()` ohne self oder `#[allow(clippy::unused_self)]` |
| `field_reassign_with_default` | Felder nach `Default::default()` zuweisen | Struct-Initializer mit allen Feldern verwenden |

**Validation:**
```bash
cargo clippy --manifest-path src/rust_modules/omega_rust/Cargo.toml -- -D warnings
```

### E12: FFI-Modul nicht gebaut vor Python-Tests (PR #24 Wave 3)
```yaml
# ❌ Falsch: Python-Tests ohne omega_rust Build-Abhängigkeit
test:
  needs: [lint]  # FEHLT: rust-build nicht in needs!
  steps:
    - run: pytest  # Schlägt fehl: omega_rust nicht vorhanden

# ✅ Option A: Rust-Build als Abhängigkeit hinzufügen
test:
  needs: [lint, rust-build]
  steps:
    - name: Download omega_rust wheel
      uses: actions/download-artifact@v4
      with:
        name: omega-rust-wheel
    - run: pip install *.whl
    - run: pytest

# ✅ Option B: Tests mit skipif-Marker für fehlende Module
test:
  needs: [lint]
  steps:
    # Tests mit Rust-Markern werden übersprungen wenn omega_rust fehlt
    - run: pytest -m "not rust_backend"
```

**Entscheidungsmatrix:**
| Szenario | Empfehlung |
|----------|------------|
| Rust ist optional (Fallback auf Python) | Option B: skipif-Marker |
| Rust ist mandatory für Feature | Option A: Build-Abhängigkeit |
| Hybrid (manche Tests brauchen Rust) | Kombination: Test-Marker + separater rust-test Job |

---

## ⚡ Pre-Flight Code Quality Checks (KRITISCH - PR #24 Learning)

**BEVOR** Workflows analysiert werden, lokale Code-Qualität prüfen!

### Schritt 0: Lokale Formatierung verifizieren

```bash
# Python-Formatierung prüfen
black --check src/ tests/

# Import-Sortierung prüfen
isort --check src/ tests/

# Rust-Formatierung prüfen
cargo fmt --manifest-path src/rust_modules/omega_rust/Cargo.toml --check

# Rust-Linting prüfen (KRITISCH - PR #24 Wave 3)
cargo clippy --manifest-path src/rust_modules/omega_rust/Cargo.toml -- -D warnings
```

**Falls Fehler gefunden werden → ERST lokal fixen:**
```bash
black src/ tests/
isort src/ tests/
cargo fmt --manifest-path src/rust_modules/omega_rust/Cargo.toml
```

### Schritt 0.5: FFI-Test-Kompatibilität prüfen (NEU - PR #24 Wave 3)

```bash
# Finde Tests die omega_rust importieren ohne skipif-Guard
grep -rn "import omega_rust" tests/ --include="*.py" | grep -v "try:\|except\|skipif"

# Finde Tests die omega_julia importieren ohne skipif-Guard  
grep -rn "import omega_julia" tests/ --include="*.py" | grep -v "try:\|except\|skipif"

# Prüfe ob rust_backend Marker existiert für Rust-Tests
grep -rn "@pytest.mark.rust_backend\|pytest.mark.skipif.*RUST" tests/ --include="*.py"
```

**Falls ungeschützte FFI-Imports gefunden:**
1. Füge `try/except ImportError` Block am Modul-Anfang hinzu
2. Füge `@pytest.mark.skipif(not RUST_AVAILABLE, reason="...")` zu Tests hinzu
3. ODER verschiebe Tests in separaten `test_*_rust.py` mit eigenem CI-Job

### Häufige Formatierungsfehler (aus PR #24)

| Fehlertyp | Symptom | Lösung |
|-----------|---------|--------|
| Black assert-Formatierung | `assert x == y, (msg)` falsch formatiert | `black` reformatiert zu korrekter Form |
| Trailing Whitespace | Leerzeichen am Zeilenende | Black ersetzt durch Newlines |
| Import-Gruppierung | Imports nicht nach isort sortiert | `isort` gruppiert korrekt |
| Rust Method-Chaining | `.method1().method2()` auf einer Zeile | `cargo fmt` bricht korrekt um |

### Code-Qualitätsfehler vs. Workflow-Konfigurationsfehler

**Code-Qualitätsfehler** (lokaler Fix, kein YAML ändern):
- `black --check` / `isort --check` → Formatierung
- `cargo fmt --check` / `clippy` → Rust-Qualität
- `flake8` / `mypy` Warnungen → Code korrigieren

**Workflow-Konfigurationsfehler** (YAML ändern):
- Fehlende Setup-Schritte (Python vor Rust)
- Veraltete Action-Versionen
- Fehlende pytest-Marker
- Falsche Pfade/Trigger

---

## Korrektur-Checkliste

Für jeden gefundenen Fehler dokumentieren:

```markdown
### [PRIORITÄT] Fehler in `<workflow>.yml`

**Problem:** <Beschreibung>
**Zeile:** <Zeilennummer>
**Ursache:** <Technische Erklärung>

**Fix:**
```yaml
# Vorher
<alter Code>

# Nachher
<neuer Code>
```

**Validierung:** <Wie prüfen wir den Fix?>
```

---

## Validierungs-Schritte

Nach allen Korrekturen ausführen:

### 1. YAML-Syntax validieren
```bash
# yamllint installieren und prüfen
pip install yamllint
yamllint .github/workflows/
```

### 2. Referenzierte Pfade prüfen
```bash
# Pfade aus Workflows extrahieren und prüfen
grep -rh "working-directory:" .github/workflows/ | sort -u
grep -rh "path:" .github/workflows/ | sort -u
```

### 3. Actions-Verfügbarkeit prüfen
```bash
# Alle verwendeten Actions auflisten
grep -rh "uses:" .github/workflows/ | sed 's/.*uses: //' | sort -u
```

### 4. FFI-Test-Guard-Validierung (NEU - PR #24 Wave 3)
```bash
# Finde alle FFI-Imports in Tests
echo "=== omega_rust Imports ==="
grep -rn "import omega_rust\|from omega_rust" tests/ --include="*.py"

echo ""
echo "=== Geschützte Imports (OK) ==="
grep -B5 "import omega_rust" tests/ --include="*.py" | grep -E "try:|skipif|RUST_AVAILABLE"

echo ""
echo "=== UNGESCHÜTZTE Imports (FIX NEEDED) ==="
for file in $(grep -l "import omega_rust" tests/ --include="*.py"); do
  if ! grep -q "RUST_AVAILABLE\|skipif.*rust\|try:.*omega_rust" "$file"; then
    echo "⚠️  $file - Kein Import-Guard gefunden!"
  fi
done
```

### 5. Clippy-Validierung vor Push
```bash
# Muss OHNE Fehler durchlaufen für Rust-Workflows
cargo clippy --manifest-path src/rust_modules/omega_rust/Cargo.toml -- -D warnings 2>&1 | tee clippy_output.txt
if grep -q "error\[" clippy_output.txt; then
  echo "❌ Clippy-Fehler gefunden - Fix vor Push!"
  exit 1
fi
echo "✅ Clippy OK"
```

### 6. Lokaler Dry-Run (wenn möglich)
```bash
# Mit act (GitHub Actions lokal testen)
act -l  # Jobs auflisten
act push --dry-run  # Dry-run für push events
```

---

## Ausgabe-Format

### Zusammenfassung
```markdown
## Workflow-Analyse Ergebnis

### Geprüfte Workflows
- [ ] ci.yml
- [ ] rust-build.yml
- [ ] julia-tests.yml
- [ ] cross-platform-ci.yml
- [ ] benchmarks.yml
- [ ] release.yml

### Gefundene Probleme

| # | Workflow | Priorität | Problem | Status |
|---|----------|-----------|---------|--------|
| 1 | ci.yml | 🔴 Kritisch | ... | ⏳ Offen |
| 2 | rust-build.yml | 🟡 Wichtig | ... | ✅ Behoben |

### Durchgeführte Änderungen
1. `ci.yml`: <Beschreibung der Änderung>
2. ...

### Empfehlungen für zukünftige Verbesserungen
- ...

### Commit-Message Vorschlag
```
fix(ci): resolve workflow failures for [component]

- Fix 1: <Beschreibung>
- Fix 2: <Beschreibung>

Closes #<issue-number>
```
```

---

## Omega-spezifische Guardrails

### Nicht anfassen ohne explizite Bestätigung:
1. **Release-Workflow** (`release.yml`) – Deployment-kritisch
2. **Secrets-Referenzen** – Keine neuen `secrets.*` ohne Review
3. **Environment-Definitionen** – staging/production Guards
4. **MT5-Credentials** – Windows-spezifische Konfiguration

### Immer prüfen:
1. **`var/`-Pfade sind gitignored** – Runtime-State nicht in CI
2. **Backtest-Determinismus** – Keine `random()` ohne Seed in Tests
3. **Cross-Platform-Kompatibilität** – macOS/Linux müssen ohne MT5 laufen

---

## Quick-Reference Commands

```bash
# Workflow-Status prüfen
gh workflow list
gh run list --workflow=ci.yml --limit 5

# Fehlgeschlagenen Run analysieren
gh run view <run-id> --log-failed

# Workflow manuell triggern
gh workflow run cross-platform-ci.yml --field run_full_matrix=true

# Lokale YAML-Validierung
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# Alle Workflow-YAMLs validieren
for f in .github/workflows/*.yml; do python -c "import yaml; yaml.safe_load(open('$f'))" && echo "$f: OK"; done
```

---

## Gelernte Muster (Error → Root Cause → Fix)

| Workflow-Fehler | Root Cause | Fix |
|-----------------|------------|-----|
| `black --check` failed | Code nicht formatiert | `black src/ tests/` lokal |
| `cargo fmt --check` failed | Rust nicht formatiert | `cargo fmt` lokal |
| PyO3 `undefined symbol: _Py_DecRef` | Python nicht installiert vor Rust-Build | `setup-python@v5` VOR `rust-toolchain` |
| `pytest` failed in copilot-setup | Integration-Tests ohne Deps | `-m "not integration..."` Marker |
| Cache miss / restore error | Veraltete action Version | Alle auf `@v5` / `@v6` updaten |
| MT5 import error auf Linux | MT5 ist Windows-only | Tests mit `platform_system == "Windows"` oder pytest-Marker |
| **`ModuleNotFoundError: omega_rust`** (PR #24) | FFI-Modul nicht gebaut/installiert | `pytest.mark.skipif` Guard oder rust-build Abhängigkeit |
| **`cargo clippy` dead_code error** (PR #24) | Unbenutzte Felder/Variablen in Rust | Entfernen oder `#[allow(dead_code)]` |
| **`cargo clippy` unused_self** (PR #24) | `&self` nicht verwendet in Methode | `#[allow(clippy::unused_self)]` oder zu assoc fn |
| **`cargo clippy` field_reassign_with_default** (PR #24) | Feld-Zuweisung nach `Default::default()` | Struct-Initializer mit expliziten Feldern |
| **11 failed tests in CI** (PR #24) | Rust-Backend-Tests ohne skipif | Alle 11 Tests mit `@pytest.mark.skipif(not RUST_AVAILABLE)` |

---

## 🎯 PR #24 Wave 3 Learnings - Checkliste für Rust-Migrationen

Bei jeder Rust-FFI-Migration (Wave 1-4) diese Punkte prüfen:

### Vor dem PR:
- [ ] `cargo clippy -- -D warnings` lokal ausführen
- [ ] Alle neuen Tests mit `try/except ImportError` für FFI-Module
- [ ] `pytest.mark.skipif` für Tests die FFI-Modul benötigen
- [ ] Oder: Separater CI-Job mit `needs: [rust-build]`

### Im CI:
- [ ] Python-Setup VOR Rust-Toolchain in PyO3-Jobs
- [ ] Test-Marker: `-m "not rust_backend"` für Standard-Tests
- [ ] Oder: `needs: [rust-build]` + Artifact-Download für rust-enabled Tests

### Nach dem PR:
- [ ] Dokumentieren in `docs/WAVE_X_MIGRATION_LEARNINGS.md`
- [ ] Error-Pattern in `workflow-fix.prompt.md` ergänzen
- [ ] Golden-File-Tests für neue FFI-Funktionen

---

**Ziel:** Nach diesem Review sollen alle Workflows beim nächsten Push auf grün durchlaufen. Sei gründlich, beachte die Omega-spezifischen Anforderungen und übersehe nichts.
