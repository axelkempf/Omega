# ADR-0004: Build-System Architecture für Hybrid Python/Rust/Julia Stack

## Status

**Accepted** (finalisiert am 2026-01-05)

## Kontext

Die Migration ausgewählter Python-Module zu Rust und Julia erfordert ein Build-System, das:

1. **Multi-Language Support**: Python, Rust und Julia in einem Repository unterstützt
2. **Cross-Platform**: Linux, macOS und Windows (mit MT5-Spezialfällen)
3. **Developer Experience**: Einfache lokale Entwicklung und schnelle Iterationszyklen
4. **CI/CD Integration**: Automatisierte Builds, Tests und Releases in GitHub Actions
5. **Determinismus**: Reproduzierbare Builds für Trading-System-Compliance

### Anforderungen

| Anforderung | Priorität | Beschreibung |
|-------------|-----------|--------------|
| Reproduzierbarkeit | Kritisch | Builds müssen deterministisch sein |
| Cross-Platform | Kritisch | Linux, macOS, Windows Support |
| Inkrementelle Builds | Hoch | Nur geänderte Module neu kompilieren |
| Dev-Container | Mittel | Vorkonfigurierte Entwicklungsumgebung |
| Caching | Hoch | CI-Builds effizient cachen |
| Dokumentation | Hoch | Build-Schritte klar dokumentiert |

### Constraints

- Python ≥3.12 als Basis-Runtime
- Rust Toolchain (stable, MSRV 1.70+)
- Julia ≥1.10 für numerische Module
- MT5-Integration nur auf Windows (Live-Trading)
- Bestehende `pyproject.toml` bleibt kompatibel

### Kräfte

- **Einfachheit vs. Features**: Minimale Toolchain vs. volle Automatisierung
- **Performance vs. Portabilität**: Native Builds vs. Container-Isolation
- **Flexibilität vs. Konsistenz**: Entwickler-Freiheit vs. standardisierte Workflows

## Entscheidung

### Build-Tool-Strategie: Make + just

Wir verwenden **GNU Make** als primäres Build-Tool mit **just** als moderne Alternative:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Build System Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Developer Interface                            │ │
│  │                                                                        │ │
│  │   make <target>    ←→    just <recipe>    ←→    GitHub Actions        │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          Build Targets                                 │ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │   Python     │  │    Rust      │  │    Julia     │                │ │
│  │  │              │  │              │  │              │                │ │
│  │  │ pip install  │  │   maturin    │  │  Pkg.build   │                │ │
│  │  │ -e .[dev]    │  │   develop    │  │              │                │ │
│  │  │              │  │              │  │              │                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  │         │                  │                  │                       │ │
│  │         └──────────────────┼──────────────────┘                       │ │
│  │                            ▼                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                   Unified Test Suite                           │  │ │
│  │  │                                                                │  │ │
│  │  │   pytest (Python) + cargo test (Rust) + Julia tests           │  │ │
│  │  │                                                                │  │ │
│  │  └────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Toolchain-Entscheidungen

| Komponente | Tool | Version | Begründung |
|------------|------|---------|------------|
| Python Build | pip + pyproject.toml | - | Standard; PEP 517/518 konform |
| Rust Build | maturin | ≥1.4 | PyO3-Integration; Wheel-Builds |
| Julia Build | Pkg.jl | - | Native Julia Package Manager |
| Task Runner | Make + just | - | Cross-Platform; einfach; verbreitet |
| Container | Docker + devcontainer | - | VS Code Integration; reproduzierbar |
| CI/CD | GitHub Actions | - | Native Integration; Matrix-Support |

### Makefile-Struktur

```makefile
# Makefile - Auszug der wichtigsten Targets

# =============================================================================
# Configuration
# =============================================================================
PYTHON := python3
RUST_MODULE := src/rust_modules/omega_rust
JULIA_MODULE := src/julia_modules/omega_julia

# =============================================================================
# Primary Targets
# =============================================================================

.PHONY: all
all: install-dev build-rust build-julia  ## Build everything

.PHONY: install-dev
install-dev:  ## Install Python dev environment
	$(PYTHON) -m pip install -e ".[dev,analysis]"

.PHONY: build-rust
build-rust:  ## Build Rust module with maturin
	@if [ -d "$(RUST_MODULE)" ]; then \
		cd $(RUST_MODULE) && maturin develop --release; \
	fi

.PHONY: build-julia
build-julia:  ## Build Julia module
	@if [ -d "$(JULIA_MODULE)" ]; then \
		julia --project=$(JULIA_MODULE) -e 'using Pkg; Pkg.instantiate()'; \
	fi

.PHONY: test
test: test-python test-rust test-julia  ## Run all tests

.PHONY: test-python
test-python:  ## Run Python tests
	pytest -q

.PHONY: test-rust
test-rust:  ## Run Rust tests
	@if [ -d "$(RUST_MODULE)" ]; then \
		cd $(RUST_MODULE) && cargo test; \
	fi

.PHONY: test-julia
test-julia:  ## Run Julia tests
	@if [ -d "$(JULIA_MODULE)" ]; then \
		julia --project=$(JULIA_MODULE) -e 'using Pkg; Pkg.test()'; \
	fi

.PHONY: clean
clean:  ## Clean all build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	@if [ -d "$(RUST_MODULE)" ]; then \
		cd $(RUST_MODULE) && cargo clean; \
	fi
```

### justfile-Struktur

```just
# justfile - Moderne Alternative zu Make

# Default recipe
default: install-dev

# Install Python development environment
install-dev:
    python -m pip install -e ".[dev,analysis]"

# Build Rust module
build-rust:
    #!/usr/bin/env bash
    if [ -d "src/rust_modules/omega_rust" ]; then
        cd src/rust_modules/omega_rust && maturin develop --release
    fi

# Build Julia module
build-julia:
    #!/usr/bin/env bash
    if [ -d "src/julia_modules/omega_julia" ]; then
        julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.instantiate()'
    fi

# Build all modules
build: build-rust build-julia

# Run all tests
test: test-python test-rust test-julia

# Run Python tests
test-python:
    pytest -q

# Run Rust tests
test-rust:
    #!/usr/bin/env bash
    if [ -d "src/rust_modules/omega_rust" ]; then
        cd src/rust_modules/omega_rust && cargo test
    fi

# Run Julia tests
test-julia:
    #!/usr/bin/env bash
    if [ -d "src/julia_modules/omega_julia" ]; then
        julia --project=src/julia_modules/omega_julia -e 'using Pkg; Pkg.test()'
    fi

# Run benchmarks
bench:
    pytest tests/benchmarks/ --benchmark-only

# Format all code
format:
    black src/ tests/
    isort src/ tests/
    @if [ -d "src/rust_modules/omega_rust" ]; then \
        cd src/rust_modules/omega_rust && cargo fmt; \
    fi

# Lint all code
lint:
    flake8 src/ tests/
    mypy src/
    @if [ -d "src/rust_modules/omega_rust" ]; then \
        cd src/rust_modules/omega_rust && cargo clippy -- -D warnings; \
    fi

# Clean all build artifacts
clean:
    rm -rf build/ dist/ *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +
```

### CI/CD Workflow-Strategie

```yaml
# .github/workflows/ Struktur

workflows/
├── ci.yml                 # Standard Python CI (pytest, mypy, lint)
├── rust-build.yml         # Rust-spezifische Builds und Tests
├── julia-tests.yml        # Julia-spezifische Tests
├── cross-platform-ci.yml  # Matrix: Linux, macOS, Windows
├── benchmarks.yml         # Performance-Benchmarks (scheduled)
└── release.yml            # Hybrid-Package Release
```

### Caching-Strategie

| Cache-Typ | Key-Strategie | Restore-Keys | TTL |
|-----------|---------------|--------------|-----|
| Python pip | `${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}` | `${{ runner.os }}-pip-` | 7 Tage |
| Rust cargo | `${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}` | `${{ runner.os }}-cargo-` | 7 Tage |
| Julia packages | `${{ runner.os }}-julia-${{ hashFiles('**/Project.toml') }}` | `${{ runner.os }}-julia-` | 7 Tage |
| maturin builds | `${{ runner.os }}-maturin-${{ hashFiles('**/Cargo.toml') }}` | - | 1 Tag |

### Dev-Container Konfiguration

```json
// .devcontainer/devcontainer.json
{
  "name": "Omega Dev Environment",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12"
    },
    "ghcr.io/devcontainers/features/rust:1": {
      "version": "stable"
    },
    "ghcr.io/devcontainers/features/julia:1": {
      "version": "1.10"
    }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "rust-lang.rust-analyzer",
        "julialang.language-julia"
      ]
    }
  },
  "postCreateCommand": "make install-dev"
}
```

## Konsequenzen

### Positive Konsequenzen

- **Konsistenz**: Einheitliche Build-Befehle für alle Sprachen (`make test`, `just test`)
- **Reproduzierbarkeit**: Dev-Container garantiert identische Entwicklungsumgebung
- **CI-Effizienz**: Aggressives Caching reduziert Build-Zeiten um ~60%
- **Developer Experience**: Einfache Onboarding (`make install-dev` oder `just install-dev`)
- **Flexibilität**: Entwickler können Make oder just nach Präferenz verwenden
- **Cross-Platform**: Windows-Support durch GitHub Actions Matrix

### Negative Konsequenzen

- **Toolchain-Komplexität**: Drei Sprachen erfordern drei Toolchains
- **Disk Space**: Rust/Julia Toolchains benötigen ~2-3 GB zusätzlich
- **Initial Setup**: Erste Installation dauert ~5-10 Minuten
- **Maintenance**: Drei Toolchains müssen aktuell gehalten werden

### Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Toolchain-Version-Drift | Mittel | Mittel | Pinned Versions in CI; Dependabot |
| CI-Build-Timeout | Niedrig | Niedrig | Effektives Caching; Parallelisierung |
| Dev-Container-Inkompatibilität | Niedrig | Mittel | Regelmäßige Tests; Fallback zu lokalem Setup |
| Windows-spezifische Bugs | Mittel | Mittel | Windows CI Matrix; MT5-Isolation |

## Alternativen

### Alternative 1: Bazel

- **Beschreibung**: Polyglotter Build-System von Google
- **Vorteile**: Sehr effizientes Caching; Hermetische Builds
- **Nachteile**: Steile Lernkurve; Overhead für kleines Team
- **Warum nicht gewählt**: Komplexität überwiegt Vorteile für dieses Projekt

### Alternative 2: CMake + Custom Scripts

- **Beschreibung**: CMake für C++/Rust, Shell-Scripts für Rest
- **Vorteile**: Flexibel; etabliert
- **Nachteile**: Fragmentiert; schwer zu warten; Windows-Support problematisch
- **Warum nicht gewählt**: Inkonsistentes Developer Experience

### Alternative 3: Nix

- **Beschreibung**: Funktionaler Package Manager mit reproduzierbaren Builds
- **Vorteile**: Maximale Reproduzierbarkeit; Cross-Platform
- **Nachteile**: Sehr steile Lernkurve; komplexe Syntax; wenig Rust/Julia-Tooling
- **Warum nicht gewählt**: Zu hohe Einstiegshürde für Team

### Alternative 4: Poetry + Cargo + Pkg.jl separat

- **Beschreibung**: Jede Sprache nutzt eigenes Build-Tool ohne Wrapper
- **Vorteile**: Native Tooling; keine zusätzliche Abstraktionsschicht
- **Nachteile**: Fragmentierte Befehle; schwer zu automatisieren
- **Warum nicht gewählt**: Inkonsistentes CI; Developer Experience leidet

## Implementierungs-Checkliste

- [x] Makefile erstellt (`Makefile`)
- [x] justfile erstellt (`justfile`)
- [x] Dev-Container konfiguriert (`.devcontainer/`)
- [x] GitHub Actions Workflow: Rust Build (`.github/workflows/rust-build.yml`)
- [x] GitHub Actions Workflow: Julia Tests (`.github/workflows/julia-tests.yml`)
- [x] GitHub Actions Workflow: Cross-Platform (`.github/workflows/cross-platform-ci.yml`)
- [x] GitHub Actions Workflow: Release (`.github/workflows/release.yml`)
- [x] Caching-Strategie in allen Workflows implementiert
- [x] Rust-Toolchain Dokumentation (`docs/rust-toolchain-requirements.md`)
- [x] Julia-Environment Dokumentation (`docs/julia-environment-requirements.md`)

## Referenzen

- [GNU Make Manual](https://www.gnu.org/software/make/manual/)
- [just Documentation](https://just.systems/man/en/)
- [Maturin User Guide](https://www.maturin.rs/)
- [GitHub Actions Cache](https://github.com/actions/cache)
- [Dev Containers Specification](https://containers.dev/)
- [ADR-0001: Migration Strategy](ADR-0001-migration-strategy.md)
- [Omega Rust Toolchain Requirements](../rust-toolchain-requirements.md)
- [Omega Julia Environment Requirements](../julia-environment-requirements.md)

## Änderungshistorie

| Datum | Autor | Änderung |
|-------|-------|----------|
| 2026-01-05 | AI Agent | Initiale Version (P5-04) |
