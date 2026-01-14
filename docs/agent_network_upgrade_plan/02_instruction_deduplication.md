# 02 - Instruction Deduplication

> Single Source of Truth für alle KI-Agent-Instruktionen

**Status:** 🟢 Abgeschlossen
**Priorität:** Hoch
**Komplexität:** Niedrig
**Geschätzter Aufwand:** 3-5 Stunden
**Abgeschlossen:** 2025-01-XX

---

## Implementierte Änderungen

### Erstellte Dateien

| Datei | Zweck |
|-------|-------|
| `.github/instructions/_core/python-standards.instructions.md` | Kanonische Python-Standards (PEP 8, Type Hints, Imports) |
| `.github/instructions/_core/rust-standards.instructions.md` | Kanonische Rust-Standards (RFC 430, Ownership, clippy) |
| `.github/instructions/_core/security-standards.instructions.md` | OWASP-basierte Security-Standards |
| `.github/instructions/_core/testing-standards.instructions.md` | pytest-Standards (AAA, Determinismus, Coverage) |
| `.github/instructions/_core/error-handling.instructions.md` | Exception-Handling Patterns |
| `.github/instructions/_domain/trading-safety.instructions.md` | Trading-spezifische Sicherheitsregeln |
| `scripts/check_instruction_redundancy.py` | CI-Tool zur Redundanz-Erkennung |

### Refaktorierte Dateien

| Datei | Änderung |
|-------|----------|
| `codexer.instructions.md` | Redundante Python/Security-Abschnitte entfernt, Core-Referenzen hinzugefügt |
| `copilot-instructions.md` | Core Standards Referenztabelle hinzugefügt |

---

## Objective

Konsolidiere redundante Instruktionen zu einer **Single Source of Truth** pro Thema:
- Keine widersprüchlichen Regeln
- Einfachere Wartung
- Konsistentes Verhalten aller Agenten

---

## Current State

### Identifizierte Redundanzen

#### 1. Python-Standards (3x definiert)

| Datei | Inhalt |
|-------|--------|
| `copilot-instructions.md` | Python ≥3.12, Type Hints, PEP 8 |
| `codexer.instructions.md` | PEP 8, Type Hints, 79 char lines |
| `CLAUDE.md` | Python ≥3.12, Type Hints |

**Problem:** Leicht unterschiedliche Formulierungen, gleiche Intention.

#### 2. Security-Regeln (3x definiert)

| Datei | Inhalt |
|-------|--------|
| `security-and-owasp.instructions.md` | OWASP Top 10, Input Validation |
| `code-review-generic.instructions.md` | SQL Injection, Secrets |
| `codexer.instructions.md` | Input Sanitization, Secrets |

**Problem:** Überlappende Regeln, unterschiedliche Detailtiefe.

#### 3. Error Handling (2x definiert)

| Datei | Inhalt |
|-------|--------|
| `codexer.instructions.md` | Specific Exceptions, No silent failures |
| `code-review-generic.instructions.md` | Proper error handling, Fail fast |

**Problem:** Gleiche Regeln, unterschiedliche Formulierung.

#### 4. Testing Standards (2x definiert)

| Datei | Inhalt |
|-------|--------|
| `copilot-instructions.md` | pytest, deterministische Tests, Mocking |
| `code-review-generic.instructions.md` | Coverage, Test Names, Edge Cases |

**Problem:** Ergänzende aber verstreute Informationen.

---

## Target State

### Neue Struktur

```
.github/
├── instructions/
│   ├── _core/                          # Basis-Standards (werden referenziert)
│   │   ├── python-standards.instructions.md
│   │   ├── rust-standards.instructions.md
│   │   ├── security-standards.instructions.md
│   │   ├── testing-standards.instructions.md
│   │   └── error-handling.instructions.md
│   │
│   ├── _roles/                         # Rollen-spezifisch (referenzieren _core)
│   │   ├── architect.instructions.md
│   │   ├── implementer.instructions.md
│   │   ├── reviewer.instructions.md
│   │   ├── tester.instructions.md
│   │   └── researcher.instructions.md
│   │
│   └── _domain/                        # Domain-spezifisch
│       ├── ffi-boundaries.instructions.md
│       ├── trading-safety.instructions.md
│       └── ...
│
├── copilot-instructions.md             # Entry Point (referenziert _core + _roles)
└── prompts/                            # Task-spezifische Prompts (unverändert)
```

### Referenz-Mechanismus

Jede Instruktionsdatei verwendet Markdown-Links zu den Core-Standards:

```markdown
# Implementer Instructions

## Python Standards
See: [Python Standards](instructions/_core/python-standards.instructions.md)

## Security Requirements
See: [Security Standards](instructions/_core/security-standards.instructions.md)

## Role-Specific Rules
[... rolle-spezifische Ergänzungen ...]
```

---

## Implementation Plan

### Schritt 1: Core-Standards extrahieren

Erstelle konsolidierte Basis-Dokumente:

#### `_core/python-standards.instructions.md`

```markdown
---
description: 'Canonical Python coding standards for Omega project'
applyTo: '**/*.py'
---

# Python Standards

## Version & Compatibility
- **Required:** Python ≥3.12
- **Type Hints:** Mandatory for all public functions
- **Union Syntax:** Use `X | Y` instead of `Union[X, Y]`

## Style Guide
- **Standard:** PEP 8
- **Line Length:** 88 characters (Black default)
- **Indentation:** 4 spaces
- **Naming:**
  - `snake_case` for functions and variables
  - `CamelCase` for classes
  - `UPPER_CASE` for constants

## Type Hints
- Use `from __future__ import annotations` for forward references
- Prefer `TypedDict` over plain `dict` for structured data
- Use `Literal` for fixed string values
- Use `Final` for constants

## Imports
- Standard library first
- Third-party second
- Local imports third
- Use `isort` with profile `black`

## Docstrings
- Google Style for public functions/classes
- Required for: public APIs, complex logic
- Not required for: obvious one-liners, private helpers

## Common Patterns
```python
# Defensive optional imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Type-safe config loading
from typing import TypedDict

class StrategyConfig(TypedDict):
    symbol: str
    timeframe: str
    magic_number: int
```
```

#### `_core/security-standards.instructions.md`

```markdown
---
description: 'Security standards based on OWASP guidelines'
applyTo: '**'
---

# Security Standards

## OWASP Top 10 Compliance

### 1. Injection Prevention
- **SQL:** Always use parameterized queries
- **Command:** Never use `shell=True` with user input
- **Template:** Escape all user-provided content

### 2. Secrets Management
- **Never commit:** Passwords, API keys, tokens
- **Use:** Environment variables via `python-dotenv`
- **Placeholder:** Document required ENV vars in README

### 3. Input Validation
- **Validate at boundaries:** User input, external APIs
- **Fail fast:** Raise exceptions early
- **Sanitize:** HTML, SQL, shell characters

### 4. Authentication & Authorization
- **Check permissions** before every protected operation
- **Use established libraries** (no custom crypto)
- **Log security events** (failed logins, permission denials)

## Code Review Security Checklist
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] Parameterized queries used
- [ ] Error messages don't leak internal details
- [ ] Dependencies are up-to-date
```

#### `_core/testing-standards.instructions.md`

```markdown
---
description: 'Testing standards for Omega project'
applyTo: '**/test_*.py'
---

# Testing Standards

## Framework
- **Primary:** pytest ≥7.4
- **Coverage:** pytest-cov
- **Mocking:** unittest.mock or pytest-mock

## Test Requirements

### Naming
- Files: `test_<module>.py`
- Functions: `test_<function>_<scenario>`
- Example: `test_calculate_lot_size_with_zero_balance`

### Structure (AAA Pattern)
```python
def test_example():
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value
```

### Determinism
- **Fixed seeds:** Use `random.seed(42)` or `np.random.seed(42)`
- **No network calls:** Mock all external APIs
- **No time dependencies:** Mock `datetime.now()`

### MT5 Handling
- **Never require MT5** in tests
- **Mock MT5 calls** with fixtures
- **Skip tests** if MT5 not available:
```python
import pytest

MT5_AVAILABLE = False
try:
    import MetaTrader5
    MT5_AVAILABLE = True
except ImportError:
    pass

@pytest.mark.skipif(not MT5_AVAILABLE, reason="MT5 not available")
def test_mt5_specific_feature():
    ...
```

### Coverage Requirements
- **Core paths:** ≥80% coverage
- **New code:** Must include tests
- **Bug fixes:** Must include regression test

## Anti-Patterns
- ❌ `assert True` - Always passes
- ❌ `except: pass` - Silent failures
- ❌ Tests depending on execution order
- ❌ Tests modifying global state
```

### Schritt 2: Bestehende Dateien refactoren

Aktualisiere `copilot-instructions.md`:

```markdown
## AI Coding Agent Instructions (institutional baseline)

[... existing intro ...]

## Core Standards

This project follows centralized standards. For details, see:

- **Python:** [Python Standards](_core/python-standards.instructions.md)
- **Security:** [Security Standards](_core/security-standards.instructions.md)
- **Testing:** [Testing Standards](_core/testing-standards.instructions.md)
- **Error Handling:** [Error Handling](_core/error-handling.instructions.md)

## Project-Specific Rules

[... nur projekt-spezifische Ergänzungen ...]
```

### Schritt 3: codexer.instructions.md refactoren

Entferne redundante Abschnitte, füge Referenzen hinzu:

```markdown
# Codexer Instructions

## Core Standards
This agent follows all [Core Standards](instructions/_core/).

## Role: Researcher
Primary responsibility: Library research and documentation analysis.

## Research-Specific Guidelines
[... nur recherche-spezifische Regeln ...]
```

### Schritt 4: Redundanz-Check

Erstelle ein Script das Duplikate findet:

```python
#!/usr/bin/env python3
"""Check for instruction redundancy."""

from pathlib import Path
import re

INSTRUCTION_DIR = Path(".github/instructions")
KEYWORDS = ["PEP 8", "Type Hints", "OWASP", "SQL Injection", "pytest"]

def find_keyword_occurrences():
    results = {}
    for keyword in KEYWORDS:
        results[keyword] = []
        for file in INSTRUCTION_DIR.rglob("*.md"):
            content = file.read_text()
            if keyword.lower() in content.lower():
                results[keyword].append(file.name)

    for keyword, files in results.items():
        if len(files) > 1:
            print(f"⚠️  '{keyword}' found in {len(files)} files: {files}")

if __name__ == "__main__":
    find_keyword_occurrences()
```

---

## Migration Mapping

| Alte Datei | Redundanter Inhalt | Ziel |
|------------|-------------------|------|
| `copilot-instructions.md` | Python Standards | `_core/python-standards.md` |
| `copilot-instructions.md` | Testing Rules | `_core/testing-standards.md` |
| `codexer.instructions.md` | PEP 8, Type Hints | Referenz zu `_core/` |
| `codexer.instructions.md` | Security Rules | Referenz zu `_core/` |
| `code-review-generic.instructions.md` | Security Checklist | Referenz zu `_core/` |
| `code-review-generic.instructions.md` | Testing Checklist | Referenz zu `_core/` |

---

## Acceptance Criteria

- [ ] `_core/` Ordner mit konsolidierten Standards existiert
- [ ] Keine doppelten Regeln in verschiedenen Dateien
- [ ] Alle bestehenden Instruktionen referenzieren `_core/`
- [ ] Redundanz-Check Script läuft ohne Warnings
- [ ] Alle Agenten verhalten sich gleich (Smoke Test)

---

## Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Agenten finden Referenzen nicht | Mittel | Hoch | Inline-Kopie als Fallback |
| Inkonsistenz während Migration | Niedrig | Mittel | Atomare Commits |
| Verhalten ändert sich | Niedrig | Hoch | Before/After Tests |

---

## Rollback Plan

Falls Probleme auftreten:
1. Git revert des Refactoring-Commits
2. Alte Struktur ist wiederhergestellt
3. Keine Datenverluste möglich (nur Reorganisation)
