# 04 - Pre-Commit Validation

> Automatische Validierung von Agent-generierten Änderungen

**Status:** ✅ Abgeschlossen
**Priorität:** Mittel
**Komplexität:** Mittel
**Geschätzter Aufwand:** 4-6 Stunden
**Abgeschlossen:** 2025-01-20

---

## Implementation Summary

### Erstellte Dateien

- `scripts/hooks/__init__.py` - Modul-Definition
- `scripts/hooks/pytest_changed.py` - Test-Runner für geänderte Dateien
- `scripts/hooks/breaking_change_check.py` - API-Breaking-Change-Erkennung
- `scripts/hooks/trading_safety_check.py` - Trading-Sicherheitsprüfung
- `scripts/hooks/agent_output_validation.py` - Code-Qualitätsvalidierung (non-blocking)
- `scripts/hooks/architecture_check.py` - Architektur-Konsistenzprüfung (non-blocking)

### Geänderte Dateien

- `.pre-commit-config.yaml` - 5 neue lokale Hooks hinzugefügt
- `docs/PRE_COMMIT_HOOKS.md` - Vollständige Dokumentation erstellt

### Hook-Übersicht

| Hook | Blocking | Bypass |
|------|----------|--------|
| `pytest-changed` | ✅ | `--no-verify` |
| `breaking-change-check` | ✅ | `BREAKING:` in Commit-Message |
| `trading-safety-check` | ✅ | `SAFETY-REVIEWED:` oder `# noqa: trading-safety` |
| `agent-output-validation` | ❌ | - |
| `architecture-check` | ❌ | - |

---

## Objective

Implementiere automatische Validierung für alle Code-Änderungen (egal ob von Menschen oder Agents):
- Tests müssen grün sein
- Code-Style eingehalten
- Security-Checks bestanden
- Keine Breaking Changes ohne Flag

---

## Current State

### Vorhandene Checks

Die `.pre-commit-config.yaml` enthält bereits:

```yaml
repos:
  - repo: https://github.com/psf/black
    # ...
  - repo: https://github.com/pycqa/isort
    # ...
  - repo: https://github.com/pycqa/flake8
    # ...
  - repo: https://github.com/pre-commit/mirrors-mypy
    # ...
  - repo: https://github.com/PyCQA/bandit
    # ...
```

### Was fehlt

1. **Kein Test-Runner** - Tests werden nicht automatisch ausgeführt
2. **Kein Breaking Change Detection** - API-Änderungen werden nicht erkannt
3. **Keine Agent-spezifischen Checks** - z.B. "Hat der richtige Agent das gemacht?"
4. **Kein Output-Validation** - Erfüllt der Code die Acceptance Criteria?

---

## Target State

### Erweiterte Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml (erweitert)

repos:
  # --- Existing Hooks ---
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
        args: ["--config-file=pyproject.toml"]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.9
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

  # --- NEW: Agent Validation Hooks ---

  # Test Runner (nur geänderte Module)
  - repo: local
    hooks:
      - id: pytest-changed
        name: pytest (changed files only)
        entry: python scripts/hooks/pytest_changed.py
        language: python
        types: [python]
        pass_filenames: true
        stages: [pre-commit]
        additional_dependencies: [pytest]

  # Breaking Change Detection
  - repo: local
    hooks:
      - id: breaking-change-check
        name: Check for breaking changes
        entry: python scripts/hooks/breaking_change_check.py
        language: python
        types: [python]
        pass_filenames: true
        stages: [pre-commit]

  # Trading Safety Invariants
  - repo: local
    hooks:
      - id: trading-safety-check
        name: Trading safety invariants
        entry: python scripts/hooks/trading_safety_check.py
        language: python
        files: ^src/(hf_engine|strategies)/
        pass_filenames: true
        stages: [pre-commit]

  # Agent Output Validation
  - repo: local
    hooks:
      - id: agent-output-validation
        name: Validate agent outputs
        entry: python scripts/hooks/agent_output_validation.py
        language: python
        types: [python]
        pass_filenames: true
        stages: [pre-commit]
        verbose: true

  # Architecture Consistency
  - repo: local
    hooks:
      - id: architecture-check
        name: Check architecture.md consistency
        entry: python scripts/hooks/architecture_check.py
        language: python
        files: ^src/
        pass_filenames: false
        stages: [pre-commit]
```

---

## Implementation Plan

### Schritt 1: Hook Scripts erstellen

Erstelle `scripts/hooks/` Ordner:

```
scripts/hooks/
├── __init__.py
├── pytest_changed.py
├── breaking_change_check.py
├── trading_safety_check.py
├── agent_output_validation.py
└── architecture_check.py
```

#### `pytest_changed.py`

```python
#!/usr/bin/env python3
"""Run pytest only for changed files."""

import subprocess
import sys
from pathlib import Path


def find_test_file(source_file: Path) -> Path | None:
    """Find corresponding test file for a source file."""

    # src/backtest_engine/core/portfolio.py -> tests/test_portfolio.py
    name = source_file.stem
    test_candidates = [
        Path(f"tests/test_{name}.py"),
        Path(f"tests/{source_file.parent.name}/test_{name}.py"),
    ]

    for candidate in test_candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    changed_files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    if not changed_files:
        return 0

    # Collect test files
    test_files: set[Path] = set()

    for source_file in changed_files:
        # If it's already a test file, add it
        if source_file.name.startswith("test_"):
            test_files.add(source_file)
            continue

        # Find corresponding test file
        test_file = find_test_file(source_file)
        if test_file:
            test_files.add(test_file)

    if not test_files:
        print("No test files found for changed files, skipping...")
        return 0

    # Run pytest
    cmd = ["pytest", "-q", "--tb=short"] + [str(f) for f in test_files]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
```

#### `breaking_change_check.py`

```python
#!/usr/bin/env python3
"""Detect breaking changes in public APIs."""

import ast
import subprocess
import sys
from pathlib import Path


class APIExtractor(ast.NodeVisitor):
    """Extract public API signatures from Python files."""

    def __init__(self):
        self.functions: dict[str, str] = {}
        self.classes: dict[str, list[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not node.name.startswith("_"):
            args = [a.arg for a in node.args.args]
            self.functions[node.name] = f"{node.name}({', '.join(args)})"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not node.name.startswith("_"):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    args = [a.arg for a in item.args.args]
                    methods.append(f"{item.name}({', '.join(args)})")
            self.classes[node.name] = methods
        self.generic_visit(node)


def get_api(file_path: Path) -> tuple[dict, dict]:
    """Extract API from a Python file."""

    try:
        tree = ast.parse(file_path.read_text())
        extractor = APIExtractor()
        extractor.visit(tree)
        return extractor.functions, extractor.classes
    except SyntaxError:
        return {}, {}


def get_previous_version(file_path: Path) -> str | None:
    """Get the previous version of a file from git."""

    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{file_path}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def main() -> int:
    changed_files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    breaking_changes: list[str] = []

    for file_path in changed_files:
        if not file_path.exists():
            continue

        # Get current API
        current_funcs, current_classes = get_api(file_path)

        # Get previous API
        prev_content = get_previous_version(file_path)
        if not prev_content:
            continue  # New file, no breaking change possible

        # Parse previous version
        try:
            prev_tree = ast.parse(prev_content)
            prev_extractor = APIExtractor()
            prev_extractor.visit(prev_tree)
            prev_funcs = prev_extractor.functions
            prev_classes = prev_extractor.classes
        except SyntaxError:
            continue

        # Compare functions
        for name, signature in prev_funcs.items():
            if name not in current_funcs:
                breaking_changes.append(f"REMOVED: {file_path}:{name}")
            elif current_funcs[name] != signature:
                breaking_changes.append(
                    f"CHANGED: {file_path}:{name}\n"
                    f"  Was: {signature}\n"
                    f"  Now: {current_funcs[name]}"
                )

        # Compare classes
        for class_name, methods in prev_classes.items():
            if class_name not in current_classes:
                breaking_changes.append(f"REMOVED CLASS: {file_path}:{class_name}")
            else:
                for method in methods:
                    method_name = method.split("(")[0]
                    if not any(m.startswith(method_name) for m in current_classes[class_name]):
                        breaking_changes.append(
                            f"REMOVED METHOD: {file_path}:{class_name}.{method_name}"
                        )

    if breaking_changes:
        print("⚠️  POTENTIAL BREAKING CHANGES DETECTED:")
        print("-" * 50)
        for change in breaking_changes:
            print(change)
        print("-" * 50)
        print("If intentional, add 'BREAKING:' to your commit message.")

        # Check if commit message contains BREAKING:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%s", "HEAD"],
                capture_output=True,
                text=True
            )
            if "BREAKING:" in result.stdout:
                print("✅ Breaking change acknowledged in commit message.")
                return 0
        except Exception:
            pass

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### `trading_safety_check.py`

```python
#!/usr/bin/env python3
"""Check trading safety invariants."""

import re
import sys
from pathlib import Path

# Patterns that indicate potentially unsafe changes
UNSAFE_PATTERNS = [
    (r"magic_number\s*=\s*\d+", "Hardcoded magic_number - should be from config"),
    (r"lot_size\s*=\s*\d+\.?\d*", "Hardcoded lot_size - should be calculated"),
    (r"\.send_order\(.*\)", "Direct order sending - ensure risk checks are in place"),
    (r"position\.close\(\)", "Position closing - ensure proper logging"),
    (r"time\.sleep\(", "time.sleep in trading code - may cause missed signals"),
    (r"except:\s*$", "Bare except - may hide trading errors"),
    (r"except:\s*pass", "Silent exception - may hide trading errors"),
]

# Files that require extra scrutiny
CRITICAL_FILES = [
    "execution_engine.py",
    "risk_manager.py",
    "lot_size_calculator.py",
    "mt5_adapter.py",
]


def check_file(file_path: Path) -> list[str]:
    """Check a file for trading safety issues."""

    issues: list[str] = []
    content = file_path.read_text()
    lines = content.split("\n")

    is_critical = file_path.name in CRITICAL_FILES

    for i, line in enumerate(lines, 1):
        for pattern, message in UNSAFE_PATTERNS:
            if re.search(pattern, line):
                severity = "🔴 CRITICAL" if is_critical else "🟡 WARNING"
                issues.append(f"{severity} {file_path}:{i}: {message}")
                issues.append(f"    {line.strip()}")

    return issues


def main() -> int:
    changed_files = [
        Path(f) for f in sys.argv[1:]
        if f.endswith(".py") and ("hf_engine" in f or "strategies" in f)
    ]

    if not changed_files:
        return 0

    all_issues: list[str] = []

    for file_path in changed_files:
        if file_path.exists():
            issues = check_file(file_path)
            all_issues.extend(issues)

    if all_issues:
        print("⚠️  TRADING SAFETY CHECK:")
        print("-" * 50)
        for issue in all_issues:
            print(issue)
        print("-" * 50)
        print("Review these issues before committing.")
        print("If intentional, add 'SAFETY-REVIEWED:' to your commit message.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### `agent_output_validation.py`

```python
#!/usr/bin/env python3
"""Validate that agent outputs meet quality standards."""

import re
import sys
from pathlib import Path

# Quality checks for agent-generated code
QUALITY_CHECKS = [
    # Type hints required
    (
        r"def\s+\w+\([^)]*\)\s*:",  # Function without return type
        r"def\s+\w+\([^)]*\)\s*->",  # Function with return type
        "Missing return type hint"
    ),
    # Docstrings for public functions
    (
        r"def\s+(?!_)\w+\([^)]*\).*:\n\s*(?!\"\"\")",  # Public function without docstring
        None,
        "Missing docstring for public function"
    ),
]


def check_file(file_path: Path) -> list[str]:
    """Check a file for quality issues."""

    issues: list[str] = []
    content = file_path.read_text()

    # Check for missing type hints on functions
    func_pattern = r"def\s+(\w+)\([^)]*\)\s*:"
    typed_pattern = r"def\s+\w+\([^)]*\)\s*->"

    functions = re.findall(func_pattern, content)
    typed_functions = len(re.findall(typed_pattern, content))

    if functions and typed_functions < len(functions) * 0.8:
        issues.append(
            f"⚠️  {file_path}: Only {typed_functions}/{len(functions)} "
            f"functions have return type hints"
        )

    return issues


def main() -> int:
    changed_files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    all_issues: list[str] = []

    for file_path in changed_files:
        if file_path.exists() and not file_path.name.startswith("test_"):
            issues = check_file(file_path)
            all_issues.extend(issues)

    if all_issues:
        print("📋 AGENT OUTPUT VALIDATION:")
        print("-" * 50)
        for issue in all_issues:
            print(issue)
        print("-" * 50)
        return 1

    print("✅ Agent output validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### `architecture_check.py`

```python
#!/usr/bin/env python3
"""Check that architecture.md is up to date."""

import subprocess
import sys
from pathlib import Path


def get_src_structure() -> set[str]:
    """Get current src/ directory structure."""

    src_path = Path("src")
    if not src_path.exists():
        return set()

    structure = set()
    for path in src_path.rglob("*.py"):
        if "__pycache__" not in str(path):
            # Get relative path from src/
            rel_path = path.relative_to(src_path)
            # Add directory structure
            for parent in rel_path.parents:
                if parent != Path("."):
                    structure.add(str(parent) + "/")
            structure.add(str(rel_path))

    return structure


def get_architecture_structure() -> set[str]:
    """Extract structure documented in architecture.md."""

    arch_path = Path("architecture.md")
    if not arch_path.exists():
        return set()

    content = arch_path.read_text()
    structure = set()

    # Simple extraction - look for file/directory mentions
    import re

    # Match patterns like `- `file.py`` or `- `directory/``
    pattern = r"`([^`]+\.py)`|`([^`]+/)`"
    matches = re.findall(pattern, content)

    for py_file, directory in matches:
        if py_file:
            structure.add(py_file)
        if directory:
            structure.add(directory)

    return structure


def main() -> int:
    # Check if any src/ files were changed
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", "src/"],
        capture_output=True,
        text=True
    )

    if not result.stdout.strip():
        # No src/ changes, skip check
        return 0

    # Get structures
    actual = get_src_structure()
    documented = get_architecture_structure()

    # Find new files not in architecture.md
    # (Simplified check - just warn about new directories)
    actual_dirs = {p for p in actual if p.endswith("/")}
    documented_dirs = {p for p in documented if p.endswith("/")}

    new_dirs = actual_dirs - documented_dirs

    if new_dirs:
        print("⚠️  ARCHITECTURE CHECK:")
        print("-" * 50)
        print("New directories not documented in architecture.md:")
        for d in sorted(new_dirs):
            print(f"  - {d}")
        print("-" * 50)
        print("Consider updating architecture.md")
        # Warning only, don't block

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Schritt 2: Pre-commit config aktualisieren

Aktualisiere `.pre-commit-config.yaml` mit den neuen Hooks.

### Schritt 3: Dokumentation

Erstelle `docs/PRE_COMMIT_HOOKS.md` mit Erklärungen zu allen Hooks.

---

## Acceptance Criteria

- [ ] Alle 5 neuen Hooks sind implementiert
- [ ] `.pre-commit-config.yaml` ist aktualisiert
- [ ] `pre-commit run -a` läuft ohne Fehler
- [ ] Hooks blockieren bei echten Problemen
- [ ] Hooks lassen legitime Änderungen durch
- [ ] Dokumentation für alle Hooks vorhanden

---

## Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Zu restriktiv | Hoch | Mittel | Bypass mit BREAKING: oder SAFETY-REVIEWED: |
| Zu langsam | Mittel | Niedrig | Nur geänderte Dateien prüfen |
| False Positives | Mittel | Niedrig | Whitelist für bekannte Patterns |

---

## Testing

```bash
# Test einzelner Hook
python scripts/hooks/trading_safety_check.py src/hf_engine/core/execution/execution_engine.py

# Test aller Hooks
pre-commit run -a --verbose

# Test mit spezifischem Hook
pre-commit run pytest-changed --all-files
```
