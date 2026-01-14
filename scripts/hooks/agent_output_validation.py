#!/usr/bin/env python3
"""Validate that agent outputs meet quality standards.

This hook checks agent-generated code for:
- Type hint coverage (>= 80% of functions should have return types)
- Docstring presence for public functions
- Import organization
- Basic code quality metrics
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityMetrics:
    """Quality metrics for a Python file."""

    total_functions: int = 0
    typed_functions: int = 0
    documented_functions: int = 0
    public_functions: int = 0
    issues: list[str] | None = None

    def __post_init__(self) -> None:
        if self.issues is None:
            self.issues = []

    @property
    def type_coverage(self) -> float:
        """Calculate type hint coverage percentage."""
        if self.total_functions == 0:
            return 100.0
        return (self.typed_functions / self.total_functions) * 100

    @property
    def doc_coverage(self) -> float:
        """Calculate docstring coverage for public functions."""
        if self.public_functions == 0:
            return 100.0
        return (self.documented_functions / self.public_functions) * 100


class QualityChecker(ast.NodeVisitor):
    """AST visitor to check code quality metrics."""

    def __init__(self) -> None:
        self.metrics = QualityMetrics()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function definitions."""
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze async function definitions."""
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Check a function for quality issues."""
        self.metrics.total_functions += 1

        # Check for return type annotation
        if node.returns is not None:
            self.metrics.typed_functions += 1

        # Check if public (not starting with _)
        is_public = not node.name.startswith("_")
        if is_public:
            self.metrics.public_functions += 1

            # Check for docstring
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                self.metrics.documented_functions += 1


def check_file(file_path: Path) -> QualityMetrics:
    """Check a file for quality issues.

    Args:
        file_path: Path to the Python file.

    Returns:
        QualityMetrics with analysis results.
    """
    metrics = QualityMetrics()

    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except (SyntaxError, OSError) as e:
        metrics.issues.append(f"Could not parse {file_path}: {e}")
        return metrics

    checker = QualityChecker()
    checker.visit(tree)
    metrics = checker.metrics

    # Check thresholds
    if metrics.total_functions > 0:
        if metrics.type_coverage < 80:
            metrics.issues.append(
                f"Type hint coverage: {metrics.typed_functions}/{metrics.total_functions} "
                f"({metrics.type_coverage:.0f}%) - should be >= 80%"
            )

        if metrics.public_functions > 0 and metrics.doc_coverage < 70:
            metrics.issues.append(
                f"Docstring coverage: {metrics.documented_functions}/{metrics.public_functions} "
                f"public functions ({metrics.doc_coverage:.0f}%) - should be >= 70%"
            )

    # Check for common issues
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        # Check for TODO without ticket reference
        if re.search(r"#\s*TODO(?!.*[A-Z]+-\d+)", line):
            metrics.issues.append(
                f"Line {i}: TODO without ticket reference (use TODO(TICKET-123))"
            )

        # Check for print statements (should use logging)
        if re.search(r"^\s*print\(", line) and "test" not in str(file_path).lower():
            metrics.issues.append(
                f"Line {i}: print() statement - consider using logging"
            )

    return metrics


def main() -> int:
    """Main entry point for the hook."""
    changed_files = [
        Path(f)
        for f in sys.argv[1:]
        if f.endswith(".py") and not f.startswith("test_")
    ]

    # Skip test files and __init__.py
    changed_files = [
        f
        for f in changed_files
        if not f.name.startswith("test_")
        and f.name != "__init__.py"
        and "tests/" not in str(f)
    ]

    if not changed_files:
        return 0

    all_issues: list[tuple[Path, list[str]]] = []

    for file_path in changed_files:
        if file_path.exists():
            metrics = check_file(file_path)
            if metrics.issues:
                all_issues.append((file_path, metrics.issues))

    if all_issues:
        print("📋 AGENT OUTPUT VALIDATION:")
        print("=" * 60)

        for file_path, issues in all_issues:
            print(f"\n📄 {file_path}:")
            for issue in issues:
                print(f"   ⚠️  {issue}")

        print("\n" + "=" * 60)
        print("These are quality suggestions - not blocking commit.")
        print("Consider addressing them to maintain code quality.")
        # Return 0 to not block (these are suggestions)
        return 0

    print("✅ Agent output validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
