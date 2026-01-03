#!/usr/bin/env python3
"""AST-based type coverage analysis for the trading stack."""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class FileStats:
    """Type coverage statistics for a single file."""

    path: str
    total_functions: int = 0
    functions_with_return_type: int = 0
    total_parameters: int = 0
    parameters_with_type: int = 0
    total_class_attributes: int = 0
    class_attributes_with_type: int = 0

    @property
    def function_coverage(self) -> float:
        if self.total_functions == 0:
            return 100.0
        return (self.functions_with_return_type / self.total_functions) * 100

    @property
    def parameter_coverage(self) -> float:
        if self.total_parameters == 0:
            return 100.0
        return (self.parameters_with_type / self.total_parameters) * 100


class TypeCoverageVisitor(ast.NodeVisitor):
    """AST visitor that collects type hint statistics."""

    def __init__(self) -> None:
        self.stats = FileStats(path="")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._analyze_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._analyze_function(node)
        self.generic_visit(node)

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Skip private/dunder methods for cleaner metrics
        if node.name.startswith("_") and not node.name.startswith("__"):
            return

        self.stats.total_functions += 1

        # Check return type
        if node.returns is not None:
            self.stats.functions_with_return_type += 1

        # Check parameters (skip self/cls)
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            self.stats.total_parameters += 1
            if arg.annotation is not None:
                self.stats.parameters_with_type += 1

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Class-level annotated assignment
        self.stats.total_class_attributes += 1
        self.stats.class_attributes_with_type += 1
        self.generic_visit(node)


def analyze_file(path: Path) -> FileStats:
    """Analyze a single Python file for type coverage."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"⚠️  Could not parse {path}: {e}", file=sys.stderr)
        return FileStats(path=str(path))

    visitor = TypeCoverageVisitor()
    visitor.stats.path = str(path)
    visitor.visit(tree)
    return visitor.stats


def find_python_files(root: Path) -> Iterator[Path]:
    """Find all Python files in directory, excluding __pycache__."""
    for path in root.rglob("*.py"):
        if "__pycache__" not in str(path):
            yield path


def main() -> None:
    src_root = Path("src")
    if not src_root.exists():
        print("❌ src/ directory not found", file=sys.stderr)
        sys.exit(1)

    all_stats: list[FileStats] = []

    for py_file in find_python_files(src_root):
        stats = analyze_file(py_file)
        all_stats.append(stats)

    # Aggregate stats
    total_funcs = sum(s.total_functions for s in all_stats)
    typed_funcs = sum(s.functions_with_return_type for s in all_stats)
    total_params = sum(s.total_parameters for s in all_stats)
    typed_params = sum(s.parameters_with_type for s in all_stats)

    func_coverage = (typed_funcs / total_funcs * 100) if total_funcs > 0 else 100
    param_coverage = (typed_params / total_params * 100) if total_params > 0 else 100

    # Print summary
    print("\n" + "=" * 60)
    print("TYPE COVERAGE REPORT")
    print("=" * 60)
    print(f"Files analyzed: {len(all_stats)}")
    print(
        f"Functions with return types: {typed_funcs}/{total_funcs} ({func_coverage:.1f}%)"
    )
    print(
        f"Parameters with type hints: {typed_params}/{total_params} ({param_coverage:.1f}%)"
    )
    print("=" * 60)

    # Top 10 files needing attention
    files_by_coverage = sorted(
        [s for s in all_stats if s.total_functions > 0],
        key=lambda s: s.function_coverage,
    )

    print("\nFiles needing most attention:")
    for stats in files_by_coverage[:10]:
        print(f"  {stats.function_coverage:5.1f}% - {stats.path}")

    # Export JSON report
    report = {
        "summary": {
            "files_analyzed": len(all_stats),
            "function_coverage_percent": round(func_coverage, 1),
            "parameter_coverage_percent": round(param_coverage, 1),
            "total_functions": total_funcs,
            "typed_functions": typed_funcs,
            "total_parameters": total_params,
            "typed_parameters": typed_params,
        },
        "files": [
            {
                "path": s.path,
                "function_coverage": round(s.function_coverage, 1),
                "parameter_coverage": round(s.parameter_coverage, 1),
                "total_functions": s.total_functions,
                "typed_functions": s.functions_with_return_type,
            }
            for s in all_stats
        ],
    }

    report_path = Path("var/reports/type_coverage.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n📄 Full report: {report_path}")


if __name__ == "__main__":
    main()
