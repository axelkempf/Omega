#!/usr/bin/env python3
"""Detect breaking changes in public APIs.

This hook compares the current version of changed files with their
previous version in git to detect:
- Removed public functions
- Changed function signatures
- Removed public classes or methods

If breaking changes are detected, the commit can still proceed if
'BREAKING:' is included in the commit message.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


class APIExtractor(ast.NodeVisitor):
    """Extract public API signatures from Python files."""

    def __init__(self) -> None:
        self.functions: dict[str, str] = {}
        self.classes: dict[str, list[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function signatures."""
        if not node.name.startswith("_"):
            args = [a.arg for a in node.args.args]
            self.functions[node.name] = f"{node.name}({', '.join(args)})"
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async function signatures."""
        if not node.name.startswith("_"):
            args = [a.arg for a in node.args.args]
            self.functions[node.name] = f"async {node.name}({', '.join(args)})"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class and method signatures."""
        if not node.name.startswith("_"):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        args = [a.arg for a in item.args.args]
                        methods.append(f"{item.name}({', '.join(args)})")
            self.classes[node.name] = methods
        self.generic_visit(node)


def get_api(content: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Extract API from Python source code.

    Args:
        content: Python source code as string.

    Returns:
        Tuple of (functions dict, classes dict).
    """
    try:
        tree = ast.parse(content)
        extractor = APIExtractor()
        extractor.visit(tree)
        return extractor.functions, extractor.classes
    except SyntaxError:
        return {}, {}


def get_previous_version(file_path: Path) -> str | None:
    """Get the previous version of a file from git.

    Args:
        file_path: Path to the file.

    Returns:
        Previous file content or None if not in git.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{file_path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def check_commit_message_for_breaking() -> bool:
    """Check if the staged commit message contains BREAKING:.

    Returns:
        True if BREAKING: is acknowledged.
    """
    # Check COMMIT_EDITMSG if available (during commit)
    commit_msg_path = Path(".git/COMMIT_EDITMSG")
    if commit_msg_path.exists():
        content = commit_msg_path.read_text()
        if "BREAKING:" in content:
            return True

    return False


def main() -> int:
    """Main entry point for the hook."""
    changed_files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    breaking_changes: list[str] = []

    for file_path in changed_files:
        if not file_path.exists():
            continue

        # Skip test files
        if "test" in file_path.name.lower():
            continue

        # Get current content
        current_content = file_path.read_text()
        current_funcs, current_classes = get_api(current_content)

        # Get previous version
        prev_content = get_previous_version(file_path)
        if not prev_content:
            continue  # New file, no breaking change possible

        prev_funcs, prev_classes = get_api(prev_content)

        # Compare functions
        for name, signature in prev_funcs.items():
            if name not in current_funcs:
                breaking_changes.append(f"REMOVED FUNCTION: {file_path}:{name}")
            elif current_funcs[name] != signature:
                breaking_changes.append(
                    f"CHANGED SIGNATURE: {file_path}:{name}\n"
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
                    current_methods = current_classes[class_name]
                    if not any(m.startswith(f"{method_name}(") for m in current_methods):
                        breaking_changes.append(
                            f"REMOVED METHOD: {file_path}:{class_name}.{method_name}"
                        )

    if breaking_changes:
        print("⚠️  POTENTIAL BREAKING CHANGES DETECTED:")
        print("-" * 60)
        for change in breaking_changes:
            print(change)
        print("-" * 60)
        print("\nIf intentional, add 'BREAKING:' to your commit message.")

        # Check if acknowledged
        if check_commit_message_for_breaking():
            print("✅ Breaking change acknowledged in commit message.")
            return 0

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
