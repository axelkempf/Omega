#!/usr/bin/env python3
"""Check that architecture.md is up to date.

This hook verifies that significant structural changes to the src/
directory are reflected in architecture.md. It's a reminder to update
documentation when the codebase structure changes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def get_staged_src_files() -> list[str]:
    """Get list of staged src/ files.

    Returns:
        List of staged file paths in src/.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", "src/"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []
    except Exception:
        return []


def get_new_directories() -> set[str]:
    """Get directories that are new (don't exist in HEAD).

    Returns:
        Set of new directory paths.
    """
    new_dirs: set[str] = set()

    try:
        # Get list of staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-status", "--", "src/"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return new_dirs

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] == "A":  # Added file
                file_path = Path(parts[1])
                # Check if parent directory is new
                parent = file_path.parent
                if str(parent).startswith("src/"):
                    # Check if directory existed in HEAD
                    check = subprocess.run(
                        ["git", "ls-tree", "-d", "HEAD", str(parent)],
                        capture_output=True,
                        text=True,
                    )
                    if not check.stdout.strip():
                        new_dirs.add(str(parent))

    except Exception:
        pass

    return new_dirs


def get_documented_paths() -> set[str]:
    """Extract paths documented in architecture.md.

    Returns:
        Set of documented paths.
    """
    arch_path = Path("architecture.md")
    if not arch_path.exists():
        return set()

    content = arch_path.read_text()
    paths: set[str] = set()

    # Match patterns like `src/module/` or `src/module/file.py`
    pattern = r"`(src/[^`]+)`"
    matches = re.findall(pattern, content)

    for match in matches:
        paths.add(match.rstrip("/"))
        # Also add parent directories
        path = Path(match)
        for parent in path.parents:
            if str(parent).startswith("src"):
                paths.add(str(parent))

    return paths


def check_architecture_updates_needed() -> list[str]:
    """Check if architecture.md needs updates.

    Returns:
        List of warnings about undocumented changes.
    """
    warnings: list[str] = []

    # Get new directories
    new_dirs = get_new_directories()
    if not new_dirs:
        return warnings

    # Get documented paths
    documented = get_documented_paths()

    # Find undocumented new directories
    for new_dir in new_dirs:
        normalized = new_dir.rstrip("/")
        if normalized not in documented:
            # Check if any parent is documented
            path = Path(normalized)
            parent_documented = False
            for parent in path.parents:
                if str(parent) in documented:
                    parent_documented = True
                    break

            if not parent_documented:
                warnings.append(f"New directory not in architecture.md: {new_dir}")

    return warnings


def main() -> int:
    """Main entry point for the hook."""
    # Check if any src/ files were staged
    staged_files = get_staged_src_files()
    if not staged_files:
        return 0

    # Check for architecture updates needed
    warnings = check_architecture_updates_needed()

    if warnings:
        print("📐 ARCHITECTURE CHECK:")
        print("=" * 60)
        print("\nThe following structural changes may need documentation:\n")

        for warning in warnings:
            print(f"  ⚠️  {warning}")

        print("\n" + "=" * 60)
        print("Consider updating architecture.md to reflect these changes.")
        print("This is a reminder only - not blocking commit.")

    return 0  # Never block, just remind


if __name__ == "__main__":
    sys.exit(main())
