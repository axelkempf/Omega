#!/usr/bin/env python3
"""Run pytest only for changed files.

This hook finds corresponding test files for changed source files
and runs them. This provides faster feedback than running the full
test suite on every commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_test_file(source_file: Path) -> Path | None:
    """Find corresponding test file for a source file.

    Searches for test files following common naming conventions:
    - tests/test_<name>.py
    - tests/<parent_dir>/test_<name>.py

    Args:
        source_file: Path to the source file.

    Returns:
        Path to the test file if found, None otherwise.
    """
    name = source_file.stem
    test_candidates = [
        Path(f"tests/test_{name}.py"),
        Path(f"tests/{source_file.parent.name}/test_{name}.py"),
        Path(f"tests/unit/test_{name}.py"),
        Path(f"tests/integration/test_{name}.py"),
    ]

    for candidate in test_candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    """Main entry point for the hook."""
    changed_files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]

    if not changed_files:
        return 0

    # Collect test files
    test_files: set[Path] = set()

    for source_file in changed_files:
        # If it's already a test file, add it
        if source_file.name.startswith("test_"):
            if source_file.exists():
                test_files.add(source_file)
            continue

        # Find corresponding test file
        test_file = find_test_file(source_file)
        if test_file:
            test_files.add(test_file)

    if not test_files:
        print("ℹ️  No test files found for changed files, skipping...")
        return 0

    # Run pytest
    cmd = ["pytest", "-q", "--tb=short"] + [str(f) for f in sorted(test_files)]
    print(f"🧪 Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
