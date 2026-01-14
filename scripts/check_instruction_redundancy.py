#!/usr/bin/env python3
"""
Check instruction files for redundant content.

This script scans .github/instructions/ for potential redundancies
that should be consolidated into _core/ standards.

Usage:
    python scripts/check_instruction_redundancy.py
    python scripts/check_instruction_redundancy.py --verbose
    python scripts/check_instruction_redundancy.py --fix  # Show fix suggestions

Exit codes:
    0: No redundancies found
    1: Redundancies detected (CI should fail)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Patterns that indicate redundant content
# These should be consolidated in _core/ files
REDUNDANCY_PATTERNS: Final[dict[str, list[str]]] = {
    "python-standards": [
        r"Python\s*[≥>=]+\s*3\.1[0-2]",  # Python version requirement
        r"PEP\s*8",  # PEP 8 mention
        r"Type\s+Hints?",  # Type hints requirement
        r"from\s+__future__\s+import\s+annotations",  # Future annotations
        r"\bblack\b.*\bformat",  # Black formatter mention
        r"\bisort\b",  # isort mention
        r"snake_case|CamelCase",  # Naming conventions
    ],
    "security-standards": [
        r"OWASP",  # OWASP mention
        r"SQL\s+[Ii]njection",  # SQL injection
        r"[Ss]ecrets?\s+[Mm]anagement",  # Secrets management
        r"Input\s+[Vv]alidation",  # Input validation
        r"[Pp]arameterized\s+[Qq]ueries?",  # Parameterized queries
        r"\.env",  # Environment files
        r"[Nn]ever\s+[Cc]ommit.*[Ss]ecrets?",  # Secret commit warnings
    ],
    "testing-standards": [
        r"\bpytest\b",  # pytest mention
        r"[Dd]eterministic\s+[Tt]ests?",  # Deterministic tests
        r"[Ff]ixed?\s+[Ss]eeds?",  # Fixed seeds
        r"[Mm]ock.*MT5|MT5.*[Mm]ock",  # MT5 mocking
        r"AAA\s+[Pp]attern|Arrange.Act.Assert",  # AAA pattern
        r"[Cc]overage",  # Coverage mention
    ],
    "error-handling": [
        r"[Ff]ail\s+[Ff]ast",  # Fail fast principle
        r"[Ss]pecific\s+[Ee]xceptions?",  # Specific exceptions
        r"try.*except.*pass",  # Anti-pattern
        r"[Rr]aise\s+Exception\(",  # Generic exception anti-pattern
        r"[Cc]ontext\s+[Mm]anager",  # Context managers
    ],
}

# Files that are allowed to contain these patterns (core files)
EXEMPT_FILES: Final[set[str]] = {
    "_core/python-standards.instructions.md",
    "_core/security-standards.instructions.md",
    "_core/testing-standards.instructions.md",
    "_core/error-handling.instructions.md",
    "_domain/trading-safety.instructions.md",
}

# Files that are expected to reference core standards
SHOULD_REFERENCE_CORE: Final[set[str]] = {
    "codexer.instructions.md",
    "code-review-generic.instructions.md",
    "tester.instructions.md",
}


@dataclass
class RedundancyMatch:
    """A detected redundancy in an instruction file."""

    file: Path
    line_number: int
    line_content: str
    category: str
    pattern: str


def find_redundancies(
    instructions_dir: Path,
    verbose: bool = False,
) -> list[RedundancyMatch]:
    """Find all redundancies in instruction files."""
    matches: list[RedundancyMatch] = []

    # Find all markdown files
    for md_file in instructions_dir.rglob("*.md"):
        # Skip exempt files
        relative_path = md_file.relative_to(instructions_dir)
        if str(relative_path) in EXEMPT_FILES:
            if verbose:
                print(f"  [SKIP] {relative_path} (exempt)")
            continue

        # Read file content
        content = md_file.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Check each pattern category
        for category, patterns in REDUNDANCY_PATTERNS.items():
            for pattern in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                for line_num, line in enumerate(lines, start=1):
                    if regex.search(line):
                        matches.append(
                            RedundancyMatch(
                                file=md_file,
                                line_number=line_num,
                                line_content=line.strip()[:80],
                                category=category,
                                pattern=pattern,
                            )
                        )

    return matches


def check_core_references(instructions_dir: Path) -> list[str]:
    """Check if files that should reference _core/ actually do."""
    missing_references: list[str] = []

    for filename in SHOULD_REFERENCE_CORE:
        file_path = instructions_dir / filename
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")

        # Check for reference to _core/
        if "_core/" not in content and "Core Standards" not in content:
            missing_references.append(filename)

    return missing_references


def format_report(
    matches: list[RedundancyMatch],
    missing_refs: list[str],
    show_fix: bool = False,
) -> str:
    """Format the redundancy report."""
    lines: list[str] = []

    if matches:
        lines.append("=" * 60)
        lines.append("REDUNDANCIES DETECTED")
        lines.append("=" * 60)
        lines.append("")

        # Group by file
        by_file: dict[Path, list[RedundancyMatch]] = {}
        for match in matches:
            by_file.setdefault(match.file, []).append(match)

        for file, file_matches in sorted(by_file.items()):
            lines.append(f"📄 {file.name}")
            for match in file_matches:
                lines.append(
                    f"   Line {match.line_number}: [{match.category}] {match.line_content}"
                )
            lines.append("")

        if show_fix:
            lines.append("-" * 60)
            lines.append("SUGGESTED FIX:")
            lines.append("")
            lines.append("Remove redundant content and add reference to _core/:")
            lines.append("")
            lines.append("```markdown")
            lines.append("## Core Standards")
            lines.append("")
            lines.append("This file follows the Omega Core Standards:")
            lines.append("")
            lines.append(
                "- Python: [_core/python-standards.instructions.md](_core/python-standards.instructions.md)"
            )
            lines.append(
                "- Security: [_core/security-standards.instructions.md](_core/security-standards.instructions.md)"
            )
            lines.append(
                "- Testing: [_core/testing-standards.instructions.md](_core/testing-standards.instructions.md)"
            )
            lines.append(
                "- Errors: [_core/error-handling.instructions.md](_core/error-handling.instructions.md)"
            )
            lines.append("```")
            lines.append("")

    if missing_refs:
        lines.append("=" * 60)
        lines.append("MISSING CORE REFERENCES")
        lines.append("=" * 60)
        lines.append("")
        for filename in missing_refs:
            lines.append(f"  ⚠️  {filename} should reference _core/ standards")
        lines.append("")

    if not matches and not missing_refs:
        lines.append("✅ No redundancies detected!")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check instruction files for redundant content"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including skipped files",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show fix suggestions for detected redundancies",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(".github/instructions"),
        help="Path to instructions directory",
    )
    args = parser.parse_args()

    # Find instructions directory
    instructions_dir = args.path
    if not instructions_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        instructions_dir = script_dir / ".github" / "instructions"

    if not instructions_dir.exists():
        print(f"❌ Instructions directory not found: {instructions_dir}")
        return 1

    print(f"Scanning: {instructions_dir}")
    print()

    # Find redundancies
    matches = find_redundancies(instructions_dir, verbose=args.verbose)

    # Check core references
    missing_refs = check_core_references(instructions_dir)

    # Format and print report
    report = format_report(matches, missing_refs, show_fix=args.fix)
    print(report)

    # Return exit code
    if matches or missing_refs:
        print(
            f"Found {len(matches)} redundancies and {len(missing_refs)} missing references"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
