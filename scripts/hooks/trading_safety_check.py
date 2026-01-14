#!/usr/bin/env python3
"""Check trading safety invariants.

This hook scans trading-related code for potentially unsafe patterns:
- Hardcoded magic numbers or lot sizes
- Direct order sending without visible risk checks
- Silent exception handling
- time.sleep in trading code

Critical files (execution, risk management) are flagged with higher severity.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patterns that indicate potentially unsafe changes
# Format: (pattern, message, is_critical, excluded_files)
# excluded_files: Set of filenames where this pattern is legitimate
UNSAFE_PATTERNS: list[tuple[str, str, bool, set[str]]] = [
    (
        r"magic_number\s*=\s*\d+(?!\s*#)",
        "Hardcoded magic_number - should be from config",
        True,
        set(),
    ),
    (
        r"lot_size\s*=\s*\d+\.?\d*(?!\s*#)",
        "Hardcoded lot_size - should be calculated dynamically",
        True,
        {"lot_size_calculator.py"},  # Legitimate: calculating lot_size internally
    ),
    (
        r"\.order_send\(",
        "Direct MT5 order_send - ensure risk checks are applied",
        True,
        {"mt5_adapter.py"},  # Legitimate: the MT5 adapter is the designated place
    ),
    (
        r"position\.close\(\s*\)",
        "Position closing - ensure proper logging and validation",
        False,
        set(),
    ),
    (
        r"time\.sleep\(\s*\d",
        "time.sleep in trading code - may cause missed signals",
        False,
        {"multi_strategy_controller.py"},  # Legitimate: pacing loops, reconnection
    ),
    (
        r"except:\s*$",
        "Bare except clause - may hide trading errors",
        True,
        set(),
    ),
    (
        r"except\s+Exception\s*:\s*\n\s*pass",
        "Silent exception handling - may hide trading errors",
        True,
        set(),
    ),
    (
        r"raise\s+SystemExit",
        "SystemExit in trading code - use proper shutdown procedures",
        False,
        set(),
    ),
]

# Files that require extra scrutiny
CRITICAL_FILES = {
    "execution_engine.py",
    "risk_manager.py",
    "lot_size_calculator.py",
    "mt5_adapter.py",
    "order_manager.py",
    "position_manager.py",
}

# Whitelist patterns (comments that acknowledge the issue)
WHITELIST_PATTERNS = [
    r"#\s*noqa:\s*trading-safety",
    r"#\s*SAFETY-REVIEWED",
    r"#\s*intentional",
]


def is_whitelisted(line: str) -> bool:
    """Check if a line is whitelisted via comment.

    Args:
        line: Source code line to check.

    Returns:
        True if the line contains a whitelist comment.
    """
    for pattern in WHITELIST_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def check_file(file_path: Path) -> list[str]:
    """Check a file for trading safety issues.

    Args:
        file_path: Path to the Python file.

    Returns:
        List of issue messages.
    """
    issues: list[str] = []

    try:
        content = file_path.read_text()
    except Exception:
        return issues

    lines = content.split("\n")
    is_critical_file = file_path.name in CRITICAL_FILES

    for i, line in enumerate(lines, 1):
        # Skip whitelisted lines
        if is_whitelisted(line):
            continue

        for unsafe_pattern in UNSAFE_PATTERNS:
            pattern = unsafe_pattern[0]
            message = unsafe_pattern[1]
            is_critical_pattern = unsafe_pattern[2]
            excluded_files = unsafe_pattern[3]
            # Skip if this file is excluded for this pattern
            if file_path.name in excluded_files:
                continue

            if re.search(pattern, line):
                # Determine severity
                if is_critical_file and is_critical_pattern:
                    severity = "🔴 CRITICAL"
                elif is_critical_file or is_critical_pattern:
                    severity = "🟠 WARNING"
                else:
                    severity = "🟡 INFO"

                issues.append(f"{severity} {file_path}:{i}")
                issues.append(f"    Pattern: {message}")
                issues.append(f"    Line: {line.strip()[:80]}")
                issues.append("")

    return issues


def check_commit_message_for_safety_review() -> bool:
    """Check if SAFETY-REVIEWED: is in commit message.

    Returns:
        True if safety review is acknowledged.
    """
    commit_msg_path = Path(".git/COMMIT_EDITMSG")
    if commit_msg_path.exists():
        content = commit_msg_path.read_text()
        if "SAFETY-REVIEWED:" in content:
            return True
    return False


def main() -> int:
    """Main entry point for the hook."""
    # Only check trading-related files
    changed_files = [
        Path(f)
        for f in sys.argv[1:]
        if f.endswith(".py") and ("hf_engine" in f or "strategies" in f)
    ]

    if not changed_files:
        return 0

    all_issues: list[str] = []
    has_critical = False

    for file_path in changed_files:
        if file_path.exists():
            issues = check_file(file_path)
            if issues:
                all_issues.extend(issues)
                if any("CRITICAL" in issue for issue in issues):
                    has_critical = True

    if all_issues:
        print("⚠️  TRADING SAFETY CHECK:")
        print("=" * 60)
        for issue in all_issues:
            print(issue)
        print("=" * 60)
        print("\nReview these issues before committing.")
        print("Add '# noqa: trading-safety' to whitelist specific lines.")
        print("Add 'SAFETY-REVIEWED:' to commit message to acknowledge all.")

        # Check if acknowledged
        if check_commit_message_for_safety_review():
            print("\n✅ Safety review acknowledged in commit message.")
            return 0

        # Only block on critical issues
        if has_critical:
            return 1

        print("\nℹ️  No critical issues - proceeding (warnings only).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
