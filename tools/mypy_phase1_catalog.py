from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ERROR_LINE_RE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+): (?P<kind>error|note):")


@dataclass(frozen=True)
class PackageStats:
    package: str
    file_count: int
    error_count: int
    note_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _src_dir(root: Path) -> Path:
    return root / "src"


def _iter_py_files(package_dir: Path) -> list[Path]:
    if not package_dir.exists():
        return []
    return sorted(
        p
        for p in package_dir.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.endswith(".pyi")
    )


def _run_mypy_for_path(*, path: Path, config_file: Path) -> tuple[list[str], int]:
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(config_file),
        str(path),
    ]

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=_repo_root(),
    )

    # mypy uses stdout for findings; stderr is typically for crashes
    output = (proc.stdout or "").splitlines()
    if proc.stderr:
        output.extend(proc.stderr.splitlines())

    return output, proc.returncode


def _count_findings(lines: list[str]) -> tuple[int, int, dict[str, int]]:
    error_count = 0
    note_count = 0
    by_file: dict[str, int] = {}

    for line in lines:
        m = _ERROR_LINE_RE.match(line)
        if not m:
            continue
        kind = m.group("kind")
        if kind == "error":
            error_count += 1
            path = m.group("path")
            by_file[path] = by_file.get(path, 0) + 1
        elif kind == "note":
            note_count += 1

    return error_count, note_count, by_file


def _subpackage_key(src_root: Path, file_path: str) -> str:
    p = Path(file_path)
    try:
        rel = p.resolve().relative_to(src_root)
    except Exception:
        # best-effort fallback: strip leading parts until we find "src"
        parts = p.parts
        if "src" in parts:
            rel = Path(*parts[parts.index("src") + 1 :])
        else:
            rel = p

    if len(rel.parts) < 2:
        # e.g. backtest_engine/__init__.py
        return rel.parts[0] if rel.parts else "(unknown)"

    top = rel.parts[0]
    sub = rel.parts[1]
    return f"{top}.{sub}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 helper: catalog mypy error surface for packages that are still "
            "configured with ignore_errors in pyproject.toml."
        )
    )
    parser.add_argument(
        "--config",
        default="tools/mypy_phase1.ini",
        help="Path to mypy config file used for baseline measurement.",
    )
    parser.add_argument(
        "--out-json",
        default="reports/mypy_baseline/p1-01_ignore_errors_catalog.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=["hf_engine", "backtest_engine", "ui_engine"],
        help="Top-level src packages to scan.",
    )

    args = parser.parse_args()

    root = _repo_root()
    src = _src_dir(root)
    config_file = (root / args.config).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"mypy config not found: {config_file}")

    report: dict[str, Any] = {
        "generated_by": str(Path(__file__).relative_to(root)),
        "mypy_config": str(config_file.relative_to(root)),
        "packages": {},
        "subpackage_breakdown": {},
        "subpackage_stats": {},
    }

    subpkg_file_counts: dict[str, int] = {}
    subpkg_error_counts: dict[str, int] = {}

    for pkg in args.packages:
        pkg_dir = src / pkg
        py_files = _iter_py_files(pkg_dir)

        for p in py_files:
            key = _subpackage_key(src, str(p))
            subpkg_file_counts[key] = subpkg_file_counts.get(key, 0) + 1

        lines, exit_code = _run_mypy_for_path(path=pkg_dir, config_file=config_file)
        err_count, note_count, by_file = _count_findings(lines)

        report["packages"][pkg] = {
            "path": str(pkg_dir.relative_to(root)),
            "file_count": len(py_files),
            "mypy_exit_code": int(exit_code),
            "error_count": int(err_count),
            "note_count": int(note_count),
            "error_density": (
                (float(err_count) / float(len(py_files))) if py_files else 0.0
            ),
        }

        # Aggregate a first-order subpackage breakdown (pkg.<subdir>)
        for file_path, cnt in by_file.items():
            key = _subpackage_key(src, file_path)
            report["subpackage_breakdown"][key] = report["subpackage_breakdown"].get(
                key, 0
            ) + int(cnt)

            subpkg_error_counts[key] = subpkg_error_counts.get(key, 0) + int(cnt)

    # Add richer stats without breaking the existing flat breakdown.
    all_subpkg_keys = set(subpkg_file_counts) | set(subpkg_error_counts)
    for key in sorted(all_subpkg_keys):
        files = subpkg_file_counts.get(key, 0)
        errors = subpkg_error_counts.get(key, 0)
        report["subpackage_stats"][key] = {
            "file_count": int(files),
            "error_count": int(errors),
            "error_density": (float(errors) / float(files)) if files else 0.0,
        }

    out_path = (root / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote: {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
