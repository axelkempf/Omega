#!/usr/bin/env python3
"""P0-05: Document test coverage for migration candidates.

This script builds on P0-04 (candidate list) and produces:
- A candidate-level coverage summary (weighted by statements)
- A simple gap analysis (lowest-coverage or missing files per candidate)

It is intentionally conservative and reproducible:
- Uses pytest-cov JSON report as the single source of coverage data
- Skips integration tests by default ("-m not integration") to avoid environment coupling

No external deps are required beyond the project's dev extras (pytest + pytest-cov).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES_JSON = ROOT / "reports" / "migration_candidates" / "p0-04_candidates.json"


@dataclass(frozen=True)
class CoverageThresholds:
    candidate_warn_percent: float = 80.0
    file_warn_percent: float = 70.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_pytest_cov_json(
    *,
    coverage_json_path: Path,
    marker_expression: str,
) -> None:
    # Keep output quiet to avoid noise in CI logs; failures still surface.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        marker_expression,
        "--cov=src",
        f"--cov-report=json:{coverage_json_path}",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        # Surface pytest output for debugging only on failure.
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def _file_summary(coverage_payload: dict[str, Any], relpath: str) -> dict[str, Any] | None:
    files = coverage_payload.get("files")
    if not isinstance(files, dict):
        return None
    entry = files.get(relpath)
    if not isinstance(entry, dict):
        return None
    summary = entry.get("summary")
    if not isinstance(summary, dict):
        return None

    num_statements = summary.get("num_statements")
    covered_lines = summary.get("covered_lines")
    percent_covered = summary.get("percent_covered")

    if not isinstance(num_statements, int) or not isinstance(covered_lines, int):
        return None

    if not isinstance(percent_covered, (int, float)):
        # Fall back to computed percent.
        percent_covered = (covered_lines / num_statements) * 100.0 if num_statements else 100.0

    return {
        "path": relpath,
        "num_statements": num_statements,
        "covered_lines": covered_lines,
        "percent_covered": float(percent_covered),
    }


def build_report(
    *,
    candidates_json: Path,
    run_tests: bool,
    marker_expression: str,
    thresholds: CoverageThresholds,
) -> dict[str, Any]:
    candidates_payload = _load_json(candidates_json)
    candidates = candidates_payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Unexpected candidates format in {candidates_json}")

    tmp_dir = ROOT / "var" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    coverage_json_path = tmp_dir / "p0-05_coverage_raw.json"

    if run_tests:
        _run_pytest_cov_json(coverage_json_path=coverage_json_path, marker_expression=marker_expression)

    if not coverage_json_path.exists():
        raise FileNotFoundError(
            "Coverage JSON not found. Either run with --run-tests (default) "
            "or ensure var/tmp/p0-05_coverage_raw.json exists."
        )

    coverage_payload = _load_json(coverage_json_path)

    rows: list[dict[str, Any]] = []

    for c in candidates:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or c.get("key") or "<unknown>")
        key = str(c.get("key") or name)
        target = str(c.get("target") or "-")
        recommended_priority = str(c.get("recommended_priority") or "-")
        src_files = c.get("src_files")
        if not isinstance(src_files, list):
            src_files = []

        summaries: list[dict[str, Any]] = []
        missing_files: list[str] = []

        for p in src_files:
            rel = str(p)
            s = _file_summary(coverage_payload, rel)
            if s is None:
                missing_files.append(rel)
                continue
            summaries.append(s)

        total_statements = sum(int(s["num_statements"]) for s in summaries)
        total_covered = sum(int(s["covered_lines"]) for s in summaries)
        candidate_percent = (total_covered / total_statements) * 100.0 if total_statements else 100.0

        low_files = [
            {
                "path": s["path"],
                "percent_covered": round(float(s["percent_covered"]), 1),
                "num_statements": int(s["num_statements"]),
            }
            for s in sorted(summaries, key=lambda x: float(x["percent_covered"]))
            if float(s["percent_covered"]) < thresholds.file_warn_percent
        ]

        rows.append(
            {
                "key": key,
                "name": name,
                "target": target,
                "recommended_priority": recommended_priority,
                "candidate_percent_covered": round(candidate_percent, 1),
                "total_statements": total_statements,
                "covered_lines": total_covered,
                "measured_files": len(summaries),
                "missing_files": missing_files,
                "low_files": low_files,
            }
        )

    # Sort by priority first, then by lowest candidate coverage (biggest gaps first)
    prio_rank = {"High": 0, "Medium": 1, "Low": 2}

    def _sort_key(r: dict[str, Any]) -> tuple[int, float, str]:
        return (
            prio_rank.get(str(r.get("recommended_priority")), 9),
            float(r.get("candidate_percent_covered") or 0.0),
            str(r.get("name")),
        )

    rows.sort(key=_sort_key)

    return {
        "meta": {
            "generated_by": "tools/migration_test_coverage.py",
            "scope": "P0-05",
            "candidates_json": str(candidates_json.relative_to(ROOT)),
            "coverage_json": str(coverage_json_path.relative_to(ROOT)),
            "pytest_marker_expression": marker_expression,
        },
        "thresholds": {
            "candidate_warn_percent": thresholds.candidate_warn_percent,
            "file_warn_percent": thresholds.file_warn_percent,
        },
        "candidates": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    thresholds = report.get("thresholds", {})
    candidate_warn = thresholds.get("candidate_warn_percent", 80.0)
    file_warn = thresholds.get("file_warn_percent", 70.0)

    lines: list[str] = []
    lines.append("# Migration Candidate Test Coverage (P0-05)")
    lines.append("")
    lines.append("Dieser Report dokumentiert die Test-Coverage der P0-04 Migrations-Kandidaten.")
    lines.append("")
    lines.append("Konventionen:")
    lines.append("")
    lines.append(f"- Candidate-Warnschwelle: {candidate_warn:.0f}% (gewichtete Statements)")
    lines.append(f"- File-Warnschwelle: {file_warn:.0f}%")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Kandidat | Priority | Target | Coverage% | Statements | Gemessen | Fehlend |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")

    for r in report.get("candidates", []):
        cov = float(r.get("candidate_percent_covered") or 0.0)
        missing = r.get("missing_files") or []
        measured = int(r.get("measured_files") or 0)
        warn_flag = "⚠" if cov < float(candidate_warn) else ""
        lines.append(
            "| {name} | {prio} | {target} | {cov:.1f}{warn} | {stmts} | {measured} | {missing} |".format(
                name=str(r.get("name")),
                prio=str(r.get("recommended_priority")),
                target=str(r.get("target")),
                cov=cov,
                warn=warn_flag,
                stmts=int(r.get("total_statements") or 0),
                measured=measured,
                missing=len(missing),
            )
        )

    lines.append("")
    lines.append("## Gap-Analyse")
    lines.append("")
    lines.append("Fokus: Kandidaten mit niedriger Coverage oder fehlenden Coverage-Daten.")
    lines.append("")

    for r in report.get("candidates", []):
        cov = float(r.get("candidate_percent_covered") or 0.0)
        missing: list[str] = list(r.get("missing_files") or [])
        low_files: list[dict[str, Any]] = list(r.get("low_files") or [])

        needs_attention = cov < float(candidate_warn) or bool(missing) or bool(low_files)
        if not needs_attention:
            continue

        lines.append(f"### {r.get('name')}")
        lines.append("")
        lines.append(f"- Coverage: {cov:.1f}%")
        if missing:
            lines.append("- Fehlende Coverage-Einträge (evtl. nicht importiert/ausgeführt):")
            for p in missing[:15]:
                lines.append(f"  - `{p}`")
            if len(missing) > 15:
                lines.append(f"  - … (+{len(missing) - 15} weitere)")
        if low_files:
            lines.append(f"- Low-Coverage Dateien (< {float(file_warn):.0f}%):")
            for lf in low_files[:10]:
                lines.append(
                    "  - `{path}`: {cov:.1f}% ({stmts} stmts)".format(
                        path=str(lf.get("path")),
                        cov=float(lf.get("percent_covered") or 0.0),
                        stmts=int(lf.get("num_statements") or 0),
                    )
                )
            if len(low_files) > 10:
                lines.append(f"  - … (+{len(low_files) - 10} weitere)")
        lines.append("")

    lines.append("## Reproduzieren")
    lines.append("")
    lines.append("- JSON: `tools/migration_test_coverage.py --format json`")
    lines.append("- Markdown: `tools/migration_test_coverage.py --format md`")
    lines.append("")
    lines.append("Hinweis: Standardmäßig werden Integrationstests via Marker ausgeschlossen: `-m not integration`.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-05 migration candidate coverage")
    parser.add_argument("--format", choices=("json", "md"), default="md")
    parser.add_argument(
        "--candidates-json",
        default=str(DEFAULT_CANDIDATES_JSON),
        help="Path to P0-04 candidates JSON.",
    )
    parser.add_argument(
        "--marker",
        default="not integration",
        help="Pytest marker expression (default excludes integration tests).",
    )
    parser.add_argument(
        "--run-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pytest to generate coverage JSON (default: true).",
    )
    args = parser.parse_args()

    report = build_report(
        candidates_json=Path(args.candidates_json),
        run_tests=bool(args.run_tests),
        marker_expression=str(args.marker),
        thresholds=CoverageThresholds(),
    )

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=False))
        return

    print(render_markdown(report))


if __name__ == "__main__":
    main()
