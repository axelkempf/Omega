#!/usr/bin/env python3
"""P0-04: Identify & prioritize Rust/Julia migration candidates.

This script combines:
- Performance baselines from `reports/performance_baselines/p0-01_*.json`
- Type readiness (AST-based type coverage) from files under `src/`

It produces a compact, evidence-based candidate ranking. The output is intentionally
heuristic: it is a *planning aid*, not a performance oracle.

Design goals:
- No external deps
- Deterministic output
- Works on macOS/Linux without MT5
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
BASELINES_DIR = ROOT / "reports" / "performance_baselines"


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    name: str
    target: str  # Rust | Julia | TBD
    src_globs: tuple[str, ...]
    baseline_file: str | None
    notes: str


@dataclass
class TypeTotals:
    total_functions: int = 0
    typed_returns: int = 0
    total_parameters: int = 0
    typed_parameters: int = 0

    def function_coverage(self) -> float:
        if self.total_functions == 0:
            return 100.0
        return (self.typed_returns / self.total_functions) * 100.0

    def parameter_coverage(self) -> float:
        if self.total_parameters == 0:
            return 100.0
        return (self.typed_parameters / self.total_parameters) * 100.0


class _CoverageVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.totals = TypeTotals()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._analyze_fn(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._analyze_fn(node)
        self.generic_visit(node)

    def _analyze_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Keep alignment with tools/type_coverage.py for comparable metrics.
        if node.name.startswith("_") and not node.name.startswith("__"):
            return

        self.totals.total_functions += 1
        if node.returns is not None:
            self.totals.typed_returns += 1

        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            self.totals.total_parameters += 1
            if arg.annotation is not None:
                self.totals.typed_parameters += 1


def _iter_candidate_files(globs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in globs:
        matches = sorted(SRC_ROOT.glob(pattern))
        for m in matches:
            if m.is_dir():
                files.extend(sorted(m.rglob("*.py")))
            elif m.suffix == ".py":
                files.append(m)
    # De-dup while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in files:
        if "__pycache__" in str(p):
            continue
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _analyze_types(paths: list[Path]) -> TypeTotals:
    totals = TypeTotals()
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            # Ignore unparsable files for planning report (rare in src/).
            continue
        visitor = _CoverageVisitor()
        visitor.visit(tree)
        totals.total_functions += visitor.totals.total_functions
        totals.typed_returns += visitor.totals.typed_returns
        totals.total_parameters += visitor.totals.total_parameters
        totals.typed_parameters += visitor.totals.typed_parameters
    return totals


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_impact_seconds(payload: dict[str, Any]) -> float | None:
    # Steady-state runners
    if isinstance(payload.get("second_run_seconds"), (int, float)):
        return float(payload["second_run_seconds"])

    # Common nested shapes
    for key in ("results", "walkforward_window"):
        if isinstance(payload.get(key), dict) and isinstance(
            payload[key].get("seconds"), (int, float)
        ):
            return float(payload[key]["seconds"])

    # Optimizer baseline
    if isinstance(payload.get("robust_zone_analysis"), dict) or isinstance(
        payload.get("final_param_selector"), dict
    ):
        total = 0.0
        for part in ("robust_zone_analysis", "final_param_selector"):
            if isinstance(payload.get(part), dict) and isinstance(
                payload[part].get("seconds"), (int, float)
            ):
                total += float(payload[part]["seconds"])
        return total if total > 0 else None

    # Slippage + fee baseline
    if isinstance(payload.get("slippage_model"), dict) and isinstance(
        payload.get("fee_model"), dict
    ):
        s = payload["slippage_model"].get("seconds")
        f = payload["fee_model"].get("seconds")
        if isinstance(s, (int, float)) and isinstance(f, (int, float)):
            return float(s) + float(f)

    # Indicator cache / rating: operations
    ops = payload.get("operations")
    if isinstance(ops, dict) and ops:
        init_seconds = float(payload.get("init_seconds") or 0.0)

        # indicator_cache style
        first_calls: list[float] = []
        seconds_list: list[float] = []
        for v in ops.values():
            if not isinstance(v, dict):
                continue
            if isinstance(v.get("first_call_seconds"), (int, float)):
                first_calls.append(float(v["first_call_seconds"]))
            if isinstance(v.get("seconds"), (int, float)):
                seconds_list.append(float(v["seconds"]))

        if first_calls:
            return init_seconds + max(first_calls)
        if seconds_list:
            return init_seconds + sum(seconds_list)

    return None


def _perf_bucket(impact_seconds: float | None) -> str:
    if impact_seconds is None:
        return "Unknown"
    if impact_seconds >= 1.0:
        return "High"
    if impact_seconds >= 0.15:
        return "Medium"
    return "Low"


def _type_bucket(func_cov: float, param_cov: float) -> str:
    if func_cov >= 80.0 and param_cov >= 90.0:
        return "High"
    if func_cov >= 50.0 and param_cov >= 80.0:
        return "Medium"
    return "Low"


def _priority_bucket(perf: str, type_ready: str) -> str:
    # Conservative: only call something High when both signals are good.
    if perf == "High" and type_ready in ("High", "Medium"):
        return "High"
    if perf == "Medium" and type_ready == "High":
        return "Medium"
    if perf == "High" and type_ready == "Low":
        return "Medium"
    if perf == "Medium" and type_ready == "Medium":
        return "Medium"
    if perf == "Unknown":
        return "Low"
    return "Low"


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        key="indicator_cache",
        name="Indicator Cache",
        target="Rust",
        src_globs=("backtest_engine/core/indicator_cache.py",),
        baseline_file="p0-01_indicator_cache.json",
        notes="Hot-path für Indikatoren + Cache; starker Kandidat für Rust (ndarray/Arrow).",
    ),
    CandidateSpec(
        key="multi_symbol_slice",
        name="Multi-Symbol Slice",
        target="Rust",
        src_globs=("backtest_engine/core/multi_symbol_slice.py",),
        baseline_file="p0-01_multi_symbol_slice.json",
        notes="Auffällig teuer in Baseline; Kandidat für Rust (Vectorisierung/Zero-copy).",
    ),
    CandidateSpec(
        key="symbol_data_slicer",
        name="Symbol Data Slicer",
        target="Rust",
        src_globs=("backtest_engine/core/symbol_data_slicer.py",),
        baseline_file="p0-01_symbol_data_slicer.json",
        notes="Hohe Call-Frequenz im Core-Loop; Rust kann Branching/Indexing beschleunigen.",
    ),
    CandidateSpec(
        key="event_engine",
        name="Event Engine",
        target="Rust",
        src_globs=("backtest_engine/core/event_engine.py",),
        baseline_file="p0-01_event_engine.json",
        notes="Core-Loop; sehr sensibel – Migration erst nach Interface-Spec + Tests.",
    ),
    CandidateSpec(
        key="portfolio",
        name="Portfolio",
        target="Rust",
        src_globs=("backtest_engine/core/portfolio.py",),
        baseline_file="p0-01_portfolio.json",
        notes="Stateful Hot-path; Ownership/Mutability muss sauber spezifiziert werden.",
    ),
    CandidateSpec(
        key="execution_simulator",
        name="Execution Simulator",
        target="Rust",
        src_globs=("backtest_engine/core/execution_simulator.py",),
        baseline_file="p0-01_execution_simulator.json",
        notes="Trade-Matching + Exits; gute Rust-Kandidatur nach klarer I/O-Spec.",
    ),
    CandidateSpec(
        key="slippage_and_fee",
        name="Slippage & Fee",
        target="Rust",
        src_globs=("backtest_engine/core/slippage_and_fee.py",),
        baseline_file="p0-01_slippage_and_fee.json",
        notes="Reine Mathematik; sehr gut als frühes, kleines Rust-Pilot-Modul.",
    ),
    CandidateSpec(
        key="rating",
        name="Rating Modules",
        target="Rust",
        src_globs=("backtest_engine/rating",),
        baseline_file="p0-01_rating.json",
        notes="Viele numerische Scores; geeignet für Rust, aber erst Schema/Output-CSV beachten.",
    ),
    CandidateSpec(
        key="optimizer",
        name="Optimizer (Final Selection / Robust Zone)",
        target="Julia",
        src_globs=("backtest_engine/optimizer",),
        baseline_file="p0-01_optimizer.json",
        notes="Explorativ/Research-lastig; Julia kann Iteration beschleunigen (aber FFI/Arrow klären).",
    ),
    CandidateSpec(
        key="walkforward",
        name="Walkforward (stubbed window)",
        target="Julia",
        src_globs=("backtest_engine/optimizer/walkforward.py",),
        baseline_file="p0-01_walkforward_stub.json",
        notes="Orchestrierung; primär I/O + Pipeline – eher später, nach Stabilisierung.",
    ),
    CandidateSpec(
        key="analysis_pipelines",
        name="Analysis Pipelines",
        target="Julia",
        src_globs=("backtest_engine/analysis",),
        baseline_file=None,
        notes="Research/Plots; Julia lohnt sich, aber Performance-Impact schwerer zu messen (noch keine Baseline).",
    ),
)


def build_report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for c in CANDIDATES:
        files = _iter_candidate_files(c.src_globs)
        types = _analyze_types(files)
        func_cov = round(types.function_coverage(), 1)
        param_cov = round(types.parameter_coverage(), 1)
        type_bucket = _type_bucket(func_cov, param_cov)

        impact_seconds: float | None = None
        perf_bucket = "Unknown"
        baseline_path: str | None = None
        if c.baseline_file is not None:
            p = BASELINES_DIR / c.baseline_file
            baseline_path = str(p.relative_to(ROOT))
            payload = _load_json(p)
            impact_seconds = _extract_impact_seconds(payload)
            perf_bucket = _perf_bucket(impact_seconds)

        priority = _priority_bucket(perf_bucket, type_bucket)

        rows.append(
            {
                "key": c.key,
                "name": c.name,
                "target": c.target,
                "src_globs": list(c.src_globs),
                "src_files": [str(p.relative_to(ROOT)) for p in files],
                "baseline": baseline_path,
                "impact_seconds": (
                    round(impact_seconds, 6)
                    if isinstance(impact_seconds, float)
                    else None
                ),
                "perf_bucket": perf_bucket,
                "function_coverage": func_cov,
                "parameter_coverage": param_cov,
                "type_bucket": type_bucket,
                "recommended_priority": priority,
                "notes": c.notes,
            }
        )

    # Sort: priority desc, then perf desc, then type desc, then name
    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    perf_rank = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    type_rank = {"High": 0, "Medium": 1, "Low": 2}

    def _sort_key(r: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            priority_rank.get(str(r["recommended_priority"]), 9),
            perf_rank.get(str(r["perf_bucket"]), 9),
            type_rank.get(str(r["type_bucket"]), 9),
            str(r["name"]),
        )

    rows.sort(key=_sort_key)

    return {
        "meta": {
            "generated_by": "tools/migration_candidates.py",
            "scope": "P0-04",
            "baselines_dir": str(BASELINES_DIR.relative_to(ROOT)),
            "src_root": str(SRC_ROOT.relative_to(ROOT)),
        },
        "candidates": rows,
        "rules": {
            "perf_bucket": {"High": ">= 1.0s", "Medium": ">= 0.15s", "Low": "< 0.15s"},
            "type_bucket": {
                "High": "return>=80% AND params>=90%",
                "Medium": "return>=50% AND params>=80%",
                "Low": "else",
            },
            "recommended_priority": "Derived from perf_bucket + type_bucket (conservative).",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Migration Candidates (P0-04)")
    lines.append("")
    lines.append(
        "Diese Liste priorisiert Migrations-Kandidaten für Rust/Julia basierend auf:"
    )
    lines.append("")
    lines.append(
        "- Performance-Baselines: `reports/performance_baselines/p0-01_*.json`"
    )
    lines.append(
        "- Type-Readiness: AST-basierte Type-Coverage (Return- und Parameter-Annotationen)"
    )
    lines.append("")
    lines.append(
        "Hinweis: Das ist ein Planungsartefakt. Die finale Reihenfolge muss zusätzlich"
    )
    lines.append(
        "Interfaces/Serialisierung, Golden-File-Determinismus und FFI-Risiken berücksichtigen."
    )
    lines.append("")

    lines.append("## Regeln")
    rules = report.get("rules", {})
    lines.append("")
    lines.append("- Perf-Bucket: High (>= 1.0s), Medium (>= 0.15s), Low (< 0.15s)")
    lines.append(
        "- Type-Bucket: High (Return >= 80% & Params >= 90%), Medium (Return >= 50% & Params >= 80%), sonst Low"
    )
    lines.append("- Recommended Priority: konservativ aus Perf + Type abgeleitet")
    lines.append("")

    lines.append("## Kandidaten")
    lines.append("")
    lines.append(
        "| Kandidat | Target | Perf | Impact (s) | Type | Return% | Param% | Priority | Baseline |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for r in report.get("candidates", []):
        baseline = r.get("baseline") or "-"
        impact = r.get("impact_seconds")
        impact_s = f"{impact:.6f}" if isinstance(impact, (int, float)) else "-"
        lines.append(
            "| {name} | {target} | {perf} | {impact} | {typeb} | {fcov} | {pcov} | {prio} | `{baseline}` |".format(
                name=str(r.get("name")),
                target=str(r.get("target")),
                perf=str(r.get("perf_bucket")),
                impact=impact_s,
                typeb=str(r.get("type_bucket")),
                fcov=str(r.get("function_coverage")),
                pcov=str(r.get("parameter_coverage")),
                prio=str(r.get("recommended_priority")),
                baseline=str(baseline),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for r in report.get("candidates", []):
        lines.append(f"- **{r.get('name')}**: {r.get('notes')}")

    lines.append("")
    lines.append("## Reproduzieren")
    lines.append("")
    lines.append(
        "- JSON (machine-readable): `tools/migration_candidates.py --format json`"
    )
    lines.append(
        "- Markdown (human-readable): `tools/migration_candidates.py --format md`"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-04 migration candidate report")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args()

    report = build_report()

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=False))
        return

    print(render_markdown(report))


if __name__ == "__main__":
    main()
