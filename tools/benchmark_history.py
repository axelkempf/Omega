"""
Benchmark History Tracking für Omega Trading System.

P3-12: Ermöglicht das Speichern, Laden und Analysieren von Benchmark-Ergebnissen
über Zeit. Dient zur Erkennung von Performance-Regressionen bei der FFI-Migration.

Features:
- JSON-basierte Speicherung von Benchmark-Ergebnissen
- Trend-Analyse über mehrere Commits/Releases
- Regression-Detection mit konfigurierbaren Schwellwerten
- Report-Generierung für CI/CD
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pfade
BENCHMARK_HISTORY_DIR = Path("reports/performance_baselines")
BENCHMARK_HISTORY_FILE = BENCHMARK_HISTORY_DIR / "benchmark_history.json"


@dataclass
class BenchmarkRun:
    """Einzelner Benchmark-Lauf."""

    name: str
    mean: float  # Sekunden
    stddev: float
    min: float
    max: float
    rounds: int
    iterations: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkRun:
        return cls(**data)

    @classmethod
    def from_pytest_benchmark(cls, benchmark: Dict[str, Any]) -> BenchmarkRun:
        """Erstellt BenchmarkRun aus pytest-benchmark JSON."""
        stats = benchmark["stats"]
        return cls(
            name=benchmark["name"],
            mean=stats["mean"],
            stddev=stats["stddev"],
            min=stats["min"],
            max=stats["max"],
            rounds=stats["rounds"],
            iterations=stats.get("iterations", 1),
        )


@dataclass
class BenchmarkSnapshot:
    """Snapshot aller Benchmarks zu einem Zeitpunkt."""

    timestamp: str
    commit_hash: str
    commit_message: str
    branch: str
    python_version: str
    benchmarks: List[BenchmarkRun]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["benchmarks"] = [b.to_dict() for b in self.benchmarks]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkSnapshot:
        benchmarks = [BenchmarkRun.from_dict(b) for b in data["benchmarks"]]
        return cls(
            timestamp=data["timestamp"],
            commit_hash=data["commit_hash"],
            commit_message=data["commit_message"],
            branch=data["branch"],
            python_version=data["python_version"],
            benchmarks=benchmarks,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegressionResult:
    """Ergebnis einer Regression-Analyse."""

    benchmark_name: str
    baseline_mean: float
    current_mean: float
    change_percent: float
    is_regression: bool
    threshold: float


class BenchmarkHistoryTracker:
    """
    Tracker für Benchmark-Historie.

    Speichert Benchmark-Ergebnisse über Zeit und ermöglicht
    Trend-Analyse und Regression-Detection.
    """

    def __init__(
        self,
        history_file: Path = BENCHMARK_HISTORY_FILE,
        max_history: int = 100,
    ):
        self.history_file = history_file
        self.max_history = max_history
        self._history: List[BenchmarkSnapshot] = []
        self._load_history()

    def _load_history(self) -> None:
        """Lädt bestehende Historie aus Datei."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self._history = [
                    BenchmarkSnapshot.from_dict(s) for s in data.get("snapshots", [])
                ]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load benchmark history: {e}")
                self._history = []
        else:
            self._history = []

    def _save_history(self) -> None:
        """Speichert Historie in Datei."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Begrenze Historie auf max_history Einträge
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "snapshots": [s.to_dict() for s in self._history],
        }

        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _get_git_info() -> Tuple[str, str, str]:
        """Holt Git-Informationen (commit hash, message, branch)."""
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_hash = "unknown"

        try:
            commit_message = (
                subprocess.check_output(
                    ["git", "log", "-1", "--pretty=%B"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
                .split("\n")[0][:100]
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_message = "unknown"

        try:
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            branch = "unknown"

        return commit_hash, commit_message, branch

    def add_snapshot(
        self,
        benchmarks: List[BenchmarkRun],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkSnapshot:
        """
        Fügt neuen Benchmark-Snapshot zur Historie hinzu.

        Args:
            benchmarks: Liste von BenchmarkRun Objekten.
            metadata: Optionale zusätzliche Metadaten.

        Returns:
            Der erstellte BenchmarkSnapshot.
        """
        import sys

        commit_hash, commit_message, branch = self._get_git_info()

        snapshot = BenchmarkSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            commit_hash=commit_hash,
            commit_message=commit_message,
            branch=branch,
            python_version=sys.version.split()[0],
            benchmarks=benchmarks,
            metadata=metadata or {},
        )

        self._history.append(snapshot)
        self._save_history()

        return snapshot

    def add_from_pytest_benchmark_json(
        self,
        json_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkSnapshot:
        """
        Fügt Snapshot aus pytest-benchmark JSON hinzu.

        Args:
            json_path: Pfad zur pytest-benchmark JSON-Datei.
            metadata: Optionale zusätzliche Metadaten.

        Returns:
            Der erstellte BenchmarkSnapshot.
        """
        with open(json_path) as f:
            data = json.load(f)

        benchmarks = [
            BenchmarkRun.from_pytest_benchmark(b) for b in data.get("benchmarks", [])
        ]

        return self.add_snapshot(benchmarks, metadata)

    def get_latest_snapshot(self) -> Optional[BenchmarkSnapshot]:
        """Gibt den neuesten Snapshot zurück."""
        return self._history[-1] if self._history else None

    def get_baseline(self, branch: str = "main") -> Optional[BenchmarkSnapshot]:
        """
        Gibt den neuesten Snapshot für einen Branch zurück.

        Args:
            branch: Name des Branches.

        Returns:
            Der neueste Snapshot für den Branch oder None.
        """
        for snapshot in reversed(self._history):
            if snapshot.branch == branch:
                return snapshot
        return None

    def detect_regressions(
        self,
        current: List[BenchmarkRun],
        baseline: Optional[BenchmarkSnapshot] = None,
        threshold_percent: float = 20.0,
    ) -> List[RegressionResult]:
        """
        Erkennt Performance-Regressionen.

        Args:
            current: Aktuelle Benchmark-Ergebnisse.
            baseline: Baseline zum Vergleich. Wenn None, wird Main-Branch verwendet.
            threshold_percent: Schwellwert für Regression (default: 20%).

        Returns:
            Liste von RegressionResult Objekten.
        """
        if baseline is None:
            baseline = self.get_baseline("main")

        if baseline is None:
            return []

        # Erstelle Lookup für Baseline-Benchmarks
        baseline_lookup = {b.name: b for b in baseline.benchmarks}

        results = []
        for bench in current:
            if bench.name not in baseline_lookup:
                continue

            baseline_bench = baseline_lookup[bench.name]
            baseline_mean = baseline_bench.mean
            current_mean = bench.mean

            if baseline_mean > 0:
                change_percent = ((current_mean - baseline_mean) / baseline_mean) * 100
            else:
                change_percent = 0.0

            is_regression = change_percent > threshold_percent

            results.append(
                RegressionResult(
                    benchmark_name=bench.name,
                    baseline_mean=baseline_mean,
                    current_mean=current_mean,
                    change_percent=change_percent,
                    is_regression=is_regression,
                    threshold=threshold_percent,
                )
            )

        return results

    def get_trend(
        self,
        benchmark_name: str,
        n_snapshots: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Gibt Trend für einen Benchmark zurück.

        Args:
            benchmark_name: Name des Benchmarks.
            n_snapshots: Anzahl der Snapshots für Trend.

        Returns:
            Liste von (timestamp, mean) Tupeln.
        """
        trend = []
        for snapshot in self._history[-n_snapshots:]:
            for bench in snapshot.benchmarks:
                if bench.name == benchmark_name:
                    trend.append((snapshot.timestamp, bench.mean))
                    break
        return trend

    def generate_report(
        self,
        current: List[BenchmarkRun],
        baseline: Optional[BenchmarkSnapshot] = None,
        threshold_percent: float = 20.0,
    ) -> str:
        """
        Generiert einen Markdown-Report.

        Args:
            current: Aktuelle Benchmark-Ergebnisse.
            baseline: Baseline zum Vergleich.
            threshold_percent: Schwellwert für Regression.

        Returns:
            Markdown-formatierter Report.
        """
        regressions = self.detect_regressions(current, baseline, threshold_percent)

        lines = []
        lines.append("# Benchmark Report\n")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}\n")

        # Summary
        regression_count = sum(1 for r in regressions if r.is_regression)
        improvement_count = sum(
            1 for r in regressions if r.change_percent < -threshold_percent
        )

        if regression_count > 0:
            lines.append(f"\n⚠️ **{regression_count} regression(s) detected**\n")
        else:
            lines.append("\n✅ **No significant regressions**\n")

        if improvement_count > 0:
            lines.append(f"🚀 **{improvement_count} improvement(s) detected**\n")

        # Details Table
        lines.append("\n## Benchmark Results\n")
        lines.append(
            "| Benchmark | Baseline (ms) | Current (ms) | Change | Status |"
        )
        lines.append("|-----------|---------------|--------------|--------|--------|")

        for r in sorted(regressions, key=lambda x: -abs(x.change_percent)):
            baseline_ms = r.baseline_mean * 1000
            current_ms = r.current_mean * 1000

            if r.is_regression:
                status = "⚠️ Regression"
            elif r.change_percent < -threshold_percent:
                status = "🚀 Improved"
            else:
                status = "✅ OK"

            change_str = f"{r.change_percent:+.1f}%"
            lines.append(
                f"| {r.benchmark_name} | {baseline_ms:.3f} | {current_ms:.3f} | {change_str} | {status} |"
            )

        # Current-only benchmarks (no baseline)
        baseline_names = {r.benchmark_name for r in regressions}
        new_benchmarks = [b for b in current if b.name not in baseline_names]

        if new_benchmarks:
            lines.append("\n## New Benchmarks (no baseline)\n")
            lines.append("| Benchmark | Mean (ms) | StdDev (ms) |")
            lines.append("|-----------|-----------|-------------|")
            for b in new_benchmarks:
                lines.append(f"| {b.name} | {b.mean * 1000:.3f} | {b.stddev * 1000:.3f} |")

        return "\n".join(lines)


# ==============================================================================
# CLI Interface
# ==============================================================================


def main():
    """CLI für Benchmark History Tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark History Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # add command
    add_parser = subparsers.add_parser("add", help="Add benchmark results")
    add_parser.add_argument("json_file", type=Path, help="pytest-benchmark JSON file")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("json_file", type=Path, help="pytest-benchmark JSON file")
    report_parser.add_argument(
        "--threshold", type=float, default=20.0, help="Regression threshold (%)"
    )

    # trend command
    trend_parser = subparsers.add_parser("trend", help="Show trend for benchmark")
    trend_parser.add_argument("benchmark_name", help="Name of benchmark")
    trend_parser.add_argument(
        "--n", type=int, default=10, help="Number of snapshots"
    )

    args = parser.parse_args()

    tracker = BenchmarkHistoryTracker()

    if args.command == "add":
        snapshot = tracker.add_from_pytest_benchmark_json(args.json_file)
        print(f"Added snapshot: {snapshot.commit_hash[:8]} ({len(snapshot.benchmarks)} benchmarks)")

    elif args.command == "report":
        with open(args.json_file) as f:
            data = json.load(f)
        benchmarks = [
            BenchmarkRun.from_pytest_benchmark(b) for b in data.get("benchmarks", [])
        ]
        report = tracker.generate_report(benchmarks, threshold_percent=args.threshold)
        print(report)

    elif args.command == "trend":
        trend = tracker.get_trend(args.benchmark_name, args.n)
        if trend:
            print(f"Trend for '{args.benchmark_name}':")
            for ts, mean in trend:
                print(f"  {ts}: {mean * 1000:.3f} ms")
        else:
            print(f"No data found for '{args.benchmark_name}'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
