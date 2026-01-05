"""
Tests für Benchmark History Tracking (P3-12).

Testet die BenchmarkHistoryTracker-Klasse und zugehörige Funktionen.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List
from unittest.mock import patch

import pytest

# Import from tools - adjust path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
from tools.benchmark_history import (
    BenchmarkHistoryTracker,
    BenchmarkRun,
    BenchmarkSnapshot,
    RegressionResult,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_benchmark_run() -> BenchmarkRun:
    """Erstellt einen Sample-BenchmarkRun."""
    return BenchmarkRun(
        name="test_benchmark",
        mean=0.001,
        stddev=0.0001,
        min=0.0009,
        max=0.0011,
        rounds=100,
        iterations=10,
    )


@pytest.fixture
def sample_benchmark_runs() -> List[BenchmarkRun]:
    """Erstellt mehrere Sample-BenchmarkRuns."""
    return [
        BenchmarkRun(
            name="bench_indicator_ema",
            mean=0.0005,
            stddev=0.00005,
            min=0.0004,
            max=0.0006,
            rounds=100,
        ),
        BenchmarkRun(
            name="bench_indicator_rsi",
            mean=0.0008,
            stddev=0.00008,
            min=0.0007,
            max=0.0009,
            rounds=100,
        ),
        BenchmarkRun(
            name="bench_scoring_rating",
            mean=0.002,
            stddev=0.0002,
            min=0.0018,
            max=0.0022,
            rounds=100,
        ),
    ]


@pytest.fixture
def sample_snapshot(sample_benchmark_runs: List[BenchmarkRun]) -> BenchmarkSnapshot:
    """Erstellt einen Sample-BenchmarkSnapshot."""
    return BenchmarkSnapshot(
        timestamp=datetime.utcnow().isoformat(),
        commit_hash="abc123def456",
        commit_message="Test commit",
        branch="main",
        python_version=sys.version.split()[0],
        benchmarks=sample_benchmark_runs,
        metadata={"test": True},
    )


@pytest.fixture
def temp_history_file(tmp_path: Path) -> Path:
    """Erstellt eine temporäre History-Datei."""
    return tmp_path / "benchmark_history.json"


@pytest.fixture
def tracker(temp_history_file: Path) -> BenchmarkHistoryTracker:
    """Erstellt einen Tracker mit temporärer Datei."""
    return BenchmarkHistoryTracker(history_file=temp_history_file)


@pytest.fixture
def pytest_benchmark_json(tmp_path: Path, sample_benchmark_runs: List[BenchmarkRun]) -> Path:
    """Erstellt eine pytest-benchmark JSON-Datei."""
    json_path = tmp_path / "benchmark_results.json"

    data = {
        "machine_info": {"node": "test-node"},
        "commit_info": {"id": "abc123"},
        "benchmarks": [
            {
                "name": run.name,
                "fullname": f"tests/benchmarks/test_performance.py::{run.name}",
                "stats": {
                    "mean": run.mean,
                    "stddev": run.stddev,
                    "min": run.min,
                    "max": run.max,
                    "rounds": run.rounds,
                    "iterations": run.iterations,
                },
            }
            for run in sample_benchmark_runs
        ],
    }

    with open(json_path, "w") as f:
        json.dump(data, f)

    return json_path


# ==============================================================================
# BenchmarkRun Tests
# ==============================================================================


class TestBenchmarkRun:
    """Tests für BenchmarkRun Dataclass."""

    def test_creation(self, sample_benchmark_run: BenchmarkRun):
        """Test BenchmarkRun Erstellung."""
        assert sample_benchmark_run.name == "test_benchmark"
        assert sample_benchmark_run.mean == 0.001
        assert sample_benchmark_run.rounds == 100

    def test_to_dict(self, sample_benchmark_run: BenchmarkRun):
        """Test Serialisierung zu Dict."""
        data = sample_benchmark_run.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test_benchmark"
        assert data["mean"] == 0.001

    def test_from_dict(self, sample_benchmark_run: BenchmarkRun):
        """Test Deserialisierung von Dict."""
        data = sample_benchmark_run.to_dict()
        restored = BenchmarkRun.from_dict(data)

        assert restored.name == sample_benchmark_run.name
        assert restored.mean == sample_benchmark_run.mean
        assert restored.stddev == sample_benchmark_run.stddev

    def test_from_pytest_benchmark(self):
        """Test Erstellung aus pytest-benchmark Format."""
        pytest_data = {
            "name": "bench_test",
            "stats": {
                "mean": 0.002,
                "stddev": 0.0002,
                "min": 0.0018,
                "max": 0.0022,
                "rounds": 50,
                "iterations": 5,
            },
        }

        run = BenchmarkRun.from_pytest_benchmark(pytest_data)
        assert run.name == "bench_test"
        assert run.mean == 0.002
        assert run.rounds == 50
        assert run.iterations == 5

    def test_roundtrip_serialization(self, sample_benchmark_run: BenchmarkRun):
        """Test vollständige Serialisierung/Deserialisierung."""
        data = sample_benchmark_run.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = BenchmarkRun.from_dict(restored_data)

        assert restored.name == sample_benchmark_run.name
        assert restored.mean == sample_benchmark_run.mean


# ==============================================================================
# BenchmarkSnapshot Tests
# ==============================================================================


class TestBenchmarkSnapshot:
    """Tests für BenchmarkSnapshot Dataclass."""

    def test_creation(self, sample_snapshot: BenchmarkSnapshot):
        """Test BenchmarkSnapshot Erstellung."""
        assert sample_snapshot.branch == "main"
        assert len(sample_snapshot.benchmarks) == 3
        assert sample_snapshot.metadata == {"test": True}

    def test_to_dict(self, sample_snapshot: BenchmarkSnapshot):
        """Test Serialisierung zu Dict."""
        data = sample_snapshot.to_dict()
        assert isinstance(data, dict)
        assert data["branch"] == "main"
        assert len(data["benchmarks"]) == 3

    def test_from_dict(self, sample_snapshot: BenchmarkSnapshot):
        """Test Deserialisierung von Dict."""
        data = sample_snapshot.to_dict()
        restored = BenchmarkSnapshot.from_dict(data)

        assert restored.branch == sample_snapshot.branch
        assert len(restored.benchmarks) == len(sample_snapshot.benchmarks)
        assert restored.metadata == sample_snapshot.metadata

    def test_roundtrip_serialization(self, sample_snapshot: BenchmarkSnapshot):
        """Test vollständige Serialisierung/Deserialisierung."""
        data = sample_snapshot.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = BenchmarkSnapshot.from_dict(restored_data)

        assert restored.commit_hash == sample_snapshot.commit_hash
        assert restored.benchmarks[0].name == sample_snapshot.benchmarks[0].name


# ==============================================================================
# BenchmarkHistoryTracker Tests
# ==============================================================================


class TestBenchmarkHistoryTracker:
    """Tests für BenchmarkHistoryTracker."""

    def test_initialization_empty(self, tracker: BenchmarkHistoryTracker):
        """Test Initialisierung mit leerer Historie."""
        assert tracker.get_latest_snapshot() is None

    def test_add_snapshot(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Hinzufügen eines Snapshots."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("abc123", "Test commit", "main"),
        ):
            snapshot = tracker.add_snapshot(sample_benchmark_runs)

        assert snapshot.commit_hash == "abc123"
        assert len(snapshot.benchmarks) == 3
        assert tracker.get_latest_snapshot() == snapshot

    def test_add_from_pytest_benchmark_json(
        self,
        tracker: BenchmarkHistoryTracker,
        pytest_benchmark_json: Path,
    ):
        """Test Hinzufügen aus pytest-benchmark JSON."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("def456", "Another commit", "feature"),
        ):
            snapshot = tracker.add_from_pytest_benchmark_json(pytest_benchmark_json)

        assert snapshot.commit_hash == "def456"
        assert len(snapshot.benchmarks) == 3

    def test_persistence(
        self,
        temp_history_file: Path,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Persistenz über Neuladen."""
        # Erstelle Tracker und füge Snapshot hinzu
        tracker1 = BenchmarkHistoryTracker(history_file=temp_history_file)
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("abc123", "Commit 1", "main"),
        ):
            tracker1.add_snapshot(sample_benchmark_runs)

        # Erstelle neuen Tracker und prüfe ob Daten geladen wurden
        tracker2 = BenchmarkHistoryTracker(history_file=temp_history_file)
        latest = tracker2.get_latest_snapshot()

        assert latest is not None
        assert latest.commit_hash == "abc123"
        assert len(latest.benchmarks) == 3

    def test_max_history_limit(
        self,
        temp_history_file: Path,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Max-History-Limit."""
        tracker = BenchmarkHistoryTracker(
            history_file=temp_history_file, max_history=3
        )

        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("commit1", "C1", "main"),
        ):
            for i in range(5):
                with patch.object(
                    BenchmarkHistoryTracker,
                    "_get_git_info",
                    return_value=(f"commit{i}", f"C{i}", "main"),
                ):
                    tracker.add_snapshot(sample_benchmark_runs)

        # Neuladen und prüfen ob nur die letzten 3 vorhanden sind
        tracker2 = BenchmarkHistoryTracker(
            history_file=temp_history_file, max_history=3
        )
        assert len(tracker2._history) <= 3

    def test_get_baseline(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Baseline-Ermittlung für Branch."""
        # Füge Snapshots für verschiedene Branches hinzu
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("main1", "Main commit 1", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("feat1", "Feature commit", "feature"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("main2", "Main commit 2", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        baseline = tracker.get_baseline("main")
        assert baseline is not None
        assert baseline.commit_hash == "main2"

        feature_baseline = tracker.get_baseline("feature")
        assert feature_baseline is not None
        assert feature_baseline.commit_hash == "feat1"


# ==============================================================================
# Regression Detection Tests
# ==============================================================================


class TestRegressionDetection:
    """Tests für Regression-Erkennung."""

    def test_no_regression(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test ohne Regression."""
        # Füge Baseline hinzu
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        # Gleiche Werte -> keine Regression
        results = tracker.detect_regressions(sample_benchmark_runs)

        for r in results:
            assert not r.is_regression
            assert abs(r.change_percent) < 0.1

    def test_detect_regression(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Regression-Erkennung."""
        # Füge Baseline hinzu
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        # Erstelle langsamere Version (50% langsamer)
        slower_runs = [
            BenchmarkRun(
                name=r.name,
                mean=r.mean * 1.5,  # 50% langsamer
                stddev=r.stddev,
                min=r.min * 1.5,
                max=r.max * 1.5,
                rounds=r.rounds,
            )
            for r in sample_benchmark_runs
        ]

        results = tracker.detect_regressions(slower_runs, threshold_percent=20.0)

        for r in results:
            assert r.is_regression
            assert r.change_percent == pytest.approx(50.0, rel=0.1)

    def test_detect_improvement(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Verbesserung-Erkennung."""
        # Füge Baseline hinzu
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        # Erstelle schnellere Version (50% schneller)
        faster_runs = [
            BenchmarkRun(
                name=r.name,
                mean=r.mean * 0.5,  # 50% schneller
                stddev=r.stddev,
                min=r.min * 0.5,
                max=r.max * 0.5,
                rounds=r.rounds,
            )
            for r in sample_benchmark_runs
        ]

        results = tracker.detect_regressions(faster_runs, threshold_percent=20.0)

        for r in results:
            assert not r.is_regression
            assert r.change_percent == pytest.approx(-50.0, rel=0.1)

    def test_threshold_boundary(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Schwellwert-Grenze."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        # Genau 20% langsamer (Grenzfall)
        boundary_runs = [
            BenchmarkRun(
                name=r.name,
                mean=r.mean * 1.20,  # Genau 20%
                stddev=r.stddev,
                min=r.min,
                max=r.max,
                rounds=r.rounds,
            )
            for r in sample_benchmark_runs
        ]

        # Mit 20% Threshold: 20% ist keine Regression (>20 wäre es)
        results = tracker.detect_regressions(boundary_runs, threshold_percent=20.0)

        for r in results:
            assert not r.is_regression


# ==============================================================================
# Trend Analysis Tests
# ==============================================================================


class TestTrendAnalysis:
    """Tests für Trend-Analyse."""

    def test_get_trend(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Trend-Ermittlung."""
        # Füge mehrere Snapshots hinzu
        for i in range(5):
            modified_runs = [
                BenchmarkRun(
                    name=r.name,
                    mean=r.mean * (1 + i * 0.1),  # Steigender Trend
                    stddev=r.stddev,
                    min=r.min,
                    max=r.max,
                    rounds=r.rounds,
                )
                for r in sample_benchmark_runs
            ]
            with patch.object(
                BenchmarkHistoryTracker,
                "_get_git_info",
                return_value=(f"commit{i}", f"Commit {i}", "main"),
            ):
                tracker.add_snapshot(modified_runs)

        trend = tracker.get_trend("bench_indicator_ema", n_snapshots=5)

        assert len(trend) == 5
        # Prüfe steigenden Trend
        means = [t[1] for t in trend]
        for i in range(len(means) - 1):
            assert means[i + 1] > means[i]

    def test_get_trend_nonexistent_benchmark(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Trend für nicht existierenden Benchmark."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("commit1", "Commit", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        trend = tracker.get_trend("nonexistent_benchmark")
        assert len(trend) == 0


# ==============================================================================
# Report Generation Tests
# ==============================================================================


class TestReportGeneration:
    """Tests für Report-Generierung."""

    def test_generate_report_no_regression(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Report ohne Regressionen."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        report = tracker.generate_report(sample_benchmark_runs)

        assert "# Benchmark Report" in report
        assert "✅ **No significant regressions**" in report
        assert "bench_indicator_ema" in report

    def test_generate_report_with_regression(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Report mit Regressionen."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs)

        slower_runs = [
            BenchmarkRun(
                name=r.name,
                mean=r.mean * 1.5,
                stddev=r.stddev,
                min=r.min,
                max=r.max,
                rounds=r.rounds,
            )
            for r in sample_benchmark_runs
        ]

        report = tracker.generate_report(slower_runs)

        assert "# Benchmark Report" in report
        assert "⚠️" in report
        assert "regression(s) detected" in report

    def test_generate_report_new_benchmarks(
        self,
        tracker: BenchmarkHistoryTracker,
        sample_benchmark_runs: List[BenchmarkRun],
    ):
        """Test Report mit neuen Benchmarks ohne Baseline."""
        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot(sample_benchmark_runs[:1])  # Nur ein Benchmark

        report = tracker.generate_report(sample_benchmark_runs)

        assert "## New Benchmarks" in report
        # Die anderen Benchmarks sollten als neu erscheinen


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests für Randfälle."""

    def test_empty_history_regression_detection(
        self, tracker: BenchmarkHistoryTracker, sample_benchmark_runs: List[BenchmarkRun]
    ):
        """Test Regression-Detection ohne Historie."""
        results = tracker.detect_regressions(sample_benchmark_runs)
        assert len(results) == 0

    def test_corrupted_history_file(self, temp_history_file: Path):
        """Test mit korrupter History-Datei."""
        # Schreibe ungültiges JSON
        with open(temp_history_file, "w") as f:
            f.write("not valid json {{{")

        # Sollte nicht abstürzen
        tracker = BenchmarkHistoryTracker(history_file=temp_history_file)
        assert tracker.get_latest_snapshot() is None

    def test_zero_baseline_mean(
        self, tracker: BenchmarkHistoryTracker
    ):
        """Test mit Baseline-Mean von 0."""
        zero_run = BenchmarkRun(
            name="zero_bench",
            mean=0.0,
            stddev=0.0,
            min=0.0,
            max=0.0,
            rounds=1,
        )

        with patch.object(
            BenchmarkHistoryTracker,
            "_get_git_info",
            return_value=("baseline", "Baseline", "main"),
        ):
            tracker.add_snapshot([zero_run])

        current_runs = [
            BenchmarkRun(
                name="zero_bench",
                mean=0.001,
                stddev=0.0001,
                min=0.0009,
                max=0.0011,
                rounds=1,
            )
        ]

        # Sollte nicht durch Division durch 0 abstürzen
        results = tracker.detect_regressions(current_runs)
        assert len(results) == 1
        assert results[0].change_percent == 0.0
