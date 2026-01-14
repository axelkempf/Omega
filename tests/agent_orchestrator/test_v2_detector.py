"""Unit tests for V2Detector component.

Tests pattern matching, confidence scoring, and V2 context detection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent_orchestrator.v2_detector import (
    V2_CRATES,
    V2_KEYWORDS,
    V2_PATH_PATTERNS,
    V2DetectionResult,
    V2Detector,
)


class TestV2PathPatterns:
    """Test V2 path pattern definitions."""

    def test_rust_core_patterns_exist(self) -> None:
        """Verify rust_core patterns are defined."""
        assert "rust_core/" in V2_PATH_PATTERNS
        assert "rust_core/crates/" in V2_PATH_PATTERNS

    def test_python_bt_patterns_exist(self) -> None:
        """Verify python/bt patterns are defined."""
        assert "python/bt/" in V2_PATH_PATTERNS

    def test_v2_doc_patterns_exist(self) -> None:
        """Verify V2 documentation patterns are defined."""
        assert any("OMEGA_V2" in pattern for pattern in V2_PATH_PATTERNS)


class TestV2Keywords:
    """Test V2 keyword definitions."""

    def test_contains_core_keywords(self) -> None:
        """Verify core V2 keywords are present."""
        core_keywords = ["run_backtest", "BacktestConfig", "BacktestResult"]
        for keyword in core_keywords:
            assert keyword in V2_KEYWORDS, f"Missing keyword: {keyword}"

    def test_contains_ffi_keywords(self) -> None:
        """Verify FFI-related keywords are present."""
        ffi_keywords = ["PyO3", "maturin", "pyo3"]
        assert any(kw in V2_KEYWORDS for kw in ffi_keywords)


class TestV2Crates:
    """Test V2 crate definitions."""

    def test_contains_all_core_crates(self) -> None:
        """Verify all 10 V2 crates are defined."""
        expected_crates = [
            "types",
            "data",
            "indicators",
            "execution",
            "portfolio",
            "trade_mgmt",
            "strategy",
            "backtest",
            "metrics",
            "ffi",
        ]
        for crate in expected_crates:
            assert crate in V2_CRATES, f"Missing crate: {crate}"


class TestV2DetectionResult:
    """Test V2DetectionResult dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test default values."""
        result = V2DetectionResult(is_v2_context=False, confidence=0.0)
        assert result.is_v2_context is False
        assert result.confidence == 0.0
        assert result.detected_patterns == []
        assert result.affected_crates == []
        assert result.recommended_instructions == []

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields populated."""
        result = V2DetectionResult(
            is_v2_context=True,
            confidence=0.95,
            detected_patterns=["rust_core/", "python/bt/"],
            affected_crates=["types", "data"],
            recommended_instructions=["omega-v2-backtest.instructions.md"],
        )
        assert result.is_v2_context is True
        assert result.confidence == 0.95
        assert len(result.detected_patterns) == 2
        assert len(result.affected_crates) == 2


class TestV2Detector:
    """Test V2Detector class."""

    @pytest.fixture
    def detector(self) -> V2Detector:
        """Create a fresh detector instance."""
        return V2Detector()

    def test_detect_empty_input(self, detector: V2Detector) -> None:
        """Test detection with empty input."""
        result = detector.detect(files=[], content="")
        assert result.is_v2_context is False
        assert result.confidence == 0.0

    def test_detect_v1_only_content(self, detector: V2Detector) -> None:
        """Test detection with V1-only content."""
        v1_content = """
        from src.hf_engine.core import TradeManager
        from src.strategies.mean_reversion import Strategy
        """
        result = detector.detect(files=[], content=v1_content)
        assert result.is_v2_context is False
        assert result.confidence < 0.5

    def test_detect_rust_core_path(self, detector: V2Detector) -> None:
        """Test detection with rust_core paths."""
        files = [Path("rust_core/crates/types/src/lib.rs")]
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True
        assert result.confidence >= 0.7
        assert "types" in result.affected_crates

    def test_detect_python_bt_path(self, detector: V2Detector) -> None:
        """Test detection with python/bt paths."""
        files = [Path("python/bt/runner.py")]
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True
        assert result.confidence >= 0.6

    def test_detect_v2_keywords_in_content(self, detector: V2Detector) -> None:
        """Test detection with V2 keywords in content."""
        v2_content = """
        use pyo3::prelude::*;
        
        #[pyfunction]
        fn run_backtest(config_json: &str) -> PyResult<String> {
            let config: BacktestConfig = serde_json::from_str(config_json)?;
            Ok(serde_json::to_string(&result)?)
        }
        """
        result = detector.detect(files=[], content=v2_content)
        assert result.is_v2_context is True
        assert result.confidence >= 0.5

    def test_detect_mixed_v1_v2_prefers_v2(self, detector: V2Detector) -> None:
        """Test that V2 context is detected when both V1 and V2 patterns present."""
        files = [
            Path("src/hf_engine/core/execution.py"),  # V1
            Path("rust_core/crates/execution/src/lib.rs"),  # V2
        ]
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True
        assert "execution" in result.affected_crates

    def test_detect_crate_extraction(self, detector: V2Detector) -> None:
        """Test that affected crates are correctly extracted."""
        files = [
            Path("rust_core/crates/types/src/lib.rs"),
            Path("rust_core/crates/data/src/loader.rs"),
            Path("rust_core/crates/execution/src/fill.rs"),
        ]
        result = detector.detect(files=files, content="")
        assert "types" in result.affected_crates
        assert "data" in result.affected_crates
        assert "execution" in result.affected_crates

    def test_detect_recommendations(self, detector: V2Detector) -> None:
        """Test that appropriate instructions are recommended."""
        files = [Path("rust_core/crates/ffi/src/lib.rs")]
        result = detector.detect(files=files, content="")
        assert len(result.recommended_instructions) > 0
        assert any("v2" in instr.lower() for instr in result.recommended_instructions)

    def test_detect_v2_plan_documents(self, detector: V2Detector) -> None:
        """Test detection of V2 plan documents."""
        files = [Path("docs/OMEGA_V2_ARCHITECTURE_PLAN.md")]
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True

    def test_confidence_scaling(self, detector: V2Detector) -> None:
        """Test that confidence scales with evidence."""
        # Single pattern
        result1 = detector.detect(files=[Path("rust_core/Cargo.toml")], content="")

        # Multiple patterns
        result2 = detector.detect(
            files=[
                Path("rust_core/crates/types/src/lib.rs"),
                Path("rust_core/crates/data/src/lib.rs"),
                Path("python/bt/runner.py"),
            ],
            content="run_backtest BacktestConfig PyO3",
        )

        assert result2.confidence >= result1.confidence

    def test_detect_golden_file_paths(self, detector: V2Detector) -> None:
        """Test detection of golden test file paths."""
        files = [Path("python/bt/tests/golden/expected/scenario_1/trades.json")]
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True


class TestV2DetectorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def detector(self) -> V2Detector:
        """Create a fresh detector instance."""
        return V2Detector()

    def test_handles_none_content(self, detector: V2Detector) -> None:
        """Test handling of None content."""
        # Should not raise
        result = detector.detect(files=[], content=None)  # type: ignore
        assert result.is_v2_context is False

    def test_handles_empty_file_list(self, detector: V2Detector) -> None:
        """Test handling of empty file list."""
        result = detector.detect(files=[], content="run_backtest")
        assert result.is_v2_context is True  # Keywords still detected

    def test_handles_path_strings(self, detector: V2Detector) -> None:
        """Test handling of string paths instead of Path objects."""
        files = ["rust_core/crates/types/src/lib.rs"]  # type: ignore
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True

    def test_case_insensitive_keywords(self, detector: V2Detector) -> None:
        """Test that keyword detection is case-appropriate."""
        # Rust types are typically CamelCase
        result = detector.detect(files=[], content="BacktestConfig BacktestResult")
        assert result.is_v2_context is True

    def test_partial_path_matching(self, detector: V2Detector) -> None:
        """Test that partial paths are handled correctly."""
        # Should match even with absolute paths
        files = [Path("/Users/axelkempf/Omega/rust_core/crates/types/src/lib.rs")]
        result = detector.detect(files=files, content="")
        assert result.is_v2_context is True
