"""Tests for agent_memory models."""
from __future__ import annotations

import pytest

from src.agent_memory.models import ChunkType, CodeChunk, SearchResult


class TestChunkType:
    """Tests for ChunkType enum."""

    def test_chunk_types_exist(self) -> None:
        """All expected chunk types are defined."""
        assert ChunkType.FUNCTION.value == "function"
        assert ChunkType.CLASS.value == "class"
        assert ChunkType.METHOD.value == "method"
        assert ChunkType.MODULE.value == "module"
        assert ChunkType.DOCSTRING.value == "docstring"
        assert ChunkType.COMMENT.value == "comment"
        assert ChunkType.ADR.value == "adr"
        assert ChunkType.DOCUMENTATION.value == "documentation"


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_create_minimal(self) -> None:
        """Create CodeChunk with minimal fields."""
        chunk = CodeChunk(
            id="test-123",
            content="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            file_path="test.py",
        )
        assert chunk.id == "test-123"
        assert chunk.content == "def foo(): pass"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.file_path == "test.py"
        assert chunk.name is None
        assert chunk.start_line is None
        assert chunk.end_line is None
        assert chunk.docstring is None
        assert chunk.callees == []
        assert chunk.metadata == {}

    def test_create_full(self) -> None:
        """Create CodeChunk with all fields."""
        chunk = CodeChunk(
            id="test-456",
            content="def bar(x: int) -> int:\n    return x * 2",
            chunk_type=ChunkType.FUNCTION,
            file_path="utils.py",
            name="bar",
            start_line=10,
            end_line=12,
            docstring="Doubles a number.",
            callees=["multiply"],
            metadata={"complexity": 1},
        )
        assert chunk.name == "bar"
        assert chunk.start_line == 10
        assert chunk.end_line == 12
        assert chunk.docstring == "Doubles a number."
        assert chunk.callees == ["multiply"]
        assert chunk.metadata == {"complexity": 1}

    def test_to_dict(self) -> None:
        """Test CodeChunk.to_dict() method."""
        chunk = CodeChunk(
            id="dict-test",
            content="pass",
            chunk_type=ChunkType.MODULE,
            file_path="mod.py",
            metadata={"key": "value"},
        )
        d = chunk.to_dict()
        assert d["id"] == "dict-test"
        assert d["chunk_type"] == "module"
        assert d["file_path"] == "mod.py"
        assert d["metadata"] == {"key": "value"}


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self) -> None:
        """Create SearchResult with chunk and score."""
        chunk = CodeChunk(
            id="result-1",
            content="test",
            chunk_type=ChunkType.FUNCTION,
            file_path="f.py",
        )
        result = SearchResult(chunk=chunk, score=0.85)
        assert result.chunk is chunk
        assert result.score == 0.85
        assert result.highlights == []

    def test_search_result_with_highlights(self) -> None:
        """Create SearchResult with highlights."""
        chunk = CodeChunk(
            id="result-2",
            content="highlighted content",
            chunk_type=ChunkType.DOCSTRING,
            file_path="doc.py",
        )
        result = SearchResult(
            chunk=chunk,
            score=0.92,
            highlights=["highlighted", "content"],
        )
        assert len(result.highlights) == 2
        assert "highlighted" in result.highlights
