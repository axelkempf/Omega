"""Tests for RAGLayer integration.

Note: These tests require chromadb and sentence-transformers.
Run with: pip install -e .[agent]
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Check for optional dependencies
DEPS_AVAILABLE = False
try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    DEPS_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="chromadb and/or sentence-transformers not installed",
)


@pytest.fixture
def temp_persist_dir(tmp_path: Path) -> Path:
    """Create temporary persistence directory."""
    return tmp_path / "agent_memory"


@pytest.fixture
def rag_layer(temp_persist_dir: Path):
    """Create RAGLayer with temporary persistence."""
    from src.agent_memory.rag import RAGLayer

    return RAGLayer(persist_directory=temp_persist_dir)


@pytest.fixture
def sample_codebase(tmp_path: Path) -> Path:
    """Create a sample codebase for indexing."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create sample files
    utils_file = src_dir / "utils.py"
    utils_file.write_text('''"""Utility functions for calculations."""

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two numbers."""
    return a * b
''')

    models_file = src_dir / "models.py"
    models_file.write_text('''"""Data models."""

class User:
    """Represents a user in the system."""

    def __init__(self, name: str, age: int):
        """Initialize user with name and age."""
        self.name = name
        self.age = age

    def greet(self) -> str:
        """Return a greeting message."""
        return f"Hello, {self.name}!"
''')

    return src_dir


class TestRAGLayerInit:
    """Tests for RAGLayer initialization."""

    def test_creates_persist_directory(self, temp_persist_dir: Path) -> None:
        """Test that persistence directory is created."""
        from src.agent_memory.rag import RAGLayer

        rag = RAGLayer(persist_directory=temp_persist_dir)
        # Access client to trigger creation
        _ = rag.client
        assert temp_persist_dir.exists()

    def test_default_persist_directory(self) -> None:
        """Test default persistence directory path."""
        from src.agent_memory.rag import RAGLayer

        rag = RAGLayer()
        expected = Path("var/agent_memory")
        assert rag.persist_directory == expected


class TestRAGLayerIndexing:
    """Tests for codebase indexing."""

    def test_index_codebase(
        self, rag_layer, sample_codebase: Path
    ) -> None:
        """Test indexing a sample codebase."""
        count = rag_layer.index_codebase(sample_codebase)
        assert count > 0

    def test_index_codebase_creates_collection(
        self, rag_layer, sample_codebase: Path
    ) -> None:
        """Test that indexing creates code collection."""
        rag_layer.index_codebase(sample_codebase)

        # Collection should exist and have items
        collection = rag_layer.code_collection
        assert collection.count() > 0


class TestRAGLayerSearch:
    """Tests for semantic search."""

    def test_semantic_search_finds_relevant(
        self, rag_layer, sample_codebase: Path
    ) -> None:
        """Test semantic search finds relevant results."""
        rag_layer.index_codebase(sample_codebase)

        results = rag_layer.semantic_search("calculate sum", n_results=5)

        assert len(results) > 0
        # Should find calculate_sum function
        found_names = [r.chunk.name for r in results]
        assert any("sum" in (name or "").lower() for name in found_names)

    def test_semantic_search_empty_codebase(self, rag_layer) -> None:
        """Test semantic search on empty codebase."""
        results = rag_layer.semantic_search("anything", n_results=5)
        assert results == []

    def test_semantic_search_with_filter(
        self, rag_layer, sample_codebase: Path
    ) -> None:
        """Test semantic search with chunk type filter."""
        rag_layer.index_codebase(sample_codebase)

        results = rag_layer.semantic_search(
            "user",
            n_results=10,
            chunk_types=["class"],
        )

        # All results should be class chunks
        for r in results:
            assert r.chunk.chunk_type.value == "class"


class TestRAGLayerSimilar:
    """Tests for find_similar functionality."""

    def test_find_similar_code(
        self, rag_layer, sample_codebase: Path
    ) -> None:
        """Test finding similar code."""
        rag_layer.index_codebase(sample_codebase)

        code = "def add(x, y): return x + y"
        results = rag_layer.find_similar(code, n_results=3)

        assert len(results) > 0
        # Should find calculate_sum as similar
        found_names = [r.chunk.name for r in results]
        assert any("sum" in (name or "").lower() for name in found_names)


class TestRAGLayerMemory:
    """Tests for remember/recall functionality."""

    def test_remember_and_recall(self, rag_layer) -> None:
        """Test storing and retrieving information."""
        key = "test_decision"
        content = "We decided to use ChromaDB for vector storage."

        rag_layer.remember(key, content, metadata={"type": "decision"})
        recalled = rag_layer.recall(key)

        assert recalled is not None
        assert recalled["content"] == content
        assert recalled["metadata"]["type"] == "decision"

    def test_recall_nonexistent(self, rag_layer) -> None:
        """Test recalling non-existent key."""
        recalled = rag_layer.recall("nonexistent_key")
        assert recalled is None

    def test_remember_overwrites(self, rag_layer) -> None:
        """Test that remember overwrites existing key."""
        key = "overwrite_test"

        rag_layer.remember(key, "First value")
        rag_layer.remember(key, "Second value")

        recalled = rag_layer.recall(key)
        assert recalled["content"] == "Second value"


class TestRAGLayerContext:
    """Tests for module context retrieval."""

    def test_get_module_context(
        self, rag_layer, sample_codebase: Path
    ) -> None:
        """Test getting module context."""
        rag_layer.index_codebase(sample_codebase)

        context = rag_layer.get_module_context("utils")

        assert len(context) > 0
        # Should find chunks from utils.py
        file_paths = [c.file_path for c in context]
        assert any("utils" in p for p in file_paths)
