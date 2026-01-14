"""Tests for CodeEmbedder.

Note: These tests require sentence-transformers to be installed.
Run with: pip install -e .[agent]
"""
from __future__ import annotations

import pytest

# Mark all tests as requiring optional dependencies
pytestmark = pytest.mark.skipif(
    True,  # Will be replaced with actual import check
    reason="sentence-transformers not installed",
)

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Update skipif with actual availability
pytestmark = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed (pip install -e .[agent])",
)


@pytest.fixture
def embedder():
    """Create CodeEmbedder instance."""
    from src.agent_memory.embedder import CodeEmbedder

    return CodeEmbedder()


class TestCodeEmbedder:
    """Tests for CodeEmbedder class."""

    def test_model_name_default(self, embedder) -> None:
        """Test default model name."""
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_model_lazy_loading(self, embedder) -> None:
        """Test that model is loaded lazily."""
        # Model should not be loaded until first use
        assert embedder._model is None

    def test_embed_text_returns_list(self, embedder) -> None:
        """Test embed_text returns a list of floats."""
        embedding = embedder.embed_text("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_dimension(self, embedder) -> None:
        """Test embedding dimension is correct for model."""
        embedding = embedder.embed_text("Test text")

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert len(embedding) == 384

    def test_embed_code_includes_metadata(self, embedder) -> None:
        """Test embed_code prepends context."""
        code = "def foo(): pass"
        embedding = embedder.embed_code(code, "foo", "function")

        # Should still return valid embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_embed_batch_multiple(self, embedder) -> None:
        """Test embed_batch with multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_embed_batch_empty(self, embedder) -> None:
        """Test embed_batch with empty list."""
        embeddings = embedder.embed_batch([])
        assert embeddings == []

    def test_similar_text_similar_embeddings(self, embedder) -> None:
        """Test that similar texts produce similar embeddings."""
        import numpy as np

        emb1 = embedder.embed_text("Calculate the sum of two numbers")
        emb2 = embedder.embed_text("Compute the addition of two values")
        emb3 = embedder.embed_text("The quick brown fox jumps over the lazy dog")

        # Cosine similarity helper
        def cosine_sim(a: list, b: list) -> float:
            a_arr = np.array(a)
            b_arr = np.array(b)
            return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

        sim_related = cosine_sim(emb1, emb2)
        sim_unrelated = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_related > sim_unrelated
