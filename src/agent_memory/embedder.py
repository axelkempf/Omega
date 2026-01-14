"""Code embedding using sentence transformers.

This module provides embedding generation for code and documentation,
enabling semantic similarity search across the codebase.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Generates embeddings for code and documentation.

    Uses sentence-transformers models for generating vector representations
    of code snippets and documentation that can be used for semantic search.

    Args:
        model_name: Name of the sentence-transformers model to use.
            Defaults to "all-MiniLM-L6-v2" which provides a good balance
            of speed and quality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text string.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        return self.model.encode(text).tolist()

    def embed_code(self, code: str, context: str = "") -> list[float]:
        """Generate embedding for code with optional context.

        Combines code with context (like docstrings or comments) to create
        a richer embedding that captures both syntax and semantics.

        Args:
            code: The source code to embed.
            context: Optional context like docstring or comments.

        Returns:
            List of floats representing the embedding vector.
        """
        combined = f"{context}\n\n{code}" if context else code
        return self.embed_text(combined)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Uses batch processing for better performance when embedding
        many texts at once.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        return self.model.encode(texts).tolist()
