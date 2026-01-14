"""RAG layer for intelligent codebase search.

This module provides Retrieval-Augmented Generation capabilities for semantic
search across the Omega codebase, enabling AI agents to find relevant code,
documentation, and stored knowledge.

Components:
    - RAGLayer: Main interface for semantic search and knowledge storage
    - CodeEmbedder: Generates embeddings using sentence-transformers
    - CodeIndexer: AST-based code chunking and indexing
    - CodeChunk: Semantic unit of code with metadata
    - ChunkType: Enum for categorizing chunk types
"""

from __future__ import annotations

from .models import ChunkType, CodeChunk, SearchResult
from .embedder import CodeEmbedder
from .indexer import CodeIndexer
from .rag import RAGLayer

__all__ = [
    "RAGLayer",
    "CodeChunk",
    "ChunkType",
    "SearchResult",
    "CodeEmbedder",
    "CodeIndexer",
]
