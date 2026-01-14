"""Data models for the RAG layer.

This module defines the core data structures used throughout the agent memory
system for representing code chunks, search results, and chunk types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(Enum):
    """Types of code chunks that can be indexed."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    ADR = "adr"
    DOCUMENTATION = "documentation"


@dataclass
class CodeChunk:
    """A semantic chunk of code with metadata and embedding.

    Attributes:
        id: Unique identifier (hash-based)
        file_path: Path to the source file
        chunk_type: Type of the chunk (function, class, etc.)
        content: Raw source code or text
        embedding: Vector representation for semantic search
        metadata: Additional key-value metadata
        name: Name of the function/class/module
        signature: Function/method signature
        docstring: Documentation string if present
        dependencies: List of imported modules/functions
        callers: Functions that call this one
        callees: Functions this one calls
    """

    id: str
    file_path: str
    chunk_type: ChunkType | str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    dependencies: list[str] = field(default_factory=list)
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Result from a semantic search operation.

    Attributes:
        chunk: The matched code chunk
        score: Similarity score (0-1, higher is better)
        highlights: Relevant text snippets for display
    """

    chunk: CodeChunk
    score: float
    highlights: list[str] = field(default_factory=list)
