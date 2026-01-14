"""Main RAG interface for codebase search.

This module provides the main Retrieval-Augmented Generation interface
for semantic search across the Omega codebase and knowledge storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .embedder import CodeEmbedder
from .indexer import CodeIndexer
from .models import ChunkType, CodeChunk, SearchResult

logger = logging.getLogger(__name__)


class RAGLayer:
    """Retrieval-Augmented Generation for codebase search.

    Provides semantic search capabilities over the codebase using ChromaDB
    for vector storage and sentence-transformers for embeddings.

    Args:
        persist_dir: Directory for ChromaDB persistence.
        embedding_model: Name of the sentence-transformers model.
    """

    def __init__(
        self,
        persist_dir: Path | str = Path("var/agent_memory"),
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._code_collection = None
        self._docs_collection = None

        # Components
        self.embedder = CodeEmbedder(model_name=embedding_model)
        self.indexer = CodeIndexer(self.embedder)

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                logger.info(f"Initializing ChromaDB at {self.persist_dir}")
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False)
                )
            except ImportError as e:
                raise ImportError(
                    "chromadb is required for RAG functionality. "
                    "Install with: pip install chromadb"
                ) from e
        return self._client

    @property
    def code_collection(self):
        """Get or create the code chunks collection."""
        if self._code_collection is None:
            self._code_collection = self.client.get_or_create_collection(
                name="code_chunks",
                metadata={"description": "Code chunks from the Omega codebase"}
            )
        return self._code_collection

    @property
    def docs_collection(self):
        """Get or create the documentation collection."""
        if self._docs_collection is None:
            self._docs_collection = self.client.get_or_create_collection(
                name="documentation",
                metadata={"description": "Documentation, ADRs, and stored knowledge"}
            )
        return self._docs_collection

    def index_codebase(
        self,
        src_path: Path | str = Path("src"),
        clear_existing: bool = False
    ) -> int:
        """Index the entire codebase.

        Args:
            src_path: Root directory to index.
            clear_existing: If True, clear existing index before re-indexing.

        Returns:
            Number of chunks indexed.
        """
        src_path = Path(src_path)
        logger.info(f"Indexing codebase at {src_path}")

        if clear_existing:
            logger.info("Clearing existing code index")
            self.client.delete_collection("code_chunks")
            self._code_collection = None

        chunks = self.indexer.index_directory(src_path)

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.code_collection.add(
                ids=[chunk.id for chunk in batch],
                embeddings=[chunk.embedding for chunk in batch],
                metadatas=[{
                    "file_path": chunk.file_path,
                    "chunk_type": chunk.chunk_type.value if isinstance(chunk.chunk_type, ChunkType) else chunk.chunk_type,
                    "name": chunk.name or "",
                    "signature": chunk.signature or "",
                    "docstring": (chunk.docstring or "")[:500],
                    "lineno": chunk.metadata.get("lineno", 0)
                } for chunk in batch],
                documents=[chunk.content for chunk in batch]
            )

        logger.info(f"Indexed {len(chunks)} chunks to ChromaDB")
        return len(chunks)

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        chunk_types: list[ChunkType] | None = None,
        file_pattern: str | None = None
    ) -> list[SearchResult]:
        """Search for relevant code using semantic similarity.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            chunk_types: Filter by chunk types (function, class, etc.)
            file_pattern: Filter by file path pattern.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        query_embedding = self.embedder.embed_text(query)

        # Build filter
        where_filter = None
        if chunk_types or file_pattern:
            conditions = []
            if chunk_types:
                conditions.append({
                    "chunk_type": {"$in": [t.value for t in chunk_types]}
                })
            if file_pattern:
                conditions.append({
                    "file_path": {"$contains": file_pattern}
                })
            where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        results = self.code_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                chunk = CodeChunk(
                    id=results["ids"][0][i],
                    file_path=metadata["file_path"],
                    chunk_type=ChunkType(metadata["chunk_type"]),
                    content=doc,
                    embedding=[],  # Don't return embedding for search results
                    metadata=metadata,
                    name=metadata.get("name"),
                    signature=metadata.get("signature"),
                    docstring=metadata.get("docstring")
                )

                # Convert distance to similarity (1 - distance for cosine)
                score = 1 - results["distances"][0][i]

                search_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    highlights=self._extract_highlights(doc, query)
                ))

        return search_results

    def find_similar(
        self,
        code_snippet: str,
        n_results: int = 5,
        min_similarity: float = 0.7
    ) -> list[SearchResult]:
        """Find code similar to a given snippet.

        Args:
            code_snippet: Code to find similar examples for.
            n_results: Maximum number of results.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of SearchResult objects with similar code.
        """
        results = self.semantic_search(code_snippet, n_results=n_results * 2)

        # Filter by similarity threshold
        return [r for r in results if r.score >= min_similarity][:n_results]

    def get_module_context(self, module_path: str) -> dict[str, Any]:
        """Get contextual information about a module.

        Args:
            module_path: Path or pattern to match module files.

        Returns:
            Dictionary with module context (functions, classes, etc.)
        """
        results = self.code_collection.get(
            where={"file_path": {"$contains": module_path}},
            include=["documents", "metadatas"]
        )

        context: dict[str, Any] = {
            "module": module_path,
            "functions": [],
            "classes": [],
            "methods": [],
            "description": ""
        }

        if results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"]):
                chunk_type = metadata.get("chunk_type", "")
                if chunk_type == "function":
                    context["functions"].append({
                        "name": metadata.get("name", ""),
                        "signature": metadata.get("signature", ""),
                        "docstring": metadata.get("docstring", "")
                    })
                elif chunk_type == "class":
                    context["classes"].append({
                        "name": metadata.get("name", ""),
                        "signature": metadata.get("signature", ""),
                        "docstring": metadata.get("docstring", "")
                    })
                elif chunk_type == "method":
                    context["methods"].append({
                        "name": metadata.get("name", ""),
                        "signature": metadata.get("signature", "")
                    })
                elif chunk_type == "module":
                    context["description"] = results["documents"][i][:500] if results["documents"] else ""

        return context

    def remember(self, key: str, value: str, category: str = "knowledge") -> None:
        """Store a piece of knowledge for future reference.

        Args:
            key: Unique identifier for this knowledge.
            value: The knowledge content to store.
            category: Category for organizing knowledge.
        """
        embedding = self.embedder.embed_text(value)
        doc_id = f"{category}_{key}"

        # Check if exists and update, or add new
        try:
            self.docs_collection.delete(ids=[doc_id])
        except Exception:
            pass

        self.docs_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"key": key, "category": category}],
            documents=[value]
        )

        logger.info(f"Stored knowledge: {key} in category {category}")

    def recall(
        self,
        query: str,
        category: str | None = None,
        n_results: int = 5
    ) -> list[tuple[str, str, float]]:
        """Recall relevant knowledge.

        Args:
            query: Query to search for relevant knowledge.
            category: Optional category filter.
            n_results: Maximum number of results.

        Returns:
            List of (key, value, score) tuples.
        """
        query_embedding = self.embedder.embed_text(query)

        where_filter = {"category": category} if category else None

        results = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        recalls = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                key = results["metadatas"][0][i].get("key", "")
                score = 1 - results["distances"][0][i]
                recalls.append((key, doc, score))

        return recalls

    def _extract_highlights(self, document: str, query: str, context_chars: int = 100) -> list[str]:
        """Extract relevant snippets from a document for display.

        Args:
            document: The full document text.
            query: The search query.
            context_chars: Characters of context around matches.

        Returns:
            List of highlighted snippets.
        """
        highlights = []
        query_terms = query.lower().split()
        doc_lower = document.lower()

        for term in query_terms:
            if len(term) < 3:
                continue
            idx = doc_lower.find(term)
            if idx >= 0:
                start = max(0, idx - context_chars)
                end = min(len(document), idx + len(term) + context_chars)
                snippet = document[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(document):
                    snippet = snippet + "..."
                highlights.append(snippet)

        return highlights[:3]  # Limit to 3 highlights
