# 05 - RAG Layer

> Retrieval-Augmented Generation für intelligente Codebase-Suche

**Status:** 🔴 Offen
**Priorität:** Niedrig
**Komplexität:** Hoch
**Geschätzter Aufwand:** 2-3 Tage

---

## Objective

Implementiere einen **RAG-Layer** (Retrieval-Augmented Generation) der:
- Codebase semantisch durchsuchbar macht
- Relevanten Kontext für Agents findet
- Projekt-Wissen über Sessions hinweg speichert
- Ähnlichen Code und Patterns erkennt

---

## Current State

### Problem

Aktuell sind Codebase-Suchen **keyword-basiert**:

```
[Agent] ──► "Suche Portfolio-Handling"
              │
              ▼
         grep "portfolio"
              │
              ▼
         [Viele irrelevante Treffer]
```

### Limitierungen

1. **Keine semantische Suche** - "Portfolio-Handling" findet nicht "PositionManager"
2. **Kein Projekt-Gedächtnis** - Architektur-Entscheidungen gehen verloren
3. **Keine Code-Ähnlichkeit** - Duplizierter Code wird nicht erkannt
4. **Kein Kontext** - Zusammenhänge zwischen Modulen unklar

---

## Target State

### RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Layer                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                    Query Interface                          ││
│  │  semantic_search("Portfolio-Handling") → [relevant_files]  ││
│  │  find_similar("def calculate_lot()") → [similar_funcs]     ││
│  │  get_context("strategies module") → [architecture_info]    ││
│  └─────────────────────────┬──────────────────────────────────┘│
│                            │                                    │
│  ┌─────────────────────────▼──────────────────────────────────┐│
│  │                  Vector Database                            ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ ││
│  │  │ Code Chunks  │  │  Docstrings  │  │  ADRs/Docs       │ ││
│  │  │ (embeddings) │  │ (embeddings) │  │  (embeddings)    │ ││
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ ││
│  └─────────────────────────┬──────────────────────────────────┘│
│                            │                                    │
│  ┌─────────────────────────▼──────────────────────────────────┐│
│  │                  Embedding Pipeline                         ││
│  │  1. Code Parser → AST Extraction                           ││
│  │  2. Chunker → Semantic Chunks                              ││
│  │  3. Embedder → Vector Representations                      ││
│  │  4. Indexer → Vector DB Storage                            ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Datenmodell

```python
@dataclass
class CodeChunk:
    """A semantic chunk of code for embedding."""

    id: str                      # Unique identifier
    file_path: str               # Source file
    chunk_type: ChunkType        # function, class, module, docstring
    content: str                 # Raw code/text
    embedding: list[float]       # Vector representation
    metadata: dict[str, Any]     # Additional context

    # Extracted information
    name: str | None             # Function/class name
    signature: str | None        # Function signature
    docstring: str | None        # Associated docstring
    dependencies: list[str]      # Imported modules
    callers: list[str]           # Functions that call this
    callees: list[str]           # Functions this calls


class ChunkType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    ADR = "adr"
    DOCUMENTATION = "documentation"
```

---

## Implementation Plan

### Schritt 1: Vector Database Setup

Wähle eine leichtgewichtige Vector DB:

| Option | Pros | Cons | Empfehlung |
|--------|------|------|------------|
| **ChromaDB** | Einfach, lokal, Python-native | Weniger Features | ✅ Empfohlen |
| Qdrant | Performant, Feature-reich | Mehr Setup | Für Scale |
| Weaviate | GraphQL API, Schema | Overkill | Nein |
| FAISS | Schnell, von Meta | Low-level | Nur Embedding |

#### Setup ChromaDB

```python
# src/agent_memory/__init__.py
"""RAG layer for intelligent codebase search."""

from .rag import RAGLayer, CodeChunk, ChunkType
from .embedder import CodeEmbedder
from .indexer import CodeIndexer

__all__ = ["RAGLayer", "CodeChunk", "ChunkType", "CodeEmbedder", "CodeIndexer"]
```

```python
# src/agent_memory/rag.py
"""Main RAG interface."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embedder import CodeEmbedder
from .indexer import CodeIndexer


@dataclass
class SearchResult:
    """Result from a semantic search."""

    chunk: CodeChunk
    score: float
    highlights: list[str]


class RAGLayer:
    """Retrieval-Augmented Generation for codebase search."""

    def __init__(
        self,
        persist_dir: Path = Path("var/agent_memory"),
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_dir),
            anonymized_telemetry=False
        ))

        # Collections
        self.code_collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"description": "Code chunks from the codebase"}
        )
        self.docs_collection = self.client.get_or_create_collection(
            name="documentation",
            metadata={"description": "Documentation and ADRs"}
        )

        # Components
        self.embedder = CodeEmbedder(model_name=embedding_model)
        self.indexer = CodeIndexer(self.embedder)

    def index_codebase(self, src_path: Path = Path("src")) -> int:
        """Index the entire codebase."""

        chunks = self.indexer.index_directory(src_path)

        # Add to ChromaDB
        for chunk in chunks:
            self.code_collection.add(
                ids=[chunk.id],
                embeddings=[chunk.embedding],
                metadatas=[{
                    "file_path": chunk.file_path,
                    "chunk_type": chunk.chunk_type.value,
                    "name": chunk.name or "",
                    "signature": chunk.signature or ""
                }],
                documents=[chunk.content]
            )

        return len(chunks)

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        chunk_types: list[ChunkType] | None = None
    ) -> list[SearchResult]:
        """Search for relevant code using semantic similarity."""

        query_embedding = self.embedder.embed_text(query)

        # Build filter
        where_filter = None
        if chunk_types:
            where_filter = {
                "chunk_type": {"$in": [t.value for t in chunk_types]}
            }

        results = self.code_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i, doc in enumerate(results["documents"][0]):
            chunk = CodeChunk(
                id=results["ids"][0][i],
                file_path=results["metadatas"][0][i]["file_path"],
                chunk_type=ChunkType(results["metadatas"][0][i]["chunk_type"]),
                content=doc,
                embedding=[],
                metadata=results["metadatas"][0][i],
                name=results["metadatas"][0][i].get("name"),
                signature=results["metadatas"][0][i].get("signature"),
                docstring=None,
                dependencies=[],
                callers=[],
                callees=[]
            )

            search_results.append(SearchResult(
                chunk=chunk,
                score=1 - results["distances"][0][i],  # Convert distance to similarity
                highlights=[]
            ))

        return search_results

    def find_similar(
        self,
        code_snippet: str,
        n_results: int = 5,
        min_similarity: float = 0.7
    ) -> list[SearchResult]:
        """Find code similar to a given snippet."""

        results = self.semantic_search(code_snippet, n_results=n_results * 2)

        # Filter by similarity threshold
        return [r for r in results if r.score >= min_similarity][:n_results]

    def get_module_context(self, module_path: str) -> dict[str, Any]:
        """Get contextual information about a module."""

        # Search for all chunks in this module
        results = self.code_collection.get(
            where={"file_path": {"$contains": module_path}},
            include=["documents", "metadatas"]
        )

        context = {
            "module": module_path,
            "functions": [],
            "classes": [],
            "dependencies": set(),
            "description": ""
        }

        for i, metadata in enumerate(results["metadatas"]):
            if metadata["chunk_type"] == "function":
                context["functions"].append({
                    "name": metadata["name"],
                    "signature": metadata["signature"]
                })
            elif metadata["chunk_type"] == "class":
                context["classes"].append({
                    "name": metadata["name"]
                })
            elif metadata["chunk_type"] == "module":
                context["description"] = results["documents"][i][:500]

        return context

    def remember(self, key: str, value: str, category: str = "knowledge") -> None:
        """Store a piece of knowledge for future reference."""

        embedding = self.embedder.embed_text(value)

        self.docs_collection.add(
            ids=[f"{category}_{key}"],
            embeddings=[embedding],
            metadatas=[{"key": key, "category": category}],
            documents=[value]
        )

    def recall(self, query: str, category: str | None = None) -> list[str]:
        """Recall relevant knowledge."""

        query_embedding = self.embedder.embed_text(query)

        where_filter = {"category": category} if category else None

        results = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where=where_filter,
            include=["documents"]
        )

        return results["documents"][0] if results["documents"] else []
```

### Schritt 2: Code Embedder

```python
# src/agent_memory/embedder.py
"""Code embedding using sentence transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class CodeEmbedder:
    """Generates embeddings for code and documentation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text."""

        return self.model.encode(text).tolist()

    def embed_code(self, code: str, context: str = "") -> list[float]:
        """Generate embedding for code with optional context."""

        # Combine code with context for richer embedding
        combined = f"{context}\n\n{code}" if context else code
        return self.embed_text(combined)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        return self.model.encode(texts).tolist()
```

### Schritt 3: Code Indexer

```python
# src/agent_memory/indexer.py
"""Code indexing using AST parsing."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embedder import CodeEmbedder


class ChunkType:
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


@dataclass
class CodeChunk:
    """A semantic chunk of code."""

    id: str
    file_path: str
    chunk_type: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    dependencies: list[str] = field(default_factory=list)
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)


class CodeIndexer:
    """Indexes Python code into semantic chunks."""

    def __init__(self, embedder: CodeEmbedder):
        self.embedder = embedder

    def index_directory(self, path: Path) -> list[CodeChunk]:
        """Index all Python files in a directory."""

        chunks = []

        for py_file in path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                file_chunks = self.index_file(py_file)
                chunks.extend(file_chunks)
            except SyntaxError:
                continue

        return chunks

    def index_file(self, file_path: Path) -> list[CodeChunk]:
        """Index a single Python file."""

        content = file_path.read_text()
        tree = ast.parse(content)

        chunks = []

        # Module-level docstring
        if (ast.get_docstring(tree)):
            chunks.append(self._create_chunk(
                file_path=str(file_path),
                chunk_type=ChunkType.MODULE,
                content=ast.get_docstring(tree),
                name=file_path.stem
            ))

        # Functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = self._extract_function(node, str(file_path), content)
                chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, str(file_path), content)
                chunks.append(chunk)

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: str,
        source: str
    ) -> CodeChunk:
        """Extract a function as a chunk."""

        # Get source code
        code = ast.get_source_segment(source, node) or ""

        # Build signature
        args = [a.arg for a in node.args.args]
        signature = f"{node.name}({', '.join(args)})"

        # Get return type if present
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        return self._create_chunk(
            file_path=file_path,
            chunk_type=ChunkType.FUNCTION,
            content=code,
            name=node.name,
            signature=signature,
            docstring=ast.get_docstring(node)
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        source: str
    ) -> CodeChunk:
        """Extract a class as a chunk."""

        code = ast.get_source_segment(source, node) or ""

        # Get method names
        methods = [
            n.name for n in node.body
            if isinstance(n, ast.FunctionDef)
        ]

        return self._create_chunk(
            file_path=file_path,
            chunk_type=ChunkType.CLASS,
            content=code,
            name=node.name,
            signature=f"class {node.name}",
            docstring=ast.get_docstring(node),
            metadata={"methods": methods}
        )

    def _create_chunk(
        self,
        file_path: str,
        chunk_type: str,
        content: str,
        name: str | None = None,
        signature: str | None = None,
        docstring: str | None = None,
        metadata: dict | None = None
    ) -> CodeChunk:
        """Create a CodeChunk with embedding."""

        # Generate ID from content hash
        chunk_id = hashlib.md5(
            f"{file_path}:{chunk_type}:{name}".encode()
        ).hexdigest()[:12]

        # Generate embedding
        embed_text = f"{name or ''}\n{signature or ''}\n{docstring or ''}\n{content}"
        embedding = self.embedder.embed_text(embed_text)

        return CodeChunk(
            id=chunk_id,
            file_path=file_path,
            chunk_type=chunk_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            name=name,
            signature=signature,
            docstring=docstring
        )
```

### Schritt 4: CLI und Integration

```python
# src/agent_memory/cli.py
"""CLI for RAG layer."""

import argparse
from pathlib import Path

from .rag import RAGLayer


def main():
    parser = argparse.ArgumentParser(description="Agent Memory RAG")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index the codebase")
    index_parser.add_argument("--path", default="src", help="Path to index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", type=int, default=5, help="Number of results")

    # Remember command
    remember_parser = subparsers.add_parser("remember", help="Store knowledge")
    remember_parser.add_argument("key", help="Knowledge key")
    remember_parser.add_argument("value", help="Knowledge value")

    args = parser.parse_args()

    rag = RAGLayer()

    if args.command == "index":
        count = rag.index_codebase(Path(args.path))
        print(f"Indexed {count} chunks")

    elif args.command == "search":
        results = rag.semantic_search(args.query, n_results=args.n)
        for r in results:
            print(f"\n[{r.score:.2f}] {r.chunk.file_path}")
            print(f"  {r.chunk.name}: {r.chunk.signature}")

    elif args.command == "remember":
        rag.remember(args.key, args.value)
        print(f"Remembered: {args.key}")


if __name__ == "__main__":
    main()
```

---

## Acceptance Criteria

- [ ] ChromaDB persistiert in `var/agent_memory/`
- [ ] Codebase kann indexiert werden (`index` Command)
- [ ] Semantische Suche funktioniert (`search` Command)
- [ ] Ähnlicher Code wird gefunden (`find_similar`)
- [ ] Wissen kann gespeichert/abgerufen werden
- [ ] Integration mit Orchestrator (optional)

---

## Dependencies

Neue Dependencies für `pyproject.toml`:

```toml
[project.optional-dependencies]
agent = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
]
```

---

## Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Langsames Indexing | Mittel | Niedrig | Inkrementelles Update |
| Große Speichernutzung | Mittel | Mittel | Chunk-Size Limits |
| Schlechte Embeddings | Niedrig | Hoch | Code-spezifisches Model |

---

## Future Enhancements

1. **Code-spezifisches Embedding Model** (z.B. CodeBERT)
2. **Graph-basierte Suche** (Call-Graphs, Import-Graphs)
3. **Automatic Re-indexing** bei Dateiänderungen
4. **Cross-Repository Search** für Multi-Repo Setups
