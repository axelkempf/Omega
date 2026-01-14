"""Code indexing using AST parsing.

This module provides AST-based code analysis and chunking for Python files,
extracting semantic units like functions, classes, and modules for indexing.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Any

from .embedder import CodeEmbedder
from .models import ChunkType, CodeChunk

logger = logging.getLogger(__name__)


class CodeIndexer:
    """Indexes Python code into semantic chunks.

    Parses Python source files using AST to extract meaningful code units
    (functions, classes, modules) and generates embeddings for semantic search.

    Args:
        embedder: CodeEmbedder instance for generating embeddings.
    """

    def __init__(self, embedder: CodeEmbedder):
        self.embedder = embedder

    def index_directory(
        self,
        path: Path,
        exclude_patterns: list[str] | None = None
    ) -> list[CodeChunk]:
        """Index all Python files in a directory.

        Args:
            path: Root directory to index.
            exclude_patterns: Patterns to exclude (default: __pycache__, .git)

        Returns:
            List of CodeChunk objects representing the indexed code.
        """
        if exclude_patterns is None:
            exclude_patterns = ["__pycache__", ".git", ".venv", "node_modules"]

        chunks = []

        for py_file in path.rglob("*.py"):
            # Skip excluded patterns
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            try:
                file_chunks = self.index_file(py_file)
                chunks.extend(file_chunks)
                logger.debug(f"Indexed {len(file_chunks)} chunks from {py_file}")
            except SyntaxError as e:
                logger.warning(f"Syntax error in {py_file}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error indexing {py_file}: {e}")
                continue

        logger.info(f"Indexed {len(chunks)} total chunks from {path}")
        return chunks

    def index_file(self, file_path: Path) -> list[CodeChunk]:
        """Index a single Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            List of CodeChunk objects from this file.
        """
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)

        chunks = []

        # Module-level docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.append(self._create_chunk(
                file_path=str(file_path),
                chunk_type=ChunkType.MODULE,
                content=module_docstring,
                name=file_path.stem
            ))

        # Functions and classes at module level
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                chunk = self._extract_function(node, str(file_path), content)
                chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, str(file_path), content)
                chunks.append(chunk)
                # Also index methods within the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        method_chunk = self._extract_method(
                            item, node.name, str(file_path), content
                        )
                        chunks.append(method_chunk)

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        source: str
    ) -> CodeChunk:
        """Extract a function as a chunk."""
        code = ast.get_source_segment(source, node) or ""

        # Build signature
        args = []
        for a in node.args.args:
            arg_str = a.arg
            if a.annotation:
                arg_str += f": {ast.unparse(a.annotation)}"
            args.append(arg_str)

        signature = f"{node.name}({', '.join(args)})"

        # Add return type if present
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        # Add async prefix if applicable
        if isinstance(node, ast.AsyncFunctionDef):
            signature = f"async {signature}"

        # Extract callees (functions this function calls)
        callees = self._extract_callees(node)

        return self._create_chunk(
            file_path=file_path,
            chunk_type=ChunkType.FUNCTION,
            content=code,
            name=node.name,
            signature=signature,
            docstring=ast.get_docstring(node),
            metadata={"callees": callees, "lineno": node.lineno}
        )

    def _extract_method(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_name: str,
        file_path: str,
        source: str
    ) -> CodeChunk:
        """Extract a method as a chunk."""
        code = ast.get_source_segment(source, node) or ""

        # Build signature
        args = []
        for a in node.args.args:
            arg_str = a.arg
            if a.annotation:
                arg_str += f": {ast.unparse(a.annotation)}"
            args.append(arg_str)

        signature = f"{class_name}.{node.name}({', '.join(args)})"

        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        if isinstance(node, ast.AsyncFunctionDef):
            signature = f"async {signature}"

        return self._create_chunk(
            file_path=file_path,
            chunk_type=ChunkType.METHOD,
            content=code,
            name=f"{class_name}.{node.name}",
            signature=signature,
            docstring=ast.get_docstring(node),
            metadata={"class": class_name, "lineno": node.lineno}
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
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        ]

        # Get base classes
        bases = [ast.unparse(b) for b in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        return self._create_chunk(
            file_path=file_path,
            chunk_type=ChunkType.CLASS,
            content=code,
            name=node.name,
            signature=signature,
            docstring=ast.get_docstring(node),
            metadata={"methods": methods, "bases": bases, "lineno": node.lineno}
        )

    def _extract_callees(self, node: ast.AST) -> list[str]:
        """Extract names of functions called within a node."""
        callees = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    callees.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    callees.append(child.func.attr)
        return list(set(callees))

    def _create_chunk(
        self,
        file_path: str,
        chunk_type: ChunkType,
        content: str,
        name: str | None = None,
        signature: str | None = None,
        docstring: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> CodeChunk:
        """Create a CodeChunk with embedding.

        Args:
            file_path: Path to the source file.
            chunk_type: Type of the chunk.
            content: Raw source code or text.
            name: Name of the function/class/module.
            signature: Function/method signature.
            docstring: Documentation string.
            metadata: Additional metadata.

        Returns:
            CodeChunk with generated embedding.
        """
        # Generate ID from content hash
        chunk_id = hashlib.md5(
            f"{file_path}:{chunk_type.value}:{name or content[:50]}".encode()
        ).hexdigest()[:12]

        # Generate embedding from combined text
        embed_text = "\n".join(filter(None, [
            name or "",
            signature or "",
            docstring or "",
            content[:2000]  # Limit content length for embedding
        ]))
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
