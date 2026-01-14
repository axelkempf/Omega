"""Tests for CodeIndexer AST parsing."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.agent_memory.indexer import CodeIndexer
from src.agent_memory.models import ChunkType


class TestCodeIndexer:
    """Tests for CodeIndexer."""

    @pytest.fixture
    def indexer(self) -> CodeIndexer:
        """Create a CodeIndexer instance."""
        return CodeIndexer()

    @pytest.fixture
    def sample_python_file(self, tmp_path: Path) -> Path:
        """Create a sample Python file for testing."""
        code = '''"""Module docstring."""

def simple_function():
    """A simple function."""
    pass


def function_with_call():
    """Calls another function."""
    simple_function()
    return 42


class MyClass:
    """A sample class."""

    def __init__(self, value: int):
        """Initialize with value."""
        self.value = value

    def method(self) -> int:
        """Return the value."""
        return self.value
'''
        file_path = tmp_path / "sample.py"
        file_path.write_text(code)
        return file_path

    def test_index_file_extracts_functions(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that index_file extracts functions."""
        chunks = indexer.index_file(sample_python_file)

        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        function_names = [c.name for c in function_chunks]

        assert "simple_function" in function_names
        assert "function_with_call" in function_names

    def test_index_file_extracts_classes(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that index_file extracts classes."""
        chunks = indexer.index_file(sample_python_file)

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "MyClass"

    def test_index_file_extracts_methods(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that index_file extracts methods."""
        chunks = indexer.index_file(sample_python_file)

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        method_names = [c.name for c in method_chunks]

        assert "__init__" in method_names
        assert "method" in method_names

    def test_index_file_captures_docstrings(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that docstrings are captured."""
        chunks = indexer.index_file(sample_python_file)

        func_chunk = next(
            (c for c in chunks if c.name == "simple_function"), None
        )
        assert func_chunk is not None
        assert func_chunk.docstring == "A simple function."

    def test_index_file_captures_callees(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that function calls are captured."""
        chunks = indexer.index_file(sample_python_file)

        func_chunk = next(
            (c for c in chunks if c.name == "function_with_call"), None
        )
        assert func_chunk is not None
        assert "simple_function" in func_chunk.callees

    def test_index_file_captures_line_numbers(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that line numbers are captured."""
        chunks = indexer.index_file(sample_python_file)

        func_chunk = next(
            (c for c in chunks if c.name == "simple_function"), None
        )
        assert func_chunk is not None
        assert func_chunk.start_line is not None
        assert func_chunk.end_line is not None
        assert func_chunk.start_line < func_chunk.end_line

    def test_index_file_includes_module_chunk(
        self, indexer: CodeIndexer, sample_python_file: Path
    ) -> None:
        """Test that module-level chunk is included."""
        chunks = indexer.index_file(sample_python_file)

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE]
        assert len(module_chunks) == 1
        assert "Module docstring" in module_chunks[0].content

    def test_index_nonexistent_file(self, indexer: CodeIndexer) -> None:
        """Test that nonexistent file returns empty list."""
        chunks = indexer.index_file(Path("/nonexistent/file.py"))
        assert chunks == []

    def test_index_invalid_python(
        self, indexer: CodeIndexer, tmp_path: Path
    ) -> None:
        """Test that invalid Python syntax returns empty list."""
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("def broken(:\n    pass")

        chunks = indexer.index_file(invalid_file)
        assert chunks == []

    def test_index_directory(
        self, indexer: CodeIndexer, tmp_path: Path
    ) -> None:
        """Test indexing a directory with multiple files."""
        # Create subdirectory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = subdir / "file2.py"
        file2.write_text("def func2(): pass")

        # Non-Python file should be ignored
        other_file = tmp_path / "readme.txt"
        other_file.write_text("Not Python")

        chunks = indexer.index_directory(tmp_path)

        # Should find functions from both Python files
        func_names = [c.name for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert "func1" in func_names
        assert "func2" in func_names

    def test_index_directory_with_exclusions(
        self, indexer: CodeIndexer, tmp_path: Path
    ) -> None:
        """Test that __pycache__ and hidden dirs are excluded."""
        # Create __pycache__ directory
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        cached_file = pycache / "cached.py"
        cached_file.write_text("def cached(): pass")

        # Create hidden directory
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        hidden_file = hidden / "secret.py"
        hidden_file.write_text("def secret(): pass")

        # Create normal file
        normal = tmp_path / "normal.py"
        normal.write_text("def normal(): pass")

        chunks = indexer.index_directory(tmp_path)
        func_names = [c.name for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        assert "normal" in func_names
        assert "cached" not in func_names
        assert "secret" not in func_names
