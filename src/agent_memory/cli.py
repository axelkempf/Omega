"""CLI for RAG layer operations.

This module provides command-line tools for indexing the codebase,
performing semantic searches, and managing stored knowledge.

Usage:
    python -m src.agent_memory.cli index [--path PATH] [--clear]
    python -m src.agent_memory.cli search QUERY [-n N] [--type TYPE]
    python -m src.agent_memory.cli similar CODE [-n N]
    python -m src.agent_memory.cli remember KEY VALUE [--category CATEGORY]
    python -m src.agent_memory.cli recall QUERY [--category CATEGORY]
    python -m src.agent_memory.cli context MODULE_PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import ChunkType
from .rag import RAGLayer


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Omega Agent Memory RAG - Semantic codebase search",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("var/agent_memory"),
        help="Directory for ChromaDB persistence"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Index the codebase for semantic search"
    )
    index_parser.add_argument(
        "--path",
        type=Path,
        default=Path("src"),
        help="Root directory to index (default: src)"
    )
    index_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before re-indexing"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Semantic search across the codebase"
    )
    search_parser.add_argument("query", help="Natural language search query")
    search_parser.add_argument(
        "-n", "--results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    search_parser.add_argument(
        "--type",
        choices=["function", "class", "method", "module"],
        help="Filter by chunk type"
    )
    search_parser.add_argument(
        "--file",
        help="Filter by file path pattern"
    )

    # Similar command
    similar_parser = subparsers.add_parser(
        "similar",
        help="Find code similar to a given snippet"
    )
    similar_parser.add_argument("code", help="Code snippet to find similar code for")
    similar_parser.add_argument(
        "-n", "--results",
        type=int,
        default=5,
        help="Number of results"
    )
    similar_parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum similarity score (0-1)"
    )

    # Remember command
    remember_parser = subparsers.add_parser(
        "remember",
        help="Store knowledge for future reference"
    )
    remember_parser.add_argument("key", help="Unique key for this knowledge")
    remember_parser.add_argument("value", help="Knowledge content to store")
    remember_parser.add_argument(
        "--category",
        default="knowledge",
        help="Category for organizing (default: knowledge)"
    )

    # Recall command
    recall_parser = subparsers.add_parser(
        "recall",
        help="Recall stored knowledge"
    )
    recall_parser.add_argument("query", help="Query to search for")
    recall_parser.add_argument(
        "--category",
        help="Filter by category"
    )
    recall_parser.add_argument(
        "-n", "--results",
        type=int,
        default=5,
        help="Number of results"
    )

    # Context command
    context_parser = subparsers.add_parser(
        "context",
        help="Get context about a module"
    )
    context_parser.add_argument("module", help="Module path or pattern")

    # Stats command
    subparsers.add_parser(
        "stats",
        help="Show index statistics"
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Initialize RAG layer
    rag = RAGLayer(persist_dir=args.persist_dir)

    if args.command == "index":
        return cmd_index(rag, args)
    elif args.command == "search":
        return cmd_search(rag, args)
    elif args.command == "similar":
        return cmd_similar(rag, args)
    elif args.command == "remember":
        return cmd_remember(rag, args)
    elif args.command == "recall":
        return cmd_recall(rag, args)
    elif args.command == "context":
        return cmd_context(rag, args)
    elif args.command == "stats":
        return cmd_stats(rag, args)

    return 0


def cmd_index(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle index command."""
    print(f"Indexing codebase at: {args.path}")
    if args.clear:
        print("Clearing existing index...")

    count = rag.index_codebase(args.path, clear_existing=args.clear)
    print(f"✅ Indexed {count} code chunks")
    return 0


def cmd_search(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle search command."""
    chunk_types = None
    if args.type:
        chunk_types = [ChunkType(args.type)]

    results = rag.semantic_search(
        args.query,
        n_results=args.results,
        chunk_types=chunk_types,
        file_pattern=args.file
    )

    if not results:
        print("No results found.")
        return 0

    print(f"\n🔍 Search results for: '{args.query}'\n")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.score:.2f}] {r.chunk.file_path}")
        print(f"   Type: {r.chunk.chunk_type.value}")
        if r.chunk.name:
            print(f"   Name: {r.chunk.name}")
        if r.chunk.signature:
            print(f"   Signature: {r.chunk.signature}")
        if r.chunk.docstring:
            docstring_preview = r.chunk.docstring[:100].replace("\n", " ")
            print(f"   Doc: {docstring_preview}...")
        if r.highlights:
            print(f"   Highlight: {r.highlights[0]}")

    return 0


def cmd_similar(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle similar command."""
    results = rag.find_similar(
        args.code,
        n_results=args.results,
        min_similarity=args.min_score
    )

    if not results:
        print("No similar code found.")
        return 0

    print(f"\n🔍 Similar code found:\n")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.score:.2f}] {r.chunk.file_path}")
        if r.chunk.signature:
            print(f"   {r.chunk.signature}")
        # Show first few lines of content
        content_preview = r.chunk.content.split("\n")[:5]
        for line in content_preview:
            print(f"   {line}")
        if len(r.chunk.content.split("\n")) > 5:
            print("   ...")

    return 0


def cmd_remember(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle remember command."""
    rag.remember(args.key, args.value, category=args.category)
    print(f"✅ Remembered: {args.key} (category: {args.category})")
    return 0


def cmd_recall(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle recall command."""
    results = rag.recall(
        args.query,
        category=args.category,
        n_results=args.results
    )

    if not results:
        print("No matching knowledge found.")
        return 0

    print(f"\n🧠 Recalled knowledge for: '{args.query}'\n")
    print("=" * 60)

    for key, value, score in results:
        print(f"\n[{score:.2f}] {key}")
        print(f"   {value[:200]}...")

    return 0


def cmd_context(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle context command."""
    context = rag.get_module_context(args.module)

    print(f"\n📦 Module Context: {args.module}\n")
    print("=" * 60)

    if context["description"]:
        print(f"\nDescription:\n  {context['description'][:300]}...")

    if context["classes"]:
        print(f"\nClasses ({len(context['classes'])}):")
        for cls in context["classes"]:
            print(f"  - {cls['signature']}")
            if cls.get("docstring"):
                print(f"    {cls['docstring'][:80]}...")

    if context["functions"]:
        print(f"\nFunctions ({len(context['functions'])}):")
        for func in context["functions"]:
            print(f"  - {func['signature']}")

    if context["methods"]:
        print(f"\nMethods ({len(context['methods'])}):")
        for method in context["methods"][:10]:  # Limit display
            print(f"  - {method['signature']}")
        if len(context["methods"]) > 10:
            print(f"  ... and {len(context['methods']) - 10} more")

    return 0


def cmd_stats(rag: RAGLayer, args: argparse.Namespace) -> int:
    """Handle stats command."""
    print("\n📊 RAG Index Statistics\n")
    print("=" * 60)

    code_count = rag.code_collection.count()
    docs_count = rag.docs_collection.count()

    print(f"Code chunks indexed: {code_count}")
    print(f"Documentation entries: {docs_count}")
    print(f"Persist directory: {rag.persist_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
