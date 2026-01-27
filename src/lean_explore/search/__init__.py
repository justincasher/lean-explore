"""Search package for Lean Explore.

This package provides hybrid search for Lean declarations using BM25 lexical
matching and FAISS semantic search, combined via Reciprocal Rank Fusion.

Modules:
    engine: Core SearchEngine class with hybrid retrieval and cross-encoder reranking.
    scoring: Score normalization and fusion algorithms (RRF, weighted fusion).
    service: Service layer wrapper for search operations.
    tokenization: Text tokenization utilities for Lean declaration names.

Note: SearchEngine and Service are lazily imported to avoid loading FAISS at module
import time, which helps prevent OpenMP library conflicts with torch on macOS.
"""

from lean_explore.models import SearchResponse, SearchResult


def __getattr__(name: str):
    """Lazy import SearchEngine and Service to avoid FAISS loading at import time."""
    if name == "SearchEngine":
        from lean_explore.search.engine import SearchEngine

        return SearchEngine
    if name == "Service":
        from lean_explore.search.service import Service

        return Service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SearchEngine", "Service", "SearchResponse", "SearchResult"]
