"""Search package for lean explore.

This package contains the core search engine functionality including semantic search,
BM25 lexical matching, PageRank scoring, and the service layer for search operations.
"""

from lean_explore.search.engine import SearchEngine
from lean_explore.search.service import Service
from lean_explore.search.types import SearchResponse, SearchResult

__all__ = ["SearchEngine", "Service", "SearchResponse", "SearchResult"]
