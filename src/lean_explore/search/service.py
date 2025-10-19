"""Service layer for search operations."""

import time

from lean_explore.models import SearchResponse, SearchResult
from lean_explore.search.engine import SearchEngine


class Service:
    """Service wrapper for search operations.

    Provides a clean interface for searching and retrieving declarations.
    """

    def __init__(self, engine: SearchEngine | None = None):
        """Initialize the search service.

        Args:
            engine: SearchEngine instance. Defaults to new engine.
        """
        self.engine = engine or SearchEngine()

    def search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.4,
        pagerank_weight: float = 0.3,
        lexical_weight: float = 0.3,
    ) -> SearchResponse:
        """Search for Lean declarations.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            semantic_weight: Weight for semantic similarity (0-1).
            pagerank_weight: Weight for PageRank score (0-1).
            lexical_weight: Weight for lexical matching (0-1).

        Returns:
            SearchResponse containing results and metadata.
        """
        start_time = time.time()

        results = self.engine.search(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            pagerank_weight=pagerank_weight,
            lexical_weight=lexical_weight,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return SearchResponse(
            query=query,
            results=results,
            count=len(results),
            processing_time_ms=processing_time_ms,
        )

    def get_by_id(self, declaration_id: int) -> SearchResult | None:
        """Retrieve a declaration by ID.

        Args:
            declaration_id: The declaration ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        return self.engine.get_by_id(declaration_id)
