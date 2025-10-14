"""Service layer for search operations.

Provides a simple interface for search, retrieval, and dependency operations.
"""

from typing import List, Optional

from lean_explore.search.engine import SearchEngine
from lean_explore.search.types import SearchResult


class Service:
    """Service wrapper for search operations.

    Provides a clean interface for searching, retrieving declarations,
    and fetching dependencies.
    """

    def __init__(self, engine: Optional[SearchEngine] = None):
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
    ) -> List[SearchResult]:
        """Search for Lean declarations.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            semantic_weight: Weight for semantic similarity (0-1).
            pagerank_weight: Weight for PageRank score (0-1).
            lexical_weight: Weight for lexical matching (0-1).

        Returns:
            List of SearchResult objects, ranked by combined score.
        """
        return self.engine.search(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            pagerank_weight=pagerank_weight,
            lexical_weight=lexical_weight,
        )

    def get_by_id(self, declaration_id: int) -> Optional[SearchResult]:
        """Retrieve a declaration by ID.

        Args:
            declaration_id: The declaration ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        return self.engine.get_by_id(declaration_id)

    def get_dependencies(self, declaration_id: int) -> List[SearchResult]:
        """Get dependencies for a declaration.

        Args:
            declaration_id: The declaration ID.

        Returns:
            List of SearchResult objects for dependencies.
        """
        return self.engine.get_dependencies(declaration_id)
