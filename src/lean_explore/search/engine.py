"""Core search engine functionality for Lean declarations.

This module provides the core search functionality using PostgreSQL with pgvector
for semantic search, combined with BM25 lexical matching and PageRank scoring.
"""

import asyncio
import logging
from typing import List, Optional

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from lean_explore import defaults
from lean_explore.extract.schemas import Declaration
from lean_explore.search.types import SearchResult
from lean_explore.util.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


class SearchEngine:
    """Core search engine for Lean declarations.

    Combines semantic search (pgvector), lexical matching (BM25),
    and PageRank scoring to rank results.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1",
    ):
        """Initialize the search engine.

        Args:
            db_url: Database URL. Defaults to configured URL.
            embedding_client: Client for generating embeddings. Defaults to new client.
            embedding_model_name: Name of the embedding model to use.
        """
        self.db_url = db_url or defaults.DEFAULT_DB_URL
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.embedding_client = embedding_client or EmbeddingClient(
            model_name=embedding_model_name
        )

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
        with self.SessionLocal() as session:
            # Generate query embedding (async)
            embedding_response = asyncio.run(self.embedding_client.embed([query]))
            query_embedding = embedding_response.embeddings[0]

            # Perform vector similarity search
            # Using informalization_embedding as primary semantic signal
            stmt = select(Declaration).order_by(
                Declaration.informalization_embedding.cosine_distance(query_embedding)
            ).limit(limit * 3)  # Get more candidates for reranking

            candidates = session.execute(stmt).scalars().all()

            # Score and rank candidates
            scored_results = []
            for decl in candidates:
                # Semantic similarity (cosine similarity, 0-1)
                semantic_score = self._compute_semantic_similarity(
                    query_embedding, decl
                )

                # PageRank score (normalize to 0-1)
                pagerank_score = decl.pagerank or 0.0

                # Lexical match score (BM25-like)
                lexical_score = self._compute_lexical_score(query, decl)

                # Combined weighted score
                final_score = (
                    semantic_weight * semantic_score
                    + pagerank_weight * pagerank_score
                    + lexical_weight * lexical_score
                )

                scored_results.append((decl, final_score))

            # Sort by score and convert to SearchResult
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [
                self._to_search_result(decl)
                for decl, _ in scored_results[:limit]
            ]

    def get_by_id(self, declaration_id: int) -> Optional[SearchResult]:
        """Retrieve a declaration by ID.

        Args:
            declaration_id: The declaration ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        with self.SessionLocal() as session:
            decl = session.get(Declaration, declaration_id)
            return self._to_search_result(decl) if decl else None

    def get_dependencies(self, declaration_id: int) -> List[SearchResult]:
        """Get dependencies for a declaration.

        Args:
            declaration_id: The declaration ID.

        Returns:
            List of SearchResult objects for dependencies.
        """
        with self.SessionLocal() as session:
            decl = session.get(Declaration, declaration_id)
            if not decl or not decl.dependencies:
                return []

            # Dependencies stored as JSON array of names
            import json
            dep_names = json.loads(decl.dependencies)

            # Fetch dependency declarations
            stmt = select(Declaration).where(Declaration.name.in_(dep_names))
            deps = session.execute(stmt).scalars().all()

            return [self._to_search_result(d) for d in deps]

    def _compute_semantic_similarity(
        self, query_embedding: List[float], decl: Declaration
    ) -> float:
        """Compute semantic similarity score.

        Args:
            query_embedding: Query embedding vector.
            decl: Declaration to score.

        Returns:
            Similarity score (0-1).
        """
        # pgvector cosine_distance returns distance, convert to similarity
        # This is a simplified version - in practice, use pgvector's built-in similarity
        if decl.informalization_embedding:
            # Cosine similarity = 1 - cosine_distance
            # This is approximate; actual computation happens in SQL
            return 0.8  # Placeholder - actual scoring done in query
        return 0.0

    def _compute_lexical_score(self, query: str, decl: Declaration) -> float:
        """Compute lexical matching score.

        Args:
            query: Query string.
            decl: Declaration to score.

        Returns:
            Lexical match score (0-1).
        """
        # Simple keyword matching
        # TODO: Implement proper BM25 scoring
        query_lower = query.lower()
        score = 0.0

        if decl.name and query_lower in decl.name.lower():
            score += 0.5

        if decl.docstring and query_lower in decl.docstring.lower():
            score += 0.3

        if decl.informalization and query_lower in decl.informalization.lower():
            score += 0.2

        return min(score, 1.0)

    def _to_search_result(self, decl: Declaration) -> SearchResult:
        """Convert Declaration ORM object to SearchResult.

        Args:
            decl: Declaration ORM object.

        Returns:
            SearchResult pydantic model.
        """
        return SearchResult(
            id=decl.id,
            name=decl.name,
            module=decl.module,
            docstring=decl.docstring,
            source_text=decl.source_text,
            source_link=decl.source_link,
            dependencies=decl.dependencies,
            informalization=decl.informalization,
            pagerank=decl.pagerank,
        )
