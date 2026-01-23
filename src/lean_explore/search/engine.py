"""Core search engine functionality for Lean declarations.

This module provides the core search functionality using SQLite for storage,
FAISS for semantic search, combined with BM25 lexical matching and PageRank scoring.

Note: On macOS, torch and FAISS have OpenMP library conflicts. To avoid segfaults:
- FAISS is imported lazily (not at module level)
- When semantic search is needed, torch/embeddings are loaded FIRST, then FAISS
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from lean_explore.config import Config
from lean_explore.models import Declaration, SearchResult

if TYPE_CHECKING:
    import faiss

    from lean_explore.util import EmbeddingClient

logger = logging.getLogger(__name__)


class SearchEngine:
    """Core search engine for Lean declarations.

    Combines semantic search (FAISS), lexical matching (BM25),
    and PageRank scoring to rank results.
    """

    def __init__(
        self,
        db_url: str | None = None,
        embedding_client: "EmbeddingClient | None" = None,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        faiss_index_path: Path | None = None,
        faiss_ids_map_path: Path | None = None,
        use_local_data: bool = True,
    ):
        """Initialize the search engine.

        Args:
            db_url: Database URL. Defaults to configured URL.
            embedding_client: Client for generating embeddings. Created lazily if None.
            embedding_model_name: Name of the embedding model to use.
            faiss_index_path: Path to FAISS index. Defaults to config path.
            faiss_ids_map_path: Path to FAISS ID mapping. Defaults to config path.
            use_local_data: If True, use DATA_DIRECTORY paths. If False, use
                CACHE_DIRECTORY paths (for downloaded remote data).
        """
        # Store for lazy embedding client creation
        self._embedding_client = embedding_client
        self._embedding_model_name = embedding_model_name

        # Select paths based on data source
        if use_local_data:
            base_path = Config.ACTIVE_DATA_PATH
            default_db_url = Config.EXTRACTION_DATABASE_URL
        else:
            base_path = Config.ACTIVE_CACHE_PATH
            default_db_url = Config.DATABASE_URL

        self.db_url = db_url or default_db_url
        self.engine: AsyncEngine = create_async_engine(self.db_url)

        # Store paths for lazy FAISS loading (to avoid torch/FAISS OpenMP conflict)
        self._faiss_index_path = faiss_index_path or (
            base_path / "informalization_faiss.index"
        )
        self._faiss_ids_map_path = faiss_ids_map_path or (
            base_path / "informalization_faiss_ids_map.json"
        )
        self._faiss_index: faiss.Index | None = None
        self._faiss_id_to_declaration_id: list[int] | None = None

        # Validate paths exist (fail fast)
        if not self._faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._faiss_index_path}. "
                "Please run 'lean-explore download' to fetch the data."
            )
        if not self._faiss_ids_map_path.exists():
            raise FileNotFoundError(
                f"FAISS ID mapping not found at {self._faiss_ids_map_path}. "
                "Please run 'lean-explore download' to fetch the data."
            )

    @property
    def embedding_client(self) -> "EmbeddingClient":
        """Lazily create the embedding client to avoid loading torch at import time.

        This prevents OpenMP library conflicts between torch and FAISS on macOS.
        The embedding client (and torch) must be loaded BEFORE FAISS.
        """
        if self._embedding_client is None:
            from lean_explore.util import EmbeddingClient

            self._embedding_client = EmbeddingClient(
                model_name=self._embedding_model_name,
                max_length=512,
            )
        return self._embedding_client

    def _ensure_faiss_loaded(self) -> None:
        """Load FAISS index and ID mapping if not already loaded.

        Important: This must be called AFTER torch is loaded (via embedding_client)
        to avoid OpenMP library conflicts on macOS.
        """
        if self._faiss_index is not None:
            return

        import faiss

        logger.info(f"Loading FAISS index from {self._faiss_index_path}")
        self._faiss_index = faiss.read_index(str(self._faiss_index_path))

        logger.info(f"Loading FAISS ID mapping from {self._faiss_ids_map_path}")
        with open(self._faiss_ids_map_path) as file:
            self._faiss_id_to_declaration_id = json.load(file)

    @property
    def faiss_index(self) -> "faiss.Index":
        """Get the FAISS index, loading it if necessary."""
        self._ensure_faiss_loaded()
        return self._faiss_index  # type: ignore[return-value]

    @property
    def faiss_id_to_declaration_id(self) -> list[int]:
        """Get the FAISS ID to declaration ID mapping, loading it if necessary."""
        self._ensure_faiss_loaded()
        return self._faiss_id_to_declaration_id  # type: ignore[return-value]

    async def search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.4,
        pagerank_weight: float = 0.3,
        lexical_weight: float = 0.3,
    ) -> list[SearchResult]:
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
        # Generate embedding FIRST (loads torch before FAISS to avoid OpenMP conflict)
        embedding_response = await self.embedding_client.embed([query])
        query_embedding = np.array([embedding_response.embeddings[0]], dtype=np.float32)

        # Now safe to load/use FAISS (torch is already loaded)
        num_candidates = limit * 3
        distances, faiss_indices = self.faiss_index.search(
            query_embedding, num_candidates
        )

        declaration_ids = [
            self.faiss_id_to_declaration_id[idx]
            for idx in faiss_indices[0]
            if idx < len(self.faiss_id_to_declaration_id)
        ]

        semantic_scores = {}
        for idx, distance in zip(faiss_indices[0], distances[0]):
            if idx < len(self.faiss_id_to_declaration_id):
                declaration_id = self.faiss_id_to_declaration_id[idx]
                similarity = 1 / (1 + distance)
                semantic_scores[declaration_id] = similarity

        async with AsyncSession(self.engine) as session:
            stmt = select(Declaration).where(Declaration.id.in_(declaration_ids))
            result = await session.execute(stmt)
            candidates = result.scalars().all()

            scored_results = []
            for decl in candidates:
                semantic_score = semantic_scores.get(decl.id, 0.0)
                pagerank_score = decl.pagerank or 0.0
                lexical_score = self._compute_lexical_score(query, decl)

                final_score = (
                    semantic_weight * semantic_score
                    + pagerank_weight * pagerank_score
                    + lexical_weight * lexical_score
                )

                scored_results.append((decl, final_score))

            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [self._to_search_result(decl) for decl, _ in scored_results[:limit]]

    async def get_by_id(self, declaration_id: int) -> SearchResult | None:
        """Retrieve a declaration by ID.

        Args:
            declaration_id: The declaration ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        async with AsyncSession(self.engine) as session:
            decl = await session.get(Declaration, declaration_id)
            return self._to_search_result(decl) if decl else None

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
