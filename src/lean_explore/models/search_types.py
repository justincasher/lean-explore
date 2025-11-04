"""Type definitions for search results and related data structures."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A search result representing a Lean declaration.

    This model represents the core information returned from a search query,
    mirroring the essential fields from the database Declaration model.
    """

    id: int
    """Primary key identifier."""

    name: str
    """Fully qualified Lean name (e.g., 'Nat.add')."""

    kind: str
    """Declaration kind (e.g., 'theorem', 'def', 'structure', 'class')."""

    module: str
    """Module name (e.g., 'Mathlib.Data.List.Basic')."""

    docstring: str | None
    """Documentation string from the source code, if available."""

    source_text: str
    """The actual Lean source code for this declaration."""

    source_link: str
    """GitHub URL to the declaration source code."""

    dependencies: str | None
    """JSON array of declaration names this declaration depends on."""

    informalization: str | None
    """Natural language description of the declaration."""

    pagerank: float | None
    """PageRank score based on dependency graph."""

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class SearchResponse(BaseModel):
    """Response from a search operation containing results and metadata."""

    query: str
    """The original search query string."""

    results: list[SearchResult]
    """List of search results."""

    count: int
    """Number of results returned."""

    processing_time_ms: int | None = None
    """Processing time in milliseconds, if available."""
