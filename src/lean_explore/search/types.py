"""Type definitions for search results and related data structures."""

from typing import Optional

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

    module: str
    """Module name (e.g., 'Mathlib.Data.List.Basic')."""

    docstring: Optional[str]
    """Documentation string from the source code, if available."""

    source_text: str
    """The actual Lean source code for this declaration."""

    source_link: str
    """GitHub URL to the declaration source code."""

    dependencies: Optional[str]
    """JSON array of declaration names this declaration depends on."""

    informalization: Optional[str]
    """Natural language description of the declaration."""

    pagerank: Optional[float]
    """PageRank score based on dependency graph."""

    class Config:
        """Pydantic configuration."""

        from_attributes = True
