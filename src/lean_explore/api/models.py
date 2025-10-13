# src/lean_explore/models/api.py

"""Pydantic models for API data interchange.

This module defines the Pydantic models that represent the structure of
request and response bodies for the remote Lean Explore API. These models
are used by the API client for data validation and serialization.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class APIPrimaryDeclarationInfo(BaseModel):
    """Minimal information about a primary declaration within an API response."""

    lean_name: Optional[str] = None
    """The Lean name of the primary declaration, if available."""


class APISearchResultItem(BaseModel):
    """Represents a single statement group item as returned by API endpoints."""

    id: int
    """The unique identifier of the statement group."""

    primary_declaration: APIPrimaryDeclarationInfo
    """Information about the primary declaration."""

    source_file: str
    """The source file where the statement group is located."""

    range_start_line: int
    """Start line of statement group in source file."""

    display_statement_text: Optional[str] = None
    """Display-friendly statement text, if available."""

    statement_text: str
    """The full canonical statement text."""

    docstring: Optional[str] = None
    """The docstring associated with the statement group, if available."""

    informal_description: Optional[str] = None
    """Informal description of the statement group, if available."""


class APISearchResponse(BaseModel):
    """Represents the complete response structure for a search API call."""

    query: str
    """The original search query string submitted by the user."""

    packages_applied: Optional[List[str]] = None
    """List of package filters applied to the search, if any."""

    results: List[APISearchResultItem]
    """A list of search result items."""

    count: int
    """The number of results returned in the current response."""

    total_candidates_considered: int
    """The total number of potential candidates considered by the search algorithm."""

    processing_time_ms: int
    """Server processing time for search request, in milliseconds."""


class APICitationsResponse(BaseModel):
    """Represents the response structure for a dependencies (citations) API call."""

    source_group_id: int
    """ID of the statement group for which citations were requested."""

    citations: List[APISearchResultItem]
    """A list of statement groups that are cited by the source group."""

    count: int
    """The number of citations found and returned."""
