# src/lean_explore/shared/models/api.py

"""Pydantic models for API data interchange.

This module defines the Pydantic models that represent the structure of
request and response bodies for the remote Lean Explore API. These models
are used by the API client for data validation and serialization.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class APIPrimaryDeclarationInfo(BaseModel):
    """Minimal information about a primary declaration within an API response.

    Attributes:
        lean_name: The Lean name of the primary declaration, if available.
    """

    lean_name: Optional[str] = Field(
        None, description="The Lean name of the primary declaration."
    )


class APISearchResultItem(BaseModel):
    """Represents a single statement group item as returned by API endpoints.

    This model is used for items in search results and for the direct
    retrieval of a statement group by its ID.

    Attributes:
        id: The unique identifier of the statement group.
        primary_declaration: Information about the primary declaration.
        source_file: The source file where the statement group is located.
        range_start_line: Start line of statement group in source file.
        display_statement_text: Display-friendly statement text, if available.
        statement_text: The full canonical statement text.
        docstring: The docstring associated with the statement group, if available.
        informal_description: Informal description of the statement group, if available.
    """

    id: int = Field(..., description="Unique identifier for the statement group.")
    primary_declaration: APIPrimaryDeclarationInfo = Field(
        ...,
        description="Information about the primary declaration of the statement group.",
    )
    source_file: str = Field(
        ..., description="The source file path for the statement group."
    )
    range_start_line: int = Field(
        ...,
        description="Line number of statement group in its source file.",
    )
    display_statement_text: Optional[str] = Field(
        None, description="A display-optimized version of the statement text."
    )
    statement_text: str = Field(
        ..., description="The complete canonical text of the statement group."
    )
    docstring: Optional[str] = Field(
        None, description="The docstring associated with the statement group."
    )
    informal_description: Optional[str] = Field(
        None,
        description="An informal, human-readable description of the statement group.",
    )


class APISearchResponse(BaseModel):
    """Represents the complete response structure for a search API call.

    Attributes:
        query: The original search query string submitted by the user.
        packages_applied: List of package filters applied to the search, if any.
        results: A list of search result items.
        count: The number of results returned in the current response.
        total_candidates_considered: The total number of potential candidates
            considered by the search algorithm before limiting results.
        processing_time_ms: Server processing time for search request, in milliseconds.
    """

    query: str = Field(..., description="The search query that was executed.")
    packages_applied: Optional[List[str]] = Field(
        None, description="List of package filters applied to the search."
    )
    results: List[APISearchResultItem] = Field(
        ..., description="A list of search results."
    )
    count: int = Field(
        ..., description="The number of results provided in this response."
    )
    total_candidates_considered: int = Field(
        ..., description="Total number of candidate results before truncation."
    )
    processing_time_ms: int = Field(
        ..., description="Server-side processing time for the search in milliseconds."
    )


class APICitationsResponse(BaseModel):
    """Represents the response structure for a dependencies (citations) API call.

    Attributes:
        source_group_id: ID of the statement group for which citations were requested.
        citations: A list of statement groups that are cited by the source group.
        count: The number of citations found and returned.
    """

    source_group_id: int = Field(
        ..., description="The ID of the statement group whose citations are listed."
    )
    citations: List[APISearchResultItem] = Field(
        ..., description="A list of statement groups cited by the source group."
    )
    count: int = Field(..., description="The number of citations provided.")
