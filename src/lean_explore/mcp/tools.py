"""Defines MCP tools for interacting with the Lean Explore search engine."""

import asyncio
import logging
from typing import TypedDict

from mcp.server.fastmcp import Context as MCPContext

from lean_explore.mcp.app import AppContext, BackendServiceType, mcp_app
from lean_explore.models import SearchResponse, SearchResult
from lean_explore.models.search_types import (
    SearchResultSummary,
    SearchSummaryResponse,
    extract_bold_description,
)


class SearchResultSummaryDict(TypedDict, total=False):
    """Serialized SearchResultSummary for slim MCP search responses."""

    id: int
    name: str
    description: str | None


class SearchSummaryResponseDict(TypedDict, total=False):
    """Serialized SearchSummaryResponse for slim MCP search responses."""

    query: str
    results: list[SearchResultSummaryDict]
    count: int
    processing_time_ms: int | None


class SearchResultDict(TypedDict, total=False):
    """Serialized SearchResult for verbose MCP tool responses."""

    id: int
    name: str
    module: str
    docstring: str | None
    source_text: str
    source_link: str
    dependencies: str | None
    informalization: str | None


class SearchResponseDict(TypedDict, total=False):
    """Serialized SearchResponse for verbose MCP tool responses."""

    query: str
    results: list[SearchResultDict]
    count: int
    processing_time_ms: int | None


logger = logging.getLogger(__name__)


async def _get_backend_from_context(ctx: MCPContext) -> BackendServiceType:
    """Retrieves the backend service from the MCP context.

    Args:
        ctx: The MCP context provided to the tool.

    Returns:
        The configured backend service (ApiClient or Service).

    Raises:
        RuntimeError: If the backend service is not available in the context.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    backend = app_ctx.backend_service
    if not backend:
        logger.error("MCP Tool Error: Backend service is not available.")
        raise RuntimeError("Backend service not configured or available for MCP tool.")
    return backend


async def _execute_backend_search(
    backend: BackendServiceType,
    query: str,
    limit: int,
    rerank_top: int | None,
    packages: list[str] | None,
) -> SearchResponse:
    """Execute a search on the backend, handling both async and sync backends.

    Args:
        backend: The backend service (ApiClient or Service).
        query: The search query string.
        limit: Maximum number of results.
        rerank_top: Number of candidates to rerank with cross-encoder.
        packages: Optional package filter.

    Returns:
        The search response from the backend.

    Raises:
        RuntimeError: If the backend does not support search.
    """
    if not hasattr(backend, "search"):
        logger.error("Backend service does not have a 'search' method.")
        raise RuntimeError("Search functionality not available on configured backend.")

    if asyncio.iscoroutinefunction(backend.search):
        return await backend.search(
            query=query, limit=limit, rerank_top=rerank_top, packages=packages
        )
    return backend.search(
        query=query, limit=limit, rerank_top=rerank_top, packages=packages
    )


@mcp_app.tool()
async def search(
    ctx: MCPContext,
    query: str,
    limit: int = 10,
    rerank_top: int | None = 50,
    packages: list[str] | None = None,
) -> SearchSummaryResponseDict:
    """Searches Lean declarations and returns concise results.

    Returns slim results (id, name, short description) to minimize token usage.
    Use get_by_id to retrieve full details for specific declarations, or
    search_verbose to get all fields upfront.

    Args:
        ctx: The MCP context, providing access to the backend service.
        query: A search query string, e.g., "continuous function".
        limit: The maximum number of search results to return. Defaults to 10.
        rerank_top: Number of candidates to rerank with cross-encoder. Set to 0 or
            None to skip reranking. Defaults to 50. Only used with local backend.
        packages: Filter results to specific packages (e.g., ["Mathlib", "Std"]).
            Defaults to None (all packages).

    Returns:
        A dictionary containing slim search results with id, name, and description.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'search' called with query: '{query}', limit: {limit}, "
        f"rerank_top: {rerank_top}, packages: {packages}"
    )

    response = await _execute_backend_search(
        backend, query, limit, rerank_top, packages
    )

    # Convert full results to slim summaries
    summary_results = [
        SearchResultSummary(
            id=result.id,
            name=result.name,
            description=extract_bold_description(result.informalization),
        )
        for result in response.results
    ]
    summary_response = SearchSummaryResponse(
        query=response.query,
        results=summary_results,
        count=response.count,
        processing_time_ms=response.processing_time_ms,
    )

    return summary_response.model_dump(exclude_none=True)


@mcp_app.tool()
async def search_verbose(
    ctx: MCPContext,
    query: str,
    limit: int = 10,
    rerank_top: int | None = 50,
    packages: list[str] | None = None,
) -> SearchResponseDict:
    """Searches Lean declarations and returns full results with all fields.

    Returns complete results including source code, dependencies, module info,
    and full informalization. Use this when you need all details upfront. For
    a more concise overview, use search instead.

    Args:
        ctx: The MCP context, providing access to the backend service.
        query: A search query string, e.g., "continuous function".
        limit: The maximum number of search results to return. Defaults to 10.
        rerank_top: Number of candidates to rerank with cross-encoder. Set to 0 or
            None to skip reranking. Defaults to 50. Only used with local backend.
        packages: Filter results to specific packages (e.g., ["Mathlib", "Std"]).
            Defaults to None (all packages).

    Returns:
        A dictionary containing the full search response with all fields.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'search_verbose' called with query: '{query}', limit: {limit}, "
        f"rerank_top: {rerank_top}, packages: {packages}"
    )

    response = await _execute_backend_search(
        backend, query, limit, rerank_top, packages
    )

    return response.model_dump(exclude_none=True)


@mcp_app.tool()
async def get_by_id(
    ctx: MCPContext,
    declaration_id: int,
) -> SearchResultDict | None:
    """Retrieves a specific declaration by its unique identifier.

    Returns the full declaration including source code, dependencies, module
    info, and informalization. Use this to expand results from the search tool.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The unique integer identifier of the declaration.

    Returns:
        A dictionary representing the SearchResult, or None if not found.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(f"MCP Tool 'get_by_id' called for declaration_id: {declaration_id}")

    if not hasattr(backend, "get_by_id"):
        logger.error("Backend service does not have a 'get_by_id' method.")
        raise RuntimeError(
            "Get by ID functionality not available on configured backend."
        )

    # Call backend get_by_id (handle both async and sync)
    if asyncio.iscoroutinefunction(backend.get_by_id):
        result: SearchResult | None = await backend.get_by_id(
            declaration_id=declaration_id
        )
    else:
        result: SearchResult | None = backend.get_by_id(declaration_id=declaration_id)

    # Return as dict for MCP, or None
    return result.model_dump(exclude_none=True) if result else None
