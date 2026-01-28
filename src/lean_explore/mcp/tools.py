"""Defines MCP tools for interacting with the Lean Explore search engine."""

import asyncio
import logging
from typing import TypedDict

from mcp.server.fastmcp import Context as MCPContext

from lean_explore.mcp.app import AppContext, BackendServiceType, mcp_app
from lean_explore.models import SearchResponse, SearchResult


class SearchResultDict(TypedDict, total=False):
    """Serialized SearchResult for MCP tool responses."""

    id: int
    name: str
    module: str
    docstring: str | None
    source_text: str
    source_link: str
    dependencies: str | None
    informalization: str | None


class SearchResponseDict(TypedDict, total=False):
    """Serialized SearchResponse for MCP tool responses."""

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


@mcp_app.tool()
async def search(
    ctx: MCPContext,
    query: str,
    limit: int = 10,
    rerank_top: int | None = 50,
) -> SearchResponseDict:
    """Searches Lean declarations by a query string.

    Args:
        ctx: The MCP context, providing access to the backend service.
        query: A search query string, e.g., "continuous function".
        limit: The maximum number of search results to return. Defaults to 10.
        rerank_top: Number of candidates to rerank with cross-encoder. Set to 0 or
            None to skip reranking. Defaults to 50. Only used with local backend.

    Returns:
        A dictionary containing the search response with results.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'search' called with query: '{query}', limit: {limit}, "
        f"rerank_top: {rerank_top}"
    )

    if not hasattr(backend, "search"):
        logger.error("Backend service does not have a 'search' method.")
        raise RuntimeError("Search functionality not available on configured backend.")

    # Call backend search (handle both async and sync)
    if asyncio.iscoroutinefunction(backend.search):
        response: SearchResponse = await backend.search(
            query=query, limit=limit, rerank_top=rerank_top
        )
    else:
        response: SearchResponse = backend.search(
            query=query, limit=limit, rerank_top=rerank_top
        )

    # Return as dict for MCP
    return response.model_dump(exclude_none=True)


@mcp_app.tool()
async def get_by_id(
    ctx: MCPContext,
    declaration_id: int,
) -> SearchResultDict | None:
    """Retrieves a specific declaration by its unique identifier.

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
