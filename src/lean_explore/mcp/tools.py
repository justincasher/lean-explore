"""Defines MCP tools for interacting with the Lean Explore search engine."""

import asyncio
import logging
from typing import Any, Dict

from mcp.server.fastmcp import Context as MCPContext

from lean_explore.mcp.app import AppContext, BackendServiceType, mcp_app
from lean_explore.search.types import SearchResponse, SearchResult

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
) -> Dict[str, Any]:
    """Searches Lean declarations by a query string.

    Args:
        ctx: The MCP context, providing access to the backend service.
        query: A search query string, e.g., "continuous function".
        limit: The maximum number of search results to return. Defaults to 10.

    Returns:
        A dictionary containing the search response with results.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(f"MCP Tool 'search' called with query: '{query}', limit: {limit}")

    if not hasattr(backend, "search"):
        logger.error("Backend service does not have a 'search' method.")
        raise RuntimeError("Search functionality not available on configured backend.")

    # Call backend search (handle both async and sync)
    if asyncio.iscoroutinefunction(backend.search):
        response: SearchResponse = await backend.search(query=query, limit=limit)
    else:
        response: SearchResponse = backend.search(query=query, limit=limit)

    # Return as dict for MCP
    return response.model_dump(exclude_none=True)


@mcp_app.tool()
async def get_by_id(
    ctx: MCPContext,
    declaration_id: int,
) -> Dict[str, Any] | None:
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
        raise RuntimeError("Get by ID functionality not available on configured backend.")

    # Call backend get_by_id (handle both async and sync)
    if asyncio.iscoroutinefunction(backend.get_by_id):
        result: SearchResult | None = await backend.get_by_id(declaration_id=declaration_id)
    else:
        result: SearchResult | None = backend.get_by_id(declaration_id=declaration_id)

    # Return as dict for MCP, or None
    return result.model_dump(exclude_none=True) if result else None
