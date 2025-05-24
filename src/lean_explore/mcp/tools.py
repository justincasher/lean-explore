# src/lean_explore/mcp/tools.py

"""Defines MCP tools for interacting with the Lean Explore search engine.

These tools provide functionalities such as searching for statement groups,
retrieving specific groups by ID, and getting their dependencies. They
utilize a backend service (either an API client or a local service)
made available through the MCP application context.
"""

import asyncio  # Needed for asyncio.iscoroutinefunction
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context as MCPContext

from lean_explore.mcp.app import AppContext, BackendServiceType, mcp_app

# Import Pydantic models for type hinting and for creating response dicts
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APISearchResponse,
    APISearchResultItem,
)

logger = logging.getLogger(__name__)


async def _get_backend_from_context(ctx: MCPContext) -> BackendServiceType:
    """Retrieves the backend service from the MCP context.

    Args:
        ctx: The MCP context provided to the tool.

    Returns:
        The configured backend service (APIClient or LocalService).
        Guaranteed to be non-None if this function returns, otherwise
        it raises an exception.

    Raises:
        RuntimeError: If the backend service is not available in the context,
                      indicating a server configuration issue.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    backend = app_ctx.backend_service
    if not backend:
        logger.error(
            "MCP Tool Error: Backend service is not available in lifespan_context."
        )
        raise RuntimeError("Backend service not configured or available for MCP tool.")
    return backend


def _prepare_mcp_result_item(backend_item: APISearchResultItem) -> APISearchResultItem:
    """Prepares an APISearchResultItem for MCP response.

    This helper ensures that the item sent over MCP does not include
    the display_statement_text, as the full statement_text is preferred
    for model consumption.

    Args:
        backend_item: The item as received from the backend service.

    Returns:
        A new APISearchResultItem instance suitable for MCP responses.
    """
    # Create a new instance or use .model_copy(update=...) for Pydantic v2
    return APISearchResultItem(
        id=backend_item.id,
        primary_declaration=backend_item.primary_declaration.model_copy()
        if backend_item.primary_declaration
        else None,
        source_file=backend_item.source_file,
        range_start_line=backend_item.range_start_line,
        statement_text=backend_item.statement_text,
        docstring=backend_item.docstring,
        informal_description=backend_item.informal_description,
        display_statement_text=None,  # Ensure this is not sent over MCP
    )


@mcp_app.tool()
async def search(
    ctx: MCPContext,
    query: str,
    package_filters: Optional[List[str]] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Searches Lean statement groups by a query string.

    This tool allows for filtering by package names and limits the number
    of results returned.

    Args:
        ctx: The MCP context, providing access to shared resources like the
             backend service.
        query: The search query string. For example, "continuous function" or
               "prime number theorem".
        package_filters: An optional list of package names to filter the search
                         results by. For example, `["Mathlib.Analysis",
                         "Mathlib.Order"]`. If None or empty, no package filter
                         is applied.
        limit: The maximum number of search results to return from this tool.
               Defaults to 10. Must be a positive integer.

    Returns:
        A dictionary corresponding to the APISearchResponse model, containing
        the search results (potentially truncated by the `limit` parameter of
        this tool), and metadata about the search operation. The
        `display_statement_text` field within each result item is omitted.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'search' called with query: '{query}', "
        f"packages: {package_filters}, tool_limit: {limit}"
    )

    if not hasattr(backend, "search"):
        logger.error("Backend service does not have a 'search' method.")
        # This should ideally return a structured error for MCP if possible.
        # For now, FastMCP will convert this RuntimeError.
        raise RuntimeError("Search functionality not available on configured backend.")

    tool_limit = max(1, limit)  # Ensure limit is at least 1 for slicing
    api_response_pydantic: Optional[APISearchResponse]

    # Conditionally await based on the backend's search method type
    if asyncio.iscoroutinefunction(backend.search):
        api_response_pydantic = await backend.search(
            query=query,
            package_filters=package_filters,
            # The backend.search method uses its own internal default for limit
            # if None is passed, or the passed limit.
            # The MCP tool will truncate the results later using tool_limit.
        )
    else:
        api_response_pydantic = backend.search(
            query=query, package_filters=package_filters
        )

    if not api_response_pydantic:
        logger.warning("Backend search returned None, responding with empty results.")
        empty_response = APISearchResponse(
            query=query,
            packages_applied=package_filters or [],
            results=[],
            count=0,
            total_candidates_considered=0,
            processing_time_ms=0,
        )
        return empty_response.model_dump(exclude_none=True)

    actual_backend_results = api_response_pydantic.results

    mcp_results_list = []
    for backend_item in actual_backend_results[:tool_limit]:  # Apply MCP tool's limit
        mcp_results_list.append(_prepare_mcp_result_item(backend_item))

    final_mcp_response = APISearchResponse(
        query=api_response_pydantic.query,
        packages_applied=api_response_pydantic.packages_applied,
        results=mcp_results_list,
        count=len(mcp_results_list),  # Count is after this tool's truncation
        total_candidates_considered=api_response_pydantic.total_candidates_considered,
        processing_time_ms=api_response_pydantic.processing_time_ms,
    )

    return final_mcp_response.model_dump(exclude_none=True)


@mcp_app.tool()
async def get_by_id(ctx: MCPContext, group_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a specific statement group by its unique identifier.

    The `display_statement_text` field is omitted from the response.

    Args:
        ctx: The MCP context, providing access to the backend service.
        group_id: The unique integer identifier of the statement group to retrieve.
                  For example, `12345`.

    Returns:
        A dictionary corresponding to the APISearchResultItem model if a
        statement group with the given ID is found (with
        `display_statement_text` omitted). Returns None (which will be
        serialized as JSON null by MCP) if no such group exists.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(f"MCP Tool 'get_by_id' called for group_id: {group_id}")

    backend_item: Optional[APISearchResultItem]
    if asyncio.iscoroutinefunction(backend.get_by_id):
        backend_item = await backend.get_by_id(group_id=group_id)
    else:
        backend_item = backend.get_by_id(group_id=group_id)

    if backend_item:
        mcp_item = _prepare_mcp_result_item(backend_item)
        return mcp_item.model_dump(exclude_none=True)
    return None


@mcp_app.tool()
async def get_dependencies(ctx: MCPContext, group_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves the direct dependencies (citations) for a specific statement group.

    The `display_statement_text` field within each cited item is omitted
    from the response.

    Args:
        ctx: The MCP context, providing access to the backend service.
        group_id: The unique integer identifier of the statement group for which
                  to fetch its direct dependencies. For example, `12345`.

    Returns:
        A dictionary corresponding to the APICitationsResponse model, which
        contains a list of cited statement groups (each with
        `display_statement_text` omitted), if the source group_id
        is found and has dependencies. Returns None (serialized as JSON null
        by MCP) if the source group is not found or has no dependencies.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(f"MCP Tool 'get_dependencies' called for group_id: {group_id}")

    backend_response: Optional[APICitationsResponse]
    if asyncio.iscoroutinefunction(backend.get_dependencies):
        backend_response = await backend.get_dependencies(group_id=group_id)
    else:
        backend_response = backend.get_dependencies(group_id=group_id)

    if backend_response:
        mcp_citations_list = []
        for backend_item in backend_response.citations:
            mcp_citations_list.append(_prepare_mcp_result_item(backend_item))

        final_mcp_response = APICitationsResponse(
            source_group_id=backend_response.source_group_id,
            citations=mcp_citations_list,
            count=len(mcp_citations_list),
        )
        return final_mcp_response.model_dump(exclude_none=True)
    return None
