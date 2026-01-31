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


class SourceCodeResultDict(TypedDict):
    """Result containing declaration id, name, and source code."""

    id: int
    name: str
    source_text: str


class SourceLinkResultDict(TypedDict):
    """Result containing declaration id, name, and GitHub source link."""

    id: int
    name: str
    source_link: str


class DocstringResultDict(TypedDict):
    """Result containing declaration id, name, and docstring."""

    id: int
    name: str
    docstring: str | None


class DescriptionResultDict(TypedDict):
    """Result containing declaration id, name, and informalization."""

    id: int
    name: str
    informalization: str | None


class ModuleResultDict(TypedDict):
    """Result containing declaration id, name, and module path."""

    id: int
    name: str
    module: str


class DependenciesResultDict(TypedDict):
    """Result containing declaration id, name, and dependencies."""

    id: int
    name: str
    dependencies: str | None


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


async def _execute_backend_get_by_id(
    backend: BackendServiceType,
    declaration_id: int,
) -> SearchResult | None:
    """Execute get_by_id on the backend, handling both async and sync backends.

    Args:
        backend: The backend service (ApiClient or Service).
        declaration_id: The numeric id of the declaration to retrieve.

    Returns:
        The SearchResult from the backend, or None if not found.

    Raises:
        RuntimeError: If the backend does not support get_by_id.
    """
    if not hasattr(backend, "get_by_id"):
        logger.error("Backend service does not have a 'get_by_id' method.")
        raise RuntimeError(
            "Get by ID functionality not available on configured backend."
        )

    if asyncio.iscoroutinefunction(backend.get_by_id):
        return await backend.get_by_id(declaration_id=declaration_id)
    return backend.get_by_id(declaration_id=declaration_id)


@mcp_app.tool()
async def search(
    ctx: MCPContext,
    query: str,
    limit: int = 10,
    rerank_top: int | None = 50,
    packages: list[str] | None = None,
) -> SearchResponseDict:
    """Search Lean 4 declarations and return full results including source code.

    Accepts two kinds of queries:
      - By name: a full or partial Lean declaration name, e.g., "List.map",
        "Nat.Prime", "CategoryTheory.Functor.map".
      - By meaning: an informal natural language description, e.g.,
        "continuous function on a compact set", "sum of a geometric series",
        "a group homomorphism preserving multiplication".

    The search engine handles both styles simultaneously via hybrid retrieval
    (lexical name matching + semantic similarity), so you do not need to
    specify which kind of query you are making.

    Returns full results including source code, module, dependencies, and
    informalization for every hit. If you only need names and short
    descriptions, prefer search_summary to save tokens, then use the
    per-field tools (get_source_code, get_docstring, get_description,
    get_module, get_dependencies) for the entries you care about.

    Args:
        ctx: The MCP context, providing access to the backend service.
        query: A Lean declaration name (e.g., "List.filter") or an informal
            natural language description (e.g., "prime number divisibility").
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
        f"MCP Tool 'search' called with query: '{query}', limit: {limit}, "
        f"rerank_top: {rerank_top}, packages: {packages}"
    )

    response = await _execute_backend_search(
        backend, query, limit, rerank_top, packages
    )

    return response.model_dump(exclude_none=True)


@mcp_app.tool()
async def search_summary(
    ctx: MCPContext,
    query: str,
    limit: int = 10,
    rerank_top: int | None = 50,
    packages: list[str] | None = None,
) -> SearchSummaryResponseDict:
    """Search Lean 4 declarations and return concise results (recommended first step).

    This is the preferred starting point for search. Returns only id, name,
    and a short natural language description for each hit, keeping token
    usage low. After reviewing these summaries, use the per-field tools
    (get_source_code, get_docstring, get_description, get_module,
    get_dependencies) for the entries you need details on.

    Accepts two kinds of queries:
      - By name: a full or partial Lean declaration name, e.g., "List.map",
        "Nat.Prime", "CategoryTheory.Functor.map".
      - By meaning: an informal natural language description, e.g.,
        "continuous function on a compact set", "sum of a geometric series",
        "a group homomorphism preserving multiplication".

    The search engine handles both styles simultaneously via hybrid retrieval
    (lexical name matching + semantic similarity), so you do not need to
    specify which kind of query you are making.

    Args:
        ctx: The MCP context, providing access to the backend service.
        query: A Lean declaration name (e.g., "List.filter") or an informal
            natural language description (e.g., "prime number divisibility").
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
        f"MCP Tool 'search_summary' called with query: '{query}', limit: {limit}, "
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
async def get_source_code(
    ctx: MCPContext,
    declaration_id: int,
) -> SourceCodeResultDict | None:
    """Retrieve the Lean source code for a declaration by id.

    Returns the declaration name and its Lean 4 source code. Use this after
    calling search_summary to inspect the actual implementation.

    The id values come from the search or search_summary result lists.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The numeric id from a search or search_summary result.

    Returns:
        A dictionary with id, name, and source_text, or None if the id
        does not exist.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'get_source_code' called for declaration_id: {declaration_id}"
    )

    result = await _execute_backend_get_by_id(backend, declaration_id)
    if result is None:
        return None

    return SourceCodeResultDict(
        id=result.id,
        name=result.name,
        source_text=result.source_text,
    )


@mcp_app.tool()
async def get_source_link(
    ctx: MCPContext,
    declaration_id: int,
) -> SourceLinkResultDict | None:
    """Retrieve the GitHub source link for a declaration by id.

    Returns the declaration name and a URL to the source code on GitHub.
    Use this when you need to reference or link to the original source.

    The id values come from the search or search_summary result lists.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The numeric id from a search or search_summary result.

    Returns:
        A dictionary with id, name, and source_link, or None if the id
        does not exist.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'get_source_link' called for declaration_id: {declaration_id}"
    )

    result = await _execute_backend_get_by_id(backend, declaration_id)
    if result is None:
        return None

    return SourceLinkResultDict(
        id=result.id,
        name=result.name,
        source_link=result.source_link,
    )


@mcp_app.tool()
async def get_docstring(
    ctx: MCPContext,
    declaration_id: int,
) -> DocstringResultDict | None:
    """Retrieve the docstring for a declaration by id.

    Returns the declaration name and its documentation string from the Lean
    source code. Use this to check what documentation exists without
    fetching the full source code.

    The id values come from the search or search_summary result lists.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The numeric id from a search or search_summary result.

    Returns:
        A dictionary with id, name, and docstring, or None if the id
        does not exist.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'get_docstring' called for declaration_id: {declaration_id}"
    )

    result = await _execute_backend_get_by_id(backend, declaration_id)
    if result is None:
        return None

    return DocstringResultDict(
        id=result.id,
        name=result.name,
        docstring=result.docstring,
    )


@mcp_app.tool()
async def get_description(
    ctx: MCPContext,
    declaration_id: int,
) -> DescriptionResultDict | None:
    """Retrieve the natural language description for a declaration by id.

    Returns the declaration name and its informalization, an AI-generated
    plain-English explanation of what the declaration states or does.

    The id values come from the search or search_summary result lists.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The numeric id from a search or search_summary result.

    Returns:
        A dictionary with id, name, and informalization, or None if the id
        does not exist.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'get_description' called for declaration_id: {declaration_id}"
    )

    result = await _execute_backend_get_by_id(backend, declaration_id)
    if result is None:
        return None

    return DescriptionResultDict(
        id=result.id,
        name=result.name,
        informalization=result.informalization,
    )


@mcp_app.tool()
async def get_module(
    ctx: MCPContext,
    declaration_id: int,
) -> ModuleResultDict | None:
    """Retrieve the module path for a declaration by id.

    Returns the declaration name and the Lean module it belongs to
    (e.g., 'Mathlib.Data.List.Basic'). Use this to find where a
    declaration lives in the package structure.

    The id values come from the search or search_summary result lists.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The numeric id from a search or search_summary result.

    Returns:
        A dictionary with id, name, and module, or None if the id does
        not exist.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'get_module' called for declaration_id: {declaration_id}"
    )

    result = await _execute_backend_get_by_id(backend, declaration_id)
    if result is None:
        return None

    return ModuleResultDict(
        id=result.id,
        name=result.name,
        module=result.module,
    )


@mcp_app.tool()
async def get_dependencies(
    ctx: MCPContext,
    declaration_id: int,
) -> DependenciesResultDict | None:
    """Retrieve the dependencies for a declaration by id.

    Returns the declaration name and a JSON array of other declaration
    names that this declaration depends on. Use this to understand what
    a declaration builds upon.

    The id values come from the search or search_summary result lists.

    Args:
        ctx: The MCP context, providing access to the backend service.
        declaration_id: The numeric id from a search or search_summary result.

    Returns:
        A dictionary with id, name, and dependencies, or None if the id
        does not exist.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'get_dependencies' called for declaration_id: {declaration_id}"
    )

    result = await _execute_backend_get_by_id(backend, declaration_id)
    if result is None:
        return None

    return DependenciesResultDict(
        id=result.id,
        name=result.name,
        dependencies=result.dependencies,
    )
