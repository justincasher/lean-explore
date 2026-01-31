"""Initializes the FastMCP application and its lifespan context."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from lean_explore.api import ApiClient
from lean_explore.search import Service

logger = logging.getLogger(__name__)

# Define a type for the backend service
BackendServiceType = ApiClient | Service | None


@dataclass
class AppContext:
    """Dataclass to hold application-level context for MCP tools.

    Attributes:
        backend_service: The initialized backend service (either ApiClient or
                         Service) that tools will use to perform actions.
    """

    backend_service: BackendServiceType


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Asynchronous context manager for the MCP application's lifespan.

    Args:
        server: The FastMCP application instance.

    Yields:
        AppContext: The application context containing the backend service.

    Raises:
        RuntimeError: If the backend service has not been initialized.
    """
    logger.info("MCP application lifespan starting...")

    backend_service_instance: BackendServiceType = getattr(
        server, "_lean_explore_backend_service", None
    )

    if backend_service_instance is None:
        logger.error(
            "Backend service not found on the FastMCP app instance. "
            "The MCP server script must set this attribute before running."
        )
        raise RuntimeError(
            "Backend service not initialized for MCP app. "
            "Ensure the server script correctly sets the backend service attribute."
        )

    app_context = AppContext(backend_service=backend_service_instance)

    try:
        yield app_context
    finally:
        logger.info("MCP application lifespan shutting down...")


# Create the FastMCP application instance
mcp_app = FastMCP(
    name="LeanExploreMCPServer",
    instructions=(
        "MCP Server for searching Lean 4 mathematical declarations (theorems, "
        "definitions, lemmas, instances, etc.) from Mathlib and other Lean "
        "packages.\n\n"
        "The search engine is hybrid: it matches by declaration name (e.g., "
        "'List.map', 'Nat.add') AND by informal natural language meaning (e.g., "
        "'a continuous function on a compact set', 'prime number divisibility'). "
        "You can use either style of query.\n\n"
        "Recommended workflow:\n"
        "1. Use search_summary to browse results (low token cost).\n"
        "2. Use per-field tools to fetch only what you need:\n"
        "   - get_source_code: Lean source code\n"
        "   - get_source_link: GitHub link to source\n"
        "   - get_docstring: documentation string\n"
        "   - get_description: natural language description\n"
        "   - get_module: module path in the package\n"
        "   - get_dependencies: declarations this depends on\n"
        "3. Use search only when you need full details for all results "
        "at once."
    ),
    lifespan=app_lifespan,
)
