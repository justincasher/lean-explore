"""Initializes the FastMCP application and its lifespan context."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Union

from mcp.server.fastmcp import FastMCP

from lean_explore.api import ApiClient
from lean_explore.search import Service

logger = logging.getLogger(__name__)

# Define a type for the backend service
BackendServiceType = Union[ApiClient, Service, None]


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
    "LeanExploreMCPServer",
    version="0.1.0",
    description="MCP Server for Lean Explore, providing tools to search Lean declarations.",
    lifespan=app_lifespan,
)

mcp_app.lifespan = app_lifespan
