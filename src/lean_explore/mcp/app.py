# src/lean_explore/mcp/app.py

"""Initializes the FastMCP application and its lifespan context.

This module creates the main FastMCP application instance and defines
a lifespan context manager. The lifespan manager is responsible for
making the configured backend service (API client or local service)
available to MCP tools via the request context. The actual backend
instance will be set by the server startup script before running the app.
"""

import logging
from contextlib import asynccontextmanager
from typing import Union, AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context as MCPContext

# Import your backend service types for type hinting
from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService

logger = logging.getLogger(__name__)

# Define a type for the backend service to be used by tools
BackendServiceType = Union[APIClient, LocalService, None]

@dataclass
class AppContext:
    """Dataclass to hold application-level context for MCP tools.

    Attributes:
        backend_service: The initialized backend service (either APIClient or
                         LocalService) that tools will use to perform actions.
                         Will be None if not properly initialized by the server script.
    """
    backend_service: BackendServiceType


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Asynchronous context manager for the MCP application's lifespan.

    This function is called by FastMCP when the server starts and stops.
    It retrieves the backend service instance (which should have been
    initialized and attached to an attribute of the `server` instance,
    e.g., `server._lean_explore_backend_service`, by the main server script)
    and makes it available in the AppContext.

    Args:
        server: The FastMCP application instance.

    Yields:
        AppContext: The application context containing the backend service.
    
    Raises:
        RuntimeError: If the backend service has not been initialized and
                      set on an attribute of the `server` instance prior to
                      the app running.
    """
    logger.info("MCP application lifespan starting...")
    
    # The main server script (mcp/server.py) is expected to instantiate
    # the backend (APIClient or LocalService) based on its startup arguments
    # and store it as an attribute on the mcp_app instance (e.g., 
    # mcp_app._lean_explore_backend_service) before mcp_app.run() is called.
    backend_service_instance: BackendServiceType = getattr(server, '_lean_explore_backend_service', None)

    if backend_service_instance is None:
        logger.error(
            "Backend service not found on the FastMCP app instance. "
            "The MCP server script must set this attribute (e.g., "
            "'_lean_explore_backend_service') before running the app."
        )
        raise RuntimeError(
            "Backend service not initialized for MCP app. "
            "Ensure the server script correctly sets the backend service attribute "
            "on the FastMCP app instance."
        )

    app_context = AppContext(backend_service=backend_service_instance)
    
    try:
        yield app_context
    finally:
        logger.info("MCP application lifespan shutting down...")
        pass


# Create the FastMCP application instance
# The lifespan manager will be associated with this app.
mcp_app = FastMCP(
    "LeanExploreMCPServer",
    version="0.1.0",
    description="MCP Server for Lean Explore, providing tools to search and query Lean mathematical data.",
    lifespan=app_lifespan,
)

mcp_app.lifespan = app_lifespan