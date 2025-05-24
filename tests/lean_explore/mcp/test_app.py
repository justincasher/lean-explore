# tests/lean_explore/mcp/test_app.py

"""Tests for the FastMCP application instance and its lifespan context.

This module verifies that the `FastMCP` application in `lean_explore.mcp.app`
correctly handles its lifespan, specifically focusing on the proper
initialization and availability of the backend service within the application context.
"""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService
from lean_explore.mcp.app import AppContext, mcp_app


@pytest.mark.asyncio
class TestFastMCPAppLifespan:
    """Test suite for the FastMCP application's lifespan context.

    These tests verify how the `app_lifespan` context manager, which is
    responsible for setting up the backend service within the application's
    lifecycle, behaves under various conditions.
    """

    async def test_app_lifespan_initializes_backend_service_api(
        self, mocker: "MockerFixture"
    ):
        """Verifies that the lifespan context correctly picks up an APIClient backend.

        This test simulates the MCP server having set an APIClient instance
        on the `mcp_app` object before starting the lifespan.
        """
        mock_api_client = mocker.MagicMock(spec=APIClient)
        mcp_app._lean_explore_backend_service = mock_api_client

        async with mcp_app.lifespan(mcp_app) as app_context:
            assert isinstance(app_context, AppContext)
            assert app_context.backend_service is mock_api_client
            assert isinstance(app_context.backend_service, APIClient)

        del mcp_app._lean_explore_backend_service  # Clean up the attribute

    async def test_app_lifespan_initializes_backend_service_local(
        self, mocker: "MockerFixture"
    ):
        """Verifies that the lifespan context correctly picks up a LocalService backend.

        This test simulates the MCP server having set a LocalService instance
        on the `mcp_app` object before starting the lifespan.
        """
        mock_local_service = mocker.MagicMock(spec=LocalService)
        mcp_app._lean_explore_backend_service = mock_local_service

        async with mcp_app.lifespan(mcp_app) as app_context:
            assert isinstance(app_context, AppContext)
            assert app_context.backend_service is mock_local_service
            assert isinstance(app_context.backend_service, LocalService)

        del mcp_app._lean_explore_backend_service  # Clean up the attribute

    async def test_app_lifespan_raises_error_if_backend_not_set(
        self, mocker: "MockerFixture"
    ):
        """Tests that lifespan raises RuntimeError if no backend service is set.

        This ensures the application fails fast if the necessary backend
        dependency is not provided by the server startup script.
        """
        # Ensure _lean_explore_backend_service is not set or is None
        if hasattr(mcp_app, "_lean_explore_backend_service"):
            del mcp_app._lean_explore_backend_service

        # Mock getattr to simulate the backend service not being
        # found on the app instance
        mocker.patch("lean_explore.mcp.app.getattr", return_value=None)

        with pytest.raises(RuntimeError) as exc_info:
            async with mcp_app.lifespan(mcp_app):
                # This line should not be reached as the error occurs on entry
                pass

        assert "Backend service not initialized for MCP app." in str(exc_info.value)
        assert (
            "Ensure the server script correctly sets the backend service attribute"
            in str(exc_info.value)
        )

    async def test_app_lifespan_with_non_none_backend_type(
        self, mocker: "MockerFixture"
    ):
        """Verifies lifespan correctly handles a backend that is not None.

        While the main app logic handles `None`, this test ensures robustness
        if a different, unexpected object is set as the backend.
        """
        mock_unrecognized_backend = (
            MagicMock()
        )  # A generic mock without a specific spec
        mcp_app._lean_explore_backend_service = mock_unrecognized_backend

        async with mcp_app.lifespan(mcp_app) as app_context:
            assert app_context.backend_service is mock_unrecognized_backend
            # Assert that it's not of the expected types, as the original code
            # allows any type for BackendServiceType. This test just confirms
            # it passes the object through if it's not None.
            assert not isinstance(app_context.backend_service, APIClient)
            assert not isinstance(app_context.backend_service, LocalService)

        del mcp_app._lean_explore_backend_service  # Clean up the attribute

    async def test_app_context_dataclass(self):
        """Verifies the structure and mutability of the AppContext dataclass."""
        test_backend_service = MagicMock()
        app_context = AppContext(backend_service=test_backend_service)

        assert app_context.backend_service is test_backend_service
        # Verify it's a dataclass instance
        assert hasattr(app_context, "__dataclass_fields__")

        # Test that the attribute can be reassigned (dataclasses are
        # not frozen by default)
        another_backend = MagicMock()
        app_context.backend_service = another_backend
        assert app_context.backend_service is another_backend
