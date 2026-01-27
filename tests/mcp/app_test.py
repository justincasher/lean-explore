"""Tests for the MCP app module.

These tests verify the FastMCP application setup and lifespan context.
"""

from unittest.mock import MagicMock

import pytest

from lean_explore.mcp.app import AppContext, app_lifespan, mcp_app


class TestAppContext:
    """Tests for the AppContext dataclass."""

    def test_app_context_creation(self):
        """Test creating an AppContext with a backend service."""
        mock_backend = MagicMock()
        context = AppContext(backend_service=mock_backend)
        assert context.backend_service is mock_backend

    def test_app_context_none_backend(self):
        """Test AppContext can be created with None backend."""
        context = AppContext(backend_service=None)
        assert context.backend_service is None


class TestBackendServiceType:
    """Tests for the BackendServiceType type alias."""

    def test_type_allows_api_client(self):
        """Test that type annotation accepts ApiClient."""
        from lean_explore.api import ApiClient

        # This is a type check - just verify the import works
        assert ApiClient is not None

    def test_type_allows_service(self):
        """Test that type annotation accepts Service."""
        from lean_explore.search import Service

        # This is a type check - just verify the import works
        assert Service is not None


class TestAppLifespan:
    """Tests for the app_lifespan context manager."""

    async def test_lifespan_yields_context(self):
        """Test that lifespan yields an AppContext."""
        mock_backend = MagicMock()
        mock_server = MagicMock()
        mock_server._lean_explore_backend_service = mock_backend

        async with app_lifespan(mock_server) as context:
            assert isinstance(context, AppContext)
            assert context.backend_service is mock_backend

    async def test_lifespan_raises_without_backend(self):
        """Test that lifespan raises RuntimeError if backend not set."""
        mock_server = MagicMock(spec=[])  # No attributes

        with pytest.raises(RuntimeError, match="Backend service not initialized"):
            async with app_lifespan(mock_server):
                pass

    async def test_lifespan_raises_with_none_backend(self):
        """Test that lifespan raises RuntimeError if backend is None."""
        mock_server = MagicMock()
        mock_server._lean_explore_backend_service = None

        with pytest.raises(RuntimeError, match="Backend service not initialized"):
            async with app_lifespan(mock_server):
                pass


class TestMcpApp:
    """Tests for the mcp_app FastMCP instance."""

    def test_mcp_app_exists(self):
        """Test that mcp_app is created."""
        assert mcp_app is not None

    def test_mcp_app_has_name(self):
        """Test that mcp_app has name configured."""
        assert mcp_app.name == "LeanExploreMCPServer"

    def test_mcp_app_name(self):
        """Test mcp_app has correct name."""
        assert "LeanExplore" in mcp_app.name or "lean" in mcp_app.name.lower()
