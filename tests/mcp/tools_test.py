"""Tests for the MCP tools module.

These tests verify the MCP tool definitions for search and get_by_id operations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_explore.mcp.tools import _get_backend_from_context, get_by_id, search
from lean_explore.models import SearchResponse, SearchResult


class TestGetBackendFromContext:
    """Tests for the _get_backend_from_context helper function."""

    async def test_get_backend_success(self):
        """Test successful backend retrieval from context."""
        mock_backend = MagicMock()

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        result = await _get_backend_from_context(mock_ctx)
        assert result is mock_backend

    async def test_get_backend_not_available(self):
        """Test error when backend is not available."""
        mock_app_context = MagicMock()
        mock_app_context.backend_service = None

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        with pytest.raises(RuntimeError, match="Backend service not configured"):
            await _get_backend_from_context(mock_ctx)


class TestSearchTool:
    """Tests for the search MCP tool."""

    @pytest.fixture
    def mock_context_with_backend(self):
        """Create a mock MCP context with a backend service."""
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(
            return_value=SearchResponse(
                query="test query",
                results=[
                    SearchResult(
                        id=1,
                        name="Test.result",
                        module="Test.Module",
                        docstring="A test result",
                        source_text="def test := 1",
                        source_link="https://example.com",
                        dependencies=None,
                        informalization="Test informalization",
                    )
                ],
                count=1,
                processing_time_ms=42,
            )
        )

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        return mock_ctx, mock_backend

    async def test_search_calls_backend(self, mock_context_with_backend):
        """Test that search tool calls the backend search method."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search(mock_ctx, query="test query", limit=10)

        mock_backend.search.assert_called_once_with(
            query="test query", limit=10, rerank_top=50, packages=None
        )

    async def test_search_returns_dict(self, mock_context_with_backend):
        """Test that search returns a dictionary response."""
        mock_ctx, _ = mock_context_with_backend

        result = await search(mock_ctx, query="test query", limit=10)

        assert isinstance(result, dict)
        assert "results" in result
        assert "query" in result
        assert result["query"] == "test query"

    async def test_search_default_limit(self, mock_context_with_backend):
        """Test search with default limit and rerank_top parameters."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search(mock_ctx, query="test")

        mock_backend.search.assert_called_once_with(
            query="test", limit=10, rerank_top=50, packages=None
        )

    async def test_search_with_packages_filter(self, mock_context_with_backend):
        """Test search with packages filter."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search(mock_ctx, query="test", packages=["Mathlib", "Std"])

        mock_backend.search.assert_called_once_with(
            query="test", limit=10, rerank_top=50, packages=["Mathlib", "Std"]
        )

    async def test_search_backend_without_method(self):
        """Test error when backend lacks search method."""
        mock_backend = MagicMock(spec=[])  # No methods

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        with pytest.raises(RuntimeError, match="Search functionality not available"):
            await search(mock_ctx, query="test")

    async def test_search_with_sync_backend(self):
        """Test search with a synchronous backend."""
        mock_response = SearchResponse(
            query="test",
            results=[],
            count=0,
            processing_time_ms=5,
        )

        mock_backend = MagicMock()
        # Make search a regular function, not async
        mock_backend.search = MagicMock(return_value=mock_response)

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        result = await search(mock_ctx, query="test")

        mock_backend.search.assert_called_once()
        assert result["count"] == 0


class TestGetByIdTool:
    """Tests for the get_by_id MCP tool."""

    @pytest.fixture
    def mock_context_with_backend(self):
        """Create a mock MCP context with a backend service."""
        mock_result = SearchResult(
            id=42,
            name="Test.declaration",
            module="Test.Module",
            docstring="A test declaration",
            source_text="def test := 42",
            source_link="https://example.com",
            dependencies=None,
            informalization="Test informalization",
        )

        mock_backend = MagicMock()
        mock_backend.get_by_id = AsyncMock(return_value=mock_result)

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        return mock_ctx, mock_backend, mock_result

    async def test_get_by_id_calls_backend(self, mock_context_with_backend):
        """Test that get_by_id tool calls the backend method."""
        mock_ctx, mock_backend, _ = mock_context_with_backend

        await get_by_id(mock_ctx, declaration_id=42)

        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_get_by_id_returns_dict(self, mock_context_with_backend):
        """Test that get_by_id returns a dictionary when found."""
        mock_ctx, _, _ = mock_context_with_backend

        result = await get_by_id(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"

    async def test_get_by_id_not_found(self):
        """Test get_by_id returns None when declaration not found."""
        mock_backend = MagicMock()
        mock_backend.get_by_id = AsyncMock(return_value=None)

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        result = await get_by_id(mock_ctx, declaration_id=99999)

        assert result is None

    async def test_get_by_id_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_backend = MagicMock(spec=[])  # No methods

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        with pytest.raises(RuntimeError, match="Get by ID functionality not available"):
            await get_by_id(mock_ctx, declaration_id=1)

    async def test_get_by_id_with_sync_backend(self):
        """Test get_by_id with a synchronous backend."""
        mock_result = SearchResult(
            id=1,
            name="Sync.result",
            module="Test",
            docstring=None,
            source_text="def sync := 1",
            source_link="https://example.com",
            dependencies=None,
            informalization=None,
        )

        mock_backend = MagicMock()
        # Make get_by_id a regular function, not async
        mock_backend.get_by_id = MagicMock(return_value=mock_result)

        mock_app_context = MagicMock()
        mock_app_context.backend_service = mock_backend

        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = mock_app_context

        result = await get_by_id(mock_ctx, declaration_id=1)

        mock_backend.get_by_id.assert_called_once()
        assert result["id"] == 1
