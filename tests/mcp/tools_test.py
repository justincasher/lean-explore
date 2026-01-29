"""Tests for the MCP tools module.

These tests verify the MCP tool definitions for search, search_verbose,
and get_by_id operations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_explore.mcp.tools import (
    _get_backend_from_context,
    get_by_id,
    search,
    search_verbose,
)
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


def _make_search_response(
    query: str = "test query",
    informalization: str | None = "**Test Title.** A test informalization.",
) -> SearchResponse:
    """Create a SearchResponse with a single result for testing.

    Args:
        query: The query string for the response.
        informalization: The informalization text for the result.

    Returns:
        A SearchResponse with one result.
    """
    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                id=1,
                name="Test.result",
                module="Test.Module",
                docstring="A test result",
                source_text="def test := 1",
                source_link="https://example.com",
                dependencies=None,
                informalization=informalization,
            )
        ],
        count=1,
        processing_time_ms=42,
    )


def _make_mock_context(backend: MagicMock) -> MagicMock:
    """Create a mock MCP context wrapping a backend service.

    Args:
        backend: The mock backend service.

    Returns:
        A mock MCP context with the backend attached.
    """
    mock_app_context = MagicMock()
    mock_app_context.backend_service = backend

    mock_ctx = MagicMock()
    mock_ctx.request_context.lifespan_context = mock_app_context
    return mock_ctx


class TestSearchTool:
    """Tests for the search MCP tool (slim results)."""

    @pytest.fixture
    def mock_context_with_backend(self):
        """Create a mock MCP context with a backend service."""
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=_make_search_response())

        mock_ctx = _make_mock_context(mock_backend)
        return mock_ctx, mock_backend

    async def test_search_calls_backend(self, mock_context_with_backend):
        """Test that search tool calls the backend search method."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search(mock_ctx, query="test query", limit=10)

        mock_backend.search.assert_called_once_with(
            query="test query", limit=10, rerank_top=50, packages=None
        )

    async def test_search_returns_slim_dict(self, mock_context_with_backend):
        """Test that search returns a slim dict with only id, name, description."""
        mock_ctx, _ = mock_context_with_backend

        result = await search(mock_ctx, query="test query", limit=10)

        assert isinstance(result, dict)
        assert result["query"] == "test query"
        assert result["count"] == 1

        # Verify slim format: only id, name, description
        search_result = result["results"][0]
        assert search_result["id"] == 1
        assert search_result["name"] == "Test.result"
        assert search_result["description"] == "Test Title."

        # Verify full fields are NOT present
        assert "module" not in search_result
        assert "source_text" not in search_result
        assert "source_link" not in search_result
        assert "docstring" not in search_result
        assert "dependencies" not in search_result
        assert "informalization" not in search_result

    async def test_search_extracts_bold_description(self):
        """Test that search extracts the bold header from informalization."""
        response = _make_search_response(
            informalization="**Continuous Map Between Topological Spaces.** "
            "A function that preserves the topology."
        )
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search(mock_ctx, query="continuous")

        description = result["results"][0]["description"]
        assert description == "Continuous Map Between Topological Spaces."

    async def test_search_handles_no_informalization(self):
        """Test that search handles results without informalization."""
        response = _make_search_response(informalization=None)
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search(mock_ctx, query="test")

        # description should be excluded from output (exclude_none=True)
        assert "description" not in result["results"][0]

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

        mock_ctx = _make_mock_context(mock_backend)

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
        mock_backend.search = MagicMock(return_value=mock_response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search(mock_ctx, query="test")

        mock_backend.search.assert_called_once()
        assert result["count"] == 0


class TestSearchVerboseTool:
    """Tests for the search_verbose MCP tool (full results)."""

    @pytest.fixture
    def mock_context_with_backend(self):
        """Create a mock MCP context with a backend service."""
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=_make_search_response())

        mock_ctx = _make_mock_context(mock_backend)
        return mock_ctx, mock_backend

    async def test_search_verbose_calls_backend(self, mock_context_with_backend):
        """Test that search_verbose calls the backend search method."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search_verbose(mock_ctx, query="test query", limit=10)

        mock_backend.search.assert_called_once_with(
            query="test query", limit=10, rerank_top=50, packages=None
        )

    async def test_search_verbose_returns_full_dict(self, mock_context_with_backend):
        """Test that search_verbose returns all fields."""
        mock_ctx, _ = mock_context_with_backend

        result = await search_verbose(mock_ctx, query="test query", limit=10)

        assert isinstance(result, dict)
        assert result["query"] == "test query"
        assert result["count"] == 1

        # Verify full fields are present
        search_result = result["results"][0]
        assert search_result["id"] == 1
        assert search_result["name"] == "Test.result"
        assert search_result["module"] == "Test.Module"
        assert search_result["source_text"] == "def test := 1"
        assert search_result["source_link"] == "https://example.com"
        assert search_result["docstring"] == "A test result"
        assert (
            search_result["informalization"]
            == "**Test Title.** A test informalization."
        )

    async def test_search_verbose_default_limit(self, mock_context_with_backend):
        """Test search_verbose with default limit and rerank_top parameters."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search_verbose(mock_ctx, query="test")

        mock_backend.search.assert_called_once_with(
            query="test", limit=10, rerank_top=50, packages=None
        )

    async def test_search_verbose_with_packages_filter(self, mock_context_with_backend):
        """Test search_verbose with packages filter."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search_verbose(mock_ctx, query="test", packages=["Mathlib", "Std"])

        mock_backend.search.assert_called_once_with(
            query="test", limit=10, rerank_top=50, packages=["Mathlib", "Std"]
        )

    async def test_search_verbose_backend_without_method(self):
        """Test error when backend lacks search method."""
        mock_backend = MagicMock(spec=[])  # No methods

        mock_ctx = _make_mock_context(mock_backend)

        with pytest.raises(RuntimeError, match="Search functionality not available"):
            await search_verbose(mock_ctx, query="test")

    async def test_search_verbose_with_sync_backend(self):
        """Test search_verbose with a synchronous backend."""
        mock_response = SearchResponse(
            query="test",
            results=[],
            count=0,
            processing_time_ms=5,
        )

        mock_backend = MagicMock()
        mock_backend.search = MagicMock(return_value=mock_response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search_verbose(mock_ctx, query="test")

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

        mock_ctx = _make_mock_context(mock_backend)

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

        mock_ctx = _make_mock_context(mock_backend)

        result = await get_by_id(mock_ctx, declaration_id=99999)

        assert result is None

    async def test_get_by_id_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_backend = MagicMock(spec=[])  # No methods

        mock_ctx = _make_mock_context(mock_backend)

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
        mock_backend.get_by_id = MagicMock(return_value=mock_result)
        mock_ctx = _make_mock_context(mock_backend)

        result = await get_by_id(mock_ctx, declaration_id=1)

        mock_backend.get_by_id.assert_called_once()
        assert result["id"] == 1
