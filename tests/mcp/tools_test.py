"""Tests for the MCP tools module.

These tests verify the MCP tool definitions for search, search_summary,
and per-field retrieval operations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_explore.mcp.tools import (
    _get_backend_from_context,
    get_dependencies,
    get_description,
    get_docstring,
    get_module,
    get_source_code,
    get_source_link,
    search,
    search_summary,
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
    """Tests for the search MCP tool (full results)."""

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

    async def test_search_returns_full_dict(self, mock_context_with_backend):
        """Test that search returns all fields."""
        mock_ctx, _ = mock_context_with_backend

        result = await search(mock_ctx, query="test query", limit=10)

        assert isinstance(result, dict)
        assert result["query"] == "test query"
        assert result["count"] == 1

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


class TestSearchSummaryTool:
    """Tests for the search_summary MCP tool (slim results)."""

    @pytest.fixture
    def mock_context_with_backend(self):
        """Create a mock MCP context with a backend service."""
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=_make_search_response())

        mock_ctx = _make_mock_context(mock_backend)
        return mock_ctx, mock_backend

    async def test_search_summary_calls_backend(self, mock_context_with_backend):
        """Test that search_summary calls the backend search method."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search_summary(mock_ctx, query="test query", limit=10)

        mock_backend.search.assert_called_once_with(
            query="test query", limit=10, rerank_top=50, packages=None
        )

    async def test_search_summary_returns_slim_dict(self, mock_context_with_backend):
        """Test that search_summary returns only id, name, description."""
        mock_ctx, _ = mock_context_with_backend

        result = await search_summary(mock_ctx, query="test query", limit=10)

        assert isinstance(result, dict)
        assert result["query"] == "test query"
        assert result["count"] == 1

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

    async def test_search_summary_extracts_bold_description(self):
        """Test that search_summary extracts the bold header."""
        response = _make_search_response(
            informalization="**Continuous Map Between Topological Spaces.** "
            "A function that preserves the topology."
        )
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search_summary(mock_ctx, query="continuous")

        description = result["results"][0]["description"]
        assert description == "Continuous Map Between Topological Spaces."

    async def test_search_summary_handles_no_informalization(self):
        """Test that search_summary handles results without informalization."""
        response = _make_search_response(informalization=None)
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search_summary(mock_ctx, query="test")

        assert "description" not in result["results"][0]

    async def test_search_summary_default_limit(self, mock_context_with_backend):
        """Test search_summary with default parameters."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search_summary(mock_ctx, query="test")

        mock_backend.search.assert_called_once_with(
            query="test", limit=10, rerank_top=50, packages=None
        )

    async def test_search_summary_with_packages_filter(self, mock_context_with_backend):
        """Test search_summary with packages filter."""
        mock_ctx, mock_backend = mock_context_with_backend

        await search_summary(mock_ctx, query="test", packages=["Mathlib", "Std"])

        mock_backend.search.assert_called_once_with(
            query="test", limit=10, rerank_top=50, packages=["Mathlib", "Std"]
        )

    async def test_search_summary_backend_without_method(self):
        """Test error when backend lacks search method."""
        mock_backend = MagicMock(spec=[])  # No methods

        mock_ctx = _make_mock_context(mock_backend)

        with pytest.raises(RuntimeError, match="Search functionality not available"):
            await search_summary(mock_ctx, query="test")

    async def test_search_summary_with_sync_backend(self):
        """Test search_summary with a synchronous backend."""
        mock_response = SearchResponse(
            query="test",
            results=[],
            count=0,
            processing_time_ms=5,
        )

        mock_backend = MagicMock()
        mock_backend.search = MagicMock(return_value=mock_response)
        mock_ctx = _make_mock_context(mock_backend)

        result = await search_summary(mock_ctx, query="test")

        mock_backend.search.assert_called_once()
        assert result["count"] == 0


_MOCK_DECLARATION = SearchResult(
    id=42,
    name="Test.declaration",
    module="Test.Module",
    docstring="A test declaration",
    source_text="def test := 42",
    source_link="https://example.com/source",
    dependencies='["Dep.one", "Dep.two"]',
    informalization="**Test Declaration.** A test description.",
)


def _make_get_by_id_context(
    result: SearchResult | None = _MOCK_DECLARATION,
    use_async: bool = True,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock MCP context with a backend that supports get_by_id.

    Args:
        result: The SearchResult to return, or None for not-found cases.
        use_async: If True, use AsyncMock; if False, use MagicMock.

    Returns:
        A tuple of (mock_context, mock_backend).
    """
    mock_backend = MagicMock()
    if use_async:
        mock_backend.get_by_id = AsyncMock(return_value=result)
    else:
        mock_backend.get_by_id = MagicMock(return_value=result)

    mock_ctx = _make_mock_context(mock_backend)
    return mock_ctx, mock_backend


class TestGetSourceCodeTool:
    """Tests for the get_source_code MCP tool."""

    async def test_calls_backend(self):
        """Test that get_source_code calls the backend get_by_id method."""
        mock_ctx, mock_backend = _make_get_by_id_context()
        await get_source_code(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_returns_correct_fields(self):
        """Test that get_source_code returns id, name, and source_text."""
        mock_ctx, _ = _make_get_by_id_context()
        result = await get_source_code(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"
        assert result["source_text"] == "def test := 42"
        assert set(result.keys()) == {"id", "name", "source_text"}

    async def test_not_found(self):
        """Test get_source_code returns None when declaration not found."""
        mock_ctx, _ = _make_get_by_id_context(result=None)
        result = await get_source_code(mock_ctx, declaration_id=99999)
        assert result is None

    async def test_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_ctx = _make_mock_context(MagicMock(spec=[]))
        with pytest.raises(RuntimeError, match="Get by ID functionality"):
            await get_source_code(mock_ctx, declaration_id=1)

    async def test_with_sync_backend(self):
        """Test get_source_code with a synchronous backend."""
        mock_ctx, mock_backend = _make_get_by_id_context(use_async=False)
        result = await get_source_code(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once()
        assert result["source_text"] == "def test := 42"


class TestGetSourceLinkTool:
    """Tests for the get_source_link MCP tool."""

    async def test_calls_backend(self):
        """Test that get_source_link calls the backend get_by_id method."""
        mock_ctx, mock_backend = _make_get_by_id_context()
        await get_source_link(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_returns_correct_fields(self):
        """Test that get_source_link returns id, name, and source_link."""
        mock_ctx, _ = _make_get_by_id_context()
        result = await get_source_link(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"
        assert result["source_link"] == "https://example.com/source"
        assert set(result.keys()) == {"id", "name", "source_link"}

    async def test_not_found(self):
        """Test get_source_link returns None when declaration not found."""
        mock_ctx, _ = _make_get_by_id_context(result=None)
        result = await get_source_link(mock_ctx, declaration_id=99999)
        assert result is None

    async def test_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_ctx = _make_mock_context(MagicMock(spec=[]))
        with pytest.raises(RuntimeError, match="Get by ID functionality"):
            await get_source_link(mock_ctx, declaration_id=1)

    async def test_with_sync_backend(self):
        """Test get_source_link with a synchronous backend."""
        mock_ctx, mock_backend = _make_get_by_id_context(use_async=False)
        result = await get_source_link(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once()
        assert result["source_link"] == "https://example.com/source"


class TestGetDocstringTool:
    """Tests for the get_docstring MCP tool."""

    async def test_calls_backend(self):
        """Test that get_docstring calls the backend get_by_id method."""
        mock_ctx, mock_backend = _make_get_by_id_context()
        await get_docstring(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_returns_correct_fields(self):
        """Test that get_docstring returns id, name, and docstring."""
        mock_ctx, _ = _make_get_by_id_context()
        result = await get_docstring(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"
        assert result["docstring"] == "A test declaration"
        assert set(result.keys()) == {"id", "name", "docstring"}

    async def test_not_found(self):
        """Test get_docstring returns None when declaration not found."""
        mock_ctx, _ = _make_get_by_id_context(result=None)
        result = await get_docstring(mock_ctx, declaration_id=99999)
        assert result is None

    async def test_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_ctx = _make_mock_context(MagicMock(spec=[]))
        with pytest.raises(RuntimeError, match="Get by ID functionality"):
            await get_docstring(mock_ctx, declaration_id=1)

    async def test_with_sync_backend(self):
        """Test get_docstring with a synchronous backend."""
        mock_ctx, mock_backend = _make_get_by_id_context(use_async=False)
        result = await get_docstring(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once()
        assert result["docstring"] == "A test declaration"


class TestGetDescriptionTool:
    """Tests for the get_description MCP tool."""

    async def test_calls_backend(self):
        """Test that get_description calls the backend get_by_id method."""
        mock_ctx, mock_backend = _make_get_by_id_context()
        await get_description(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_returns_correct_fields(self):
        """Test that get_description returns id, name, and informalization."""
        mock_ctx, _ = _make_get_by_id_context()
        result = await get_description(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"
        assert result["informalization"] == (
            "**Test Declaration.** A test description."
        )
        assert set(result.keys()) == {"id", "name", "informalization"}

    async def test_not_found(self):
        """Test get_description returns None when declaration not found."""
        mock_ctx, _ = _make_get_by_id_context(result=None)
        result = await get_description(mock_ctx, declaration_id=99999)
        assert result is None

    async def test_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_ctx = _make_mock_context(MagicMock(spec=[]))
        with pytest.raises(RuntimeError, match="Get by ID functionality"):
            await get_description(mock_ctx, declaration_id=1)

    async def test_with_sync_backend(self):
        """Test get_description with a synchronous backend."""
        mock_ctx, mock_backend = _make_get_by_id_context(use_async=False)
        result = await get_description(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once()
        assert result["informalization"] == (
            "**Test Declaration.** A test description."
        )


class TestGetModuleTool:
    """Tests for the get_module MCP tool."""

    async def test_calls_backend(self):
        """Test that get_module calls the backend get_by_id method."""
        mock_ctx, mock_backend = _make_get_by_id_context()
        await get_module(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_returns_correct_fields(self):
        """Test that get_module returns id, name, and module."""
        mock_ctx, _ = _make_get_by_id_context()
        result = await get_module(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"
        assert result["module"] == "Test.Module"
        assert set(result.keys()) == {"id", "name", "module"}

    async def test_not_found(self):
        """Test get_module returns None when declaration not found."""
        mock_ctx, _ = _make_get_by_id_context(result=None)
        result = await get_module(mock_ctx, declaration_id=99999)
        assert result is None

    async def test_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_ctx = _make_mock_context(MagicMock(spec=[]))
        with pytest.raises(RuntimeError, match="Get by ID functionality"):
            await get_module(mock_ctx, declaration_id=1)

    async def test_with_sync_backend(self):
        """Test get_module with a synchronous backend."""
        mock_ctx, mock_backend = _make_get_by_id_context(use_async=False)
        result = await get_module(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once()
        assert result["module"] == "Test.Module"


class TestGetDependenciesTool:
    """Tests for the get_dependencies MCP tool."""

    async def test_calls_backend(self):
        """Test that get_dependencies calls the backend get_by_id method."""
        mock_ctx, mock_backend = _make_get_by_id_context()
        await get_dependencies(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once_with(declaration_id=42)

    async def test_returns_correct_fields(self):
        """Test that get_dependencies returns id, name, and dependencies."""
        mock_ctx, _ = _make_get_by_id_context()
        result = await get_dependencies(mock_ctx, declaration_id=42)

        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["name"] == "Test.declaration"
        assert result["dependencies"] == '["Dep.one", "Dep.two"]'
        assert set(result.keys()) == {"id", "name", "dependencies"}

    async def test_not_found(self):
        """Test get_dependencies returns None when declaration not found."""
        mock_ctx, _ = _make_get_by_id_context(result=None)
        result = await get_dependencies(mock_ctx, declaration_id=99999)
        assert result is None

    async def test_backend_without_method(self):
        """Test error when backend lacks get_by_id method."""
        mock_ctx = _make_mock_context(MagicMock(spec=[]))
        with pytest.raises(RuntimeError, match="Get by ID functionality"):
            await get_dependencies(mock_ctx, declaration_id=1)

    async def test_with_sync_backend(self):
        """Test get_dependencies with a synchronous backend."""
        mock_ctx, mock_backend = _make_get_by_id_context(use_async=False)
        result = await get_dependencies(mock_ctx, declaration_id=42)
        mock_backend.get_by_id.assert_called_once()
        assert result["dependencies"] == '["Dep.one", "Dep.two"]'
