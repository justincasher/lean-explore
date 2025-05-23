# tests/lean_explore/mcp/test_tools.py

"""Tests for the MCP tools defined in `lean_explore.mcp.tools`.

This module verifies the functionality of the MCP tools (search, get_by_id,
get_dependencies), ensuring they correctly interact with the backend service
(either APIClient or LocalService) via the application context and return
responses in the expected format for MCP consumers. External service calls
are mocked to isolate test execution.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from mcp.server.fastmcp import Context as MCPContext
from lean_explore.mcp.app import AppContext
from lean_explore.mcp.tools import search, get_by_id, get_dependencies, _get_backend_from_context, _prepare_mcp_result_item
from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService
from lean_explore.shared.models.api import (
    APISearchResponse,
    APISearchResultItem,
    APIPrimaryDeclarationInfo,
    APICitationsResponse
)

@pytest.fixture
def backend_service_mock(request):
    """
    Resolve the indirect parametrize values “mock_api_backend” and
    “mock_local_backend” by fetching those fixtures by name.
    """
    return request.getfixturevalue(request.param)

@pytest.fixture
def mock_mcp_context(mocker: "MockerFixture") -> MCPContext:
    """Provides a mock MCPContext for testing tools."""
    mock_app_context = AppContext(backend_service=None) # Will be set by other fixtures
    mock_request_context = mocker.MagicMock()
    mock_request_context.lifespan_context = mock_app_context

    mock_ctx = mocker.MagicMock(spec=MCPContext)
    mock_ctx.request_context = mock_request_context
    return mock_ctx


@pytest.fixture
def mock_api_backend(mocker: "MockerFixture") -> APIClient:
    """Provides a mock APIClient instance as a backend service."""
    mock_client = mocker.MagicMock(spec=APIClient)
    # Set default mock return values for methods that are awaited
    mock_client.search = AsyncMock()
    mock_client.get_by_id = AsyncMock()
    mock_client.get_dependencies = AsyncMock()
    return mock_client


@pytest.fixture
def mock_local_backend(mocker: "MockerFixture") -> LocalService:
    """Provides a mock LocalService instance as a backend service."""
    mock_service = mocker.MagicMock(spec=LocalService)
    # Set default mock return values for methods that are not awaited (synchronous)
    mock_service.search = MagicMock()
    mock_service.get_by_id = MagicMock()
    mock_service.get_dependencies = MagicMock()
    return mock_service


@pytest.fixture
def set_backend_on_context(mock_mcp_context: MCPContext, backend_service_mock: MagicMock):
    """Sets the provided backend service mock onto the MCP context."""
    mock_mcp_context.request_context.lifespan_context.backend_service = backend_service_mock
    return mock_mcp_context


@pytest.mark.asyncio
class TestBackendRetrievalHelper:
    """Test suite for the internal `_get_backend_from_context` helper function."""

    async def test_get_backend_from_context_successful(self, mock_mcp_context: MCPContext, mock_api_backend: APIClient):
        """Verifies that the helper correctly retrieves the backend service."""
        mock_mcp_context.request_context.lifespan_context.backend_service = mock_api_backend
        backend = await _get_backend_from_context(mock_mcp_context)
        assert backend is mock_api_backend

    async def test_get_backend_from_context_raises_error_if_none(self, mock_mcp_context: MCPContext):
        """Verifies that the helper raises an error if the backend service is None."""
        mock_mcp_context.request_context.lifespan_context.backend_service = None
        with pytest.raises(RuntimeError, match="Backend service not configured or available for MCP tool."):
            await _get_backend_from_context(mock_mcp_context)


class TestPrepareMcpResultItemHelper:
    """Test suite for the internal `_prepare_mcp_result_item` helper function."""

    def test_prepare_mcp_result_item_omits_display_statement_text(self):
        """Verifies that `display_statement_text` is set to None in the prepared item."""
        original_item = APISearchResultItem(
            id=1,
            primary_declaration=APIPrimaryDeclarationInfo(lean_name="Test.Declaration"),
            source_file="Test.lean",
            range_start_line=10,
            statement_text="def Test.Declaration := ...",
            display_statement_text="def Test.Declaration := ... (formatted)",
            docstring="A test declaration.",
            informal_description=None
        )

        prepared_item = _prepare_mcp_result_item(original_item)

        assert prepared_item.id == original_item.id
        assert prepared_item.statement_text == original_item.statement_text
        assert prepared_item.display_statement_text is None
        assert prepared_item.docstring == original_item.docstring
        assert prepared_item.primary_declaration.lean_name == original_item.primary_declaration.lean_name


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_service_mock", ["mock_api_backend", "mock_local_backend"], indirect=True)
class TestSearchTool:
    """Test suite for the `search` MCP tool."""

    async def test_search_successful_no_filters_no_limit(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies successful search without package filters or explicit limit."""
        ctx = set_backend_on_context
        query = "function.continuous"
        
        mock_backend_response = APISearchResponse(
            query=query, packages_applied=[],
            results=[
                APISearchResultItem(
                    id=1, primary_declaration=APIPrimaryDeclarationInfo(lean_name="Foo"),
                    statement_text="def Foo := ...", display_statement_text="def Foo := ... disp",
                    source_file="Foo.lean", range_start_line=1
                ),
                APISearchResultItem(
                    id=2, primary_declaration=APIPrimaryDeclarationInfo(lean_name="Bar"),
                    statement_text="def Bar := ...", display_statement_text="def Bar := ... disp",
                    source_file="Bar.lean", range_start_line=1
                )
            ],
            count=2, total_candidates_considered=2, processing_time_ms=50
        )
        backend_service_mock.search.return_value = mock_backend_response

        result = await search(ctx, query=query)

        if asyncio.iscoroutinefunction(backend_service_mock.search):
            backend_service_mock.search.assert_awaited_once_with(query=query, package_filters=None)
        else:
            backend_service_mock.search.assert_called_once_with(query=query, package_filters=None)

        assert result["query"] == query
        assert len(result["results"]) == 2
        assert "display_statement_text" not in result["results"][0]
        assert result["count"] == 2 # Tool's count after limit applied
        assert result["total_candidates_considered"] == 2 # From backend
        assert result["processing_time_ms"] > 0

    async def test_search_successful_with_package_filters_and_limit(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies successful search with package filters and a specific limit."""
        ctx = set_backend_on_context
        query = "nat.prime"
        package_filters = ["Mathlib.Data.Nat", "Mathlib.NumberTheory"]
        limit = 1

        mock_backend_response = APISearchResponse(
            query=query, packages_applied=package_filters,
            results=[
                APISearchResultItem(
                    id=10, primary_declaration=APIPrimaryDeclarationInfo(lean_name="Nat.prime"),
                    statement_text="def Nat.prime := ...", display_statement_text="def Nat.prime := ... disp",
                    source_file="Mathlib/Data/Nat/Prime.lean", range_start_line=100
                ),
                APISearchResultItem(
                    id=11, primary_declaration=APIPrimaryDeclarationInfo(lean_name="prime_factorization"),
                    statement_text="def prime_factorization := ...", display_statement_text="def prime_factorization := ... disp",
                    source_file="Mathlib/NumberTheory/Prime.lean", range_start_line=200
                )
            ],
            count=2, total_candidates_considered=2, processing_time_ms=75
        )
        backend_service_mock.search.return_value = mock_backend_response

        result = await search(ctx, query=query, package_filters=package_filters, limit=limit)

        if asyncio.iscoroutinefunction(backend_service_mock.search):
            backend_service_mock.search.assert_awaited_once_with(query=query, package_filters=package_filters)
        else:
            backend_service_mock.search.assert_called_once_with(query=query, package_filters=package_filters)

        assert result["query"] == query
        assert result["packages_applied"] == package_filters
        assert len(result["results"]) == limit
        assert result["results"][0]["id"] == 10 # Verify limit applied correctly
        assert "display_statement_text" not in result["results"][0]
        assert result["count"] == limit
        assert result["total_candidates_considered"] == 2 # Backend's total
        assert result["processing_time_ms"] > 0

    async def test_search_returns_empty_if_backend_returns_none(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies search returns an empty response if the backend returns None."""
        ctx = set_backend_on_context
        query = "nonexistent"
        backend_service_mock.search.return_value = None

        result = await search(ctx, query=query)

        assert result["query"] == query
        assert result["packages_applied"] == []
        assert len(result["results"]) == 0
        assert result["count"] == 0
        assert result["total_candidates_considered"] == 0
        assert result["processing_time_ms"] == 0

    async def test_search_handles_empty_results_from_backend(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies search handles an empty results list from the backend."""
        ctx = set_backend_on_context
        query = "another_nonexistent"
        mock_backend_response = APISearchResponse(
            query=query, packages_applied=[], results=[],
            count=0, total_candidates_considered=0, processing_time_ms=10
        )
        backend_service_mock.search.return_value = mock_backend_response

        result = await search(ctx, query=query)

        assert result["query"] == query
        assert len(result["results"]) == 0
        assert result["count"] == 0
        assert result["total_candidates_considered"] == 0
        assert result["processing_time_ms"] > 0

    async def test_search_propagates_backend_exceptions(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies search propagates exceptions raised by the backend service."""
        ctx = set_backend_on_context
        query = "error_query"
        backend_service_mock.search.side_effect = ValueError("Backend search failed")

        with pytest.raises(ValueError, match="Backend search failed"):
            await search(ctx, query=query)

@pytest.mark.asyncio
async def test_search_raises_error_if_backend_has_no_search_method_outside_class(mock_mcp_context: MCPContext):
    """Verifies search raises an error if the backend service lacks a 'search' method.
    This test is outside the parametrized class because it specifically tests
    a scenario where the backend doesn't conform to the expected interface.
    """
    class NoSearchBackend:
        pass
    mock_ill_formed_backend = MagicMock(spec_set=NoSearchBackend())
    mock_mcp_context.request_context.lifespan_context.backend_service = mock_ill_formed_backend

    with pytest.raises(RuntimeError, match="Search functionality not available on configured backend."):
        await search(mock_mcp_context, query="any")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_service_mock", ["mock_api_backend", "mock_local_backend"], indirect=True)
class TestGetByIdTool:
    """Test suite for the `get_by_id` MCP tool."""

    async def test_get_by_id_successful(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies successful retrieval of a statement group by ID."""
        ctx = set_backend_on_context
        group_id = 123
        
        mock_backend_item = APISearchResultItem(
            id=group_id, primary_declaration=APIPrimaryDeclarationInfo(lean_name="Lean.Definition"),
            statement_text="def Lean.Definition := ...", display_statement_text="def Lean.Definition := ... disp",
            source_file="Lean.lean", range_start_line=50
        )
        backend_service_mock.get_by_id.return_value = mock_backend_item

        result = await get_by_id(ctx, group_id=group_id)

        if asyncio.iscoroutinefunction(backend_service_mock.get_by_id):
            backend_service_mock.get_by_id.assert_awaited_once_with(group_id=group_id)
        else:
            backend_service_mock.get_by_id.assert_called_once_with(group_id=group_id)

        assert isinstance(result, dict)
        assert result["id"] == group_id
        assert result["primary_declaration"]["lean_name"] == "Lean.Definition"
        assert "display_statement_text" not in result

    async def test_get_by_id_not_found(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies `get_by_id` returns None if the backend returns None."""
        ctx = set_backend_on_context
        group_id = 999
        backend_service_mock.get_by_id.return_value = None

        result = await get_by_id(ctx, group_id=group_id)

        assert result is None
        if asyncio.iscoroutinefunction(backend_service_mock.get_by_id):
            backend_service_mock.get_by_id.assert_awaited_once_with(group_id=group_id)
        else:
            backend_service_mock.get_by_id.assert_called_once_with(group_id=group_id)


    async def test_get_by_id_propagates_backend_exceptions(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies `get_by_id` propagates exceptions raised by the backend service."""
        ctx = set_backend_on_context
        group_id = 500
        backend_service_mock.get_by_id.side_effect = RuntimeError("Backend get_by_id failed")

        with pytest.raises(RuntimeError, match="Backend get_by_id failed"):
            await get_by_id(ctx, group_id=group_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_service_mock", ["mock_api_backend", "mock_local_backend"], indirect=True)
class TestGetDependenciesTool:
    """Test suite for the `get_dependencies` MCP tool."""

    async def test_get_dependencies_successful_with_citations(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies successful retrieval of dependencies with citations."""
        ctx = set_backend_on_context
        group_id = 456
        
        mock_backend_response = APICitationsResponse(
            source_group_id=group_id,
            citations=[
                APISearchResultItem(
                    id=701, primary_declaration=APIPrimaryDeclarationInfo(lean_name="Dep.One"),
                    statement_text="def Dep.One := ...", display_statement_text="def Dep.One := ... disp",
                    source_file="Dep1.lean", range_start_line=10
                ),
                APISearchResultItem(
                    id=702, primary_declaration=APIPrimaryDeclarationInfo(lean_name="Dep.Two"),
                    statement_text="def Dep.Two := ...", display_statement_text="def Dep.Two := ... disp",
                    source_file="Dep2.lean", range_start_line=20
                )
            ],
            count=2
        )
        backend_service_mock.get_dependencies.return_value = mock_backend_response

        result = await get_dependencies(ctx, group_id=group_id)

        if asyncio.iscoroutinefunction(backend_service_mock.get_dependencies):
            backend_service_mock.get_dependencies.assert_awaited_once_with(group_id=group_id)
        else:
            backend_service_mock.get_dependencies.assert_called_once_with(group_id=group_id)

        assert isinstance(result, dict)
        assert result["source_group_id"] == group_id
        assert len(result["citations"]) == 2
        assert result["citations"][0]["id"] == 701
        assert "display_statement_text" not in result["citations"][0]
        assert result["count"] == 2

    async def test_get_dependencies_successful_no_citations(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies successful retrieval of dependencies when no citations are found."""
        ctx = set_backend_on_context
        group_id = 457
        
        mock_backend_response = APICitationsResponse(
            source_group_id=group_id,
            citations=[],
            count=0
        )
        backend_service_mock.get_dependencies.return_value = mock_backend_response

        result = await get_dependencies(ctx, group_id=group_id)

        assert isinstance(result, dict)
        assert result["source_group_id"] == group_id
        assert len(result["citations"]) == 0
        assert result["count"] == 0

    async def test_get_dependencies_not_found(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies `get_dependencies` returns None if the backend returns None (source group not found)."""
        ctx = set_backend_on_context
        group_id = 998
        backend_service_mock.get_dependencies.return_value = None

        result = await get_dependencies(ctx, group_id=group_id)

        assert result is None

    async def test_get_dependencies_propagates_backend_exceptions(self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock):
        """Verifies `get_dependencies` propagates exceptions raised by the backend service."""
        ctx = set_backend_on_context
        group_id = 501
        backend_service_mock.get_dependencies.side_effect = Exception("Backend dependencies failed")

        with pytest.raises(Exception, match="Backend dependencies failed"):
            await get_dependencies(ctx, group_id=group_id)