# tests/lean_explore/api/test_client.py

"""Tests for the API client in `lean_explore.api.client`.

This module verifies the functionality of the `Client` class,
ensuring it correctly initializes, constructs API requests,
handles responses (both successful and erroneous), and parses
data into the appropriate Pydantic models. External HTTP calls
are mocked to isolate test execution.
"""

from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from lean_explore.api.client import _DEFAULT_API_BASE_URL, Client
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APISearchResponse,
    APISearchResultItem,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.asyncio
class TestAPIClient:
    """Test suite for the lean_explore.api.client.Client class."""

    def test_client_initialization(self):
        """Verifies correct initialization of Client attributes.

        Checks that api_key, timeout (default and custom), base_url,
        and HTTP headers are set as expected upon client instantiation.
        """
        api_key = "test_api_key_123"
        custom_timeout = 20.0

        client_default_timeout = Client(api_key=api_key)
        assert client_default_timeout.api_key == api_key
        assert client_default_timeout.timeout == 10.0  # Default timeout
        assert client_default_timeout.base_url == _DEFAULT_API_BASE_URL
        assert client_default_timeout._headers == {"Authorization": f"Bearer {api_key}"}

        client_custom_timeout = Client(api_key=api_key, timeout=custom_timeout)
        assert client_custom_timeout.timeout == custom_timeout

    async def test_search_successful_no_filters(self, mocker: "MockerFixture"):
        """Tests successful search without package filters.

        Verifies correct API endpoint, parameters, headers, and response parsing.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        api_key = "test_key"
        query = "Nat.add"
        client = Client(api_key=api_key)

        mock_response_json: Dict[str, Any] = {
            "query": query,
            "packages_applied": None,
            "results": [
                {
                    "id": 1,
                    "primary_declaration": {"lean_name": "Nat.add"},
                    "source_file": "Nat.lean",
                    "range_start_line": 10,
                    "statement_text": "def Nat.add ...",
                    "display_statement_text": "def Nat.add ...",
                    "docstring": "Adds two natural numbers.",
                    "informal_description": "Function for addition.",
                }
            ],
            "count": 1,
            "total_candidates_considered": 1,
            "processing_time_ms": 100,
        }

        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_json
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)

        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        response_model = await client.search(query=query)

        mock_async_client_constructor.assert_called_once_with(timeout=client.timeout)
        mock_async_client_instance.get.assert_awaited_once_with(
            f"{client.base_url}/search", params={"q": query}, headers=client._headers
        )
        mock_http_response.raise_for_status.assert_called_once()
        assert isinstance(response_model, APISearchResponse)
        assert response_model.query == query
        assert len(response_model.results) == 1
        assert response_model.results[0].id == 1
        assert response_model.results[0].primary_declaration.lean_name == "Nat.add"

    async def test_search_successful_with_package_filters(
        self, mocker: "MockerFixture"
    ):
        """Tests successful search with package filters applied.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        api_key = "test_key"
        query = "is_prime"
        package_filters = ["Mathlib.NumberTheory", "Mathlib.Data.Nat"]
        client = Client(api_key=api_key)

        mock_response_json: Dict[str, Any] = {
            "query": query,
            "packages_applied": package_filters,
            "results": [],
            "count": 0,
            "total_candidates_considered": 0,
            "processing_time_ms": 50,
        }
        mock_http_response = MagicMock(spec=httpx.Response, status_code=200)
        mock_http_response.json.return_value = mock_response_json
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        await client.search(query=query, package_filters=package_filters)

        mock_async_client_instance.get.assert_awaited_once_with(
            f"{client.base_url}/search",
            params={"q": query, "pkg": package_filters},
            headers=client._headers,
        )

    async def test_search_http_error(self, mocker: "MockerFixture"):
        """Tests handling of HTTP errors during search.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=500)
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_http_response
        )
        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.search(query="any")

    async def test_search_request_error(self, mocker: "MockerFixture"):
        """Tests handling of httpx.RequestError during search.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(
            side_effect=httpx.RequestError("Connection failed", request=MagicMock())
        )
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        with pytest.raises(httpx.RequestError, match="Connection failed"):
            await client.search(query="any")

    async def test_get_by_id_successful(self, mocker: "MockerFixture"):
        """Tests successful retrieval of a statement group by its ID.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        api_key = "test_key"
        group_id = 123
        client = Client(api_key=api_key)

        mock_response_json: Dict[str, Any] = {
            "id": group_id,
            "primary_declaration": {"lean_name": "Specific.Item"},
            "source_file": "Specific.lean",
            "range_start_line": 1,
            "statement_text": "def Specific.Item ...",
            "display_statement_text": "def Specific.Item ...",
            "docstring": "A specific item.",
            "informal_description": None,
        }
        mock_http_response = MagicMock(spec=httpx.Response, status_code=200)
        mock_http_response.json.return_value = mock_response_json
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        result = await client.get_by_id(group_id=group_id)

        mock_async_client_instance.get.assert_awaited_once_with(
            f"{client.base_url}/statement_groups/{group_id}", headers=client._headers
        )
        assert isinstance(result, APISearchResultItem)
        assert result.id == group_id
        assert result.primary_declaration.lean_name == "Specific.Item"

    async def test_get_by_id_not_found_404(self, mocker: "MockerFixture"):
        """Tests get_by_id returning None for a 404 response.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=404)
        mock_http_response.raise_for_status = MagicMock()  # Should not be called

        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        result = await client.get_by_id(group_id=404)
        assert result is None
        mock_http_response.raise_for_status.assert_not_called()

    async def test_get_by_id_http_error_other_than_404(self, mocker: "MockerFixture"):
        """Tests get_by_id raising HTTPStatusError for non-404 errors.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=401)
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_http_response
        )
        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_by_id(group_id=123)
        mock_http_response.raise_for_status.assert_called_once()

    async def test_get_dependencies_successful(self, mocker: "MockerFixture"):
        """Tests successful retrieval of dependencies for a statement group.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        api_key = "test_key"
        group_id = 456
        client = Client(api_key=api_key)

        mock_response_json: Dict[str, Any] = {
            "source_group_id": group_id,
            "citations": [
                {
                    "id": 789,
                    "primary_declaration": {"lean_name": "Dependency.One"},
                    "source_file": "Dep.lean",
                    "range_start_line": 1,
                    "statement_text": "def Dependency.One ...",
                    "display_statement_text": None,
                    "docstring": None,
                    "informal_description": None,
                }
            ],
            "count": 1,
        }
        mock_http_response = MagicMock(spec=httpx.Response, status_code=200)
        mock_http_response.json.return_value = mock_response_json
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        result = await client.get_dependencies(group_id=group_id)

        mock_async_client_instance.get.assert_awaited_once_with(
            f"{client.base_url}/statement_groups/{group_id}/dependencies",
            headers=client._headers,
        )
        assert isinstance(result, APICitationsResponse)
        assert result.source_group_id == group_id
        assert len(result.citations) == 1
        assert result.citations[0].id == 789

    async def test_get_dependencies_not_found_404(self, mocker: "MockerFixture"):
        """Tests get_dependencies returning None for a 404 response.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=404)
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        result = await client.get_dependencies(group_id=404)
        assert result is None
        mock_http_response.raise_for_status.assert_not_called()

    async def test_get_dependencies_http_error_other_than_404(
        self, mocker: "MockerFixture"
    ):
        """Tests get_dependencies raising HTTPStatusError for non-404 errors.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=503)
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_http_response
        )
        mock_async_client_instance = MagicMock()
        mock_async_client_instance.get = AsyncMock(return_value=mock_http_response)
        mock_async_client_constructor = mocker.patch("httpx.AsyncClient")
        mock_async_client_constructor.return_value.__aenter__.return_value = (
            mock_async_client_instance
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_dependencies(group_id=123)
        mock_http_response.raise_for_status.assert_called_once()
