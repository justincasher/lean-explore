"""Tests for the API client module.

These tests verify the ApiClient class for interacting with the remote
Lean Explore API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lean_explore.api.client import ApiClient
from lean_explore.models import SearchResponse, SearchResult


class TestApiClientInit:
    """Tests for ApiClient initialization."""

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key passed as parameter."""
        client = ApiClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        assert client._headers["Authorization"] == "Bearer test-key-123"

    def test_init_with_env_variable(self):
        """Test initialization with API key from environment variable."""
        with patch.dict("os.environ", {"LEANEXPLORE_API_KEY": "env-key-456"}):
            client = ApiClient()
            assert client.api_key == "env-key-456"

    def test_init_parameter_overrides_env(self):
        """Test that parameter API key takes precedence over env variable."""
        with patch.dict("os.environ", {"LEANEXPLORE_API_KEY": "env-key"}):
            client = ApiClient(api_key="param-key")
            assert client.api_key == "param-key"

    def test_init_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("os.getenv", return_value=""):
                with pytest.raises(ValueError, match="API key required"):
                    ApiClient()

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = ApiClient(api_key="test-key", timeout=30.0)
        assert client.timeout == 30.0

    def test_init_default_timeout(self):
        """Test default timeout value."""
        client = ApiClient(api_key="test-key")
        assert client.timeout == 10.0

    def test_init_sets_base_url(self):
        """Test that base URL is set from Config."""
        client = ApiClient(api_key="test-key")
        assert client.base_url is not None
        assert "leanexplore" in client.base_url.lower()


class TestApiClientSearch:
    """Tests for ApiClient.search method."""

    @pytest.fixture
    def client(self):
        """Create an ApiClient instance for testing."""
        return ApiClient(api_key="test-api-key")

    @pytest.fixture
    def mock_search_response(self):
        """Create a mock API search response."""
        return {
            "results": [
                {
                    "id": 1,
                    "name": "Nat.add",
                    "module": "Init.Data.Nat.Basic",
                    "docstring": "Addition of natural numbers",
                    "source_text": "def add (a b : Nat) : Nat := a + b",
                    "source_link": "https://github.com/example#L100",
                    "dependencies": None,
                    "informalization": "Adds two natural numbers",
                },
                {
                    "id": 2,
                    "name": "Nat.mul",
                    "module": "Init.Data.Nat.Basic",
                    "docstring": "Multiplication of natural numbers",
                    "source_text": "def mul (a b : Nat) : Nat := a * b",
                    "source_link": "https://github.com/example#L110",
                    "dependencies": None,
                    "informalization": "Multiplies two natural numbers",
                },
            ],
            "processing_time_ms": 42,
        }

    async def test_search_success(self, client, mock_search_response):
        """Test successful search request."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            response = await client.search(query="natural numbers", limit=10)

            assert isinstance(response, SearchResponse)
            assert response.query == "natural numbers"
            assert response.count == 2
            assert len(response.results) == 2
            assert response.processing_time_ms == 42
            assert response.results[0].name == "Nat.add"

    async def test_search_empty_results(self, client):
        """Test search with no results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [], "processing_time_ms": 5}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            response = await client.search(query="nonexistent query")

            assert response.count == 0
            assert response.results == []

    async def test_search_passes_parameters(self, client):
        """Test that search passes correct parameters to API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            await client.search(query="test query", limit=25)

            mock_async_client.get.assert_called_once()
            call_args = mock_async_client.get.call_args
            assert call_args.kwargs["params"]["q"] == "test query"
            assert call_args.kwargs["params"]["limit"] == 25

    async def test_search_includes_auth_header(self, client):
        """Test that search includes authorization header."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            await client.search(query="test")

            call_args = mock_async_client.get.call_args
            assert "Authorization" in call_args.kwargs["headers"]
            assert "Bearer" in call_args.kwargs["headers"]["Authorization"]

    async def test_search_http_error(self, client):
        """Test that HTTP errors are propagated."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            with pytest.raises(httpx.HTTPStatusError):
                await client.search(query="test")

    async def test_search_default_limit(self, client):
        """Test default limit parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            await client.search(query="test")

            call_args = mock_async_client.get.call_args
            assert call_args.kwargs["params"]["limit"] == 20


class TestApiClientGetById:
    """Tests for ApiClient.get_by_id method."""

    @pytest.fixture
    def client(self):
        """Create an ApiClient instance for testing."""
        return ApiClient(api_key="test-api-key")

    @pytest.fixture
    def mock_declaration_response(self):
        """Create a mock API declaration response."""
        return {
            "id": 42,
            "name": "List.map",
            "module": "Init.Data.List.Basic",
            "docstring": "Maps a function over a list",
            "source_text": "def map (f : a -> b) (xs : List a) : List b := ...",
            "source_link": "https://github.com/example#L200",
            "dependencies": '["List"]',
            "informalization": "Applies a function to each element",
        }

    async def test_get_by_id_found(self, client, mock_declaration_response):
        """Test successful retrieval of a declaration by ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_declaration_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            result = await client.get_by_id(declaration_id=42)

            assert isinstance(result, SearchResult)
            assert result.id == 42
            assert result.name == "List.map"

    async def test_get_by_id_not_found(self, client):
        """Test retrieval of non-existent declaration returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            result = await client.get_by_id(declaration_id=99999)

            assert result is None

    async def test_get_by_id_http_error(self, client):
        """Test that HTTP errors (non-404) are propagated."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_by_id(declaration_id=42)

    async def test_get_by_id_includes_auth_header(self, client):
        """Test that get_by_id includes authorization header."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            await client.get_by_id(declaration_id=1)

            call_args = mock_async_client.get.call_args
            assert "Authorization" in call_args.kwargs["headers"]

    async def test_get_by_id_correct_endpoint(self, client):
        """Test that get_by_id uses correct endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_async_client

            await client.get_by_id(declaration_id=123)

            call_args = mock_async_client.get.call_args
            endpoint = call_args.args[0]
            assert "/declarations/123" in endpoint
