"""Client for interacting with the remote Lean Explore API."""

import httpx

from lean_explore.search.types import SearchResponse, SearchResult

_DEFAULT_API_BASE_URL = "https://www.leanexplore.com/api/v1"


class ApiClient:
    """Async client for the remote Lean Explore API.

    This client handles making HTTP requests to the API, authenticating
    with an API key, and parsing responses into SearchResult objects.
    """

    def __init__(self, api_key: str, timeout: float = 10.0):
        """Initialize the API client.

        Args:
            api_key: The API key for authentication.
            timeout: Default timeout for HTTP requests in seconds.
        """
        self.base_url: str = _DEFAULT_API_BASE_URL
        self.api_key: str = api_key
        self.timeout: float = timeout
        self._headers: dict = {"Authorization": f"Bearer {self.api_key}"}

    async def search(
        self,
        query: str,
        limit: int = 20,
    ) -> SearchResponse:
        """Search for Lean declarations via the API.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            SearchResponse containing results and metadata.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status.
            httpx.RequestError: For network-related issues.
        """
        endpoint = f"{self.base_url}/search"
        params = {"q": query, "limit": limit}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint, params=params, headers=self._headers)
            response.raise_for_status()
            data = response.json()

            # Parse API response into our types
            results = [SearchResult(**item) for item in data.get("results", [])]

            return SearchResponse(
                query=query,
                results=results,
                count=len(results),
                processing_time_ms=data.get("processing_time_ms"),
            )

    async def get_by_id(self, declaration_id: int) -> SearchResult | None:
        """Retrieve a declaration by ID via the API.

        Args:
            declaration_id: The declaration ID.

        Returns:
            SearchResult if found, None otherwise.

        Raises:
            httpx.HTTPStatusError: If the API returns an error (except 404).
            httpx.RequestError: For network-related issues.
        """
        endpoint = f"{self.base_url}/declarations/{declaration_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint, headers=self._headers)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return SearchResult(**response.json())
