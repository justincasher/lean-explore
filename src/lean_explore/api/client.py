# src/lean_explore/api/client.py

"""Provides a client for interacting with the remote Lean Explore API.

This module contains the Client class, which facilitates
communication with the backend Lean Explore search engine API for
performing searches and retrieving detailed information.
"""

from typing import List, Optional

import httpx

from lean_explore.shared.models.api import (
    APICitationsResponse,
    APISearchResponse,
    APISearchResultItem,
)

_DEFAULT_API_BASE_URL = "https://www.leanexplore.com/api/v1"


class Client:
    """An asynchronous client for the Lean Explore backend API.

    This client handles making HTTP requests to the production API base URL,
    authenticating with an API key, and parsing responses into Pydantic models.

    Attributes:
        api_key: The API key used for authenticating requests.
        timeout: The timeout for HTTP requests in seconds.
        base_url: The hardcoded base URL for the API.
    """

    def __init__(self, api_key: str, timeout: float = 10.0):
        """Initializes the API Client.

        Args:
            api_key: The API key for authentication.
            timeout: Default timeout for HTTP requests in seconds.
        """
        self.base_url: str = _DEFAULT_API_BASE_URL
        self.api_key: str = api_key
        self.timeout: float = timeout
        self._headers: dict = {"Authorization": f"Bearer {self.api_key}"}

    async def search(
        self, query: str, package_filters: Optional[List[str]] = None
    ) -> APISearchResponse:
        """Performs a search for statement groups via the API.

        Args:
            query: The search query string.
            package_filters: An optional list of package names to filter the
                search by.

        Returns:
            An APISearchResponse object containing the search results and
            associated metadata.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status (4xx or 5xx).
            httpx.RequestError: For network-related issues or other request errors.
        """
        endpoint = f"{self.base_url}/search"
        params = {"q": query}
        if package_filters:
            params["pkg"] = package_filters

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint, params=params, headers=self._headers)
            response.raise_for_status()
            return APISearchResponse(**response.json())

    async def get_by_id(self, group_id: int) -> Optional[APISearchResultItem]:
        """Retrieves a specific statement group by its unique ID via the API.

        Args:
            group_id: The unique identifier of the statement group.

        Returns:
            An APISearchResultItem object if the statement group is found,
            otherwise None if a 404 error is received.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status
                                   other than 404 (e.g., 401, 403, 5xx).
            httpx.RequestError: For network-related issues or other request errors.
        """
        endpoint = f"{self.base_url}/statement_groups/{group_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint, headers=self._headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return APISearchResultItem(**response.json())

    async def get_dependencies(self, group_id: int) -> Optional[APICitationsResponse]:
        """Retrieves the dependencies (citations) for a specific statement group.

        This method fetches the statement groups that the specified 'group_id'
        depends on (i.e., cites).

        Args:
            group_id: The unique identifier of the statement group for which
                to fetch dependencies.

        Returns:
            An APICitationsResponse object containing the list of dependencies
            (cited items) if the source statement group is found. Returns None
            if the source statement group itself is not found (receives a 404).

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status
                                   other than 404 (e.g., 401, 403, 5xx).
            httpx.RequestError: For network-related issues or other request errors.
        """
        endpoint = f"{self.base_url}/statement_groups/{group_id}/dependencies"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint, headers=self._headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return APICitationsResponse(**response.json())
