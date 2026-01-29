"""Tests for the search service module.

These tests verify the Service wrapper class that provides a clean interface
for search operations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_explore.models import SearchResponse, SearchResult
from lean_explore.search.service import Service


class TestService:
    """Tests for Service class."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SearchEngine."""
        engine = MagicMock()
        engine.search = AsyncMock(return_value=[])
        engine.get_by_id = AsyncMock(return_value=None)
        return engine

    @pytest.fixture
    def service(self, mock_engine):
        """Create a Service with mock engine."""
        return Service(engine=mock_engine)

    async def test_search_returns_response(self, service, mock_engine):
        """Test that search returns a SearchResponse."""
        mock_result = SearchResult(
            id=1,
            name="Nat.add",
            module="Init.Data.Nat",
            docstring="Addition",
            source_text="def add := ...",
            source_link="https://example.com",
            dependencies=None,
            informalization="Adds two numbers",
        )
        mock_engine.search.return_value = [mock_result]

        response = await service.search("add")

        assert isinstance(response, SearchResponse)
        assert response.query == "add"
        assert response.count == 1
        assert len(response.results) == 1
        assert response.processing_time_ms is not None

    async def test_search_empty_results(self, service, mock_engine):
        """Test search with no results."""
        mock_engine.search.return_value = []

        response = await service.search("nonexistent")

        assert response.count == 0
        assert response.results == []

    async def test_search_passes_parameters(self, service, mock_engine):
        """Test that search passes parameters to engine."""
        await service.search("test", limit=10, rerank_top=25)

        mock_engine.search.assert_called_once_with(
            query="test",
            limit=10,
            rerank_top=25,
            packages=None,
        )

    async def test_search_passes_packages_filter(self, service, mock_engine):
        """Test that search passes packages filter to engine."""
        await service.search("test", limit=10, packages=["Mathlib", "Std"])

        mock_engine.search.assert_called_once_with(
            query="test",
            limit=10,
            rerank_top=50,
            packages=["Mathlib", "Std"],
        )

    async def test_get_by_id_found(self, service, mock_engine):
        """Test retrieving a declaration by ID."""
        mock_result = SearchResult(
            id=42,
            name="Test",
            module="Test",
            docstring=None,
            source_text="def test := 1",
            source_link="https://example.com",
            dependencies=None,
            informalization=None,
        )
        mock_engine.get_by_id.return_value = mock_result

        result = await service.get_by_id(42)

        assert result is not None
        assert result.id == 42
        mock_engine.get_by_id.assert_called_once_with(42)

    async def test_get_by_id_not_found(self, service, mock_engine):
        """Test retrieving a non-existent declaration."""
        mock_engine.get_by_id.return_value = None

        result = await service.get_by_id(999)

        assert result is None
