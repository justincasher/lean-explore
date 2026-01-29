"""Tests for search type models and utility functions."""

from lean_explore.models.search_types import (
    SearchResultSummary,
    SearchSummaryResponse,
    extract_bold_description,
)


class TestExtractBoldDescription:
    """Tests for extract_bold_description utility function."""

    def test_extracts_bold_header(self):
        """Test extraction from a standard informalization."""
        text = "**Group Homomorphism.** A function that preserves group structure."
        assert extract_bold_description(text) == "Group Homomorphism."

    def test_extracts_bold_header_without_trailing_period(self):
        """Test extraction when bold header has no trailing period."""
        text = "**Linear Map** A linear transformation between vector spaces."
        assert extract_bold_description(text) == "Linear Map"

    def test_returns_none_for_none_input(self):
        """Test that None input returns None."""
        assert extract_bold_description(None) is None

    def test_returns_none_for_empty_string(self):
        """Test that empty string returns None."""
        assert extract_bold_description("") is None

    def test_returns_none_when_no_bold_markers(self):
        """Test that text without bold markers returns None."""
        text = "A plain description with no bold header."
        assert extract_bold_description(text) is None

    def test_extracts_first_bold_section_only(self):
        """Test that only the first bold section is extracted."""
        text = "**First Bold.** Some text **Second Bold.** More text."
        assert extract_bold_description(text) == "First Bold."

    def test_handles_bold_with_special_characters(self):
        """Test extraction with special characters in bold text."""
        text = "**Nat.add (Addition of Natural Numbers).** Adds two natural numbers."
        assert (
            extract_bold_description(text) == "Nat.add (Addition of Natural Numbers)."
        )


class TestSearchResultSummary:
    """Tests for SearchResultSummary model."""

    def test_create_with_description(self):
        """Test creating a summary with all fields."""
        summary = SearchResultSummary(
            id=1,
            name="Nat.add",
            description="Addition of Natural Numbers.",
        )
        assert summary.id == 1
        assert summary.name == "Nat.add"
        assert summary.description == "Addition of Natural Numbers."

    def test_create_with_none_description(self):
        """Test creating a summary with no description."""
        summary = SearchResultSummary(
            id=42,
            name="List.map",
            description=None,
        )
        assert summary.id == 42
        assert summary.name == "List.map"
        assert summary.description is None

    def test_model_dump(self):
        """Test serialization to dictionary."""
        summary = SearchResultSummary(
            id=1,
            name="Nat.add",
            description="Addition.",
        )
        data = summary.model_dump()
        assert data == {"id": 1, "name": "Nat.add", "description": "Addition."}

    def test_model_dump_excludes_none(self):
        """Test that model_dump with exclude_none omits None fields."""
        summary = SearchResultSummary(id=1, name="Nat.add", description=None)
        data = summary.model_dump(exclude_none=True)
        assert data == {"id": 1, "name": "Nat.add"}


class TestSearchSummaryResponse:
    """Tests for SearchSummaryResponse model."""

    def test_create_response(self):
        """Test creating a summary response with results."""
        results = [
            SearchResultSummary(id=1, name="Nat.add", description="Addition."),
            SearchResultSummary(id=2, name="Nat.mul", description="Multiplication."),
        ]
        response = SearchSummaryResponse(
            query="natural number",
            results=results,
            count=2,
            processing_time_ms=150,
        )
        assert response.query == "natural number"
        assert len(response.results) == 2
        assert response.count == 2
        assert response.processing_time_ms == 150

    def test_create_response_without_processing_time(self):
        """Test that processing_time_ms defaults to None."""
        response = SearchSummaryResponse(
            query="test",
            results=[],
            count=0,
        )
        assert response.processing_time_ms is None

    def test_model_dump(self):
        """Test serialization of full response."""
        results = [
            SearchResultSummary(id=1, name="Nat.add", description="Addition."),
        ]
        response = SearchSummaryResponse(
            query="add",
            results=results,
            count=1,
            processing_time_ms=100,
        )
        data = response.model_dump(exclude_none=True)
        assert data == {
            "query": "add",
            "results": [{"id": 1, "name": "Nat.add", "description": "Addition."}],
            "count": 1,
            "processing_time_ms": 100,
        }
