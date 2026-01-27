"""Tests for the CLI display module.

These tests verify the text formatting and display functions used to render
search results in the terminal.
"""

from io import StringIO

import pytest
from rich.console import Console

from lean_explore.cli.display import (
    _format_text_for_panel,
    _wrap_line,
    display_search_results,
)
from lean_explore.models import SearchResponse, SearchResult


class TestWrapLine:
    """Tests for the _wrap_line helper function."""

    def test_wrap_empty_line(self):
        """Test that empty lines return padded empty string."""
        result = _wrap_line("", width=80)
        assert len(result) == 1
        assert result[0] == " " * 80

    def test_wrap_whitespace_only_line(self):
        """Test that whitespace-only lines return padded empty string."""
        result = _wrap_line("   ", width=80)
        assert len(result) == 1
        assert result[0] == " " * 80

    def test_wrap_short_line(self):
        """Test that short lines are padded to width."""
        result = _wrap_line("Hello", width=80)
        assert len(result) == 1
        assert result[0] == "Hello" + " " * 75
        assert len(result[0]) == 80

    def test_wrap_long_line(self):
        """Test that long lines are wrapped and each segment padded."""
        long_text = "This is a very long line that should be wrapped " * 3
        result = _wrap_line(long_text, width=40)

        assert len(result) > 1
        for segment in result:
            assert len(segment) == 40

    def test_wrap_exact_width_line(self):
        """Test line that exactly matches width."""
        exact_line = "x" * 40
        result = _wrap_line(exact_line, width=40)
        assert len(result) == 1
        assert result[0] == exact_line


class TestFormatTextForPanel:
    """Tests for the _format_text_for_panel function."""

    def test_format_none_content(self):
        """Test that None content returns empty padded line."""
        result = _format_text_for_panel(None, width=80)
        assert result == " " * 80

    def test_format_empty_string(self):
        """Test that empty string returns empty padded line."""
        result = _format_text_for_panel("", width=80)
        assert result == " " * 80

    def test_format_single_line(self):
        """Test formatting a single line of text."""
        result = _format_text_for_panel("Hello world", width=80)
        lines = result.split("\n")
        assert len(lines) == 1
        assert lines[0].startswith("Hello world")
        assert len(lines[0]) == 80

    def test_format_multiple_lines(self):
        """Test formatting text with multiple lines."""
        text = "Line one\nLine two\nLine three"
        result = _format_text_for_panel(text, width=80)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("Line one")
        assert lines[1].startswith("Line two")
        assert lines[2].startswith("Line three")

    def test_format_paragraphs(self):
        """Test formatting text with paragraph breaks (double newlines)."""
        text = "First paragraph.\n\nSecond paragraph."
        result = _format_text_for_panel(text, width=80)
        lines = result.split("\n")

        # Should have: "First paragraph", empty line, "Second paragraph"
        assert len(lines) == 3
        assert lines[0].startswith("First paragraph")
        assert lines[1].strip() == ""
        assert lines[2].startswith("Second paragraph")

    def test_format_all_lines_same_width(self):
        """Test that all output lines have consistent width."""
        text = "Short\n\nThis is a much longer line of text\n\nAnother short"
        result = _format_text_for_panel(text, width=50)
        lines = result.split("\n")

        for line in lines:
            assert len(line) == 50

    def test_format_preserves_paragraph_structure(self):
        """Test that paragraph structure is preserved in output."""
        text = "Para 1 line 1\nPara 1 line 2\n\nPara 2 line 1"
        result = _format_text_for_panel(text, width=80)
        lines = result.split("\n")

        # Para 1 line 1, Para 1 line 2, blank, Para 2 line 1
        assert len(lines) == 4
        assert lines[0].strip() == "Para 1 line 1"
        assert lines[1].strip() == "Para 1 line 2"
        assert lines[2].strip() == ""
        assert lines[3].strip() == "Para 2 line 1"


class TestDisplaySearchResults:
    """Tests for the display_search_results function."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console that captures output."""
        output = StringIO()
        return Console(file=output, force_terminal=True, width=120)

    @pytest.fixture
    def sample_search_result(self):
        """Create a sample search result for testing."""
        return SearchResult(
            id=1,
            name="Nat.add",
            module="Init.Data.Nat.Basic",
            docstring="Adds two natural numbers.",
            source_text="def add (a b : Nat) : Nat := a + b",
            source_link="https://github.com/example/lean4#L100",
            dependencies=None,
            informalization="Adds two natural numbers together.",
        )

    @pytest.fixture
    def sample_response(self, sample_search_result):
        """Create a sample search response."""
        return SearchResponse(
            query="add",
            results=[sample_search_result],
            count=1,
            processing_time_ms=42,
        )

    def test_display_creates_console_if_none(self, sample_response):
        """Test that display creates a console if none provided."""
        # Should not raise an error
        display_search_results(sample_response, display_limit=5, console=None)

    def test_display_uses_provided_console(self, sample_response, mock_console):
        """Test that display uses the provided console."""
        display_search_results(sample_response, display_limit=5, console=mock_console)
        # If we got here without error, the console was used

    def test_display_empty_results(self, mock_console):
        """Test display with no search results."""
        response = SearchResponse(
            query="nonexistent",
            results=[],
            count=0,
            processing_time_ms=10,
        )
        display_search_results(response, display_limit=5, console=mock_console)

    def test_display_respects_limit(self, mock_console):
        """Test that display_limit is respected."""
        results = [
            SearchResult(
                id=i,
                name=f"Result.{i}",
                module="Test",
                docstring=None,
                source_text=f"def r{i} := {i}",
                source_link="https://example.com",
                dependencies=None,
                informalization=None,
            )
            for i in range(10)
        ]
        response = SearchResponse(
            query="test",
            results=results,
            count=10,
            processing_time_ms=50,
        )

        # The function should only display up to display_limit results
        display_search_results(response, display_limit=3, console=mock_console)

    def test_display_handles_none_fields(self, mock_console):
        """Test display with None optional fields."""
        result = SearchResult(
            id=1,
            name="Test",
            module="Test",
            docstring=None,
            source_text="def test := 1",
            source_link="https://example.com",
            dependencies=None,
            informalization=None,
        )
        response = SearchResponse(
            query="test",
            results=[result],
            count=1,
            processing_time_ms=None,
        )
        display_search_results(response, display_limit=5, console=mock_console)

    def test_display_with_all_optional_fields(self, mock_console):
        """Test display with all optional fields populated."""
        result = SearchResult(
            id=1,
            name="Nat.add",
            module="Init.Data.Nat",
            docstring="Addition function",
            source_text="def add := ...",
            source_link="https://example.com",
            dependencies='["Nat"]',
            informalization="Adds two numbers",
        )
        response = SearchResponse(
            query="add",
            results=[result],
            count=1,
            processing_time_ms=42,
        )
        display_search_results(response, display_limit=5, console=mock_console)
