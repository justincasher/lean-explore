"""Tests for the main CLI module.

These tests verify the core CLI commands including search and MCP server launch.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from lean_explore.cli.main import _get_console, _search_async, app
from lean_explore.models import SearchResponse, SearchResult

runner = CliRunner()


class TestGetConsole:
    """Tests for the _get_console helper function."""

    def test_get_console_stdout(self):
        """Test creating a stdout console."""
        console = _get_console(use_stderr=False)
        assert console is not None
        assert not console.stderr

    def test_get_console_stderr(self):
        """Test creating a stderr console."""
        console = _get_console(use_stderr=True)
        assert console is not None
        assert console.stderr


class TestSearchCommand:
    """Tests for the search command."""

    @pytest.fixture
    def mock_api_client(self):
        """Create a mock API client."""
        client = MagicMock()
        client.search = AsyncMock(
            return_value=SearchResponse(
                query="test",
                results=[
                    SearchResult(
                        id=1,
                        name="Test.result",
                        module="Test",
                        docstring="A test result",
                        source_text="def test := 1",
                        source_link="https://example.com",
                        dependencies=None,
                        informalization="A test",
                    )
                ],
                count=1,
                processing_time_ms=42,
            )
        )
        return client

    async def test_search_command_success(self, mock_api_client):
        """Test successful search command execution."""
        with (
            patch("lean_explore.cli.main.ApiClient", return_value=mock_api_client),
            patch("lean_explore.cli.main.display_search_results"),
        ):
            # Test the async implementation directly
            await _search_async(query_string="test query", limit=5, packages=None)
            mock_api_client.search.assert_called_once_with(
                query="test query", limit=5, packages=None
            )

    async def test_search_command_with_limit(self, mock_api_client):
        """Test search command with custom limit."""
        with (
            patch("lean_explore.cli.main.ApiClient", return_value=mock_api_client),
            patch("lean_explore.cli.main.display_search_results"),
        ):
            await _search_async(query_string="test query", limit=10, packages=None)
            mock_api_client.search.assert_called_once_with(
                query="test query", limit=10, packages=None
            )

    async def test_search_command_with_packages(self, mock_api_client):
        """Test search command with package filter."""
        with (
            patch("lean_explore.cli.main.ApiClient", return_value=mock_api_client),
            patch("lean_explore.cli.main.display_search_results"),
        ):
            await _search_async(
                query_string="test query", limit=5, packages=["Mathlib", "Std"]
            )
            mock_api_client.search.assert_called_once_with(
                query="test query", limit=5, packages=["Mathlib", "Std"]
            )

    async def test_search_command_api_key_error(self):
        """Test search command when API key is missing."""
        with patch(
            "lean_explore.cli.main.ApiClient",
            side_effect=ValueError("API key required"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                await _search_async(query_string="test query", limit=5, packages=None)
            assert exc_info.value.exit_code == 1


class TestMcpServeCommand:
    """Tests for the MCP serve command."""

    def test_mcp_serve_missing_api_key(self):
        """Test MCP serve fails without API key for api backend."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing LEANEXPLORE_API_KEY
            with patch("os.getenv", return_value=None):
                result = runner.invoke(app, ["mcp", "serve", "--backend", "api"])
                assert result.exit_code != 0

    def test_mcp_serve_with_api_key_env(self):
        """Test MCP serve with API key from environment."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("os.getenv", return_value="test-api-key"),
            patch("subprocess.run", return_value=mock_result) as mock_run,
        ):
            result = runner.invoke(app, ["mcp", "serve", "--backend", "api"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_mcp_serve_with_api_key_option(self):
        """Test MCP serve with API key from command line option."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = runner.invoke(
                app, ["mcp", "serve", "--backend", "api", "--api-key", "my-key"]
            )
            assert result.exit_code == 0
            mock_run.assert_called_once()
            # Check that --api-key was passed to subprocess
            call_args = mock_run.call_args[0][0]
            assert "--api-key" in call_args
            assert "my-key" in call_args

    def test_mcp_serve_local_backend(self):
        """Test MCP serve with local backend (no API key needed)."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["mcp", "serve", "--backend", "local"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "--backend" in call_args
            assert "local" in call_args

    def test_mcp_serve_subprocess_failure(self):
        """Test MCP serve handles subprocess failures."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with (
            patch("os.getenv", return_value="test-api-key"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = runner.invoke(app, ["mcp", "serve", "--backend", "api"])
            assert result.exit_code == 1

    def test_mcp_serve_backend_case_insensitive(self):
        """Test that backend option is case insensitive."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["mcp", "serve", "--backend", "LOCAL"])
            assert result.exit_code == 0
            call_args = mock_run.call_args[0][0]
            assert "local" in call_args


class TestCliApp:
    """Tests for the CLI app structure."""

    def test_app_help(self):
        """Test that app help displays correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lean-explore" in result.output.lower() or "search" in result.output

    def test_search_help(self):
        """Test that search command help displays correctly."""
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_mcp_help(self):
        """Test that mcp subcommand help displays correctly."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower()

    def test_data_subcommand_exists(self):
        """Test that data subcommand is registered."""
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
