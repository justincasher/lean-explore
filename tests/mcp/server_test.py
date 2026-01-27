"""Tests for the MCP server module.

These tests verify the MCP server initialization and argument parsing.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from lean_explore.mcp.server import _get_error_console, _parse_arguments


class TestGetErrorConsole:
    """Tests for the _get_error_console helper function."""

    def test_returns_console(self):
        """Test that _get_error_console returns a Console instance."""
        console = _get_error_console()
        assert console is not None
        assert console.stderr is True


class TestParseArguments:
    """Tests for the _parse_arguments function."""

    def test_parse_api_backend(self):
        """Test parsing with api backend."""
        test_argv = ["server", "--backend", "api", "--api-key", "k"]
        with patch.object(sys, "argv", test_argv):
            args = _parse_arguments()
            assert args.backend == "api"
            assert args.api_key == "k"

    def test_parse_local_backend(self):
        """Test parsing with local backend."""
        with patch.object(sys, "argv", ["server", "--backend", "local"]):
            args = _parse_arguments()
            assert args.backend == "local"
            assert args.api_key is None

    def test_parse_log_level(self):
        """Test parsing custom log level."""
        with patch.object(
            sys, "argv", ["server", "--backend", "local", "--log-level", "DEBUG"]
        ):
            args = _parse_arguments()
            assert args.log_level == "DEBUG"

    def test_default_log_level(self):
        """Test default log level is ERROR."""
        with patch.object(sys, "argv", ["server", "--backend", "local"]):
            args = _parse_arguments()
            assert args.log_level == "ERROR"

    def test_missing_backend_exits(self):
        """Test that missing backend argument causes exit."""
        with patch.object(sys, "argv", ["server"]):
            with pytest.raises(SystemExit):
                _parse_arguments()

    def test_invalid_backend_exits(self):
        """Test that invalid backend choice causes exit."""
        with patch.object(sys, "argv", ["server", "--backend", "invalid"]):
            with pytest.raises(SystemExit):
                _parse_arguments()

    def test_invalid_log_level_exits(self):
        """Test that invalid log level causes exit."""
        with patch.object(
            sys, "argv", ["server", "--backend", "local", "--log-level", "INVALID"]
        ):
            with pytest.raises(SystemExit):
                _parse_arguments()


class TestMainFunction:
    """Tests for the main function initialization logic."""

    def test_api_backend_missing_key_exits(self):
        """Test that api backend without key exits with error."""
        from lean_explore.mcp.server import main

        with patch.object(
            sys, "argv", ["server", "--backend", "api"]
        ), pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_local_backend_missing_files_exits(self):
        """Test that local backend with missing files exits with error."""
        from lean_explore.mcp.server import main

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path.resolve.return_value = "/fake/path"

        with patch.object(
            sys, "argv", ["server", "--backend", "local"]
        ), patch(
            "lean_explore.mcp.server.Config.DATABASE_PATH", mock_path
        ), patch(
            "lean_explore.mcp.server.Config.ACTIVE_VERSION", "v0.1.0"
        ), patch(
            "lean_explore.mcp.server.Config.ACTIVE_CACHE_PATH", mock_path
        ), pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
