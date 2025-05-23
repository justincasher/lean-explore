# tests/lean_explore/mcp/test_server.py

"""Tests for the Lean Explore MCP server entry point in `lean_explore.mcp.server`.

This module verifies the correct command-line argument parsing,
backend service initialization (API or local), and the overall
execution flow of the MCP server. It utilizes mocking to isolate
file system interactions, backend service instantiation, and
the `FastMCP` application's `run` method.
"""

import pytest
import sys
import argparse
import logging
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# Assume these are available via conftest or direct import if not using fixtures for them
from lean_explore.mcp.app import mcp_app, BackendServiceType
from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService
import lean_explore.defaults # Used for default paths in pre-checks

# The module under test
from lean_explore.mcp.server import main, parse_arguments

class TestServerArguments:
    """Test suite for parsing command-line arguments in `lean_explore.mcp.server`."""

    def test_parse_arguments_api_backend_with_key(self):
        """Verifies correct parsing for API backend with an API key."""
        test_args = ["--backend", "api", "--api-key", "test_api_key_123", "--log-level", "INFO"]
        with patch.object(sys, 'argv', ['server_script.py'] + test_args):
            args = parse_arguments()
            assert args.backend == "api"
            assert args.api_key == "test_api_key_123"
            assert args.log_level == "INFO"

    def test_parse_arguments_local_backend_no_key(self):
        """Verifies correct parsing for local backend without an API key."""
        test_args = ["--backend", "local", "--log-level", "DEBUG"]
        with patch.object(sys, 'argv', ['server_script.py'] + test_args):
            args = parse_arguments()
            assert args.backend == "local"
            assert args.api_key is None
            assert args.log_level == "DEBUG"

    @pytest.mark.parametrize("backend_choice", ["api", "local"])
    def test_parse_arguments_backend_required(self, backend_choice: str):
        """Ensures that the --backend argument is required."""
        test_args = ["--api-key", "dummy", "--log-level", "WARNING"] if backend_choice == "api" else ["--log-level", "WARNING"]
        with patch.object(sys, 'argv', ['server_script.py'] + test_args), \
             pytest.raises(SystemExit) as sysexit: # argparse raises SystemExit on error
            parse_arguments()
        assert sysexit.type == SystemExit
        assert sysexit.value.code == 2 # Standard exit code for argparse errors

    def test_parse_arguments_api_key_required_for_api_backend(self):
        """Ensures --api-key is required when --backend is 'api'."""
        test_args = ["--backend", "api", "--log-level", "ERROR"]
        with patch.object(sys, 'argv', ['server_script.py'] + test_args), \
             patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as sysexit:
            main() # Call main to test the full argument validation logic
        mock_print.assert_any_call("--api-key is required when using the 'api' backend.", file=sys.stderr)
        assert sysexit.type == SystemExit
        assert sysexit.value.code == 1

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_parse_arguments_log_level_choices(self, log_level: str):
        """Verifies valid log level choices are accepted."""
        test_args = ["--backend", "local", "--log-level", log_level]
        with patch.object(sys, 'argv', ['server_script.py'] + test_args):
            args = parse_arguments()
            assert args.log_level == log_level

    def test_parse_arguments_invalid_log_level(self):
        """Ensures invalid log levels are rejected."""
        test_args = ["--backend", "local", "--log-level", "INVALID"]
        with patch.object(sys, 'argv', ['server_script.py'] + test_args), \
             pytest.raises(SystemExit) as sysexit:
            parse_arguments()
        assert sysexit.type == SystemExit
        assert sysexit.value.code == 2


class TestServerMainFunction:
    """Test suite for the main execution function of the MCP server."""

    @pytest.fixture(autouse=True)
    def setup_main_mocks(self, mocker):
        """Sets up common mocks for the main function tests."""
        # Mock sys.argv to control command-line arguments
        self.mock_sys_argv = mocker.patch.object(sys, 'argv', ['server_script.py'])
        # Mock sys.exit to prevent actual program exit during tests
        self.mock_sys_exit = mocker.patch.object(sys, 'exit')
        # Mock logging setup to prevent actual logging configuration interference
        self.mock_logging_config = mocker.patch('logging.basicConfig')
        # Mock rich.console.Console.print to capture error messages
        self.mock_error_console_print = mocker.patch('rich.console.Console.print')
        # Mock the mcp_app.run method to prevent the server from actually starting
        self.mock_mcp_app_run = mocker.patch('lean_explore.mcp.app.mcp_app.run')
        # Ensure _lean_explore_backend_service is clean before each test
        if hasattr(mcp_app, '_lean_explore_backend_service'):
            del mcp_app._lean_explore_backend_service

        # Mocks for backend initialization
        self.mock_api_client_init = mocker.patch('lean_explore.api.client.Client')
        self.mock_local_service_init = mocker.patch('lean_explore.local.service.Service')

        # Mock defaults paths and existence checks for local backend pre-checks
        self.mock_default_db_path = mocker.patch.object(lean_explore.defaults, 'DEFAULT_DB_PATH', new_callable=MagicMock)
        self.mock_default_faiss_index_path = mocker.patch.object(lean_explore.defaults, 'DEFAULT_FAISS_INDEX_PATH', new_callable=MagicMock)
        self.mock_default_faiss_map_path = mocker.patch.object(lean_explore.defaults, 'DEFAULT_FAISS_MAP_PATH', new_callable=MagicMock)
        self.mock_lean_explore_toolchains_base_dir = mocker.patch.object(lean_explore.defaults, 'LEAN_EXPLORE_TOOLCHAINS_BASE_DIR', new_callable=MagicMock)
        self.mock_default_active_toolchain_version = mocker.patch.object(lean_explore.defaults, 'DEFAULT_ACTIVE_TOOLCHAIN_VERSION', 'test_version')

        # By default, assume local files exist
        self.mock_default_db_path.exists.return_value = True
        self.mock_default_faiss_index_path.exists.return_value = True
        self.mock_default_faiss_map_path.exists.return_value = True
        self.mock_lean_explore_toolchains_base_dir.return_value.__truediv__.return_value.resolve.return_value = Path("/tmp/test_toolchain")


    def test_main_runs_api_backend_successfully(self):
        """Verifies the main function successfully initializes and runs with API backend."""
        self.mock_sys_argv.extend(["--backend", "api", "--api-key", "my_api_key", "--log-level", "INFO"])
        
        mock_api_client_instance = MagicMock(spec=APIClient)
        self.mock_api_client_init.return_value = mock_api_client_instance

        main()

        self.mock_logging_config.assert_called_with(
            level=logging.INFO,
            format=mocker.ANY,
            datefmt=mocker.ANY,
            stream=sys.stderr,
            force=True
        )
        self.mock_api_client_init.assert_called_once_with(api_key="my_api_key")
        assert mcp_app._lean_explore_backend_service is mock_api_client_instance
        self.mock_mcp_app_run.assert_called_once_with(transport='stdio')
        self.mock_sys_exit.assert_not_called()

    def test_main_runs_local_backend_successfully(self):
        """Verifies the main function successfully initializes and runs with local backend."""
        self.mock_sys_argv.extend(["--backend", "local", "--log-level", "DEBUG"])

        mock_local_service_instance = MagicMock(spec=LocalService)
        self.mock_local_service_init.return_value = mock_local_service_instance

        main()

        self.mock_logging_config.assert_called_with(
            level=logging.DEBUG,
            format=mocker.ANY,
            datefmt=mocker.ANY,
            stream=sys.stderr,
            force=True
        )
        self.mock_default_db_path.exists.assert_called_once()
        self.mock_default_faiss_index_path.exists.assert_called_once()
        self.mock_default_faiss_map_path.exists.assert_called_once()
        self.mock_local_service_init.assert_called_once_with()
        assert mcp_app._lean_explore_backend_service is mock_local_service_instance
        self.mock_mcp_app_run.assert_called_once_with(transport='stdio')
        self.mock_sys_exit.assert_not_called()

    def test_main_exits_if_api_key_missing_for_api_backend(self):
        """Verifies that the main function exits if API key is missing for API backend."""
        self.mock_sys_argv.extend(["--backend", "api"]) # Missing --api-key

        main()

        self.mock_error_console_print.assert_not_called() # rich console is not used for this message
        # Use patch('builtins.print') instead for exact verification of message to stderr
        # or rely on sys.exit being called with specific code.
        self.mock_sys_exit.assert_called_once_with(1)
        self.mock_api_client_init.assert_not_called()
        self.mock_mcp_app_run.assert_not_called()


    def test_main_exits_if_local_backend_files_missing(self):
        """Verifies main function exits if essential local backend data files are missing."""
        self.mock_sys_argv.extend(["--backend", "local", "--log-level", "INFO"])
        self.mock_default_db_path.exists.return_value = False # Simulate missing DB file

        main()

        self.mock_default_db_path.exists.assert_called_once()
        self.mock_local_service_init.assert_not_called()
        self.mock_mcp_app_run.assert_not_called()
        self.mock_sys_exit.assert_called_once_with(1)
        self.mock_error_console_print.assert_called_once()
        assert "Essential data files for the local backend are missing." in self.mock_error_console_print.call_args[0][0]
        assert "Please run `leanexplore data fetch`" in self.mock_error_console_print.call_args[0][0]

    def test_main_exits_if_local_service_initialization_fails_filenotfound(self):
        """Verifies main function exits if LocalService init fails with FileNotFoundError."""
        self.mock_sys_argv.extend(["--backend", "local", "--log-level", "ERROR"])
        # Pre-checks pass, but LocalService itself raises FileNotFoundError
        self.mock_local_service_init.side_effect = FileNotFoundError("mock_db_corrupted.db")

        main()

        self.mock_local_service_init.assert_called_once()
        self.mock_mcp_app_run.assert_not_called()
        self.mock_sys_exit.assert_called_once_with(1)
        # Check if error is logged critically
        assert any(
            "LocalService initialization failed due to an unexpected missing file" in record.message
            for record in self.mock_logging_config.call_args[0] if isinstance(record, logging.LogRecord)
        )

    def test_main_exits_if_local_service_initialization_fails_runtime_error(self):
        """Verifies main function exits if LocalService init fails with generic RuntimeError."""
        self.mock_sys_argv.extend(["--backend", "local", "--log-level", "ERROR"])
        self.mock_local_service_init.side_effect = RuntimeError("Generic initialization error")

        main()

        self.mock_local_service_init.assert_called_once()
        self.mock_mcp_app_run.assert_not_called()
        self.mock_sys_exit.assert_called_once_with(1)
        assert any(
            "LocalService initialization failed: Generic initialization error" in record.message
            for record in self.mock_logging_config.call_args[0] if isinstance(record, logging.LogRecord)
        )

    def test_main_exits_if_api_client_initialization_fails(self):
        """Verifies main function exits if APIClient initialization fails."""
        self.mock_sys_argv.extend(["--backend", "api", "--api-key", "my_key", "--log-level", "ERROR"])
        self.mock_api_client_init.side_effect = Exception("API client init failed")

        main()

        self.mock_api_client_init.assert_called_once()
        self.mock_mcp_app_run.assert_not_called()
        self.mock_sys_exit.assert_called_once_with(1)
        assert any(
            "An unexpected error occurred while initializing APIClient: API client init failed" in record.message
            for record in self.mock_logging_config.call_args[0] if isinstance(record, logging.LogRecord)
        )

    def test_main_handles_mcp_app_run_exception(self):
        """Verifies main function handles exceptions during mcp_app.run."""
        self.mock_sys_argv.extend(["--backend", "local", "--log-level", "INFO"])
        self.mock_local_service_init.return_value = MagicMock(spec=LocalService)
        self.mock_mcp_app_run.side_effect = Exception("MCP app crashed")

        main()

        self.mock_mcp_app_run.assert_called_once_with(transport='stdio')
        self.mock_sys_exit.assert_called_once_with(1)
        assert any(
            "MCP server exited with an unexpected error: MCP app crashed" in record.message
            for record in self.mock_logging_config.call_args[0] if isinstance(record, logging.LogRecord)
        )

    def test_main_log_level_configuration(self):
        """Verifies that the logging level is correctly configured."""
        self.mock_sys_argv.extend(["--backend", "local", "--log-level", "WARNING"])

        main()

        # Check that basicConfig was called with the correct level
        self.mock_logging_config.assert_called_with(
            level=logging.WARNING, # Assert on this specific argument
            format=mocker.ANY,
            datefmt=mocker.ANY,
            stream=sys.stderr,
            force=True
        )
        self.mock_sys_exit.assert_not_called()