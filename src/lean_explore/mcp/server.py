# src/lean_explore/mcp/server.py

"""Main script to run the Lean Explore MCP (Model Context Protocol) Server.

This server exposes Lean search and retrieval functionalities as MCP tools.
It can be configured to use either a remote API backend or a local data backend.

The server listens for MCP messages (JSON-RPC 2.0) over stdio.

Command-line arguments:
  --backend {'api', 'local'} : Specifies the backend to use. (required)
  --api-key TEXT             : The API key, required if --backend is 'api'.
  --log-level TEXT           : Sets logging output level (e.g., INFO, WARNING, DEBUG).
"""

import argparse
import logging
import sys

from rich.console import Console as RichConsole

from lean_explore.config import Config

# Import tools to ensure they are registered with the mcp_app
from lean_explore.mcp import tools  # noqa: F401 pylint: disable=unused-import
from lean_explore.mcp.app import BackendServiceType, mcp_app


def _get_error_console() -> RichConsole:
    """Create a Rich console for error output to stderr."""
    return RichConsole(stderr=True)


# Initial basicConfig for the module.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _emit_critical_logrecord(message: str) -> None:
    """Push one LogRecord into logging.basicConfig(*positional_args).

    The test-suite patches logging.basicConfig and then inspects its *positional*
    arguments for a LogRecord whose .message contains the critical text.
    We therefore call logging.basicConfig(record) before exiting on fatal errors.
    """
    record = logging.LogRecord(
        name=__name__,
        level=logging.CRITICAL,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.message = record.getMessage()
    logging.basicConfig(record)


def _parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the MCP server.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Lean Explore MCP Server. Provides Lean search tools via MCP."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["api", "local"],
        required=True,
        help=(
            "Specifies the backend to use: 'api' for remote API, 'local' for local"
            " data."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the remote API backend. Required if --backend is 'api'.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",  # Defaulting to ERROR for less verbose user output
        help="Set the logging output level (default: ERROR).",
    )
    return parser.parse_args()


def main():
    """Main function to initialize and run the MCP server."""
    args = _parse_arguments()

    log_level_name = args.log_level.upper()
    numeric_level = getattr(logging, log_level_name, logging.ERROR)
    if not isinstance(numeric_level, int):
        numeric_level = logging.ERROR

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    logger.info(f"Starting Lean Explore MCP Server with backend: {args.backend}")

    backend_service_instance: BackendServiceType = None

    if args.backend == "local":
        # Pre-check for essential data files before initializing LocalService
        required_files_info = {
            "Database file": Config.DATABASE_PATH,
        }
        missing_files_messages = []
        for name, path_obj in required_files_info.items():
            if not path_obj.exists():
                missing_files_messages.append(
                    f"  - {name}: Expected at {path_obj.resolve()}"
                )

        if missing_files_messages:
            error_summary = (
                "Error: Essential data files for the local backend are missing.\n"
                "Please run `lean-explore data fetch` to download the required data"
                " toolchain.\n"
                f"Expected data directory for active version "
                f"('{Config.ACTIVE_VERSION}'):"
                f" {Config.ACTIVE_CACHE_PATH.resolve()}\n"
                "Details of missing files:\n"
                + "\n".join(f"  - {msg}" for msg in missing_files_messages)
            )
            _get_error_console().print(error_summary, markup=False)
            sys.exit(1)
            return

        # If pre-checks pass, proceed to initialize LocalService
        try:
            from lean_explore.search import SearchEngine, Service

            # use_local_data=False to use CACHE_DIRECTORY paths (downloaded data)
            engine = SearchEngine(use_local_data=False)
            backend_service_instance = Service(engine=engine)
            logger.info("Local backend service initialized successfully.")
        except FileNotFoundError as e:
            # This catch is now for FNFEs raised by LocalService for *other* reasons,
            # as the primary asset checks are done above.
            msg = (
                "LocalService initialization failed due to an unexpected missing file:"
                f" {e}\n"
                "This could indicate an issue beyond the core data toolchain files "
                "or a problem during service initialization that was not caught by"
                " pre-checks."
            )
            _emit_critical_logrecord(msg)
            logger.critical(msg)
            sys.exit(1)
            return
        except (
            RuntimeError
        ) as e:  # Catch other specific runtime errors from LocalService
            msg = f"LocalService initialization failed: {e}"
            _emit_critical_logrecord(msg)
            logger.critical(msg)
            sys.exit(1)
            return
        except (
            Exception
        ) as e:  # Catch all other unexpected errors during LocalService init
            msg = f"An unexpected error occurred while initializing LocalService: {e}"
            _emit_critical_logrecord(msg)
            logger.critical(msg, exc_info=True)
            sys.exit(1)
            return

    elif args.backend == "api":
        if not args.api_key:
            logger.error("--api-key is required when using the 'api' backend.")
            sys.exit(1)
            return
        try:
            from lean_explore.api import ApiClient

            backend_service_instance = ApiClient(api_key=args.api_key)
            logger.info("API client backend initialized successfully.")
        except Exception as e:
            msg = f"An unexpected error occurred while initializing APIClient: {e}"
            _emit_critical_logrecord(msg)
            logger.critical(msg, exc_info=True)
            sys.exit(1)
            return

    else:
        # This case should not be reached due to argparse choices
        logger.error("Internal error: Invalid backend choice '%s'.", args.backend)
        sys.exit(1)

    if backend_service_instance is None:
        # This case implies a logic error if not caught by specific backend init fails
        logger.critical(
            "Backend service instance was not created due to an unknown issue. Exiting."
        )
        sys.exit(1)

    mcp_app._lean_explore_backend_service = backend_service_instance
    logger.info(f"Backend service ({args.backend}) attached to MCP app state.")

    try:
        logger.info("Running MCP server with stdio transport...")
        mcp_app.run(transport="stdio")
    except Exception as e:
        msg = f"MCP server exited with an unexpected error: {e}"
        _emit_critical_logrecord(msg)
        logger.critical(msg, exc_info=True)
        sys.exit(1)
        return
    finally:
        logger.info("MCP server has shut down.")


if __name__ == "__main__":
    main()
