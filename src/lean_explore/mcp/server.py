# src/lean_explore/mcp/server.py

"""Main script to run the Lean Explore MCP (Model Context Protocol) Server.

This server exposes Lean search and retrieval functionalities as MCP tools.
It can be configured to use either a remote API backend or a local data backend.

The server listens for MCP messages (JSON-RPC 2.0) over stdio.

Command-line arguments:
  --backend {'api', 'local'} : Specifies the backend to use. (required)
  --api-key TEXT             : The API key, required if --backend is 'api'.
  --log-level TEXT           : Sets the logging output level (e.g., INFO, WARNING, DEBUG).
"""

import argparse
import logging
import sys
import pathlib # For path checks
from rich.console import Console as RichConsole
error_console = RichConsole(stderr=True)

# Import the FastMCP app instance from mcp.app
# This instance has the lifespan manager configured.
from lean_explore.mcp.app import mcp_app, BackendServiceType

# Import backend clients/services
from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService

# Import tools to ensure they are registered with the mcp_app
from lean_explore.mcp import tools  # noqa: F401 pylint: disable=unused-import

# Import defaults for checking local file paths
from lean_explore import defaults

# Initial basicConfig for the module.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
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
        help="Specifies the backend to use: 'api' for remote API, 'local' for local data."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the remote API backend. Required if --backend is 'api'."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR", # Defaulting to ERROR for less verbose user output
        help="Set the logging output level (default: ERROR)."
    )
    return parser.parse_args()


def main():
    """Main function to initialize and run the MCP server."""
    args = parse_arguments()

    log_level_name = args.log_level.upper()
    numeric_level = getattr(logging, log_level_name, logging.ERROR)
    if not isinstance(numeric_level, int):
        numeric_level = logging.ERROR

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True
    )

    logger.info(f"Starting Lean Explore MCP Server with backend: {args.backend}")

    backend_service_instance: BackendServiceType = None

    if args.backend == "local":
        # Pre-check for essential data files before attempting to initialize LocalService
        required_files_info = {
            "Database file": defaults.DEFAULT_DB_PATH,
            "FAISS index file": defaults.DEFAULT_FAISS_INDEX_PATH,
            "FAISS ID map file": defaults.DEFAULT_FAISS_MAP_PATH,
        }
        missing_files_messages = []
        for name, path_obj in required_files_info.items():
            if not path_obj.exists():
                missing_files_messages.append(f"  - {name}: Expected at {path_obj.resolve()}")

        if missing_files_messages:
            expected_toolchain_dir = (
                defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR /
                defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION
            )
            error_summary = (
            f"[bold red]Error: Essential data files for the local backend are missing.[/bold red]\n"
            f"Please run [bold cyan]`leanexplore data fetch`[/bold cyan] to download the required data toolchain.\n"
            f"Expected data directory for active toolchain ('{defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION}'): "
            f"{expected_toolchain_dir.resolve()}\n"
            f"[yellow]Details of missing files:[/yellow]\n" +
            "\n".join([f"  - {msg}" for msg in missing_files_messages])
        )
            error_console.print(error_summary)
            sys.exit(1)

        # If pre-checks pass, proceed to initialize LocalService
        try:
            backend_service_instance = LocalService()
            logger.info("Local backend service initialized successfully.")
        except FileNotFoundError as e:
            # This catch is now for FNFEs raised by LocalService for *other* reasons,
            # as the primary asset checks are done above.
            logger.critical(
                f"LocalService initialization failed due to an unexpected missing file: {e}\n"
                "This could indicate an issue beyond the core data toolchain files "
                "or a problem during service initialization that was not caught by pre-checks."
            )
            sys.exit(1)
        except RuntimeError as e: # Catch other specific runtime errors from LocalService
            logger.critical(f"LocalService initialization failed: {e}")
            sys.exit(1)
        except Exception as e: # Catch all other unexpected errors during LocalService init
            logger.critical(f"An unexpected error occurred while initializing LocalService: {e}", exc_info=True)
            sys.exit(1)

    elif args.backend == "api":
        if not args.api_key:
            print("--api-key is required when using the 'api' backend.", file=sys.stderr)
            sys.exit(1)
        try:
            backend_service_instance = APIClient(api_key=args.api_key)
            logger.info("API client backend initialized successfully.")
        except Exception as e:
            logger.critical(f"An unexpected error occurred while initializing APIClient: {e}", exc_info=True)
            sys.exit(1)
    else:
        # This case should not be reached due to argparse choices
        print(f"Internal error: Invalid backend choice '{args.backend}'.", file=sys.stderr)
        sys.exit(1)

    if backend_service_instance is None:
        # This case implies a logic error if not caught by specific backend init failures
        logger.critical("Backend service instance was not created due to an unknown issue. Exiting.")
        sys.exit(1)

    mcp_app._lean_explore_backend_service = backend_service_instance
    logger.info(f"Backend service ({args.backend}) attached to MCP app state.")

    try:
        logger.info("Running MCP server with stdio transport...")
        mcp_app.run(transport='stdio')
    except Exception as e:
        logger.critical(f"MCP server exited with an unexpected error: {e}", exc_info=True)
        sys.exit(1) # Ensure non-zero exit code for any server run error
    finally:
        logger.info("MCP server has shut down.")


if __name__ == "__main__":
    main()