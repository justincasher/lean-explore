"""Command-Line Interface for Lean Explore.

Provides commands to search for Lean declarations via the remote API,
interact with AI agents, and manage local data.
"""

import asyncio
import logging
import os
import subprocess
import sys

import typer
from rich.console import Console

from lean_explore.api import ApiClient
from lean_explore.cli import data_commands
from lean_explore.cli.display import display_search_results

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="lean-explore",
    help="A CLI tool to explore and search Lean mathematical libraries.",
    add_completion=False,
    rich_markup_mode="markdown",
)

mcp_app = typer.Typer(
    name="mcp", help="Manage and run the Model Context Protocol (MCP) server."
)
app.add_typer(mcp_app)

app.add_typer(
    data_commands.app,
    name="data",
    help="Manage local data toolchains.",
)


def _get_console(use_stderr: bool = False) -> Console:
    """Create a Rich console instance for output.

    Args:
        use_stderr: If True, output to stderr instead of stdout.

    Returns:
        A configured Console instance.
    """
    return Console(stderr=use_stderr)


@app.command("search")
def search_command(
    query_string: str = typer.Argument(..., help="The search query string."),
    limit: int = typer.Option(
        5, "--limit", "-n", help="Number of search results to display."
    ),
    packages: list[str] | None = typer.Option(
        None, "--package", "-p", help="Filter by package (e.g., -p Mathlib -p Std)."
    ),
):
    """Search for Lean declarations using the Lean Explore API."""
    asyncio.run(_search_async(query_string, limit, packages))


async def _search_async(
    query_string: str, limit: int, packages: list[str] | None
) -> None:
    """Async implementation of search command."""
    console = _get_console()
    error_console = _get_console(use_stderr=True)

    try:
        client = ApiClient()
    except ValueError as error:
        logger.error("Failed to initialize API client: %s", error)
        error_console.print(f"[bold red]Error: {error}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Searching for: '{query_string}'...")
    response = await client.search(query=query_string, limit=limit, packages=packages)
    display_search_results(response, display_limit=limit, console=console)


@mcp_app.command("serve")
def mcp_serve_command(
    backend: str = typer.Option(
        "api",
        "--backend",
        "-b",
        help="Backend to use for the MCP server: 'api' or 'local'. Default is 'api'.",
        case_sensitive=False,
        show_choices=True,
    ),
    api_key_override: str | None = typer.Option(
        None,
        "--api-key",
        help="API key to use if backend is 'api'. Overrides env var.",
    ),
):
    """Launch the Lean Explore MCP (Model Context Protocol) server."""
    error_console = _get_console(use_stderr=True)

    command_parts = [
        sys.executable,
        "-m",
        "lean_explore.mcp.server",
        "--backend",
        backend.lower(),
    ]

    if backend.lower() == "api":
        effective_api_key = api_key_override or os.getenv("LEANEXPLORE_API_KEY")
        if not effective_api_key:
            logger.error("API key required for 'api' backend but not provided")
            error_console.print(
                "[bold red]API key required for 'api' backend.[/bold red]\n"
                "Set LEANEXPLORE_API_KEY or use --api-key option."
            )
            raise typer.Abort()
        if api_key_override:
            command_parts.extend(["--api-key", api_key_override])

    logger.info("Starting MCP server with backend: %s", backend.lower())
    result = subprocess.run(command_parts, check=False)

    if result.returncode != 0:
        logger.error("MCP server exited with code %d", result.returncode)
        raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
