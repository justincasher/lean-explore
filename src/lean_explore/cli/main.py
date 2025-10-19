"""Command-Line Interface for Lean Explore.

Provides commands to search for Lean declarations via the remote API,
interact with AI agents, and manage local data.
"""

import os
import subprocess
import sys

import typer
from rich.console import Console

from lean_explore.api import ApiClient
from lean_explore.cli import data_commands
from lean_explore.cli.display import display_search_results

# Initialize Typer app and Rich console
app = typer.Typer(
    name="leanexplore",
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

console = Console()
error_console = Console(stderr=True)


@app.command("search")
async def search_command(
    query_string: str = typer.Argument(..., help="The search query string."),
    limit: int = typer.Option(
        5, "--limit", "-n", help="Number of search results to display."
    ),
):
    """Search for Lean declarations using the Lean Explore API."""
    try:
        client = ApiClient()
    except ValueError as e:
        error_console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Searching for: '{query_string}'...")
    response = await client.search(query=query_string, limit=limit)
    display_search_results(response, display_limit=limit)


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
            error_console.print(
                "[bold red]API key required for 'api' backend.[/bold red]\n"
                "Set LEANEXPLORE_API_KEY or use --api-key option."
            )
            raise typer.Abort()
        if api_key_override:
            command_parts.extend(["--api-key", api_key_override])

    subprocess.run(command_parts, check=False)


if __name__ == "__main__":
    app()
