"""Command-Line Interface for Lean Explore.

Provides commands to search for Lean declarations via the remote API,
interact with AI agents, and manage local data.
"""

import asyncio
import functools
import os
import subprocess
import sys
import textwrap

import typer
from rich.console import Console
from rich.panel import Panel

from lean_explore.api import ApiClient
from lean_explore.cli import data_commands
from lean_explore.search.types import SearchResponse


def async_command(f):
    """Decorator to run async Typer commands."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


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

PANEL_CONTENT_WIDTH = 80


def _get_api_client() -> ApiClient | None:
    """Loads Lean Explore API key from environment and initializes the ApiClient."""
    api_key = os.getenv("LEANEXPLORE_API_KEY")
    if not api_key:
        error_console.print(
            "[bold yellow]Lean Explore API key not found.[/bold yellow]\n"
            "Please set the LEANEXPLORE_API_KEY environment variable."
        )
        return None
    return ApiClient(api_key=api_key)


def _format_text_for_fixed_panel(text_content: str | None, width: int) -> str:
    """Wraps text and pads lines to ensure fixed content width for a Panel."""
    if not text_content:
        return " " * width

    final_output_lines = []
    paragraphs = text_content.split("\n\n")

    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip() and i < len(paragraphs) - 1:
            final_output_lines.append(" " * width)
            continue

        lines_in_paragraph = paragraph.splitlines()
        if not lines_in_paragraph and paragraph.strip() == "":
            final_output_lines.append(" " * width)
            continue
        if not lines_in_paragraph and not paragraph:
            final_output_lines.append(" " * width)
            continue

        for line in lines_in_paragraph:
            if not line.strip():
                final_output_lines.append(" " * width)
                continue

            wrapped_segments = textwrap.wrap(
                line,
                width=width,
                replace_whitespace=True,
                drop_whitespace=True,
                break_long_words=True,
                break_on_hyphens=True,
            )
            if not wrapped_segments:
                final_output_lines.append(" " * width)
            else:
                for segment in wrapped_segments:
                    final_output_lines.append(segment.ljust(width))

        if i < len(paragraphs) - 1 and (
            paragraph.strip() or (not paragraph.strip() and not lines_in_paragraph)
        ):
            final_output_lines.append(" " * width)

    if not final_output_lines and text_content.strip():
        return " " * width

    return "\n".join(final_output_lines)


def _display_search_results(response: SearchResponse, display_limit: int = 5):
    """Displays search results using fixed-width Panels for each item."""
    console.print(
        Panel(
            f"[bold cyan]Search Query:[/bold cyan] {response.query}",
            expand=False,
            border_style="dim",
        )
    )

    num_results_to_show = min(len(response.results), display_limit)
    time_info = (
        f"Time: {response.processing_time_ms}ms" if response.processing_time_ms else ""
    )
    console.print(
        f"Showing {num_results_to_show} of {response.count} results. {time_info}"
    )

    if not response.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print("")

    for i, item in enumerate(response.results):
        if i >= display_limit:
            break

        console.rule(f"[bold]Result {i + 1}[/bold]", style="dim")
        console.print(f"[bold cyan]ID:[/bold cyan] [dim]{item.id}[/dim]")
        console.print(f"[bold cyan]Name:[/bold cyan] {item.name}")
        console.print(f"[bold cyan]Module:[/bold cyan] [green]{item.module}[/green]")
        source_formatted = (
            f"[bold cyan]Source:[/bold cyan] "
            f"[link={item.source_link}]{item.source_link}[/link]"
        )
        console.print(source_formatted)

        if item.source_text:
            formatted_code = _format_text_for_fixed_panel(
                item.source_text, PANEL_CONTENT_WIDTH
            )
            console.print(
                Panel(
                    formatted_code,
                    title="[bold green]Code[/bold green]",
                    border_style="green",
                    expand=False,
                    padding=(0, 1),
                )
            )

        if item.docstring:
            formatted_doc = _format_text_for_fixed_panel(
                item.docstring, PANEL_CONTENT_WIDTH
            )
            console.print(
                Panel(
                    formatted_doc,
                    title="[bold blue]Docstring[/bold blue]",
                    border_style="blue",
                    expand=False,
                    padding=(0, 1),
                )
            )

        if item.informalization:
            formatted_informal = _format_text_for_fixed_panel(
                item.informalization, PANEL_CONTENT_WIDTH
            )
            console.print(
                Panel(
                    formatted_informal,
                    title="[bold magenta]Informalization[/bold magenta]",
                    border_style="magenta",
                    expand=False,
                    padding=(0, 1),
                )
            )

        if i < num_results_to_show - 1:
            console.print("")

    console.rule(style="dim")
    if len(response.results) > num_results_to_show:
        console.print(
            f"...and {len(response.results) - num_results_to_show} more results "
            "received but not shown due to limit."
        )


@app.command("search")
@async_command
async def search_command(
    query_string: str = typer.Argument(..., help="The search query string."),
    limit: int = typer.Option(
        5, "--limit", "-n", help="Number of search results to display."
    ),
):
    """Search for Lean declarations using the Lean Explore API."""
    client = _get_api_client()
    if not client:
        raise typer.Exit(code=1)

    console.print(f"Searching for: '{query_string}'...")
    response = await client.search(query=query_string, limit=limit)
    _display_search_results(response, display_limit=limit)


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
