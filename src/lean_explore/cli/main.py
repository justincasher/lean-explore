# src/lean_explore/cli/main.py

"""Command-Line Interface for Lean Explore.

Provides commands to configure the CLI, search for Lean statement groups
via the remote API, interact with AI agents, manage local data, and other utilities.
"""

import asyncio
import functools # For the decorator
import typer
from typing import List, Optional
import textwrap
import subprocess # For running the MCP server
import sys # For sys.executable

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import httpx

from lean_explore.api.client import Client as APIClient
from lean_explore.shared.models.api import APISearchResponse, APISearchResultItem, APICitationsResponse
from lean_explore.cli import config_utils
from lean_explore.cli import data_commands # For data management subcommands
from lean_explore import defaults
# Import the specific command function and its async wrapper from agent.py
from .agent import agent_chat_command, typer_async as agent_typer_async

# Initialize Typer app and Rich console
app = typer.Typer(
    name="leanexplore",
    help="A CLI tool to explore and search Lean mathematical libraries.",
    add_completion=False,
    rich_markup_mode="markdown" # Enables rich markup in help text
)
configure_app = typer.Typer(name="configure", help="Configure leanexplore CLI settings.")
app.add_typer(configure_app)

mcp_app = typer.Typer(name="mcp", help="Manage and run the Model Context Protocol (MCP) server.")
app.add_typer(mcp_app)

# Add the data_commands.app as a subcommand group named "data"
app.add_typer(
    data_commands.app,
    name="data",
    help="Manage local data toolchains (e.g., download, list, select)."
)

# Register the agent_chat_command directly on the main app as "chat"
# The agent_chat_command is already decorated with @typer_async in agent.py
app.command("chat", help="Interact with an AI agent using Lean Explore tools.")(agent_chat_command)


console = Console()

# Content width for panels.
PANEL_CONTENT_WIDTH = 80


@configure_app.command("lean-key")
def configure_lean_explore_api_key(
    api_key: Optional[str] = typer.Option(
        None,
        prompt="Please enter your Lean Explore API key",
        help="Your personal API key for accessing Lean Explore services.",
        hide_input=True,
        confirmation_prompt=True,
    )
):
    """
    Configure and save your Lean Explore API key.
    """
    if not api_key:
        console.print("[bold red]Lean Explore API key cannot be empty.[/bold red]")
        raise typer.Abort()

    if config_utils.save_api_key(api_key):
        config_path = config_utils.get_config_file_path()
        console.print(f"[bold green]Lean Explore API key saved successfully to: {config_path}[/bold green]")
    else:
        console.print("[bold red]Failed to save Lean Explore API key. Check logs or permissions.[/bold red]")
        raise typer.Abort()


@configure_app.command("openai-key")
def configure_openai_api_key(
    api_key: Optional[str] = typer.Option(
        None,
        prompt="Please enter your OpenAI API key",
        help="Your personal API key for OpenAI services (e.g., GPT-4).",
        hide_input=True,
        confirmation_prompt=True,
    )
):
    """
    Configure and save your OpenAI API key.

    This key is used by agent functionalities that leverage OpenAI models.
    """
    if not api_key:
        console.print("[bold red]OpenAI API key cannot be empty.[/bold red]")
        raise typer.Abort()

    if config_utils.save_openai_api_key(api_key):
        config_path = config_utils.get_config_file_path()
        console.print(f"[bold green]OpenAI API key saved successfully to: {config_path}[/bold green]")
    else:
        console.print("[bold red]Failed to save OpenAI API key. Check logs or permissions.[/bold red]")
        raise typer.Abort()


def _get_api_client() -> Optional[APIClient]:
    """
    Loads Lean Explore API key and initializes the APIClient.

    Returns:
        APIClient instance if key is found, None otherwise.
    """
    api_key = config_utils.load_api_key()
    if not api_key:
        config_path = config_utils.get_config_file_path()
        console.print(
            "[bold yellow]Lean Explore API key not configured. Please run:[/bold yellow]\n"
            f"  `leanexplore configure lean-key`\n"
            f"Your API key will be stored in: {config_path}"
        )
        return None
    return APIClient(api_key=api_key)


def _format_text_for_fixed_panel(text_content: Optional[str], width: int) -> str:
    """Wraps text and pads lines to ensure fixed content width for a Panel."""
    if not text_content:
        return " " * width

    final_output_lines = []
    paragraphs = text_content.split('\n\n')

    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip() and i < len(paragraphs) -1 :
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
                break_on_hyphens=True
            )
            if not wrapped_segments:
                final_output_lines.append(" " * width)
            else:
                for segment in wrapped_segments:
                    final_output_lines.append(segment.ljust(width))

        if i < len(paragraphs) - 1 and (paragraph.strip() or (not paragraph.strip() and not lines_in_paragraph)):
            final_output_lines.append(" " * width)

    if not final_output_lines and text_content.strip():
        return " " * width

    return "\n".join(final_output_lines)


def _display_search_results(response: APISearchResponse, display_limit: int = 5):
    """Displays search results using fixed-width Panels for each item."""
    console.print(Panel(f"[bold cyan]Search Query:[/bold cyan] {response.query}", expand=False, border_style="dim"))
    if response.packages_applied:
        console.print(f"[bold cyan]Package Filters:[/bold cyan] {', '.join(response.packages_applied)}")

    num_results_to_show = min(len(response.results), display_limit)
    console.print(
        f"Showing {num_results_to_show} of {response.count} "
        f"(out of {response.total_candidates_considered} candidates considered by server). "
        f"Time: {response.processing_time_ms}ms"
    )

    if not response.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print("")

    for i, item in enumerate(response.results):
        if i >= display_limit:
            break

        lean_name = item.primary_declaration.lean_name if item.primary_declaration else "N/A"

        console.rule(f"[bold]Result {i+1}[/bold]", style="dim")
        console.print(f"[bold cyan]ID:[/bold cyan] [dim]{item.id}[/dim]")
        console.print(f"[bold cyan]Name:[/bold cyan] {lean_name}")
        console.print(f"[bold cyan]File:[/bold cyan] [green]{item.source_file}[/green]:[dim]{item.range_start_line}[/dim]")

        code_to_display = item.display_statement_text or item.statement_text
        if code_to_display:
            formatted_code = _format_text_for_fixed_panel(code_to_display, PANEL_CONTENT_WIDTH)
            console.print(Panel(formatted_code, title="[bold green]Code[/bold green]", border_style="green", expand=False, padding=(0,1)))

        if item.docstring:
            formatted_doc = _format_text_for_fixed_panel(item.docstring, PANEL_CONTENT_WIDTH)
            console.print(Panel(formatted_doc, title="[bold blue]Docstring[/bold blue]", border_style="blue", expand=False, padding=(0,1)))

        if item.informal_description:
            formatted_informal = _format_text_for_fixed_panel(item.informal_description, PANEL_CONTENT_WIDTH)
            console.print(Panel(formatted_informal, title="[bold magenta]Informal Description[/bold magenta]", border_style="magenta", expand=False, padding=(0,1)))
        elif not item.docstring and not code_to_display:
            console.print(f"[dim]No further textual details (docstring, informal description, code) available for this item.[/dim]")

        if i < num_results_to_show - 1:
             console.print("")

    console.rule(style="dim")
    if len(response.results) > num_results_to_show:
        console.print(f"...and {len(response.results) - num_results_to_show} more results received from server but not shown due to limit.")
    elif response.count > len(response.results):
         console.print(f"...and {response.count - len(response.results)} more results available on server.")

@app.command("search")
@agent_typer_async
async def search_command(
    query_string: str = typer.Argument(..., help="The search query string."),
    package: Optional[List[str]] = typer.Option(None, "--package", "-p", help="Filter by package name(s). Can be used multiple times."),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of search results to display.")
):
    """
    Search for Lean statement groups using the Lean Explore API.
    """
    client = _get_api_client()
    if not client:
        raise typer.Exit(code=1)

    console.print(f"Searching for: '{query_string}'...")
    try:
        response = await client.search(query=query_string, package_filters=package)
        _display_search_results(response, display_limit=limit)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            console.print(f"[bold red]API Error {e.response.status_code}: Unauthorized. Your API key might be invalid or expired.[/bold red]")
            console.print("Please reconfigure your API key using: `leanexplore configure lean-key`")
        else:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
                console.print(f"[bold red]API Error {e.response.status_code}: {error_detail}[/bold red]")
            except Exception:
                console.print(f"[bold red]API Error {e.response.status_code}: {e.response.text}[/bold red]")
        raise typer.Exit(code=1)
    except httpx.RequestError as e:
        console.print(f"[bold red]Network Error: Could not connect to the API. {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("get")
@agent_typer_async
async def get_by_id_command(
    group_id: int = typer.Argument(..., help="The ID of the statement group to retrieve.")
):
    """
    Get detailed information about a specific statement group by its ID.
    """
    client = _get_api_client()
    if not client:
        raise typer.Exit(code=1)

    console.print(f"Fetching statement group ID: {group_id}...")
    try:
        item = await client.get_by_id(group_id)
        if item:
            console.print(Panel(f"[bold green]Statement Group ID: {item.id}[/bold green]", expand=False, border_style="dim"))
            console.print(f"  [bold cyan]Lean Name:[/bold cyan] {item.primary_declaration.lean_name if item.primary_declaration else 'N/A'}")
            console.print(f"  [bold cyan]Source File:[/bold cyan] [green]{item.source_file}[/green]:[dim]{item.range_start_line}[/dim]")

            if item.statement_text:
                formatted_stmt_text = _format_text_for_fixed_panel(item.statement_text, PANEL_CONTENT_WIDTH)
                console.print(Panel(formatted_stmt_text, title="[bold green]Statement Text[/bold green]", border_style="green", expand=False, padding=(0,1)))

            if item.display_statement_text and item.display_statement_text != item.statement_text:
                 formatted_display_stmt_text = _format_text_for_fixed_panel(item.display_statement_text, PANEL_CONTENT_WIDTH)
                 console.print(Panel(formatted_display_stmt_text, title="[bold green]Display Statement Text (Formatted)[/bold green]", border_style="green", expand=False, padding=(0,1)))

            if item.docstring:
                formatted_docstring = _format_text_for_fixed_panel(item.docstring, PANEL_CONTENT_WIDTH)
                console.print(Panel(formatted_docstring, title="[bold blue]Docstring[/bold blue]", border_style="blue", expand=False, padding=(0,1)))

            if item.informal_description:
                formatted_informal = _format_text_for_fixed_panel(item.informal_description, PANEL_CONTENT_WIDTH)
                console.print(Panel(formatted_informal, title="[bold magenta]Informal Description[/bold magenta]", border_style="magenta", expand=False, padding=(0,1)))

        else:
            console.print(f"[yellow]Statement group with ID {group_id} not found.[/yellow]")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            console.print(f"[bold red]API Error {e.response.status_code}: Unauthorized. Your API key might be invalid or expired.[/bold red]")
        else:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
                console.print(f"[bold red]API Error {e.response.status_code}: {error_detail}[/bold red]")
            except Exception:
                console.print(f"[bold red]API Error {e.response.status_code}: {e.response.text}[/bold red]")
        raise typer.Exit(code=1)
    except httpx.RequestError as e:
        console.print(f"[bold red]Network Error: Could not connect to the API. {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("dependencies")
@agent_typer_async
async def get_dependencies_command(
    group_id: int = typer.Argument(..., help="The ID of the statement group to get dependencies for.")
):
    """
    Get dependencies (citations) for a specific statement group by its ID.
    """
    client = _get_api_client()
    if not client:
        raise typer.Exit(code=1)

    console.print(f"Fetching dependencies for statement group ID: {group_id}...")
    try:
        response = await client.get_dependencies(group_id)
        if response:
            console.print(Panel(f"[bold green]Citations for Statement Group ID: {response.source_group_id}[/bold green]", expand=False, border_style="dim"))
            console.print(f"Found {response.count} direct citations.")

            if response.citations:
                citation_table = Table(show_header=True, header_style="bold magenta")
                citation_table.add_column("ID", style="dim", width=6)
                citation_table.add_column("Cited Lean Name", width=40)
                citation_table.add_column("File", style="green")
                citation_table.add_column("Line", style="dim")

                for item in response.citations:
                    lean_name = item.primary_declaration.lean_name if item.primary_declaration else "N/A"
                    citation_table.add_row(
                        str(item.id),
                        lean_name,
                        item.source_file,
                        str(item.range_start_line)
                    )
                console.print(citation_table)
            else:
                console.print("[yellow]No citations found for this group.[/yellow]")
        else:
            console.print(f"[yellow]Statement group with ID {group_id} not found or no citations data available.[/yellow]")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            console.print(f"[bold red]API Error {e.response.status_code}: Unauthorized. Your API key might be invalid or expired.[/bold red]")
        else:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
                console.print(f"[bold red]API Error {e.response.status_code}: {error_detail}[/bold red]")
            except Exception:
                console.print(f"[bold red]API Error {e.response.status_code}: {e.response.text}[/bold red]")
        raise typer.Exit(code=1)
    except httpx.RequestError as e:
        console.print(f"[bold red]Network Error: Could not connect to the API. {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise typer.Exit(code=1)

# The placeholder download-db command is removed as its functionality
# will be covered by the `leanexplore data fetch` command.

@mcp_app.command("serve")
def mcp_serve_command(
    backend: str = typer.Option(
        "api",
        "--backend",
        "-b",
        help="Backend to use for the MCP server: 'api' or 'local'. Default is 'api'.",
        case_sensitive=False,
        show_choices=True
    ),
    api_key_override: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key to use if backend is 'api'. Overrides stored key. Not used for 'local' backend."
    )
):
    """
    Launch the Lean Explore MCP (Model Context Protocol) server.

    The server communicates via stdio and provides Lean search functionalities
    as MCP tools. The actual checks for local data presence or API key validity
    are handled by the 'lean_explore.mcp.server' module when it starts.
    """
    command_parts = [sys.executable, "-m", "lean_explore.mcp.server", "--backend", backend.lower()]

    if backend.lower() == "api":
        # API key will be loaded by mcp.server or passed if overridden
        # We still check here to provide immediate feedback if --api-key is needed but not stored.
        effective_lean_explore_api_key = api_key_override or config_utils.load_api_key()
        if not effective_lean_explore_api_key:
            console.print(
                "[bold red]Lean Explore API key is required for 'api' backend.[/bold red]\n"
                "Please configure it using `leanexplore configure lean-key` "
                "or provide it with the `--api-key` option for this command."
            )
            raise typer.Abort()
        # Pass the override if provided; otherwise, mcp.server will load it.
        if api_key_override:
            command_parts.extend(["--api-key", api_key_override])
    elif backend.lower() == "local":
        # The mcp.server module will now handle checks for local data existence
        # and provide user guidance. No explicit checks needed here in main.py.
        logger_name = "lean_explore.cli.main" # Use a more specific logger if available, or general console
        console.print(f"[dim]Attempting to start MCP server with 'local' backend. "
                      f"The server will verify local data availability.[/dim]")
    else:
        # This case should ideally not be reached due to Typer's choice validation,
        # but as a safeguard.
        console.print(f"[bold red]Invalid backend: '{backend}'. Must be 'api' or 'local'.[/bold red]")
        raise typer.Abort()

    console.print(f"[green]Launching MCP server subprocess with '{backend}' backend...[/green]")
    console.print("[dim]The server will now take over stdio. To stop it, the connected MCP client should disconnect, or you may need to manually terminate this process (e.g., Ctrl+C if no client is managing it).[/dim]")

    try:
        process_result = subprocess.run(command_parts, check=False)
        if process_result.returncode != 0:
            # mcp.server should have printed its own detailed error message
            # before exiting with a non-zero code.
            console.print(f"[bold red]MCP server subprocess exited with code: {process_result.returncode}. "
                          f"Check server logs above for details.[/bold red]")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Could not find Python interpreter '{sys.executable}' or the MCP server module 'lean_explore.mcp.server'.[/bold red]")
        console.print("Please ensure the package is installed correctly and `python -m lean_explore.mcp.server` is runnable.")
    except Exception as e:
        console.print(f"[bold red]An error occurred while trying to launch or run the MCP server: {e}[/bold red]")


if __name__ == "__main__":
    app()