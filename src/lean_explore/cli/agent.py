# src/lean_explore/cli/agent.py

"""Command-line interface logic for interacting with an AI agent.

This module provides the agent_chat_command and supporting functions for the chat
interaction, intended to be registered with a main Typer application.
"""

import asyncio
import functools
import logging
import os
import pathlib
import shutil
import sys
import textwrap
from typing import Optional

import typer

# Ensure 'openai-agents' is installed
try:
    from agents import Agent, Runner
    from agents.exceptions import AgentsException, UserError
    from agents.mcp import MCPServerStdio
except ImportError:
    print(
        "Fatal Error: The 'openai-agents' library or its expected exceptions "
        "are not installed/found. Please install 'openai-agents' correctly "
        "(e.g., 'pip install openai-agents')",
        file=sys.stderr,
    )
    raise typer.Exit(code=1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print(
        "Fatal Error: The 'rich' library is not installed/found. "
        "Please install 'rich' (e.g., 'pip install rich')",
        file=sys.stderr,
    )
    raise typer.Exit(code=1)


# Attempt to import your project's config_utils for API key loading
config_utils_imported = False
try:
    from lean_explore.cli import config_utils

    config_utils_imported = True
except ImportError:
    # BasicConfig for this warning, actual command configures logger later
    logging.basicConfig(level=logging.WARNING)
    logging.warning(
        "Could not import 'lean_explore.cli.config_utils'. "
        "Automatic loading/saving of stored API keys will be disabled. "
        "Ensure 'lean_explore' package is installed correctly and accessible "
        "in PYTHONPATH (e.g., by running 'pip install -e .' from the project root)."
    )

    class MockConfigUtils:
        """A mock for config_utils if it cannot be imported."""

        def load_api_key(self) -> Optional[str]:
            """Loads Lean Explore API key."""
            return None

        def load_openai_api_key(self) -> Optional[str]:
            """Loads OpenAI API key."""
            return None

        def save_api_key(self, api_key: str) -> bool:
            """Saves Lean Explore API key."""
            return False

        def save_openai_api_key(self, api_key: str) -> bool:
            """Saves OpenAI API key."""
            return False

    config_utils = MockConfigUtils()


# --- Async Wrapper for Typer Commands ---
def typer_async(f):
    """A decorator to allow Typer commands to be async functions.

    It wraps the async function in `asyncio.run()`.

    Args:
        f: The asynchronous function to wrap.

    Returns:
        The wrapped function that can be called synchronously.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# --- ANSI Color Codes ---
class Colors:
    """ANSI color codes for terminal output for enhanced readability."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


agent_cli_app = typer.Typer(
    name="agent_cli_utils",
    help="Utilities related to AI agent interactions.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)
console = Console()
CHAT_CONTENT_WIDTH = 76  # Consistent with main.py's PANEL_CONTENT_WIDTH


def _format_chat_text_for_panel(text_content: str, width: int) -> str:
    """Wraps text for chat display, padding lines to fill panel width.

    This function processes text line by line, wraps content that exceeds the
    specified width, and pads each resulting line with spaces to ensure a
    uniform block appearance within a Rich Panel. Empty lines in the input
    are preserved as padded blank lines.

    Args:
        text_content: The text content to wrap.
        width: The target width for text wrapping and padding.

    Returns:
        A string with wrapped and padded text.
    """
    if not text_content.strip():
        # For empty or whitespace-only input, provide a single padded blank line
        return " " * width

    input_lines = text_content.splitlines()
    output_panel_lines = []

    if not input_lines:
        return " " * width  # Should be caught by strip(), but safeguard

    for line_text in input_lines:
        if not line_text.strip():  # An intentionally blank line in the input
            output_panel_lines.append(" " * width)
        else:
            wrapped_segments = textwrap.wrap(
                line_text,
                width=width,
                replace_whitespace=True,  # Collapse multiple spaces within line
                drop_whitespace=True,  # Remove leading/trailing space from segments
                break_long_words=True,  # Break words that exceed width
                break_on_hyphens=True,  # Allow breaking on hyphens
            )
            if not wrapped_segments:
                # If wrapping a non-blank line results in nothing (e.g. only whitespace)
                output_panel_lines.append(" " * width)
            else:
                for segment in wrapped_segments:
                    output_panel_lines.append(segment.ljust(width))

    if not output_panel_lines:
        # Fallback if all processing led to no lines (e.g. input was "\n \n")
        return " " * width

    return "\n".join(output_panel_lines)


def _handle_server_connection_error(
    error: Exception,
    lean_backend_type: str,
    debug_mode: bool,
    context: str = "server startup",
):
    """Handles MCP server connection errors by logging and user-friendly messages."""
    logger.error(
        f"CRITICAL: Error during MCP {context}: {type(error).__name__}: {error}",
        exc_info=debug_mode,
    )

    error_str = str(error).lower()
    is_timeout_error = "timed out" in error_str or "timeout" in error_str

    if is_timeout_error:
        console.print(
            Text.from_markup(
                "[bold red]Error: The Lean Explore server failed to start "
                "or respond promptly.[/bold red]"
            ),
            stderr=True,
        )
        if lean_backend_type == "local":
            console.print(
                Text.from_markup(
                    "[yellow]This often occurs with the 'local' backend due to missing "
                    "or corrupted data files.[/yellow]"
                ),
                stderr=True,
            )
            console.print(
                Text.from_markup("[yellow]Please try the following steps:[/yellow]"),
                stderr=True,
            )
            console.print(
                Text.from_markup(
                    "[yellow]  1. Run 'leanexplore data fetch' to download or update "
                    "the required data.[/yellow]"
                ),
                stderr=True,
            )
            console.print(
                Text.from_markup("[yellow]  2. Try this chat command again.[/yellow]"),
                stderr=True,
            )
            console.print(
                Text.from_markup(
                    "[yellow]  3. If the problem persists, run 'leanexplore mcp serve "
                    "--backend local --log-level DEBUG' directly in another terminal "
                    "to see detailed server startup logs.[/yellow]"
                ),
                stderr=True,
            )
        else:  # api backend or other cases
            console.print(
                Text.from_markup(
                    "[yellow]Please check your network connection and ensure the API "
                    "server is accessible.[/yellow]"
                ),
                stderr=True,
            )
    elif isinstance(error, UserError):
        console.print(
            Text.from_markup(
                f"[bold red]Error: SDK usage problem during {context}: "
                f"{error}[/bold red]"
            ),
            stderr=True,
        )
    elif isinstance(error, AgentsException):
        console.print(
            Text.from_markup(
                f"[bold red]Error: An SDK error occurred during {context}: "
                f"{error}[/bold red]"
            ),
            stderr=True,
        )
    else:
        console.print(
            Text.from_markup(
                f"[bold red]An unexpected error occurred during {context}: "
                f"{error}[/bold red]"
            ),
            stderr=True,
        )

    if debug_mode:
        console.print(
            Text.from_markup(
                f"[magenta]Error Details ({type(error).__name__}): {error}[/magenta]"
            ),
            stderr=True,
        )
    raise typer.Exit(code=1)


# --- Core Agent Logic ---
async def _run_agent_session(
    lean_backend_type: str,
    lean_explore_api_key_arg: Optional[str] = None,
    debug_mode: bool = False,
    log_level_for_mcp_server: str = "WARNING",
):
    """Internal function to set up and run the OpenAI Agent session.

    Args:
        lean_backend_type: The backend ('api' or 'local') for the Lean Explore server.
        lean_explore_api_key_arg: API key for Lean Explore (if 'api' backend),
                                  already resolved from CLI arg or ENV.
        debug_mode: If True, enables more verbose logging for this client and
                    the MCP server.
        log_level_for_mcp_server: The log level to pass to the MCP server.
    """
    internal_server_script_path = (
        pathlib.Path(__file__).parent.parent / "mcp" / "server.py"
    ).resolve()

    # --- OpenAI API Key Acquisition ---
    openai_api_key = None
    if config_utils_imported:
        logger.debug("Attempting to load OpenAI API key from CLI configuration...")
        try:
            openai_api_key = config_utils.load_openai_api_key()
            if openai_api_key:
                logger.info("Loaded OpenAI API key from CLI configuration.")
            else:
                logger.debug("No OpenAI API key found in CLI configuration.")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                f"Error loading OpenAI API key from CLI configuration: {e}",
                exc_info=debug_mode,
            )

    if not openai_api_key:
        console.print(
            Text.from_markup(
                "[yellow]OpenAI API key not found in configuration.[/yellow]"
            )
        )
        openai_api_key = typer.prompt(
            "Please enter your OpenAI API key", hide_input=True
        )
        if not openai_api_key:
            console.print(
                Text.from_markup(
                    "[bold red]OpenAI API key cannot be empty. Exiting.[/bold red]"
                ),
                stderr=True,
            )
            raise typer.Exit(code=1)
        logger.info("Using OpenAI API key provided via prompt.")
        if config_utils_imported:
            if typer.confirm(
                "Would you like to save this OpenAI API key for future use?"
            ):
                if config_utils.save_openai_api_key(openai_api_key):
                    console.print(
                        Text.from_markup(
                            "[green]OpenAI API key saved successfully.[/green]"
                        )
                    )
                else:
                    console.print(
                        Text.from_markup("[red]Failed to save OpenAI API key.[/red]"),
                        stderr=True,
                    )
            else:
                console.print("OpenAI API key will be used for this session only.")
        else:
            console.print(
                Text.from_markup(
                    "[yellow]Note: config_utils not available, "
                    "OpenAI API key cannot be saved.[/yellow]"
                )
            )

    os.environ["OPENAI_API_KEY"] = openai_api_key

    # --- Lean Explore Server Script and Executable Validation ---
    if not internal_server_script_path.exists():
        error_msg = (
            "Lean Explore MCP server script not found at calculated path: "
            f"{internal_server_script_path}"
        )
        logger.error(error_msg)
        console.print(
            Text.from_markup(f"[bold red]Error: {error_msg}[/bold red]"), stderr=True
        )
        raise typer.Exit(code=1)

    python_executable = sys.executable
    if not python_executable or not shutil.which(python_executable):
        error_msg = (
            f"Python executable '{python_executable}' not found or not executable. "
            "Ensure Python is correctly installed and in your PATH."
        )
        logger.error(error_msg)
        console.print(
            Text.from_markup(f"[bold red]Error: {error_msg}[/bold red]"), stderr=True
        )
        raise typer.Exit(code=1)

    # --- Lean Explore API Key Acquisition (if API backend) ---
    effective_lean_api_key = lean_explore_api_key_arg
    if lean_backend_type == "api":
        if not effective_lean_api_key and config_utils_imported:
            logger.debug(
                "Lean Explore API key not provided via CLI option or ENV. "
                "Attempting to load from CLI configuration..."
            )
            try:
                stored_key = config_utils.load_api_key()
                if stored_key:
                    effective_lean_api_key = stored_key
                    logger.debug(
                        "Successfully loaded Lean Explore API key from "
                        "CLI configuration."
                    )
                else:
                    logger.debug("No Lean Explore API key found in CLI configuration.")
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    f"Error loading Lean Explore API key from CLI configuration: {e}",
                    exc_info=debug_mode,
                )

        if not effective_lean_api_key:
            console.print(
                Text.from_markup(
                    "[yellow]Lean Explore API key is required for the 'api' backend "
                    "and was not found through CLI option, environment variable, "
                    "or configuration.[/yellow]"
                )
            )
            effective_lean_api_key = typer.prompt(
                "Please enter your Lean Explore API key", hide_input=True
            )
            if not effective_lean_api_key:
                console.print(
                    Text.from_markup(
                        "[bold red]Lean Explore API key cannot be empty for 'api' "
                        "backend. Exiting.[/bold red]"
                    ),
                    stderr=True,
                )
                raise typer.Exit(code=1)
            logger.info("Using Lean Explore API key provided via prompt.")
            if config_utils_imported:
                if typer.confirm(
                    "Would you like to save this Lean Explore API key for future use?"
                ):
                    if config_utils.save_api_key(effective_lean_api_key):
                        console.print(
                            Text.from_markup(
                                "[green]Lean Explore API key saved successfully."
                                "[/green]"
                            )
                        )
                    else:
                        console.print(
                            Text.from_markup(
                                "[red]Failed to save Lean Explore API key.[/red]"
                            ),
                            stderr=True,
                        )
                else:
                    console.print(
                        "Lean Explore API key will be used for this session only."
                    )
            else:
                console.print(
                    Text.from_markup(
                        "[yellow]Note: config_utils not available, "
                        "Lean Explore API key "
                        "cannot be saved.[/yellow]"
                    )
                )

    # --- MCP Server Setup ---
    mcp_server_args = [
        str(internal_server_script_path),
        "--backend",
        lean_backend_type,
        "--log-level",
        log_level_for_mcp_server,
    ]
    if lean_backend_type == "api" and effective_lean_api_key:
        mcp_server_args.extend(["--api-key", effective_lean_api_key])

    lean_explore_mcp_server = MCPServerStdio(
        name="LeanExploreSearchServer",
        params={
            "command": python_executable,
            "args": mcp_server_args,
            "cwd": str(internal_server_script_path.parent),
        },
        cache_tools_list=True,
        client_session_timeout_seconds=10.0,
    )

    # --- Agent Interaction Loop ---
    try:
        async with lean_explore_mcp_server as server_instance:
            logger.debug(
                f"MCP server '{server_instance.name}' connection initiated. "
                "Listing tools..."
            )
            tools = []
            try:
                tools = await server_instance.list_tools()
                if not tools or not any(tools):
                    logger.warning(
                        "MCP Server connected but reported no tools. "
                        "Agent may lack expected capabilities."
                    )
                else:
                    logger.debug(
                        f"Available tools from {server_instance.name}: "
                        f"{[tool.name for tool in tools]}"
                    )
            except (UserError, AgentsException, Exception) as e_list_tools:
                _handle_server_connection_error(
                    e_list_tools, lean_backend_type, debug_mode, context="tool listing"
                )

            agent_model = "gpt-4.1"
            agent_object_name = "Assistant"
            agent_display_name = (
                f"{Colors.BOLD}{Colors.GREEN}{agent_object_name}{Colors.ENDC}"
            )

            agent = Agent(
                name=agent_object_name,
                model=agent_model,
                instructions=(
                    "You are a CLI assistant for searching a Lean 4 mathematical "
                    "library.\n"
                    "**Goal:** Find relevant Lean statements, understand them "
                    "(including dependencies), and explain them conversationally "
                    "to the user.\n"
                    "**Output:** CLI-friendly (plain text, simple lists). "
                    "NO complex Markdown/LaTeX.\n\n"
                    "**Packages:** Use exact top-level names for filters (Batteries, "
                    "Init, Lean, Mathlib, PhysLean, Std). Map subpackage mentions "
                    "to top-level (e.g., 'Mathlib.Analysis' -> 'Mathlib').\n\n"
                    "**Core Workflow:**\n"
                    "1.  **Search & Analyze:**\n"
                    "    * Execute multiple distinct `search` queries for each user "
                    "request (e.g., using full statements, rephrasing). Set `limit` "
                    ">= 10 for each search.\n"
                    "    * From all search results, select the statement(s) most "
                    "helpful to the user.\n"
                    "    * For each selected statement, **MUST** use "
                    "`get_dependencies` to understand its context before "
                    "explaining.\n\n"
                    "2.  **Explain Results (Conversational & CLI-Friendly):**\n"
                    "    * Briefly state your search approach (e.g., 'I looked into X "
                    "in Mathlib...').\n"
                    "    * For each selected statement:\n"
                    "        * Introduce it (e.g., Lean name: "
                    "`primary_declaration.lean_name`).\n"
                    "        * Explain its meaning (use `docstring`, "
                    "`informal_description`, `statement_text`).\n"
                    "        * Provide the full Lean code (`statement_text`).\n"
                    "        * Explain key dependencies (what they are, their role, "
                    "using `statement_text` or `display_statement_text` from "
                    "`get_dependencies` output).\n"
                    "3.  **Specific User Follow-ups (If Asked):**\n"
                    "    * **`get_by_id`:** For a specific ID, provide: ID, Lean name, "
                    "statement text, source/line, docstring, informal description "
                    "(structured CLI format).\n"
                    "    * **`get_dependencies` (Direct Request):** For all "
                    "dependencies of an ID, list: ID, Lean name, statement "
                    "text/summary. State total count.\n\n"
                    "Always be concise, helpful, and clear."
                ),
                mcp_servers=[server_instance],
            )
            console.print(
                Text.from_markup(
                    "[bold]Lean Search Assistant[/bold] (powered by [green]"
                    f"{agent_model}[/green] and [green]{server_instance.name}[/green]) "
                    "is ready."
                )
            )
            console.print(
                "Ask me to search for Lean statements (e.g., 'find definitions "
                "of a scheme')."
            )
            if not debug_mode and lean_backend_type == "local":
                console.print(
                    Text.from_markup(
                        "[yellow]Note: The local search server might print startup "
                        "logs. "
                        "For a quieter experience, use --debug to see detailed logs or "
                        "ensure the server's default log level is WARNING.[/yellow]"
                    )
                )
            console.print("Type 'exit' or 'quit' to end the session.")
            console.print()

            while True:
                try:
                    user_styled_name = typer.style(
                        "You", fg=typer.colors.BLUE, bold=True
                    )
                    user_input = typer.prompt(
                        user_styled_name, default="", prompt_suffix=": "
                    ).strip()

                    if user_input.lower() in ["exit", "quit"]:
                        logger.debug("Exiting chat loop.")
                        break
                    if not user_input:
                        continue

                    formatted_user_input = _format_chat_text_for_panel(
                        user_input, CHAT_CONTENT_WIDTH
                    )
                    console.print(
                        Panel(
                            formatted_user_input,
                            title="You",
                            border_style="blue",
                            title_align="left",
                            expand=False,
                        )
                    )
                    console.print()

                    thinking_line_str_ansi = (
                        f"{agent_display_name}: {Colors.YELLOW}Thinking...{Colors.ENDC}"
                    )
                    sys.stdout.write(thinking_line_str_ansi)
                    sys.stdout.flush()

                    result = await Runner.run(starting_agent=agent, input=user_input)

                    thinking_len_to_clear = Text.from_ansi(
                        thinking_line_str_ansi
                    ).cell_len
                    sys.stdout.write("\r" + " " * thinking_len_to_clear + "\r")
                    sys.stdout.flush()

                    assistant_output = (
                        "No specific textual output from the agent for this turn."
                    )
                    if result.final_output is not None:
                        assistant_output = result.final_output
                    else:
                        logger.warning(
                            "Agent run completed without error, but final_output "
                            "is None."
                        )
                        assistant_output = (
                            "(Agent action completed; no specific text message "
                            "for this turn.)"
                        )

                    formatted_assistant_output = _format_chat_text_for_panel(
                        assistant_output, CHAT_CONTENT_WIDTH
                    )
                    console.print(
                        Panel(
                            formatted_assistant_output,
                            title=agent_object_name,
                            border_style="green",
                            title_align="left",
                            expand=False,
                        )
                    )
                    console.print()

                except typer.Abort:
                    console.print(
                        Text.from_markup(
                            "\n[yellow]Chat interrupted by user. Exiting.[/yellow]"
                        )
                    )
                    logger.debug("Chat interrupted by user (typer.Abort). Exiting.")
                    break
                except KeyboardInterrupt:
                    console.print(
                        Text.from_markup(
                            "\n[yellow]Chat interrupted by user. Exiting.[/yellow]"
                        )
                    )
                    logger.debug(
                        "Chat interrupted by user (KeyboardInterrupt). Exiting."
                    )
                    break
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        f"An error occurred in the chat loop: {e}", exc_info=debug_mode
                    )
                    console.print(
                        Text.from_markup(
                            f"[bold red]An unexpected error occurred: {e}[/bold red]"
                        )
                    )
                    break
    except (UserError, AgentsException, Exception) as e_startup:
        _handle_server_connection_error(
            e_startup,
            lean_backend_type,
            debug_mode,
            context="server startup or connection",
        )

    console.print(
        Text.from_markup("[bold]Lean Search Assistant session has ended.[/bold]")
    )


@typer_async
async def agent_chat_command(
    ctx: typer.Context,
    lean_backend: str = typer.Option(
        "api",
        "--backend",
        "-lb",
        help="Backend for the Lean Explore MCP server ('api' or 'local'). "
        "Default: api.",
        case_sensitive=False,
    ),
    lean_api_key: Optional[str] = typer.Option(
        None,
        "--lean-api-key",
        help="API key for Lean Explore (if 'api' backend). Overrides env var/config.",
        show_default=False,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable detailed debug logging for this script and the MCP server.",
    ),
):
    """Start an interactive chat session with the Lean Search Assistant.

    The assistant uses the Lean Explore MCP server to search for Lean statements.
    An OpenAI API key must be available (prompts if not found). If using `--backend api`
    (default), a Lean Explore API key is also needed (prompts if not found).
    """
    client_log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=client_log_level,
        format="%(asctime)s - %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger.setLevel(client_log_level)

    library_log_level_for_client = logging.DEBUG if debug else logging.WARNING
    logging.getLogger("httpx").setLevel(library_log_level_for_client)
    logging.getLogger("httpcore").setLevel(library_log_level_for_client)
    logging.getLogger("openai").setLevel(library_log_level_for_client)
    logging.getLogger("agents").setLevel(library_log_level_for_client)

    mcp_server_log_level_str = "DEBUG" if debug else "WARNING"

    if not config_utils_imported and not debug:
        if not os.getenv("OPENAI_API_KEY"):
            console.print(
                Text.from_markup(
                    "[yellow]Warning: Automatic loading of stored OpenAI API key "
                    "is disabled (config module not found). OPENAI_API_KEY env "
                    "var is not set. You will be prompted if no key is found "
                    "in config.[/yellow]"
                ),
                stderr=True,
            )
        if lean_backend == "api" and not (
            lean_api_key or os.getenv("LEAN_EXPLORE_API_KEY")
        ):
            console.print(
                Text.from_markup(
                    "[yellow]Warning: Automatic loading of stored Lean Explore "
                    "API key is disabled (config module not found). If using "
                    "--backend api, and key is not in env or via option, you "
                    "will be prompted.[/yellow]"
                ),
                stderr=True,
            )

    resolved_lean_api_key = lean_api_key
    if resolved_lean_api_key is None and lean_backend == "api":
        env_key = os.getenv("LEAN_EXPLORE_API_KEY")
        if env_key:
            logger.debug(
                "Using Lean Explore API key from LEAN_EXPLORE_API_KEY environment "
                "variable for agent session."
            )
            resolved_lean_api_key = env_key

    await _run_agent_session(
        lean_backend_type=lean_backend,
        lean_explore_api_key_arg=resolved_lean_api_key,
        debug_mode=debug,
        log_level_for_mcp_server=mcp_server_log_level_str,
    )
