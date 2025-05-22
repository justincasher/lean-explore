# src/lean_explore/cli/agent.py

"""
Command-line interface logic for interacting with an AI agent
powered by the Lean Explore MCP search tools.

This module provides the agent_chat_command and supporting functions
for the chat interaction, intended to be registered with a main Typer application.
"""

import asyncio
import os
import sys
import shutil
import typer
from typing import Optional
import logging
import functools
import pathlib

# Ensure 'openai-agents' is installed
try:
    from agents import Agent, Runner
    from agents.mcp import MCPServerStdio
    from agents.exceptions import AgentsException, UserError
except ImportError:
    print(
        "Fatal Error: The 'openai-agents' library or its expected exceptions are not installed/found. "
        "Please install 'openai-agents' correctly (e.g., 'pip install openai-agents')",
        file=sys.stderr
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
        "Ensure 'lean_explore' package is installed correctly and accessible in PYTHONPATH "
        "(e.g., by running 'pip install -e .' from the project root)."
    )
    class MockConfigUtils:
        """A mock for config_utils if it cannot be imported."""
        def load_api_key(self) -> Optional[str]: # For Lean Explore key
            return None
        def load_openai_api_key(self) -> Optional[str]: # For OpenAI key
            return None
        def save_api_key(self, api_key: str) -> bool:
            return False
        def save_openai_api_key(self, api_key: str) -> bool:
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
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

agent_cli_app = typer.Typer(
    name="agent_cli_utils",
    help="Utilities related to AI agent interactions.",
    no_args_is_help=True
)

logger = logging.getLogger(__name__)


def _handle_server_connection_error(
    error: Exception,
    lean_backend_type: str,
    debug_mode: bool,
    context: str = "server startup"
):
    """Handles MCP server connection errors by logging and printing user-friendly messages."""
    logger.error(f"CRITICAL: Error during MCP {context}: {type(error).__name__}: {error}", exc_info=debug_mode)

    error_str = str(error).lower()
    is_timeout_error = "timed out" in error_str or "timeout" in error_str

    if is_timeout_error:
        typer.echo(typer.style(
            "Error: The Lean Explore server failed to start or respond promptly.",
            fg=typer.colors.RED, bold=True
        ), err=True)
        if lean_backend_type == "local":
            typer.echo(typer.style(
                "This often occurs with the 'local' backend due to missing or corrupted data files.",
                fg=typer.colors.YELLOW
            ), err=True)
            typer.echo(typer.style(
                "Please try the following steps:",
                fg=typer.colors.YELLOW
            ), err=True)
            typer.echo(typer.style(
                "  1. Run 'leanexplore data fetch' to download or update the required data.",
                fg=typer.colors.YELLOW
            ), err=True)
            typer.echo(typer.style(
                "  2. Try this chat command again.",
                fg=typer.colors.YELLOW
            ), err=True)
            typer.echo(typer.style(
                "  3. If the problem persists, run 'leanexplore mcp serve --backend local --log-level DEBUG' directly in another terminal to see detailed server startup logs.",
                fg=typer.colors.YELLOW
            ), err=True)
        else: # api backend or other cases
            typer.echo(typer.style(
                "Please check your network connection and ensure the API server is accessible.",
                fg=typer.colors.YELLOW
            ), err=True)
    elif isinstance(error, UserError):
        typer.echo(typer.style(f"Error: SDK usage problem during {context}: {error}", fg=typer.colors.RED, bold=True), err=True)
    elif isinstance(error, AgentsException):
        typer.echo(typer.style(f"Error: An SDK error occurred during {context}: {error}", fg=typer.colors.RED, bold=True), err=True)
    else:
        typer.echo(typer.style(f"An unexpected error occurred during {context}: {error}", fg=typer.colors.RED, bold=True), err=True)

    if debug_mode:
        typer.echo(typer.style(f"Error Details ({type(error).__name__}): {error}", fg=typer.colors.MAGENTA), err=True)
    raise typer.Exit(code=1)


# --- Core Agent Logic ---
async def _run_agent_session(
    lean_backend_type: str,
    lean_explore_api_key_arg: Optional[str] = None,
    debug_mode: bool = False,
    log_level_for_mcp_server: str = "WARNING"
):
    """Internal function to set up and run the OpenAI Agent session.

    Args:
        lean_backend_type: The backend ('api' or 'local') for the Lean Explore server.
        lean_explore_api_key_arg: API key for Lean Explore (if 'api' backend),
                                  already resolved from CLI arg or ENV.
        debug_mode: If True, enables more verbose logging for this client and the MCP server.
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
        except Exception as e: # pylint: disable=broad-except
            logger.error(f"Error loading OpenAI API key from CLI configuration: {e}", exc_info=debug_mode)

    if not openai_api_key:
        typer.echo(typer.style(
            "OpenAI API key not found in configuration.",
            fg=typer.colors.YELLOW
        ))
        openai_api_key = typer.prompt("Please enter your OpenAI API key", hide_input=True)
        if not openai_api_key:
            typer.echo(typer.style("OpenAI API key cannot be empty. Exiting.", fg=typer.colors.RED, bold=True), err=True)
            raise typer.Exit(code=1)
        logger.info("Using OpenAI API key provided via prompt.")
        if config_utils_imported:
            if typer.confirm("Would you like to save this OpenAI API key for future use?"):
                if config_utils.save_openai_api_key(openai_api_key):
                    typer.echo(typer.style("OpenAI API key saved successfully.", fg=typer.colors.GREEN))
                else:
                    typer.echo(typer.style("Failed to save OpenAI API key.", fg=typer.colors.RED), err=True)
            else:
                typer.echo("OpenAI API key will be used for this session only.")
        else:
            typer.echo(typer.style("Note: config_utils not available, OpenAI API key cannot be saved.", fg=typer.colors.YELLOW))

    os.environ["OPENAI_API_KEY"] = openai_api_key


    # --- Lean Explore Server Script and Executable Validation ---
    if not internal_server_script_path.exists():
        error_msg = f"Lean Explore MCP server script not found at calculated path: {internal_server_script_path}"
        logger.error(error_msg)
        typer.echo(typer.style(f"Error: {error_msg}", fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)

    python_executable = sys.executable
    if not python_executable or not shutil.which(python_executable):
        error_msg = (f"Python executable '{python_executable}' not found or not executable. "
                     "Ensure Python is correctly installed and in your PATH.")
        logger.error(error_msg)
        typer.echo(typer.style(f"Error: {error_msg}", fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)

    # --- Lean Explore API Key Acquisition (if API backend) ---
    effective_lean_api_key = lean_explore_api_key_arg
    if lean_backend_type == "api":
        if not effective_lean_api_key and config_utils_imported:
            logger.debug("Lean Explore API key not provided via CLI option or ENV. Attempting to load from CLI configuration...")
            try:
                stored_key = config_utils.load_api_key()
                if stored_key:
                    effective_lean_api_key = stored_key
                    logger.debug("Successfully loaded Lean Explore API key from CLI configuration.")
                else:
                    logger.debug("No Lean Explore API key found in CLI configuration.")
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Error loading Lean Explore API key from CLI configuration: {e}", exc_info=debug_mode)

        if not effective_lean_api_key:
            typer.echo(typer.style(
                "Lean Explore API key is required for the 'api' backend and was not found through CLI option, environment variable, or configuration.",
                fg=typer.colors.YELLOW
            ))
            effective_lean_api_key = typer.prompt("Please enter your Lean Explore API key", hide_input=True)
            if not effective_lean_api_key:
                typer.echo(typer.style("Lean Explore API key cannot be empty for 'api' backend. Exiting.", fg=typer.colors.RED, bold=True), err=True)
                raise typer.Exit(code=1)
            logger.info("Using Lean Explore API key provided via prompt.")
            if config_utils_imported:
                if typer.confirm("Would you like to save this Lean Explore API key for future use?"):
                    if config_utils.save_api_key(effective_lean_api_key):
                        typer.echo(typer.style("Lean Explore API key saved successfully.", fg=typer.colors.GREEN))
                    else:
                        typer.echo(typer.style("Failed to save Lean Explore API key.", fg=typer.colors.RED), err=True)
                else:
                    typer.echo("Lean Explore API key will be used for this session only.")
            else:
                typer.echo(typer.style("Note: config_utils not available, Lean Explore API key cannot be saved.", fg=typer.colors.YELLOW))


    # --- MCP Server Setup ---
    mcp_server_args = [
        str(internal_server_script_path),
        "--backend", lean_backend_type,
        "--log-level", log_level_for_mcp_server
    ]
    if lean_backend_type == "api" and effective_lean_api_key:
        mcp_server_args.extend(["--api-key", effective_lean_api_key])

    lean_explore_mcp_server = MCPServerStdio(
        name="LeanExploreSearchServer",
        params={"command": python_executable, "args": mcp_server_args, "cwd": str(internal_server_script_path.parent)},
        cache_tools_list=True,
        client_session_timeout_seconds=10.0
    )

    # --- Agent Interaction Loop ---
    try:
        async with lean_explore_mcp_server as server_instance:
            logger.debug(f"MCP server '{server_instance.name}' connection initiated. Listing tools...")
            tools = []
            try:
                tools = await server_instance.list_tools()
                if not tools or not any(tools):
                    logger.warning("MCP Server connected but reported no tools. Agent may lack expected capabilities.")
                else:
                    logger.debug(f"Available tools from {server_instance.name}: {[tool.name for tool in tools]}")
            except (UserError, AgentsException, Exception) as e_list_tools:
                 _handle_server_connection_error(
                    e_list_tools,
                    lean_backend_type,
                    debug_mode,
                    context="tool listing"
                )

            agent_model = "gpt-4.1"
            agent_object_name = "Assistant"
            agent_display_name = f"{Colors.BOLD}{Colors.GREEN}{agent_object_name}{Colors.ENDC}"

            agent = Agent(
                name=agent_object_name, model=agent_model,
                instructions=(
                    "You are a CLI assistant for searching a Lean 4 mathematical library.\n"
                    "**Goal:** Find relevant Lean statements, understand them (including dependencies), and explain them conversationally to the user.\n"
                    "**Output:** CLI-friendly (plain text, simple lists). NO complex Markdown/LaTeX.\n\n"

                    "**Packages:** Use exact top-level names for filters (Batteries, Init, Lean, Mathlib, PhysLean, Std). Map subpackage mentions to top-level (e.g., 'Mathlib.Analysis' -> 'Mathlib').\n\n"

                    "**Core Workflow:**\n"
                    "1.  **Search & Analyze:**\n"
                    "    * Execute multiple distinct `search` queries for each user request (e.g., using full statements, rephrasing). Set `limit` >= 10 for each search.\n"
                    "    * From all search results, select the statement(s) most helpful to the user.\n"
                    "    * For each selected statement, **MUST** use `get_dependencies` to understand its context before explaining.\n\n"

                    "2.  **Explain Results (Conversational & CLI-Friendly):**\n"
                    "    * Briefly state your search approach (e.g., 'I looked into X in Mathlib...').\n"
                    "    * For each selected statement:\n"
                    "        * Introduce it (e.g., Lean name: `primary_declaration.lean_name`).\n"
                    "        * Explain its meaning (use `docstring`, `informal_description`, `statement_text`).\n"
                    "        * Provide the full Lean code (`statement_text`).\n"
                    "        * Explain key dependencies (what they are, their role, using `statement_text` or `display_statement_text` from `get_dependencies` output).\n"

                    "3.  **Specific User Follow-ups (If Asked):**\n"
                    "    * **`get_by_id`:** For a specific ID, provide: ID, Lean name, statement text, source/line, docstring, informal description (structured CLI format).\n"
                    "    * **`get_dependencies` (Direct Request):** For all dependencies of an ID, list: ID, Lean name, statement text/summary. State total count.\n\n"
                    "Always be concise, helpful, and clear."
                ),
                mcp_servers=[server_instance]
            )

            typer.echo(typer.style(f"Lean Search Assistant", bold=True) + f" (powered by {Colors.GREEN}{agent_model}{Colors.ENDC} and {Colors.GREEN}{server_instance.name}{Colors.ENDC}) is ready.")
            typer.echo("Ask me to search for Lean statements (e.g., 'find definitions of a scheme').")
            if not debug_mode and lean_backend_type == 'local':
                 typer.echo(typer.style(
                     "Note: The local search server might print startup logs. For a quieter experience, "
                     "use --debug to see detailed logs or ensure the server's default log level is WARNING.",
                     fg=typer.colors.YELLOW
                 ))
            typer.echo("Type 'exit' or 'quit' to end the session.\n")

            while True:
                try:
                    user_input = typer.prompt(typer.style("You", fg=typer.colors.BLUE, bold=True), default="", prompt_suffix=": ").strip()
                    if user_input.lower() in ["exit", "quit"]:
                        logger.debug("Exiting chat loop.")
                        break
                    if not user_input:
                        continue

                    typer.echo()
                    typer.echo(f"{agent_display_name}: {Colors.YELLOW}Thinking...{Colors.ENDC}")

                    result = await Runner.run(starting_agent=agent, input=user_input)

                    assistant_output = "No specific textual output from the agent for this turn."
                    if result.final_output is not None:
                        assistant_output = result.final_output
                    else:
                        logger.warning("Agent run completed without error, but final_output is None.")
                        assistant_output = "(Agent action completed; no specific text message for this turn.)"

                    thinking_line_approx_len = len(agent_object_name) + len(": Thinking...") + len(Colors.BOLD+Colors.GREEN+Colors.ENDC+Colors.YELLOW+Colors.ENDC) + 5
                    sys.stdout.write("\r" + " " * thinking_line_approx_len + "\r")
                    sys.stdout.flush()

                    typer.echo(f"{agent_display_name}: {assistant_output}\n")

                except typer.Abort:
                    typer.echo(f"\n{Colors.YELLOW}Chat interrupted by user. Exiting.{Colors.ENDC}")
                    logger.debug("Chat interrupted by user (typer.Abort). Exiting.")
                    break
                except KeyboardInterrupt:
                    typer.echo(f"\n{Colors.YELLOW}Chat interrupted by user. Exiting.{Colors.ENDC}")
                    logger.debug("Chat interrupted by user (KeyboardInterrupt). Exiting.")
                    break
                except Exception as e: # pylint: disable=broad-except
                    logger.error(f"An error occurred in the chat loop: {e}", exc_info=debug_mode)
                    typer.echo(typer.style(f"An unexpected error occurred: {e}", fg=typer.colors.RED, bold=True))
                    break
    except (UserError, AgentsException, Exception) as e_startup: # Catches errors from MCPServerStdio.__aenter__
        _handle_server_connection_error(
            e_startup,
            lean_backend_type,
            debug_mode,
            context="server startup or connection"
        )

    typer.echo(typer.style("Lean Search Assistant session has ended.", bold=True))


@typer_async
async def agent_chat_command(
    ctx: typer.Context,
    lean_backend: str = typer.Option(
        "api",
        "--backend",
        "-lb",
        help="Backend for the Lean Explore MCP server ('api' or 'local'). Default: api.",
        case_sensitive=False,
    ),
    lean_api_key: Optional[str] = typer.Option(
        None,
        "--lean-api-key",
        help=("API key for Lean Explore (if 'api' backend). Overrides env var/config."),
        show_default=False
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable detailed debug logging for this script and the MCP server."
    )
):
    """
    Start an interactive chat session with the Lean Search Assistant.

    The assistant uses the Lean Explore MCP server to search for Lean statements.
    An OpenAI API key must be available (prompts if not found).
    If using --backend api (default), a Lean Explore API key is also needed (prompts if not found).
    """
    client_log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=client_log_level,
        format="%(asctime)s - %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logger.setLevel(client_log_level)

    library_log_level_for_client = logging.DEBUG if debug else logging.WARNING
    logging.getLogger("httpx").setLevel(library_log_level_for_client)
    logging.getLogger("httpcore").setLevel(library_log_level_for_client)
    logging.getLogger("openai").setLevel(library_log_level_for_client)
    logging.getLogger("agents").setLevel(library_log_level_for_client) # Covers agents.mcp, agents.exceptions etc.

    mcp_server_log_level_str = "DEBUG" if debug else "WARNING"

    if not config_utils_imported and not debug:
        if not os.getenv("OPENAI_API_KEY"):
            typer.echo(typer.style(
                "Warning: Automatic loading of stored OpenAI API key is disabled (config module not found). "
                "OPENAI_API_KEY env var is not set. You will be prompted if no key is found in config.",
                fg=typer.colors.YELLOW
            ), err=True)
        if lean_backend == "api" and not (lean_api_key or os.getenv("LEAN_EXPLORE_API_KEY")):
             typer.echo(typer.style(
                "Warning: Automatic loading of stored Lean Explore API key is disabled (config module not found). "
                "If using --backend api, and key is not in env or via option, you will be prompted.",
                fg=typer.colors.YELLOW
            ), err=True)


    resolved_lean_api_key = lean_api_key
    if resolved_lean_api_key is None and lean_backend == "api":
        env_key = os.getenv("LEAN_EXPLORE_API_KEY")
        if env_key:
            logger.debug("Using Lean Explore API key from LEAN_EXPLORE_API_KEY environment variable for agent session.")
            resolved_lean_api_key = env_key

    await _run_agent_session(
        lean_backend_type=lean_backend,
        lean_explore_api_key_arg=resolved_lean_api_key,
        debug_mode=debug,
        log_level_for_mcp_server=mcp_server_log_level_str
    )