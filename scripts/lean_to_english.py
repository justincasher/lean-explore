# File: scripts/lean_to_english.py

"""Processes StatementGroups to generate informal English descriptions using an LLM.

Identifies groups needing descriptions, processes them in a topological sort
order based on pre-computed group-level dependencies (with configurable
library-based filtering), and constructs prompts using a template, group text,
and prerequisite descriptions. It calls a Gemini LLM concurrently (with rate
limiting and cost tracking) for translation and updates the
'informal_description' field in the database. Progress is displayed using tqdm.
An optional test mode processes a limited number of items with LLM calls but
without database writes. A startover mode allows re-processing all groups.
If cycles are detected in the dependency graph, a fallback mechanism attempts
to process the remaining items.
"""

import argparse
import asyncio
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- Project Path Setup for Imports ---
# (Ensure 'sys' and 'from pathlib import Path' have been imported above this block)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to sys.path to reliably import 'dev_tools'
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Dependency Imports ---
try:
    from sqlalchemy import create_engine, inspect, select, update
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session, selectinload, sessionmaker
except ImportError:
    # pylint: disable=broad-exception-raised
    print(
        "Error: Missing 'sqlalchemy'. Install with 'pip install sqlalchemy'",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # pylint: disable=broad-exception-raised
    print("Error: Missing 'tqdm'. Install with 'pip install tqdm'", file=sys.stderr)
    sys.exit(1)


# --- Project Model & Config/LLM Imports ---
try:
    # For 'lean_explore' modules, ensure 'lean_explore' is installed
    # (e.g., 'pip install -e .') or the 'src/' directory is added to sys.path.
    # 'pip install -e .' is recommended.
    # For 'dev_tools.llm_caller', ensure PROJECT_ROOT is in sys.path.
    from dev_tools.llm_caller import GeminiClient, GeminiCostTracker
    from .config import APP_CONFIG, get_gemini_api_key
    from lean_explore.shared.models.db import (
        Declaration,
        StatementGroup,
        StatementGroupDependency,
    )
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Could not import project modules: {e}\n\n"
        "POSSIBLE CAUSES & SOLUTIONS:\n"
        "1. RUNNING LOCATION: Ensure you are running this script from the "
        "project's ROOT directory,\n"
        "   e.g., 'python scripts/lean_to_english.py'.\n\n"
        "2. FOR 'lean_explore' (e.g., models, config):\n"
        "   - RECOMMENDED: The 'lean_explore' package might not be installed "
        "in your current Python environment.\n"
        "     From your project root, run: 'pip install -e .'\n"
        "     This installs it in editable mode, making it available.\n"
        "   - Alternatively (if not using editable install), ensure the 'src/' "
        "directory is correctly added to sys.path.\n\n"
        "3. FOR 'dev_tools.llm_caller':\n"
        "   - The 'dev_tools/' directory (containing 'llm_caller.py' and an "
        "'__init__.py' file)\n"
        "     must be at the project root.\n"
        "   - The script attempts to add the project root to sys.path (this "
        "should now happen before the import attempt);\n"
        "     verify this is working for your environment if issues persist.\n"
        "     You might need to set your PYTHONPATH environment variable to "
        "include the project root.\n\n"
        f"Current sys.path (at time of error): {sys.path}",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:  # pylint: disable=broad-except
    print(
        f"An unexpected error occurred during project module import: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 20
DEFAULT_PROMPT_TEMPLATE_PATH = "scripts/prompt_template.txt"

# Library categorization constants based on the first path component of source_file
FOUNDATIONAL_LIBRARY_COMPONENTS = {"Init", "Std", "Lean", "Batteries"}
THEORY_LIBRARY_COMPONENTS = {"Mathlib", "PhysLean"}


# --- Data Structures ---
@dataclass
class GroupData:
    """Holds necessary data for a StatementGroup during processing.

    Attributes:
        id: The database ID of the StatementGroup.
        statement_text: The full Lean statement text of the group.
        source_file: The source file path of the StatementGroup.
        primary_decl_name: The Lean name of the primary declaration for context.
        primary_decl_type: The type of the primary declaration for context.
        informal_description: The existing or newly generated informal
            description. This is None initially if it needs processing or if
            startover mode is active.
        primary_decl_docstring: The docstring of the primary declaration, if any.
    """

    id: int
    statement_text: str
    source_file: str
    primary_decl_name: str = "UnknownPrimaryDeclaration"
    primary_decl_type: str = "UnknownType"
    informal_description: Optional[str] = None
    primary_decl_docstring: Optional[str] = None


# --- Helper Functions ---
def get_library_category(source_file: str) -> Optional[str]:
    """Determines the library category from the first path component of source_file.

    Categories can be "FOUNDATIONAL", "THEORY", "OTHER", or None if
    source_file is empty or invalid.

    Args:
        source_file: The source_file string from a StatementGroup.

    Returns:
        The category string ("FOUNDATIONAL", "THEORY", "OTHER") or None.
    """
    if not source_file:
        return None

    parts = source_file.split("/", 1)
    first_component = parts[0]

    if not first_component:
        logger.warning(
            "Could not determine first_component from source_file: '%s'", source_file
        )
        return "OTHER"

    if first_component in FOUNDATIONAL_LIBRARY_COMPONENTS:
        return "FOUNDATIONAL"
    elif first_component in THEORY_LIBRARY_COMPONENTS:
        return "THEORY"
    else:
        return "OTHER"


# --- Core Logic ---


def load_group_graph_data(
    session: Session, startover: bool = False
) -> Tuple[
    Dict[int, GroupData],
    Set[int],
    Dict[int, int],
    Dict[int, List[int]],
    Dict[int, List[int]],
]:
    """Loads StatementGroup data and builds the group-level dependency graph.

    Fetches StatementGroups, identifies those needing processing, and builds the
    dependency graph using pre-computed `StatementGroupDependency` links.
    Dependencies from "FOUNDATIONAL" libraries to "THEORY" libraries can be
    filtered out to adjust processing order.

    Args:
        session: The SQLAlchemy session object.
        startover: If True, all groups with statement_text are considered for
            processing.

    Returns:
        A tuple containing:
            all_group_data: Dictionary mapping all group IDs to their GroupData.
            groups_to_process: Set of IDs of StatementGroups needing processing.
            group_in_degree: Dictionary of in-degree counts for candidates.
            group_successors: Successor map for candidate groups.
            group_predecessors: Predecessor map for candidate groups.

    Raises:
        SQLAlchemyError: If database operations fail.
    """
    logger.info("Loading StatementGroup data with primary declaration details...")
    all_group_data: Dict[int, GroupData] = {}
    groups_to_process_ids: Set[int] = set()

    group_stmt = select(StatementGroup).options(
        selectinload(StatementGroup.primary_declaration).load_only(
            Declaration.lean_name, Declaration.decl_type, Declaration.docstring
        )
    )
    db_groups = session.execute(group_stmt).scalars().unique().all()

    for group in db_groups:
        if not group.statement_text:
            logger.warning(
                "StatementGroup %d has no statement_text, skipping.", group.id
            )
            continue

        primary_name = f"UnnamedPrimary_{group.primary_decl_id}"
        primary_type = "UnknownType"
        primary_docstring = None
        if group.primary_declaration:
            primary_name = group.primary_declaration.lean_name or primary_name
            primary_type = group.primary_declaration.decl_type or primary_type
            primary_docstring = group.primary_declaration.docstring

        group_data_item = GroupData(
            id=group.id,
            statement_text=group.statement_text,
            source_file=group.source_file,
            primary_decl_name=primary_name,
            primary_decl_type=primary_type,
            informal_description=(None if startover else group.informal_description),
            primary_decl_docstring=primary_docstring,
        )
        all_group_data[group.id] = group_data_item

        if startover or not group.informal_description:
            groups_to_process_ids.add(group.id)

    logger.info("Loaded data for %d StatementGroups.", len(all_group_data))
    if startover:
        logger.info(
            "Startover mode: All %d groups with statement text will be processed.",
            len(groups_to_process_ids),
        )
    else:
        logger.info(
            "Found %d StatementGroups needing processing (missing description).",
            len(groups_to_process_ids),
        )

    if not groups_to_process_ids:
        return {}, set(), {}, {}, {}

    logger.info("Loading dependencies from 'statement_group_dependencies' table...")
    group_in_degree: Dict[int, int] = {gp_id: 0 for gp_id in groups_to_process_ids}
    group_successors: Dict[int, List[int]] = {
        gp_id: [] for gp_id in groups_to_process_ids
    }
    group_predecessors: Dict[int, List[int]] = {
        gp_id: [] for gp_id in groups_to_process_ids
    }

    stmt_group_deps_query = select(
        StatementGroupDependency.source_statement_group_id,
        StatementGroupDependency.target_statement_group_id,
    ).where(
        StatementGroupDependency.source_statement_group_id.in_(groups_to_process_ids),
        StatementGroupDependency.target_statement_group_id.in_(groups_to_process_ids),
    )
    direct_group_dependencies = session.execute(stmt_group_deps_query).all()
    logger.info(
        "Loaded %d group dependency links for processing candidates.",
        len(direct_group_dependencies),
    )

    valid_group_dependency_count = 0
    processed_edges: Set[Tuple[int, int]] = set()

    for dependent_sg_id, prerequisite_sg_id in direct_group_dependencies:
        if dependent_sg_id == prerequisite_sg_id:
            logger.warning(
                "Ignoring self-dependency for group ID %d from "
                "statement_group_dependencies.",
                dependent_sg_id,
            )
            continue

        dependent_group_data = all_group_data.get(dependent_sg_id)
        prerequisite_group_data = all_group_data.get(prerequisite_sg_id)

        if not dependent_group_data or not prerequisite_group_data:
            logger.warning(
                "Missing GroupData for group IDs %d or %d from "
                "statement_group_dependencies. Skipping edge.",
                dependent_sg_id,
                prerequisite_sg_id,
            )
            continue

        dependent_category = get_library_category(dependent_group_data.source_file)
        prerequisite_category = get_library_category(
            prerequisite_group_data.source_file
        )

        if dependent_category == "THEORY" and prerequisite_category == "FOUNDATIONAL":
            logger.debug(
                "Ignoring dependency edge from FOUNDATIONAL group %d (%s) "
                "to THEORY group %d (%s) for graph building "
                "(from statement_group_dependencies).",
                prerequisite_sg_id,
                prerequisite_group_data.source_file,
                dependent_sg_id,
                dependent_group_data.source_file,
            )
            continue

        edge = (prerequisite_sg_id, dependent_sg_id)
        if edge not in processed_edges:
            group_in_degree[dependent_sg_id] = (
                group_in_degree.get(dependent_sg_id, 0) + 1
            )
            group_successors.setdefault(prerequisite_sg_id, []).append(dependent_sg_id)
            group_predecessors.setdefault(dependent_sg_id, []).append(
                prerequisite_sg_id
            )
            processed_edges.add(edge)
            valid_group_dependency_count += 1

    logger.info(
        "Built group graph with %d unique inter-group dependencies among "
        "processing candidates.",
        valid_group_dependency_count,
    )
    return (
        all_group_data,
        groups_to_process_ids,
        group_in_degree,
        group_successors,
        group_predecessors,
    )


def format_prerequisites_for_prompt(
    prerequisite_group_ids: List[int],
    all_group_data: Dict[int, GroupData],
    generated_group_translations: Dict[int, str],
    current_group_id: int,
) -> str:
    """Formats prerequisite StatementGroup information for the LLM prompt.

    Constructs a markdown-formatted string listing prerequisites.
    Prerequisites from "FOUNDATIONAL" libraries are filtered out if the
    current group belongs to a "THEORY" library.

    Args:
        prerequisite_group_ids: List of IDs of prerequisite StatementGroups.
        all_group_data: Dictionary mapping all group IDs to their GroupData.
        generated_group_translations: Cache of recently generated descriptions.
        current_group_id: The ID of the StatementGroup for which the prompt
            is being generated.

    Returns:
        A formatted string describing the prerequisites.
    """
    lean_to_english_config = APP_CONFIG.get("lean_to_english", {})
    max_items = lean_to_english_config.get("max_dependencies_in_prompt", 20)
    max_chars_per_desc = lean_to_english_config.get(
        "max_dependency_description_length", 150
    )
    total_char_limit = lean_to_english_config.get(
        "total_prerequisite_context_char_limit", 10000
    )

    current_group_data = all_group_data.get(current_group_id)
    current_category = "OTHER"
    if current_group_data:
        current_category = (
            get_library_category(current_group_data.source_file) or "OTHER"
        )

    if not prerequisite_group_ids:
        return (
            "This statement block has no direct prerequisite blocks that "
            "required processing.\n"
        )

    header = (
        "This statement block depends on the following concepts "
        "(from prerequisite blocks):\n"
    )
    prompt_lines = [header]
    current_chars = len(header)
    items_added = 0
    limit_reason = ""
    filtered_prereq_ids = []

    for prereq_id in sorted(prerequisite_group_ids):
        prereq_data = all_group_data.get(prereq_id)
        if not prereq_data:
            logger.warning(
                "Could not find data for prerequisite group ID %d during "
                "prompt formatting.",
                prereq_id,
            )
            continue

        prereq_category = "OTHER"
        if prereq_data.source_file:
            prereq_category = get_library_category(prereq_data.source_file) or "OTHER"

        if current_category == "THEORY" and prereq_category == "FOUNDATIONAL":
            logger.debug(
                "Skipping FOUNDATIONAL prerequisite group %d (%s) for THEORY "
                "group %d (%s) in prompt.",
                prereq_id,
                prereq_data.source_file,
                current_group_id,
                current_group_data.source_file if current_group_data else "N/A",
            )
            continue
        filtered_prereq_ids.append(prereq_id)

    num_total_prereqs_considered = len(prerequisite_group_ids)
    num_shown_or_elligible = len(filtered_prereq_ids)

    for prereq_id_to_show in filtered_prereq_ids:
        if items_added >= max_items:
            limit_reason = f"item limit ({max_items})"
            break

        prereq_data = all_group_data.get(prereq_id_to_show)  # Should exist

        desc = generated_group_translations.get(
            prereq_id_to_show, prereq_data.informal_description if prereq_data else None
        )
        if not desc or not desc.strip():
            desc = "[Description not available]"
        elif len(desc) > max_chars_per_desc:
            desc = desc[:max_chars_per_desc].rsplit(" ", 1)[0] + "..."

        line = (
            f"- **{prereq_data.primary_decl_name}** ({prereq_data.primary_decl_type} "
            f"block): {desc}\n"
        )

        if current_chars + len(line) > total_char_limit:
            limit_reason = f"total character limit ({total_char_limit} chars)"
            break

        prompt_lines.append(line)
        current_chars += len(line)
        items_added += 1

    if items_added < num_shown_or_elligible and limit_reason:
        remaining = num_shown_or_elligible - items_added
        ellipsis_line = (
            f"- ... (and {remaining} more eligible prerequisite block(s) not "
            f"shown due to {limit_reason})\n"
        )
        if current_chars + len(ellipsis_line) <= total_char_limit:
            prompt_lines.append(ellipsis_line)
    elif items_added == 0 and num_shown_or_elligible > 0:
        generic_ellipsis = "- ... (eligible prerequisites not shown due to limits)\n"
        if current_chars + len(generic_ellipsis) <= total_char_limit:
            prompt_lines.append(generic_ellipsis)
    elif num_shown_or_elligible == 0 and num_total_prereqs_considered > 0:
        if len(prompt_lines) == 1 and prompt_lines[0] == header:
            prompt_lines.append(
                "- (All direct prerequisites were filtered based on library "
                "category rules.)\n"
            )
    elif num_shown_or_elligible == 0 and num_total_prereqs_considered == 0:
        if len(prompt_lines) == 1 and prompt_lines[0] == header:
            return (
                "This statement block has no direct prerequisite blocks that "
                "required processing or were eligible for display.\n"
            )

    return "".join(prompt_lines)


async def _process_single_statement_group_with_llm(
    group_id: int,
    all_group_data: Dict[int, GroupData],
    group_predecessors: Dict[int, List[int]],
    generated_group_translations: Dict[int, str],
    client: GeminiClient,
    semaphore: asyncio.Semaphore,
    prompt_template: str,
    is_test_mode: bool,
) -> Tuple[int, Optional[str], Optional[Exception]]:
    """Processes one StatementGroup: formats prompt, calls LLM, handles result.

    This internal helper is rate-limited by the provided semaphore.

    Args:
        group_id: The ID of the StatementGroup to process.
        all_group_data: Data for all loaded groups.
        group_predecessors: Map of group IDs to their prerequisite group IDs.
        generated_group_translations: Cache of recently generated descriptions.
        client: The GeminiClient instance.
        semaphore: The asyncio Semaphore for concurrent call limiting.
        prompt_template: The loaded prompt template string.
        is_test_mode: Flag indicating if running in test mode.

    Returns:
        A tuple: (group_id, generated_description or None, error or None).
    """
    current_group = all_group_data.get(group_id)
    if not current_group:
        return (
            group_id,
            None,
            ValueError(f"Data for StatementGroup ID {group_id} not found."),
        )
    if not current_group.statement_text:
        logger.warning(
            "StatementGroup %d (%s) has no statement_text, cannot generate "
            "description.",
            group_id,
            current_group.primary_decl_name,
        )
        return group_id, None, None  # No error, but no translation

    try:
        prereq_ids = group_predecessors.get(group_id, [])
        prereq_context = format_prerequisites_for_prompt(
            prereq_ids, all_group_data, generated_group_translations, group_id
        )
        docstring_context = (
            current_group.primary_decl_docstring
            if current_group.primary_decl_docstring
            else "No docstring available for the primary declaration."
        )

        prompt = prompt_template.format(
            primary_lean_name=current_group.primary_decl_name,
            primary_decl_type=current_group.primary_decl_type,
            statement_text=current_group.statement_text,
            prerequisites_context=prereq_context,
            docstring_context=docstring_context,
        )

        if is_test_mode:
            print(
                f"\n--- LLM PROMPT for Group: {current_group.primary_decl_name} "
                f"(ID: {group_id}) ---\n"
                f"Source File: {current_group.source_file}\n"
                f"{prompt}"
                f"\n--- END LLM PROMPT ---"
            )

        async with semaphore:
            generated_text = await client.generate(prompt=prompt)

        if generated_text and generated_text.strip():
            translation = generated_text.strip()
            if is_test_mode:
                print(
                    f"\n--- Test Output for Group: {current_group.primary_decl_name} "
                    f"(ID: {group_id}) ---\n"
                    f"Statement Text:\n{current_group.statement_text}\n"
                    f"Generated Description:\n{translation}\n"
                    f"---------------------------------------------"
                )
            return group_id, translation, None

        logger.warning(
            "LLM returned empty description for Group %d (%s).",
            group_id,
            current_group.primary_decl_name,
        )
        return group_id, None, None  # No error, but no translation

    except Exception as llm_err:  # pylint: disable=broad-except
        logger.error(
            "LLM call failed for Group %d (%s): %s",
            group_id,
            current_group.primary_decl_name,
            llm_err,
            exc_info=is_test_mode,  # Provide more info in test mode for debugging
        )
        return group_id, None, llm_err


async def process_statement_groups_with_llm(
    db_url: str,
    commit_batch_size: int,
    test_limit: Optional[int] = None,
    startover: bool = False,
) -> bool:
    """Orchestrates StatementGroup processing with LLM.

    Loads group data, builds a dependency graph, and processes groups using
    Kahn's algorithm for topological sort. If cycles prevent full processing,
    a fallback mechanism attempts to process remaining groups. Calls an LLM
    concurrently. Updates the database with generated descriptions or prints
    to console (test mode).

    Args:
        db_url: SQLAlchemy database URL.
        commit_batch_size: Number of database updates to commit in one batch.
        test_limit: If set, process at most N items in test mode.
        startover: If True, re-translate all applicable groups.

    Returns:
        True if all targeted groups were attempted, False otherwise.
    """
    is_test_mode = test_limit is not None
    mode_desc_str = (
        f"Test Mode (limit {test_limit}, NO DB writes)" if is_test_mode else "Live Mode"
    )
    if startover:
        mode_desc_str += " [STARTOVER]"
    logger.info("--- Starting LLM Group Processing Workflow (%s) ---", mode_desc_str)
    logger.info("Using database: %s", db_url)
    if not is_test_mode:
        logger.info("Database commit batch size: %d", commit_batch_size)

    lean_to_english_cfg = APP_CONFIG.get("lean_to_english", {})
    prompt_template_path_str = lean_to_english_cfg.get(
        "prompt_template_path", DEFAULT_PROMPT_TEMPLATE_PATH
    )
    try:
        template_path = Path(prompt_template_path_str).resolve()
        prompt_template = template_path.read_text(encoding="utf-8")
        logger.info("Loaded prompt template from %s", template_path)
    except FileNotFoundError:
        logger.error(
            "Prompt template file '%s' (resolved to '%s') not found. "
            "Check 'prompt_template_path' in config or default location.",
            prompt_template_path_str,
            template_path,
        )
        return False
    except Exception as e:
        logger.error(
            "Error loading prompt template %s: %s",
            template_path if "template_path" in locals() else prompt_template_path_str,
            e,
            exc_info=True,
        )
        return False

    max_concurrent_calls = lean_to_english_cfg.get(
        "max_concurrent_llm_calls", DEFAULT_MAX_CONCURRENT_LLM_CALLS
    )
    if not (isinstance(max_concurrent_calls, int) and max_concurrent_calls > 0):
        logger.warning(
            "Invalid max_concurrent_llm_calls '%s', using default %d.",
            max_concurrent_calls,
            DEFAULT_MAX_CONCURRENT_LLM_CALLS,
        )
        max_concurrent_calls = DEFAULT_MAX_CONCURRENT_LLM_CALLS
    logger.info("Max concurrent LLM calls: %d", max_concurrent_calls)
    llm_semaphore = asyncio.Semaphore(max_concurrent_calls)

    max_gather_batch = lean_to_english_cfg.get("max_gather_batch_size")
    if max_gather_batch is not None and not (
        isinstance(max_gather_batch, int) and max_gather_batch > 0
    ):
        logger.warning(
            "Invalid max_gather_batch_size '%s', applying no limit.", max_gather_batch
        )
        max_gather_batch = None
    logger.info("Max items per gather batch: %s", max_gather_batch or "No limit")

    api_key = get_gemini_api_key()
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment or configuration.")
        return False
    client: GeminiClient
    try:
        llm_config = APP_CONFIG.get("llm", {})
        generation_model_name = llm_config.get("generation_model")

        if not generation_model_name:
            logger.error(
                "Configuration error: 'llm.generation_model' is not set in your "
                "config.yml or via the DEFAULT_GENERATION_MODEL environment variable. "
                "Cannot initialize GeminiClient."
            )
            return False  # Abort if model name isn't found in the loaded config

        cost_tracker = GeminiCostTracker(model_costs_override=APP_CONFIG.get("costs"))
        client = GeminiClient(
            api_key=api_key,
            cost_tracker=cost_tracker,
            default_generation_model=generation_model_name,
        )
        # This log line should now correctly reflect the model passed.
        # If GeminiClient sets an attribute like 'self.default_generation_model',
        # this will show it.
        logger.info(
            "GeminiClient initialized to use Generation Model: %s",
            generation_model_name,
        )

    except (
        ValueError
    ) as ve:  # Catch the specific ValueError from GeminiClient if it still occurs
        logger.error(
            "Failed to initialize GeminiClient: %s. This might happen if the "
            "model name is invalid or other internal GeminiClient setup fails.",
            ve,
            exc_info=True,
        )
        logger.error(
            "Please ensure 'llm.generation_model' is correctly specified in your "
            "'config.yml' (e.g., 'gemini-1.5-pro-latest') or set the "
            "'DEFAULT_GENERATION_MODEL' environment variable."
        )
        return False
    except Exception as e:  # Catch other unexpected errors during initialization
        logger.error(
            "An unexpected error occurred while initializing GeminiClient: %s",
            e,
            exc_info=True,
        )
        return False

    engine = None
    SessionLocal = None
    try:
        engine = create_engine(db_url, echo=False)
        with engine.connect() as connection:  # Verify connection early
            inspector = inspect(engine)
            required_tables = [
                Declaration.__tablename__,
                StatementGroup.__tablename__,
                StatementGroupDependency.__tablename__,
            ]
            if not all(inspector.has_table(t) for t in required_tables):
                missing = [t for t in required_tables if not inspector.has_table(t)]
                logger.error(
                    "Database missing one or more required tables: %s", missing
                )
                return False
            sg_columns = [
                col["name"]
                for col in inspector.get_columns(StatementGroup.__tablename__)
            ]
            if "informal_description" not in sg_columns:
                logger.error(
                    "DB table '%s' missing 'informal_description' column.",
                    StatementGroup.__tablename__,
                )
                return False
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database connection successful and schema verified.")
    except Exception as e:
        logger.error("Failed to setup database connection: %s", e, exc_info=True)
        return False

    processed_count = 0
    translation_success_count = 0
    translation_failure_count = 0
    generated_group_translations: Dict[int, str] = {}
    pending_db_updates: List[Tuple[int, str]] = []
    pbar: Optional[tqdm] = None
    session_obj_for_processing: Optional[Session] = None
    launched_task_ids: Set[int] = set()

    try:
        with SessionLocal() as session_obj_for_processing:
            (
                all_group_data,
                groups_to_process_ids,
                group_in_degree,
                group_successors,
                group_predecessors,
            ) = load_group_graph_data(session_obj_for_processing, startover=startover)

            total_to_process_initially = len(groups_to_process_ids)
            if total_to_process_initially == 0:
                logger.info("No StatementGroups found needing processing. Exiting.")
                return True

            pbar_total = (
                min(test_limit, total_to_process_initially)
                if is_test_mode
                else total_to_process_initially
            )
            pbar = tqdm(total=pbar_total, desc="Processing Groups", unit="group")
            initial_cost_str = f"${client.cost_tracker.get_total_cost():.4f}"
            pbar.set_postfix_str(
                f"Cost: {initial_cost_str} | OK: 0 | Fail: 0 | "
                f"{mode_desc_str.split(' [')[0]}"
            )

            # Phase 1: Topological Sort
            logger.info("Starting topological sort phase...")
            current_processing_queue = deque(
                sorted(
                    [
                        gid
                        for gid in groups_to_process_ids
                        if group_in_degree.get(gid, 0) == 0
                    ]
                )
            )

            if not current_processing_queue and groups_to_process_ids:
                logger.info(
                    "No groups with in-degree 0 found for topological sort. "
                    "All items will be handled by fallback if applicable."
                )

            phase_name = "Topological"
            while current_processing_queue:
                if is_test_mode and processed_count >= test_limit:
                    logger.info(
                        "Reached test limit of %d items during %s phase.",
                        test_limit,
                        phase_name,
                    )
                    break

                num_in_active_queue = len(current_processing_queue)
                current_gather_size = num_in_active_queue
                if max_gather_batch is not None:
                    current_gather_size = min(current_gather_size, max_gather_batch)

                if is_test_mode:
                    remaining_in_test_limit = test_limit - processed_count
                    current_gather_size = min(
                        current_gather_size, remaining_in_test_limit
                    )

                if current_gather_size <= 0:
                    break

                current_batch_ids = [
                    current_processing_queue.popleft()
                    for _ in range(current_gather_size)
                ]
                launched_task_ids.update(current_batch_ids)

                tasks = [
                    _process_single_statement_group_with_llm(
                        gid,
                        all_group_data,
                        group_predecessors,
                        generated_group_translations,
                        client,
                        llm_semaphore,
                        prompt_template,
                        is_test_mode,
                    )
                    for gid in current_batch_ids
                ]
                results_from_gather = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                processed_count += len(current_batch_ids)

                batch_succeeded_ids: List[int] = []
                for res_item in results_from_gather:
                    if isinstance(res_item, Exception):
                        translation_failure_count += 1
                        logger.error(
                            "Asyncio.gather task raised an exception: %s", res_item
                        )
                        continue
                    res_group_id, translation, error_obj = res_item
                    if error_obj:
                        translation_failure_count += 1
                    elif translation:
                        generated_group_translations[res_group_id] = translation
                        translation_success_count += 1
                        batch_succeeded_ids.append(res_group_id)
                        if not is_test_mode:
                            pending_db_updates.append((res_group_id, translation))
                    else:  # No translation, no error (e.g. skipped, empty LLM response)
                        translation_failure_count += 1

                if pbar:
                    pbar.update(
                        len(current_batch_ids)
                    )  # Update based on tasks launched
                cost_str = f"${client.cost_tracker.get_total_cost():.4f}"
                pbar.set_postfix_str(
                    f"Cost: {cost_str} | OK: {translation_success_count} | "
                    f"Fail: {translation_failure_count} | "
                    f"{mode_desc_str.split(' [')[0]}"
                )

                if not is_test_mode and len(pending_db_updates) >= commit_batch_size:
                    logger.info(
                        "Committing batch of %d group updates...",
                        len(pending_db_updates),
                    )
                    update_stmt_mappings = [
                        {"id": gid, "informal_description": desc}
                        for gid, desc in pending_db_updates
                    ]
                    session_obj_for_processing.execute(
                        update(StatementGroup), update_stmt_mappings
                    )
                    session_obj_for_processing.commit()
                    pending_db_updates.clear()

                # Kahn's algorithm: update successors only in topological phase
                if phase_name == "Topological":
                    for proc_gid in batch_succeeded_ids:
                        for successor_gid in group_successors.get(proc_gid, []):
                            if successor_gid in group_in_degree:
                                group_in_degree[successor_gid] -= 1
                                if group_in_degree[successor_gid] == 0:
                                    if (
                                        successor_gid not in launched_task_ids
                                        and successor_gid
                                        not in current_processing_queue
                                    ):
                                        current_processing_queue.append(successor_gid)

            if phase_name == "Topological":  # End of topological while loop
                logger.info(
                    "Topological sort phase finished. Items attempted in this "
                    "phase: %d.",
                    len(launched_task_ids),
                )

            # Phase 2: Fallback for items not launched by topological sort
            unlaunched_ids = groups_to_process_ids - launched_task_ids
            if unlaunched_ids:
                if is_test_mode and processed_count >= test_limit:
                    logger.info(
                        "Test limit reached, skipping fallback phase for %d items.",
                        len(unlaunched_ids),
                    )
                else:
                    logger.warning(
                        "%d group(s) remain unprocessed after topological sort, "
                        "likely due to cycles. Attempting to process these now by "
                        "breaking cycles heuristically (processing in ID order).",
                        len(unlaunched_ids),
                    )
                    phase_name = (
                        "Fallback"  # For logging or conditional behavior if any
                    )
                    current_processing_queue = deque(
                        sorted(list(unlaunched_ids))
                    )  # New queue for fallback

                    # Repeat the processing loop structure for the fallback queue
                    while current_processing_queue:  # Loop for fallback items
                        if is_test_mode and processed_count >= test_limit:
                            logger.info(
                                "Reached test limit of %d items during %s phase.",
                                test_limit,
                                phase_name,
                            )
                            break

                        num_in_active_queue = len(current_processing_queue)
                        current_gather_size = num_in_active_queue
                        if max_gather_batch is not None:
                            current_gather_size = min(
                                current_gather_size, max_gather_batch
                            )

                        if is_test_mode:
                            remaining_in_test_limit = test_limit - processed_count
                            current_gather_size = min(
                                current_gather_size, remaining_in_test_limit
                            )

                        if current_gather_size <= 0:
                            break

                        current_batch_ids = [
                            current_processing_queue.popleft()
                            for _ in range(current_gather_size)
                        ]
                        # These IDs are from unlaunched_ids, so they add to
                        # launched_task_ids now
                        launched_task_ids.update(current_batch_ids)

                        tasks = [
                            _process_single_statement_group_with_llm(
                                gid,
                                all_group_data,
                                group_predecessors,
                                generated_group_translations,
                                client,
                                llm_semaphore,
                                prompt_template,
                                is_test_mode,
                            )
                            for gid in current_batch_ids
                        ]
                        results_from_gather = await asyncio.gather(
                            *tasks, return_exceptions=True
                        )

                        processed_count += len(current_batch_ids)

                        # Re-use result processing logic
                        for res_item in results_from_gather:
                            if isinstance(res_item, Exception):
                                translation_failure_count += 1
                                logger.error(
                                    "Asyncio.gather task raised an exception during "
                                    "fallback: %s",
                                    res_item,
                                )
                                continue
                            res_group_id, translation, error_obj = res_item
                            if error_obj:
                                translation_failure_count += 1
                            elif translation:
                                generated_group_translations[res_group_id] = translation
                                translation_success_count += 1
                                # batch_succeeded_ids is not used for Kahn's here
                                if not is_test_mode:
                                    pending_db_updates.append(
                                        (res_group_id, translation)
                                    )
                            else:
                                translation_failure_count += 1

                        if pbar:
                            pbar.update(len(current_batch_ids))
                        cost_str = f"${client.cost_tracker.get_total_cost():.4f}"
                        pbar.set_postfix_str(
                            f"Cost: {cost_str} | OK: {translation_success_count} | "
                            f"Fail: {translation_failure_count} | "
                            f"{mode_desc_str.split(' [')[0]}"
                        )

                        if (
                            not is_test_mode
                            and len(pending_db_updates) >= commit_batch_size
                        ):
                            logger.info(
                                "Committing batch of %d group updates (fallback)...",
                                len(pending_db_updates),
                            )
                            update_stmt_mappings = [
                                {"id": gid, "informal_description": desc}
                                for gid, desc in pending_db_updates
                            ]
                            session_obj_for_processing.execute(
                                update(StatementGroup), update_stmt_mappings
                            )
                            session_obj_for_processing.commit()
                            pending_db_updates.clear()
                    logger.info("Fallback processing phase finished.")

            # Final commit for any remaining updates
            if not is_test_mode and pending_db_updates:
                logger.info(
                    "Committing final batch of %d group updates...",
                    len(pending_db_updates),
                )
                final_update_mappings = [
                    {"id": gid, "informal_description": desc}
                    for gid, desc in pending_db_updates
                ]
                session_obj_for_processing.execute(
                    update(StatementGroup), final_update_mappings
                )
                session_obj_for_processing.commit()
                pending_db_updates.clear()

    except SQLAlchemyError as db_err:
        logger.error("Database error during processing: %s", db_err, exc_info=True)
        if session_obj_for_processing and session_obj_for_processing.is_active:
            session_obj_for_processing.rollback()
        return False
    except Exception as e:
        logger.error("Unexpected error during group processing: %s", e, exc_info=True)
        if session_obj_for_processing and session_obj_for_processing.is_active:
            session_obj_for_processing.rollback()
        return False
    finally:
        if pbar:
            pbar.close()
            print()  # For newline after tqdm
        if client and client.cost_tracker:
            summary = client.cost_tracker.get_summary()
            logger.info("--- Final Estimated LLM Costs (%s) ---", mode_desc_str)
            logger.info(
                "Total Estimated Cost: $%.4f", summary.get("total_estimated_cost", 0.0)
            )
            logger.info(
                "LLM Calls Attempted (Processed Count): %d", processed_count
            )  # Total tasks created
            logger.info("LLM Translations Succeeded: %d", translation_success_count)
            logger.info(
                "LLM Calls Failed or Yielded No Translation: %d",
                translation_failure_count,
            )
            for model_name, usage in summary.get("usage_by_model", {}).items():
                logger.info(
                    "    Model: %s -> Calls: %d, Cost: $%.4f",
                    model_name,
                    usage.get("calls", 0),
                    usage.get("estimated_cost", 0.0),
                )
            logger.info("------------------------------------------")

    if is_test_mode:
        logger.info(
            "Test mode finished. Attempted %d items (limit was %d). Successes: %d, "
            "Failures: %d.",
            processed_count,
            test_limit or 0,
            translation_success_count,
            translation_failure_count,
        )
        return True

    # In live mode, success means all initially targeted groups were ATTEMPTED.
    if processed_count >= total_to_process_initially:
        if (
            translation_failure_count == 0
            and translation_success_count == total_to_process_initially
        ):
            logger.info(
                "Live mode: Successfully processed and translated all %d "
                "identified groups.",
                total_to_process_initially,
            )
        elif (
            translation_success_count + translation_failure_count
            == total_to_process_initially
        ):  # All attempted had some outcome
            logger.warning(
                "Live mode: Attempted all %d identified groups. %d translations "
                "succeeded, %d LLM calls failed/yielded no translation.",
                total_to_process_initially,
                translation_success_count,
                translation_failure_count,
            )
        else:  # Should not happen if processed_count >= total_to_process_initially
            logger.info(
                "Live mode: Processed %d groups. Successes: %d, Failures: %d. "
                "Initial total: %d",
                processed_count,
                translation_success_count,
                translation_failure_count,
                total_to_process_initially,
            )

        return True  # All were attempted

    logger.error(
        "Live mode: Processing INCOMPLETE. Attempted (%d) != Initial Total (%d). "
        "Successes: %d, Failures: %d. Check logs for errors.",
        processed_count,
        total_to_process_initially,
        translation_success_count,
        translation_failure_count,
    )
    return False


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script.

    Returns:
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate informal English descriptions for Lean StatementGroups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-url", type=str, required=True, help="SQLAlchemy database URL."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Database commit batch size (for live mode).",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Enable test mode: process at most N items. LLM calls are active, "
        "but no database writes are performed.",
    )
    parser.add_argument(
        "--startover",
        action="store_true",
        help="Ignore existing translations in the database and re-translate all "
        "applicable StatementGroups.",
    )
    return parser.parse_args()


async def main_async():
    """Asynchronous main function to run the processing logic."""
    args = parse_arguments()

    startover_msg_part = " [STARTOVER ENABLED]" if args.startover else ""
    if args.test is not None:
        logger.info(
            "--- TEST MODE (Max %d items, LLM ON, DB OFF)%s ---",
            args.test,
            startover_msg_part,
        )
    elif args.startover:  # Only log if not already covered by test mode log
        logger.info("--- LIVE MODE%s ---", startover_msg_part)
    else:
        logger.info("--- LIVE MODE ---")

    if args.db_url.startswith("sqlite:///"):
        db_path_str = args.db_url.split("sqlite:///", 1)[1]
        if db_path_str != ":memory:":
            db_file = Path(db_path_str).resolve()
            logger.info("Using SQLite database: %s", db_file)
            if (
                not db_file.exists() and not args.test
            ):  # Only warn if not test mode, as test might not need existing DB
                logger.warning("SQLite database file not found: %s", db_file)
        else:
            logger.info("Using in-memory SQLite database.")

    success = await process_statement_groups_with_llm(
        db_url=args.db_url,
        commit_batch_size=args.batch_size,
        test_limit=args.test,
        startover=args.startover,
    )

    current_mode_desc = "Test Mode" if args.test is not None else "Live Mode"
    if success:
        logger.info(
            "--- Processing Completed (%s%s) ---", current_mode_desc, startover_msg_part
        )
        sys.exit(0)
    else:
        logger.error(
            "--- Processing FAILED or Incomplete (%s%s) ---",
            current_mode_desc,
            startover_msg_part,
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user (KeyboardInterrupt).")
        sys.exit(130)
    except Exception as e:  # pylint: disable=broad-except
        logger.critical(
            "An unexpected critical error occurred in main execution: %s",
            e,
            exc_info=True,
        )
        sys.exit(1)
