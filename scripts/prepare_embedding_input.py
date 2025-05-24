# File: scripts/prepare_embedding_input.py

"""Prepares a JSON file from StatementGroups for embedding generation.

Connects to a database, queries StatementGroups, and for each group,
conditionally generates entries for its Lean code representation,
informal English description, and docstring. Entries are only created if the
respective text content is non-empty and not excluded by command-line flags.

The output JSON file contains objects with 'id', 'source_statement_group_id',
'text_type' ('lean', 'informal_description', or 'docstring'), and 'text' fields.
This file serves as input for subsequent embedding generation.
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

# --- Dependency Imports ---
try:
    from sqlalchemy import create_engine, select
    from sqlalchemy.exc import OperationalError, SQLAlchemyError
    from sqlalchemy.orm import sessionmaker
    from tqdm import tqdm
except ImportError as e:
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please install them by running: pip install sqlalchemy tqdm",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Project Model & Config Imports ---
try:
    from .config import APP_CONFIG
    from lean_explore.shared.models.db import StatementGroup
except ImportError as e:
    print(
        f"Error: Could not import project modules (StatementGroup, APP_CONFIG): {e}\n"
        "Ensure 'lean_explore' is installed (e.g., 'pip install -e .') "
        "and all dependencies are met.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Suppress verbose SQLAlchemy logging unless specifically enabled via script's log level
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# --- Path Constants ---
# Assumes this script is in 'scripts/', project root is one level up.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_FILENAME = "embedding_input.json"
DEFAULT_OUTPUT_FILE_PATH = PROJECT_ROOT / "data" / DEFAULT_OUTPUT_FILENAME


def determine_lean_text(statement_group: StatementGroup) -> str:
    """Determines the Lean text to use for embedding from a StatementGroup.

    Prioritizes `display_statement_text` if available and non-empty.
    Otherwise, falls back to `statement_text`.

    Args:
        statement_group: The StatementGroup ORM object.

    Returns:
        str: The selected Lean text, stripped of whitespace, or an empty
        string if no suitable text is found.
    """
    lean_text = ""
    if statement_group.display_statement_text:
        stripped_display_text = statement_group.display_statement_text.strip()
        if stripped_display_text:
            lean_text = stripped_display_text

    if not lean_text and statement_group.statement_text:
        stripped_statement_text = statement_group.statement_text.strip()
        if stripped_statement_text:
            lean_text = stripped_statement_text
    return lean_text


def prepare_data(
    db_url: str,
    output_file_path: pathlib.Path,
    limit: Optional[int] = None,
    exclude_lean: bool = False,
    exclude_english: bool = False,
    exclude_docstrings: bool = False,
) -> None:
    """Fetches StatementGroups and prepares a JSON file for embedding.

    For each StatementGroup, this function conditionally creates entries for
    Lean code, informal descriptions, and docstrings, based on content
    availability and exclusion flags.

    Args:
        db_url: SQLAlchemy database URL.
        output_file_path: Path to write the output JSON file.
        limit: Optional maximum number of StatementGroups to process.
        exclude_lean: If True, Lean code text will not be included.
        exclude_english: If True, informal English descriptions will not be included.
        exclude_docstrings: If True, docstrings will not be included.

    Returns:
        None. The function writes to a file or exits on error.
    """
    db_url_display = f"...{db_url[-30:]}" if len(db_url) > 30 else db_url
    logger.info("Starting data preparation for embedding input.")
    logger.info("Database URL: %s", db_url_display)
    logger.info("Output file: %s", output_file_path)
    if limit:
        logger.info("Processing limit: %d StatementGroups", limit)
    if exclude_lean:
        logger.info("Excluding Lean code text.")
    if exclude_english:
        logger.info("Excluding informal English descriptions.")
    if exclude_docstrings:
        logger.info("Excluding docstrings.")

    engine = None
    try:
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        output_records: List[Dict[str, Any]] = []

        with SessionLocal() as session:
            query = select(StatementGroup).order_by(StatementGroup.id)
            if limit:
                query = query.limit(limit)

            logger.info("Fetching StatementGroups from the database...")
            statement_groups = session.execute(query).scalars().all()

            if not statement_groups:
                logger.info("No StatementGroups found. Writing empty JSON array.")
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
                logger.info("Empty JSON array written to %s", output_file_path)
                return

            logger.info("Processing %d StatementGroups...", len(statement_groups))
            for sg_obj in tqdm(statement_groups, desc="Preparing records", unit="sg"):
                sg_id: int = sg_obj.id

                # 1. Handle Lean Code Text
                if not exclude_lean:
                    lean_text_content = determine_lean_text(sg_obj)
                    if lean_text_content:
                        output_records.append(
                            {
                                "id": f"sg_{sg_id}_lean",
                                "source_statement_group_id": sg_id,
                                "text_type": "lean",
                                "text": lean_text_content,
                            }
                        )
                    else:
                        logger.debug(
                            "SG ID %d: No non-empty lean text found. Skipping entry.",
                            sg_id,
                        )
                elif exclude_lean:  # Explicitly log if excluded by flag
                    logger.debug(
                        "SG ID %d: Lean text excluded by flag. Skipping lean entry.",
                        sg_id,
                    )

                # 2. Handle Informal Description Text
                if not exclude_english:
                    informal_text_content = ""
                    if sg_obj.informal_description:
                        informal_text_content = sg_obj.informal_description.strip()

                    if informal_text_content:
                        output_records.append(
                            {
                                "id": f"sg_{sg_id}_informal",
                                "source_statement_group_id": sg_id,
                                "text_type": "informal_description",
                                "text": informal_text_content,
                            }
                        )
                    else:
                        logger.debug(
                            "SG ID %d: No non-empty informal description found. "
                            "Skipping entry.",
                            sg_id,
                        )
                elif exclude_english:
                    logger.debug(
                        "SG ID %d: Informal description excluded by flag. "
                        "Skipping entry.",
                        sg_id,
                    )

                # 3. Handle Docstring Text
                if not exclude_docstrings:
                    docstring_text_content = ""
                    if sg_obj.docstring:
                        docstring_text_content = sg_obj.docstring.strip()

                    if docstring_text_content:
                        output_records.append(
                            {
                                "id": f"sg_{sg_id}_docstring",
                                "source_statement_group_id": sg_id,
                                "text_type": "docstring",
                                "text": docstring_text_content,
                            }
                        )
                    else:
                        logger.debug(
                            "SG ID %d: No non-empty docstring found. Skipping entry.",
                            sg_id,
                        )
                elif exclude_docstrings:
                    logger.debug(
                        "SG ID %d: Docstring excluded by flag. Skipping entry.",
                        sg_id,
                    )

        logger.info("Prepared %d records for JSON output.", len(output_records))

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_records, f, indent=2)

        logger.info("Successfully wrote embedding input data to %s", output_file_path)

    except OperationalError as e:
        logger.error("Database connection failed or operational error: %s", e)
        logger.error(
            "Please check the database URL and ensure the database server is running."
        )
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error(
            "A database error occurred during data preparation: %s", e, exc_info=True
        )
        sys.exit(1)
    except OSError as e:
        logger.error(
            "File I/O error when writing to %s: %s", output_file_path, e, exc_info=True
        )
        sys.exit(1)
    except Exception as e:
        logger.critical("An unexpected critical error occurred: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare JSON input for embedding generation from StatementGroups, "
            "allowing selective inclusion of Lean code, informal descriptions, "
            "and docstrings."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # APP_CONFIG should be loaded by this point from the project imports
    db_url_default = APP_CONFIG.get("database", {}).get("url")

    parser.add_argument(
        "--db-url",
        type=str,
        default=db_url_default,
        help="SQLAlchemy database URL. Overrides application config if provided.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_FILE_PATH,
        help="Path for the output JSON file (default: data/embedding_input.json).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Optional: Limit the number of StatementGroups to process.",
    )
    parser.add_argument(
        "--exclude-lean",
        action="store_true",
        help="Exclude Lean code text from the output.",
    )
    parser.add_argument(
        "--exclude-english",
        action="store_true",
        help="Exclude informal English descriptions from the output.",
    )
    parser.add_argument(
        "--exclude-docstrings",
        action="store_true",
        help="Exclude docstrings from the output.",
    )

    args = parser.parse_args()
    if not args.db_url:
        logger.error(
            "Database URL is required. Provide via --db-url or in application config."
        )
        sys.exit(1)
    return args


if __name__ == "__main__":
    cli_args = parse_arguments()

    # Resolve output file path to ensure it's absolute before passing
    resolved_output_file = cli_args.output_file.resolve()

    prepare_data(
        db_url=cli_args.db_url,
        output_file_path=resolved_output_file,
        limit=cli_args.limit,
        exclude_lean=cli_args.exclude_lean,
        exclude_english=cli_args.exclude_english,
        exclude_docstrings=cli_args.exclude_docstrings,
    )
    logger.info("--- Embedding input preparation finished ---")
