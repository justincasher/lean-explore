"""Populates a relational database from Lean declaration and dependency files.

This script orchestrates a multi-phase database population process:
1.  Initial population of declarations (from 'declarations.jsonl').
2.  Refinement of declaration source locations and statement text using
    .ast.json and .lean source files (from toolchain and Lake packages).
3.  Grouping of declarations into StatementGroups and determination of primary
    declarations for these groups.
4.  Population of inter-declaration dependencies (from 'dependencies.jsonl').

Database connection details and file paths can be provided via command-line
arguments. Lean toolchain source path can be auto-detected from
'extractor/extractor_config.json' or overridden.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import model and phase-specific functions ---
try:
    from dev_tools.population.phase1_tasks import phase1_populate_declarations_initial
    from dev_tools.population.phase2_tasks import (
        phase2_refine_declarations_source_info,
    )
    from dev_tools.population.phase3_tasks import phase3_group_statements
    from dev_tools.population.phase4_tasks import populate_dependencies
    from dev_tools.population.phase5_tasks import (
        populate_statement_group_dependencies,
    )
    from lean_explore.shared.models.db import Base  # For table creation/dropping
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Could not import necessary modules: {e}\n\n"
        "POSSIBLE CAUSES & SOLUTIONS:\n"
        "1. RUNNING LOCATION: Ensure you are running this script from the project's"
        " ROOT directory,\n"
        "   e.g., 'python scripts/populate_db.py'.\n\n"
        "2. FOR 'lean_explore' (e.g., models):\n"
        "   - RECOMMENDED: The 'lean_explore' package might not be installed in your"
        " current Python environment.\n"
        "     From your project root, run: 'pip install -e .'\n"
        "     This installs it in editable mode, making it available.\n"
        "   - Alternatively (if not using editable install), ensure the 'src/'"
        " directory is correctly added to sys.path.\n\n"
        "3. FOR 'dev_tools.population' (e.g., phaseX_tasks):\n"
        "   - The 'dev_tools/' directory (containing 'population/' and '__init__.py'"
        " files)\n"
        "     must be at the project root.\n"
        "   - The script attempts to add the project root to sys.path; verify this is"
        " working for your environment.\n"
        "     You might need to set your PYTHONPATH environment variable to include the"
        " project root.\n\n"
        f"Current sys.path: {sys.path}",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Default Path Constants (relative to project root) ---
DATA_DIR = PROJECT_ROOT / "data"

# Default locations for primary data files
DEFAULT_DECLARATIONS_FILE = DATA_DIR / "declarations.jsonl"
DEFAULT_DEPENDENCIES_FILE = DATA_DIR / "dependencies.jsonl"
DEFAULT_AST_JSON_BASE_ABS = DATA_DIR / "AST"
DEFAULT_DB_URL_STR = "sqlite:///data/lean_explore_data.db"

# Base paths related to a potential 'extractor' Lean project and its artifacts.
# Used for default Lake packages location.
EXTRACTOR_PROJECT_ROOT = PROJECT_ROOT / "extractor"
DEFAULT_LAKE_PACKAGES_SRC_BASE = EXTRACTOR_PROJECT_ROOT / ".lake" / "packages"

# --- Logging Setup ---
# Logger instance is created in main() after parsing args for log level.
# A module-level logger can be obtained using logging.getLogger(__name__) if needed
# elsewhere.


def get_toolchain_src_from_config(
    project_root: Path, config_logger: logging.Logger
) -> Optional[Path]:
    """Reads and derives the Lean toolchain source base from extractor_config.json.

    The path in the config ('toolchainCoreSrcDir') typically points to '.../src',
    but core modules like Init, Lean, Std are usually under '.../src/lean/'.
    This function appends 'lean' to the configured path.

    Args:
        project_root: The root directory of the 'lean-explore' project.
        config_logger: Logger instance for logging messages during config processing.

    Returns:
        Optional[Path]: The derived absolute path to the toolchain source base
                        (e.g., '.../src/lean/'), or None if not found or invalid.
    """
    config_file = project_root / "extractor" / "extractor_config.json"
    if config_file.is_file():
        try:
            with open(config_file, encoding="utf-8") as f:
                config_data = json.load(f)
            raw_path_str = config_data.get("toolchainCoreSrcDir")
            if raw_path_str:
                # Core modules (Init, Lean, Std) are under the 'lean' subdirectory
                toolchain_base = Path(raw_path_str) / "lean"
                resolved_toolchain_base = toolchain_base.expanduser().resolve()
                if resolved_toolchain_base.is_dir():
                    config_logger.info(
                        "Successfully read and derived toolchain source base from %s:"
                        " %s",
                        config_file,
                        resolved_toolchain_base,
                    )
                    return resolved_toolchain_base
                else:
                    config_logger.warning(
                        "Path derived from 'toolchainCoreSrcDir' in %s ('%s') is not"
                        " a valid directory.",
                        config_file,
                        resolved_toolchain_base,
                    )
            else:
                config_logger.warning(
                    "Key 'toolchainCoreSrcDir' not found in %s.", config_file
                )
        except json.JSONDecodeError as jde:
            config_logger.warning("Error decoding JSON from %s: %s", config_file, jde)
        except Exception as e:  # pylint: disable=broad-except
            config_logger.warning("Error reading or processing %s: %s", config_file, e)
    else:
        config_logger.info(
            "Extractor config file not found at %s. Toolchain source path must be"
            " provided via --lean-toolchain-src if needed.",
            config_file,
        )
    return None


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the database population script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Populates a database from Lean project export files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=DEFAULT_DB_URL_STR,
        help="SQLAlchemy database URL. Defaults to an SQLite DB at"
        " 'data/lean_explore_data.db'.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Drop existing tables (if any) and create new ones. CAUTION: DELETES"
        " ALL EXISTING DATA IN THE TARGET DATABASE.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to batch before committing to the database.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the script and imported modules.",
    )
    parser.add_argument(
        "--declarations-file",
        type=Path,
        default=DEFAULT_DECLARATIONS_FILE,
        help="Path to the declarations.jsonl file.",
    )
    parser.add_argument(
        "--dependencies-file",
        type=Path,
        default=DEFAULT_DEPENDENCIES_FILE,
        help="Path to the dependencies.jsonl file.",
    )
    parser.add_argument(
        "--lean-toolchain-src",
        type=Path,
        default=None,  # Auto-detected from config if not provided
        help="Absolute base path to Lean toolchain's .lean source files (e.g., for"
        " Init, Lean, Std). Overrides auto-detection from"
        " 'extractor/extractor_config.json'. Path should point to the directory"
        " containing 'Init', 'Lean', 'Std' subdirectories (e.g., .../src/lean/).",
    )
    parser.add_argument(
        "--lean-lake-src",
        type=Path,
        default=DEFAULT_LAKE_PACKAGES_SRC_BASE,
        help="Absolute base path to the directory containing Lake package sources"
        " (e.g., 'mathlib', 'batteries'). Typically '.lake/packages/' within a Lean"
        " project. Used in Phase 2.",
    )
    parser.add_argument(
        "--ast-json-base",
        type=Path,
        default=DEFAULT_AST_JSON_BASE_ABS,
        help="Absolute base path to .ast.json files (e.g., 'data/AST/'). Used in"
        " Phase 2.",
    )
    return parser.parse_args()


def main():
    """Main execution function for the database population script.

    Orchestrates the multi-phase population process:
    1. Initial declaration population.
    2. Refinement of declaration source information using .lean and .ast.json files.
    3. Grouping of declarations into statement groups.
    4. Population of dependencies between declarations.
    """
    args = parse_arguments()

    # Setup main logger for this script
    # Configure logging for the entire application, including imported modules
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    script_logger = logging.getLogger(
        __name__
    )  # Logger specific to this script's messages

    script_logger.info("--- Starting Database Population Script (Multi-Phase) ---")
    db_url_display = (
        args.db_url
        if len(args.db_url) < 70
        else f"{args.db_url[:30]}...{args.db_url[-30:]}"
    )
    script_logger.info("Database URL: %s", db_url_display)
    script_logger.info("Create tables: %s", args.create_tables)
    script_logger.info("Batch size: %d", args.batch_size)
    script_logger.info("Log level: %s", args.log_level.upper())

    # Resolve paths and log them
    declarations_file_abs = args.declarations_file.resolve()
    dependencies_file_abs = args.dependencies_file.resolve()
    ast_json_base_abs = args.ast_json_base.resolve()

    script_logger.info("Declarations file: %s", declarations_file_abs)
    script_logger.info("Dependencies file: %s", dependencies_file_abs)
    script_logger.info("Base for .ast.json files: %s", ast_json_base_abs)

    # Determine Lean source paths
    lean_toolchain_src_abs: Optional[Path] = None
    if args.lean_toolchain_src:
        lean_toolchain_src_abs = args.lean_toolchain_src.resolve()
        script_logger.info(
            "Using user-provided Lean toolchain source base: %s",
            lean_toolchain_src_abs,
        )
    else:
        script_logger.info(
            "Attempting to determine Lean toolchain source base from"
            " 'extractor/extractor_config.json'..."
        )
        lean_toolchain_src_abs = get_toolchain_src_from_config(
            PROJECT_ROOT, script_logger
        )
        if not lean_toolchain_src_abs:
            script_logger.warning(
                "Lean toolchain source base not provided via --lean-toolchain-src and"
                " could not be determined from config. Phase 2 may not find toolchain"
                " .lean files."
            )

    lean_lake_packages_root_abs: Optional[Path] = None
    if args.lean_lake_src:
        lean_lake_packages_root_abs = args.lean_lake_src.resolve()
        script_logger.info(
            "Using Lean Lake packages source base: %s", lean_lake_packages_root_abs
        )
    else:
        script_logger.info(
            "Lean Lake packages source base not provided via --lean-lake-src. "
            "Phase 2 may not find .lean files from Lake packages."
        )

    # Perform critical path checks early
    paths_ok = True
    if not declarations_file_abs.is_file():
        script_logger.error(
            "Declarations file not found: %s. This file is required for Phase 1.",
            declarations_file_abs,
        )
        paths_ok = False
    if not dependencies_file_abs.is_file():
        # This is not immediately critical if only running early phases, but log
        # prominently.
        script_logger.warning(
            "Dependencies file not found: %s. Phase 4 (dependency population) will be"
            " impacted.",
            dependencies_file_abs,
        )
    if not ast_json_base_abs.is_dir():
        script_logger.error(
            ".ast.json base path not a directory: %s. This is required for Phase 2.",
            ast_json_base_abs,
        )
        paths_ok = False

    if not paths_ok:
        script_logger.critical("Essential path checks failed. Aborting.")
        sys.exit(1)

    # Further checks for optional/Phase 2 specific paths
    if lean_toolchain_src_abs and not lean_toolchain_src_abs.is_dir():
        script_logger.warning(
            "Specified/Detected Lean toolchain source base is not a directory: %s."
            " Phase 2 impact on toolchain files.",
            lean_toolchain_src_abs,
        )
    if lean_lake_packages_root_abs and not lean_lake_packages_root_abs.is_dir():
        script_logger.warning(
            "Specified Lake packages source base is not a directory: %s. Phase 2"
            " impact on package files.",
            lean_lake_packages_root_abs,
        )

    try:
        engine = create_engine(args.db_url, echo=(args.log_level.upper() == "DEBUG"))
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        if args.create_tables:
            script_logger.warning(
                "Option --create-tables: DROPPING existing tables and CREATING new"
                " ones."
            )
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            script_logger.info("Tables dropped and created successfully.")
        else:
            # Test connection by trying to connect
            with engine.connect():
                script_logger.info(
                    "Database connection verified (existing tables will "
                    "be used/updated)."
                )

        # --- Phase 1: Initial Declaration Population ---
        script_logger.info("--- Executing Phase 1: Initial Declaration Population ---")
        lean_name_to_id_map = phase1_populate_declarations_initial(
            SessionLocal, declarations_file_abs, args.batch_size, args.create_tables
        )
        if lean_name_to_id_map is None:
            script_logger.critical("Aborting: Critical error during Phase 1.")
            sys.exit(1)
        if not lean_name_to_id_map:
            script_logger.warning(
                "Phase 1: Declaration map is empty (no declarations processed or"
                " found). Subsequent phases might have limited or no data to process."
            )
        else:
            script_logger.info(
                "Phase 1 completed. Found/Processed %d declarations into map.",
                len(lean_name_to_id_map),
            )

        # --- Phase 2: Refine Declaration Source Info ---
        script_logger.info("--- Executing Phase 2: Refine Declaration Source Info ---")
        lean_source_bases_for_phase2: List[Path] = []
        if lean_toolchain_src_abs and lean_toolchain_src_abs.is_dir():
            lean_source_bases_for_phase2.append(lean_toolchain_src_abs)
        if lean_lake_packages_root_abs and lean_lake_packages_root_abs.is_dir():
            script_logger.info(
                "Scanning Lake packages directory for Phase 2 sources: %s",
                lean_lake_packages_root_abs,
            )
            for package_dir in lean_lake_packages_root_abs.iterdir():
                if package_dir.is_dir():
                    lean_source_bases_for_phase2.append(package_dir)
                    script_logger.debug(
                        "Added Lake package source base for Phase 2: %s", package_dir
                    )

        if not lean_source_bases_for_phase2:
            script_logger.warning(
                "Phase 2: No valid .lean source base paths configured or found. Source"
                " text refinement will likely be skipped for most declarations."
            )

        success_phase2 = phase2_refine_declarations_source_info(
            SessionLocal,
            lean_source_bases_for_phase2,
            ast_json_base_abs,
            args.batch_size,
            lean_toolchain_src_abs,
        )
        if not success_phase2:
            script_logger.warning(
                "Phase 2: Refinement may have encountered errors. Review logs."
                " Subsequent data may be impacted."
            )
        else:
            script_logger.info(
                "Phase 2: Declaration Source Info Refinement processing completed."
            )

        # --- Phase 3: Statement Grouping ---
        script_logger.info("--- Executing Phase 3: Statement Grouping ---")
        success_phase3 = phase3_group_statements(SessionLocal, args.batch_size)
        if not success_phase3:
            script_logger.warning(
                "Phase 3: Grouping may have encountered errors. Review logs."
                " Declarations may not be optimally grouped."
            )
        else:
            script_logger.info("Phase 3: Statement Grouping completed.")

        # --- Phase 4: Populate Dependencies ---
        script_logger.info("--- Executing Phase 4: Populate Dependencies ---")
        if not lean_name_to_id_map:  # Check if Phase 1 yielded any declarations
            script_logger.warning(
                "Skipping Phase 4: Dependency Population (declaration map from Phase 1"
                " is empty)."
            )
        elif not dependencies_file_abs.is_file():
            script_logger.warning(
                "Skipping Phase 4: Dependency Population (dependencies file not found"
                " at %s).",
                dependencies_file_abs,
            )
        else:
            success_phase4 = populate_dependencies(
                SessionLocal,
                dependencies_file_abs,
                lean_name_to_id_map,
                args.batch_size,
                args.create_tables,
            )
            if not success_phase4:
                script_logger.error(
                    "Population failed during Phase 4: Dependency Population. Review"
                    " logs."
                )
                # Decide if this should be a sys.exit(1) based on criticality
            else:
                script_logger.info(
                    "Phase 4: Dependency Population processing completed."
                )

        # --- Phase 5: Populate StatementGroup Dependencies ---
        script_logger.info(
            "--- Executing Phase 5: Populate StatementGroup Dependencies ---"
        )
        if (
            not success_phase3 or not success_phase4
        ):  # Or specific checks if these phases must succeed
            script_logger.warning(
                "Skipping Phase 5: StatementGroup Dependency Population due to failures"
                " or incomplete data in prior phases."
            )
        else:
            success_phase5 = populate_statement_group_dependencies(
                SessionLocal,
                args.batch_size,
                args.create_tables,  # Pass create_tables to allow skipping pre-fetch if
                # tables new
            )
            if not success_phase5:
                script_logger.error(
                    "Population failed during Phase 5: StatementGroup Dependency"
                    " Population. Review logs."
                )
                # Decide if this should be a sys.exit(1)
            else:
                script_logger.info(
                    "Phase 5: StatementGroup Dependency Population processing"
                    " completed."
                )

        script_logger.info("--- Database Population Script Finished ---")

    except SQLAlchemyError as e:
        script_logger.critical(
            "A top-level database error occurred: %s", e, exc_info=True
        )
        sys.exit(1)
    except (
        FileNotFoundError
    ) as e:  # Should be caught by earlier checks, but as a fallback
        script_logger.critical("A required file was not found: %s", e, exc_info=True)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        script_logger.critical(
            "An unexpected critical error occurred in main script execution: %s",
            e,
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
