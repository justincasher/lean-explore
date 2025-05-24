# File: src/lean_explore/population/phase4_tasks.py

"""Population Phase 4: Processing and storing inter-declaration dependencies.

Reads declaration dependencies from a 'dependencies.jsonl' file,
maps Lean names to database IDs (using a provided map from Phase 1),
and populates the 'dependencies' table in the database. Handles batch
commits and checks for existing dependencies if not in `create_tables` mode
(i.e., if not performing a fresh database population).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker  # Session for type hinting
from tqdm import tqdm

from lean_explore.shared.models.db import Dependency

logger = logging.getLogger(__name__)

# --- Phase 4 Main Function ---


def populate_dependencies(
    session_factory: sessionmaker[Session],
    dependencies_file: Path,
    lean_name_to_db_id: Dict[str, int],
    batch_size: int,
    create_tables_flag: bool,
) -> bool:
    """Populates the 'dependencies' table from a 'dependencies.jsonl' file.

    Processes a JSONL file where each line represents a dependency between two
    Lean declarations. It uses a pre-built mapping of Lean names to database IDs
    (from Phase 1) to link the source and target declarations in the database.

    Args:
        session_factory: SQLAlchemy session factory for creating database sessions.
        dependencies_file: Path to the 'dependencies.jsonl' input file.
        lean_name_to_db_id: A dictionary mapping fully qualified Lean names to
            their corresponding primary key IDs in the 'declarations' table.
        batch_size: The number of dependency records to process before committing
            them to the database in a single batch.
        create_tables_flag: A boolean flag. If True, assumes database tables were
            newly created in this run, and skips fetching existing dependencies.
            If False, it attempts to fetch existing dependencies to avoid
            inserting duplicates, relying on database constraints as a fallback.

    Returns:
        bool: True if the dependency population process completed successfully
        (which can include skipping items due to non-critical errors like
        missing declaration IDs or duplicate entries). False if a critical,
        unrecoverable error occurred (e.g., file not found, major DB issue).
    """
    logger.info(
        "Starting Phase 4: Population of dependencies from %s...", dependencies_file
    )
    if not dependencies_file.is_file():
        logger.error("Dependencies file not found: %s", dependencies_file)
        return False
    if not lean_name_to_db_id:
        logger.warning(
            "Phase 4: Dependency population called with empty lean_name_to_db_id map. "
            "No dependencies can be added. This is normal if Phase 1 found no "
            "declarations."
        )
        return True  # Not a failure of Phase 4 itself

    try:
        total_lines = 0
        with open(dependencies_file, encoding="utf-8") as f_count:
            for _ in f_count:
                total_lines += 1
        if total_lines == 0:
            logger.info(
                "Dependencies file %s is empty. Skipping Phase 4 population.",
                dependencies_file,
            )
            return True
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Error counting lines in %s: %s", dependencies_file, e, exc_info=True
        )
        return False

    added_count = 0
    skipped_missing_decl_id = 0
    skipped_duplicates_integrity = 0
    log_sample_frequency = 10000

    existing_deps_set: Set[Tuple[int, int, str]] = set()
    if not create_tables_flag:
        try:
            with session_factory() as session:
                logger.info("Phase 4: Fetching existing dependencies from database...")
                results = session.execute(
                    select(
                        Dependency.source_decl_id,
                        Dependency.target_decl_id,
                        Dependency.dependency_type,
                    )
                )
                existing_deps_set = {(r[0], r[1], r[2]) for r in results}  # type: ignore[misc]
            logger.info(
                "Phase 4: Found %d existing dependencies in the database.",
                len(existing_deps_set),
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Phase 4: Could not fetch existing dependencies: %s. "
                "Proceeding without pre-check for duplicates; relying on DB "
                "constraints.",
                e,
            )

    try:
        with open(
            dependencies_file, encoding="utf-8"
        ) as f_deps, session_factory() as session:
            with tqdm(
                total=total_lines, desc="Phase 4: Dependencies", unit="line"
            ) as pbar:
                current_batch_to_add_payloads: List[Dict[str, Any]] = []

                for i, line in enumerate(f_deps):
                    pbar.update(1)
                    log_this_sample = (i + 1) % log_sample_frequency == 0

                    if log_this_sample:
                        logger.debug(
                            "[Phase 4 Sample Record #%d] Raw JSON: %s",
                            i + 1,
                            line.strip()[:300],
                        )
                    try:
                        data: Dict[str, Any] = json.loads(line)
                        source_lean_name = data.get("source_lean_name")
                        target_lean_name = data.get("target_lean_name")
                        dep_type = data.get(
                            "dependency_type", "Direct"
                        )  # Default if missing

                        if not source_lean_name or not target_lean_name:
                            logger.warning(
                                "Skipping dependency line %d: Missing source or "
                                "target lean_name. Line: %s",
                                i + 1,
                                line.strip(),
                            )
                            continue

                        source_id = lean_name_to_db_id.get(source_lean_name)
                        target_id = lean_name_to_db_id.get(target_lean_name)

                        if log_this_sample:
                            logger.debug(
                                "  [Phase 4 Sample Record #%d] Source: '%s' (ID: %s), "
                                "Target: '%s' (ID: %s), Type: '%s'",
                                i + 1,
                                source_lean_name,
                                source_id,
                                target_lean_name,
                                target_id,
                                dep_type,
                            )

                        if source_id is None or target_id is None:
                            skipped_missing_decl_id += 1
                            if (
                                skipped_missing_decl_id <= 20
                            ):  # Log first few, then suppress
                                logger.debug(
                                    "  Skipping dependency from '%s' to '%s' "
                                    "(line %d): "
                                    "DB ID not found for source or target.",
                                    source_lean_name,
                                    target_lean_name,
                                    i + 1,
                                )
                            elif skipped_missing_decl_id == 21:
                                logger.debug(
                                    "  (Further 'DB ID not found' messages will be "
                                    "suppressed)"
                                )
                            continue

                        dep_tuple = (source_id, target_id, dep_type)
                        if dep_tuple in existing_deps_set:
                            if log_this_sample:
                                logger.debug(
                                    "  Dependency %s already exists. Skipping.",
                                    dep_tuple,
                                )
                            skipped_duplicates_integrity += 1
                            continue

                        dep_payload = {
                            "source_decl_id": source_id,
                            "target_decl_id": target_id,
                            "dependency_type": dep_type,
                        }
                        current_batch_to_add_payloads.append(dep_payload)
                        existing_deps_set.add(
                            dep_tuple
                        )  # Add to set to avoid duplicate attempts within run

                        if len(current_batch_to_add_payloads) >= batch_size:
                            session.bulk_insert_mappings(
                                Dependency, current_batch_to_add_payloads
                            )
                            session.commit()
                            added_count += len(current_batch_to_add_payloads)
                            pbar.set_postfix_str(
                                f"Batch adds: {len(current_batch_to_add_payloads)}",
                                refresh=False,
                            )
                            current_batch_to_add_payloads = []

                    except json.JSONDecodeError:
                        logger.error(
                            "Failed to parse JSON dependency at line %d in %s: %s",
                            i + 1,
                            dependencies_file,
                            line.strip(),
                        )
                    except IntegrityError:
                        session.rollback()
                        logger.warning(
                            "IntegrityError for dependency at line %d (likely "
                            "duplicate). Source: '%s', Target: '%s'",
                            i + 1,
                            source_lean_name,
                            target_lean_name,
                        )
                        skipped_duplicates_integrity += 1
                    except SQLAlchemyError as e_item:
                        session.rollback()
                        logger.error(
                            "Database error processing dependency line %d: %s. "
                            "Skipping item.",
                            i + 1,
                            e_item,
                        )
                    except Exception as e_item_unexpected:  # pylint: disable=broad-except
                        session.rollback()
                        logger.error(
                            "Unexpected error for dependency line %d: %s. "
                            "Skipping item.",
                            i + 1,
                            e_item_unexpected,
                            exc_info=True,
                        )

                # Final batch commit
                if current_batch_to_add_payloads:
                    session.bulk_insert_mappings(
                        Dependency, current_batch_to_add_payloads
                    )
                    session.commit()
                    added_count += len(current_batch_to_add_payloads)

                pbar.set_postfix_str(
                    f"Adds: {added_count}, Skips: {skipped_missing_decl_id} + "
                    f"{skipped_duplicates_integrity}",
                    refresh=True,
                )

            logger.info(
                "Phase 4 Summary: Processed %d lines from %s. "
                "Added %d new dependencies. "
                "Skipped (missing declaration ID): %d. "
                "Skipped (duplicate/integrity error): %d.",
                total_lines,
                dependencies_file,
                added_count,
                skipped_missing_decl_id,
                skipped_duplicates_integrity,
            )
            return True

    except FileNotFoundError:  # Should be caught by initial check
        logger.error(
            "Dependencies file not found during Phase 4 execution: %s",
            dependencies_file,
        )
    except SQLAlchemyError as e_db_critical:
        logger.error(
            "A critical database error occurred during Phase 4: %s",
            e_db_critical,
            exc_info=True,
        )
    except Exception as e_critical:  # pylint: disable=broad-except
        logger.error(
            "A critical unexpected error occurred during Phase 4: %s",
            e_critical,
            exc_info=True,
        )
    return False
