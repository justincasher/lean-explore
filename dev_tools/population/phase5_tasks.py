# File: src/lean_explore/population/phase5_tasks.py

"""Population Phase 5: Deriving and storing dependencies between StatementGroups.

This phase processes existing declaration-level dependencies and statement group
assignments to infer and populate dependencies directly between StatementGroups.
"""

import logging
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from lean_explore.shared.models.db import Declaration, Dependency, StatementGroupDependency

logger = logging.getLogger(__name__)

DEFAULT_SG_DEPENDENCY_TYPE = "DerivedFromDecl"


def populate_statement_group_dependencies(
    session_factory: sessionmaker[Session],
    batch_size: int,
    create_tables_flag: bool,
) -> bool:
    """Populates the 'statement_group_dependencies' table.

    This function derives dependencies between StatementGroups based on
    existing declaration-level dependencies. If a declaration D1 in
    StatementGroup SG1 depends on a declaration D2 in StatementGroup SG2,
    and SG1 is different from SG2, then a dependency from SG1 to SG2
    is recorded.

    Args:
        session_factory: SQLAlchemy session factory for creating database sessions.
        batch_size: Number of records to process in a batch before committing.
        create_tables_flag: If True, assumes tables were just created and
            skips fetching existing statement group dependencies for comparison.

    Returns:
        bool: True if the process completed successfully, False on critical error.
    """
    logger.info("Starting Phase 5: Population of StatementGroup dependencies...")

    existing_sg_deps_set: Set[Tuple[int, int, str]] = set()
    if not create_tables_flag:
        try:
            with session_factory() as session:
                logger.info(
                    "Phase 5: Fetching existing statement_group_dependencies "
                    "from database..."
                )
                stmt_existing = select(
                    StatementGroupDependency.source_statement_group_id,
                    StatementGroupDependency.target_statement_group_id,
                    StatementGroupDependency.dependency_type,
                )
                results_existing = session.execute(stmt_existing).fetchall()
                existing_sg_deps_set = {
                    (sg_source_id, sg_target_id, dep_type)
                    for sg_source_id, sg_target_id, dep_type in results_existing
                }
                logger.info(
                    "Phase 5: Found %d existing statement_group_dependencies.",
                    len(existing_sg_deps_set),
                )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Phase 5: Could not fetch existing statement_group_dependencies: %s. "
                "Proceeding and relying on DB constraints for duplicates.",
                e,
            )
            existing_sg_deps_set.clear()  # Ensure it's empty if fetch failed

    decl_id_to_sg_id_map: Dict[int, int] = {}
    declaration_links: List[Tuple[int, int]] = []

    try:
        with session_factory() as session:
            logger.info("Phase 5: Fetching declaration to statement_group_id map...")
            stmt_map = select(Declaration.id, Declaration.statement_group_id).where(
                Declaration.statement_group_id.isnot(None)
            )
            results_map = session.execute(stmt_map).fetchall()
            decl_id_to_sg_id_map = {
                decl_id: sg_id
                for decl_id, sg_id in results_map
                if decl_id is not None and sg_id is not None
            }
            logger.info(
                "Phase 5: Built map for %d declarations linked to statement groups.",
                len(decl_id_to_sg_id_map),
            )

            if not decl_id_to_sg_id_map:
                logger.warning(
                    "Phase 5: No declarations found linked to statement groups. "
                    "Cannot derive group dependencies."
                )
                return True  # Not a failure of this phase, but no work to do.

            logger.info("Phase 5: Fetching declaration-level dependencies...")
            stmt_deps = select(Dependency.source_decl_id, Dependency.target_decl_id)
            declaration_links = session.execute(stmt_deps).fetchall()  # type: ignore
            logger.info(
                "Phase 5: Found %d declaration-level dependencies to process.",
                len(declaration_links),
            )
            if not declaration_links:
                logger.info(
                    "Phase 5: No declaration-level dependencies found. "
                    "No statement group dependencies to derive."
                )
                return True

    except SQLAlchemyError as e_fetch:
        logger.error(
            "Phase 5: Database error during initial data fetching: %s",
            e_fetch,
            exc_info=True,
        )
        return False
    except Exception as e_unexpected_fetch:  # pylint: disable=broad-except
        logger.error(
            "Phase 5: Unexpected error during initial data fetching: %s",
            e_unexpected_fetch,
            exc_info=True,
        )
        return False

    new_sg_dependency_payloads: List[Dict[str, Any]] = []
    processed_in_this_run_sg_deps_set: Set[Tuple[int, int, str]] = set()
    added_count = 0
    skipped_due_to_missing_sg_link = 0
    skipped_due_to_self_loop = 0
    skipped_as_existing_or_duplicate = 0

    try:
        with session_factory() as session:
            with tqdm(
                total=len(declaration_links),
                desc="Phase 5: Deriving SG Dependencies",
                unit="decl_link",
            ) as pbar:
                for source_decl_id, target_decl_id in declaration_links:
                    pbar.update(1)

                    source_sg_id = decl_id_to_sg_id_map.get(source_decl_id)
                    target_sg_id = decl_id_to_sg_id_map.get(target_decl_id)

                    if source_sg_id is None or target_sg_id is None:
                        skipped_due_to_missing_sg_link += 1
                        continue

                    if source_sg_id == target_sg_id:
                        skipped_due_to_self_loop += 1
                        # Skip dependencies within the same statement group
                        continue

                    sg_dep_tuple = (
                        source_sg_id,
                        target_sg_id,
                        DEFAULT_SG_DEPENDENCY_TYPE,
                    )

                    if (
                        sg_dep_tuple in existing_sg_deps_set
                        or sg_dep_tuple in processed_in_this_run_sg_deps_set
                    ):
                        skipped_as_existing_or_duplicate += 1
                        continue

                    payload = {
                        "source_statement_group_id": source_sg_id,
                        "target_statement_group_id": target_sg_id,
                        "dependency_type": DEFAULT_SG_DEPENDENCY_TYPE,
                    }
                    new_sg_dependency_payloads.append(payload)
                    processed_in_this_run_sg_deps_set.add(sg_dep_tuple)

                    if len(new_sg_dependency_payloads) >= batch_size:
                        try:
                            session.bulk_insert_mappings(
                                StatementGroupDependency, new_sg_dependency_payloads
                            )
                            session.commit()
                            added_count += len(new_sg_dependency_payloads)
                            logger.debug(
                                "Phase 5: Committed batch of %d statement group "
                                "dependencies.",
                                len(new_sg_dependency_payloads),
                            )
                            new_sg_dependency_payloads.clear()
                        except IntegrityError:
                            session.rollback()
                            logger.warning(
                                "Phase 5: IntegrityError on batch insert (likely a "
                                "concurrent duplicate not caught by pre-check). "
                                "Skipping duplicates in this batch and continuing."
                            )
                            # If an integrity error occurs, we might lose the
                            # non-duplicates in this specific batch unless we insert
                            # one by one, or re-filter. For now, clearing and
                            # continuing. A more robust approach might filter out the
                            # specific duplicates and retry.
                            new_sg_dependency_payloads.clear()  # Discard batch on error
                        except SQLAlchemyError as e_batch:
                            session.rollback()
                            logger.error(
                                "Phase 5: Database error during batch insert: %s. "
                                "Skipping batch.",
                                e_batch,
                            )
                            new_sg_dependency_payloads.clear()  # Discard batch

                # Commit any remaining items
                if new_sg_dependency_payloads:
                    try:
                        session.bulk_insert_mappings(
                            StatementGroupDependency, new_sg_dependency_payloads
                        )
                        session.commit()
                        added_count += len(new_sg_dependency_payloads)
                        logger.debug(
                            "Phase 5: Committed final batch of %d statement group "
                            "dependencies.",
                            len(new_sg_dependency_payloads),
                        )
                        new_sg_dependency_payloads.clear()
                    except IntegrityError:
                        session.rollback()
                        logger.warning(
                            "Phase 5: IntegrityError on final batch insert. Some "
                            "items may not have been added."
                        )
                    except SQLAlchemyError as e_final_batch:
                        session.rollback()
                        logger.error(
                            "Phase 5: Database error during final batch insert: %s.",
                            e_final_batch,
                        )

            logger.info(
                "Phase 5: StatementGroup dependency population completed. "
                "Added: %d. Skipped (missing SG link): %d. Skipped (self-loop): %d. "
                "Skipped (existing/duplicate): %d.",
                added_count,
                skipped_due_to_missing_sg_link,
                skipped_due_to_self_loop,
                skipped_as_existing_or_duplicate,
            )
            return True

    except SQLAlchemyError as e_db_critical:
        logger.error(
            "Phase 5: A critical database error occurred: %s",
            e_db_critical,
            exc_info=True,
        )
        return False
    except Exception as e_critical:  # pylint: disable=broad-except
        logger.error(
            "Phase 5: An unexpected critical error occurred: %s",
            e_critical,
            exc_info=True,
        )
        return False
