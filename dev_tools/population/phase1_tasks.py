# File: src/lean_explore/population/phase1_tasks.py

"""Population Phase 1: Initial processing and synchronization of Lean declarations.

Reads declarations from a 'declarations.jsonl' file, processes them,
and populates the 'declarations' table in the database. It handles
both new insertions and updates to existing declarations based on
'lean_name'. If not in 'create_tables_flag' mode, this phase also
identifies and removes declarations from the database that are not present
in the input file, ensuring dependent `StatementGroup` integrity.
Returns a mapping of Lean names (for active declarations) to their database IDs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Type

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.expression import ColumnElement  # Used for type hinting columns
from tqdm import tqdm

from lean_explore.shared.models.db import Declaration, StatementGroup

logger = logging.getLogger(__name__)

# Default chunk size for IN clauses, adjust if needed based on DB limits
DEFAULT_CHUNK_SIZE_FOR_IN_CLAUSE = 500


def module_name_to_rel_path(module_name: str) -> Optional[str]:
    """Converts a Lean module name string to a relative file path string.

    Args:
        module_name: The Lean module name (e.g., 'Mathlib.Data.Nat.Basic').

    Returns:
        Optional[str]: The relative file path (e.g.,
        'Mathlib/Data/Nat/Basic.lean') or None if conversion fails.
    """
    if not module_name:
        return None
    try:
        if module_name.startswith("«.lake».packages."):
            pass
        elif module_name.startswith("«.lake»."):
            module_name = module_name.replace("«.lake».", ".lake/", 1)

        parts = module_name.split(".")
        if not parts:
            return None
        rel_path_obj = Path(*parts)
        rel_path_with_ext = rel_path_obj.with_suffix(".lean")
        return rel_path_with_ext.as_posix()
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error converting module name '%s' to path: %s", module_name, e)
        return None


def _choose_primary_declaration_for_phase1(
    declarations_in_block: List[Dict[str, Any]],
) -> Optional[int]:
    """Chooses the database ID of the primary declaration from a list of dicts.

    Args:
        declarations_in_block: A list of dictionaries, where each dictionary
            represents a declaration and must contain 'id', 'lean_name',
            'decl_type', and 'is_internal' keys.

    Returns:
        Optional[int]: The database ID of the chosen primary `Declaration`,
        or None if no suitable candidate is found.
    """
    if not declarations_in_block:
        return None

    candidates = [decl for decl in declarations_in_block if not decl.get("is_internal")]
    if not candidates:
        candidates = declarations_in_block

    preferred_types_order = [
        "definition",
        "def",
        "theorem",
        "thm",
        "lemma",
        "inductive",
        "structure",
        "class",
        "instance",
        "abbreviation",
        "abbrev",
        "opaque",
        "axiom",
        "constructor",
        "ctor",
        "example",
    ]
    type_priority_map = {dtype: i for i, dtype in enumerate(preferred_types_order)}

    candidates.sort(
        key=lambda d: (
            type_priority_map.get(
                d.get("decl_type", ""), len(preferred_types_order) + 1
            ),
            d.get("lean_name", ""),
        )
    )

    if candidates:
        primary_decl_id = candidates[0].get("id")
        if primary_decl_id is not None:
            return int(primary_decl_id)
    return None


def _execute_select_in_chunks(
    session: Session,
    select_stmt: Any,  # sqlalchemy.sql.selectable.Select
    filter_column: ColumnElement,  # e.g. Declaration.lean_name
    values: Sequence[Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE_FOR_IN_CLAUSE,
) -> List[Any]:
    """Executes a SELECT statement with a large IN clause by chunking values.

    Args:
        session: The SQLAlchemy session.
        select_stmt: The base SELECT statement (without the IN clause applied
                     to filter_column). Any existing WHERE clauses on
                     select_stmt will be preserved and combined with AND.
        filter_column: The SQLAlchemy column object to apply the IN filter on.
        values: A sequence of values for the IN clause.
        chunk_size: The number of values per chunk.

    Returns:
        A list of all results combined from all chunks.
    """
    if not values:
        logger.debug(
            "_execute_select_in_chunks: No values provided for IN clause, "
            "attempting to execute base statement or returning empty if IN is "
            "essential."
        )
        # If the IN clause is essential and values are empty, it means no results.
        # An empty IN () is often invalid SQL or means false.
        # If select_stmt has no other where clauses, this might select all if
        # not handled. To be safe, if values is empty, we add a 'WHERE false'
        # effectively. However, if the original select_stmt is simple (e.g.,
        # just selecting columns), and the IN is the only filter, returning []
        # is correct. Let's refine to add an always-false condition if values
        # is empty. This needs to be done carefully to not break complex
        # existing where clauses. For simplicity here, if it's an IN filter
        # that's empty, usually means "no match for this part".
        # If the overall query relies on this IN to return anything, then empty
        # list is correct.
        return []

    all_results = []
    value_list = list(values)

    for i in range(0, len(value_list), chunk_size):
        chunk = value_list[i : i + chunk_size]
        # Apply the IN clause to a copy or by reconstructing if necessary
        # to avoid modifying the original select_stmt's where clause repeatedly.
        # A simple way: clone the statement or build the final where clause with and_().
        current_stmt = select_stmt.where(filter_column.in_(chunk))
        all_results.extend(session.execute(current_stmt).fetchall())
    return all_results


def _execute_delete_in_chunks(
    session: Session,
    model_class: Type[Any],
    filter_column: ColumnElement,
    values: Sequence[Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE_FOR_IN_CLAUSE,
) -> int:
    """Executes a DELETE statement with a large IN clause by chunking values.

    Args:
        session: The SQLAlchemy session.
        model_class: The ORM model class to delete from.
        filter_column: The SQLAlchemy column object to apply the IN filter on.
        values: A sequence of values for the IN clause.
        chunk_size: The number of values per chunk.

    Returns:
        The total number of rows deleted.
    """
    if not values:
        logger.debug(
            "_execute_delete_in_chunks: No values provided for IN clause, "
            "no rows deleted."
        )
        return 0

    total_deleted_count = 0
    value_list = list(values)

    for i in range(0, len(value_list), chunk_size):
        chunk = value_list[i : i + chunk_size]
        stmt = delete(model_class).where(filter_column.in_(chunk))
        result = session.execute(stmt)
        total_deleted_count += result.rowcount
    return total_deleted_count


def phase1_populate_declarations_initial(
    session_factory: sessionmaker[Session],
    declarations_file: Path,
    batch_size: int,
    create_tables_flag: bool,
) -> Optional[Dict[str, int]]:
    """Populates and synchronizes the 'declarations' table.

    Reads declaration data from the JSONL file, upserts records into the
    'declarations' table. If `create_tables_flag` is False, it then
    identifies and removes declarations from the database that are not present
    in the input file. During this removal, it attempts to maintain the
    integrity of `StatementGroup` entities by updating their primary
    declaration if the original one is removed, or deleting the group if it
    becomes empty or its primary cannot be reassigned.

    Args:
        session_factory: SQLAlchemy session factory for creating database sessions.
        declarations_file: Path to the 'declarations.jsonl' input file.
        batch_size: Number of records to process in a batch before committing.
        create_tables_flag: If True, assumes tables are new and skips fetching
            existing declarations for comparison and skips deletion of stale data.

    Returns:
        Optional[Dict[str, int]]: A dictionary mapping `lean_name` to database ID
        for all active declarations after synchronization. Returns an empty
        dictionary if the input file is empty. Returns None on critical error.
    """
    logger.info(
        "Starting Phase 1: Initial population and synchronization of "
        "declarations from %s...",
        declarations_file,
    )
    if not declarations_file.is_file():
        logger.error("Declarations file not found: %s", declarations_file)
        return None

    try:
        total_lines = 0
        with open(declarations_file, encoding="utf-8") as f_count:
            for _f in f_count:
                total_lines += 1
        if total_lines == 0:
            logger.info(
                "Declarations file %s is empty. Phase 1 skipping population.",
                declarations_file,
            )
            if not create_tables_flag:
                logger.info(
                    "Declarations file is empty, proceeding to delete all "
                    "existing declarations as it's the source of truth."
                )
                try:
                    with session_factory() as session:
                        all_db_decl_results = session.execute(
                            select(Declaration.id)
                        ).fetchall()
                        all_db_decl_ids = {id_val for (id_val,) in all_db_decl_results}
                        if all_db_decl_ids:
                            _delete_stale_declarations_and_manage_sg_integrity(
                                session, set(), all_db_decl_ids
                            )
                            session.commit()
                        else:
                            logger.info(
                                "No declarations found in the database to delete."
                            )
                except Exception as e_delete_all:  # pylint: disable=broad-except
                    logger.error(
                        "Error during deletion of all declarations: %s",
                        e_delete_all,
                        exc_info=True,
                    )
                    return None
            return {}
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Error counting lines in %s: %s", declarations_file, e, exc_info=True
        )
        return None

    active_lean_names_from_input: Set[str] = set()
    db_adds = 0
    db_updates = 0
    log_sample_frequency = 10000

    existing_declarations_map: Dict[str, Declaration] = {}
    if not create_tables_flag:
        try:
            with session_factory() as session:
                logger.info(
                    "Fetching existing declarations from database for Phase 1 "
                    "comparison..."
                )
                results = session.execute(select(Declaration)).scalars().all()
                for decl in results:
                    if decl.lean_name:
                        existing_declarations_map[decl.lean_name] = decl
                logger.info(
                    "Found %d existing declarations for comparison.",
                    len(existing_declarations_map),
                )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Could not fetch existing declarations for Phase 1: %s. "
                "Proceeding as if no existing declarations found for "
                "comparison part.",
                e,
            )

    try:
        with open(
            declarations_file, encoding="utf-8"
        ) as f_decls, session_factory() as session:
            with tqdm(
                total=total_lines, desc="Phase 1: Upserting Declarations", unit="line"
            ) as pbar:
                current_batch_data_to_add: List[Dict[str, Any]] = []
                current_batch_data_to_update: List[Dict[str, Any]] = []

                for i, line in enumerate(f_decls):
                    pbar.update(1)
                    lean_name_for_log = f"UNKNOWN_IN_LINE_{i + 1}"
                    try:
                        data: Dict[str, Any] = json.loads(line)
                        lean_name_for_log = data.get("lean_name", lean_name_for_log)
                        decl_type = data.get("decl_type")
                        log_this_sample = i % log_sample_frequency == 0

                        if log_this_sample:
                            logger.debug(
                                "[Phase 1 Sample Record #%d] Raw JSON: %s",
                                i + 1,
                                line.strip()[:300],
                            )

                        if (
                            not lean_name_for_log
                            or lean_name_for_log.startswith("UNKNOWN_IN_LINE_")
                            or not decl_type
                        ):
                            logger.warning(
                                "Skipping line %d: Missing 'lean_name' or "
                                "'decl_type'. Line: %s",
                                i + 1,
                                line.strip(),
                            )
                            continue

                        active_lean_names_from_input.add(lean_name_for_log)

                        module_name = data.get("module_name")
                        source_file_rel = data.get("source_file")
                        if not source_file_rel and module_name:
                            source_file_rel = module_name_to_rel_path(module_name)

                        attributes_list = data.get("attributes", [])
                        is_projection_val = (
                            "projection" in attributes_list
                            if isinstance(attributes_list, list)
                            else False
                        )

                        auto_generated_suffixes = [
                            ".noConfusion",
                            ".noConfusionType",
                            ".rec",
                            ".recOn",
                            ".casesOn",
                            ".brecOn",
                            ".below",
                            ".IBelow",
                            ".ndrec",
                            ".ndrecOn",
                            ".match_1",
                            ".match_2",
                            ".matcher",
                            ".mk.inj",
                            ".mk.inj_arrow",
                            ".sizeOf_spec",
                            "_uniq",
                            ".internal",
                        ]
                        core_prefixes = ["Lean.", "Init."]
                        is_internal_val = (
                            any(lean_name_for_log.startswith(p) for p in core_prefixes)
                            or any(
                                lean_name_for_log.endswith(s)
                                for s in auto_generated_suffixes
                            )
                            or "._match" in lean_name_for_log
                            or "._proof_" in lean_name_for_log
                            or "._example" in lean_name_for_log
                        )

                        name_parts = lean_name_for_log.split(".")
                        if (
                            len(name_parts) > 1
                            and name_parts[-1].startswith("eq_")
                            and name_parts[-1][3:].isdigit()
                        ):
                            is_internal_val = True
                        if ".Internal." in name_parts:
                            is_internal_val = True

                        decl_data_dict = {
                            "lean_name": lean_name_for_log,
                            "decl_type": decl_type,
                            "module_name": module_name,
                            "source_file": source_file_rel,
                            "docstring": data.get("docstring"),
                            "range_start_line": data.get("range_start_line"),
                            "range_start_col": data.get("range_start_col"),
                            "range_end_line": data.get("range_end_line"),
                            "range_end_col": data.get("range_end_col"),
                            "is_internal": is_internal_val,
                            "is_protected": data.get("is_protected", False),
                            "is_deprecated": data.get("is_deprecated", False),
                            "is_projection": is_projection_val,
                        }

                        if log_this_sample:
                            log_dict_sample = {
                                k: v
                                for k, v in decl_data_dict.items()
                                if k != "docstring"
                            }
                            logger.debug(
                                "[Phase 1 Sample Record #%d] Prepared dict for "
                                "'%s': %s",
                                i + 1,
                                lean_name_for_log,
                                log_dict_sample,
                            )

                        existing_decl = existing_declarations_map.get(lean_name_for_log)
                        if existing_decl:
                            changed = False
                            update_payload = {"id": existing_decl.id}
                            for key, value in decl_data_dict.items():
                                if value != getattr(existing_decl, key, None):
                                    update_payload[key] = value
                                    changed = True
                            if changed:
                                if log_this_sample:
                                    logger.debug(
                                        "Queuing UPDATE for '%s'.", lean_name_for_log
                                    )
                                current_batch_data_to_update.append(update_payload)
                        else:
                            if log_this_sample:
                                logger.debug(
                                    "Queuing NEW insert for '%s'.", lean_name_for_log
                                )
                            current_batch_data_to_add.append(decl_data_dict)

                        if (
                            len(current_batch_data_to_add)
                            + len(current_batch_data_to_update)
                        ) >= batch_size:
                            if current_batch_data_to_add:
                                session.bulk_insert_mappings(
                                    Declaration, current_batch_data_to_add
                                )
                                db_adds += len(current_batch_data_to_add)
                            if current_batch_data_to_update:
                                session.bulk_update_mappings(
                                    Declaration, current_batch_data_to_update
                                )
                                db_updates += len(current_batch_data_to_update)
                            session.commit()
                            pbar.set_postfix_str(
                                f"Adds: {db_adds}, Updates: {db_updates}", refresh=False
                            )
                            current_batch_data_to_add = []
                            current_batch_data_to_update = []
                    except json.JSONDecodeError:
                        logger.error(
                            "Failed to parse JSON at line %d: %s", i + 1, line.strip()
                        )
                    except IntegrityError:
                        session.rollback()
                        logger.warning(
                            "IntegrityError for '%s' at line %d (likely duplicate).",
                            lean_name_for_log,
                            i + 1,
                        )
                    except SQLAlchemyError as e_item:
                        session.rollback()
                        logger.error(
                            "DB error for '%s' at line %d: %s. Skipping.",
                            lean_name_for_log,
                            i + 1,
                            e_item,
                        )
                    except Exception as e_item_unexpected:  # pylint: disable=broad-except
                        session.rollback()
                        logger.error(
                            "Unexpected error for '%s' at line %d: %s. Skipping.",
                            lean_name_for_log,
                            i + 1,
                            e_item_unexpected,
                            exc_info=True,
                        )

                if current_batch_data_to_add:
                    session.bulk_insert_mappings(Declaration, current_batch_data_to_add)
                    db_adds += len(current_batch_data_to_add)
                if current_batch_data_to_update:
                    session.bulk_update_mappings(
                        Declaration, current_batch_data_to_update
                    )
                    db_updates += len(current_batch_data_to_update)
                if current_batch_data_to_add or current_batch_data_to_update:
                    session.commit()
                pbar.set_postfix_str(
                    f"Adds: {db_adds}, Updates: {db_updates}. Upserts done.",
                    refresh=True,
                )

        logger.info(
            "Phase 1: Upsert processing finished. Processed %d lines. DB "
            "additions: %d, DB updates: %d.",
            total_lines,
            db_adds,
            db_updates,
        )
        logger.info(
            "Collected %d unique lean_names from input file for active set.",
            len(active_lean_names_from_input),
        )

    except FileNotFoundError:
        logger.error(
            "Declarations file not found during Phase 1 execution: %s",
            declarations_file,
        )
        return None
    except SQLAlchemyError as e_db_critical_upsert:
        logger.error(
            "A critical database error occurred during Phase 1 (upsert part): %s",
            e_db_critical_upsert,
            exc_info=True,
        )
        return None
    except Exception as e_critical_upsert:  # pylint: disable=broad-except
        logger.error(
            "A critical unexpected error occurred during Phase 1 (upsert part): %s",
            e_critical_upsert,
            exc_info=True,
        )
        return None

    if not create_tables_flag:
        logger.info("Phase 1: Starting stale data deletion process...")
        try:
            with session_factory() as session:
                all_db_declarations_info = session.execute(
                    select(Declaration.id, Declaration.lean_name)
                ).fetchall()

                db_lean_names_to_ids = {
                    name: id_val for id_val, name in all_db_declarations_info if name
                }
                db_lean_names = set(db_lean_names_to_ids.keys())

                stale_lean_names = db_lean_names - active_lean_names_from_input
                stale_declaration_ids = {
                    db_lean_names_to_ids[name]
                    for name in stale_lean_names
                    if name in db_lean_names_to_ids
                }

                if stale_declaration_ids:
                    logger.info(
                        "Found %d stale declarations to process for deletion.",
                        len(stale_declaration_ids),
                    )
                    _delete_stale_declarations_and_manage_sg_integrity(
                        session, active_lean_names_from_input, stale_declaration_ids
                    )
                    session.commit()
                else:
                    logger.info("No stale declarations found to delete.")
            logger.info("Phase 1: Stale data deletion process finished.")
        except Exception as e_delete_critical:  # pylint: disable=broad-except
            logger.error(
                "Critical error during Phase 1 (stale data deletion part): %s. "
                "Database may be inconsistent.",
                e_delete_critical,
                exc_info=True,
            )
            return None

    final_lean_name_to_db_id: Dict[str, int] = {}
    try:
        with session_factory() as session:
            logger.info(
                "Phase 1: Rebuilding lean_name to database ID map from current "
                "DB state..."
            )

            # Base statement for rebuilding the map, targeting only active declarations
            rebuild_stmt_base = select(Declaration.id, Declaration.lean_name)

            if not active_lean_names_from_input:
                if (
                    not create_tables_flag and total_lines == 0
                ):  # Input was empty, and deletions occurred
                    logger.info(
                        "Input was empty and stale data potentially deleted; "
                        "final map will be empty if DB is empty."
                    )
                    # Query all remaining declarations (could be none)
                    results_rebuild = session.execute(rebuild_stmt_base).fetchall()

                elif create_tables_flag:  # create_tables_flag true, input empty
                    logger.info(
                        "Input was empty and create_tables_flag is true; final "
                        "map is empty."
                    )
                    results_rebuild = []
                else:  # Non-empty input file yielded no active_lean_names
                    # Or if initial file processing failed to
                    # populate active_lean_names_from_input
                    logger.warning(
                        "No active lean names collected from input, or input "
                        "processing error. Rebuilding map from all current DB "
                        "entries as a fallback."
                    )
                    results_rebuild = session.execute(rebuild_stmt_base).fetchall()

            else:  # Normal case: active_lean_names_from_input is populated
                results_rebuild = _execute_select_in_chunks(
                    session,
                    rebuild_stmt_base,  # The base SELECT statement
                    Declaration.lean_name,  # The column to filter with IN
                    list(active_lean_names_from_input),  # The values for the IN clause
                )

            final_lean_name_to_db_id = {
                name: id_val for id_val, name in results_rebuild if name
            }

            logger.info(
                "Phase 1: Built final map with %d entries after all operations.",
                len(final_lean_name_to_db_id),
            )
            return final_lean_name_to_db_id
    except Exception as e_rebuild_map:  # pylint: disable=broad-except
        logger.error(
            "Phase 1: Error rebuilding final lean_name_to_db_id map: %s",
            e_rebuild_map,
            exc_info=True,
        )
        return None


def _delete_stale_declarations_and_manage_sg_integrity(
    session: Session,
    active_lean_names_from_input: Set[str],
    stale_declaration_ids: Set[int],
):
    """Manages StatementGroup integrity and deletes stale declarations.

    Args:
        session: The active SQLAlchemy session. All operations are performed
                 within this session. The caller is responsible for committing.
        active_lean_names_from_input: A set of lean_names considered active.
        stale_declaration_ids: A set of database IDs for declarations marked as stale.
    """
    if not stale_declaration_ids:
        logger.debug("_delete_stale: No stale declaration IDs to process.")
        return

    logger.info(
        "Managing StatementGroup integrity for %d stale declarations...",
        len(stale_declaration_ids),
    )

    sg_info_list_tuples = session.execute(
        select(StatementGroup.id, StatementGroup.primary_decl_id)
    ).fetchall()

    base_active_decls_stmt = select(
        Declaration.id,
        Declaration.lean_name,
        Declaration.decl_type,
        Declaration.is_internal,
        Declaration.statement_group_id,
    ).where(Declaration.statement_group_id.isnot(None))

    active_decls_in_groups_raw = _execute_select_in_chunks(
        session,
        base_active_decls_stmt,
        Declaration.lean_name,
        list(active_lean_names_from_input),
    )

    sg_id_to_active_members: Dict[int, List[Dict[str, Any]]] = {}
    for (
        decl_id,
        lean_name,
        decl_type,
        is_internal,
        sg_id_val,
    ) in active_decls_in_groups_raw:
        if sg_id_val is not None:
            if sg_id_val not in sg_id_to_active_members:
                sg_id_to_active_members[sg_id_val] = []
            sg_id_to_active_members[sg_id_val].append(
                {
                    "id": decl_id,
                    "lean_name": lean_name,
                    "decl_type": decl_type,
                    "is_internal": is_internal,
                }
            )

    statement_groups_to_update_payloads: List[Dict[str, Any]] = []
    statement_group_ids_to_delete: Set[int] = set()

    for sg_id, sg_primary_decl_id in sg_info_list_tuples:
        if sg_primary_decl_id in stale_declaration_ids:
            logger.debug(
                "StatementGroup ID %d primary declaration ID %d is stale.",
                sg_id,
                sg_primary_decl_id,
            )
            active_members_for_this_sg = sg_id_to_active_members.get(sg_id, [])
            if not active_members_for_this_sg:
                logger.debug(
                    "StatementGroup ID %d has no remaining active members. "
                    "Marking for deletion.",
                    sg_id,
                )
                statement_group_ids_to_delete.add(sg_id)
            else:
                new_primary_decl_id = _choose_primary_declaration_for_phase1(
                    active_members_for_this_sg
                )
                if new_primary_decl_id is not None:
                    if sg_primary_decl_id != new_primary_decl_id:
                        logger.debug(
                            "StatementGroup ID %d: Assigning new primary "
                            "declaration ID %d.",
                            sg_id,
                            new_primary_decl_id,
                        )
                        statement_groups_to_update_payloads.append(
                            {"id": sg_id, "primary_decl_id": new_primary_decl_id}
                        )
                else:
                    logger.warning(
                        "StatementGroup ID %d: Could not choose a new primary. "
                        "Marking for deletion.",
                        sg_id,
                    )
                    statement_group_ids_to_delete.add(sg_id)

    if statement_groups_to_update_payloads:
        logger.info(
            "Updating primary_decl_id for %d StatementGroups.",
            len(statement_groups_to_update_payloads),
        )
        try:
            session.bulk_update_mappings(
                StatementGroup, statement_groups_to_update_payloads
            )
        except SQLAlchemyError as e_sg_update:
            logger.error(
                "Error updating StatementGroup primary_decl_ids: %s. Changes "
                "for SGs in this batch might be lost.",
                e_sg_update,
                exc_info=True,
            )
            session.rollback()  # Rollback this specific set of SG updates
            # Decide if this is critical enough to halt all further deletions.
            # For now, we'll log and attempt to continue with other deletions.
            # Or re-raise to abort the entire deletion process in
            # phase1_populate_declarations_initial

    logger.info(
        "Preparing to delete %d stale declarations...", len(stale_declaration_ids)
    )
    deleted_decl_count = _execute_delete_in_chunks(
        session, Declaration, Declaration.id, list(stale_declaration_ids)
    )
    logger.info("Actually deleted %d stale declarations.", deleted_decl_count)

    if (
        active_lean_names_from_input
    ):  # Only try to find orphans if there's a basis for "active"
        base_active_linked_sgs_stmt = select(
            Declaration.statement_group_id.distinct()
        ).where(Declaration.statement_group_id.isnot(None))
        active_declarations_still_linked_to_sgs_tuples = _execute_select_in_chunks(
            session,
            base_active_linked_sgs_stmt,
            Declaration.lean_name,
            list(active_lean_names_from_input),
        )
        active_sg_ids_via_links = {
            sg_id
            for (sg_id,) in active_declarations_still_linked_to_sgs_tuples
            if sg_id is not None
        }

        all_current_sg_ids = {sg_id for sg_id, _ in sg_info_list_tuples}

        orphaned_sg_ids = all_current_sg_ids - active_sg_ids_via_links
        orphaned_sg_ids = (
            orphaned_sg_ids - statement_group_ids_to_delete
        )  # Avoid re-adding
        if orphaned_sg_ids:
            logger.info(
                "Identified %d additional orphaned StatementGroups for deletion.",
                len(orphaned_sg_ids),
            )
            statement_group_ids_to_delete.update(orphaned_sg_ids)

    elif (
        not active_lean_names_from_input and sg_info_list_tuples
    ):  # Input was empty (or all names stale)
        logger.info(
            "Input file was empty or all names stale, marking all %d "
            "StatementGroups for deletion.",
            len(sg_info_list_tuples),
        )
        statement_group_ids_to_delete.update(sg_id for sg_id, _ in sg_info_list_tuples)

    if statement_group_ids_to_delete:
        logger.info(
            "Preparing to delete %d stale/orphaned StatementGroups...",
            len(statement_group_ids_to_delete),
        )
        deleted_sg_count = _execute_delete_in_chunks(
            session,
            StatementGroup,
            StatementGroup.id,
            list(statement_group_ids_to_delete),
        )
        logger.info(
            "Actually deleted %d stale/orphaned StatementGroups.", deleted_sg_count
        )
    # The main caller `phase1_populate_declarations_initial` will commit the
    # session after this function.
