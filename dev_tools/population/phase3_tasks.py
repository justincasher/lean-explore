# File: src/lean_explore/population/phase3_tasks.py

"""Population Phase 3: Statement Grouping and Display Text Generation.

Identifies unique statement blocks from declarations refined in Phase 2.
For each unique block, it creates or updates a `StatementGroup` entry,
determines a primary declaration for the group, cleans and sets the group's
`display_statement_text` (based on the primary declaration's signature or
the full block text), and updates the `docstring`. Finally, it links all
declarations belonging to that block to their respective `StatementGroup`.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from lean_explore.shared.models.db import Declaration, StatementGroup

logger = logging.getLogger(__name__)

# --- Helper Functions ---


def _remove_attributes_from_text(text: str) -> str:
    """Removes attribute annotations (e.g., @[...]) from text.

    Manually parses text to remove attribute blocks, handling nested brackets
    `[]` within attribute content. Also removes whitespace immediately
    following an attribute block.

    Args:
        text: The input string potentially containing attribute annotations.

    Returns:
        str: The string with attribute blocks removed.
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "@" and i + 1 < n and text[i + 1] == "[":
            attribute_start_index = i
            scan_index = i + 2  # Start after '@['
            bracket_level = 1
            found_end = False
            while scan_index < n:
                char = text[scan_index]
                if char == "[":
                    bracket_level += 1
                elif char == "]":
                    bracket_level -= 1
                    if bracket_level == 0:
                        found_end = True
                        scan_index += 1  # Move past ']'
                        while (
                            scan_index < n and text[scan_index].isspace()
                        ):  # Skip trailing whitespace
                            scan_index += 1
                        i = scan_index
                        break
                scan_index += 1
            if not found_end:
                logger.debug(
                    "Unclosed attribute at index %d. Treating '@' literally.",
                    attribute_start_index,
                )
                result.append(text[attribute_start_index])
                i = attribute_start_index + 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def _remove_all_comments_from_text(text: str) -> str:
    """Removes all types of comments from Lean code.

    Manually parses text to remove nested block comments (`/- ... -/`),
    line comments (`-- ...`), ignoring comment markers within string literals.
    Resulting empty lines are removed.

    Args:
        text: The input string potentially containing comments.

    Returns:
        str: The string with comments removed, stripped of leading/trailing
        whitespace and empty lines.
    """
    # This is a complex manual parser. Original logic preserved.
    result_chars = []
    i = 0
    n = len(text)
    in_string_literal = False
    is_escaped = False
    block_comment_nest_level = 0

    while i < n:
        char = text[i]
        next_char = text[i + 1] if i + 1 < n else None

        if in_string_literal:
            result_chars.append(char)
            if is_escaped:
                is_escaped = False
            elif char == "\\":
                is_escaped = True
            elif char == '"':
                in_string_literal = False
            i += 1
            continue

        if block_comment_nest_level > 0:
            if char == "-" and next_char == "/":
                block_comment_nest_level -= 1
                i += 2
            elif char == "/" and next_char == "-":
                block_comment_nest_level += 1
                i += 2
                if text[i] == "-":
                    i += 1  # For /--
            else:
                i += 1
            continue

        if char == '"':
            in_string_literal = True
            is_escaped = False
            result_chars.append(char)
            i += 1
        elif char == "/" and next_char == "-":
            block_comment_nest_level = 1
            i += 2
            if text[i] == "-":
                i += 1  # For /--
        elif char == "-" and next_char == "-":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            if i < n and text[i] == "\n":
                result_chars.append(text[i])
                i += 1
        else:
            result_chars.append(char)
            i += 1

    processed_text = "".join(result_chars)
    lines = processed_text.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines).strip()


def _calculate_text_hash(text: str) -> str:
    r"""Calculates SHA256 hash for a given text string.

    Normalizes line endings to LF ('\n') before hashing to ensure
    consistency across different operating systems.

    Args:
        text: The input string to hash.

    Returns:
        str: A hexadecimal string digest of the SHA256 hash.
    """
    normalized_text = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _choose_primary_declaration(
    declarations_in_block: List[Declaration],
) -> Optional[Declaration]:
    """Chooses the most representative ("primary") declaration from a list.

    Applies heuristics to select the primary declaration from a list of
    declarations originating from the same statement block. Prioritizes
    non-internal declarations, then specific declaration types (e.g.,
    'definition' over 'theorem'), and finally uses Lean name for tie-breaking.

    Args:
        declarations_in_block: A list of `Declaration` objects from the same
            source code block.

    Returns:
        Optional[Declaration]: The chosen primary `Declaration` object, or None
        if the input list is empty or no suitable candidate is found.
    """
    if not declarations_in_block:
        return None

    # Prefer non-internal declarations
    candidates = [decl for decl in declarations_in_block if not decl.is_internal]
    if not candidates:  # If all are internal, consider all of them
        candidates = declarations_in_block

    # Prioritize by declaration type, then by Lean name for tie-breaking
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

    # Sort candidates: lower priority number first (more preferred type), then by name
    candidates.sort(
        key=lambda d: (
            type_priority_map.get(d.decl_type, len(preferred_types_order) + 1),
            d.lean_name,  # Alphabetical by name as a tie-breaker
        )
    )

    if candidates:
        primary_decl = candidates[0]
        logger.debug(
            "  Chosen primary: '%s' (Type: %s, Internal: %s) from %d declarations"
            " in block.",
            primary_decl.lean_name,
            primary_decl.decl_type,
            primary_decl.is_internal,
            len(declarations_in_block),
        )
        return primary_decl

    logger.warning(
        "  Could not choose a primary declaration from %d declarations using current"
        " heuristic.",
        len(declarations_in_block),
    )
    return None


# --- Phase 3 Main Function ---


def phase3_group_statements(
    session_factory: sessionmaker[Session], batch_size: int
) -> bool:
    """Groups declarations into StatementGroups based on source block information.

    Processes `Declaration` entries (refined by Phase 2), identifies unique
    statement blocks by hashing their `statement_text`, chooses a primary
    declaration for each block, and creates or updates `StatementGroup`
    entries in the database. It also generates a cleaned `display_statement_text`
    for each group and links all declarations from a block to their
    respective `StatementGroup`.

    Args:
        session_factory: SQLAlchemy session factory for creating database sessions.
        batch_size: Number of declaration-to-group link updates to batch before
            committing to the database.

    Returns:
        bool: True if the grouping process completed successfully, False otherwise.
    """
    logger.info("Starting Phase 3: Statement Grouping and Display Text Population...")

    declarations_to_process: List[Declaration] = []
    try:
        with session_factory() as session:
            logger.info("Phase 3: Fetching declarations for grouping...")
            # Select declarations that haven't been grouped and have
            # necessary source info
            stmt = (
                select(Declaration)
                .where(Declaration.statement_group_id.is_(None))
                .where(Declaration.statement_text.isnot(None))
                .where(Declaration.source_file.isnot(None))
                .where(Declaration.range_start_line.isnot(None))
                .where(Declaration.range_start_col.isnot(None))
                .where(Declaration.range_end_line.isnot(None))
                .where(Declaration.range_end_col.isnot(None))
                .order_by(  # Order for deterministic processing if it matters
                    Declaration.source_file,
                    Declaration.range_start_line,
                    Declaration.range_start_col,
                )
            )
            declarations_to_process = session.execute(stmt).scalars().all()
        logger.info(
            "Phase 3: Found %d declarations to process for grouping.",
            len(declarations_to_process),
        )
        if not declarations_to_process:
            logger.info("Phase 3: No declarations require grouping. Phase complete.")
            return True
    except SQLAlchemyError as e:
        logger.error(
            "Phase 3: DB error fetching declarations for grouping: %s", e, exc_info=True
        )
        return False
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Phase 3: Unexpected error fetching declarations: %s", e, exc_info=True
        )
        return False

    # Group declarations by their canonical block identifier (location + text)
    blocks_map: Dict[Tuple[str, int, int, int, int, str], List[Declaration]] = {}
    for decl in declarations_to_process:
        # Defensive check, though query should ensure these are not None
        if not all(
            [
                decl.source_file,
                decl.range_start_line is not None,
                decl.range_start_col is not None,
                decl.range_end_line is not None,
                decl.range_end_col is not None,
                decl.statement_text is not None,
            ]
        ):
            logger.warning(
                "Declaration ID %d ('%s') missing required block info. Skipping.",
                decl.id,
                decl.lean_name,
            )
            continue
        block_key = (
            decl.source_file,
            decl.range_start_line,
            decl.range_start_col,
            decl.range_end_line,
            decl.range_end_col,
            decl.statement_text,  # statement_text is crucial for defining the block
        )
        blocks_map.setdefault(block_key, []).append(decl)

    logger.info("Phase 3: Identified %d unique statement blocks.", len(blocks_map))

    created_group_count = 0
    updated_group_count = 0
    links_to_commit_count = 0
    declaration_update_batch: List[Dict[str, Any]] = []

    with tqdm(total=len(blocks_map), desc="Phase 3: Grouping", unit="block") as pbar:
        for block_key_tuple, decls_in_block in blocks_map.items():
            pbar.update(1)
            (
                block_src_file,
                blk_start_line,
                blk_start_col,
                blk_end_line,
                blk_end_col,
                blk_stmt_text,
            ) = block_key_tuple
            block_hash = _calculate_text_hash(blk_stmt_text)

            try:
                primary_decl = _choose_primary_declaration(decls_in_block)
                if not primary_decl or primary_decl.id is None:
                    logger.warning(
                        "  Could not determine valid primary declaration for block hash"
                        " %s. Skipping.",
                        block_hash[:8],
                    )
                    continue

                # Determine display text: use signature if available, else
                # full block text
                # Then clean attributes and comments for display.
                base_display_text = (
                    primary_decl.declaration_signature
                    if primary_decl.declaration_signature
                    and primary_decl.declaration_signature.strip()
                    else blk_stmt_text
                )
                cleaned_display_text = _remove_attributes_from_text(base_display_text)
                cleaned_display_text = _remove_all_comments_from_text(
                    cleaned_display_text
                )

                with session_factory() as session:
                    sg_entry = session.execute(
                        select(StatementGroup).filter_by(text_hash=block_hash)
                    ).scalar_one_or_none()

                    sg_id_for_linking: Optional[int] = None
                    if sg_entry:  # Update existing StatementGroup
                        sg_id_for_linking = sg_entry.id
                        needs_update = False
                        if sg_entry.primary_decl_id != primary_decl.id:
                            sg_entry.primary_decl_id = primary_decl.id
                            needs_update = True
                        if (
                            sg_entry.docstring != primary_decl.docstring
                        ):  # Handles None comparison
                            sg_entry.docstring = primary_decl.docstring
                            needs_update = True
                        if sg_entry.display_statement_text != cleaned_display_text:
                            sg_entry.display_statement_text = cleaned_display_text
                            needs_update = True
                        if needs_update:
                            session.commit()
                            updated_group_count += 1
                            logger.debug("  Updated StatementGroup ID %d.", sg_entry.id)
                    else:  # Create new StatementGroup
                        new_sg = StatementGroup(
                            text_hash=block_hash,
                            statement_text=blk_stmt_text,
                            display_statement_text=cleaned_display_text,
                            source_file=block_src_file,
                            range_start_line=blk_start_line,
                            range_start_col=blk_start_col,
                            range_end_line=blk_end_line,
                            range_end_col=blk_end_col,
                            primary_decl_id=primary_decl.id,
                            docstring=primary_decl.docstring,
                        )
                        session.add(new_sg)
                        try:
                            session.commit()  # Commit to get ID
                            sg_id_for_linking = new_sg.id
                            created_group_count += 1
                        except (
                            IntegrityError
                        ):  # Handle rare race if hash collision by other means
                            session.rollback()
                            logger.warning(
                                "IntegrityError on new SG commit (hash %s). "
                                "Refetching.",
                                block_hash[:8],
                            )
                            sg_refetched = session.execute(
                                select(StatementGroup).filter_by(text_hash=block_hash)
                            ).scalar_one_or_none()
                            if sg_refetched:
                                sg_id_for_linking = sg_refetched.id
                            else:
                                logger.error(
                                    "Refetch failed for hash %s. Skipping linking.",
                                    block_hash[:8],
                                )

                    # Link declarations to this StatementGroup
                    if sg_id_for_linking is not None:
                        for decl_to_link in decls_in_block:
                            if (
                                decl_to_link.id is not None
                                and decl_to_link.statement_group_id != sg_id_for_linking
                            ):
                                declaration_update_batch.append(
                                    {
                                        "id": decl_to_link.id,
                                        "statement_group_id": sg_id_for_linking,
                                    }
                                )
            except SQLAlchemyError as e_block_db:
                logger.error(
                    "DB error processing block (hash %s): %s. Rolling back session for"
                    " this block.",
                    block_hash[:8] if "block_hash" in locals() else "N/A",
                    e_block_db,
                )
                if "session" in locals() and session.is_active:
                    session.rollback()
            except Exception as e_block_unexpected:  # pylint: disable=broad-except
                logger.error(
                    "Unexpected error processing block (hash %s): %s",
                    block_hash[:8] if "block_hash" in locals() else "N/A",
                    e_block_unexpected,
                    exc_info=True,
                )

    # Final commit for all declaration-to-group links
    if declaration_update_batch:
        logger.info(
            "Phase 3: Committing %d declaration-to-group links in batches of %d...",
            len(declaration_update_batch),
            batch_size,
        )
        try:
            with session_factory() as session:
                committed_links_total = 0
                for i in range(0, len(declaration_update_batch), batch_size):
                    batch_payload = declaration_update_batch[i : i + batch_size]
                    if batch_payload:
                        session.bulk_update_mappings(Declaration, batch_payload)
                        session.commit()
                        committed_links_total += len(batch_payload)
                        if ((i // batch_size) + 1) % 10 == 0 or (i + batch_size) >= len(
                            declaration_update_batch
                        ):
                            logger.info(
                                "  Committed link batch %d, total links committed: %d",
                                (i // batch_size) + 1,
                                committed_links_total,
                            )
                links_to_commit_count = (
                    committed_links_total  # Reflect actual committed
                )
            logger.info("Finished committing declaration links.")
        except SQLAlchemyError as e_final_commit:
            logger.error(
                "Phase 3: DB error on final declaration link commit: %s",
                e_final_commit,
                exc_info=True,
            )
            return False  # Indicate partial failure

    logger.info(
        "Phase 3: Statement Grouping completed. "
        "Processed %d unique blocks. "
        "Created %d new StatementGroups. Updated %d existing StatementGroups. "
        "Established %d declaration-to-group links.",
        len(blocks_map),
        created_group_count,
        updated_group_count,
        links_to_commit_count,
    )
    return True
