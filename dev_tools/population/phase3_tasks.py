# File: src/lean_explore/population/phase3_tasks.py

"""Population Phase 3: Statement Grouping and Display Text Generation.

Identifies unique statement blocks from declarations refined in Phase 2.
For each unique block, it creates or updates a `StatementGroup` entry,
determines a primary declaration for the group using a multi-phase heuristic,
cleans and sets the group's `display_statement_text`, and updates the
`docstring`. Finally, it links all declarations belonging to that block to
their respective `StatementGroup`.
"""

import hashlib
import logging
import re  # Added for regex operations
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
                if i < n and text[i] == "-":  # Check boundary for text[i]
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
            if i < n and text[i] == "-":  # Check boundary for text[i]
                i += 1  # For /--
        elif char == "-" and next_char == "-":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            if i < n and text[i] == "\n":
                result_chars.append(
                    text[i]
                )  # Keep the newline itself if line comment ends line
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


def _is_word_in_text(word: str, text: str) -> bool:
    """Checks if a word is in text as a whole word.

    Args:
        word (str): The word to search for.
        text (str): The text to search within.

    Returns:
        bool: True if the word is found as a whole word, False otherwise.
    """
    if not word or not text:
        return False
    escaped_word = re.escape(word)
    pattern = r"\b" + escaped_word + r"\b"
    return bool(re.search(pattern, text))


def _get_declarations_found_in_code_hierarchically(
    declarations: List[Declaration], statement_text: str
) -> List[Declaration]:
    """Performs a hierarchical search for declaration names in statement text.

    It iterates through suffix levels of declaration names (from full FQN
    down to the last component). If matches are found at a certain level,
    only those declarations are returned, and deeper levels are not checked.

    Args:
        declarations (List[Declaration]): The list of declarations to check.
        statement_text (str): The text of the statement group.

    Returns:
        List[Declaration]: A list of declarations found at the earliest
                           (longest name part) suffix level. Empty if none found.
    """
    s_found_in_code: List[Declaration] = []
    if not declarations or not statement_text:
        return s_found_in_code

    max_levels = 0
    for decl in declarations:
        if decl.lean_name:
            max_levels = max(max_levels, len(decl.lean_name.split(".")))

    if max_levels == 0:
        return s_found_in_code

    for level in range(max_levels):  # level 0 to max_levels-1
        current_level_matches: List[Declaration] = []
        for decl in declarations:
            if not decl.lean_name:
                continue

            name_parts = decl.lean_name.split(".")
            if level >= len(name_parts):
                name_part_to_check = None
            else:
                name_part_to_check = ".".join(name_parts[level:])

            if name_part_to_check and _is_word_in_text(
                name_part_to_check, statement_text
            ):
                current_level_matches.append(decl)

        if current_level_matches:
            s_found_in_code = current_level_matches
            break

    return s_found_in_code


def _choose_primary_declaration(
    declarations_in_block: List[Declaration], block_statement_text: str
) -> Optional[Declaration]:
    """Chooses the most representative ("primary") declaration from a list.

    This function applies a multi-phase heuristic. First, it determines an
    initial candidate based on non-internal status, shortest Lean name, and
    declaration type priority. Then, it checks for the presence of declaration
    names (or their parts) in the `block_statement_text` using a hierarchical
    matching approach. If declarations are found in the code, the selection is
    refined based on prefix relationships and name length among those found.
    If no names are found in the code, the initial heuristic choice is used.

    Args:
        declarations_in_block: A list of `Declaration` objects from the same
            source code block.
        block_statement_text: The full source text of the statement block.

    Returns:
        Optional[Declaration]: The chosen primary `Declaration` object, or None
        if no suitable candidate is found.
    """
    if not declarations_in_block:
        return None

    candidate_pool = [decl for decl in declarations_in_block if not decl.is_internal]
    if not candidate_pool:
        candidate_pool = list(declarations_in_block)

    if not candidate_pool:
        logger.warning("  No candidates for primary declaration in block.")
        return None

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

    candidate_pool.sort(
        key=lambda d: (
            len(d.lean_name) if d.lean_name else float("inf"),
            type_priority_map.get(d.decl_type, len(preferred_types_order) + 1),
            d.lean_name if d.lean_name else "",
            d.id if d.id is not None else float("inf"),
        )
    )
    heuristic_choice_decl = candidate_pool[0]

    s_found_in_code = _get_declarations_found_in_code_hierarchically(
        candidate_pool, block_statement_text
    )

    final_primary_decl: Optional[Declaration]

    if not s_found_in_code:
        final_primary_decl = heuristic_choice_decl
    else:
        if len(s_found_in_code) == 1:
            final_primary_decl = s_found_in_code[0]
        else:
            eligible_candidates = list(s_found_in_code)

            prefix_based_candidates: List[Declaration] = []
            is_any_candidate_a_prefix = False
            for d1 in eligible_candidates:
                is_d1_prefix_of_other = False
                for d2 in eligible_candidates:
                    if d1.id == d2.id or not d1.lean_name or not d2.lean_name:
                        continue
                    if len(d1.lean_name) < len(
                        d2.lean_name
                    ) and d2.lean_name.startswith(d1.lean_name):
                        is_d1_prefix_of_other = True
                        is_any_candidate_a_prefix = True
                        break
                if is_d1_prefix_of_other:
                    prefix_based_candidates.append(d1)

            if is_any_candidate_a_prefix:
                eligible_candidates = prefix_based_candidates

            eligible_candidates.sort(
                key=lambda d: (
                    len(d.lean_name) if d.lean_name else float("inf"),
                    d.id if d.id is not None else float("inf"),
                )
            )

            if not eligible_candidates:
                final_primary_decl = heuristic_choice_decl
            else:
                best_after_sort = eligible_candidates[0]

                shortest_len = (
                    len(best_after_sort.lean_name)
                    if best_after_sort.lean_name
                    else float("inf")
                )

                tied_shortest_candidates = [
                    decl
                    for decl in eligible_candidates
                    if (len(decl.lean_name) if decl.lean_name else float("inf"))
                    == shortest_len
                ]

                is_heuristic_choice_chosen = False
                if len(tied_shortest_candidates) > 1 and heuristic_choice_decl:
                    for tied_decl in tied_shortest_candidates:
                        if tied_decl.id == heuristic_choice_decl.id:
                            final_primary_decl = tied_decl
                            is_heuristic_choice_chosen = True
                            break

                if not is_heuristic_choice_chosen:
                    final_primary_decl = best_after_sort

    if final_primary_decl:
        logger.debug(
            "  Chosen primary (new logic): '%s' (ID: %s, Type: %s, Internal: %s) from "
            "%d declarations in block. Heuristic first choice was '%s'. "
            "Found in code: %s.",
            final_primary_decl.lean_name,
            final_primary_decl.id,
            final_primary_decl.decl_type,
            final_primary_decl.is_internal,
            len(declarations_in_block),
            heuristic_choice_decl.lean_name if heuristic_choice_decl else "N/A",
            bool(s_found_in_code),
        )
    else:
        logger.warning(
            "  Could not choose a primary declaration from %d declarations using new"
            " logic.",
            len(declarations_in_block),
        )
    return final_primary_decl


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

    content_hash_to_id_map: Dict[str, int] = {}
    declarations_to_process: List[Declaration] = []
    try:
        with session_factory() as session:
            logger.info(
                "Phase 3: Building in-memory hash map of existing statement groups..."
            )
            existing_groups = session.execute(
                select(StatementGroup.id, StatementGroup.statement_text)
            ).fetchall()
            content_hash_to_id_map = {
                _calculate_text_hash(text): group_id
                for group_id, text in existing_groups
            }
            logger.info(
                "Phase 3: Mapped %d existing groups.", len(content_hash_to_id_map)
            )

            logger.info("Phase 3: Fetching declarations for grouping...")
            stmt = (
                select(Declaration)
                .where(Declaration.statement_group_id.is_(None))
                .where(Declaration.statement_text.isnot(None))
                .where(Declaration.source_file.isnot(None))
                .where(Declaration.range_start_line.isnot(None))
                .where(Declaration.range_start_col.isnot(None))
                .where(Declaration.range_end_line.isnot(None))
                .where(Declaration.range_end_col.isnot(None))
                .order_by(
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
            "Phase 3: DB error during initial data fetch: %s", e, exc_info=True
        )
        return False
    except Exception as e:
        logger.error(
            "Phase 3: Unexpected error during initial data fetch: %s", e, exc_info=True
        )
        return False

    blocks_map: Dict[Tuple[str, int, int, int, int, str], List[Declaration]] = {}
    for decl in declarations_to_process:
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
            decl.statement_text,
        )
        blocks_map.setdefault(block_key, []).append(decl)

    logger.info("Phase 3: Identified %d unique statement blocks.", len(blocks_map))

    created_group_count = 0
    updated_group_count = 0
    links_to_commit_count = 0
    declaration_update_batch: List[Dict[str, Any]] = []

    with tqdm(total=len(blocks_map), desc="Phase 3: Grouping", unit="block") as pbar:
        with session_factory() as session:
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
                    primary_decl = _choose_primary_declaration(
                        decls_in_block, blk_stmt_text
                    )
                    if not primary_decl or primary_decl.id is None:
                        logger.warning(
                            "  Could not determine valid primary declaration for block"
                            " hash %s (text: '%s...'). Skipping.",
                            block_hash[:8],
                            blk_stmt_text[:50].replace("\n", "\\n"),
                        )
                        continue

                    base_display_text = (
                        primary_decl.declaration_signature
                        if primary_decl.declaration_signature
                        and primary_decl.declaration_signature.strip()
                        else blk_stmt_text
                    )
                    cleaned_display_text = _remove_attributes_from_text(
                        base_display_text
                    )
                    cleaned_display_text = _remove_all_comments_from_text(
                        cleaned_display_text
                    )

                    existing_group_id = content_hash_to_id_map.get(block_hash)
                    sg_id_for_linking: Optional[int] = None

                    if existing_group_id:
                        sg_entry = session.get(StatementGroup, existing_group_id)
                        if sg_entry:
                            sg_id_for_linking = sg_entry.id
                            sg_entry.source_file = block_src_file
                            sg_entry.range_start_line = blk_start_line
                            sg_entry.range_start_col = blk_start_col
                            sg_entry.range_end_line = blk_end_line
                            sg_entry.range_end_col = blk_end_col
                            sg_entry.primary_decl_id = primary_decl.id
                            sg_entry.docstring = primary_decl.docstring
                            sg_entry.display_statement_text = cleaned_display_text
                            needs_update = True
                        if (
                            sg_entry.source_file != block_src_file
                            or sg_entry.range_start_line != blk_start_line
                            or sg_entry.range_start_col != blk_start_col
                            or sg_entry.range_end_line != blk_end_line
                            or sg_entry.range_end_col != blk_end_col
                        ):
                            sg_entry.source_file = block_src_file
                            sg_entry.range_start_line = blk_start_line
                            sg_entry.range_start_col = blk_start_col
                            sg_entry.range_end_line = blk_end_line
                            sg_entry.range_end_col = blk_end_col
                            needs_update = True
                        if needs_update:
                            session.commit()
                            updated_group_count += 1
                        else:
                            logger.warning(
                                "Found hash for SG ID %d but could not fetch the "
                                "record. A new group will be created.",
                                existing_group_id,
                            )
                            existing_group_id = None

                    if not existing_group_id:
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
                        new_sg.declarations = decls_in_block
                        session.add(new_sg)
                        session.flush()
                        sg_id_for_linking = new_sg.id
                        content_hash_to_id_map[block_hash] = new_sg.id
                        created_group_count += 1

                    if sg_id_for_linking is not None:
                        for decl_to_link in decls_in_block:
                            if decl_to_link.id is not None:
                                declaration_update_batch.append(
                                    {
                                        "id": decl_to_link.id,
                                        "statement_group_id": sg_id_for_linking,
                                    }
                                )

                except IntegrityError as e:
                    logger.warning(
                        "Integrity error for block hash %s: %s. Rolling back.",
                        block_hash[:8],
                        e,
                    )
                    session.rollback()
                except SQLAlchemyError as e_block_db:
                    logger.error(
                        "DB error processing block (hash %s): %s. Rolling back.",
                        block_hash[:8],
                        e_block_db,
                    )
                    session.rollback()
                except Exception as e_block_unexpected:
                    logger.error(
                        "Unexpected error processing block (hash %s): %s",
                        block_hash[:8],
                        e_block_unexpected,
                        exc_info=True,
                    )
                    session.rollback()
            session.commit()

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
                links_to_commit_count = committed_links_total
            logger.info("Finished committing declaration links.")
        except SQLAlchemyError as e_final_commit:
            logger.error(
                "Phase 3: DB error on final declaration link commit: %s",
                e_final_commit,
                exc_info=True,
            )
            return False

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
