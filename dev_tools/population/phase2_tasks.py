# File: src/lean_explore/population/phase2_tasks.py

"""Population Phase 2: Refinement of declaration source and signature.

Uses .ast.json files (with pre-calculated byte spans) and original .lean
source files to refine declaration source locations (line/column spans),
extract the full `statement_text` for the code block a declaration belongs to,
and derive a `declaration_signature` for proof-bearing declaration types.
This phase employs multiprocessing for parallel processing of source files.
"""

import bisect
import functools
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from lean_explore.shared.models.db import Declaration

logger = logging.getLogger(__name__)

# --- Preprocessing Helper Functions for Signature Extraction (Unchanged) ---


def _remove_attributes_from_text(text: str) -> str:
    """Removes attribute annotations (e.g., @[...]) from text.

    This function manually parses the text to identify and remove attribute
    blocks, correctly handling nested brackets `[]` within the attribute
    content. It also removes any whitespace immediately following the
    attribute block.

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
                        # Skip trailing whitespace
                        while scan_index < n and text[scan_index].isspace():
                            scan_index += 1
                        i = scan_index
                        break
                scan_index += 1

            if not found_end:
                logger.debug(
                    "Unclosed attribute found starting at index %d. Treating "
                    "'@' literally.",
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

    Args:
        text: The input string potentially containing comments.

    Returns:
        str: The string with all comments removed.
    """
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
                if i < n and text[i] == "-":
                    i += 1
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
            if i < n and text[i] == "-":
                i += 1
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


def _find_top_level_delimiter_index(text: str) -> Optional[int]:
    """Finds the index of the first top-level ':=' (definition) sequence.

    Args:
        text: The preprocessed text string to search.

    Returns:
        Optional[int]: The starting index of the ':=' sequence if found.
    """
    paren_level, bracket_level, brace_level = 0, 0, 0
    n = len(text)
    for i in range(n - 1):
        char = text[i]
        if char == "(":
            paren_level += 1
        elif char == ")":
            paren_level = max(0, paren_level - 1)
        elif char == "[":
            bracket_level += 1
        elif char == "]":
            bracket_level = max(0, bracket_level - 1)
        elif char == "{":
            brace_level += 1
        elif char == "}":
            brace_level = max(0, brace_level - 1)
        elif char == ":" and text[i + 1] == "=":
            if paren_level == 0 and bracket_level == 0 and brace_level == 0:
                return i
    return None


# --- Configuration for Signature Extraction (Unchanged) ---
PROOF_BEARING_DECL_TYPES: Set[str] = {"theorem", "lemma", "example"}

# --- Helper Functions (Specific to Phase 2 AST Processing) ---


def remove_docstring_from_text(text: str) -> str:
    """Removes a leading Lean docstring (`/-- ... -/`) from text."""
    stripped_text = text.lstrip()
    if stripped_text.startswith("/--"):
        docstring_block_start_idx = text.find("/--")
        end_marker_nl_idx = text.find("-/\n", docstring_block_start_idx + 3)
        end_marker_eof_idx = text.find("-/", docstring_block_start_idx + 3)
        actual_end_idx, advance_chars = -1, 0
        if end_marker_nl_idx != -1 and (
            end_marker_eof_idx == -1 or end_marker_nl_idx < end_marker_eof_idx
        ):
            actual_end_idx, advance_chars = end_marker_nl_idx, 3
        elif end_marker_eof_idx != -1:
            actual_end_idx, advance_chars = end_marker_eof_idx, 2
        if actual_end_idx != -1:
            substring_before_end = text[docstring_block_start_idx + 3 : actual_end_idx]
            if "/-" not in substring_before_end:
                return text[actual_end_idx + advance_chars :]
    return text


# --- SourceFilePositionMapper Class (Unchanged) ---
class SourceFilePositionMapper:
    """Efficiently maps byte offsets to line/column details in a source file.

    This class pre-processes the file content once to build an index of
    line start offsets (both byte and character based). Subsequent calls to
    convert byte spans use this index for faster lookups compared to
    re-scanning the entire file each time.

    Attributes:
        file_content_str: The full string content of the source file.
        total_bytes: Total number of bytes in the UTF-8 encoded file content.
        total_chars: Total number of characters in the file content.
        line_start_byte_offsets: List where element `i` is the byte offset
                                 of the start of line `i+1`.
        line_start_char_offsets: List where element `i` is the character offset
                                 of the start of line `i+1`.
    """

    def __init__(self, file_content_str: str):
        """Initializes the mapper by pre-processing the file content.

        Args:
            file_content_str: The full string content of the source file.
        """
        self.file_content_str = file_content_str
        self.line_start_byte_offsets: List[int] = [0]
        self.line_start_char_offsets: List[int] = [0]

        current_b_offset = 0
        current_c_offset = 0
        for char_val in self.file_content_str:
            try:
                char_len_bytes = len(char_val.encode("utf-8"))
            except UnicodeEncodeError:  # Should be rare for valid strings
                char_len_bytes = 1  # Fallback

            current_b_offset += char_len_bytes
            current_c_offset += 1

            if char_val == "\n":
                self.line_start_byte_offsets.append(current_b_offset)
                self.line_start_char_offsets.append(current_c_offset)

        self.total_bytes = current_b_offset
        self.total_chars = current_c_offset

    def _get_single_pos_details(self, target_byte_offset: int) -> Tuple[int, int, int]:
        """Convert a byte offset to (1-based line, 0-based col, 0-based char_idx).

        Uses precomputed line start offsets for efficiency.

        Args:
            target_byte_offset: The UTF-8 byte offset to convert.

        Returns:
            A tuple (line_number, column_number, char_idx_in_file).
        """
        line_idx = (
            bisect.bisect_right(self.line_start_byte_offsets, target_byte_offset) - 1
        )
        line_idx = max(0, line_idx)

        line_num_1_based = line_idx + 1

        byte_offset_at_line_start = self.line_start_byte_offsets[line_idx]
        char_offset_at_line_start = self.line_start_char_offsets[line_idx]

        current_bytes_on_line = 0
        col_num_0_based = 0
        char_idx_in_file_for_target = char_offset_at_line_start

        line_char_end_exclusive = (
            self.line_start_char_offsets[line_idx + 1]
            if (line_idx + 1) < len(self.line_start_char_offsets)
            else self.total_chars
        )

        for c_idx_in_file in range(char_offset_at_line_start, line_char_end_exclusive):
            char_val = self.file_content_str[c_idx_in_file]
            try:
                char_len_bytes = len(char_val.encode("utf-8"))
            except UnicodeEncodeError:
                char_len_bytes = 1

            if (
                byte_offset_at_line_start + current_bytes_on_line
            ) >= target_byte_offset:
                char_idx_in_file_for_target = c_idx_in_file
                break

            if (
                byte_offset_at_line_start + current_bytes_on_line + char_len_bytes
            ) > target_byte_offset:
                char_idx_in_file_for_target = c_idx_in_file
                break

            current_bytes_on_line += char_len_bytes
            col_num_0_based += 1
            char_idx_in_file_for_target = c_idx_in_file + 1
        else:
            if (
                byte_offset_at_line_start + current_bytes_on_line
            ) == target_byte_offset:
                pass

        if target_byte_offset == byte_offset_at_line_start:
            col_num_0_based = 0
            char_idx_in_file_for_target = char_offset_at_line_start

        return line_num_1_based, col_num_0_based, char_idx_in_file_for_target

    def convert_byte_span_to_details(
        self, target_byte_start: Optional[int], target_byte_end: Optional[int]
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        """Converts a UTF-8 byte span to line/column numbers and character indices.

        Uses an optimized approach with precomputed line offsets.

        Args:
            target_byte_start: The starting byte offset of the span.
            target_byte_end: The ending byte offset of the span. (Often exclusive,
                             pointing to the byte after the last char of the span).

        Returns:
            A tuple (start_line, start_col, end_line, end_col,
            start_char_idx, end_char_idx). Line numbers are 1-based,
            column and char indices are 0-based. Returns Nones if input
            span is invalid.
        """
        if (
            target_byte_start is None
            or target_byte_end is None
            or target_byte_start < 0
            or target_byte_end < target_byte_start
        ):
            logger.debug(
                "Invalid byte span provided: start=%s, end=%s",
                target_byte_start,
                target_byte_end,
            )
            return None, None, None, None, None, None

        clamped_byte_start = max(0, min(target_byte_start, self.total_bytes))
        clamped_byte_end = max(
            clamped_byte_start, min(target_byte_end, self.total_bytes)
        )

        s_line, s_col, s_char_idx = self._get_single_pos_details(clamped_byte_start)

        e_line: Optional[int]
        e_col: Optional[int]
        e_char_idx: Optional[int]

        if clamped_byte_start == clamped_byte_end:
            e_line, e_col, e_char_idx = s_line, s_col, s_char_idx
        else:
            e_line, e_col, e_char_idx = self._get_single_pos_details(clamped_byte_end)

        return s_line, s_col, e_line, e_col, s_char_idx, e_char_idx


# --- Worker Function for Parallel Processing ---


def _process_single_file(
    file_key_data: Tuple[str, List[Tuple[int, str, Optional[int], str]]],
    lean_src_bases: List[Path],
    ast_json_base: Path,
    toolchain_src_path_for_comparison: Optional[Path],
) -> List[Dict[str, Any]]:
    """Processes a single source file to refine its declarations' information."""
    func_total_start_time = time.perf_counter()
    source_file_rel_str, decls_in_file_tuples = file_key_data
    payloads_for_this_file: List[Dict[str, Any]] = []
    file_id_for_log = f"'{source_file_rel_str}'"

    t_start_file_resolution = time.perf_counter()
    lean_file_abs: Optional[Path] = None

    for base_path_attempt in lean_src_bases:
        if not isinstance(base_path_attempt, Path):
            continue
        candidate_path = base_path_attempt / source_file_rel_str
        if candidate_path.is_file():
            lean_file_abs = candidate_path
            break
    logger.debug(
        f"TIMING {file_id_for_log}: Lean file resolution took: "
        f"{time.perf_counter() - t_start_file_resolution:.4f}s"
    )

    if not lean_file_abs:
        logger.warning(
            "Worker: Skipping %s; .lean file not found in any provided source bases: "
            "%s. Searched for relative path: '%s'",
            file_id_for_log,
            [str(b.resolve()) for b in lean_src_bases if isinstance(b, Path)],
            source_file_rel_str,
        )
        logger.debug(
            f"TIMING {file_id_for_log}: Total processing time (skipped .lean): "
            f"{time.perf_counter() - func_total_start_time:.4f}s"
        )
        return []

    t_start_ast_path_construction = time.perf_counter()
    effective_rel_str = source_file_rel_str
    path_obj_effective = Path(effective_rel_str)
    package_identifier_for_ast_dir: str
    if not path_obj_effective.parts:
        logger.warning(
            "Worker: Skipping %s; effective relative path '%s' is empty.",
            file_id_for_log,
            effective_rel_str,
        )
        logger.debug(
            f"TIMING {file_id_for_log}: Total processing time (skipped empty "
            f"effective path): {time.perf_counter() - func_total_start_time:.4f}s"
        )
        return []

    first_segment = path_obj_effective.parts[0]
    if "/" in effective_rel_str:
        package_identifier_for_ast_dir = first_segment
    else:  # Handles files directly in the root of a "library" like Init.lean, Std.lean
        package_identifier_for_ast_dir = first_segment.removesuffix(".lean")

    # Packages whose ASTs are in a subdir named after the package itself, e.g.,
    # AST/Mathlib/Mathlib/...
    doubled_name_pkgs = {
        "Mathlib",
        "Batteries",
        "PhysLean",
    }
    final_ast_path_str_relative_to_ast_base: str

    if package_identifier_for_ast_dir in doubled_name_pkgs:
        final_ast_path_str_relative_to_ast_base = str(
            Path(package_identifier_for_ast_dir) / effective_rel_str
        )
    else:
        final_ast_path_str_relative_to_ast_base = effective_rel_str

    ast_json_file_abs = (
        ast_json_base / final_ast_path_str_relative_to_ast_base
    ).with_suffix(".ast.json")
    logger.debug(
        f"TIMING {file_id_for_log}: AST path construction took: "
        f"{time.perf_counter() - t_start_ast_path_construction:.4f}s"
    )

    t_start_ast_file_check = time.perf_counter()
    ast_file_exists = ast_json_file_abs.is_file()
    logger.debug(
        f"TIMING {file_id_for_log}: AST file existence check took: "
        f"{time.perf_counter() - t_start_ast_file_check:.4f}s"
    )

    if not ast_file_exists:
        logger.warning(
            "Worker: Skipping %s; .ast.json file not found at '%s' (Lean file was "
            "found at: '%s'). Original source_file_rel_str: '%s'. Used final "
            "relative path for AST: '%s'.",
            file_id_for_log,
            ast_json_file_abs,
            lean_file_abs,
            source_file_rel_str,
            final_ast_path_str_relative_to_ast_base,
        )
        logger.debug(
            f"TIMING {file_id_for_log}: Total processing time (skipped .ast.json): "
            f"{time.perf_counter() - func_total_start_time:.4f}s"
        )
        return []

    try:
        t_start_read_lean = time.perf_counter()
        lean_content = lean_file_abs.read_text("utf-8")
        logger.debug(
            f"TIMING {file_id_for_log}: Reading .lean file took: "
            f"{time.perf_counter() - t_start_read_lean:.4f}s"
        )

        t_start_mapper_init = time.perf_counter()
        position_mapper = SourceFilePositionMapper(lean_content)
        logger.debug(
            f"TIMING {file_id_for_log}: SourceFilePositionMapper init took: "
            f"{time.perf_counter() - t_start_mapper_init:.4f}s"
        )

        t_start_read_ast_json = time.perf_counter()
        with open(ast_json_file_abs, encoding="utf-8") as f_ast:
            ast_data = json.load(f_ast)
        logger.debug(
            f"TIMING {file_id_for_log}: Reading and parsing .ast.json took: "
            f"{time.perf_counter() - t_start_read_ast_json:.4f}s"
        )

        command_asts_from_json = ast_data.get("commandASTs", [])

        if not command_asts_from_json:
            logger.debug(
                "Worker: No commandASTs found in '%s' for %s.",
                ast_json_file_abs,
                file_id_for_log,
            )
            logger.debug(
                f"TIMING {file_id_for_log}: Total processing time (no commandASTs): "
                f"{time.perf_counter() - func_total_start_time:.4f}s"
            )
            return []

        t_start_command_processing = time.perf_counter()
        ast_command_details = []
        for cmd_idx, cmd_entry_json in enumerate(command_asts_from_json):
            t_loop_cmd_start = time.perf_counter()

            if not isinstance(cmd_entry_json, dict):
                logger.warning(
                    f"Worker: Expected a dictionary for command AST entry in "
                    f"{file_id_for_log}, cmd_idx {cmd_idx}, "
                    f"got {type(cmd_entry_json)}. "
                    "Skipping."
                )
                continue

            byte_s_raw = cmd_entry_json.get("byteStart")
            byte_e_raw = cmd_entry_json.get("byteEnd")

            byte_s_val = int(byte_s_raw) if byte_s_raw is not None else None
            byte_e_val = int(byte_e_raw) if byte_e_raw is not None else None

            current_byte_span = (byte_s_val, byte_e_val)

            if (
                not current_byte_span
                or current_byte_span[0] is None
                or current_byte_span[1] is None
            ):
                logger.debug(
                    f"Worker: Missing byteStart/byteEnd in commandASTs entry {cmd_idx} "
                    f"for {file_id_for_log}. Skipping."
                )
                continue

            t_pos_conv_start = time.perf_counter()
            conv_res = position_mapper.convert_byte_span_to_details(
                current_byte_span[0], current_byte_span[1]
            )
            t_pos_conv_end = time.perf_counter()
            if cmd_idx % 20 == 0:
                logger.debug(
                    f"TIMING {file_id_for_log} CMD_IDX {cmd_idx}: "
                    f"convert_byte_span_to_details took: "
                    f"{t_pos_conv_end - t_pos_conv_start:.4f}s"
                )

            if not all(val is not None for val in conv_res):  # Check all 6 components
                logger.debug(
                    f"Worker: Byte span conversion failed for a command in "
                    f"{file_id_for_log}, cmd_idx {cmd_idx}. Span: {current_byte_span}, "
                    f"ConvRes: {conv_res}"
                )
                continue
            s_line, s_col, e_line, e_col, char_s_idx, char_e_idx = conv_res  # type: ignore

            command_syntax_data = cmd_entry_json.get("commandSyntax")
            kind = "UnknownKind"  # Default
            if isinstance(command_syntax_data, dict):
                kind = command_syntax_data.get("kind", "UnknownKind")
            else:
                logger.warning(
                    f"Worker: 'commandSyntax' field missing or not a dict in AST entry "
                    f"for {file_id_for_log}, cmd_idx {cmd_idx}. Using default kind."
                )

            ast_command_details.append(
                {
                    "byte_start": current_byte_span[0],
                    "byte_end": current_byte_span[1],
                    "char_start_idx": char_s_idx,
                    "char_end_idx": char_e_idx,
                    "line_start": s_line,
                    "col_start": s_col,
                    "line_end": e_line,
                    "col_end": e_col,
                    "idx": cmd_idx,
                    "kind": kind,
                }
            )
            if cmd_idx % 20 == 0:
                logger.debug(
                    f"TIMING {file_id_for_log} CMD_IDX {cmd_idx}: Rest of command loop "
                    f"iteration took: {time.perf_counter() - t_pos_conv_end:.4f}s"
                )
                logger.debug(
                    f"TIMING {file_id_for_log} CMD_IDX {cmd_idx}: Total for this "
                    f"command iter: {time.perf_counter() - t_loop_cmd_start:.4f}s"
                )

        logger.debug(
            f"TIMING {file_id_for_log}: Processing {len(command_asts_from_json)} "
            f"command_asts took: "
            f"{time.perf_counter() - t_start_command_processing:.4f}s. "
            f"Found {len(ast_command_details)} valid command details."
        )

        t_start_decl_processing = time.perf_counter()
        num_decls_processed_in_file = 0
        for decl_idx, (db_id, lean_name, prelim_line, decl_type) in enumerate(
            decls_in_file_tuples
        ):
            t_loop_decl_start = time.perf_counter()
            if prelim_line is None:
                continue

            t_cand_sel_start = time.perf_counter()
            candidate_cmds = [
                ast_cmd
                for ast_cmd in ast_command_details
                if ast_cmd["line_start"] <= prelim_line <= ast_cmd["line_end"]
            ]
            if not candidate_cmds:
                logger.debug(
                    "Worker: No AST command found for decl '%s' (ID: %s, prelim_line: "
                    "%s) in %s. Available AST cmd lines: %s",
                    lean_name,
                    db_id,
                    prelim_line,
                    file_id_for_log,
                    sorted(
                        list(
                            set(
                                (c["line_start"], c["line_end"])
                                for c in ast_command_details
                            )
                        )
                    ),
                )
                continue

            best_ast_cmd = min(
                candidate_cmds,
                key=lambda c: (
                    c["line_end"] - c["line_start"],
                    c["char_end_idx"] - c["char_start_idx"],
                ),
            )
            t_cand_sel_end = time.perf_counter()
            if decl_idx % 10 == 0:
                logger.debug(
                    f"TIMING {file_id_for_log} DECL_IDX {decl_idx} ({lean_name}): "
                    f"Candidate command selection took: "
                    f"{t_cand_sel_end - t_cand_sel_start:.4f}s"
                )

            char_s = best_ast_cmd["char_start_idx"]
            char_e = best_ast_cmd["char_end_idx"]
            statement_text_for_db: Optional[str] = None

            t_stmt_extract_start = time.perf_counter()
            if not (
                char_s is not None
                and char_e is not None
                and 0 <= char_s <= char_e <= len(lean_content)
            ):
                logger.warning(
                    "Worker: Invalid character slice [%s:%s] (from byte span [%s:%s]) "
                    "for lean_content of length %s in %s for decl %s. "
                    "Skipping statement_text.",
                    char_s,
                    char_e,
                    best_ast_cmd.get("byte_start"),
                    best_ast_cmd.get("byte_end"),
                    len(lean_content),
                    file_id_for_log,
                    lean_name,
                )
                statement_text_for_db = ""
            else:
                contextual_text_raw = lean_content[char_s:char_e]
                statement_text_for_db = remove_docstring_from_text(
                    contextual_text_raw
                ).strip()
            t_stmt_extract_end = time.perf_counter()
            if decl_idx % 10 == 0:
                logger.debug(
                    f"TIMING {file_id_for_log} DECL_IDX {decl_idx} ({lean_name}): "
                    f"Statement text extraction took: "
                    f"{t_stmt_extract_end - t_stmt_extract_start:.4f}s"
                )

            extracted_signature: Optional[str] = None
            t_sig_extract_start = time.perf_counter()
            if decl_type in PROOF_BEARING_DECL_TYPES and statement_text_for_db:
                text_no_attrs = _remove_attributes_from_text(statement_text_for_db)
                text_no_comments = _remove_all_comments_from_text(text_no_attrs)
                delimiter_idx = _find_top_level_delimiter_index(text_no_comments)
                if delimiter_idx is not None:
                    extracted_signature = text_no_comments[:delimiter_idx].strip()
            t_sig_extract_end = time.perf_counter()
            if decl_idx % 10 == 0 and decl_type in PROOF_BEARING_DECL_TYPES:
                logger.debug(
                    f"TIMING {file_id_for_log} DECL_IDX {decl_idx} ({lean_name}): "
                    f"Signature extraction took: "
                    f"{t_sig_extract_end - t_sig_extract_start:.4f}s"
                )

            payloads_for_this_file.append(
                {
                    "id": db_id,
                    "range_start_line": best_ast_cmd["line_start"],
                    "range_start_col": best_ast_cmd["col_start"],
                    "range_end_line": best_ast_cmd["line_end"],
                    "range_end_col": best_ast_cmd["col_end"],
                    "statement_text": statement_text_for_db,
                    "declaration_signature": extracted_signature,
                }
            )
            num_decls_processed_in_file += 1
            if decl_idx % 10 == 0:
                logger.debug(
                    f"TIMING {file_id_for_log} DECL_IDX {decl_idx} ({lean_name}): "
                    f"Payload append and end of loop took: "
                    f"{time.perf_counter() - t_sig_extract_end:.4f}s"
                )
                logger.debug(
                    f"TIMING {file_id_for_log} DECL_IDX {decl_idx} ({lean_name}): "
                    f"Total for this decl iter: "
                    f"{time.perf_counter() - t_loop_decl_start:.4f}s"
                )

        logger.debug(
            f"TIMING {file_id_for_log}: Processing {num_decls_processed_in_file}/"
            f"{len(decls_in_file_tuples)} declarations took: "
            f"{time.perf_counter() - t_start_decl_processing:.4f}s"
        )

    except FileNotFoundError:
        logger.error(
            "Worker: File not found for %s (should have been caught by initial "
            "checks).",
            file_id_for_log,
            exc_info=False,
        )
    except json.JSONDecodeError:
        logger.error(
            "Worker: Failed to parse AST JSON for %s.", file_id_for_log, exc_info=False
        )
    except Exception as e:
        logger.error(
            "Worker error processing file %s: %s", file_id_for_log, e, exc_info=True
        )

    logger.debug(
        f"TIMING {file_id_for_log}: Total processing time for file: "
        f"{time.perf_counter() - func_total_start_time:.4f}s. Generated "
        f"{len(payloads_for_this_file)} payloads."
    )
    return payloads_for_this_file


# --- Phase 2 Main Function (Parallelized) ---


def phase2_refine_declarations_source_info(
    session_factory: sessionmaker[Session],
    lean_src_bases: List[Path],
    ast_json_base: Path,
    batch_size: int,
    toolchain_src_path: Optional[Path],
) -> bool:
    """Refines declaration details using ASTs and source files via parallel processing.

    Args:
        session_factory: SQLAlchemy session factory.
        lean_src_bases: A list of absolute base paths to search for .lean
                        source files, in order of priority.
        ast_json_base: Absolute base path to .ast.json files.
        batch_size: Number of DB updates to commit in one batch.
        toolchain_src_path: Resolved absolute path to the Lean toolchain source
                            directory. (Passed to worker, its usage there might be
                            minimal now).

    Returns:
        bool: True if the phase completed successfully, False on critical error.
    """
    logger.info(
        "Starting Phase 2: Refining declaration source information (Parallel)..."
    )

    if not ast_json_base.is_dir():
        logger.error(
            "AST JSON base path (%s) not found or not a directory. Aborting Phase 2.",
            ast_json_base,
        )
        return False
    if not lean_src_bases:
        logger.warning(
            "Phase 2: No Lean source base paths provided. Refinement of .lean file "
            "content (statement_text, signature) will be skipped for all files. "
            "AST-based range updates might still occur if .ast.json files are "
            "processed."
        )

    db_declarations_for_refinement: List[Tuple[int, str, str, Optional[int], str]] = []
    try:
        with session_factory() as session:
            stmt = select(
                Declaration.id,
                Declaration.lean_name,
                Declaration.source_file,
                Declaration.range_start_line,
                Declaration.decl_type,
            ).where(Declaration.source_file.isnot(None))
            results = session.execute(stmt).fetchall()
            db_declarations_for_refinement = [
                (r[0], r[1], r[2], r[3], r[4])
                for r in results
                if r[2] and r[4]  # Ensure source_file and decl_type are not None
            ]
            logger.info(
                "Found %d declarations with source files for Phase 2 refinement.",
                len(db_declarations_for_refinement),
            )
    except Exception as e:
        logger.error("Failed to query declarations for Phase 2: %s", e, exc_info=True)
        return False

    if not db_declarations_for_refinement:
        logger.info("No declarations require Phase 2 refinement.")
        return True

    declarations_grouped_by_file: Dict[
        str, List[Tuple[int, str, Optional[int], str]]
    ] = {}
    for (
        db_id,
        lean_name,
        src_file,
        prelim_line,
        decl_type,
    ) in db_declarations_for_refinement:
        if src_file:  # Should always be true due to query and list comprehension filter
            declarations_grouped_by_file.setdefault(src_file, []).append(
                (db_id, lean_name, prelim_line, decl_type)
            )

    items_to_process = list(declarations_grouped_by_file.items())
    num_files_to_process = len(items_to_process)
    if num_files_to_process == 0:
        logger.info("No files need processing in Phase 2 after grouping.")
        return True

    worker_func_partial = functools.partial(
        _process_single_file,
        lean_src_bases=lean_src_bases,
        ast_json_base=ast_json_base,
        toolchain_src_path_for_comparison=toolchain_src_path,
    )

    all_update_payloads: List[Dict[str, Any]] = []
    # Determine number of processes: use all available CPUs up to number of files
    num_cpus = os.cpu_count() or 1
    num_processes = min(num_cpus, num_files_to_process)

    logger.info(
        "Starting parallel processing of %d files using %d workers for Phase 2...",
        num_files_to_process,
        num_processes,
    )

    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(
                total=num_files_to_process, desc="Phase 2 Files (Parallel)", unit="file"
            ) as pbar:
                for result_list in pool.imap_unordered(
                    worker_func_partial, items_to_process
                ):
                    if result_list:  # result_list is List[Dict[str, Any]]
                        all_update_payloads.extend(result_list)
                    pbar.update(1)
    except Exception as e:
        logger.error(
            "Critical error during parallel processing pool for Phase 2: %s",
            e,
            exc_info=True,
        )
        return False

    logger.info(
        "Phase 2 parallel processing finished. Collected %d potential declaration "
        "updates.",
        len(all_update_payloads),
    )

    db_updates_committed_count = 0
    if all_update_payloads:
        logger.info(
            "Committing %d updates from Phase 2 to the database in batches of %d...",
            len(all_update_payloads),
            batch_size,
        )
        try:
            with session_factory() as session:
                for i in range(0, len(all_update_payloads), batch_size):
                    batch_data = all_update_payloads[i : i + batch_size]
                    if batch_data:
                        session.bulk_update_mappings(Declaration, batch_data)
                        session.commit()
                        db_updates_committed_count += len(batch_data)
                        if ((i // batch_size) + 1) % 10 == 0 or (i + batch_size) >= len(
                            all_update_payloads
                        ):
                            logger.info(
                                "   Committed Phase 2 batch %d, total committed: %d",
                                (i // batch_size) + 1,
                                db_updates_committed_count,
                            )
                logger.info("Finished committing updates for Phase 2.")
        except Exception as e:
            logger.error(
                "Error committing batch updates for Phase 2: %s", e, exc_info=True
            )
            return False
    else:
        logger.info("No updates to commit from Phase 2 processing.")

    logger.info(
        "Phase 2 Summary: Processed %d files. Generated %d update payloads. "
        "Committed %d updates to the database.",
        num_files_to_process,
        len(all_update_payloads),
        db_updates_committed_count,
    )
    return True
