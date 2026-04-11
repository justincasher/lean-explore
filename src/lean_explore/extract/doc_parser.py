"""Parser for Lean doc-gen4 output files.

This module parses doc-gen4 output and extracts Lean source code to produce
Declaration objects ready for database insertion.

Supports two doc-gen4 output formats:
- SQLite database (api-docs.db): Used by doc-gen4 >= v4.29.0-rc2
- BMP JSON files (.bmp): Used by doc-gen4 < v4.29.0-rc2
"""

import json
import logging
import re
import sqlite3
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.extract.types import Declaration
from lean_explore.models import Declaration as DBDeclaration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RenderedCode BLOB parser
#
# Doc-gen4 stores declaration type signatures in the `name_info.type` column
# as a binary BLOB using leansqlite's ToBinary serialization format.
#
# The type is  RenderedCode = TaggedText RenderedCode.Tag  where:
#   TaggedText:  text(0) String | tag(1) Tag TaggedText | append(2) Array
#   Tag:         keyword(0) | string(1) | const(2) Name | sort-none(3)
#                | sort-type(4) | sort-prop(5) | sort-sort(6) | otherExpr(7)
#   Name:        anonymous(0) | str(1) Name String | num(2) Name Nat
#
# Encoding primitives (big-endian, leansqlite Classes.lean):
#   Nat  – variable-length 7-bit chunks, high bit = continuation
#   String – Nat(utf8_byte_length) + raw UTF-8 bytes
#   Array  – Nat(count) + elements
# ---------------------------------------------------------------------------


class _BlobReader:
    """Minimal reader for leansqlite ToBinary format."""

    __slots__ = ("_data", "_cursor")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._cursor = 0

    def _read_byte(self) -> int:
        if self._cursor >= len(self._data):
            raise ValueError("Unexpected end of BLOB data")
        value = self._data[self._cursor]
        self._cursor += 1
        return value

    def _read_nat(self) -> int:
        """Read a variable-length natural number (7-bit chunks, MSB = more)."""
        result = 0
        shift = 0
        while True:
            byte = self._read_byte()
            if byte >= 128:
                result |= (byte & 0x7F) << shift
            else:
                result |= byte << shift
                break
            shift += 7
        return result

    def _read_string(self) -> str:
        byte_length = self._read_nat()
        if self._cursor + byte_length > len(self._data):
            raise ValueError("String extends past end of BLOB data")
        raw = self._data[self._cursor : self._cursor + byte_length]
        self._cursor += byte_length
        return raw.decode("utf-8")

    def _read_name(self) -> str:
        """Read a Lean Name and return its dot-separated string form."""
        tag = self._read_byte()
        if tag == 0:  # anonymous
            return ""
        if tag == 1:  # str parent s
            parent = self._read_name()
            component = self._read_string()
            return f"{parent}.{component}" if parent else component
        if tag == 2:  # num parent n
            parent = self._read_name()
            number = self._read_nat()
            return f"{parent}.{number}" if parent else str(number)
        raise ValueError(f"Invalid Name tag: {tag}")

    def _skip_name(self) -> None:
        """Skip over a Name without allocating strings."""
        tag = self._read_byte()
        if tag == 0:
            return
        if tag == 1:
            self._skip_name()
            byte_length = self._read_nat()
            self._cursor += byte_length
            return
        if tag == 2:
            self._skip_name()
            self._read_nat()
            return
        raise ValueError(f"Invalid Name tag: {tag}")


def _extract_names_from_rendered_code(blob: bytes) -> list[str]:
    """Extract referenced declaration names from a RenderedCode BLOB.

    Walks the TaggedText tree and collects Lean Names from every
    RenderedCode.Tag.const node (tag byte 2).

    Args:
        blob: Raw bytes of the RenderedCode BLOB from name_info.type.

    Returns:
        De-duplicated list of fully-qualified Lean names referenced in the type.
    """
    reader = _BlobReader(blob)
    names: list[str] = []
    seen: set[str] = set()

    def walk_tagged_text() -> None:
        tag = reader._read_byte()
        if tag == 0:  # text
            reader._read_string()
        elif tag == 1:  # tag
            walk_tag()
            walk_tagged_text()
        elif tag == 2:  # append
            count = reader._read_nat()
            for _ in range(count):
                walk_tagged_text()
        else:
            raise ValueError(f"Invalid TaggedText tag: {tag}")

    def walk_tag() -> None:
        tag = reader._read_byte()
        if tag <= 1 or (3 <= tag <= 7):
            # keyword(0), string(1), sort-none(3), sort-type(4),
            # sort-prop(5), sort-sort(6), otherExpr(7) — no payload
            return
        if tag == 2:  # const
            name = reader._read_name()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
            return
        raise ValueError(f"Invalid RenderedCode.Tag tag: {tag}")

    try:
        walk_tagged_text()
    except (ValueError, IndexError):
        logger.debug("Failed to parse RenderedCode BLOB (%d bytes)", len(blob))
        return []

    return names


def _strip_lean_comments(source_text: str) -> str:
    """Strip Lean comments from source text for comparison.

    Removes:
    - Line comments: -- to end of line
    - Block comments: /- ... -/ (including nested)
    - Doc comments: /-- ... -/ (just a special form of block comments)

    Returns normalized text with collapsed whitespace for reliable comparison.
    """
    result = []
    i = 0
    length = len(source_text)

    while i < length:
        # Check for block comment (includes doc comments /-- ... -/)
        if i < length - 1 and source_text[i : i + 2] == "/-":
            # Skip the opening /-
            i += 2
            nesting_level = 1
            while i < length and nesting_level > 0:
                if i < length - 1 and source_text[i : i + 2] == "/-":
                    nesting_level += 1
                    i += 2
                elif i < length - 1 and source_text[i : i + 2] == "-/":
                    nesting_level -= 1
                    i += 2
                else:
                    i += 1
            continue

        # Check for line comment
        if i < length - 1 and source_text[i : i + 2] == "--":
            # Skip to end of line
            while i < length and source_text[i] != "\n":
                i += 1
            continue

        result.append(source_text[i])
        i += 1

    # Normalize whitespace: collapse multiple spaces/newlines into single space
    text = "".join(result)
    return " ".join(text.split())


def _filter_auto_generated_projections(
    declarations: list[Declaration],
) -> tuple[list[Declaration], int]:
    """Filter out auto-generated 'to*' projections that share source text with parent.

    When a Lean structure extends another, it automatically generates projections
    like `Scheme.toLocallyRingedSpace` that point to the same source location as
    the parent `Scheme` structure. These should be filtered out.

    However, legitimate definitions like `IsOpenImmersion.toScheme` have their
    own unique source text and should be kept.

    Args:
        declarations: List of all extracted declarations.

    Returns:
        Tuple of (filtered declarations, count of removed projections).
    """
    # Build a map of stripped source text -> list of declaration names
    source_to_names: dict[str, list[str]] = {}
    for declaration in declarations:
        stripped = _strip_lean_comments(declaration.source_text)
        if stripped not in source_to_names:
            source_to_names[stripped] = []
        source_to_names[stripped].append(declaration.name)

    filtered = []
    removed_count = 0

    for declaration in declarations:
        short_name = declaration.name.rsplit(".", 1)[-1]

        # Check if this looks like a 'toFoo' projection (to + uppercase letter)
        is_to_projection = (
            len(short_name) > 2
            and short_name.startswith("to")
            and short_name[2].isupper()
        )

        if is_to_projection:
            stripped = _strip_lean_comments(declaration.source_text)
            declarations_with_same_source = source_to_names.get(stripped, [])

            # If other declarations share this source text, this is auto-generated
            if len(declarations_with_same_source) > 1:
                removed_count += 1
                continue

        filtered.append(declaration)

    return filtered, removed_count


def _build_package_cache(
    lean_root: str | Path, workspace_name: str | None = None
) -> dict[str, Path]:
    """Build a cache of package names to their actual directories.

    When workspace_name is provided, only includes packages from that specific
    workspace's .lake/packages directory. This ensures source files are resolved
    from the correct workspace, avoiding version mismatches between workspaces.

    Args:
        lean_root: Root directory containing package workspaces.
        workspace_name: If provided, only include packages from this workspace.
            If None, includes packages from all workspaces (legacy behavior).

    Returns:
        Dictionary mapping lowercase package names to their directory paths.
    """
    from lean_explore.extract.package_utils import get_extraction_order

    lean_root = Path(lean_root)
    cache = {}

    # Determine which workspaces to scan
    workspaces = [workspace_name] if workspace_name else get_extraction_order()

    # Collect packages from workspace(s)
    for ws_name in workspaces:
        packages_directory = lean_root / ws_name / ".lake" / "packages"
        if packages_directory.exists():
            for package_directory in packages_directory.iterdir():
                if package_directory.is_dir():
                    cache[package_directory.name.lower()] = package_directory

    # Add toolchain - use specified workspace or find first available
    if workspace_name:
        toolchain_workspaces = [workspace_name]
    else:
        toolchain_workspaces = get_extraction_order()
    for ws_name in toolchain_workspaces:
        toolchain_file = lean_root / ws_name / "lean-toolchain"
        if toolchain_file.exists():
            version = toolchain_file.read_text().strip().split(":")[-1]
            toolchain_path = (
                Path.home()
                / ".elan"
                / "toolchains"
                / f"leanprover--lean4---{version}"
                / "src"
                / "lean"
            )
            if toolchain_path.exists():
                cache["lean4"] = toolchain_path
                break

    return cache


def _extract_dependencies_from_html(html: str) -> list[str]:
    """Extract dependency names from HTML declaration header."""
    href_pattern = r'href="[^"]*#([^"]+)"'
    matches = re.findall(href_pattern, html)

    dependencies = []
    seen = set()
    for match in matches:
        if match not in seen:
            dependencies.append(match)
            seen.add(match)

    return dependencies


def _read_source_lines(file_path: str | Path, line_start: int, line_end: int) -> str:
    """Read specific lines from a source file.

    If the extracted text is just an attribute (like @[to_additive]), extends
    the range to include the full declaration.
    """
    file_path = Path(file_path)
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
        if line_start > len(lines) or line_end > len(lines):
            raise ValueError(
                f"Line range {line_start}-{line_end} out of bounds for {file_path}"
            )

        result = "".join(lines[line_start - 1 : line_end])

        # If result starts with an attribute, extend to get the full declaration
        stripped = result.strip()
        if stripped.startswith("@["):
            extended_end = line_end
            while extended_end < len(lines):
                extended_end += 1
                extended_result = "".join(lines[line_start - 1 : extended_end])
                if any(
                    kw in extended_result
                    for kw in [
                        " def ",
                        " theorem ",
                        " lemma ",
                        " instance ",
                        " class ",
                        " structure ",
                        " inductive ",
                        " abbrev ",
                        ":=",
                    ]
                ):
                    return extended_result.rstrip()
            return "".join(lines[line_start - 1 : extended_end]).rstrip()

        return result


def _extract_source_text(
    source_link: str, lean_root: str | Path, package_cache: dict[str, Path]
) -> str:
    """Extract source text from a Lean file given a GitHub source link."""
    lean_root = Path(lean_root)
    match = re.search(
        r"github\.com/([^/]+)/([^/]+)/blob/[^/]+/(.+\.lean)#L(\d+)-L(\d+)",
        source_link,
    )
    if not match:
        raise ValueError(f"Could not parse source link: {source_link}")

    (
        organization_name,
        package_name,
        file_path_string,
        line_start_string,
        line_end_string,
    ) = match.groups()
    line_start = int(line_start_string)
    line_end = int(line_end_string)

    candidates = []

    for variant in [
        package_name.lower(),
        package_name.rstrip("0123456789").lower(),
        package_name.replace("-", "").lower(),
    ]:
        if variant in package_cache:
            if variant == "lean4" and file_path_string.startswith("src/lean/"):
                adjusted_path = file_path_string[9:]
                candidates.append(package_cache[variant] / adjusted_path)
                continue
            if variant == "lean4" and file_path_string.startswith("src/lake/"):
                adjusted_path = file_path_string[9:]
                candidates.append(
                    package_cache[variant].parent / "lake" / adjusted_path
                )
                continue
            if variant == "lean4" and file_path_string.startswith("src/"):
                adjusted_path = file_path_string[4:]
            else:
                adjusted_path = file_path_string
            candidates.append(package_cache[variant] / adjusted_path)

    candidates.append(lean_root / file_path_string)

    for candidate in candidates:
        if candidate.exists():
            return _read_source_lines(candidate, line_start, line_end)

    for package_directory in package_cache.values():
        candidate = package_directory / file_path_string
        if candidate.exists():
            return _read_source_lines(candidate, line_start, line_end)

    raise FileNotFoundError(
        f"Could not find {file_path_string} for package {package_name}"
    )


def _read_lean_toolchain_version(workspace_path: Path) -> str | None:
    """Read the Lean version from a workspace's lean-toolchain file.

    Args:
        workspace_path: Path to the package workspace (e.g., lean/mathlib).

    Returns:
        Version string like 'v4.29.0-rc6', or None if not found.
    """
    toolchain_file = workspace_path / "lean-toolchain"
    if not toolchain_file.exists():
        return None
    try:
        content = toolchain_file.read_text().strip()
        match = re.search(r"v\d+\.\d+\.\d+(?:-rc\d+)?", content)
        return match.group() if match else None
    except OSError:
        return None


def _construct_source_link(
    module_name: str,
    module_source_url: str | None,
    start_line: int,
    end_line: int,
    lean_version: str | None = None,
) -> str | None:
    """Construct a GitHub source link from module URL and line range.

    Args:
        module_name: Lean module name from api-docs.db.
        module_source_url: GitHub URL to the module file from api-docs.db.
        start_line: Start line number in the source file.
        end_line: End line number in the source file.
        lean_version: Lean toolchain version (e.g., 'v4.29.0-rc6') used as
            the git ref for core module fallback URLs.

    Returns:
        GitHub URL with line range fragment, or None if no source URL exists.
    """
    if module_source_url:
        return f"{module_source_url}#L{start_line}-L{end_line}"

    git_ref = lean_version or "master"
    module_path = module_name.replace(".", "/")
    root = module_name.split(".", 1)[0]
    if root in {"Init", "Lean", "Std"}:
        return (
            f"https://github.com/leanprover/lean4/blob/{git_ref}/"
            f"src/lean/{module_path}.lean#L{start_line}-L{end_line}"
        )
    if root == "Lake":
        return (
            f"https://github.com/leanprover/lean4/blob/{git_ref}/"
            f"src/lake/{module_path}.lean#L{start_line}-L{end_line}"
        )

    return None


def _parse_declarations_from_sqlite(
    database_path: Path,
    lean_root: Path,
    package_cache: dict[str, Path],
    allowed_module_prefixes: list[str],
    lean_version: str | None = None,
) -> list[Declaration]:
    """Parse declarations from a doc-gen4 SQLite database (api-docs.db).

    Doc-gen4 >= v4.29.0-rc2 outputs declaration data to a SQLite database
    instead of individual BMP JSON files. This function reads that database
    and produces the same Declaration objects as the BMP parser.

    Args:
        database_path: Path to the api-docs.db SQLite database.
        lean_root: Root directory of the Lean project.
        package_cache: Dictionary mapping package names to their directories.
        allowed_module_prefixes: Module prefixes to extract (e.g., ["Mathlib"]).
        lean_version: Lean toolchain version for core module source links.

    Returns:
        List of parsed Declaration objects.
    """
    declarations = []

    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row

    try:
        # Query declarations with source ranges and both docstring types.
        # Doc-gen4 stores docstrings as either markdown text or Verso binary
        # BLOBs (never both). We prefer markdown; Verso BLOBs require a
        # complex deserializer so we detect but cannot extract them yet.
        query = """
            SELECT
                n.module_name,
                n.position,
                n.kind,
                n.name,
                n.type,
                r.start_line,
                r.end_line,
                d.text AS docstring,
                v.content AS verso_docstring,
                m.source_url
            FROM name_info n
            JOIN declaration_ranges r
                ON n.module_name = r.module_name AND n.position = r.position
            LEFT JOIN declaration_markdown_docstrings d
                ON n.module_name = d.module_name AND n.position = d.position
            LEFT JOIN declaration_verso_docstrings v
                ON n.module_name = v.module_name AND n.position = v.position
            JOIN modules m
                ON n.module_name = m.name
            WHERE n.render = 1
            ORDER BY n.module_name, n.position
        """
        rows = connection.execute(query).fetchall()

        logger.info("Found %d declarations in api-docs.db", len(rows))

        skipped_no_source = 0
        skipped_prefix = 0
        skipped_constructor = 0
        source_errors = 0
        verso_only_docstrings = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Parsing api-docs.db...", total=len(rows)
            )

            for row in rows:
                module_name = row["module_name"]
                declaration_name = row["name"]

                # Filter by module prefix
                matches_prefix = any(
                    module_name == prefix or module_name.startswith(prefix + ".")
                    for prefix in allowed_module_prefixes
                )
                if not matches_prefix:
                    skipped_prefix += 1
                    progress.update(task, advance=1)
                    continue

                # Skip auto-generated .mk constructors
                if declaration_name.endswith(".mk"):
                    skipped_constructor += 1
                    progress.update(task, advance=1)
                    continue

                source_url = row["source_url"]
                start_line = row["start_line"]
                end_line = row["end_line"]

                source_link = _construct_source_link(
                    module_name, source_url, start_line, end_line,
                    lean_version=lean_version,
                )
                if not source_link:
                    skipped_no_source += 1
                    progress.update(task, advance=1)
                    continue

                # Extract source text from local files
                try:
                    source_text = _extract_source_text(
                        source_link, lean_root, package_cache
                    )
                except (FileNotFoundError, ValueError) as error:
                    source_errors += 1
                    if source_errors <= 10:
                        logger.debug(
                            "Could not extract source for %s: %s",
                            declaration_name, error,
                        )
                    progress.update(task, advance=1)
                    continue

                # Extract dependency names from the type signature BLOB
                type_blob = row["type"]
                if type_blob:
                    dep_names = _extract_names_from_rendered_code(
                        bytes(type_blob)
                    )
                    # Filter out self-references
                    dep_names = [
                        d for d in dep_names if d != declaration_name
                    ]
                    dependencies = dep_names or None
                else:
                    dependencies = None

                # Use markdown docstring; detect Verso-only cases
                docstring = row["docstring"]
                if not docstring and row["verso_docstring"]:
                    verso_only_docstrings += 1

                declarations.append(
                    Declaration(
                        name=declaration_name,
                        module=module_name,
                        docstring=docstring,
                        source_text=source_text,
                        source_link=source_link,
                        dependencies=dependencies,
                    )
                )

                progress.update(task, advance=1)

        if skipped_prefix > 0:
            logger.info(
                "Skipped %d declarations outside allowed prefixes",
                skipped_prefix,
            )
        if skipped_constructor > 0:
            logger.info("Skipped %d .mk constructors", skipped_constructor)
        if skipped_no_source > 0:
            logger.info("Skipped %d declarations without source URL", skipped_no_source)
        if verso_only_docstrings > 0:
            logger.warning(
                "%d declarations have Verso-only docstrings "
                "(not yet supported, stored as docstring=None)",
                verso_only_docstrings,
            )
        if source_errors > 0:
            logger.warning(
                "Could not extract source text for %d declarations",
                source_errors,
            )

    finally:
        connection.close()

    return declarations


def _parse_declarations_from_files(
    bmp_files: list[Path],
    lean_root: Path,
    package_cache: dict[str, Path],
    allowed_module_prefixes: list[str],
) -> list[Declaration]:
    """Parse declarations from doc-gen4 BMP files.

    Args:
        bmp_files: List of paths to BMP files containing declaration data.
        lean_root: Root directory of the Lean project.
        package_cache: Dictionary mapping package names to their directories.
        allowed_module_prefixes: Module prefixes to extract (e.g., ["Mathlib"]).

    Returns:
        List of parsed Declaration objects.
    """
    declarations = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Parsing BMP files...", total=len(bmp_files))

        for file_path in bmp_files:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            module_name = data["name"]

            # Only extract modules matching the allowed prefixes for this workspace
            # Use prefix + "." to avoid "Lean" matching "LeanSearchClient"
            matches_prefix = any(
                module_name == prefix or module_name.startswith(prefix + ".")
                for prefix in allowed_module_prefixes
            )
            if not matches_prefix:
                progress.update(task, advance=1)
                continue

            for declaration_data in data.get("declarations", []):
                information = declaration_data["info"]
                source_text = _extract_source_text(
                    information["sourceLink"], lean_root, package_cache
                )

                header_html = declaration_data.get("header", "")
                dependencies = _extract_dependencies_from_html(header_html)

                # Filter out self-references from dependencies
                declaration_name = information["name"]
                filtered_dependencies = [
                    d for d in dependencies if d != declaration_name
                ]

                # Skip auto-generated .mk constructors
                if declaration_name.endswith(".mk"):
                    continue

                declarations.append(
                    Declaration(
                        name=declaration_name,
                        module=module_name,
                        docstring=information.get("doc"),
                        source_text=source_text,
                        source_link=information["sourceLink"],
                        dependencies=filtered_dependencies or None,
                    )
                )

            progress.update(task, advance=1)

    return declarations


async def _insert_declarations_batch(
    session: AsyncSession, declarations: list[Declaration], batch_size: int = 1000
) -> int:
    """Insert declarations into database in batches.

    Args:
        session: Active database session.
        declarations: List of declarations to insert.
        batch_size: Number of declarations to insert per batch.

    Returns:
        Number of declarations successfully inserted.
    """
    inserted_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "[green]Inserting declarations into database...",
            total=len(declarations),
        )

        async with session.begin():
            for i in range(0, len(declarations), batch_size):
                batch = declarations[i : i + batch_size]

                for declaration in batch:
                    dependencies_json = (
                        json.dumps(declaration.dependencies)
                        if declaration.dependencies
                        else None
                    )
                    statement = (
                        insert(DBDeclaration)
                        .values(
                            name=declaration.name,
                            module=declaration.module,
                            docstring=declaration.docstring,
                            source_text=declaration.source_text,
                            source_link=declaration.source_link,
                            dependencies=dependencies_json,
                        )
                        .on_conflict_do_nothing(index_elements=["name"])
                    )

                    result = await session.execute(statement)
                    inserted_count += result.rowcount
                    progress.update(task, advance=1)

    return inserted_count


_REQUIRED_DOCGEN_TABLES = {"name_info", "declaration_ranges", "modules"}


def _validate_docgen_sqlite(database_path: Path) -> bool:
    """Check that a doc-gen4 api-docs.db is a valid, usable SQLite database.

    Verifies the file is non-empty, opens as SQLite, and contains the tables
    that the extraction pipeline requires.

    Args:
        database_path: Path to the api-docs.db file.

    Returns:
        True if the database is valid and contains the required tables.
    """
    if database_path.stat().st_size == 0:
        logger.warning("api-docs.db exists but is empty: %s", database_path)
        return False

    try:
        connection = sqlite3.connect(str(database_path))
        try:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
        finally:
            connection.close()
    except sqlite3.DatabaseError as error:
        logger.warning("api-docs.db is not a valid SQLite file: %s", error)
        return False

    missing = _REQUIRED_DOCGEN_TABLES - tables
    if missing:
        logger.warning(
            "api-docs.db is missing required tables %s: %s",
            missing, database_path,
        )
        return False

    return True


def _detect_docgen_format(workspace_path: Path) -> str:
    """Detect which doc-gen4 output format a workspace uses.

    Doc-gen4 >= v4.29.0-rc2 writes to a SQLite database (api-docs.db).
    Earlier versions write individual BMP JSON files to doc-data/.

    The SQLite file is validated before returning "sqlite" to guard against
    zero-byte, corrupt, or incompatible databases left by crashed builds.

    Args:
        workspace_path: Path to the package workspace (e.g., lean/mathlib).

    Returns:
        "sqlite" if a valid api-docs.db exists, "bmp" if BMP files exist,
        "none" otherwise.
    """
    api_docs_db = workspace_path / ".lake" / "build" / "api-docs.db"
    if api_docs_db.exists():
        if _validate_docgen_sqlite(api_docs_db):
            return "sqlite"
        logger.warning(
            "Invalid api-docs.db at %s, checking for BMP fallback",
            api_docs_db,
        )

    doc_data_dir = workspace_path / ".lake" / "build" / "doc-data"
    if doc_data_dir.exists():
        bmp_files = list(doc_data_dir.glob("**/*.bmp"))
        if bmp_files:
            return "bmp"

    return "none"


async def extract_declarations(engine: AsyncEngine, batch_size: int = 1000) -> None:
    """Extract all declarations from doc-gen4 data and load into database.

    Automatically detects whether each package uses the newer SQLite format
    (api-docs.db from doc-gen4 >= v4.29.0-rc2) or the legacy BMP JSON format.

    Args:
        engine: SQLAlchemy async engine for database connection.
        batch_size: Number of declarations to insert per database transaction.
    """
    from lean_explore.extract.package_registry import PACKAGE_REGISTRY
    from lean_explore.extract.package_utils import get_extraction_order

    lean_root = Path("lean")
    all_declarations = []

    for package_name in get_extraction_order():
        package_config = PACKAGE_REGISTRY[package_name]
        workspace_path = lean_root / package_name
        docgen_format = _detect_docgen_format(workspace_path)

        if docgen_format == "none":
            logger.warning("No doc-gen4 output found for %s", package_name)
            continue

        # Build workspace-specific package cache to avoid version mismatches
        package_cache = _build_package_cache(lean_root, package_name)
        logger.info(
            "Built package cache for %s with %d packages",
            package_name, len(package_cache),
        )

        if docgen_format == "sqlite":
            api_docs_path = workspace_path / ".lake" / "build" / "api-docs.db"
            lean_version = _read_lean_toolchain_version(workspace_path)
            logger.info(
                "[%s] Using SQLite format (api-docs.db)", package_name
            )
            declarations = _parse_declarations_from_sqlite(
                api_docs_path,
                lean_root,
                package_cache,
                package_config.module_prefixes,
                lean_version=lean_version,
            )
        else:
            doc_data_dir = workspace_path / ".lake" / "build" / "doc-data"
            bmp_files = sorted(doc_data_dir.glob("**/*.bmp"))
            logger.info(
                "[%s] Using BMP format (%d files)",
                package_name, len(bmp_files),
            )
            declarations = _parse_declarations_from_files(
                bmp_files, lean_root, package_cache,
                package_config.module_prefixes,
            )

        logger.info(
            "Extracted %d declarations from %s (prefixes: %s)",
            len(declarations), package_name, package_config.module_prefixes,
        )
        all_declarations.extend(declarations)

    if not all_declarations:
        raise FileNotFoundError(
            "No declarations extracted from any package workspace"
        )

    logger.info("Total declarations extracted: %d", len(all_declarations))

    # Filter out auto-generated 'to*' projections that share source with parent
    all_declarations, projection_count = _filter_auto_generated_projections(
        all_declarations
    )
    if projection_count > 0:
        logger.info(
            "Filtered %d auto-generated 'to*' projections", projection_count
        )

    async with AsyncSession(engine) as session:
        inserted_count = await _insert_declarations_batch(
            session, all_declarations, batch_size
        )

    skipped = len(all_declarations) - inserted_count
    logger.info(
        "Inserted %d new declarations into database (skipped %d duplicates)",
        inserted_count, skipped,
    )
