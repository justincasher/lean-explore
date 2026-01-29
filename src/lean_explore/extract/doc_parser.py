"""Parser for Lean doc-gen4 output files.

This module parses doc-gen4 JSON data and extracts Lean source code
to produce Declaration objects ready for database insertion.
"""

import json
import logging
import re
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


async def extract_declarations(engine: AsyncEngine, batch_size: int = 1000) -> None:
    """Extract all declarations from doc-gen4 data and load into database.

    Looks for BMP files in each package's .lake/build/doc-data directory.
    Extracts only declarations matching the package's configured module_prefixes,
    ensuring each package's declarations come from its own workspace.

    Args:
        engine: SQLAlchemy async engine for database connection.
        batch_size: Number of declarations to insert per database transaction.
    """
    from lean_explore.extract.package_registry import PACKAGE_REGISTRY
    from lean_explore.extract.package_utils import get_extraction_order

    lean_root = Path("lean")
    all_declarations = []

    # Process each workspace separately with its own package cache
    for package_name in get_extraction_order():
        package_config = PACKAGE_REGISTRY[package_name]
        doc_data_dir = lean_root / package_name / ".lake" / "build" / "doc-data"

        if not doc_data_dir.exists():
            logger.warning(f"No doc-data directory for {package_name}: {doc_data_dir}")
            continue

        bmp_files = sorted(doc_data_dir.glob("**/*.bmp"))
        logger.info(f"Found {len(bmp_files)} BMP files in {package_name}")

        if not bmp_files:
            continue

        # Build workspace-specific package cache to avoid version mismatches
        package_cache = _build_package_cache(lean_root, package_name)
        logger.info(
            f"Built package cache for {package_name} with {len(package_cache)} packages"
        )

        declarations = _parse_declarations_from_files(
            bmp_files, lean_root, package_cache, package_config.module_prefixes
        )
        logger.info(
            f"Extracted {len(declarations)} declarations from {package_name} "
            f"(prefixes: {package_config.module_prefixes})"
        )
        all_declarations.extend(declarations)

    if not all_declarations:
        raise FileNotFoundError("No declarations extracted from any package workspace")

    logger.info(f"Total declarations extracted: {len(all_declarations)}")

    # Filter out auto-generated 'to*' projections that share source with parent
    all_declarations, projection_count = _filter_auto_generated_projections(
        all_declarations
    )
    if projection_count > 0:
        logger.info(f"Filtered {projection_count} auto-generated 'to*' projections")

    async with AsyncSession(engine) as session:
        inserted_count = await _insert_declarations_batch(
            session, all_declarations, batch_size
        )

    skipped = len(all_declarations) - inserted_count
    logger.info(
        f"Inserted {inserted_count} new declarations into database "
        f"(skipped {skipped} duplicates)"
    )
