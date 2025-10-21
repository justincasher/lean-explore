"""Parser for Lean doc-gen4 output files.

This module parses doc-gen4 JSON data and extracts Lean source code
to produce Declaration objects ready for database insertion.
"""

import json
import logging
import re
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.extract.types import Declaration
from lean_explore.models import Declaration as DBDeclaration

logger = logging.getLogger(__name__)


def _build_package_cache(lean_root: str | Path) -> dict[str, Path]:
    """Build a cache of package names to their actual directories.

    Includes both Lake packages from .lake/packages and the Lean 4 toolchain
    from the elan installation.

    Args:
        lean_root: Root directory of the Lean project.

    Returns:
        Dictionary mapping lowercase package names to their directory paths.

    Example:
        >>> cache = _build_package_cache("lean")
        >>> cache
        {"mathlib4": Path("lean/.lake/packages/mathlib4"),
         "qq": Path("lean/.lake/packages/Qq"),
         "lean4": Path("~/.elan/toolchains/leanprover--lean4---v4.23.0/src/lean")}
    """
    lean_root = Path(lean_root)
    cache = {}

    # Add Lake packages from .lake/packages
    packages_directory = lean_root / ".lake" / "packages"
    if packages_directory.exists():
        for package_directory in packages_directory.iterdir():
            if package_directory.is_dir():
                cache[package_directory.name.lower()] = package_directory

    # Add Lean 4 toolchain package from elan
    toolchain_file = lean_root / "lean-toolchain"
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

    return cache


def _extract_dependencies_from_html(html: str) -> list[str]:
    """Extract dependency names from HTML declaration header."""
    # Find all href links in the HTML
    href_pattern = r'href="[^"]*#([^"]+)"'
    matches = re.findall(href_pattern, html)

    # Filter out self-references and duplicates
    dependencies = []
    seen = set()
    for match in matches:
        if match not in seen:
            dependencies.append(match)
            seen.add(match)

    return dependencies


def _read_source_lines(file_path: str | Path, line_start: int, line_end: int) -> str:
    """Read specific lines from a source file."""
    file_path = Path(file_path)
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
        if line_start <= len(lines) and line_end <= len(lines):
            return "".join(lines[line_start - 1 : line_end])
        raise ValueError(
            f"Line range {line_start}-{line_end} out of bounds for {file_path}"
        )


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

    # Build list of candidate paths to try
    candidates = []

    # 1. Try package name variations in cache
    for variant in [
        package_name.lower(),
        package_name.rstrip("0123456789").lower(),
        package_name.replace("-", "").lower(),
    ]:
        if variant in package_cache:
            # For lean4, strip "src/" prefix if present (already handled in cache path)
            if variant == "lean4" and file_path_string.startswith("src/"):
                adjusted_path = file_path_string[4:]
            else:
                adjusted_path = file_path_string
            candidates.append(package_cache[variant] / adjusted_path)

    # 2. Main source
    candidates.append(lean_root / file_path_string)

    # Try each candidate
    for candidate in candidates:
        if candidate.exists():
            return _read_source_lines(candidate, line_start, line_end)

    # Last resort: search all packages
    for package_directory in package_cache.values():
        candidate = package_directory / file_path_string
        if candidate.exists():
            return _read_source_lines(candidate, line_start, line_end)

    raise FileNotFoundError(
        f"Could not find {file_path_string} for package {package_name}"
    )


async def extract_declarations(engine: AsyncEngine, batch_size: int = 1000) -> None:
    """Extract all declarations from doc-gen4 data and load into database.

    Automatically finds the lean/.lake directory from the root.
    Extracts all declarations, then inserts them into the database.

    Args:
        engine: SQLAlchemy async engine for database connection.
        batch_size: Number of declarations to insert per database transaction.
    """
    lean_root = Path("lean")
    documentation_data_directory = lean_root / ".lake" / "build" / "doc-data"

    if not documentation_data_directory.exists():
        raise FileNotFoundError(
            f"Doc-data directory not found: {documentation_data_directory}"
        )

    bmp_files = sorted(documentation_data_directory.glob("**/*.bmp"))

    # Build package cache once
    package_cache = _build_package_cache(lean_root)
    logger.info(f"Found {len(package_cache)} packages: {list(package_cache.keys())}")

    declarations = []
    for file_path in bmp_files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        module_name = data["name"]

        for declaration_data in data.get("declarations", []):
            information = declaration_data["info"]
            source_text = _extract_source_text(
                information["sourceLink"], lean_root, package_cache
            )

            header_html = declaration_data.get("header", "")
            dependencies = _extract_dependencies_from_html(header_html)

            declarations.append(
                Declaration(
                    name=information["name"],
                    module=module_name,
                    docstring=information.get("doc"),
                    source_text=source_text,
                    source_link=information["sourceLink"],
                    dependencies=dependencies if dependencies else None,
                )
            )
    inserted_count = 0
    async with AsyncSession(engine) as session:
        async with session.begin():
            for i in range(0, len(declarations), batch_size):
                batch = declarations[i : i + batch_size]

                # Use INSERT ON CONFLICT to skip duplicates
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

    skipped = len(declarations) - inserted_count
    logger.info(
        f"Inserted {inserted_count} new declarations into database "
        f"(skipped {skipped} duplicates)"
    )
