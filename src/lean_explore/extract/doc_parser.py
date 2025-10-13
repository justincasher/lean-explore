"""Parser for Lean doc-gen4 output files.

This module parses doc-gen4 JSON data and extracts Lean source code
to produce Declaration objects ready for database insertion.
"""

import json
import logging
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.extract.schemas import Declaration as DBDeclaration
from lean_explore.extract.types import Declaration

logger = logging.getLogger(__name__)


def _build_package_cache(lean_root: Path) -> dict[str, Path]:
    """Build a cache of package names to their actual directories."""
    cache = {}
    packages_dir = lean_root / ".lake" / "packages"
    if packages_dir.exists():
        for pkg_dir in packages_dir.iterdir():
            if pkg_dir.is_dir():
                cache[pkg_dir.name.lower()] = pkg_dir
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


def _read_source_lines(file_path: Path, line_start: int, line_end: int) -> str:
    """Read specific lines from a source file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if line_start <= len(lines) and line_end <= len(lines):
            return "".join(lines[line_start - 1 : line_end])
        raise ValueError(
            f"Line range {line_start}-{line_end} out of bounds for {file_path}"
        )


def _extract_source_text(
    source_link: str, lean_root: Path, package_cache: dict[str, Path]
) -> str:
    """Extract source text from a Lean file given a GitHub source link."""
    match = re.search(
        r"github\.com/([^/]+)/([^/]+)/blob/[^/]+/(.+\.lean)#L(\d+)-L(\d+)",
        source_link,
    )
    if not match:
        raise ValueError(f"Could not parse source link: {source_link}")

    org_name, package_name, file_path_str, line_start_str, line_end_str = match.groups()
    line_start = int(line_start_str)
    line_end = int(line_end_str)

    # Build list of candidate paths to try
    candidates = []

    # 1. Lean 4 toolchain (for leanprover/lean4)
    if org_name == "leanprover" and package_name == "lean4":
        toolchain_file = lean_root / "lean-toolchain"
        if toolchain_file.exists():
            version = toolchain_file.read_text().strip().split(":")[-1]
            lean4_path = file_path_str[4:] if file_path_str.startswith("src/") else file_path_str
            candidates.append(
                Path.home() / ".elan" / "toolchains" / f"leanprover--lean4---{version}" / "src" / "lean" / lean4_path
            )

    # 2. Package variations
    for variant in [
        package_name.lower(),
        package_name.rstrip("0123456789").lower(),
        package_name.replace("-", "").lower(),
    ]:
        if variant in package_cache:
            candidates.append(package_cache[variant] / file_path_str)

    # 3. Main source
    candidates.append(lean_root / file_path_str)

    # Try each candidate
    for candidate in candidates:
        if candidate.exists():
            return _read_source_lines(candidate, line_start, line_end)

    # Last resort: search all packages
    for pkg_dir in package_cache.values():
        candidate = pkg_dir / file_path_str
        if candidate.exists():
            return _read_source_lines(candidate, line_start, line_end)

    raise FileNotFoundError(f"Could not find {file_path_str} for package {package_name}")


async def extract_declarations(engine: AsyncEngine) -> None:
    """Extract all declarations from doc-gen4 data and load into database.

    Automatically finds the lean/.lake directory from the root.
    Extracts all declarations, then inserts them into the database.

    Args:
        engine: SQLAlchemy async engine for database connection.
    """
    lean_root = Path("lean")
    doc_data_dir = lean_root / ".lake" / "build" / "doc-data"

    if not doc_data_dir.exists():
        raise FileNotFoundError(f"Doc-data directory not found: {doc_data_dir}")

    bmp_files = sorted(doc_data_dir.glob("**/*.bmp"))

    # Build package cache once
    package_cache = _build_package_cache(lean_root)
    logger.info(f"Found {len(package_cache)} packages: {list(package_cache.keys())}")

    declarations = []
    for file_path in bmp_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        module_name = data["name"]

        for decl_data in data.get("declarations", []):
            info = decl_data["info"]
            source_text = _extract_source_text(info["sourceLink"], lean_root, package_cache)

            header_html = decl_data.get("header", "")
            dependencies = _extract_dependencies_from_html(header_html)

            declarations.append(
                Declaration(
                    name=info["name"],
                    module=module_name,
                    docstring=info.get("doc"),
                    source_text=source_text,
                    source_link=info["sourceLink"],
                    dependencies=dependencies if dependencies else None,
                )
            )

    batch_size = 1000
    inserted_count = 0
    async with AsyncSession(engine) as session:
        async with session.begin():
            for i in range(0, len(declarations), batch_size):
                batch = declarations[i : i + batch_size]

                # Use INSERT ON CONFLICT to skip duplicates
                for decl in batch:
                    stmt = insert(DBDeclaration).values(
                        name=decl.name,
                        module=decl.module,
                        docstring=decl.docstring,
                        source_text=decl.source_text,
                        source_link=decl.source_link,
                        dependencies=json.dumps(decl.dependencies) if decl.dependencies else None,
                    ).on_conflict_do_nothing(index_elements=["name"])

                    result = await session.execute(stmt)
                    inserted_count += result.rowcount

    logger.info(f"Inserted {inserted_count} new declarations into database (skipped {len(declarations) - inserted_count} duplicates)")
