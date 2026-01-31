"""Documentation generation using doc-gen4 for each package.

This module provides functionality to run doc-gen4 on each package workspace
to generate Lean documentation data for the extraction pipeline.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from lean_explore.extract.github import extract_lean_version
from lean_explore.extract.package_config import PackageConfig
from lean_explore.extract.package_registry import PACKAGE_REGISTRY
from lean_explore.extract.package_utils import (
    get_extraction_order,
    get_package_toolchain,
    update_lakefile_docgen_version,
)

logger = logging.getLogger(__name__)


def _clear_workspace_cache(workspace_path: Path) -> None:
    """Clear entire Lake cache to force complete rebuild.

    Removes the .lake/ directory and lake-manifest.json to ensure:
    1. Fresh dependency resolution (latest compatible versions)
    2. Fresh doc-gen4 output (regenerated BMP files)
    3. No stale build artifacts

    Use this for nightly updates to get a clean build from scratch.

    Args:
        workspace_path: Path to the package workspace.
    """
    manifest = workspace_path / "lake-manifest.json"
    if manifest.exists():
        logger.info(f"Removing {manifest}")
        manifest.unlink()

    lake_dir = workspace_path / ".lake"
    if lake_dir.exists():
        logger.info(f"Removing {lake_dir} to force complete rebuild")
        shutil.rmtree(lake_dir)


def _get_doc_lib_names(package_name: str) -> list[str]:
    """Get the library names to run doc-gen4 on for a package.

    Some packages have custom extract wrappers, others use upstream libraries directly.
    """
    lib_names: dict[str, list[str]] = {
        "mathlib": ["MathExtract"],
        "physlean": ["PhysExtract"],
        "flt": ["FLTExtract"],
        "formal-conjectures": ["FormalConjectures", "FormalConjecturesForMathlib"],
        "cslib": ["CslibExtract"],
    }
    return lib_names.get(package_name, [f"{package_name.title()}Extract"])


def _setup_workspace(package_config: PackageConfig) -> tuple[str, str]:
    """Fetch toolchain from GitHub and update lakefile.

    Returns:
        Tuple of (lean_toolchain, git_ref).
    """
    workspace_path = Path("lean") / package_config.name
    lakefile_path = workspace_path / "lakefile.lean"
    toolchain_file = workspace_path / "lean-toolchain"

    lean_toolchain, git_ref = get_package_toolchain(package_config)
    lean_version = extract_lean_version(lean_toolchain)

    update_lakefile_docgen_version(lakefile_path, lean_version)
    toolchain_file.write_text(lean_toolchain + "\n")

    return lean_toolchain, git_ref


def _run_lake_update_with_retry(
    workspace_path: Path,
    package_name: str,
    env: dict[str, str],
    verbose: bool = False,
    max_retries: int = 3,
    base_delay: float = 30.0,
) -> None:
    """Run ``lake update`` with retries for transient network failures.

    Large repositories like mathlib4 require cloning several gigabytes of git
    data. Transient network issues (DNS blips, connection resets, GitHub
    throttling) can cause the clone to fail with git exit code 128. Retrying
    with exponential backoff handles these cases.

    Args:
        workspace_path: Path to the Lake workspace directory.
        package_name: Name of the package (for log messages).
        env: Environment variables to pass to the subprocess.
        verbose: Log stdout from ``lake update``.
        max_retries: Maximum number of retry attempts after the initial try.
        base_delay: Seconds to wait before the first retry. Doubles each retry.
    """
    for attempt in range(1, max_retries + 2):
        logger.info(f"[{package_name}] Running lake update (attempt {attempt})...")
        result = subprocess.run(
            ["lake", "update"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            env=env,
        )
        if verbose and result.stdout:
            logger.info(result.stdout)
        if result.returncode == 0:
            return

        if attempt <= max_retries:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"[{package_name}] lake update failed (attempt {attempt}), "
                f"retrying in {delay:.0f}s..."
            )
            logger.warning(f"[{package_name}] stderr: {result.stderr.strip()}")
            time.sleep(delay)
        else:
            logger.error(result.stderr)
            raise RuntimeError(f"lake update failed for {package_name}")


def _run_lake_for_package(package_name: str, verbose: bool = False) -> None:
    """Run lake update, cache get, and doc-gen4 for a package."""
    workspace_path = Path("lean") / package_name
    package_config = PACKAGE_REGISTRY[package_name]
    env = os.environ.copy()
    env["MATHLIB_NO_CACHE_ON_UPDATE"] = "1"

    _run_lake_update_with_retry(workspace_path, package_name, env, verbose)

    # Fetch mathlib cache for packages that depend on mathlib
    if "mathlib" in package_config.depends_on or package_name == "mathlib":
        logger.info(f"[{package_name}] Fetching mathlib cache...")
        result = subprocess.run(
            ["lake", "exe", "cache", "get"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            env=env,
        )
        if verbose and result.stdout:
            logger.info(result.stdout)
        if result.returncode != 0:
            logger.warning(f"[{package_name}] Cache fetch failed (non-fatal)")

    logger.info(f"[{package_name}] Running lake build...")
    process = subprocess.Popen(
        ["lake", "build"],
        cwd=workspace_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"lake build failed for {package_name}")

    lib_names = _get_doc_lib_names(package_name)
    for lib_name in lib_names:
        logger.info(f"[{package_name}] Running doc-gen4 ({lib_name}:docs)...")

        process = subprocess.Popen(
            ["lake", "build", f"{lib_name}:docs"],
            cwd=workspace_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
        returncode = process.wait()
        if returncode != 0:
            logger.warning(
                f"[{package_name}] doc-gen4 had failures for {lib_name} "
                "(continuing with generated docs)"
            )


async def run_doc_gen4(
    packages: list[str] | None = None,
    setup: bool = True,
    fresh: bool = False,
    verbose: bool = False,
) -> None:
    """Run doc-gen4 for each package to generate documentation data.

    Args:
        packages: List of package names to process. If None, processes all packages
            in dependency order.
        setup: Whether to fetch toolchain and update lakefile before building.
        fresh: Clear cached dependencies to force fresh resolution. Use this for
            nightly updates to get the latest compatible versions of all packages.
        verbose: Enable verbose logging.

    Raises:
        RuntimeError: If any build step fails.
    """
    if packages is None:
        packages = get_extraction_order()

    logger.info(f"Running doc-gen4 for packages: {', '.join(packages)}")

    for package_name in packages:
        if package_name not in PACKAGE_REGISTRY:
            raise ValueError(f"Unknown package: {package_name}")

        config = PACKAGE_REGISTRY[package_name]
        workspace_path = Path("lean") / package_name
        logger.info(f"\n{'=' * 50}\nPackage: {package_name}\n{'=' * 50}")

        if fresh:
            _clear_workspace_cache(workspace_path)

        if setup:
            toolchain, ref = _setup_workspace(config)
            logger.info(f"Toolchain: {toolchain}, ref: {ref}")

        _run_lake_for_package(package_name, verbose)

    logger.info("doc-gen4 generation complete for all packages")
