"""Utility functions for package configuration.

This module provides helper functions for working with the package registry,
including dependency ordering, toolchain resolution, and lakefile manipulation.
"""

import logging
import re
from pathlib import Path

from lean_explore.extract.package_config import PackageConfig, VersionStrategy
from lean_explore.extract.package_registry import PACKAGE_REGISTRY

logger = logging.getLogger(__name__)


def get_package_for_module(module_name: str) -> str | None:
    """Determine which package a module belongs to.

    Args:
        module_name: Fully qualified module name (e.g., 'Mathlib.Data.List.Basic')

    Returns:
        Package name or None if not recognized.
    """
    for package_name, configuration in PACKAGE_REGISTRY.items():
        if configuration.should_include_module(module_name):
            return package_name
    return None


def get_extraction_order() -> list[str]:
    """Get packages in dependency order for extraction.

    Returns packages ordered so dependencies come before dependents.
    """
    result: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        configuration = PACKAGE_REGISTRY.get(name)
        if configuration:
            for dep in configuration.depends_on:
                visit(dep)
            result.append(name)

    for name in PACKAGE_REGISTRY:
        visit(name)

    return result


def get_package_toolchain(package_configuration: PackageConfig) -> tuple[str, str]:
    """Get the toolchain and ref for a package based on its version strategy.

    Args:
        package_configuration: Package configuration

    Returns:
        Tuple of (lean_toolchain, git_ref) where git_ref is the branch/tag to use.
    """
    from lean_explore.extract.github import fetch_latest_tag, fetch_lean_toolchain

    if package_configuration.version_strategy == VersionStrategy.LATEST:
        for branch in ["main", "master"]:
            try:
                toolchain = fetch_lean_toolchain(package_configuration.git_url, branch)
                return toolchain, branch
            except RuntimeError:
                continue
        raise RuntimeError(
            f"Could not fetch toolchain from main or master for "
            f"{package_configuration.name}"
        )
    else:
        latest_tag = fetch_latest_tag(package_configuration.git_url)
        toolchain = fetch_lean_toolchain(package_configuration.git_url, latest_tag)
        return toolchain, latest_tag


def update_lakefile_docgen_version(lakefile_path: Path, lean_version: str) -> None:
    """Update the doc-gen4 version in a lakefile to match the Lean version.

    Args:
        lakefile_path: Path to lakefile.lean
        lean_version: Lean version like 'v4.27.0'
    """
    content = lakefile_path.read_text()

    pattern = (
        r"require «doc-gen4» from git\s+"
        r'"https://github\.com/leanprover/doc-gen4"(?:\s+@\s+"[^"]*")?'
    )
    replacement = (
        f"require «doc-gen4» from git\n"
        f'  "https://github.com/leanprover/doc-gen4" @ "{lean_version}"'
    )
    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        lakefile_path.write_text(new_content)
        logger.info(f"Updated doc-gen4 version to {lean_version} in {lakefile_path}")
