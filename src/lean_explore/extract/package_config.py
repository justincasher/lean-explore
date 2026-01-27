"""Package configuration for Lean extraction.

This module defines the registry of Lean packages to extract, including their
GitHub URLs, module prefixes, and version strategies.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class VersionStrategy(Enum):
    """Strategy for selecting which version of a package to extract."""

    LATEST = "latest"
    """Use HEAD/main branch - for packages with CI that ensures main compiles."""

    TAGGED = "tagged"
    """Use the latest git tag - safer for downstream packages."""


@dataclass
class PackageConfig:
    """Configuration for a Lean package extraction."""

    name: str
    """Package name (e.g., 'mathlib', 'physlean')."""

    git_url: str
    """GitHub repository URL."""

    module_prefixes: list[str]
    """Module name prefixes that belong to this package (e.g., ['Mathlib'])."""

    version_strategy: VersionStrategy = VersionStrategy.TAGGED
    """Strategy for selecting the version to extract."""

    lean_toolchain: str | None = None
    """Override Lean toolchain version. If None, determined from package."""

    depends_on: list[str] = field(default_factory=list)
    """List of package names this package depends on (for extraction ordering)."""

    extract_core: bool = False
    """If True, also extract Init/Lean/Std modules from this package's toolchain."""

    def workspace_path(self, base_path: Path) -> Path:
        """Get the workspace path for this package."""
        return base_path / self.name

    def should_include_module(self, module_name: str) -> bool:
        """Check if a module belongs to this package based on prefixes.

        Uses exact match or prefix + "." to avoid "Lean" matching "LeanSearchClient".
        """
        return any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in self.module_prefixes
        )


# =============================================================================
# Package Registry
# =============================================================================

PACKAGE_REGISTRY: dict[str, PackageConfig] = {
    "mathlib": PackageConfig(
        name="mathlib",
        git_url="https://github.com/leanprover-community/mathlib4",
        module_prefixes=["Mathlib", "Batteries", "Init", "Lean", "Std"],
        version_strategy=VersionStrategy.LATEST,
        depends_on=[],
        extract_core=True,
    ),
    "physlean": PackageConfig(
        name="physlean",
        git_url="https://github.com/HEPLean/PhysLean",
        module_prefixes=["PhysLean"],
        version_strategy=VersionStrategy.TAGGED,
        depends_on=["mathlib"],
    ),
    "flt": PackageConfig(
        name="flt",
        git_url="https://github.com/ImperialCollegeLondon/FLT",
        module_prefixes=["FLT"],
        version_strategy=VersionStrategy.LATEST,
        depends_on=["mathlib"],
    ),
    "formal-conjectures": PackageConfig(
        name="formal-conjectures",
        git_url="https://github.com/google-deepmind/formal-conjectures",
        module_prefixes=["FormalConjectures", "FormalConjecturesForMathlib"],
        version_strategy=VersionStrategy.LATEST,
        depends_on=["mathlib"],
    ),
    "cslib": PackageConfig(
        name="cslib",
        git_url="https://github.com/leanprover/cslib",
        module_prefixes=["Cslib"],
        version_strategy=VersionStrategy.LATEST,
        depends_on=["mathlib"],
    ),
}


# =============================================================================
# Query Functions
# =============================================================================


def get_package_for_module(module_name: str) -> str | None:
    """Determine which package a module belongs to.

    Args:
        module_name: Fully qualified module name (e.g., 'Mathlib.Data.List.Basic')

    Returns:
        Package name or None if not recognized.
    """
    for package_name, config in PACKAGE_REGISTRY.items():
        if config.should_include_module(module_name):
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
        config = PACKAGE_REGISTRY.get(name)
        if config:
            for dep in config.depends_on:
                visit(dep)
            result.append(name)

    for name in PACKAGE_REGISTRY:
        visit(name)

    return result


def get_package_toolchain(package_config: PackageConfig) -> tuple[str, str]:
    """Get the toolchain and ref for a package based on its version strategy.

    Args:
        package_config: Package configuration

    Returns:
        Tuple of (lean_toolchain, git_ref) where git_ref is the branch/tag to use.
    """
    from lean_explore.extract.github import fetch_latest_tag, fetch_lean_toolchain

    if package_config.version_strategy == VersionStrategy.LATEST:
        for branch in ["main", "master"]:
            try:
                toolchain = fetch_lean_toolchain(package_config.git_url, branch)
                return toolchain, branch
            except RuntimeError:
                continue
        raise RuntimeError(
            f"Could not fetch toolchain from main or master for {package_config.name}"
        )
    else:
        latest_tag = fetch_latest_tag(package_config.git_url)
        toolchain = fetch_lean_toolchain(package_config.git_url, latest_tag)
        return toolchain, latest_tag


def update_lakefile_docgen_version(lakefile_path: Path, lean_version: str) -> None:
    """Update the doc-gen4 version in a lakefile to match the Lean version.

    Args:
        lakefile_path: Path to lakefile.lean
        lean_version: Lean version like 'v4.27.0'
    """
    content = lakefile_path.read_text()

    pattern = (
        r'require «doc-gen4» from git\s+'
        r'"https://github\.com/leanprover/doc-gen4"(?:\s+@\s+"[^"]*")?'
    )
    replacement = (
        f'require «doc-gen4» from git\n'
        f'  "https://github.com/leanprover/doc-gen4" @ "{lean_version}"'
    )
    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        lakefile_path.write_text(new_content)
        logger.info(f"Updated doc-gen4 version to {lean_version} in {lakefile_path}")
