"""Package configuration for Lean extraction.

This module defines the configuration dataclass and version strategy enum
for Lean packages to extract.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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
