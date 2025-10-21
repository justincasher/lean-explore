# src/lean_explore/config.py

"""Centralized configuration for lean_explore.

This module provides all configuration settings including paths, URLs,
and other constants used throughout the application.
"""

import os
import pathlib


class Config:
    """Application-wide configuration settings."""

    DATA_DIRECTORY: pathlib.Path = pathlib.Path(
        os.getenv(
            "LEAN_EXPLORE_DATA_DIR",
            pathlib.Path.home() / ".lean_explore" / "data",
        )
    )
    """Base directory for all lean_explore data storage.

    Can be overridden with LEAN_EXPLORE_DATA_DIR environment variable.
    Default: ~/.lean_explore/data
    """

    TOOLCHAINS_DIRECTORY: pathlib.Path = DATA_DIRECTORY / "toolchains"
    """Directory containing versioned data toolchains.

    Each toolchain version has its own subdirectory containing databases
    and other version-specific data files.
    """

    ACTIVE_TOOLCHAIN_VERSION: str = "1.0.0"
    """Version identifier for the currently active data toolchain."""

    ACTIVE_TOOLCHAIN_DIRECTORY: pathlib.Path = (
        TOOLCHAINS_DIRECTORY / ACTIVE_TOOLCHAIN_VERSION
    )
    """Directory path for the active toolchain version's data files."""

    DEFAULT_LEAN_VERSION: str = os.getenv("LEAN_EXPLORE_LEAN_VERSION", "4.23.0")
    """Default Lean version for database naming.

    Can be overridden with LEAN_EXPLORE_LEAN_VERSION environment variable.
    Default: 4.23.0
    """

    DB_BASE_URL: str = os.getenv(
        "LEAN_EXPLORE_DB_BASE_URL",
        "postgresql+asyncpg://postgres:@localhost:5432",
    )
    """Base PostgreSQL connection URL without database name.

    Can be overridden with LEAN_EXPLORE_DB_BASE_URL environment variable.
    Default: postgresql+asyncpg://postgres:@localhost:5432
    """

    DATABASE_NAME: str = f"lean_explore_{DEFAULT_LEAN_VERSION}"
    """Database name constructed from Lean version.

    Format: lean_explore_{lean_version}
    Example: lean_explore_4.23.0
    """

    DATABASE_URL: str = f"{DB_BASE_URL}/{DATABASE_NAME}"
    """Async SQLAlchemy database URL for PostgreSQL.

    Constructed from DB_BASE_URL and DATABASE_NAME.
    Default: postgresql+asyncpg://postgres:@localhost:5432/lean_explore_{version}
    """

    MANIFEST_URL: str = (
        "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"
    )
    """Remote URL for the data toolchain manifest.

    The manifest contains metadata about available toolchain versions,
    file checksums, and download locations.
    """

    R2_ASSETS_BASE_URL: str = (
        "https://pub-48b75babc4664808b15520033423c765.r2.dev"
    )
    """Base URL for Cloudflare R2 asset storage.

    Used to construct full asset URLs by appending the asset path.
    """

    API_BASE_URL: str = "https://www.leanexplore.com/api/v1"
    """Base URL for the LeanExplore remote API service.

    Used by the API client to make requests to the remote backend.
    """
