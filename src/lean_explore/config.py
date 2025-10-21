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
    """Directory containing versioned data toolchains."""

    ACTIVE_TOOLCHAIN_VERSION: str = "1.0.0"
    """Version identifier for the currently active data toolchain."""

    ACTIVE_TOOLCHAIN_DIRECTORY: pathlib.Path = (
        TOOLCHAINS_DIRECTORY / ACTIVE_TOOLCHAIN_VERSION
    )
    """Directory for the active toolchain version's data files."""

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
    """Database name constructed from Lean version (e.g., lean_explore_4.23.0)."""

    DATABASE_URL: str = f"{DB_BASE_URL}/{DATABASE_NAME}"
    """Async SQLAlchemy database URL constructed from DB_BASE_URL and DATABASE_NAME."""

    MANIFEST_URL: str = (
        "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"
    )
    """Remote URL for the data toolchain manifest."""

    R2_ASSETS_BASE_URL: str = "https://pub-48b75babc4664808b15520033423c765.r2.dev"
    """Base URL for Cloudflare R2 asset storage."""

    API_BASE_URL: str = "https://www.leanexplore.com/api/v1"
    """Base URL for the LeanExplore remote API service."""
