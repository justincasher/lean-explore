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

    Each toolchain version has its own subdirectory containing database,
    FAISS index, and other data files.
    """

    ACTIVE_TOOLCHAIN_VERSION: str = os.getenv(
        "LEAN_EXPLORE_TOOLCHAIN_VERSION", "0.2.0"
    )
    """Version identifier for the currently active data toolchain.

    Can be overridden with LEAN_EXPLORE_TOOLCHAIN_VERSION environment variable.
    Default: 0.2.0
    """

    ACTIVE_TOOLCHAIN_DIRECTORY: pathlib.Path = (
        TOOLCHAINS_DIRECTORY / ACTIVE_TOOLCHAIN_VERSION
    )
    """Directory path for the active toolchain version's data files."""

    DATABASE_URL: str = os.getenv(
        "LEAN_EXPLORE_DATABASE_URL",
        "postgresql+psycopg2://postgres:@localhost:5432/lean_explore",
    )
    """SQLAlchemy database URL for the active toolchain database.

    Can be overridden with LEAN_EXPLORE_DATABASE_URL environment variable.
    Default: postgresql+psycopg2://postgres:@localhost:5432/lean_explore
    """

    DATABASE_URL_ASYNC: str = os.getenv(
        "LEAN_EXPLORE_DATABASE_URL_ASYNC",
        "postgresql+asyncpg://postgres:@localhost:5432/lean_explore",
    )
    """Async SQLAlchemy database URL for the extract pipeline.

    Can be overridden with LEAN_EXPLORE_DATABASE_URL_ASYNC environment variable.
    Default: postgresql+asyncpg://postgres:@localhost:5432/lean_explore
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
