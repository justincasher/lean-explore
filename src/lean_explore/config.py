# src/lean_explore/config.py

"""Centralized configuration for lean_explore.

This module provides all configuration settings including paths, URLs,
and other constants used throughout the application.
"""

import os
import pathlib


class Config:
    """Application-wide configuration settings."""

    CACHE_DIRECTORY: pathlib.Path = pathlib.Path(
        os.getenv(
            "LEAN_EXPLORE_CACHE_DIR",
            pathlib.Path.home() / ".lean_explore" / "cache",
        )
    )
    """Cache directory for downloaded data (used by search engine and MCP server).

    Can be overridden with LEAN_EXPLORE_CACHE_DIR environment variable.
    Default: ~/.lean_explore/cache
    """

    DATA_DIRECTORY: pathlib.Path = pathlib.Path(
        os.getenv(
            "LEAN_EXPLORE_DATA_DIR",
            pathlib.Path(__file__).parent.parent.parent / "data",
        )
    )
    """Local data directory for extraction pipeline output.

    Can be overridden with LEAN_EXPLORE_DATA_DIR environment variable.
    Default: <repo-root>/data
    """

    DEFAULT_LEAN_VERSION: str = "4.24.0"
    """Lean version for database naming and dependency resolution."""

    ACTIVE_VERSION: str = f"v{DEFAULT_LEAN_VERSION}"
    """Active version identifier (e.g., v4.24.0)."""

    ACTIVE_CACHE_PATH: pathlib.Path = CACHE_DIRECTORY / ACTIVE_VERSION
    """Directory for the active version's cached data files."""

    ACTIVE_DATA_PATH: pathlib.Path = DATA_DIRECTORY / ACTIVE_VERSION
    """Directory for the active version's local data files."""

    DATABASE_PATH: pathlib.Path = ACTIVE_CACHE_PATH / "lean_explore.db"
    """Path to SQLite database file in cache (used by search engine)."""

    FAISS_INDEX_PATH: pathlib.Path = ACTIVE_CACHE_PATH / "informalization_faiss.index"
    """Path to FAISS index file in cache (using informalization embeddings)."""

    FAISS_IDS_MAP_PATH: pathlib.Path = (
        ACTIVE_CACHE_PATH / "informalization_faiss_ids_map.json"
    )
    """Path to FAISS ID mapping file in cache."""

    DATABASE_URL: str = f"sqlite+aiosqlite:///{DATABASE_PATH}"
    """Async SQLAlchemy database URL for SQLite (used by search engine)."""

    EXTRACTION_DATABASE_PATH: pathlib.Path = ACTIVE_DATA_PATH / "lean_explore.db"
    """Path to SQLite database file in data directory (used by extraction)."""

    EXTRACTION_DATABASE_URL: str = f"sqlite+aiosqlite:///{EXTRACTION_DATABASE_PATH}"
    """Async SQLAlchemy database URL for extraction pipeline."""

    INFORMAL_CACHE_PATH: pathlib.Path = DATA_DIRECTORY / "informal_cache.db"
    """Path to shared informalization cache database (version-independent)."""

    INFORMAL_CACHE_URL: str = f"sqlite+aiosqlite:///{INFORMAL_CACHE_PATH}"
    """Async SQLAlchemy database URL for informalization cache."""

    EXTRACT_PACKAGES: set[str] = {
        "batteries",
        "init",
        "lean4",
        "mathlib",
        "physlean",
        "std",
    }
    """Set of package names to extract from doc-gen4 output."""

    MANIFEST_URL: str = (
        "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"
    )
    """Remote URL for the data toolchain manifest."""

    R2_ASSETS_BASE_URL: str = "https://pub-48b75babc4664808b15520033423c765.r2.dev"
    """Base URL for Cloudflare R2 asset storage."""

    API_BASE_URL: str = "https://www.leanexplore.com/api/v1"
    """Base URL for the LeanExplore remote API service."""
