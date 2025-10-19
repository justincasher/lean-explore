"""Path configuration for lean_explore data storage.

This module provides centralized path configuration, with support for
environment variable overrides.
"""

import os
import pathlib
from typing import Final

# Base data directory
# Can be overridden with LEAN_EXPLORE_DATA_DIR environment variable
_DATA_DIRECTORY_DEFAULT = pathlib.Path.home() / ".lean_explore" / "data"
DATA_DIRECTORY: Final[pathlib.Path] = pathlib.Path(
    os.getenv("LEAN_EXPLORE_DATA_DIR", _DATA_DIRECTORY_DEFAULT)
)

# Toolchains directory
TOOLCHAINS_DIRECTORY: Final[pathlib.Path] = DATA_DIRECTORY / "toolchains"

# Active toolchain version
# Can be overridden with LEAN_EXPLORE_TOOLCHAIN_VERSION environment variable
ACTIVE_TOOLCHAIN_VERSION: Final[str] = os.getenv(
    "LEAN_EXPLORE_TOOLCHAIN_VERSION", "0.2.0"
)

# Active toolchain data directory
ACTIVE_TOOLCHAIN_DIRECTORY: Final[pathlib.Path] = (
    TOOLCHAINS_DIRECTORY / ACTIVE_TOOLCHAIN_VERSION
)

# Data file paths for the active toolchain
DATABASE_PATH: Final[pathlib.Path] = ACTIVE_TOOLCHAIN_DIRECTORY / "lean_explore_data.db"
FAISS_INDEX_PATH: Final[pathlib.Path] = ACTIVE_TOOLCHAIN_DIRECTORY / "main_faiss.index"
FAISS_MAP_PATH: Final[pathlib.Path] = ACTIVE_TOOLCHAIN_DIRECTORY / "faiss_ids_map.json"

# Database URL for SQLAlchemy
DATABASE_URL: Final[str] = f"sqlite:///{DATABASE_PATH.resolve()}"
