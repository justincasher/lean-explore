# src/lean_explore/defaults.py

"""Provides default paths and configuration parameters for the lean_explore package.

This module centralizes default values. The defined paths point to
a user-specific data directory where downloaded assets (database,
FAISS index, etc.) for a specific toolchain version are expected to reside.
"""

import os
import pathlib
from typing import Final

# --- User-Specific Data Directory ---
# Define a base directory within the user's home folder to store
# downloaded data assets for lean_explore.
# Example: ~/.lean_explore/data/
USER_HOME_DIR: Final[pathlib.Path] = pathlib.Path(os.path.expanduser("~"))
LEAN_EXPLORE_USER_DATA_DIR: Final[pathlib.Path] = (
    USER_HOME_DIR / ".lean_explore" / "data"
)

# --- Toolchain Specific Paths ---
# Directory within the user data directory to store versioned toolchain data.
# Example: ~/.lean_explore/data/toolchains/
LEAN_EXPLORE_TOOLCHAINS_BASE_DIR: Final[pathlib.Path] = (
    LEAN_EXPLORE_USER_DATA_DIR / "toolchains"
)

# Default active toolchain version.
# In future enhancements, this could be determined dynamically
# or from user configuration.
# For now, it's set to the initial version of data provided ("0.1.0").
DEFAULT_ACTIVE_TOOLCHAIN_VERSION: Final[str] = "0.1.0"

# Path to the data directory for the currently active toolchain version.
# Example: ~/.lean_explore/data/toolchains/0.1.0/
_ACTIVE_TOOLCHAIN_VERSION_DATA_PATH: Final[pathlib.Path] = (
    LEAN_EXPLORE_TOOLCHAINS_BASE_DIR / DEFAULT_ACTIVE_TOOLCHAIN_VERSION
)

# --- Default Filenames (names of the asset files themselves) ---
DEFAULT_DB_FILENAME: Final[str] = "lean_explore_data.db"
DEFAULT_FAISS_INDEX_FILENAME: Final[str] = "main_faiss.index"
DEFAULT_FAISS_MAP_FILENAME: Final[str] = "faiss_ids_map.json"

# --- Default Full Paths (to be used by the application for the active toolchain) ---
# These paths indicate where the package will look for its data files
# for the currently active toolchain version. The data management component
# will be responsible for downloading files to these versioned locations.

DEFAULT_DB_PATH: Final[pathlib.Path] = (
    _ACTIVE_TOOLCHAIN_VERSION_DATA_PATH / DEFAULT_DB_FILENAME
)
DEFAULT_FAISS_INDEX_PATH: Final[pathlib.Path] = (
    _ACTIVE_TOOLCHAIN_VERSION_DATA_PATH / DEFAULT_FAISS_INDEX_FILENAME
)
DEFAULT_FAISS_MAP_PATH: Final[pathlib.Path] = (
    _ACTIVE_TOOLCHAIN_VERSION_DATA_PATH / DEFAULT_FAISS_MAP_FILENAME
)

# For SQLAlchemy, the database URL needs to be a string.
# We construct the SQLite URL string from the Path object.
DEFAULT_DB_URL: Final[str] = f"sqlite:///{DEFAULT_DB_PATH.resolve()}"


# --- Remote Data Asset Defaults ---
# These constants are used by the data management commands to locate and
# manage remote toolchain data assets.

# Default URL for the master manifest file on R2.
R2_MANIFEST_DEFAULT_URL: Final[str] = (
    "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"
)

# Base URL for accessing assets on R2. Specific file paths from the manifest
# will be appended to this base.
R2_ASSETS_BASE_URL: Final[str] = "https://pub-48b75babc4664808b15520033423c765.r2.dev/"

# Filename for storing the currently selected active toolchain version.
# This file will reside in LEAN_EXPLORE_USER_DATA_DIR.
# Example: ~/.lean_explore/data/active_toolchain.txt
ACTIVE_TOOLCHAIN_CONFIG_FILENAME: Final[str] = "active_toolchain.txt"

# Full path to the active toolchain configuration file.
ACTIVE_TOOLCHAIN_CONFIG_FILE_PATH: Final[pathlib.Path] = (
    LEAN_EXPLORE_USER_DATA_DIR / ACTIVE_TOOLCHAIN_CONFIG_FILENAME
)


# --- Default Embedding Model ---
DEFAULT_EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-base-en-v1.5"


# --- Default Search Parameters ---
# These values are based on the previously discussed config.yml and search.py fallbacks.

# FAISS Search Parameters
DEFAULT_FAISS_K: Final[int] = 100  # Number of nearest neighbors from FAISS
DEFAULT_FAISS_NPROBE: Final[int] = 200  # For IVF-type FAISS indexes

# Scoring and Ranking Parameters
DEFAULT_SEM_SIM_THRESHOLD: Final[float] = 0.525
DEFAULT_PAGERANK_WEIGHT: Final[float] = 1.0
DEFAULT_TEXT_RELEVANCE_WEIGHT: Final[float] = 0.2
DEFAULT_NAME_MATCH_WEIGHT: Final[float] = 0.5

# Output Parameters
DEFAULT_RESULTS_LIMIT: Final[int] = (
    50  # Default number of final results to display/return
)


# --- Other Constants (if any emerge) ---
# Example: If your application needs other hardcoded default values,
# they can be added here.
# DEFAULT_SOME_OTHER_PARAMETER: Final[Any] = "some_value"
