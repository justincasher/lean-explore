# tests/test_defaults.py
"""Tests for the lean_explore.defaults module.

This module verifies the correctness of default path constructions and
constant values defined in `lean_explore.defaults`. It also tests the
effect of the `isolated_data_paths` fixture, ensuring that path constants
are correctly monkeypatched to use temporary directories during testing.
"""

import os
import pathlib

import pytest

from lean_explore import defaults as project_defaults


class TestOriginalDefaults:
    """Tests the original values and structures of constants in defaults.py.

    These tests mock `os.path.expanduser` to ensure that path constructions
    based on the user's home directory are predictable and testable without
    depending on the actual execution environment's home directory for
    constructing expected relative paths.
    """

    @pytest.fixture
    def mock_home_dir(self, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
        """Mocks os.path.expanduser to return a fixed mock home directory.

        Args:
            monkeypatch: Pytest fixture for modifying object attributes.

        Returns:
            pathlib.Path: The path to the mocked home directory.
        """
        mock_home = pathlib.Path("/mock/home/testuser")
        monkeypatch.setattr(
            os.path,
            "expanduser",
            lambda path_str: str(mock_home) if path_str == "~" else path_str,
        )
        return mock_home

    def test_user_specific_directory_structure(self, mock_home_dir: pathlib.Path):
        """Verifies the construction of user-specific base directories.

        This test checks if the derived paths in `project_defaults`
        (like LEAN_EXPLORE_USER_DATA_DIR) are correctly constructed
        relative to `USER_HOME_DIR`.

        Args:
            mock_home_dir: The mocked home directory path (its primary use here
                           is to enable the test of expanduser in other contexts,
                           but for these assertions, we focus on derivations from
                           the already set USER_HOME_DIR).
        """
        # The following assertions check the relative construction logic
        # based on the value of project_defaults.USER_HOME_DIR set at import time.
        assert (
            project_defaults.LEAN_EXPLORE_USER_DATA_DIR
            == project_defaults.USER_HOME_DIR / ".lean_explore" / "data"
        )
        assert (
            project_defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR
            == project_defaults.LEAN_EXPLORE_USER_DATA_DIR / "toolchains"
        )

    def test_active_toolchain_path_structure(self, mock_home_dir: pathlib.Path):
        """Verifies the active toolchain path construction.

        Args:
            mock_home_dir: The mocked home directory path.
        """
        # Similar to above, testing the logic.
        expected_base = (
            project_defaults.USER_HOME_DIR / ".lean_explore" / "data" / "toolchains"
        )
        expected_active_toolchain_path = (
            expected_base / project_defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION
        )
        assert (
            project_defaults._ACTIVE_TOOLCHAIN_VERSION_DATA_PATH
            == expected_active_toolchain_path
        )

    def test_default_asset_paths_structure(self, mock_home_dir: pathlib.Path):
        """Verifies the structure of default asset file paths.

        Args:
            mock_home_dir: The mocked home directory path.
        """
        base_path = project_defaults._ACTIVE_TOOLCHAIN_VERSION_DATA_PATH
        assert (
            project_defaults.DEFAULT_DB_PATH
            == base_path / project_defaults.DEFAULT_DB_FILENAME
        )
        assert (
            project_defaults.DEFAULT_FAISS_INDEX_PATH
            == base_path / project_defaults.DEFAULT_FAISS_INDEX_FILENAME
        )
        assert (
            project_defaults.DEFAULT_FAISS_MAP_PATH
            == base_path / project_defaults.DEFAULT_FAISS_MAP_FILENAME
        )

    def test_default_db_url_format(self, mock_home_dir: pathlib.Path):
        """Verifies the format of the default SQLite database URL.

        Args:
            mock_home_dir: The mocked home directory path.
        """
        # The .resolve() in the original DEFAULT_DB_URL definition might behave
        # differently if the path doesn't exist. For testing the format, we assume
        # DEFAULT_DB_PATH is what it would be.
        expected_db_path_str = str(project_defaults.DEFAULT_DB_PATH.resolve())
        assert project_defaults.DEFAULT_DB_URL == f"sqlite:///{expected_db_path_str}"

    def test_active_toolchain_config_file_path_structure(
        self, mock_home_dir: pathlib.Path
    ):
        """Verifies the path for the active toolchain configuration file.

        Args:
            mock_home_dir: The mocked home directory path.
        """
        expected_path = (
            project_defaults.LEAN_EXPLORE_USER_DATA_DIR
            / project_defaults.ACTIVE_TOOLCHAIN_CONFIG_FILENAME
        )
        assert project_defaults.ACTIVE_TOOLCHAIN_CONFIG_FILE_PATH == expected_path

    def test_string_constants_values(self):
        """Checks the values of important string constants."""
        assert project_defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION == "0.1.0"
        assert project_defaults.DEFAULT_DB_FILENAME == "lean_explore_data.db"
        assert project_defaults.DEFAULT_FAISS_INDEX_FILENAME == "main_faiss.index"
        assert project_defaults.DEFAULT_FAISS_MAP_FILENAME == "faiss_ids_map.json"
        assert (
            project_defaults.R2_MANIFEST_DEFAULT_URL
            == "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"
        )
        assert (
            project_defaults.R2_ASSETS_BASE_URL
            == "https://pub-48b75babc4664808b15520033423c765.r2.dev/"
        )
        assert (
            project_defaults.ACTIVE_TOOLCHAIN_CONFIG_FILENAME == "active_toolchain.txt"
        )
        assert project_defaults.DEFAULT_EMBEDDING_MODEL_NAME == "BAAI/bge-base-en-v1.5"

    def test_numeric_constants_values(self):
        """Checks the values of important numeric constants."""
        assert project_defaults.DEFAULT_FAISS_K == 100
        assert project_defaults.DEFAULT_FAISS_NPROBE == 200
        assert project_defaults.DEFAULT_SEM_SIM_THRESHOLD == 0.525
        assert project_defaults.DEFAULT_PAGERANK_WEIGHT == 1.0
        assert project_defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT == 0.2
        assert project_defaults.DEFAULT_NAME_MATCH_WEIGHT == 0.5
        assert project_defaults.DEFAULT_RESULTS_LIMIT == 50


class TestIsolatedDefaults:
    """Tests constants in defaults.py when `isolated_data_paths` fixture is active.

    These tests verify that the `isolated_data_paths` fixture correctly
    monkeypatches the path constants in `project_defaults` to point to
    locations within the temporary directory structure provided by the fixture.
    """

    def test_user_data_dir_is_isolated(
        self, isolated_data_paths: pathlib.Path, tmp_path: pathlib.Path
    ):
        """Verifies LEAN_EXPLORE_USER_DATA_DIR is redirected into tmp_path.

        Args:
            isolated_data_paths: Fixture providing the root of the isolated
                                 user data structure and patching defaults.
            tmp_path: Pytest's built-in temporary directory fixture.
        """
        # isolated_data_paths yields the root of the mocked user data structure
        # which is set as LEAN_EXPLORE_USER_DATA_DIR by the fixture.
        assert project_defaults.LEAN_EXPLORE_USER_DATA_DIR == isolated_data_paths
        assert tmp_path in project_defaults.LEAN_EXPLORE_USER_DATA_DIR.parents

    def test_toolchains_base_dir_is_isolated(
        self, isolated_data_paths: pathlib.Path, tmp_path: pathlib.Path
    ):
        """Verifies LEAN_EXPLORE_TOOLCHAINS_BASE_DIR is within the isolated structure.

        Args:
            isolated_data_paths: Fixture providing the isolated user data root.
            tmp_path: Pytest's built-in temporary directory fixture.
        """
        expected_toolchains_base = isolated_data_paths / "toolchains"
        assert (
            project_defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR
            == expected_toolchains_base
        )
        assert tmp_path in project_defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR.parents

    def test_active_toolchain_version_data_path_is_isolated(
        self, isolated_data_paths: pathlib.Path, tmp_path: pathlib.Path
    ):
        """Verifies _ACTIVE_TOOLCHAIN_VERSION_DATA_PATH is correctly isolated.

        Args:
            isolated_data_paths: Fixture providing the isolated user data root.
            tmp_path: Pytest's built-in temporary directory fixture.
        """
        expected_active_path = (
            isolated_data_paths
            / "toolchains"
            / project_defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION
        )
        assert (
            project_defaults._ACTIVE_TOOLCHAIN_VERSION_DATA_PATH == expected_active_path
        )
        assert tmp_path in project_defaults._ACTIVE_TOOLCHAIN_VERSION_DATA_PATH.parents

    def test_asset_paths_are_isolated(
        self, isolated_data_paths: pathlib.Path, tmp_path: pathlib.Path
    ):
        """Verifies that default asset paths (DB, FAISS) are isolated.

        Args:
            isolated_data_paths: Fixture providing the isolated user data root.
            tmp_path: Pytest's built-in temporary directory fixture.
        """
        mocked_active_toolchain_path = (
            project_defaults._ACTIVE_TOOLCHAIN_VERSION_DATA_PATH
        )

        expected_db_path = (
            mocked_active_toolchain_path / project_defaults.DEFAULT_DB_FILENAME
        )
        assert project_defaults.DEFAULT_DB_PATH == expected_db_path
        assert tmp_path in project_defaults.DEFAULT_DB_PATH.parents

        expected_faiss_index_path = (
            mocked_active_toolchain_path / project_defaults.DEFAULT_FAISS_INDEX_FILENAME
        )
        assert project_defaults.DEFAULT_FAISS_INDEX_PATH == expected_faiss_index_path
        assert tmp_path in project_defaults.DEFAULT_FAISS_INDEX_PATH.parents

        expected_faiss_map_path = (
            mocked_active_toolchain_path / project_defaults.DEFAULT_FAISS_MAP_FILENAME
        )
        assert project_defaults.DEFAULT_FAISS_MAP_PATH == expected_faiss_map_path
        assert tmp_path in project_defaults.DEFAULT_FAISS_MAP_PATH.parents

    def test_db_url_is_isolated(
        self, isolated_data_paths: pathlib.Path, tmp_path: pathlib.Path
    ):
        """Verifies that DEFAULT_DB_URL reflects the isolated DB path.

        Args:
            isolated_data_paths: Fixture providing the isolated user data root.
            tmp_path: Pytest's built-in temporary directory fixture.
        """
        # The isolated_data_paths fixture monkeypatches DEFAULT_DB_URL directly
        mocked_db_path_str = str(project_defaults.DEFAULT_DB_PATH.resolve())
        assert project_defaults.DEFAULT_DB_URL == f"sqlite:///{mocked_db_path_str}"
        assert str(tmp_path) in project_defaults.DEFAULT_DB_URL

    def test_active_toolchain_config_file_path_is_isolated(
        self, isolated_data_paths: pathlib.Path, tmp_path: pathlib.Path
    ):
        """Verifies ACTIVE_TOOLCHAIN_CONFIG_FILE_PATH is isolated.

        Args:
            isolated_data_paths: Fixture providing the isolated user data root.
            tmp_path: Pytest's built-in temporary directory fixture.
        """
        expected_config_path = (
            isolated_data_paths / project_defaults.ACTIVE_TOOLCHAIN_CONFIG_FILENAME
        )
        assert (
            project_defaults.ACTIVE_TOOLCHAIN_CONFIG_FILE_PATH == expected_config_path
        )
        assert tmp_path in project_defaults.ACTIVE_TOOLCHAIN_CONFIG_FILE_PATH.parents

    def test_non_path_constants_remain_unchanged(
        self, isolated_data_paths: pathlib.Path
    ):
        """Ensures non-path constants are not affected by isolated_data_paths.

        Args:
            isolated_data_paths: Fixture that patches path defaults.
        """
        # This test confirms that the isolated_data_paths fixture only targets
        # path variables and doesn't unintentionally alter other constants.
        assert project_defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION == "0.1.0"
        assert project_defaults.DEFAULT_EMBEDDING_MODEL_NAME == "BAAI/bge-base-en-v1.5"
        assert (
            project_defaults.R2_MANIFEST_DEFAULT_URL
            == "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"
        )
        assert project_defaults.DEFAULT_FAISS_K == 100
        assert project_defaults.DEFAULT_SEM_SIM_THRESHOLD == 0.525
