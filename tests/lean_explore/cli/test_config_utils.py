# tests/lean_explore/cli/test_config_utils.py

"""Tests for CLI user configuration utilities.

This module contains tests for the functions in
`lean_explore.cli.config_utils`. It verifies that user-specific settings,
such as API keys, are saved, loaded, and deleted correctly from a TOML
configuration file. Tests use fixtures to ensure operations are performed
within an isolated, temporary directory, preventing interference with the
actual user environment.
"""

import logging
import os
import pathlib
import toml
import pytest
import builtins

from lean_explore.cli import config_utils as cu

def _get_test_config_file(isolated_config_app_dir: pathlib.Path) -> pathlib.Path:
    """Constructs the full path to the config.toml file within an isolated directory.

    Args:
        isolated_config_app_dir: The path to the isolated application-specific
                                 configuration directory (e.g., 'leanexplore')
                                 yielded by a fixture.

    Returns:
        pathlib.Path: The full path to the 'config.toml' file within the
                      provided isolated directory.
    """
    return isolated_config_app_dir / cu._CONFIG_FILENAME


class TestEnsureConfigDirExists:
    """Tests for the internal `_ensure_config_dir_exists` helper function."""

    def test_directory_creation_and_idempotency(self, isolated_config_dir: pathlib.Path):
        """Verifies directory creation and ensures calling again does not error.

        The `isolated_config_dir` fixture, which monkeypatches
        `get_config_file_path`, inherently involves the creation of the
        necessary directory structure. This test confirms that
        `_ensure_config_dir_exists` executes without error in this context,
        effectively testing its idempotency if the directory already exists.

        Args:
            isolated_config_dir: A pytest fixture that provides an isolated
                                 temporary directory for configuration files and
                                 patches `get_config_file_path` accordingly.
        """
        try:
            cu._ensure_config_dir_exists()
        except OSError:
            pytest.fail(
                "cu._ensure_config_dir_exists() raised OSError unexpectedly "
                "when the directory is expected to exist or be creatable."
            )

    def test_oserror_on_mkdir(
        self,
        isolated_config_dir: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture
    ):
        """Tests that an OSError during directory creation is raised and logged.

        This test simulates a scenario where creating the configuration
        directory fails due to an OSError.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            monkeypatch: Pytest fixture for modifying object attributes.
            caplog: Pytest fixture to capture log output.
        """
        def mock_mkdir_raises_oserror(self_path, parents=False, exist_ok=False):
            """Simulates an OSError when pathlib.Path.mkdir is called."""
            raise OSError("Test-induced OSError for mkdir")

        monkeypatch.setattr(pathlib.Path, "mkdir", mock_mkdir_raises_oserror)

        with pytest.raises(OSError, match="Test-induced OSError for mkdir"):
            cu._ensure_config_dir_exists()

        assert "Failed to create configuration directory" in caplog.text
        assert "Test-induced OSError for mkdir" in caplog.text


class TestLoadConfigData:
    """Tests for the internal `_load_config_data` helper function."""

    def test_load_non_existent_file(self, tmp_path: pathlib.Path):
        """Ensures loading from a non-existent file returns an empty dictionary.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
        """
        non_existent_file = tmp_path / "non_existent.toml"
        assert cu._load_config_data(non_existent_file) == {}

    def test_load_valid_toml_file(self, tmp_path: pathlib.Path):
        """Verifies loading data from a correctly formatted TOML file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
        """
        valid_toml_file = tmp_path / "valid.toml"
        expected_data = {"section": {"key": "value"}}
        valid_toml_file.write_text(toml.dumps(expected_data), encoding="utf-8")

        assert cu._load_config_data(valid_toml_file) == expected_data

    def test_load_corrupted_toml_file(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture
    ):
        """Checks that loading a corrupted TOML file returns an empty dict and logs a warning.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            caplog: Pytest fixture to capture log output.
        """
        corrupted_toml_file = tmp_path / "corrupted.toml"
        corrupted_toml_file.write_text("this is not valid toml content {=}", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert cu._load_config_data(corrupted_toml_file) == {}
        assert f"Configuration file {corrupted_toml_file} is corrupted." in caplog.text

    def test_load_exception_during_open(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture
    ):
        """Tests error handling when an exception occurs during file opening.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            monkeypatch: Pytest fixture for modifying object attributes.
            caplog: Pytest fixture to capture log output.
        """
        file_path = tmp_path / "problematic.toml"
        file_path.touch()

        def mock_open_raises_exception(*args, **kwargs):
            """Simulates an IOError during file open."""
            raise IOError("Test-induced file read error")

        monkeypatch.setattr("builtins.open", mock_open_raises_exception)

        with caplog.at_level(logging.ERROR):
            assert cu._load_config_data(file_path) == {}
        assert f"Error reading existing config file {file_path}" in caplog.text
        assert "Test-induced file read error" in caplog.text


class TestSaveConfigData:
    """Tests for the internal `_save_config_data` helper function."""

    def test_successful_save_and_chmod(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch
    ):
        """Verifies successful data saving and that `os.chmod` is called correctly.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            monkeypatch: Pytest fixture for modifying object attributes.
        """
        save_file = tmp_path / "save_test.toml"
        data_to_save = {"user": {"name": "tester"}}
        chmod_called_with = None

        def mock_chmod_capture(path_arg, mode_arg):
            """Captures arguments passed to os.chmod."""
            nonlocal chmod_called_with
            chmod_called_with = (path_arg, mode_arg)

        monkeypatch.setattr("os.chmod", mock_chmod_capture)

        assert cu._save_config_data(save_file, data_to_save) is True
        assert save_file.exists()

        assert toml.loads(save_file.read_text(encoding="utf-8")) == data_to_save
        assert chmod_called_with == (save_file, 0o600)

    def test_oserror_during_open_for_write(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture
    ):
        """Tests handling of OSError during file open for writing.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            monkeypatch: Pytest fixture for modifying object attributes.
            caplog: Pytest fixture to capture log output.
        """
        save_file = tmp_path / "oserror_save.toml"
        data_to_save = {"error": "test"}
        original_open = builtins.open

        def mock_open_raises_oserror_for_write(file, mode="r", *args, **kwargs):
            """Simulates OSError if file is opened in write mode."""
            if 'w' in mode:
                raise OSError("Test-induced OSError on write")
            return original_open(file, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open_raises_oserror_for_write)

        with caplog.at_level(logging.ERROR):
            assert cu._save_config_data(save_file, data_to_save) is False
        assert f"OS error saving configuration to {save_file}" in caplog.text
        assert "Test-induced OSError on write" in caplog.text

    def test_oserror_during_chmod(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture
    ):
        """Tests handling of OSError during the `os.chmod` call.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            monkeypatch: Pytest fixture for modifying object attributes.
            caplog: Pytest fixture to capture log output.
        """
        save_file = tmp_path / "chmod_error.toml"
        data_to_save = {"chmod": "error"}

        def mock_chmod_raises_oserror(path, mode):
            """Simulates OSError when os.chmod is called."""
            raise OSError("Test-induced chmod OSError")

        monkeypatch.setattr("os.chmod", mock_chmod_raises_oserror)

        with caplog.at_level(logging.ERROR):
            assert cu._save_config_data(save_file, data_to_save) is False
        assert f"OS error saving configuration to {save_file}" in caplog.text
        assert "Test-induced chmod OSError" in caplog.text

    def test_exception_during_toml_dump(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture
    ):
        """Tests error handling when `toml.dump` raises an exception.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            monkeypatch: Pytest fixture for modifying object attributes.
            caplog: Pytest fixture to capture log output.
        """
        save_file = tmp_path / "dump_error.toml"
        data_to_save = {"dump": "error"}

        def mock_toml_dump_raises_exception(*args, **kwargs):
            """Simulates a TypeError during TOML dumping."""
            raise TypeError("Test-induced toml.dump error")

        monkeypatch.setattr("toml.dump", mock_toml_dump_raises_exception)

        with caplog.at_level(logging.ERROR):
            assert cu._save_config_data(save_file, data_to_save) is False
        assert f"Unexpected error saving configuration to {save_file}" in caplog.text
        assert "Test-induced toml.dump error" in caplog.text


@pytest.mark.parametrize(
    "save_func, load_func, delete_func, section_name, key_name, key_type_name",
    [
        (
            cu.save_api_key,
            cu.load_api_key,
            cu.delete_api_key,
            cu._LEAN_EXPLORE_API_SECTION_NAME,
            cu._LEAN_EXPLORE_API_KEY_NAME,
            "Lean Explore"
        ),
        (
            cu.save_openai_api_key,
            cu.load_openai_api_key,
            cu.delete_openai_api_key,
            cu._OPENAI_API_SECTION_NAME,
            cu._OPENAI_API_KEY_NAME,
            "OpenAI"
        ),
    ]
)
class TestApiKeyManagement:
    """Parametrized tests for Lean Explore and OpenAI API key management functions.

    These tests cover saving, loading, updating, and deleting API keys,
    utilizing the `isolated_config_dir` fixture for sandboxed file operations.
    """

    def test_save_and_load_new_key(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Verifies saving a new API key and subsequently loading it.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function to be tested.
            load_func: The load API key function to be tested.
            delete_func: The delete API key function (unused in this test).
            section_name: The TOML section name for this API key type.
            key_name: The TOML key name for this API key.
            key_type_name: A descriptive name for the API key type.
        """
        api_key_value = f"test_{key_type_name.lower().replace(' ', '_')}_key_12345"
        config_file = _get_test_config_file(isolated_config_dir)

        assert save_func(api_key_value) is True
        assert config_file.exists()

        loaded_key = load_func()
        assert loaded_key == api_key_value

        assert toml.loads(config_file.read_text(encoding="utf-8"))[section_name][key_name] == api_key_value

    def test_update_existing_key(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Ensures that saving a key again updates its value.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function.
            load_func: The load API key function.
            delete_func: The delete API key function (unused).
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        initial_key = f"initial_{key_type_name.lower()}_key"
        updated_key = f"updated_{key_type_name.lower()}_key_67890"

        assert save_func(initial_key) is True
        assert save_func(updated_key) is True

        assert load_func() == updated_key

    def test_save_invalid_key(
        self,
        isolated_config_dir: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Checks that attempting to save None or an empty string as a key fails and logs an error.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            caplog: Pytest fixture to capture log output.
            save_func: The save API key function.
            load_func: The load API key function.
            delete_func: The delete API key function (unused).
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        with caplog.at_level(logging.ERROR):
            assert save_func(None) is False
            assert f"Attempted to save an invalid or empty {key_type_name} API key." in caplog.text
            caplog.clear()
            assert save_func("") is False
            assert f"Attempted to save an invalid or empty {key_type_name} API key." in caplog.text

        assert load_func() is None

    def test_load_key_file_not_exists(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Tests loading a key when the configuration file itself does not exist.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function (unused).
            load_func: The load API key function.
            delete_func: The delete API key function (unused).
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        config_file = _get_test_config_file(isolated_config_dir)
        if config_file.exists():
            config_file.unlink()

        assert load_func() is None

    def test_load_key_section_or_key_not_exists_in_file(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Verifies loading when the specific section or key is missing from the config file.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function (used to save an unrelated key).
            load_func: The load API key function.
            delete_func: The delete API key function (unused).
            section_name: The TOML section name for the target key.
            key_name: The TOML key name for the target key.
            key_type_name: A descriptive name for the API key type.
        """
        config_file = _get_test_config_file(isolated_config_dir)

        # Save an unrelated key to ensure the file exists but doesn't contain the target key/section.
        if section_name == cu._LEAN_EXPLORE_API_SECTION_NAME:
            cu.save_openai_api_key("other_unrelated_key_data")
        else:
            cu.save_api_key("other_unrelated_key_data")

        assert config_file.exists()
        assert load_func() is None

        # Test with an empty relevant section
        config_file.write_text(toml.dumps({section_name: {}}), encoding="utf-8")
        assert load_func() is None

    def test_load_key_invalid_type_in_file(
        self,
        isolated_config_dir: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Checks loading when a key exists but has an incorrect data type (e.g., not a string).

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            caplog: Pytest fixture to capture log output.
            save_func: The save API key function (unused).
            load_func: The load API key function.
            delete_func: The delete API key function (unused).
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        config_file = _get_test_config_file(isolated_config_dir)
        config_file.write_text(toml.dumps({section_name: {key_name: 12345}}), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert load_func() is None
        assert f"{key_type_name} API key found in {config_file} but is not a valid string." in caplog.text

    def test_delete_existing_key_and_empty_section(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Verifies deleting an existing API key also removes its section if it becomes empty.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function.
            load_func: The load API key function.
            delete_func: The delete API key function.
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        api_key_value = f"key_to_delete_{key_type_name.lower()}"
        assert save_func(api_key_value) is True
        assert load_func() == api_key_value

        assert delete_func() is True
        assert load_func() is None

        config_file = _get_test_config_file(isolated_config_dir)
        if config_file.exists(): # File might not exist if all sections are removed
            data = toml.loads(config_file.read_text(encoding="utf-8"))
            assert section_name not in data

    def test_delete_non_existent_key_succeeds(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Ensures deleting a non-existent key is a successful no-op.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function (unused).
            load_func: The load API key function (unused).
            delete_func: The delete API key function.
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        config_file = _get_test_config_file(isolated_config_dir)
        config_file.write_text("[other_section]\nother_key = \"value\"\n", encoding="utf-8")

        assert delete_func() is True

    def test_delete_key_file_not_exists_succeeds(
        self,
        isolated_config_dir: pathlib.Path,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Checks that deleting a key when the config file doesn't exist is a successful no-op.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            save_func: The save API key function (unused).
            load_func: The load API key function (unused).
            delete_func: The delete API key function.
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        config_file = _get_test_config_file(isolated_config_dir)
        if config_file.exists():
            config_file.unlink()

        assert delete_func() is True

    def test_save_failure_after_key_deletion_in_memory(
        self,
        isolated_config_dir: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        save_func, load_func, delete_func,
        section_name, key_name, key_type_name
    ):
        """Tests behavior when saving the config file fails after a key is deleted from the in-memory representation.

        Args:
            isolated_config_dir: Fixture providing an isolated config directory.
            monkeypatch: Pytest fixture for modifying object attributes.
            save_func: The save API key function.
            load_func: The load API key function.
            delete_func: The delete API key function.
            section_name: The TOML section name.
            key_name: The TOML key name.
            key_type_name: A descriptive name for the API key type.
        """
        api_key_value = f"test_delete_save_fail_{key_type_name.lower()}"
        save_func(api_key_value)
        assert load_func() == api_key_value

        monkeypatch.setattr("lean_explore.cli.config_utils._save_config_data", lambda path, data: False)

        assert delete_func() is False
        # The key should ideally still be retrievable if the save failed,
        # meaning the file on disk was not modified.
        assert load_func() == api_key_value


class TestOriginalGetConfigFilePath:
    """Tests for the original `get_config_file_path` without fixture-based monkeypatching.

    These tests verify the path construction logic by directly mocking
    `os.path.expanduser`.
    """

    def test_path_structure_with_mocked_home(self, monkeypatch: pytest.MonkeyPatch):
        """Verifies the constructed config file path relative to a mocked home directory.

        Args:
            monkeypatch: Pytest fixture for modifying object attributes.
        """
        mock_home = "/mock/home/user"
        monkeypatch.setattr("os.path.expanduser", lambda path_str: mock_home if path_str == "~" else path_str)

        expected_path = (
            pathlib.Path(mock_home) /
            ".config" /
            cu._APP_CONFIG_DIR_NAME /
            cu._CONFIG_FILENAME
        )
        assert cu.get_config_file_path() == expected_path