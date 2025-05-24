# tests/cli/test_data_commands.py

"""Tests for CLI data management commands in `lean_explore.cli.data_commands`.

This module verifies the functionality of fetching, resolving, downloading,
verifying, and decompressing data toolchains for local use. It utilizes
mocks for external HTTP requests and isolates file operations to temporary
directories.
"""

import gzip
import hashlib
import json
import pathlib
from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests  # For requests.exceptions

# For spec in mock, requests.Response.raw is often a
# urllib3.response.HTTPResponse
import urllib3
from typer.testing import CliRunner

from lean_explore import defaults
from lean_explore.cli import data_commands as data_commands_module

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# --- Tests for Helper Functions ---
class TestDataCommandHelpers:
    """Tests for internal helper functions in data_commands.py."""

    @pytest.fixture  # type: ignore[misc]
    def mock_console_print(self, mocker: "MockerFixture") -> MagicMock:
        """Mocks the console.print function in the data_commands module.

        Args:
            mocker: Pytest-mock's mocker fixture.

        Returns:
            MagicMock: The mock object for console.print.
        """
        return mocker.patch.object(data_commands_module.console, "print")

    def test_fetch_remote_json_success(
        self, mocker: "MockerFixture", mock_console_print: MagicMock
    ) -> None:
        """Verifies successful fetching and parsing of remote JSON.

        Args:
            mocker: Pytest-mock's mocker fixture.
            mock_console_print: Mock for console.print.
        """
        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.json.return_value = {"key": "value"}
        mocker.patch.object(
            data_commands_module.requests, "get", return_value=mock_response
        )

        result = data_commands_module._fetch_remote_json("http://fakeurl.com/data.json")

        assert result == {"key": "value"}
        data_commands_module.requests.get.assert_called_once_with(
            "http://fakeurl.com/data.json", timeout=10
        )
        mock_response.raise_for_status.assert_called_once()
        mock_console_print.assert_not_called()

    @pytest.mark.parametrize(  # type: ignore[misc]
        "exception_to_raise, log_substring",
        [
            (
                requests.exceptions.RequestException("Network Error"),
                "Error fetching manifest",
            ),
            (requests.exceptions.HTTPError("Server Error"), "Error fetching manifest"),
            (json.JSONDecodeError("Decode Error", "doc", 0), "Error parsing JSON"),
        ],
    )
    def test_fetch_remote_json_failures(
        self,
        mocker: "MockerFixture",
        mock_console_print: MagicMock,
        exception_to_raise: Exception,
        log_substring: str,
    ) -> None:
        """Tests various failure scenarios for _fetch_remote_json.

        Args:
            mocker: Pytest-mock's mocker fixture.
            mock_console_print: Mock for console.print.
            exception_to_raise: The exception to be simulated.
            log_substring: The expected substring in the error print.
        """
        mock_requests_get = mocker.patch.object(data_commands_module.requests, "get")

        if isinstance(
            exception_to_raise,
            (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        ):
            mock_requests_get.side_effect = exception_to_raise
            if isinstance(exception_to_raise, requests.exceptions.HTTPError):
                mock_http_error_response = mocker.MagicMock(spec=requests.Response)
                mock_http_error_response.raise_for_status.side_effect = (
                    exception_to_raise
                )
                mock_requests_get.return_value = mock_http_error_response
                mock_requests_get.side_effect = None

        elif isinstance(exception_to_raise, json.JSONDecodeError):
            mock_json_error_response = mocker.MagicMock(spec=requests.Response)
            mock_json_error_response.json.side_effect = exception_to_raise
            mock_requests_get.return_value = mock_json_error_response

        result = data_commands_module._fetch_remote_json("http://fakeurl.com/data.json")
        assert result is None
        mock_console_print.assert_called()
        assert any(
            log_substring in str(c_arg)
            for call_args_tuple in mock_console_print.call_args_list
            for c_arg_list in call_args_tuple
            for c_arg in c_arg_list
            if isinstance(c_arg, str)
        )

    def test_resolve_toolchain_version_info(
        self, mock_console_print: MagicMock
    ) -> None:
        """Tests resolving toolchain version identifiers from manifest data.

        Args:
            mock_console_print: Mock for console.print.
        """
        manifest_data: Dict[str, Any] = {
            "default_toolchain": "0.1.0",
            "toolchains": {
                "0.1.0": {
                    "description": "Version 0.1.0",
                    "assets_base_path_r2": "assets/0.1.0/",
                },
                "0.2.0": {
                    "description": "0.2.0",
                    "assets_base_path_r2": "assets/0.2.0/",
                },
            },
        }
        info_stable = data_commands_module._resolve_toolchain_version_info(
            manifest_data, "stable"
        )
        assert info_stable is not None
        assert info_stable["description"] == "Version 0.1.0"
        assert info_stable["_resolved_key"] == "0.1.0"
        assert any(
            "Note: 'stable' currently points to version '0.1.0'" in str(c[0][0])
            for c in mock_console_print.call_args_list
        )
        mock_console_print.reset_mock()

        info_specific = data_commands_module._resolve_toolchain_version_info(
            manifest_data, "0.2.0"
        )
        assert info_specific is not None
        assert info_specific["description"] == "0.2.0"
        assert info_specific["_resolved_key"] == "0.2.0"
        mock_console_print.assert_not_called()

        info_not_found = data_commands_module._resolve_toolchain_version_info(
            manifest_data, "0.3.0"
        )
        assert info_not_found is None
        assert any(
            "Error: Version '0.3.0' (resolved from '0.3.0') not found" in str(c[0][0])
            for c in mock_console_print.call_args_list
        )
        mock_console_print.reset_mock()

        del manifest_data["default_toolchain"]
        info_no_stable_target = data_commands_module._resolve_toolchain_version_info(
            manifest_data, "stable"
        )
        assert info_no_stable_target is None
        assert any(
            "Error: Manifest does not define a 'default_toolchain' for 'stable'"
            in str(c[0][0])
            for c in mock_console_print.call_args_list
        )
        mock_console_print.reset_mock()

        info_no_toolchains = data_commands_module._resolve_toolchain_version_info(
            {"other_key": "value"}, "stable"
        )  # type: ignore[arg-type]
        assert info_no_toolchains is None
        assert any(
            "Error: Manifest is missing 'toolchains' dictionary" in str(c[0][0])
            for c in mock_console_print.call_args_list
        )

    def test_download_file_with_progress_success(
        self,
        tmp_path: pathlib.Path,
        mocker: "MockerFixture",
        mock_console_print: MagicMock,
    ) -> None:
        """Verifies successful file download and progress simulation.

        Args:
            tmp_path: Pytest fixture for a temporary directory path.
            mocker: Pytest-mock's mocker fixture.
            mock_console_print: Mock for console.print.
        """
        dest_path = tmp_path / "downloaded_file.dat"
        file_content = b"test content" * 1000
        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.headers = {"content-length": str(len(file_content))}

        mock_response.raw = mocker.MagicMock(spec=urllib3.response.HTTPResponse)

        default_chunk_size = 8192

        def stream_side_effect(
            decode_content: bool = False, amt: int = default_chunk_size
        ) -> iter:  # type: ignore[type-arg]
            current_chunk_size = amt if amt > 0 else 1
            chunks = [
                file_content[i : i + current_chunk_size]
                for i in range(0, len(file_content), current_chunk_size)
            ]
            return iter(chunks)

        mock_response.raw.stream = mocker.MagicMock(side_effect=stream_side_effect)

        mocker.patch.object(
            data_commands_module.requests, "get", return_value=mock_response
        )

        mock_progress_class = mocker.patch.object(data_commands_module, "Progress")
        mock_progress_instance = mock_progress_class.return_value.__enter__.return_value
        mock_progress_instance.add_task = mocker.MagicMock()
        mock_progress_instance.update = mocker.MagicMock()

        result = data_commands_module._download_file_with_progress(
            "http://fake.com/file",
            dest_path,
            "Test File",
            expected_size_bytes=len(file_content),
        )

        assert result is True, "Expected _download_file_with_progress to return True"
        assert dest_path.exists(), f"Expected destination file {dest_path} to exist"
        assert dest_path.read_bytes() == file_content, "File content mismatch"

        data_commands_module.requests.get.assert_called_once_with(
            "http://fake.com/file", stream=True, timeout=30
        )
        mock_response.raw.stream.assert_called_once_with(
            decode_content=False, amt=default_chunk_size
        )

        mock_progress_class.assert_called_once()
        mock_progress_instance.add_task.assert_called()
        if file_content:
            expected_update_calls = (
                len(file_content) + default_chunk_size - 1
            ) // default_chunk_size
            assert mock_progress_instance.update.call_count == expected_update_calls
        else:
            mock_progress_instance.update.assert_not_called()

    def test_verify_sha256_checksum(
        self, tmp_path: pathlib.Path, mock_console_print: MagicMock
    ) -> None:
        """Tests SHA256 checksum verification.

        Args:
            tmp_path: Pytest fixture for a temporary directory path.
            mock_console_print: Mock for console.print.
        """
        test_file = tmp_path / "checksum_test.txt"
        content = b"Hello, checksum!"
        test_file.write_bytes(content)

        hasher = hashlib.sha256()
        hasher.update(content)
        correct_checksum = hasher.hexdigest()
        incorrect_checksum = "incorrect" + correct_checksum[len("incorrect") :]

        assert (
            data_commands_module._verify_sha256_checksum(test_file, correct_checksum)
            is True
        )
        assert any(
            "Checksum verified" in str(c[0][0])
            for c in mock_console_print.call_args_list
            if isinstance(c[0][0], str)
        )
        mock_console_print.reset_mock()

        assert (
            data_commands_module._verify_sha256_checksum(test_file, incorrect_checksum)
            is False
        )
        assert any(
            "Checksum mismatch" in str(c[0][0])
            for c in mock_console_print.call_args_list
            if isinstance(c[0][0], str)
        )
        mock_console_print.reset_mock()

        with patch("builtins.open", side_effect=OSError("Read error")):
            assert (
                data_commands_module._verify_sha256_checksum(
                    test_file, correct_checksum
                )
                is False
            )
        assert any(
            "Error reading file" in str(c[0][0])
            for c in mock_console_print.call_args_list
            if isinstance(c[0][0], str)
        )

    def test_decompress_gzipped_file(
        self, tmp_path: pathlib.Path, mock_console_print: MagicMock
    ) -> None:
        """Tests gzipped file decompression.

        Args:
            tmp_path: Pytest fixture for a temporary directory path.
            mock_console_print: Mock for console.print.
        """
        original_content = b"This is original content for gzip."
        gzipped_file_path = tmp_path / "test_file.txt.gz"
        output_file_path = tmp_path / "test_file.txt"

        with gzip.open(gzipped_file_path, "wb") as f_gz:
            f_gz.write(original_content)

        assert (
            data_commands_module._decompress_gzipped_file(
                gzipped_file_path, output_file_path
            )
            is True
        )
        assert output_file_path.exists()
        assert output_file_path.read_bytes() == original_content
        assert any(
            "Decompressed" in str(c[0][0])
            for c in mock_console_print.call_args_list
            if isinstance(c[0][0], str)
        )
        mock_console_print.reset_mock()

        non_gzipped_file = tmp_path / "not_gzipped.txt.gz"
        non_gzipped_file.write_text("not gzipped")
        output_file_path_for_fail = tmp_path / "failed_output.txt"
        assert (
            data_commands_module._decompress_gzipped_file(
                non_gzipped_file, output_file_path_for_fail
            )
            is False
        )
        assert any(
            "Error decompressing" in str(c[0][0])
            for c in mock_console_print.call_args_list
            if isinstance(c[0][0], str)
        )


# --- Tests for CLI Command 'fetch' ---
class TestDataFetchCommand:
    """Tests for the `Workspace` CLI command in data_commands.py."""

    runner = CliRunner()

    @pytest.fixture  # type: ignore[misc]
    def mock_manifest_content(self) -> Dict[str, Any]:
        """Provides sample manifest content."""
        return {
            "default_toolchain": "0.1.0",
            "toolchains": {
                "0.1.0": {
                    "description": "Version 0.1.0",
                    "assets_base_path_r2": "assets/v0.1.0/",
                    "files": [
                        {
                            "local_name": "lean_explore_data.db",
                            "remote_name": "lean_explore_data.db.gz",
                            "sha256": "correct_db_checksum_gzipped",
                            "size_bytes_compressed": 1000,
                        },
                        {
                            "local_name": "main_faiss.index",
                            "remote_name": "main_faiss.index.gz",
                            "sha256": "correct_faiss_checksum_gzipped",
                            "size_bytes_compressed": 2000,
                        },
                    ],
                }
            },
        }

    @pytest.fixture  # type: ignore[misc]
    def mock_assets(
        self, mocker: "MockerFixture", mock_manifest_content: Dict[str, Any]
    ) -> MagicMock:
        """Mocks HTTP calls for manifest and assets, and file operations.

        Args:
            mocker: Pytest-mock's mocker fixture.
            mock_manifest_content: Sample manifest data.

        Returns:
            MagicMock: The mock object for requests.get.
        """
        mock_get = mocker.patch.object(data_commands_module.requests, "get")

        mocker.patch.object(
            data_commands_module, "_download_file_with_progress", return_value=True
        )
        mocker.patch.object(
            data_commands_module, "_verify_sha256_checksum", return_value=True
        )
        mocker.patch.object(
            data_commands_module, "_decompress_gzipped_file", return_value=True
        )
        mocker.patch.object(data_commands_module.console, "print")

        def get_side_effect(url: str, **kwargs: Any) -> MagicMock:
            mock_resp = MagicMock(spec=requests.Response)
            if url == defaults.R2_MANIFEST_DEFAULT_URL:
                mock_resp.json.return_value = mock_manifest_content
                mock_resp.raise_for_status = MagicMock()
            else:
                mock_resp.headers = {"content-length": "100"}
                mock_raw = MagicMock(spec=urllib3.response.HTTPResponse)
                mock_raw.stream.return_value = iter([b"gzipped_content_chunk"])
                mock_resp.raw = mock_raw
                mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_get.side_effect = get_side_effect
        return mock_get

    def test_fetch_successful_stable_version(
        self,
        isolated_data_paths: pathlib.Path,
        mock_assets: MagicMock,
        mock_manifest_content: Dict[str, Any],
    ) -> None:
        """Tests successful fetching of the 'stable' toolchain version.

        The 'catch_exceptions=False' argument to invoke is crucial for debugging;
        it ensures that if Typer/Click encounters an issue parsing arguments
        or dispatching the command (leading to exit code 2), the underlying
        Python exception (if any, beyond SystemExit) will propagate and be fully
        displayed by pytest, aiding in diagnosing issues like the
        "Got unexpected extra argument" error.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_assets: Fixture that mocks HTTP calls and file operations.
            mock_manifest_content: Sample manifest data.
        """
        result = self.runner.invoke(
            data_commands_module.app, ["fetch", "stable"], catch_exceptions=False
        )

        if result.exit_code != 0:  # pragma: no cover
            print("\nDebug Output for test_fetch_successful_stable_version:")
            print(f"Exit Code: {result.exit_code}")
            print(f"Output:\n{result.output}")
            if (
                result.exception
            ):  # If catch_exceptions=False, this might be None if SystemExit was raised
                print(f"Exception Type: {type(result.exception)}")
                print(f"Exception Value: {result.exception}")

        assert result.exit_code == 0, (
            f"Command failed unexpectedly. Typer error indicated: {result.output}"
        )

        num_files = len(mock_manifest_content["toolchains"]["0.1.0"]["files"])
        assert data_commands_module._download_file_with_progress.call_count == num_files
        assert data_commands_module._verify_sha256_checksum.call_count == num_files
        assert data_commands_module._decompress_gzipped_file.call_count == num_files

        version_dir = isolated_data_paths / "toolchains" / "0.1.0"
        for file_entry in mock_manifest_content["toolchains"]["0.1.0"]["files"]:
            expected_output_path = version_dir / file_entry["local_name"]
            expected_temp_download_path = version_dir / file_entry["remote_name"]

            decompress_call_args_list = (
                data_commands_module._decompress_gzipped_file.call_args_list
            )
            assert any(
                c_args[0][1] == expected_output_path
                for c_args in decompress_call_args_list
            ), f"Decompress not called with expected output path {expected_output_path}"

            download_call_args_list = (
                data_commands_module._download_file_with_progress.call_args_list
            )
            assert any(
                c_args[0][1] == expected_temp_download_path
                for c_args in download_call_args_list
            ), (
                "Download not called with expected temp path "
                f"{expected_temp_download_path}"
            )

    def test_fetch_manifest_request_exception(
        self, isolated_data_paths: pathlib.Path, mocker: "MockerFixture"
    ) -> None:
        """Tests fetch command failure when manifest fails with RequestException.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(
            data_commands_module.requests,
            "get",
            side_effect=requests.exceptions.RequestException("Network issue"),
        )
        mock_console_print = mocker.patch.object(data_commands_module.console, "print")

        result = self.runner.invoke(
            data_commands_module.app, ["fetch", "stable"], catch_exceptions=False
        )
        if result.exit_code != 1:  # pragma: no cover
            print(
                f"\nDebug Output for test_fetch_manifest_request_exception:\n"
                f"Exit Code: {result.exit_code}\nOutput:\n{result.output}"
            )
            if result.exception:
                print(
                    f"Exception:\n{result.exception}"
                )  # Should be SystemExit(1) if handled correctly
        assert result.exit_code == 1, (
            "Command did not exit with 1 as expected. "
            f"Typer error indicated: {result.output}"
        )
        assert any(
            "Failed to fetch or parse the manifest" in str(arg)
            for call_args_tuple in mock_console_print.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )

    def test_fetch_version_not_in_manifest(
        self,
        isolated_data_paths: pathlib.Path,
        mock_assets: MagicMock,
        mock_manifest_content: Dict[str, Any],
        mocker: "MockerFixture",  # pylint: disable=unused-argument
    ) -> None:
        """Tests fetch command failure when requested version is not in manifest.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_assets: Fixture that mocks HTTP calls.
            mock_manifest_content: Sample manifest data.
            mocker: Pytest-mock's mocker fixture.
        """
        mock_console_print_instance = data_commands_module.console.print

        result = self.runner.invoke(
            data_commands_module.app,
            ["fetch", "0.non.existent"],
            catch_exceptions=False,
        )
        if result.exit_code != 1:  # pragma: no cover
            print(
                f"\nDebug Output for test_fetch_version_not_in_manifest:\n"
                f"Exit Code: {result.exit_code}\nOutput:\n{result.output}"
            )
            if result.exception:
                print(f"Exception:\n{result.exception}")
        assert result.exit_code == 1, (
            "Command did not exit with 1 as expected. "
            f"Typer error indicated: {result.output}"
        )
        assert any(
            "Error: Version '0.non.existent' (resolved from '0.non.existent') not found"
            in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )

    def test_fetch_asset_download_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_assets: MagicMock,
        mocker: "MockerFixture",  # pylint: disable=unused-argument
    ) -> None:
        """Tests fetch command failure when an asset download fails.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_assets: Fixture that mocks HTTP calls and some helpers.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(
            data_commands_module, "_download_file_with_progress", return_value=False
        )
        mock_console_print_instance = data_commands_module.console.print

        result = self.runner.invoke(
            data_commands_module.app, ["fetch", "stable"], catch_exceptions=False
        )
        if result.exit_code != 1:  # pragma: no cover
            print(
                f"\nDebug Output for test_fetch_asset_download_fails:\n"
                f"Exit Code: {result.exit_code}\nOutput:\n{result.output}"
            )
            if result.exception:
                print(f"Exception:\n{result.exception}")
        assert result.exit_code == 1, (
            "Command did not exit with 1 as expected. "
            f"Typer error indicated: {result.output}"
        )
        assert any(
            "Failed to download" in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )
        assert any(
            "fetch process completed with some errors" in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )

    def test_fetch_checksum_mismatch(
        self,
        isolated_data_paths: pathlib.Path,
        mock_assets: MagicMock,
        mocker: "MockerFixture",  # pylint: disable=unused-argument
    ) -> None:
        """Tests fetch command failure due to checksum mismatch.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_assets: Fixture that mocks HTTP calls and some helpers.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(
            data_commands_module, "_verify_sha256_checksum", return_value=False
        )
        mock_console_print_instance = data_commands_module.console.print

        result = self.runner.invoke(
            data_commands_module.app, ["fetch", "stable"], catch_exceptions=False
        )
        if result.exit_code != 1:  # pragma: no cover
            print(
                f"\nDebug Output for test_fetch_checksum_mismatch:\n"
                f"Exit Code: {result.exit_code}\nOutput:\n{result.output}"
            )
            if result.exception:
                print(f"Exception:\n{result.exception}")
        assert result.exit_code == 1, (
            "Command did not exit with 1 as expected. "
            f"Typer error indicated: {result.output}"
        )
        assert any(
            "Checksum verification failed" in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )
        assert any(
            "fetch process completed with some errors" in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )

    def test_fetch_decompression_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_assets: MagicMock,
        mocker: "MockerFixture",  # pylint: disable=unused-argument
    ) -> None:
        """Tests fetch command failure due to decompression error.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_assets: Fixture that mocks HTTP calls and some helpers.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(
            data_commands_module, "_decompress_gzipped_file", return_value=False
        )
        mock_console_print_instance = data_commands_module.console.print

        result = self.runner.invoke(
            data_commands_module.app, ["fetch", "stable"], catch_exceptions=False
        )
        if result.exit_code != 1:  # pragma: no cover
            print(
                f"\nDebug Output for test_fetch_decompression_fails:\n"
                f"Exit Code: {result.exit_code}\nOutput:\n{result.output}"
            )
            if result.exception:
                print(f"Exception:\n{result.exception}")
        assert result.exit_code == 1, (
            "Command did not exit with 1 as expected. "
            f"Typer error indicated: {result.output}"
        )
        assert any(
            "Failed to decompress" in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )
        assert any(
            "fetch process completed with some errors" in str(arg)
            for call_args_tuple in mock_console_print_instance.call_args_list
            for arg_list in call_args_tuple
            for arg in arg_list
            if isinstance(arg, str)
        )
