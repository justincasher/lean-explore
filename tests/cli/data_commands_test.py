"""Tests for the CLI data_commands module.

These tests verify the data toolchain management commands including fetch and clean.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lean_explore.cli.data_commands import (
    _cleanup_old_versions,
    _fetch_latest_version,
    _get_console,
    _install_toolchain,
    _write_active_version,
    app,
)

runner = CliRunner()


class TestGetConsole:
    """Tests for the _get_console helper function."""

    def test_get_console_returns_console(self):
        """Test that _get_console returns a Console instance."""
        console = _get_console()
        assert console is not None


class TestFetchLatestVersion:
    """Tests for the _fetch_latest_version function."""

    def test_fetch_latest_version_success(self):
        """Test successful latest version fetch."""
        mock_response = MagicMock()
        mock_response.text = "20260127_103630\n"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = _fetch_latest_version()
            assert result == "20260127_103630"

    def test_fetch_latest_version_strips_whitespace(self):
        """Test that version string is stripped of whitespace."""
        mock_response = MagicMock()
        mock_response.text = "  20260127_103630  \n"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = _fetch_latest_version()
            assert result == "20260127_103630"

    def test_fetch_latest_version_network_error(self):
        """Test latest version fetch with network error."""
        import requests

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(ValueError, match="Failed to fetch latest version"):
                _fetch_latest_version()

    def test_fetch_latest_version_http_error(self):
        """Test latest version fetch with HTTP error."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Failed to fetch latest version"):
                _fetch_latest_version()

    def test_fetch_latest_version_timeout(self):
        """Test latest version fetch with timeout."""
        import requests

        with patch("requests.get", side_effect=requests.exceptions.Timeout()):
            with pytest.raises(ValueError, match="Failed to fetch latest version"):
                _fetch_latest_version()


class TestWriteActiveVersion:
    """Tests for the _write_active_version function."""

    def test_write_active_version_creates_file(self):
        """Test that active version is written to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            version_file = Path(tmpdir) / "active_version"

            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
            ):
                _write_active_version("20260127_103630")
                assert version_file.exists()
                assert version_file.read_text() == "20260127_103630"

    def test_write_active_version_overwrites_existing(self):
        """Test that active version file is overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            version_file = Path(tmpdir) / "active_version"
            version_file.write_text("old_version")

            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
            ):
                _write_active_version("new_version")
                assert version_file.read_text() == "new_version"


class TestCleanupOldVersions:
    """Tests for the _cleanup_old_versions function."""

    def test_cleanup_removes_old_versions(self):
        """Test that old version directories are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            old_version = cache_dir / "old_version"
            current_version = cache_dir / "current_version"
            old_version.mkdir()
            current_version.mkdir()
            (old_version / "file.txt").touch()
            (current_version / "file.txt").touch()

            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
            ):
                _cleanup_old_versions("current_version")
                assert not old_version.exists()
                assert current_version.exists()

    def test_cleanup_handles_nonexistent_cache(self):
        """Test cleanup when cache directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", nonexistent
            ):
                # Should not raise
                _cleanup_old_versions("any_version")


class TestInstallToolchain:
    """Tests for the _install_toolchain function."""

    def test_install_toolchain_version_fetch_fails(self):
        """Test error when latest version fetch fails."""
        with patch(
            "lean_explore.cli.data_commands._fetch_latest_version",
            side_effect=ValueError("Failed to fetch"),
        ):
            with pytest.raises(ValueError, match="Failed to fetch"):
                _install_toolchain()

    def test_install_toolchain_with_explicit_version(self):
        """Test that explicit version skips latest fetch."""
        import requests as requests_module

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            # Mock the download to always fail so we can verify version is used
            with (
                patch(
                    "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
                ),
                patch(
                    "lean_explore.cli.data_commands._fetch_latest_version"
                ) as mock_fetch,
                patch("requests.get") as mock_get,
            ):
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = (
                    requests_module.exceptions.HTTPError("Download fail")
                )
                mock_get.return_value = mock_response

                with pytest.raises(ValueError):
                    _install_toolchain("explicit_version")

                # Should not call fetch latest when version is explicit
                mock_fetch.assert_not_called()

    @pytest.mark.integration
    def test_install_toolchain_downloads_all_files(self):
        """Test that all required files are downloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            downloaded_urls = []

            def mock_get(url, **kwargs):
                downloaded_urls.append(url)
                response = MagicMock()
                response.headers = {"content-length": "100"}
                response.iter_content = MagicMock(return_value=[b"data"])
                return response

            with (
                patch(
                    "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
                ),
                patch(
                    "lean_explore.cli.data_commands._fetch_latest_version",
                    return_value="test_version",
                ),
                patch("requests.get", side_effect=mock_get),
            ):
                _install_toolchain()

                # Verify key files were requested
                url_paths = [url.split("/")[-1] for url in downloaded_urls]
                assert "lean_explore.db" in url_paths
                assert "informalization_faiss.index" in url_paths
                assert "bm25_ids_map.json" in url_paths


class TestFetchCommand:
    """Tests for the fetch CLI command."""

    def test_fetch_command_help(self):
        """Test fetch command help output."""
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_fetch_command_calls_install(self):
        """Test that fetch command calls _install_toolchain."""
        with patch("lean_explore.cli.data_commands._install_toolchain") as mock_install:
            runner.invoke(app, ["fetch"])
            mock_install.assert_called_once_with(None)

    def test_fetch_command_with_version(self):
        """Test fetch command with specific version."""
        with patch("lean_explore.cli.data_commands._install_toolchain") as mock_install:
            runner.invoke(app, ["fetch", "--version", "20260127_103630"])
            mock_install.assert_called_once_with("20260127_103630")


class TestCleanCommand:
    """Tests for the clean CLI command."""

    def test_clean_command_help(self):
        """Test clean command help output."""
        result = runner.invoke(app, ["clean", "--help"])
        assert result.exit_code == 0

    def test_clean_command_no_data(self):
        """Test clean when no cache directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", nonexistent
            ):
                result = runner.invoke(app, ["clean"])
                assert "No local data" in result.output

    def test_clean_command_aborted(self):
        """Test clean command when user aborts confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            (cache_dir / "test_file").touch()

            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
            ):
                runner.invoke(app, ["clean"], input="n\n")
                assert cache_dir.exists()

    def test_clean_command_confirmed(self):
        """Test clean command when user confirms deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            (cache_dir / "test_file").touch()

            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
            ):
                result = runner.invoke(app, ["clean"], input="y\n")
                assert not cache_dir.exists()
                assert "cleared" in result.output.lower()

    def test_clean_command_removes_version_file(self):
        """Test that clean also removes the active_version file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            version_file = Path(tmpdir) / "active_version"
            version_file.write_text("some_version")

            with patch(
                "lean_explore.cli.data_commands.Config.CACHE_DIRECTORY", cache_dir
            ):
                result = runner.invoke(app, ["clean"], input="y\n")
                assert not version_file.exists()
                assert "cleared" in result.output.lower()


class TestDataApp:
    """Tests for the data app structure."""

    def test_data_app_no_args_shows_help(self):
        """Test that data app with no args shows help."""
        result = runner.invoke(app)
        # Typer exits with code 2 when no_args_is_help=True and no args provided
        assert result.exit_code in (0, 2)
        # Should show help with available commands
        assert "fetch" in result.output.lower() or "clean" in result.output.lower()
