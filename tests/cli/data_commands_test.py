"""Tests for the CLI data_commands module.

These tests verify the data toolchain management commands including fetch and clean.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lean_explore.cli.data_commands import (
    _build_file_registry,
    _fetch_manifest,
    _get_console,
    _install_toolchain,
    _resolve_version,
    app,
)

runner = CliRunner()


class TestGetConsole:
    """Tests for the _get_console helper function."""

    def test_get_console_returns_console(self):
        """Test that _get_console returns a Console instance."""
        console = _get_console()
        assert console is not None


class TestFetchManifest:
    """Tests for the _fetch_manifest function."""

    def test_fetch_manifest_success(self):
        """Test successful manifest fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "default_toolchain": "0.1.0",
            "toolchains": {"0.1.0": {}},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = _fetch_manifest()
            assert result is not None
            assert result["default_toolchain"] == "0.1.0"

    def test_fetch_manifest_network_error(self):
        """Test manifest fetch with network error."""
        import requests

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
            result = _fetch_manifest()
            assert result is None

    def test_fetch_manifest_http_error(self):
        """Test manifest fetch with HTTP error."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        with patch("requests.get", return_value=mock_response):
            result = _fetch_manifest()
            assert result is None

    def test_fetch_manifest_timeout(self):
        """Test manifest fetch with timeout."""
        import requests

        with patch("requests.get", side_effect=requests.exceptions.Timeout()):
            result = _fetch_manifest()
            assert result is None


class TestResolveVersion:
    """Tests for the _resolve_version function."""

    def test_resolve_version_none_uses_default(self):
        """Test that None version resolves to default."""
        manifest = {"default_toolchain": "1.0.0"}
        result = _resolve_version(manifest, None)
        assert result == "1.0.0"

    def test_resolve_version_stable_uses_default(self):
        """Test that 'stable' resolves to default."""
        manifest = {"default_toolchain": "1.0.0"}
        result = _resolve_version(manifest, "stable")
        assert result == "1.0.0"

    def test_resolve_version_stable_case_insensitive(self):
        """Test that 'STABLE' is case insensitive."""
        manifest = {"default_toolchain": "1.0.0"}
        result = _resolve_version(manifest, "STABLE")
        assert result == "1.0.0"

    def test_resolve_version_specific(self):
        """Test that specific version is returned as-is."""
        manifest = {"default_toolchain": "1.0.0"}
        result = _resolve_version(manifest, "2.0.0")
        assert result == "2.0.0"

    def test_resolve_version_no_default(self):
        """Test error when no default and version is None."""
        manifest = {}
        with pytest.raises(ValueError, match="No default_toolchain"):
            _resolve_version(manifest, None)


class TestBuildFileRegistry:
    """Tests for the _build_file_registry function."""

    def test_build_registry_with_valid_files(self):
        """Test building registry with valid file entries."""
        version_info = {
            "files": [
                {"remote_name": "file1.gz", "sha256": "abc123", "local_name": "file1"},
                {"remote_name": "file2.gz", "sha256": "def456", "local_name": "file2"},
            ]
        }
        result = _build_file_registry(version_info)
        assert result == {
            "file1.gz": "sha256:abc123",
            "file2.gz": "sha256:def456",
        }

    def test_build_registry_skips_incomplete_entries(self):
        """Test that entries without remote_name or sha256 are skipped."""
        version_info = {
            "files": [
                {"remote_name": "file1.gz", "sha256": "abc123"},
                {"remote_name": "file2.gz"},  # Missing sha256
                {"sha256": "def456"},  # Missing remote_name
            ]
        }
        result = _build_file_registry(version_info)
        assert result == {"file1.gz": "sha256:abc123"}

    def test_build_registry_empty_files(self):
        """Test building registry with empty files list."""
        version_info = {"files": []}
        result = _build_file_registry(version_info)
        assert result == {}

    def test_build_registry_no_files_key(self):
        """Test building registry when files key is missing."""
        version_info = {}
        result = _build_file_registry(version_info)
        assert result == {}


class TestInstallToolchain:
    """Tests for the _install_toolchain function."""

    def test_install_toolchain_manifest_fetch_fails(self):
        """Test error when manifest fetch fails."""
        with patch(
            "lean_explore.cli.data_commands._fetch_manifest", return_value=None
        ):
            with pytest.raises(ValueError, match="Failed to fetch manifest"):
                _install_toolchain()

    def test_install_toolchain_version_not_found(self):
        """Test error when version is not in manifest."""
        manifest = {
            "default_toolchain": "1.0.0",
            "toolchains": {"1.0.0": {}},
        }
        with patch(
            "lean_explore.cli.data_commands._fetch_manifest", return_value=manifest
        ):
            with pytest.raises(ValueError, match="not found"):
                _install_toolchain("2.0.0")

    def test_install_toolchain_shows_available_versions(self):
        """Test that error message shows available versions."""
        manifest = {
            "default_toolchain": "1.0.0",
            "toolchains": {"1.0.0": {}, "1.1.0": {}},
        }
        with patch(
            "lean_explore.cli.data_commands._fetch_manifest", return_value=manifest
        ):
            with pytest.raises(ValueError) as exc_info:
                _install_toolchain("2.0.0")
            assert "1.0.0" in str(exc_info.value) or "Available" in str(exc_info.value)

    @pytest.mark.integration
    def test_install_toolchain_success(self):
        """Test successful toolchain installation."""
        manifest = {
            "default_toolchain": "1.0.0",
            "toolchains": {
                "1.0.0": {
                    "assets_base_path_r2": "v1",
                    "files": [
                        {
                            "remote_name": "test.gz",
                            "sha256": "abc123",
                            "local_name": "test",
                        }
                    ],
                }
            },
        }

        mock_pooch = MagicMock()
        mock_pooch.fetch = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_patch = patch(
                "lean_explore.cli.data_commands._fetch_manifest",
                return_value=manifest,
            )
            pooch_patch = patch("lean_explore.cli.data_commands.pooch.create")
            config_patch = patch(
                "lean_explore.cli.data_commands.Config.DATA_DIRECTORY",
                Path(tmpdir),
            )
            with manifest_patch, pooch_patch as mock_create, config_patch:
                mock_create.return_value = mock_pooch

                _install_toolchain("1.0.0")

                mock_create.assert_called_once()
                mock_pooch.fetch.assert_called_once()


class TestFetchCommand:
    """Tests for the fetch CLI command."""

    def test_fetch_command_help(self):
        """Test fetch command help output."""
        result = runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_fetch_command_calls_install(self):
        """Test that fetch command calls _install_toolchain."""
        with patch(
            "lean_explore.cli.data_commands._install_toolchain"
        ) as mock_install:
            runner.invoke(app, ["fetch"])
            mock_install.assert_called_once_with(None)

    def test_fetch_command_with_version(self):
        """Test fetch command with specific version."""
        with patch(
            "lean_explore.cli.data_commands._install_toolchain"
        ) as mock_install:
            runner.invoke(app, ["fetch", "--version", "1.0.0"])
            mock_install.assert_called_once_with("1.0.0")


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


class TestDataApp:
    """Tests for the data app structure."""

    def test_data_app_no_args_shows_help(self):
        """Test that data app with no args shows help."""
        result = runner.invoke(app)
        # Typer exits with code 2 when no_args_is_help=True and no args provided
        assert result.exit_code in (0, 2)
        # Should show help with available commands
        assert "fetch" in result.output.lower() or "clean" in result.output.lower()
