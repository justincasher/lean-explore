# src/lean_explore/cli/data_commands.py

"""Manages local Lean Explore data toolchains.

Provides CLI commands to download, install, and clean data files (database,
FAISS index, etc.) from remote storage using Pooch for checksums and caching.
"""

import logging
import shutil
from typing import Any

import pooch
import requests
import typer
from rich.console import Console

from lean_explore.config import Config

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="data",
    help="Manage local data toolchains for Lean Explore (e.g., download, list, "
    "select, clean).",
    no_args_is_help=True,
)


def _get_console() -> Console:
    """Create a Rich console instance for output."""
    return Console()


def _fetch_manifest() -> dict[str, Any] | None:
    """Fetches the remote data manifest.

    Returns:
        The manifest dictionary, or None if fetch fails.
    """
    console = _get_console()
    try:
        response = requests.get(Config.MANIFEST_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        logger.error("Failed to fetch manifest: %s", error)
        console.print(f"[bold red]Error fetching manifest: {error}[/bold red]")
        return None


def _resolve_version(manifest: dict[str, Any], version: str | None) -> str:
    """Resolves the version string to an actual toolchain version.

    Args:
        manifest: The manifest dictionary containing toolchain information.
        version: The requested version, or None/"stable" for default.

    Returns:
        The resolved version string.

    Raises:
        ValueError: If the version cannot be resolved.
    """
    if not version or version.lower() == "stable":
        resolved = manifest.get("default_toolchain")
        if not resolved:
            raise ValueError("No default_toolchain specified in manifest")
        return resolved
    return version


def _build_file_registry(version_info: dict[str, Any]) -> dict[str, str]:
    """Builds a Pooch registry from version info.

    Args:
        version_info: The version information from the manifest.

    Returns:
        A dictionary mapping remote filenames to SHA256 checksums.
    """
    return {
        file_entry["remote_name"]: f"sha256:{file_entry['sha256']}"
        for file_entry in version_info.get("files", [])
        if file_entry.get("remote_name") and file_entry.get("sha256")
    }


def _install_toolchain(version: str | None = None) -> None:
    """Installs the data toolchain for the specified version.

    Downloads and verifies all required data files (database, FAISS index, etc.)
    using Pooch. Files are automatically decompressed and cached locally.

    Args:
        version: The version to install. If None, uses the default version.

    Raises:
        ValueError: If manifest fetch fails or version is not found.
    """
    console = _get_console()

    manifest = _fetch_manifest()
    if not manifest:
        raise ValueError("Failed to fetch manifest")

    resolved_version = _resolve_version(manifest, version)
    version_info = manifest.get("toolchains", {}).get(resolved_version)
    if not version_info:
        available = list(manifest.get("toolchains", {}).keys())
        raise ValueError(
            f"Version '{resolved_version}' not found. Available: {available}"
        )

    registry = _build_file_registry(version_info)
    base_path = version_info.get("assets_base_path_r2", "")
    base_url = f"{Config.R2_ASSETS_BASE_URL}/{base_path}/"

    file_downloader = pooch.create(
        path=Config.CACHE_DIRECTORY / resolved_version,
        base_url=base_url,
        registry=registry,
    )

    # Download and decompress each file
    for file_entry in version_info.get("files", []):
        remote_name = file_entry.get("remote_name")
        local_name = file_entry.get("local_name")
        if remote_name and local_name:
            logger.info("Downloading %s -> %s", remote_name, local_name)
            file_downloader.fetch(
                remote_name, processor=pooch.Decompress(name=local_name)
            )

    console.print(f"[green]Installed data for version {resolved_version}[/green]")


@app.callback()
def main() -> None:
    """Lean-Explore data CLI.

    This callback exists only to prevent Typer from treating the first
    sub-command as a *default* command when there is otherwise just one.
    """
    pass


@app.command()
def fetch(
    version: str = typer.Option(
        None,
        "--version",
        "-v",
        help="Version to install (e.g., '0.1.0'). Defaults to stable/latest.",
    ),
) -> None:
    """Fetches and installs the data toolchain from the remote repository.

    Downloads the database, FAISS index, and other required data files.
    Files are verified with SHA256 checksums and automatically decompressed.
    """
    _install_toolchain(version)


@app.command("clean")
def clean_data_toolchains() -> None:
    """Removes all downloaded local data toolchains."""
    console = _get_console()

    if not Config.CACHE_DIRECTORY.exists():
        console.print("[yellow]No local data found to clean.[/yellow]")
        return

    if typer.confirm("Delete all cached data?", default=False, abort=True):
        try:
            shutil.rmtree(Config.CACHE_DIRECTORY)
            console.print("[green]Data cache cleared.[/green]")
        except OSError as error:
            logger.error("Failed to clean cache directory: %s", error)
            console.print(f"[bold red]Error cleaning data: {error}[/bold red]")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
