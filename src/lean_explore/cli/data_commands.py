# src/lean_explore/cli/data_commands.py

"""Manages local Lean Explore data toolchains.

Provides CLI commands to download, install, and clean data files (database,
FAISS index, etc.) from remote storage using Pooch for checksums and caching.
"""

import pathlib
import shutil
from typing import Any

import pooch
import requests
import typer
from rich.console import Console

# Constants
LEAN_EXPLORE_DATA_CACHE = pathlib.Path.home() / ".lean_explore" / "data"
MANIFEST_URL = "https://pub-48b75babc4664808b15520033423c765.r2.dev/manifest.json"

# Typer application for data commands
app = typer.Typer(
    name="data",
    help="Manage local data toolchains for Lean Explore (e.g., download, list, "
    "select, clean).",
    no_args_is_help=True,
)

console = Console()


# --- Helper Functions ---


def fetch_manifest() -> dict[str, Any] | None:
    """Fetches the remote data manifest.

    Returns:
        The manifest dictionary, or None if fetch fails.
    """
    try:
        response = requests.get(MANIFEST_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching manifest: {e}[/bold red]")
        return None


def install_toolchain(version: str | None = None) -> None:
    """Installs the data toolchain for the specified version.

    Downloads and verifies all required data files (database, FAISS index, etc.)
    using Pooch. Files are automatically decompressed and cached locally.

    Args:
        version: The version to install. If None, uses the default version.
    """
    manifest = fetch_manifest()
    if not manifest:
        raise ValueError("Failed to fetch manifest")

    # Resolve version
    if not version or version.lower() == "stable":
        version = manifest.get("default_toolchain")
        if not version:
            raise ValueError("No default_toolchain in manifest")

    version_info = manifest.get("toolchains", {}).get(version)
    if not version_info:
        raise ValueError(f"Version '{version}' not found in manifest")

    # Build registry
    registry = {
        f["remote_name"]: f"sha256:{f['sha256']}"
        for f in version_info.get("files", [])
        if f.get("remote_name") and f.get("sha256")
    }

    # Create Pooch instance
    base_path = version_info.get("assets_base_path_r2", "")
    base_url = f"https://pub-48b75babc4664808b15520033423c765.r2.dev/{base_path}/"
    pup = pooch.create(
        path=LEAN_EXPLORE_DATA_CACHE / version,
        base_url=base_url,
        registry=registry,
    )

    # Download files
    for file_info in version_info.get("files", []):
        if file_info.get("remote_name") and file_info.get("local_name"):
            pup.fetch(
                file_info["remote_name"],
                processor=pooch.Decompress(name=file_info["local_name"]),
            )

    console.print(f"[green]Installed data for version {version}[/green]")


# --- CLI Commands ---


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
    install_toolchain(version)


@app.command("clean")
def clean_data_toolchains() -> None:
    """Removes all downloaded local data toolchains."""
    if not LEAN_EXPLORE_DATA_CACHE.exists():
        console.print("[yellow]No local data found to clean.[/yellow]")
        return

    if typer.confirm("Delete all cached data?", default=False, abort=True):
        shutil.rmtree(LEAN_EXPLORE_DATA_CACHE)
        console.print("[green]Data cache cleared.[/green]")


if __name__ == "__main__":
    app()
