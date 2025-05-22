# src/lean_explore/cli/data_commands.py

"""Provides CLI commands for managing local Lean exploration data toolchains.

This module includes functions to fetch toolchain data (database, FAISS index, etc.)
from a remote source (Cloudflare R2), verify its integrity, decompress it,
and place it in the appropriate local directory for the application to use.
"""

import gzip
import hashlib
import json
import pathlib
import shutil
import sys # For exit
from typing import Any, Dict, List, Optional

import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from lean_explore import defaults # For R2 URLs and local paths

# Typer application for data commands
app = typer.Typer(
    name="data",
    help="Manage local data toolchains for Lean Explore (e.g., download, list, select).",
    no_args_is_help=True,
)

# Initialize console for rich output
console = Console()


# --- Internal Helper Functions ---

def _fetch_remote_json(url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Fetches JSON data from a remote URL.

    Args:
        url: The URL to fetch JSON from.
        timeout: Request timeout in seconds.

    Returns:
        A dictionary parsed from JSON, or None if an error occurs.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching manifest from {url}: {e}[/bold red]")
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error parsing JSON from {url}: {e}[/bold red]")
    return None


def _resolve_toolchain_version_info(
    manifest_data: Dict[str, Any], requested_identifier: str
) -> Optional[Dict[str, Any]]:
    """Resolves a requested version identifier to its concrete toolchain info.

    Handles aliases like "stable" by looking up "default_toolchain" in the manifest.

    Args:
        manifest_data: The parsed manifest dictionary.
        requested_identifier: The version string requested by the user (e.g., "stable", "0.1.0").

    Returns:
        The dictionary containing information for the resolved concrete toolchain
        version, or None if not found or resolvable.
    """
    toolchains_dict = manifest_data.get("toolchains")
    if not isinstance(toolchains_dict, dict):
        console.print("[bold red]Error: Manifest is missing 'toolchains' dictionary.[/bold red]")
        return None

    target_version_key = requested_identifier
    if requested_identifier.lower() == "stable":
        stable_alias_target = manifest_data.get("default_toolchain")
        if not stable_alias_target:
            console.print(
                f"[bold red]Error: Manifest does not define a 'default_toolchain' for 'stable'.[/bold red]"
            )
            return None
        target_version_key = stable_alias_target
        console.print(f"Note: 'stable' currently points to version '{target_version_key}'.")

    version_info = toolchains_dict.get(target_version_key)
    if not version_info:
        console.print(
            f"[bold red]Error: Version '{target_version_key}' (resolved from '{requested_identifier}') "
            f"not found in the manifest.[/bold red]"
        )
        return None

    # Store the resolved key for easier access by the caller
    version_info["_resolved_key"] = target_version_key
    return version_info


def _download_file_with_progress(
    url: str,
    destination_path: pathlib.Path,
    description: str,
    expected_size_bytes: Optional[int] = None,
    timeout: int = 30,
) -> bool:
    """Downloads a file from a URL with a progress bar, saving raw bytes.

    This function attempts to download the raw bytes from the server,
    especially to handle pre-gzipped files correctly without interference
    from the requests library's automatic content decoding.

    Args:
        url: The URL to download from.
        destination_path: The local path to save the downloaded file.
        description: A description of the file for the progress bar.
        expected_size_bytes: The expected size of the file in bytes for progress tracking.
                             This should typically be the size of the compressed file if
                             downloading a gzipped file.
        timeout: Request timeout in seconds for establishing connection and for read.

    Returns:
        True if download was successful, False otherwise.
    """
    console.print(f"Downloading [cyan]{description}[/cyan] from {url}...")
    try:
        # By not setting 'Accept-Encoding', we let the server decide if it wants
        # to send a Content-Encoding. We will handle the raw stream.
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            # Content-Length should refer to the size of the entity on the wire.
            # If the server sends Content-Encoding: gzip, this should be the gzipped size.
            total_size_from_header = int(r.headers.get("content-length", 0))

            display_size = total_size_from_header
            if expected_size_bytes is not None:
                if total_size_from_header > 0 and expected_size_bytes != total_size_from_header:
                    console.print(
                        f"[yellow]Warning: Expected size for [cyan]{description}[/cyan] is {expected_size_bytes} bytes, "
                        f"but server reports Content-Length: {total_size_from_header} bytes. "
                        f"Using server reported size for progress bar if available, otherwise expected size.[/yellow]"
                    )
                # Prefer expected_size_bytes if it's provided and server doesn't send Content-Length
                # or if we want to strictly adhere to manifest size for progress.
                # However, for live progress, server's content-length is usually more accurate
                # for what's being transferred.
                if total_size_from_header == 0: # If server didn't provide content-length
                     display_size = expected_size_bytes
            elif total_size_from_header == 0 and expected_size_bytes is None:
                # Cannot determine size for progress bar
                display_size = None


            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as progress:
                task_id = progress.add_task(description, total=display_size)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_bytes_count = 0
                with open(destination_path, "wb") as f:
                    # Iterate over the raw stream to prevent requests from auto-decompressing
                    # based on Content-Encoding headers.
                    for chunk in r.raw.stream(decode_content=False, amt=8192):
                        f.write(chunk)
                        downloaded_bytes_count += len(chunk)
                        progress.update(task_id, advance=len(chunk))

        # Sanity check after download
        actual_downloaded_size = destination_path.stat().st_size
        if total_size_from_header > 0 and actual_downloaded_size != total_size_from_header:
            # This might indicate an incomplete download if not all bytes were written.
            console.print(
                f"[orange3]Warning: For [cyan]{description}[/cyan], downloaded size ({actual_downloaded_size} bytes) "
                f"differs from Content-Length header ({total_size_from_header} bytes). "
                f"Checksum verification will be the final arbiter.[/orange3]"
            )
        elif expected_size_bytes is not None and actual_downloaded_size != expected_size_bytes:
             console.print(
                f"[orange3]Warning: For [cyan]{description}[/cyan], downloaded size ({actual_downloaded_size} bytes) "
                f"differs from manifest expected size ({expected_size_bytes} bytes). "
                f"Checksum verification will be the final arbiter.[/orange3]"
            )


        console.print(f"[green]Downloaded raw content for {description} successfully.[/green]")
        return True
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error downloading {description}: {e}[/bold red]")
    except IOError as e:
        console.print(f"[bold red]Error writing {description} to disk: {e}[/bold red]")
    except Exception as e: # Catch any other unexpected errors during download
        console.print(f"[bold red]An unexpected error occurred during download of {description}: {e}[/bold red]")

    if destination_path.exists():
        destination_path.unlink(missing_ok=True)
    return False


def _verify_sha256_checksum(file_path: pathlib.Path, expected_checksum: str) -> bool:
    """Verifies the SHA256 checksum of a file.

    Args:
        file_path: Path to the file to verify.
        expected_checksum: The expected SHA256 checksum string (hex digest).

    Returns:
        True if the checksum matches, False otherwise.
    """
    console.print(f"Verifying checksum for [cyan]{file_path.name}[/cyan]...")
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        calculated_checksum = sha256_hash.hexdigest()
        if calculated_checksum == expected_checksum.lower():
            console.print(f"[green]Checksum verified for {file_path.name}.[/green]")
            return True
        else:
            console.print(
                f"[bold red]Checksum mismatch for {file_path.name}:[/bold red]\n"
                f"  Expected: {expected_checksum.lower()}\n"
                f"  Got:      {calculated_checksum}"
            )
            return False
    except IOError as e:
        console.print(f"[bold red]Error reading file {file_path.name} for checksum: {e}[/bold red]")
        return False


def _decompress_gzipped_file(
    gzipped_file_path: pathlib.Path, output_file_path: pathlib.Path
) -> bool:
    """Decompresses a .gz file.

    Args:
        gzipped_file_path: Path to the .gz file.
        output_file_path: Path to save the decompressed output.

    Returns:
        True if decompression was successful, False otherwise.
    """
    console.print(f"Decompressing [cyan]{gzipped_file_path.name}[/cyan] to {output_file_path.name}...")
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gzipped_file_path, "rb") as f_in:
            with open(output_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        console.print(f"[green]Decompressed {gzipped_file_path.name} successfully.[/green]")
        return True
    except (gzip.BadGzipFile, EOFError, IOError) as e:
        console.print(f"[bold red]Error decompressing {gzipped_file_path.name}: {e}[/bold red]")
    if output_file_path.exists(): # Clean up partial decompression
        output_file_path.unlink(missing_ok=True)
    return False


# --- CLI Command Functions ---

@app.command()
def fetch(
    version: str = typer.Argument(
        "stable",
        help=(
            "The toolchain version to fetch (e.g., 'stable', '0.1.0'). "
            "'stable' will attempt to use the 'default_toolchain' from the manifest."
        ),
    )
) -> None:
    """
    Fetches and installs a specified data toolchain version from the remote repository.

    Downloads necessary assets like the database and FAISS index, verifies their
    integrity via SHA256 checksums, decompresses them, and places them into the
    appropriate local directory (e.g., ~/.lean_explore/data/toolchains/<version>/).
    """
    console.rule(f"[bold blue]Fetching Lean Explore Data Toolchain: {version}[/bold blue]")

    # 1. Fetch and Parse Manifest
    console.print(f"Fetching data manifest from {defaults.R2_MANIFEST_DEFAULT_URL}...")
    manifest_data = _fetch_remote_json(defaults.R2_MANIFEST_DEFAULT_URL)
    if not manifest_data:
        console.print("[bold red]Failed to fetch or parse the manifest. Aborting.[/bold red]")
        raise typer.Exit(code=1)
    console.print("[green]Manifest fetched successfully.[/green]")

    # 2. Resolve Target Version from Manifest
    version_info = _resolve_toolchain_version_info(manifest_data, version)
    if not version_info:
        # _resolve_toolchain_version_info already prints detailed errors
        raise typer.Exit(code=1)

    resolved_version_key = version_info["_resolved_key"] # Key like "0.1.0"
    console.print(
        f"Processing toolchain version: [bold yellow]{resolved_version_key}[/bold yellow] "
        f"('{version_info.get('description', 'N/A')}')"
    )

    # 3. Determine Local Paths and Ensure Directory Exists
    local_version_dir = defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR / resolved_version_key
    try:
        local_version_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Data will be stored in: [dim]{local_version_dir}[/dim]")
    except OSError as e:
        console.print(
            f"[bold red]Error creating local directory {local_version_dir}: {e}[/bold red]"
        )
        raise typer.Exit(code=1)

    # 4. Process Files for the Target Version
    files_to_process: List[Dict[str, Any]] = version_info.get("files", [])
    if not files_to_process:
        console.print(
            f"[yellow]No files listed in the manifest for version '{resolved_version_key}'. Nothing to do.[/yellow]"
        )
        raise typer.Exit(code=0)

    all_files_successful = True
    for file_entry in files_to_process:
        local_name = file_entry.get("local_name")
        remote_name = file_entry.get("remote_name")
        expected_checksum = file_entry.get("sha256")
        expected_size_compressed = file_entry.get("size_bytes_compressed") # This is size of .gz
        assets_r2_path_prefix = version_info.get("assets_base_path_r2", "") # e.g., "assets/0.1.0/"

        if not all([local_name, remote_name, expected_checksum]):
            console.print(
                f"[bold red]Skipping invalid file entry in manifest: {file_entry}. "
                f"Missing name, remote name, or checksum.[/bold red]"
            )
            all_files_successful = False
            continue

        console.rule(f"[bold cyan]Processing: {local_name}[/bold cyan]")

        final_local_path = local_version_dir / local_name
        temp_download_path = local_version_dir / remote_name # Path for the .gz file

        remote_url = (
            defaults.R2_ASSETS_BASE_URL.rstrip("/") + "/" +
            assets_r2_path_prefix.strip("/") + "/" +
            remote_name
        )

        if final_local_path.exists():
            console.print(
                f"[yellow]'{local_name}' already exists at {final_local_path}. Skipping download.[/yellow]\n"
                f"[dim]  (Checksum verification for existing files is not yet implemented. "
                f"Delete the file to re-download).[/dim]"
            )
            continue

        if temp_download_path.exists():
            temp_download_path.unlink(missing_ok=True)

        download_ok = _download_file_with_progress(
            remote_url, temp_download_path, description=local_name,
            expected_size_bytes=expected_size_compressed # Pass size of .gz for progress
        )
        if not download_ok:
            all_files_successful = False
            console.print(f"[bold red]Failed to download {remote_name}. Halting for this file.[/bold red]")
            continue

        checksum_ok = _verify_sha256_checksum(temp_download_path, expected_checksum)
        if not checksum_ok:
            all_files_successful = False
            console.print(
                f"[bold red]Checksum verification failed for {remote_name}. "
                f"Deleting downloaded file.[/bold red]"
            )
            temp_download_path.unlink(missing_ok=True)
            continue

        decompress_ok = _decompress_gzipped_file(temp_download_path, final_local_path)
        if not decompress_ok:
            all_files_successful = False
            console.print(
                f"[bold red]Failed to decompress {remote_name}. "
                f"Cleaning up temporary files.[/bold red]"
            )
            if final_local_path.exists():
                final_local_path.unlink(missing_ok=True)
            if temp_download_path.exists(): # Ensure .gz is also removed on decompress failure
                temp_download_path.unlink(missing_ok=True)
            continue

        if temp_download_path.exists():
            temp_download_path.unlink()
        console.print(f"[green]Successfully installed and verified {local_name} to {final_local_path}[/green]\n")

    console.rule()
    if all_files_successful:
        console.print(
            f"[bold green]Toolchain '{resolved_version_key}' fetch process completed successfully.[/bold green]"
        )
    else:
        console.print(
            f"[bold orange3]Toolchain '{resolved_version_key}' fetch process completed with some errors. "
            f"Please review the output above.[/bold orange3]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # This allows testing `python -m lean_explore.cli.data_commands fetch stable`
    # For actual CLI use, this app will be mounted in `main.py`.
    app()