"""GitHub utilities for fetching package metadata.

This module provides functions to interact with GitHub repositories
for fetching toolchain versions and release tags.
"""

import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)


def github_url_to_raw(git_url: str, branch: str, file_path: str) -> str:
    """Convert GitHub repo URL to raw file URL.

    Args:
        git_url: GitHub repository URL (e.g., https://github.com/owner/repo)
        branch: Branch or tag name
        file_path: Path to file in repo

    Returns:
        Raw GitHub URL for the file.
    """
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", git_url)
    if not match:
        raise ValueError(f"Could not parse GitHub URL: {git_url}")
    owner, repo = match.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"


def fetch_lean_toolchain(git_url: str, ref: str = "main") -> str:
    """Fetch lean-toolchain content from a GitHub repository.

    Args:
        git_url: GitHub repository URL
        ref: Branch name or tag (default: main)

    Returns:
        Content of the lean-toolchain file (e.g., 'leanprover/lean4:v4.27.0')
    """
    raw_url = github_url_to_raw(git_url, ref, "lean-toolchain")
    logger.info(f"Fetching lean-toolchain from {raw_url}")

    try:
        with urllib.request.urlopen(raw_url, timeout=30) as response:
            return response.read().decode("utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch lean-toolchain from {raw_url}: {e}")


def fetch_latest_tag(git_url: str) -> str:
    """Fetch the latest semver tag from a GitHub repository.

    Args:
        git_url: GitHub repository URL

    Returns:
        Latest tag name (e.g., 'v4.26.0')
    """
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", git_url)
    if not match:
        raise ValueError(f"Could not parse GitHub URL: {git_url}")
    owner, repo = match.groups()

    api_url = f"https://api.github.com/repos/{owner}/{repo}/tags?per_page=100"
    logger.info(f"Fetching tags from {api_url}")

    try:
        request = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch tags from {api_url}: {e}")

    if not tags:
        raise RuntimeError(f"No tags found for {git_url}")

    # Filter to semver-like tags (v*.*.*)
    semver_pattern = re.compile(r"^v?\d+\.\d+\.\d+")
    semver_tags = [t["name"] for t in tags if semver_pattern.match(t["name"])]

    if not semver_tags:
        return tags[0]["name"]

    def semver_key(tag: str) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", tag)]

    semver_tags.sort(key=semver_key, reverse=True)
    return semver_tags[0]


def extract_lean_version(toolchain: str) -> str:
    """Extract version from lean-toolchain content.

    Args:
        toolchain: Toolchain content like 'leanprover/lean4:v4.27.0'
            or 'leanprover/lean4:v4.28.0-rc1'.

    Returns:
        Version string like 'v4.27.0' or 'v4.28.0-rc1'
    """
    match = re.search(r"v\d+\.\d+\.\d+(?:-rc\d+)?", toolchain)
    if not match:
        raise ValueError(f"Could not extract version from toolchain: {toolchain}")
    return match.group()
