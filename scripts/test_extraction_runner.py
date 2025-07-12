# scripts/test_extraction_runner.py

"""Runs a proof-of-concept data extraction using lean-interact in parallel.

This script is designed to test and validate the data extraction capabilities of
the lean-interact server against a full-scale Lean project (e.g., Mathlib).
It performs the following steps:
1. Clones or updates a specified Git repository for a Lean project.
2. Creates a pool of worker processes and a shared queue for progress tracking.
3. Divides the list of all .lean files into chunks for each worker.
4. Each worker initializes a single, long-lived lean-interact AutoLeanServer
   and processes all files in its assigned chunk. After each file, it sends
   a message to the shared queue.
5. The main process monitors the queue to update a single progress bar in
   real-time, providing a "livestream" of the overall progress.
6. Once all workers are finished, the main process collects the aggregated
   results and writes them to two JSONL files:
   - declarations.jsonl: Contains one JSON object per declaration.
   - dependencies.jsonl: Contains one JSON object per unique dependency link.
"""

import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# --- Dependency Imports ---
try:
    from git import Repo, GitCommandError
    from lean_interact import (
        AutoLeanServer,
        FileCommand,
        LeanREPLConfig,
        LocalProject,
    )
    from lean_interact.interface import (
        CommandResponse,
        LeanError,
    )
    from lean_interact.interface import DeclarationInfo
except ImportError as e:
    print(f"Failed to import a required library: {e}")
    print(
        "Please install the required dependencies:\n"
        "pip install tqdm GitPython \"lean_interact @ "
        "git+https://github.com/augustepoiroux/LeanInteract.git"
        "@5c0b61a5ae8c6cc23f3bb37ae554423332112ff8\""
    )
    exit(1)


# --- Configuration ---
# The Git repository of the Lean project to be processed.
GIT_REPO_URL: str = "https://github.com/leanprover-community/mathlib4.git"

# A local directory within the scripts folder to store the cloned project.
PROJECTS_DIR: Path = Path(__file__).parent / "temp_projects"
PROJECT_PATH: Path = PROJECTS_DIR / Path(GIT_REPO_URL).stem

# Output directory for the generated .jsonl files.
OUTPUT_DIR: Path = Path(__file__).parent.parent / "data" / "test_extraction_output"

# Number of parallel workers to use. Defaults to the number of CPU cores.
NUM_WORKERS: int = 32

# Limit the number of files to process for a quick test run.
# Set to `None` to process all files in the project.
MAX_FILES_TO_PROCESS: Optional[int] = None


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Suppress overly verbose logs from dependencies for cleaner test output.
logging.getLogger("lean_interact").setLevel(logging.WARNING)


def setup_project_repo(repo_url: str, target_dir: Path) -> None:
    """Ensures the target Lean project repository is cloned and up-to-date.

    If the target directory does not exist, it clones the repository.
    If it already exists, it pulls the latest changes from the 'master' branch.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        target_dir (Path): The local path where the repository should be stored.
    """
    logger.info(f"Setting up project repository at: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info("Repository exists. Pulling latest changes...")
        try:
            repo = Repo(target_dir)
            origin = repo.remotes.origin
            origin.pull()
            logger.info("Successfully pulled latest changes.")
        except GitCommandError as e:
            logger.error(f"Failed to pull repository: {e}")
            raise
    else:
        logger.info(f"Cloning repository from {repo_url}...")
        try:
            Repo.clone_from(repo_url, target_dir, branch="master")
            logger.info("Successfully cloned repository.")
        except GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise


def is_internal_heuristic(lean_name: str) -> bool:
    """Applies a heuristic to determine if a declaration is internal.

    This logic is carried over from the original population scripts to maintain
    consistency in identifying auto-generated or core-internal declarations.

    Args:
        lean_name (str): The fully qualified name of the declaration.

    Returns:
        bool: True if the name suggests it is an internal declaration.
    """
    auto_generated_suffixes = [
        ".noConfusion", ".noConfusionType", ".rec", ".recOn", ".casesOn",
        ".brecOn", ".below", ".IBelow", ".ndrec", ".ndrecOn", ".match_1",
        ".match_2", ".matcher", ".mk.inj", ".mk.inj_arrow", ".sizeOf_spec",
        "._uniq", ".internal",
    ]
    core_prefixes = ["Lean.", "Init."]

    is_internal = (
        any(lean_name.startswith(p) for p in core_prefixes)
        or any(lean_name.endswith(s) for s in auto_generated_suffixes)
        or "._match" in lean_name
        or "._proof_" in lean_name
        or "._example" in lean_name
    )

    name_parts = lean_name.split(".")
    if (
        len(name_parts) > 1
        and name_parts[-1].startswith("eq_")
        and name_parts[-1][3:].isdigit()
    ):
        is_internal = True

    if ".Internal." in name_parts:
        is_internal = True

    return is_internal


def map_decl_info_to_dict(decl_info: DeclarationInfo, file_path: str) -> Dict:
    """Maps a DeclarationInfo object to a dictionary for JSON serialization.

    Args:
        decl_info (DeclarationInfo): The structured declaration object from
            the lean-interact server.
        file_path (str): The relative path of the source .lean file.

    Returns:
        Dict: A dictionary representing the declaration, ready for JSONL output.
    """
    return {
        "lean_name": decl_info.full_name,
        "decl_type": decl_info.kind,
        "source_file": file_path,
        "module_name": None,  # This can be derived later if needed
        "is_internal": is_internal_heuristic(decl_info.full_name),
        "docstring": (
            decl_info.modifiers.doc_string.content
            if decl_info.modifiers.doc_string
            else None
        ),
        "is_protected": (decl_info.modifiers.visibility == "protected"),
        "is_deprecated": False,  # Info not available in current extractor
        "is_projection": ("projection" in decl_info.modifiers.attributes),
        "range_start_line": decl_info.range.start.line,
        "range_start_col": decl_info.range.start.column,
        "range_end_line": decl_info.range.finish.line,
        "range_end_col": decl_info.range.finish.column,
        "statement_text": decl_info.pp,
        "declaration_signature": decl_info.signature.pp,
        "attributes": decl_info.modifiers.attributes,
    }


def process_file_chunk_worker(
    args: Tuple[List[Path], Path, multiprocessing.Queue]
) -> Tuple[List[Dict], Set[Tuple[str, str]], List[str]]:
    """Worker function to process a chunk of .lean files in a single process.

    Initializes one single, long-lived Lean server instance for the entire
    chunk. After processing each file, it places an item onto the shared
    progress queue. It returns the aggregated data at the end.

    Args:
        args (Tuple[List[Path], Path, multiprocessing.Queue]): A tuple containing
            a list of file paths to process, the project root path, and the
            shared queue for progress reporting.

    Returns:
        Tuple[List[Dict], Set[Tuple[str, str]], List[str]]: A tuple containing
            the list of extracted declaration dictionaries, the set of
            dependency pairs, and a list of paths for files that failed.
    """
    file_chunk, project_root_path, progress_queue = args
    server = None
    declarations_in_chunk = []
    dependencies_in_chunk = set()
    failed_files = []

    try:
        config = LeanREPLConfig(project=LocalProject(str(project_root_path)))
        server = AutoLeanServer(config)

        for file_path in file_chunk:
            relative_path_str = str(file_path.relative_to(project_root_path))
            try:
                cmd = FileCommand(path=relative_path_str, declarations=True)
                response = server.run(cmd)

                if isinstance(response, LeanError):
                    failed_files.append(relative_path_str)
                    continue

                if isinstance(response, CommandResponse):
                    for decl_info in response.declarations:
                        decl_dict = map_decl_info_to_dict(
                            decl_info, relative_path_str
                        )
                        declarations_in_chunk.append(decl_dict)

                        source_name = decl_info.full_name
                        dep_names = set()
                        if decl_info.type and decl_info.type.constants:
                            dep_names.update(decl_info.type.constants)
                        if decl_info.value and decl_info.value.constants:
                            dep_names.update(decl_info.value.constants)

                        for target_name in dep_names:
                            if source_name != target_name:
                                dependencies_in_chunk.add(
                                    (source_name, target_name)
                                )
            except Exception:
                failed_files.append(relative_path_str)
            finally:
                progress_queue.put(1)  # Signal one file is done

        return declarations_in_chunk, dependencies_in_chunk, failed_files
    finally:
        if server:
            server.kill()


def main():
    """Main function to run the extraction and file generation test."""
    logger.info("--- Starting Parallel Extraction Test ---")
    logger.info(f"Target Project Repository: {GIT_REPO_URL}")
    logger.info(f"Local Project Path: {PROJECT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    declarations_file = OUTPUT_DIR / "declarations.jsonl"
    dependencies_file = OUTPUT_DIR / "dependencies.jsonl"
    if declarations_file.exists():
        declarations_file.unlink()
    if dependencies_file.exists():
        dependencies_file.unlink()

    try:
        setup_project_repo(GIT_REPO_URL, PROJECT_PATH)
    except Exception:
        logger.error("Failed to set up the project repository. Aborting.")
        return

    lean_files = list(Path(PROJECT_PATH).rglob("*.lean"))
    if not lean_files:
        logger.error(f"No .lean files found in {PROJECT_PATH}. Aborting.")
        return

    if MAX_FILES_TO_PROCESS is not None:
        logger.info(
            f"Found {len(lean_files)} total .lean files. "
            f"Processing a subset of {MAX_FILES_TO_PROCESS} files."
        )
        lean_files = lean_files[:MAX_FILES_TO_PROCESS]
    else:
        logger.info(f"Found {len(lean_files)} files. Starting extraction...")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    num_files_to_process = len(lean_files)

    chunk_size = (num_files_to_process + NUM_WORKERS - 1) // NUM_WORKERS
    file_chunks = [
        lean_files[i : i + chunk_size]
        for i in range(0, len(lean_files), chunk_size)
    ]
    worker_args = [(chunk, PROJECT_PATH, progress_queue) for chunk in file_chunks]

    all_declarations: List[Dict] = []
    all_dependency_pairs: Set[Tuple[str, str]] = set()
    total_failed_files: List[str] = []
    start_time = time.time()

    logger.info(f"Starting extraction with {NUM_WORKERS} parallel workers.")
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Launch worker tasks asynchronously
        async_results = [
            pool.apply_async(process_file_chunk_worker, (arg,)) for arg in worker_args
        ]

        # Monitor progress from the shared queue
        with tqdm(total=num_files_to_process, desc="Extracting from files") as pbar:
            completed_count = 0
            while completed_count < num_files_to_process:
                try:
                    progress_queue.get(timeout=10)
                    completed_count += 1
                    pbar.update(1)
                except Empty:
                    # If queue is empty, check if all workers are done
                    if all(res.ready() for res in async_results):
                        break

        # Collect results from all workers
        for res in async_results:
            chunk_decls, chunk_deps, failed_paths = res.get()
            all_declarations.extend(chunk_decls)
            all_dependency_pairs.update(chunk_deps)
            total_failed_files.extend(failed_paths)

    processing_time = time.time() - start_time
    logger.info(f"Parallel extraction finished in {processing_time:.2f} seconds.")
    processed_count = num_files_to_process - len(total_failed_files)
    logger.info(f"Successfully processed: {processed_count} files.")
    logger.info(f"Failed to process: {len(total_failed_files)} files.")
    if total_failed_files:
        logger.warning(f"Example failed files: {total_failed_files[:5]}")

    logger.info(f"Extracted {len(all_declarations)} total declarations.")
    logger.info(f"Writing declarations to {declarations_file}...")
    with open(declarations_file, "w", encoding="utf-8") as f_decls:
        for decl_dict in tqdm(all_declarations, desc="Writing declarations"):
            f_decls.write(json.dumps(decl_dict) + "\n")

    logger.info(f"Extracted {len(all_dependency_pairs)} unique dependency pairs.")
    logger.info(f"Writing dependencies to {dependencies_file}...")
    with open(dependencies_file, "w", encoding="utf-8") as f_deps:
        for source, target in tqdm(all_dependency_pairs, desc="Writing deps"):
            dep_dict = {
                "source_lean_name": source,
                "target_lean_name": target,
                "dependency_type": "Direct",
            }
            f_deps.write(json.dumps(dep_dict) + "\n")

    logger.info("--- ✅ Test Extraction Complete ---")


if __name__ == "__main__":
    main()