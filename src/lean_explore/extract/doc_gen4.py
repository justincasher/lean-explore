"""Documentation generation using doc-gen4.

This module provides functionality to run doc-gen4 to generate Lean documentation
data that can be parsed and processed by the extraction pipeline.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


async def run_doc_gen4() -> None:
    """Run doc-gen4 to generate documentation data.

    This function runs doc-gen4 from the lean/docbuild subdirectory
    following the recommended setup from the doc-gen4 documentation.

    Steps:
    1. Updates doc-gen4 dependency to the pinned version
    2. Updates LeanExtract dependency (parent library)
    3. Generates documentation using doc-gen4

    Each step streams output in real-time.

    Raises:
        RuntimeError: If any build step fails with a non-zero exit code.
    """
    logger.info("Running doc-gen4 to generate documentation...")

    commands = [
        (["lake", "update", "doc-gen4"], "Updating doc-gen4 dependency"),
        (["lake", "update", "LeanExtract"], "Updating LeanExtract dependency"),
        (["lake", "build", "LeanExtract:docs"], "Generating documentation"),
    ]

    for command, description in commands:
        logger.info(f"{description}...")

        process = subprocess.Popen(
            command,
            cwd="lean/docbuild",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)

        returncode = process.wait()

        if returncode != 0:
            logger.error(f"{description} failed with return code {returncode}")
            raise RuntimeError(f"{description} failed")

    logger.info("doc-gen4 generation complete")
