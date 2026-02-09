"""Pipeline orchestration for Lean declaration extraction and enrichment.

This module provides functions to coordinate the complete data extraction pipeline:
1. Extract declarations from doc-gen4 output
2. Generate informal natural language descriptions
3. Generate vector embeddings for semantic search
4. Build FAISS indices for vector similarity search
"""

import asyncio
import logging
import os
from pathlib import Path

import click
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from lean_explore.config import Config
from lean_explore.models import Base
from lean_explore.util.logging import setup_logging

logger = logging.getLogger(__name__)


async def _create_database_schema(engine: AsyncEngine) -> None:
    """Create database tables if they don't exist.

    Args:
        engine: SQLAlchemy async engine instance.
    """
    logger.info("Creating database schema...")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database schema created successfully")


async def _run_doc_gen4_step(fresh: bool = False) -> None:
    """Run doc-gen4 to generate documentation.

    Args:
        fresh: Clear cached dependencies to force fresh resolution.
    """
    from lean_explore.extract.doc_gen4 import run_doc_gen4

    logger.info("Running doc-gen4...")
    await run_doc_gen4(fresh=fresh)
    logger.info("doc-gen4 complete")


async def _run_extract_step(engine: AsyncEngine) -> None:
    """Extract declarations from doc-gen4 output."""
    from lean_explore.extract.doc_parser import extract_declarations

    logger.info("Step 1: Extracting declarations from doc-gen4...")
    await extract_declarations(engine)
    logger.info("Declaration extraction complete")


async def _run_informalize_step(
    engine: AsyncEngine,
    model: str,
    batch_size: int,
    max_concurrent: int,
    limit: int | None,
) -> None:
    """Generate informal descriptions for declarations."""
    from lean_explore.extract.informalize import informalize_declarations

    logger.info("Step 2: Generating informal descriptions...")
    await informalize_declarations(
        engine,
        model=model,
        commit_batch_size=batch_size,
        max_concurrent=max_concurrent,
        limit=limit,
    )
    logger.info("Informalization complete")


async def _run_embeddings_step(
    engine: AsyncEngine,
    model_name: str,
    batch_size: int,
    limit: int | None,
    max_seq_length: int,
) -> None:
    """Generate embeddings for all declaration fields."""
    from lean_explore.extract.embeddings import generate_embeddings

    logger.info("Step 3: Generating embeddings...")
    await generate_embeddings(
        engine,
        model_name=model_name,
        batch_size=batch_size,
        limit=limit,
        max_seq_length=max_seq_length,
    )
    logger.info("Embedding generation complete")


async def _run_index_step(engine: AsyncEngine, extraction_path: Path) -> None:
    """Build search indices (FAISS and BM25).

    Args:
        engine: SQLAlchemy async engine instance.
        extraction_path: Directory to save indices (same as database location).
    """
    from lean_explore.extract.index import build_bm25_indices, build_faiss_indices

    logger.info("Step 4: Building search indices...")
    await build_faiss_indices(engine, output_directory=extraction_path)
    await build_bm25_indices(engine, output_directory=extraction_path)
    logger.info("Index building complete")


async def run_pipeline(
    database_url: str,
    extraction_path: Path,
    run_doc_gen4: bool = False,
    fresh: bool = False,
    parse_docs: bool = True,
    informalize: bool = True,
    embeddings: bool = True,
    index: bool = True,
    informalize_model: str = "google/gemini-3-flash-preview",
    informalize_batch_size: int = 1000,
    informalize_max_concurrent: int = 100,
    informalize_limit: int | None = None,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_batch_size: int = 250,
    embedding_limit: int | None = None,
    embedding_max_seq_length: int = 512,
    verbose: bool = False,
) -> None:
    """Run the Lean declaration extraction and enrichment pipeline.

    Args:
        database_url: SQLite database URL (e.g., sqlite+aiosqlite:///path/to/db)
        extraction_path: Directory containing the extraction (for saving indices).
        run_doc_gen4: Run doc-gen4 to generate documentation before parsing
        fresh: Clear cached dependencies to force fresh resolution (for nightly updates)
        parse_docs: Run doc-gen4 parsing step
        informalize: Run informalization step
        embeddings: Run embeddings generation step
        index: Run FAISS index building step
        informalize_model: LLM model for generating informalizations
        informalize_batch_size: Commit batch size for informalization
        informalize_max_concurrent: Maximum concurrent informalization requests
        informalize_limit: Limit number of declarations to informalize
        embedding_model: Sentence transformer model for embeddings
        embedding_batch_size: Batch size for embedding generation
        embedding_limit: Limit number of declarations for embeddings
        embedding_max_seq_length: Max sequence length for embeddings (lower=less mem)
        verbose: Enable verbose logging
    """
    setup_logging(verbose)

    # Validate OpenRouter API key if informalization is needed
    if informalize:
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.error(
                "OPENROUTER_API_KEY environment variable is required for "
                "informalization"
            )
            raise RuntimeError("OPENROUTER_API_KEY not set")

    steps_enabled = []
    if parse_docs:
        steps_enabled.append("parse-docs")
    if informalize:
        steps_enabled.append("informalize")
    if embeddings:
        steps_enabled.append("embeddings")
    if index:
        steps_enabled.append("index")

    logger.info("Starting Lean Explore extraction pipeline")
    logger.info(f"Database URL: {database_url}")
    logger.info(f"Steps to run: {', '.join(steps_enabled)}")

    engine = create_async_engine(database_url, echo=verbose)

    try:
        await _create_database_schema(engine)

        if run_doc_gen4:
            await _run_doc_gen4_step(fresh=fresh)

        if parse_docs:
            await _run_extract_step(engine)

        if informalize:
            await _run_informalize_step(
                engine,
                informalize_model,
                informalize_batch_size,
                informalize_max_concurrent,
                informalize_limit,
            )

        if embeddings:
            await _run_embeddings_step(
                engine,
                embedding_model,
                embedding_batch_size,
                embedding_limit,
                embedding_max_seq_length,
            )

        if index:
            await _run_index_step(engine, extraction_path)

        logger.info("Pipeline completed successfully!")

    finally:
        await engine.dispose()


@click.command()
@click.option(
    "--run-doc-gen4",
    is_flag=True,
    help="Run doc-gen4 to generate documentation before parsing",
)
@click.option(
    "--fresh",
    is_flag=True,
    help="Clear cached dependencies to fetch latest versions (use for nightly updates)",
)
@click.option(
    "--parse-docs/--no-parse-docs",
    default=None,
    help="Run doc-gen4 parsing step (creates new timestamped directory)",
)
@click.option(
    "--informalize/--no-informalize",
    default=None,
    help="Run informalization step (uses latest extraction)",
)
@click.option(
    "--embeddings/--no-embeddings",
    default=None,
    help="Run embeddings generation step (uses latest extraction)",
)
@click.option(
    "--index/--no-index",
    default=None,
    help="Run FAISS index building step (uses latest extraction)",
)
@click.option(
    "--informalize-model",
    default="google/gemini-3-flash-preview",
    help="LLM model for generating informalizations",
)
@click.option(
    "--informalize-max-concurrent",
    type=int,
    default=100,
    help="Maximum concurrent informalization requests",
)
@click.option(
    "--informalize-limit",
    type=int,
    default=None,
    help="Limit number of declarations to informalize (for testing)",
)
@click.option(
    "--embedding-model",
    default="Qwen/Qwen3-Embedding-0.6B",
    help="Sentence transformer model for embeddings",
)
@click.option(
    "--embedding-batch-size",
    type=int,
    default=250,
    help="Batch size for embedding generation (lower = less memory, default 250)",
)
@click.option(
    "--embedding-limit",
    type=int,
    default=None,
    help="Limit number of declarations for embeddings (for testing)",
)
@click.option(
    "--embedding-max-seq-length",
    type=int,
    default=512,
    help="Max sequence length for embeddings (lower = less memory, default 512)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    run_doc_gen4: bool,
    fresh: bool,
    parse_docs: bool | None,
    informalize: bool | None,
    embeddings: bool | None,
    index: bool | None,
    informalize_model: str,
    informalize_max_concurrent: int,
    informalize_limit: int | None,
    embedding_model: str,
    embedding_batch_size: int,
    embedding_limit: int | None,
    embedding_max_seq_length: int,
    verbose: bool,
) -> None:
    """Run the Lean declaration extraction and enrichment pipeline.

    Extraction creates timestamped directories (YYYYMMDD_HHMMSS format).
    Subsequent steps (informalize, embeddings, index) use the latest extraction.
    """
    # Determine if any flags were explicitly set (including --run-doc-gen4)
    step_flags = [run_doc_gen4, parse_docs, informalize, embeddings, index]
    any_flag_explicitly_set = run_doc_gen4 or any(
        flag is not None for flag in step_flags[1:]
    )

    # If no flags were explicitly set, run all pipeline steps by default
    # Otherwise, only run what was explicitly requested
    if not any_flag_explicitly_set:
        parse_docs = informalize = embeddings = index = True
    else:
        parse_docs = parse_docs if parse_docs is not None else False
        informalize = informalize if informalize is not None else False
        embeddings = embeddings if embeddings is not None else False
        index = index if index is not None else False

    # Determine extraction directory
    if parse_docs:
        # Create new timestamped directory for fresh extraction
        extraction_path = Config.create_timestamped_extraction_path()
        logger.info(f"Created new extraction directory: {extraction_path}")
    else:
        # Use latest existing extraction for subsequent steps
        extraction_path = Config.get_latest_extraction_path()
        if extraction_path is None:
            raise click.ClickException(
                "No existing extraction found. Run with --parse-docs first."
            )
        logger.info(f"Using existing extraction: {extraction_path}")

    database_path = extraction_path / "lean_explore.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"

    asyncio.run(
        run_pipeline(
            database_url=database_url,
            extraction_path=extraction_path,
            run_doc_gen4=run_doc_gen4,
            fresh=fresh,
            parse_docs=parse_docs,
            informalize=informalize,
            embeddings=embeddings,
            index=index,
            informalize_model=informalize_model,
            informalize_max_concurrent=informalize_max_concurrent,
            informalize_limit=informalize_limit,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_limit=embedding_limit,
            embedding_max_seq_length=embedding_max_seq_length,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    main()
