"""Pipeline orchestration for Lean declaration extraction and enrichment.

This module provides functions to coordinate the complete data extraction pipeline:
1. Extract declarations from doc-gen4 output
2. Calculate PageRank scores based on dependencies
3. Generate informal natural language descriptions
4. Generate vector embeddings for semantic search
5. Build FAISS indices for vector similarity search
"""

import asyncio
import importlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import click
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import lean_explore.config
from lean_explore.config import Config
from lean_explore.extract.doc_parser import extract_declarations
from lean_explore.extract.embeddings import generate_embeddings
from lean_explore.extract.index import build_faiss_indices
from lean_explore.extract.informalize import informalize_declarations
from lean_explore.extract.pagerank import calculate_pagerank
from lean_explore.models import Base
from lean_explore.util import setup_logging

logger = logging.getLogger(__name__)


async def ensure_database_exists(database_url: str, database_name: str) -> None:
    """Ensure the database exists, creating it if necessary.

    Args:
        database_url: Full database URL.
        database_name: Name of the database to create.
    """
    # Try to connect to check if database exists
    test_engine = create_async_engine(database_url, echo=False)
    try:
        async with test_engine.connect():
            logger.info(f"Database {database_name} already exists")
    except Exception as e:
        # Database doesn't exist, create it
        if "does not exist" in str(e):
            logger.info(f"Database {database_name} does not exist, creating it...")

            # Connect to default postgres database to create our database
            base_url = database_url.rsplit("/", 1)[0]
            maintenance_url = f"{base_url}/postgres"
            maintenance_engine = create_async_engine(
                maintenance_url, isolation_level="AUTOCOMMIT", echo=False
            )

            async with maintenance_engine.connect() as connection:
                await connection.execute(text(f'CREATE DATABASE "{database_name}"'))

            await maintenance_engine.dispose()
            logger.info(f"Database {database_name} created successfully")
        else:
            raise
    finally:
        await test_engine.dispose()


async def create_database_schema(engine: AsyncEngine) -> None:
    """Create database tables if they don't exist.

    Args:
        engine: SQLAlchemy async engine instance.
    """
    logger.info("Creating database schema...")
    async with engine.begin() as connection:
        # Enable pgvector extension for vector similarity search
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database schema created successfully")


async def run_doc_gen4_step() -> None:
    """Run doc-gen4 to generate documentation data."""
    logger.info("Running doc-gen4 to generate documentation...")

    # Clean up old build artifacts to prevent version conflicts
    build_directory = Path("lean") / ".lake" / "build"
    for artifact_directory in ["doc", "doc-data"]:
        artifact_path = build_directory / artifact_directory
        if artifact_path.exists():
            logger.info(f"Removing old build artifacts from {artifact_path}")
            shutil.rmtree(artifact_path)

    process = subprocess.Popen(
        ["lake", "build", "LeanExtract:docs"],
        cwd="lean",
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
        logger.error(f"doc-gen4 failed with return code {returncode}")
        raise RuntimeError("doc-gen4 generation failed")

    logger.info("doc-gen4 generation complete")


async def run_extract_step(engine: AsyncEngine) -> None:
    """Extract declarations from doc-gen4 output."""
    logger.info("Step 1: Extracting declarations from doc-gen4...")
    await extract_declarations(engine)
    logger.info("Declaration extraction complete")


async def run_pagerank_step(engine: AsyncEngine, alpha: float, batch_size: int) -> None:
    """Calculate PageRank scores for declarations."""
    logger.info("Step 2: Calculating PageRank scores...")
    await calculate_pagerank(engine, alpha=alpha, batch_size=batch_size)
    logger.info("PageRank calculation complete")


async def run_informalize_step(
    engine: AsyncEngine,
    model: str,
    batch_size: int,
    max_concurrent: int,
    limit: int | None,
) -> None:
    """Generate informal descriptions for declarations."""
    logger.info("Step 3: Generating informal descriptions...")
    await informalize_declarations(
        engine,
        model=model,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        limit=limit,
    )
    logger.info("Informalization complete")


async def run_embeddings_step(
    engine: AsyncEngine,
    model_name: str,
    batch_size: int,
    limit: int | None,
) -> None:
    """Generate embeddings for all declaration fields."""
    logger.info("Step 4: Generating embeddings...")
    await generate_embeddings(
        engine, model_name=model_name, batch_size=batch_size, limit=limit
    )
    logger.info("Embedding generation complete")


async def run_index_step(engine: AsyncEngine) -> None:
    """Build FAISS indices from embeddings."""
    logger.info("Step 5: Building FAISS indices...")
    await build_faiss_indices(engine)
    logger.info("FAISS index building complete")


async def run_pipeline(
    database_url: str,
    run_doc_gen4: bool = False,
    parse_docs: bool = True,
    pagerank: bool = True,
    informalize: bool = True,
    embeddings: bool = True,
    index: bool = True,
    pagerank_alpha: float = 0.85,
    pagerank_batch_size: int = 1000,
    informalize_model: str = "google/gemini-2.5-flash",
    informalize_batch_size: int = 1000,
    informalize_max_concurrent: int = 10,
    informalize_limit: int | None = None,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    embedding_batch_size: int = 250,
    embedding_limit: int | None = None,
    verbose: bool = False,
) -> None:
    """Run the Lean declaration extraction and enrichment pipeline.

    Args:
        database_url: SQLite database URL (e.g., sqlite+aiosqlite:///path/to/db)
        run_doc_gen4: Run doc-gen4 to generate documentation before parsing
        parse_docs: Run doc-gen4 parsing step
        pagerank: Run PageRank calculation step
        informalize: Run informalization step
        embeddings: Run embeddings generation step
        index: Run FAISS index building step
        pagerank_alpha: PageRank damping parameter
        pagerank_batch_size: Batch size for PageRank updates
        informalize_model: LLM model for generating informalizations
        informalize_batch_size: Commit batch size for informalization
        informalize_max_concurrent: Maximum concurrent informalization requests
        informalize_limit: Limit number of declarations to informalize
        embedding_model: Sentence transformer model for embeddings
        embedding_batch_size: Batch size for embedding generation
        embedding_limit: Limit number of declarations for embeddings
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
    if pagerank:
        steps_enabled.append("pagerank")
    if informalize:
        steps_enabled.append("informalize")
    if embeddings:
        steps_enabled.append("embeddings")
    if index:
        steps_enabled.append("index")

    logger.info("Starting Lean Explore extraction pipeline")
    logger.info(f"Database URL: {database_url}")
    logger.info(f"Steps to run: {', '.join(steps_enabled)}")

    # Ensure database exists before trying to connect
    database_name = database_url.rsplit("/", 1)[1]
    await ensure_database_exists(database_url, database_name)

    engine = create_async_engine(database_url, echo=verbose)

    try:
        await create_database_schema(engine)

        if run_doc_gen4:
            await run_doc_gen4_step()

        if parse_docs:
            await run_extract_step(engine)

        if pagerank:
            await run_pagerank_step(engine, pagerank_alpha, pagerank_batch_size)

        if informalize:
            await run_informalize_step(
                engine,
                informalize_model,
                informalize_batch_size,
                informalize_max_concurrent,
                informalize_limit,
            )

        if embeddings:
            await run_embeddings_step(
                engine, embedding_model, embedding_batch_size, embedding_limit
            )

        if index:
            await run_index_step(engine)

        logger.info("Pipeline completed successfully!")

    finally:
        await engine.dispose()


@click.command()
@click.option(
    "--lean-version",
    envvar="LEAN_EXPLORE_LEAN_VERSION",
    help=(
        "Lean version for database (e.g., 4.23.0). "
        "Uses config default if not specified."
    ),
)
@click.option(
    "--run-doc-gen4",
    is_flag=True,
    help="Run doc-gen4 to generate documentation before parsing",
)
@click.option(
    "--parse-docs/--no-parse-docs",
    default=None,
    help="Run doc-gen4 parsing step",
)
@click.option(
    "--pagerank/--no-pagerank",
    default=None,
    help="Run PageRank calculation step",
)
@click.option(
    "--informalize/--no-informalize",
    default=None,
    help="Run informalization step",
)
@click.option(
    "--embeddings/--no-embeddings",
    default=None,
    help="Run embeddings generation step",
)
@click.option(
    "--index/--no-index",
    default=None,
    help="Run FAISS index building step",
)
@click.option(
    "--informalize-model",
    default="google/gemini-2.5-flash",
    help="LLM model for generating informalizations",
)
@click.option(
    "--informalize-max-concurrent",
    type=int,
    default=10,
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
    default="BAAI/bge-base-en-v1.5",
    help="Sentence transformer model for embeddings",
)
@click.option(
    "--embedding-limit",
    type=int,
    default=None,
    help="Limit number of declarations for embeddings (for testing)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    lean_version: str | None,
    run_doc_gen4: bool,
    parse_docs: bool | None,
    pagerank: bool | None,
    informalize: bool | None,
    embeddings: bool | None,
    index: bool | None,
    informalize_model: str,
    informalize_max_concurrent: int,
    informalize_limit: int | None,
    embedding_model: str,
    embedding_limit: int | None,
    verbose: bool,
) -> None:
    """Run the Lean declaration extraction and enrichment pipeline."""
    # Determine if any step flags were explicitly set
    step_flags = [parse_docs, pagerank, informalize, embeddings, index]
    any_step_explicitly_set = any(flag is not None for flag in step_flags)

    # If no steps were explicitly set, run all by default
    # Otherwise, only run explicitly enabled steps (default unset to False)
    if not any_step_explicitly_set:
        parse_docs = pagerank = informalize = embeddings = index = True
    else:
        parse_docs = parse_docs if parse_docs is not None else False
        pagerank = pagerank if pagerank is not None else False
        informalize = informalize if informalize is not None else False
        embeddings = embeddings if embeddings is not None else False
        index = index if index is not None else False

    if lean_version:
        os.environ["LEAN_EXPLORE_LEAN_VERSION"] = lean_version
        importlib.reload(lean_explore.config)
        from lean_explore.config import Config as ReloadedConfig

        database_url = ReloadedConfig.DATABASE_URL
    else:
        database_url = Config.DATABASE_URL

    asyncio.run(
        run_pipeline(
            database_url=database_url,
            run_doc_gen4=run_doc_gen4,
            parse_docs=parse_docs,
            pagerank=pagerank,
            informalize=informalize,
            embeddings=embeddings,
            index=index,
            informalize_model=informalize_model,
            informalize_max_concurrent=informalize_max_concurrent,
            informalize_limit=informalize_limit,
            embedding_model=embedding_model,
            embedding_limit=embedding_limit,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    main()
