"""Pipeline orchestration for Lean declaration extraction and enrichment.

This module provides functions to coordinate the complete data extraction pipeline:
1. Extract declarations from doc-gen4 output
2. Calculate PageRank scores based on dependencies
3. Generate informal natural language descriptions
4. Generate vector embeddings for semantic search
"""

import asyncio
import importlib
import logging
import os

import click
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import lean_explore.config
from lean_explore.config import Config
from lean_explore.extract.doc_parser import extract_declarations
from lean_explore.extract.embeddings import generate_embeddings
from lean_explore.extract.informalize import informalize_declarations
from lean_explore.extract.pagerank import calculate_pagerank
from lean_explore.models import Base
from lean_explore.util import setup_logging

logger = logging.getLogger(__name__)


async def create_database_schema(engine: AsyncEngine) -> None:
    """Create database tables if they don't exist.

    Args:
        engine: SQLAlchemy async engine instance.
    """
    logger.info("Creating database schema...")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database schema created successfully")


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


async def run_pipeline(
    database_url: str,
    steps: str = "all",
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
        database_url: PostgreSQL database URL (e.g., postgresql+asyncpg://user:pass@host/db)
        steps: Steps to run - 'all', 'extract', 'pagerank', 'informalize', 'embeddings',
            or comma-separated list
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
    step_list = [s.strip().lower() for s in steps.split(",")]
    if "all" in step_list or "informalize" in step_list:
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.error(
                "OPENROUTER_API_KEY environment variable is required for "
                "informalization"
            )
            raise RuntimeError("OPENROUTER_API_KEY not set")

    logger.info("Starting Lean Explore extraction pipeline")
    logger.info(f"Database URL: {database_url}")
    logger.info(f"Steps to run: {steps}")

    engine = create_async_engine(database_url, echo=verbose)

    try:
        await create_database_schema(engine)

        run_all = "all" in step_list
        run_extract = run_all or "extract" in step_list
        run_pagerank = run_all or "pagerank" in step_list
        run_informalize = run_all or "informalize" in step_list
        run_embeddings = run_all or "embeddings" in step_list

        if run_extract:
            await run_extract_step(engine)

        if run_pagerank:
            await run_pagerank_step(engine, pagerank_alpha, pagerank_batch_size)

        if run_informalize:
            await run_informalize_step(
                engine,
                informalize_model,
                informalize_batch_size,
                informalize_max_concurrent,
                informalize_limit,
            )

        if run_embeddings:
            await run_embeddings_step(
                engine, embedding_model, embedding_batch_size, embedding_limit
            )

        logger.info("Pipeline completed successfully!")

    except Exception:
        logger.exception("Pipeline failed with error")
        raise

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
    "--steps",
    default="all",
    help=(
        "Steps to run: 'all', 'extract', 'pagerank', 'informalize', "
        "'embeddings', or comma-separated list"
    ),
)
@click.option(
    "--informalize-limit",
    type=int,
    default=None,
    help="Limit number of declarations to informalize (for testing)",
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
    steps: str,
    informalize_limit: int | None,
    embedding_limit: int | None,
    verbose: bool,
) -> None:
    """Run the Lean declaration extraction and enrichment pipeline."""
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
            steps=steps,
            informalize_limit=informalize_limit,
            embedding_limit=embedding_limit,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    main()
