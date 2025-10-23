"""Pipeline orchestration for Lean declaration extraction and enrichment.

This module provides functions to coordinate the complete data extraction pipeline:
1. Extract declarations from doc-gen4 output
2. Calculate PageRank scores based on dependencies
3. Generate informal natural language descriptions
4. Generate vector embeddings for semantic search
"""

import asyncio
import logging
import os

import click
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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
    informalize_model: str = "anthropic/claude-3.5-sonnet",
    informalize_batch_size: int = 50,
    informalize_max_concurrent: int = 10,
    informalize_limit: int | None = None,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    embedding_batch_size: int = 50,
    embedding_limit: int | None = None,
    skip_schema_creation: bool = False,
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
        informalize_batch_size: Batch size for informalization
        informalize_max_concurrent: Maximum concurrent informalization requests
        informalize_limit: Limit number of declarations to informalize
        embedding_model: Sentence transformer model for embeddings
        embedding_batch_size: Batch size for embedding generation
        embedding_limit: Limit number of declarations for embeddings
        skip_schema_creation: Skip database schema creation
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

    # Create database engine
    engine = create_async_engine(database_url, echo=verbose)

    try:
        # Create schema if needed
        if not skip_schema_creation:
            await create_database_schema(engine)

        # Determine which steps to run
        run_all = "all" in step_list
        run_extract = run_all or "extract" in step_list
        run_pagerank = run_all or "pagerank" in step_list
        run_informalize = run_all or "informalize" in step_list
        run_embeddings = run_all or "embeddings" in step_list

        # Run extraction step
        if run_extract:
            await run_extract_step(engine)

        # Run PageRank step
        if run_pagerank:
            await run_pagerank_step(engine, pagerank_alpha, pagerank_batch_size)

        # Run informalization step
        if run_informalize:
            await run_informalize_step(
                engine,
                informalize_model,
                informalize_batch_size,
                informalize_max_concurrent,
                informalize_limit,
            )

        # Run embeddings step
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
    "--database-url",
    required=True,
    envvar="DATABASE_URL",
    help="PostgreSQL database URL (e.g., postgresql+asyncpg://user:pass@host/db)",
)
@click.option(
    "--steps",
    default="all",
    help=(
        "Steps to run: 'all', 'extract', 'pagerank', 'informalize', "
        "'embeddings', or comma-separated list"
    ),
)
@click.option("--pagerank-alpha", default=0.85, help="PageRank damping parameter")
@click.option(
    "--pagerank-batch-size", default=1000, help="Batch size for PageRank updates"
)
@click.option(
    "--informalize-model",
    default="google/gemini-2.5-flash",
    help="LLM model for generating informalizations",
)
@click.option(
    "--informalize-commit-batch-size",
    default=1000,
    help="Commit batch size for informalization",
)
@click.option(
    "--informalize-max-concurrent",
    default=10,
    help="Maximum concurrent informalization requests",
)
@click.option(
    "--informalize-limit",
    type=int,
    default=None,
    help="Limit number of declarations to informalize",
)
@click.option(
    "--embedding-model",
    default="BAAI/bge-base-en-v1.5",
    help="Sentence transformer model for embeddings",
)
@click.option(
    "--embedding-batch-size",
    default=250,
    help="Batch size for embedding generation",
)
@click.option(
    "--embedding-limit",
    type=int,
    default=None,
    help="Limit number of declarations for embeddings",
)
@click.option(
    "--skip-schema-creation",
    is_flag=True,
    help="Skip database schema creation",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    database_url: str,
    steps: str,
    pagerank_alpha: float,
    pagerank_batch_size: int,
    informalize_model: str,
    informalize_commit_batch_size: int,
    informalize_max_concurrent: int,
    informalize_limit: int | None,
    embedding_model: str,
    embedding_batch_size: int,
    embedding_limit: int | None,
    skip_schema_creation: bool,
    verbose: bool,
) -> None:
    """Run the Lean declaration extraction and enrichment pipeline."""
    asyncio.run(
        run_pipeline(
            database_url=database_url,
            steps=steps,
            pagerank_alpha=pagerank_alpha,
            pagerank_batch_size=pagerank_batch_size,
            informalize_model=informalize_model,
            informalize_batch_size=informalize_commit_batch_size,
            informalize_max_concurrent=informalize_max_concurrent,
            informalize_limit=informalize_limit,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_limit=embedding_limit,
            skip_schema_creation=skip_schema_creation,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    main()
