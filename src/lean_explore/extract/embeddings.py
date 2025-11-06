"""Generate embeddings for Lean declarations.

Reads declarations from the database and generates embeddings for:
- name
- informalization (if available)
- source_text
- docstring (if available)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from lean_explore.config import Config
from lean_explore.models import Declaration
from lean_explore.util import EmbeddingClient

logger = logging.getLogger(__name__)


# --- Data Classes ---


@dataclass
class EmbeddingCaches:
    """Container for all embedding caches."""

    by_name: dict[str, list[float]]
    by_informalization: dict[str, list[float]]
    by_source_text: dict[str, list[float]]
    by_docstring: dict[str, list[float]]


# --- Cross-Database Cache Loading ---


def _discover_database_files() -> list[Path]:
    """Discover all lean_explore.db files in data/ and cache/ directories.

    Returns:
        List of paths to discovered database files
    """
    database_files = []

    # Search in data directory
    data_dir = Config.DATA_DIRECTORY
    if data_dir.exists():
        database_files.extend(data_dir.rglob("lean_explore.db"))

    # Search in cache directory
    cache_dir = Config.CACHE_DIRECTORY
    if cache_dir.exists():
        database_files.extend(cache_dir.rglob("lean_explore.db"))

    logger.info(f"Discovered {len(database_files)} database files")
    return database_files


async def _load_embedding_caches(database_files: list[Path]) -> EmbeddingCaches:
    """Load embeddings from all discovered databases.

    Builds four caches mapping text content to embeddings by scanning
    all databases for declarations that have embeddings.

    Args:
        database_files: List of database file paths to scan

    Returns:
        EmbeddingCaches with all four cache dictionaries populated
    """
    cache_by_name = {}
    cache_by_informalization = {}
    cache_by_source_text = {}
    cache_by_docstring = {}

    for db_path in database_files:
        db_url = f"sqlite+aiosqlite:///{db_path}"
        logger.info(f"Loading embedding cache from {db_path}")

        try:
            engine = create_async_engine(db_url)
            async with AsyncSession(engine) as session:
                # Load all declarations that have at least one embedding
                stmt = select(Declaration).where(
                    (Declaration.name_embedding.isnot(None))
                    | (Declaration.informalization_embedding.isnot(None))
                    | (Declaration.source_text_embedding.isnot(None))
                    | (Declaration.docstring_embedding.isnot(None))
                )
                result = await session.execute(stmt)
                declarations = result.scalars().all()

                for declaration in declarations:
                    # Cache name embedding
                    if (
                        declaration.name_embedding is not None
                        and declaration.name not in cache_by_name
                    ):
                        cache_by_name[declaration.name] = declaration.name_embedding

                    # Cache informalization embedding
                    if (
                        declaration.informalization is not None
                        and declaration.informalization_embedding is not None
                        and declaration.informalization not in cache_by_informalization
                    ):
                        cache_by_informalization[declaration.informalization] = (
                            declaration.informalization_embedding
                        )

                    # Cache source_text embedding
                    if (
                        declaration.source_text_embedding is not None
                        and declaration.source_text not in cache_by_source_text
                    ):
                        cache_by_source_text[declaration.source_text] = (
                            declaration.source_text_embedding
                        )

                    # Cache docstring embedding
                    if (
                        declaration.docstring is not None
                        and declaration.docstring_embedding is not None
                        and declaration.docstring not in cache_by_docstring
                    ):
                        cache_by_docstring[declaration.docstring] = (
                            declaration.docstring_embedding
                        )

                logger.info(f"Loaded {len(declarations)} declarations from {db_path}")

            await engine.dispose()

        except Exception as e:
            logger.warning(f"Failed to load embedding cache from {db_path}: {e}")
            continue

    logger.info(
        f"Total cache sizes - name: {len(cache_by_name)}, "
        f"informalization: {len(cache_by_informalization)}, "
        f"source_text: {len(cache_by_source_text)}, "
        f"docstring: {len(cache_by_docstring)}"
    )

    return EmbeddingCaches(
        by_name=cache_by_name,
        by_informalization=cache_by_informalization,
        by_source_text=cache_by_source_text,
        by_docstring=cache_by_docstring,
    )


async def _get_declarations_needing_embeddings(
    session: AsyncSession, limit: int | None
) -> list[Declaration]:
    """Get declarations that need at least one embedding.

    Args:
        session: Async database session
        limit: Maximum number of declarations to retrieve (None for all)

    Returns:
        List of declarations needing embeddings
    """
    stmt = select(Declaration).where(
        (Declaration.name_embedding.is_(None))
        | (Declaration.informalization_embedding.is_(None))
        | (Declaration.source_text_embedding.is_(None))
        | (Declaration.docstring_embedding.is_(None))
    )
    if limit:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _process_batch(
    session: AsyncSession,
    declarations: list[Declaration],
    client: EmbeddingClient,
    caches: EmbeddingCaches,
) -> int:
    """Process a batch of declarations and generate embeddings.

    Args:
        session: Async database session
        declarations: List of declarations to process
        client: Embedding client for generating embeddings
        caches: Embedding caches from cross-database loading

    Returns:
        Number of embeddings generated
    """
    all_texts = []
    text_metadata = []
    cached_count = 0

    for declaration in declarations:
        # Check name embedding
        if declaration.name_embedding is None:
            if declaration.name in caches.by_name:
                declaration.name_embedding = caches.by_name[declaration.name]
                cached_count += 1
            else:
                all_texts.append(declaration.name)
                text_metadata.append((declaration, "name"))

        # Check informalization embedding
        if (
            declaration.informalization
            and declaration.informalization_embedding is None
        ):
            if declaration.informalization in caches.by_informalization:
                declaration.informalization_embedding = caches.by_informalization[
                    declaration.informalization
                ]
                cached_count += 1
            else:
                all_texts.append(declaration.informalization)
                text_metadata.append((declaration, "informalization"))

        # Check source_text embedding
        if declaration.source_text_embedding is None:
            if declaration.source_text in caches.by_source_text:
                declaration.source_text_embedding = caches.by_source_text[
                    declaration.source_text
                ]
                cached_count += 1
            else:
                all_texts.append(declaration.source_text)
                text_metadata.append((declaration, "source_text"))

        # Check docstring embedding
        if declaration.docstring and declaration.docstring_embedding is None:
            if declaration.docstring in caches.by_docstring:
                declaration.docstring_embedding = caches.by_docstring[
                    declaration.docstring
                ]
                cached_count += 1
            else:
                all_texts.append(declaration.docstring)
                text_metadata.append((declaration, "docstring"))

    # Generate embeddings for texts not found in cache
    if all_texts:
        response = await client.embed(all_texts)

        for (declaration, field_name), embedding in zip(
            text_metadata, response.embeddings
        ):
            if field_name == "name":
                declaration.name_embedding = embedding
            elif field_name == "informalization":
                declaration.informalization_embedding = embedding
            elif field_name == "source_text":
                declaration.source_text_embedding = embedding
            elif field_name == "docstring":
                declaration.docstring_embedding = embedding

    await session.commit()

    if cached_count > 0:
        logger.debug(f"Reused {cached_count} embeddings from cache")

    return len(all_texts)


async def generate_embeddings(
    engine: AsyncEngine,
    model_name: str,
    batch_size: int = 250,
    limit: int | None = None,
) -> None:
    """Generate embeddings for all declarations.

    Args:
        engine: Async database engine
        model_name: Name of the sentence transformer model to use
        batch_size: Number of declarations to process in each batch (default 250)
        limit: Maximum number of declarations to process (None for all)
    """
    client = EmbeddingClient(model_name=model_name)
    logger.info(
        f"Starting embedding generation with {client.model_name} on {client.device}"
    )

    # Discover and load embedding caches from all existing databases
    logger.info("Discovering existing databases for embedding cache...")
    database_files = _discover_database_files()
    caches = await _load_embedding_caches(database_files)

    async with AsyncSession(engine) as session:
        declarations = await _get_declarations_needing_embeddings(session, limit)
        total = len(declarations)
        logger.info(f"Found {total} declarations needing embeddings")

        if not declarations:
            logger.info("No declarations to process")
            return

        total_embeddings = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Generating embeddings", total=total)

            for i in range(0, total, batch_size):
                batch = declarations[i : i + batch_size]
                count = await _process_batch(session, batch, client, caches)
                total_embeddings += count
                progress.update(task, advance=len(batch))

        logger.info(f"Generated {total_embeddings} embeddings for {total} declarations")
