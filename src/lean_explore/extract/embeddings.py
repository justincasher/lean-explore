"""Generate embeddings for Lean declarations.

Reads declarations from the database and generates informalization embeddings
for semantic search.
"""

import logging
import sqlite3
import struct
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.config import Config
from lean_explore.models import Declaration
from lean_explore.util import EmbeddingClient

logger = logging.getLogger(__name__)


class RateColumn(ProgressColumn):
    """Custom column showing embeddings per second over a rolling window."""

    def __init__(self, window_seconds: int = 300):
        """Initialize rate column.

        Args:
            window_seconds: Rolling window size in seconds for rate calculation
        """
        super().__init__()
        self.window_seconds = window_seconds
        self.history: deque[tuple[float, int]] = deque()
        self.total_count = 0

    def add_count(self, count: int) -> None:
        """Add embedding count with timestamp."""
        now = time.time()
        self.history.append((now, count))
        self.total_count += count
        # Remove old entries outside window
        cutoff = now - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def render(self, task: Task) -> Text:
        """Render the rate column."""
        if not self.history:
            return Text("-- emb/s", style="cyan")

        now = time.time()
        cutoff = now - self.window_seconds
        # Sum counts within window
        window_count = sum(c for t, c in self.history if t >= cutoff)
        # Calculate elapsed time in window
        if self.history:
            oldest_in_window = max(self.history[0][0], cutoff)
            elapsed = now - oldest_in_window
            if elapsed > 0:
                rate = window_count / elapsed
                return Text(f"{rate:.1f} emb/s", style="cyan")

        return Text("-- emb/s", style="cyan")


# --- Data Classes ---


@dataclass
class EmbeddingCaches:
    """Container for embedding caches.

    Stores embeddings as raw bytes for efficiency. Use _deserialize_embedding()
    to convert to list[float] when actually needed.
    """

    by_informalization: dict[str, bytes]


def _deserialize_embedding(data: bytes) -> list[float]:
    """Convert raw binary embedding to list[float].

    Args:
        data: Binary embedding data (float32 packed)

    Returns:
        List of float values
    """
    num_floats = len(data) // 4
    return list(struct.unpack(f"{num_floats}f", data))


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


def _load_embedding_caches(database_files: list[Path]) -> EmbeddingCaches:
    """Load embeddings from all discovered databases.

    Builds a cache mapping informalization text to raw embedding bytes by scanning
    all databases for declarations that have embeddings.

    Uses sync sqlite3 directly to avoid SQLAlchemy ORM overhead and TypeDecorator
    deserialization. Embeddings are stored as raw bytes and only deserialized
    when actually used.

    Args:
        database_files: List of database file paths to scan

    Returns:
        EmbeddingCaches with cache dictionary populated (as bytes)
    """
    cache_by_informalization: dict[str, bytes] = {}

    for db_path in database_files:
        logger.info(f"Loading embedding cache from {db_path}")

        try:
            connection = sqlite3.connect(db_path)
            cursor = connection.execute(
                """
                SELECT informalization, informalization_embedding
                FROM declarations
                WHERE informalization_embedding IS NOT NULL
                """
            )

            count = 0
            for row in cursor:
                count += 1
                (informalization, informalization_embedding) = row

                # Cache informalization embedding
                if (
                    informalization is not None
                    and informalization not in cache_by_informalization
                ):
                    cache_by_informalization[informalization] = (
                        informalization_embedding
                    )

            connection.close()
            logger.info(f"Loaded {count} declarations from {db_path}")

        except Exception as e:
            logger.warning(f"Failed to load embedding cache from {db_path}: {e}")
            continue

    logger.info(f"Total cache size - informalization: {len(cache_by_informalization)}")

    return EmbeddingCaches(by_informalization=cache_by_informalization)


async def _get_declarations_needing_embeddings(
    session: AsyncSession, limit: int | None
) -> list[Declaration]:
    """Get declarations that need informalization embeddings.

    Only returns declarations that have an informalization but no embedding yet.

    Args:
        session: Async database session
        limit: Maximum number of declarations to retrieve (None for all)

    Returns:
        List of declarations needing embeddings
    """
    stmt = select(Declaration).where(
        Declaration.informalization.isnot(None),
        Declaration.informalization_embedding.is_(None),
    )
    if limit:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _apply_cache_to_declarations(
    session: AsyncSession,
    declarations: list[Declaration],
    caches: EmbeddingCaches,
    commit_batch_size: int = 1000,
) -> tuple[int, list[Declaration]]:
    """Apply cached embeddings to declarations.

    This is a fast first pass that applies all cache hits before generating
    new embeddings, allowing the user to see exactly how many need generation.

    Args:
        session: Async database session
        declarations: List of declarations to check against cache
        caches: Embedding caches from cross-database loading
        commit_batch_size: Number of updates to batch before committing

    Returns:
        Tuple of (cache_hits_count, list of declarations still needing generation)
    """
    cache_hits = 0
    remaining: list[Declaration] = []
    batch_count = 0

    for declaration in declarations:
        if not declaration.informalization:
            continue

        if declaration.informalization in caches.by_informalization:
            declaration.informalization_embedding = _deserialize_embedding(
                caches.by_informalization[declaration.informalization]
            )
            cache_hits += 1
            batch_count += 1

            if batch_count >= commit_batch_size:
                await session.commit()
                batch_count = 0
        else:
            remaining.append(declaration)

    if batch_count > 0:
        await session.commit()

    return cache_hits, remaining


async def _process_batch(
    session: AsyncSession,
    declarations: list[Declaration],
    client: EmbeddingClient,
) -> int:
    """Process a batch of declarations and generate informalization embeddings.

    Args:
        session: Async database session
        declarations: List of declarations to process (already filtered, no cache)
        client: Embedding client for generating embeddings

    Returns:
        Number of embeddings generated
    """
    texts_to_embed = []
    declarations_to_embed = []

    for declaration in declarations:
        if not declaration.informalization:
            continue
        if declaration.informalization_embedding is not None:
            continue
        texts_to_embed.append(declaration.informalization)
        declarations_to_embed.append(declaration)

    if texts_to_embed:
        response = await client.embed(texts_to_embed)

        for declaration, embedding in zip(declarations_to_embed, response.embeddings):
            declaration.informalization_embedding = embedding

    await session.commit()

    return len(texts_to_embed)


async def generate_embeddings(
    engine: AsyncEngine,
    model_name: str,
    batch_size: int = 128,
    limit: int | None = None,
    max_seq_length: int = 512,
) -> None:
    """Generate embeddings for all declarations.

    Args:
        engine: Async database engine
        model_name: Name of the sentence transformer model to use
        batch_size: Number of declarations to process in each batch (default 250)
        limit: Maximum number of declarations to process (None for all)
        max_seq_length: Maximum sequence length for tokenization (default 512).
            Lower values reduce memory usage but may truncate long texts.
    """
    # Discover and load embedding caches from all existing databases
    logger.info("Discovering existing databases for embedding cache...")
    database_files = _discover_database_files()
    caches = _load_embedding_caches(database_files)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        declarations = await _get_declarations_needing_embeddings(session, limit)
        logger.info(f"Found {len(declarations)} declarations needing embeddings")

        if not declarations:
            logger.info("No declarations to process")
            return

        # Phase 1: Apply all cache hits first
        logger.info("Phase 1: Applying cached embeddings...")
        cache_hits, remaining = await _apply_cache_to_declarations(
            session, declarations, caches
        )
        logger.info(
            f"Applied {cache_hits} embeddings from cache, "
            f"{len(remaining)} remaining need generation"
        )

        if not remaining:
            logger.info("All embeddings served from cache, no generation needed")
            return

        # Phase 2: Generate embeddings for remaining declarations
        logger.info("Phase 2: Generating embeddings for remaining declarations...")
        client = EmbeddingClient(model_name=model_name, max_length=max_seq_length)
        logger.info(f"Using {client.model_name} on {client.device}")

        total = len(remaining)
        total_embeddings = 0
        rate_column = RateColumn(window_seconds=60)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            rate_column,
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Generating embeddings", total=total)

            for i in range(0, total, batch_size):
                batch = remaining[i : i + batch_size]
                count = await _process_batch(session, batch, client)
                total_embeddings += count
                rate_column.add_count(count)
                progress.update(task, advance=len(batch))

        logger.info(
            f"Generated {total_embeddings} new embeddings "
            f"({cache_hits} from cache, {total_embeddings} generated)"
        )
