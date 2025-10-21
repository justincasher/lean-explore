"""Generate embeddings for Lean declarations.

Reads declarations from the database and generates embeddings for:
- name
- informalization (if available)
- source_text
- docstring (if available)
"""

import logging

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.models import Declaration
from lean_explore.util import EmbeddingClient

logger = logging.getLogger(__name__)


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
) -> int:
    """Process a batch of declarations and generate embeddings.

    Args:
        session: Async database session
        declarations: List of declarations to process
        client: Embedding client for generating embeddings

    Returns:
        Number of embeddings generated
    """
    # Collect all texts that need embeddings
    all_texts = []
    # Track which declaration and field each text belongs to
    text_metadata = []

    for declaration in declarations:
        if declaration.name_embedding is None:
            all_texts.append(declaration.name)
            text_metadata.append((declaration, "name"))

        if (
            declaration.informalization
            and declaration.informalization_embedding is None
        ):
            all_texts.append(declaration.informalization)
            text_metadata.append((declaration, "informalization"))

        if declaration.source_text_embedding is None:
            all_texts.append(declaration.source_text)
            text_metadata.append((declaration, "source_text"))

        if declaration.docstring and declaration.docstring_embedding is None:
            all_texts.append(declaration.docstring)
            text_metadata.append((declaration, "docstring"))

    if not all_texts:
        return 0

    # Generate all embeddings in one call
    response = await client.embed(all_texts)

    # Assign embeddings back to declarations
    for (declaration, field_name), embedding in zip(text_metadata, response.embeddings):
        if field_name == "name":
            declaration.name_embedding = embedding
        elif field_name == "informalization":
            declaration.informalization_embedding = embedding
        elif field_name == "source_text":
            declaration.source_text_embedding = embedding
        elif field_name == "docstring":
            declaration.docstring_embedding = embedding

    # Save to database
    await session.commit()

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
                count = await _process_batch(session, batch, client)
                total_embeddings += count
                progress.update(task, advance=len(batch))

        logger.info(
            f"Generated {total_embeddings} embeddings for {total} declarations"
        )
