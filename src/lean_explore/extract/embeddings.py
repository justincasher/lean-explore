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
from sqlalchemy import select, update
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
    updates = {}

    # Collect texts that need embeddings
    names = []
    informalizations = []
    sources = []
    docstrings = []

    name_indices = []
    informalization_indices = []
    source_indices = []
    docstring_indices = []

    for index, declaration in enumerate(declarations):
        if declaration.name_embedding is None:
            names.append(declaration.name)
            name_indices.append(index)
        if (
            declaration.informalization
            and declaration.informalization_embedding is None
        ):
            informalizations.append(declaration.informalization)
            informalization_indices.append(index)
        if declaration.source_text_embedding is None:
            sources.append(declaration.source_text)
            source_indices.append(index)
        if declaration.docstring and declaration.docstring_embedding is None:
            docstrings.append(declaration.docstring)
            docstring_indices.append(index)

    # Generate embeddings
    name_response = (
        await client.embed(names) if names else None
    )
    informalization_response = (
        await client.embed(informalizations) if informalizations else None
    )
    source_response = (
        await client.embed(sources) if sources else None
    )
    docstring_response = (
        await client.embed(docstrings) if docstrings else None
    )

    # Build updates
    if name_response:
        for index, embedding in zip(name_indices, name_response.embeddings):
            declaration_id = declarations[index].id
            if declaration_id not in updates:
                updates[declaration_id] = {"id": declaration_id}
            updates[declaration_id]["name_embedding"] = embedding

    if informalization_response:
        for index, embedding in zip(
            informalization_indices,
            informalization_response.embeddings,
        ):
            declaration_id = declarations[index].id
            if declaration_id not in updates:
                updates[declaration_id] = {"id": declaration_id}
            updates[declaration_id]["informalization_embedding"] = embedding

    if source_response:
        for index, embedding in zip(source_indices, source_response.embeddings):
            declaration_id = declarations[index].id
            if declaration_id not in updates:
                updates[declaration_id] = {"id": declaration_id}
            updates[declaration_id]["source_text_embedding"] = embedding

    if docstring_response:
        for index, embedding in zip(docstring_indices, docstring_response.embeddings):
            declaration_id = declarations[index].id
            if declaration_id not in updates:
                updates[declaration_id] = {"id": declaration_id}
            updates[declaration_id]["docstring_embedding"] = embedding

    # Apply updates
    if updates:
        await session.execute(update(Declaration), list(updates.values()))
        await session.commit()

    responses = [
        name_response,
        informalization_response,
        source_response,
        docstring_response,
    ]
    total = sum(len(response.embeddings) for response in responses if response)
    return total


async def generate_embeddings(
    engine: AsyncEngine,
    model_name: str,
    batch_size: int = 50,
    limit: int | None = None,
) -> None:
    """Generate embeddings for all declarations.

    Args:
        engine: Async database engine
        model_name: Name of the sentence transformer model to use
        batch_size: Number of declarations to process in each batch
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

        logger.info(f"Generated {total_embeddings} embeddings for {total} declarations")
