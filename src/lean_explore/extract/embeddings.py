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

from lean_explore.extract.schemas import Declaration
from lean_explore.util.embedding_client import EmbeddingClient

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
    infos = []
    sources = []
    docs = []

    name_indices = []
    info_indices = []
    source_indices = []
    doc_indices = []

    for i, decl in enumerate(declarations):
        if decl.name_embedding is None:
            names.append(decl.name)
            name_indices.append(i)
        if decl.informalization and decl.informalization_embedding is None:
            infos.append(decl.informalization)
            info_indices.append(i)
        if decl.source_text_embedding is None:
            sources.append(decl.source_text)
            source_indices.append(i)
        if decl.docstring and decl.docstring_embedding is None:
            docs.append(decl.docstring)
            doc_indices.append(i)

    # Generate embeddings
    name_response = await client.embed(names) if names else None
    info_response = await client.embed(infos) if infos else None
    source_response = await client.embed(sources) if sources else None
    doc_response = await client.embed(docs) if docs else None

    # Build updates
    if name_response:
        for idx, emb in zip(name_indices, name_response.embeddings):
            decl_id = declarations[idx].id
            if decl_id not in updates:
                updates[decl_id] = {"id": decl_id}
            updates[decl_id]["name_embedding"] = emb

    if info_response:
        for idx, emb in zip(info_indices, info_response.embeddings):
            decl_id = declarations[idx].id
            if decl_id not in updates:
                updates[decl_id] = {"id": decl_id}
            updates[decl_id]["informalization_embedding"] = emb

    if source_response:
        for idx, emb in zip(source_indices, source_response.embeddings):
            decl_id = declarations[idx].id
            if decl_id not in updates:
                updates[decl_id] = {"id": decl_id}
            updates[decl_id]["source_text_embedding"] = emb

    if doc_response:
        for idx, emb in zip(doc_indices, doc_response.embeddings):
            decl_id = declarations[idx].id
            if decl_id not in updates:
                updates[decl_id] = {"id": decl_id}
            updates[decl_id]["docstring_embedding"] = emb

    # Apply updates
    if updates:
        await session.execute(update(Declaration), list(updates.values()))
        await session.commit()

    responses = [name_response, info_response, source_response, doc_response]
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
