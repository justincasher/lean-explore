"""Generate informal natural language descriptions for Lean declarations.

Reads declarations from the database, generates informal descriptions using
an LLM via OpenRouter, and updates the informalization field.
"""

import asyncio
import json
import logging
from collections import defaultdict
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
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.models import Declaration, InformalizationCache
from lean_explore.util import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class InformalizationResult:
    """Result of processing a single declaration."""

    declaration_id: int
    declaration_name: str
    informalization: str | None


def _parse_dependencies(dependencies: str | list[str] | None) -> list[str]:
    """Parse dependencies field which may be JSON string or list.

    Args:
        dependencies: Dependencies as JSON string, list, or None

    Returns:
        List of dependency names
    """
    if not dependencies:
        return []
    if isinstance(dependencies, str):
        return json.loads(dependencies)
    return dependencies


def _build_dependency_layers(
    declarations: list[Declaration],
) -> list[list[Declaration]]:
    """Build dependency layers where each layer has no dependencies on later layers.

    Returns a list of layers, where layer 0 has no dependencies, layer 1 only
    depends on layer 0, etc. Cycles are broken arbitrarily.
    """
    name_to_declaration = {
        declaration.name: declaration for declaration in declarations
    }

    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for declaration in declarations:
        in_degree[declaration.name] = 0

    for declaration in declarations:
        dependencies = _parse_dependencies(declaration.dependencies)
        for dependency_name in dependencies:
            if dependency_name in name_to_declaration:
                graph[dependency_name].append(declaration.name)
                in_degree[declaration.name] += 1

    # Process declarations layer by layer using Kahn's algorithm
    layers = []
    current_layer = [
        name_to_declaration[name] for name in in_degree if in_degree[name] == 0
    ]

    while current_layer:
        layers.append(current_layer)
        next_layer = []

        for declaration in current_layer:
            for neighbor in graph[declaration.name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_layer.append(name_to_declaration[neighbor])

        current_layer = next_layer

    # If there are nodes with non-zero in-degree, we have cycles
    # Add them as a final layer (cycle is broken by arbitrary order)
    remaining = [name_to_declaration[name] for name in in_degree if in_degree[name] > 0]
    if remaining:
        logger.warning(
            f"Found {len(remaining)} declarations in cycles, adding as final layer"
        )
        layers.append(remaining)

    return layers


async def _load_existing_informalizations(
    session: AsyncSession,
) -> list[InformalizationResult]:
    """Load all existing informalizations from the database."""
    logger.info("Loading existing informalizations...")
    stmt = select(Declaration).where(Declaration.informalization.isnot(None))
    result = await session.execute(stmt)
    declarations = result.scalars().all()
    informalizations = [
        InformalizationResult(
            declaration_id=declaration.id,
            declaration_name=declaration.name,
            informalization=declaration.informalization,
        )
        for declaration in declarations
    ]
    logger.info(f"Loaded {len(informalizations)} existing informalizations")
    return informalizations


async def _load_informalization_cache(
    session: AsyncSession, model: str
) -> dict[str, str]:
    """Load all cached informalizations for the given model.

    Args:
        session: Async database session for informalization cache
        model: LLM model name to filter cache entries

    Returns:
        Dictionary mapping source_text_hash -> informalization
    """
    logger.info(f"Loading informalization cache for model '{model}'...")
    stmt = select(InformalizationCache).where(InformalizationCache.model == model)
    result = await session.execute(stmt)
    cache_entries = result.scalars().all()
    cache_by_hash = {
        entry.source_text_hash: entry.informalization for entry in cache_entries
    }
    logger.info(f"Loaded {len(cache_by_hash)} cached informalizations")
    return cache_by_hash


async def _get_declarations_to_process(
    session: AsyncSession, limit: int | None
) -> list[Declaration]:
    """Query and return declarations that need informalization."""
    stmt = select(Declaration).where(Declaration.informalization.is_(None))
    if limit:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _process_one_declaration(
    *,
    declaration: Declaration,
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_by_name: dict[str, str],
    cache_by_hash: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> InformalizationResult:
    """Process a single declaration and generate its informalization.

    Args:
        declaration: Declaration to process
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_by_name: Map of declaration names to informalizations
        cache_by_hash: Map of source text hashes to cached informalizations
        semaphore: Concurrency control semaphore

    Returns:
        InformalizationResult with declaration info and generated informalization
    """
    if declaration.informalization is not None:
        return InformalizationResult(
            declaration_id=declaration.id,
            declaration_name=declaration.name,
            informalization=None,
        )

    source_hash = InformalizationCache.compute_hash(declaration.source_text)
    if source_hash in cache_by_hash:
        return InformalizationResult(
            declaration_id=declaration.id,
            declaration_name=declaration.name,
            informalization=cache_by_hash[source_hash],
        )

    async with semaphore:
        dependencies_text = ""
        dependencies = _parse_dependencies(declaration.dependencies)
        if dependencies:
            dependency_informalizations = []
            for dependency_name in dependencies:
                if dependency_name in informalizations_by_name:
                    informal_description = informalizations_by_name[dependency_name]
                    dependency_informalizations.append(
                        f"- {dependency_name}: {informal_description}"
                    )

            if dependency_informalizations:
                dependencies_text = "Dependencies:\n" + "\n".join(
                    dependency_informalizations
                )

        prompt = prompt_template.format(
            name=declaration.name,
            source_text=declaration.source_text,
            docstring=declaration.docstring or "No docstring available",
            dependencies=dependencies_text,
        )

        response = await client.generate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        if response.choices and response.choices[0].message.content:
            result = response.choices[0].message.content.strip()
            return InformalizationResult(
                declaration_id=declaration.id,
                declaration_name=declaration.name,
                informalization=result,
            )

        logger.warning(f"Empty response for declaration {declaration.name}")
        return InformalizationResult(
            declaration_id=declaration.id,
            declaration_name=declaration.name,
            informalization=None,
        )


async def _process_layer(
    *,
    session: AsyncSession,
    informal_session: AsyncSession,
    layer: list[Declaration],
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_by_name: dict[str, str],
    cache_by_hash: dict[str, str],
    semaphore: asyncio.Semaphore,
    progress,
    task,
    commit_batch_size: int,
) -> int:
    """Process a single dependency layer.

    Args:
        session: Async database session for search database
        informal_session: Async database session for informalization cache
        layer: List of declarations in this layer
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_by_name: Map of declaration names to informalizations
        cache_by_hash: Map of source text hashes to cached informalizations
        semaphore: Concurrency control semaphore
        progress: Rich progress bar
        task: Progress task ID
        commit_batch_size: Number of updates to batch before committing

    Returns:
        Number of declarations processed in this layer
    """
    processed = 0
    pending_updates = []
    pending_cache_inserts = []

    # Process all declarations in this layer in parallel (controlled by semaphore)
    tasks = [
        _process_one_declaration(
            declaration=declaration,
            client=client,
            model=model,
            prompt_template=prompt_template,
            informalizations_by_name=informalizations_by_name,
            cache_by_hash=cache_by_hash,
            semaphore=semaphore,
        )
        for declaration in layer
    ]
    results = await asyncio.gather(*tasks)

    for declaration, result in zip(layer, results):
        if result.informalization:
            pending_updates.append(
                {
                    "id": result.declaration_id,
                    "informalization": result.informalization,
                }
            )
            informalizations_by_name[result.declaration_name] = result.informalization

            cache_entry = InformalizationCache.create(
                source_text=declaration.source_text,
                informalization=result.informalization,
                model=model,
            )
            pending_cache_inserts.append(cache_entry)

            processed += 1

        progress.update(task, advance=1)

        if len(pending_updates) >= commit_batch_size:
            await session.execute(update(Declaration), pending_updates)
            await session.commit()
            logger.info(f"Committed batch of {len(pending_updates)} updates")
            pending_updates.clear()

            informal_session.add_all(pending_cache_inserts)
            await informal_session.commit()
            logger.info(
                f"Committed batch of {len(pending_cache_inserts)} cache entries"
            )
            pending_cache_inserts.clear()

    if pending_updates:
        await session.execute(update(Declaration), pending_updates)
        await session.commit()
        logger.info(f"Committed batch of {len(pending_updates)} updates")

    if pending_cache_inserts:
        informal_session.add_all(pending_cache_inserts)
        await informal_session.commit()
        logger.info(
            f"Committed batch of {len(pending_cache_inserts)} cache entries"
        )

    return processed


async def _process_layers(
    *,
    session: AsyncSession,
    informal_session: AsyncSession,
    layers: list[list[Declaration]],
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    existing_informalizations: list[InformalizationResult],
    cache_by_hash: dict[str, str],
    semaphore: asyncio.Semaphore,
    commit_batch_size: int,
) -> int:
    """Process declarations layer by layer with progress tracking.

    Args:
        session: Async database session for search database
        informal_session: Async database session for informalization cache
        layers: List of dependency layers to process
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        existing_informalizations: List of existing informalizations
        cache_by_hash: Map of source text hashes to cached informalizations
        semaphore: Concurrency control semaphore
        commit_batch_size: Number of updates to batch before committing to database

    Returns:
        Number of declarations processed
    """
    total = sum(len(layer) for layer in layers)
    processed = 0

    informalizations_by_name = {
        inf.declaration_name: inf.informalization
        for inf in existing_informalizations
        if inf.informalization is not None
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Informalizing declarations", total=total)

        for layer_num, layer in enumerate(layers):
            logger.info(
                f"Processing layer {layer_num + 1}/{len(layers)} "
                f"({len(layer)} declarations)"
            )
            layer_processed = await _process_layer(
                session=session,
                informal_session=informal_session,
                layer=layer,
                client=client,
                model=model,
                prompt_template=prompt_template,
                informalizations_by_name=informalizations_by_name,
                cache_by_hash=cache_by_hash,
                semaphore=semaphore,
                progress=progress,
                task=task,
                commit_batch_size=commit_batch_size,
            )
            processed += layer_processed
            logger.info(
                f"Completed layer {layer_num + 1}: "
                f"{layer_processed}/{len(layer)} declarations informalized"
            )

    return processed


async def informalize_declarations(
    search_db_engine: AsyncEngine,
    informal_db_engine: AsyncEngine,
    *,
    model: str = "google/gemini-2.5-flash",
    commit_batch_size: int = 1000,
    max_concurrent: int = 10,
    limit: int | None = None,
) -> None:
    """Generate informalizations for declarations missing them.

    Args:
        search_db_engine: Async database engine for search database (Declaration table)
        informal_db_engine: Async database engine for informalization cache
        model: LLM model to use for generation
        commit_batch_size: Number of updates to batch before committing to database
        max_concurrent: Maximum number of concurrent LLM API calls
        limit: Maximum number of declarations to process (None for all)
    """
    prompt_template = (Path(__file__).parent / "prompt.txt").read_text()
    logger.info("Starting informalization process...")
    logger.info(
        f"Model: {model}, Max concurrent: {max_concurrent}, "
        f"Commit batch size: {commit_batch_size}"
    )

    client = OpenRouterClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    async with AsyncSession(search_db_engine) as search_session:
        async with AsyncSession(informal_db_engine) as informal_session:
            existing_informalizations = await _load_existing_informalizations(
                search_session
            )
            cache_by_hash = await _load_informalization_cache(
                informal_session, model
            )
            declarations = await _get_declarations_to_process(
                search_session, limit
            )

            logger.info(
                f"Found {len(declarations)} declarations needing informalization"
            )
            if not declarations:
                logger.info("No declarations to process")
                return

            logger.info("Building dependency layers...")
            layers = _build_dependency_layers(declarations)
            logger.info(f"Built {len(layers)} dependency layers")

            processed = await _process_layers(
                session=search_session,
                informal_session=informal_session,
                layers=layers,
                client=client,
                model=model,
                prompt_template=prompt_template,
                existing_informalizations=existing_informalizations,
                cache_by_hash=cache_by_hash,
                semaphore=semaphore,
                commit_batch_size=commit_batch_size,
            )

            logger.info(
                f"Informalization complete. Processed {processed}/"
                f"{sum(len(layer) for layer in layers)} declarations"
            )
