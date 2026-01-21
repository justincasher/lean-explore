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
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from lean_explore.config import Config
from lean_explore.models import Declaration
from lean_explore.util import OpenRouterClient

logger = logging.getLogger(__name__)


# --- Data Classes ---


@dataclass
class InformalizationResult:
    """Result of processing a single declaration."""

    declaration_id: int
    declaration_name: str
    informalization: str | None


# --- Utility Functions ---


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


# --- Database Loading ---


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


async def _get_declarations_to_process(
    session: AsyncSession, limit: int | None
) -> list[Declaration]:
    """Query and return declarations that need informalization."""
    stmt = select(Declaration).where(Declaration.informalization.is_(None))
    if limit:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


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


async def _load_cache_from_databases(
    database_files: list[Path],
) -> dict[str, str]:
    """Load informalizations from all discovered databases.

    Builds a cache mapping source_text -> informalization by scanning
    all databases for declarations that have informalizations.

    Args:
        database_files: List of database file paths to scan

    Returns:
        Dictionary mapping source_text -> informalization
    """
    cache_by_source_text = {}

    for db_path in database_files:
        db_url = f"sqlite+aiosqlite:///{db_path}"
        logger.info(f"Loading cache from {db_path}")

        try:
            engine = create_async_engine(db_url)
            async with AsyncSession(engine) as session:
                stmt = select(Declaration).where(
                    Declaration.informalization.isnot(None)
                )
                result = await session.execute(stmt)
                declarations = result.scalars().all()

                for declaration in declarations:
                    # Only cache if we haven't seen this source_text before
                    if declaration.source_text not in cache_by_source_text:
                        cache_by_source_text[declaration.source_text] = (
                            declaration.informalization
                        )

                logger.info(
                    f"Loaded {len(declarations)} informalizations from {db_path}"
                )

            await engine.dispose()

        except Exception as e:
            logger.warning(f"Failed to load cache from {db_path}: {e}")
            continue

    logger.info(f"Total cache size: {len(cache_by_source_text)} unique source texts")
    return cache_by_source_text


# --- Processing Functions ---


async def _process_one_declaration(
    *,
    declaration: Declaration,
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_by_name: dict[str, str],
    cache_by_source_text: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> InformalizationResult:
    """Process a single declaration and generate its informalization.

    Args:
        declaration: Declaration to process
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_by_name: Map of declaration names to informalizations
        cache_by_source_text: Map of source_text to cached informalizations
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

    # Check cross-database cache first
    if declaration.source_text in cache_by_source_text:
        return InformalizationResult(
            declaration_id=declaration.id,
            declaration_name=declaration.name,
            informalization=cache_by_source_text[declaration.source_text],
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
    layer: list[Declaration],
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_by_name: dict[str, str],
    cache_by_source_text: dict[str, str],
    semaphore: asyncio.Semaphore,
    progress,
    total_task,
    batch_task,
    commit_batch_size: int,
) -> int:
    """Process a single dependency layer.

    Args:
        session: Async database session for search database
        layer: List of declarations in this layer
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_by_name: Map of declaration names to informalizations
        cache_by_source_text: Map of source_text to cached informalizations
        semaphore: Concurrency control semaphore
        progress: Rich progress bar
        total_task: Progress task ID for total progress
        batch_task: Progress task ID for batch progress
        commit_batch_size: Number of updates to batch before committing

    Returns:
        Number of declarations processed in this layer
    """
    processed = 0
    pending_updates = []

    # Process all declarations in this layer in parallel (controlled by semaphore)
    tasks = [
        _process_one_declaration(
            declaration=declaration,
            client=client,
            model=model,
            prompt_template=prompt_template,
            informalizations_by_name=informalizations_by_name,
            cache_by_source_text=cache_by_source_text,
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

            processed += 1

        progress.update(total_task, advance=1)
        progress.update(batch_task, advance=1)

        if len(pending_updates) >= commit_batch_size:
            await session.execute(update(Declaration), pending_updates)
            await session.commit()
            logger.info(f"Committed batch of {len(pending_updates)} updates")
            pending_updates.clear()
            # Reset batch progress
            progress.reset(batch_task)

    if pending_updates:
        await session.execute(update(Declaration), pending_updates)
        await session.commit()
        logger.info(f"Committed batch of {len(pending_updates)} updates")
        progress.reset(batch_task)

    return processed


async def _process_layers(
    *,
    session: AsyncSession,
    layers: list[list[Declaration]],
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    existing_informalizations: list[InformalizationResult],
    cache_by_source_text: dict[str, str],
    semaphore: asyncio.Semaphore,
    commit_batch_size: int,
) -> int:
    """Process declarations layer by layer with progress tracking.

    Args:
        session: Async database session for search database
        layers: List of dependency layers to process
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        existing_informalizations: List of existing informalizations
        cache_by_source_text: Map of source_text to cached informalizations
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
        total_task = progress.add_task(
            f"[cyan]Total ({total:,})", total=total
        )
        batch_task = progress.add_task(
            f"[green]Batch ({commit_batch_size:,})", total=commit_batch_size
        )

        for layer_num, layer in enumerate(layers):
            logger.info(
                f"Processing layer {layer_num + 1}/{len(layers)} "
                f"({len(layer)} declarations)"
            )
            layer_processed = await _process_layer(
                session=session,
                layer=layer,
                client=client,
                model=model,
                prompt_template=prompt_template,
                informalizations_by_name=informalizations_by_name,
                cache_by_source_text=cache_by_source_text,
                semaphore=semaphore,
                progress=progress,
                total_task=total_task,
                batch_task=batch_task,
                commit_batch_size=commit_batch_size,
            )
            processed += layer_processed
            logger.info(
                f"Completed layer {layer_num + 1}: "
                f"{layer_processed}/{len(layer)} declarations informalized"
            )

    return processed


# --- Public API ---


async def informalize_declarations(
    search_db_engine: AsyncEngine,
    *,
    model: str = "google/gemini-2.5-flash",
    commit_batch_size: int = 1000,
    max_concurrent: int = 10,
    limit: int | None = None,
) -> None:
    """Generate informalizations for declarations missing them.

    Args:
        search_db_engine: Async database engine for search database (Declaration table)
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

    # Discover and load cache from all existing databases
    logger.info("Discovering existing databases for cache...")
    database_files = _discover_database_files()
    cache_by_source_text = await _load_cache_from_databases(database_files)

    async with AsyncSession(search_db_engine) as search_session:
        existing_informalizations = await _load_existing_informalizations(
            search_session
        )
        declarations = await _get_declarations_to_process(search_session, limit)

        logger.info(f"Found {len(declarations)} declarations needing informalization")
        if not declarations:
            logger.info("No declarations to process")
            return

        logger.info("Building dependency layers...")
        layers = _build_dependency_layers(declarations)
        logger.info(f"Built {len(layers)} dependency layers")

        processed = await _process_layers(
            session=search_session,
            layers=layers,
            client=client,
            model=model,
            prompt_template=prompt_template,
            existing_informalizations=existing_informalizations,
            cache_by_source_text=cache_by_source_text,
            semaphore=semaphore,
            commit_batch_size=commit_batch_size,
        )

        logger.info(
            f"Informalization complete. Processed {processed}/"
            f"{sum(len(layer) for layer in layers)} declarations"
        )
