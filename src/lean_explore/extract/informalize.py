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


@dataclass
class DeclarationData:
    """Plain data extracted from Declaration ORM object for async processing."""

    id: int
    name: str
    source_text: str
    docstring: str | None
    dependencies: str | None
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
) -> dict[tuple[str, str], str]:
    """Load informalizations from all discovered databases.

    Builds a cache mapping (name, source_text) -> informalization by scanning
    all databases for declarations that have informalizations.

    Args:
        database_files: List of database file paths to scan

    Returns:
        Dictionary mapping (name, source_text) -> informalization
    """
    cache: dict[tuple[str, str], str] = {}

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
                    cache_key = (declaration.name, declaration.source_text)
                    if cache_key not in cache:
                        cache[cache_key] = declaration.informalization

                logger.info(
                    f"Loaded {len(declarations)} informalizations from {db_path}"
                )

            await engine.dispose()

        except Exception as e:
            logger.warning(f"Failed to load cache from {db_path}: {e}")
            continue

    logger.info(f"Total cache size: {len(cache)} unique (name, source_text) pairs")
    return cache


# --- Processing Functions ---


async def _process_one_declaration(
    *,
    declaration_data: DeclarationData,
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_by_name: dict[str, str],
    cache: dict[tuple[str, str], str],
    semaphore: asyncio.Semaphore,
) -> InformalizationResult:
    """Process a single declaration and generate its informalization.

    Args:
        declaration_data: Plain data extracted from Declaration ORM object
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_by_name: Map of declaration names to informalizations
        cache: Map of (name, source_text) to cached informalizations
        semaphore: Concurrency control semaphore

    Returns:
        InformalizationResult with declaration info and generated informalization
    """
    if declaration_data.informalization is not None:
        return InformalizationResult(
            declaration_id=declaration_data.id,
            declaration_name=declaration_data.name,
            informalization=None,
        )

    # Check cross-database cache first
    cache_key = (declaration_data.name, declaration_data.source_text)
    if cache_key in cache:
        return InformalizationResult(
            declaration_id=declaration_data.id,
            declaration_name=declaration_data.name,
            informalization=cache[cache_key],
        )

    async with semaphore:
        dependencies_text = ""
        dependencies = _parse_dependencies(declaration_data.dependencies)
        if dependencies:
            dependency_informalizations = []
            # Limit to first 20 dependencies
            for dependency_name in dependencies[:20]:
                if dependency_name in informalizations_by_name:
                    informal_description = informalizations_by_name[dependency_name]
                    # Truncate description to 256 characters
                    if len(informal_description) > 256:
                        informal_description = informal_description[:253] + "..."
                    dependency_informalizations.append(
                        f"- {dependency_name}: {informal_description}"
                    )

            if dependency_informalizations:
                dependencies_text = "Dependencies:\n" + "\n".join(
                    dependency_informalizations
                )

        prompt = prompt_template.format(
            name=declaration_data.name,
            source_text=declaration_data.source_text,
            docstring=declaration_data.docstring or "No docstring available",
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
                declaration_id=declaration_data.id,
                declaration_name=declaration_data.name,
                informalization=result,
            )

        logger.warning(f"Empty response for declaration {declaration_data.name}")
        return InformalizationResult(
            declaration_id=declaration_data.id,
            declaration_name=declaration_data.name,
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
    cache: dict[tuple[str, str], str],
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
        cache: Map of (name, source_text) to cached informalizations
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

    # Extract data from ORM objects before creating async tasks
    # This avoids SQLAlchemy session issues with concurrent access
    declaration_data_list = [
        DeclarationData(
            id=d.id,
            name=d.name,
            source_text=d.source_text,
            docstring=d.docstring,
            dependencies=d.dependencies,
            informalization=d.informalization,
        )
        for d in layer
    ]

    # Create tasks for all declarations in this layer
    tasks = [
        asyncio.create_task(
            _process_one_declaration(
                declaration_data=data,
                client=client,
                model=model,
                prompt_template=prompt_template,
                informalizations_by_name=informalizations_by_name,
                cache=cache,
                semaphore=semaphore,
            )
        )
        for data in declaration_data_list
    ]

    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro

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
    cache: dict[tuple[str, str], str],
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
        cache: Map of (name, source_text) to cached informalizations
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
        total_task = progress.add_task(f"[cyan]Total ({total:,})", total=total)
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
                cache=cache,
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


async def _apply_cache_to_declarations(
    session: AsyncSession,
    declarations: list[Declaration],
    cache: dict[tuple[str, str], str],
    commit_batch_size: int = 1000,
) -> tuple[int, list[Declaration]]:
    """Apply cached informalizations to declarations.

    This is a fast first pass that applies all cache hits before making any
    API calls, allowing the user to see exactly how many API calls will be needed.

    Args:
        session: Async database session
        declarations: List of declarations to check against cache
        cache: Map of (name, source_text) to cached informalizations
        commit_batch_size: Number of updates to batch before committing

    Returns:
        Tuple of (cache_hits_count, list of declarations still needing API calls)
    """
    from sqlalchemy import text

    # Phase 1: Match all declarations against cache in memory
    updates_to_apply: list[tuple[int, str]] = []
    remaining: list[Declaration] = []

    for declaration in declarations:
        cache_key = (declaration.name, declaration.source_text)
        if cache_key in cache:
            updates_to_apply.append((declaration.id, cache[cache_key]))
        else:
            remaining.append(declaration)

    logger.info(
        f"Cache matching complete: {len(updates_to_apply)} hits, "
        f"{len(remaining)} misses"
    )

    # Phase 2: Apply updates in batches using raw SQL for efficiency
    if updates_to_apply:
        num_updates = len(updates_to_apply)
        total_batches = (num_updates + commit_batch_size - 1) // commit_batch_size
        logger.info(
            f"Applying {len(updates_to_apply)} cached informalizations "
            f"in {total_batches} batches..."
        )
        stmt = text("UPDATE declarations SET informalization = :inf WHERE id = :id")
        for i in range(0, len(updates_to_apply), commit_batch_size):
            batch = updates_to_apply[i : i + commit_batch_size]
            params = [{"id": decl_id, "inf": inf} for decl_id, inf in batch]
            conn = await session.connection()
            await conn.execute(stmt, params)
            await session.commit()
            batch_num = i // commit_batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"Committed batch {batch_num}/{total_batches}")

    return len(updates_to_apply), remaining


async def informalize_declarations(
    search_db_engine: AsyncEngine,
    *,
    model: str = "google/gemini-3-flash-preview",
    commit_batch_size: int = 1000,
    max_concurrent: int = 100,
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

    # Discover and load cache from all existing databases
    logger.info("Discovering existing databases for cache...")
    database_files = _discover_database_files()
    cache = await _load_cache_from_databases(database_files)

    async with AsyncSession(search_db_engine, expire_on_commit=False) as search_session:
        existing_informalizations = await _load_existing_informalizations(
            search_session
        )
        declarations = await _get_declarations_to_process(search_session, limit)

        logger.info(f"Found {len(declarations)} declarations needing informalization")
        if not declarations:
            logger.info("No declarations to process")
            return

        # Phase 1: Apply all cache hits first
        logger.info("Phase 1: Applying cached informalizations...")
        cache_hits, remaining_declarations = await _apply_cache_to_declarations(
            search_session, declarations, cache, commit_batch_size
        )
        logger.info(
            f"Applied {cache_hits} informalizations from cache, "
            f"{len(remaining_declarations)} remaining need API calls"
        )

        if not remaining_declarations:
            logger.info("All declarations served from cache, no API calls needed")
            return

        # Phase 2: Process remaining declarations with API calls
        logger.info("Phase 2: Making API calls for remaining declarations...")
        client = OpenRouterClient()
        semaphore = asyncio.Semaphore(max_concurrent)

        # Reload existing informalizations (now includes cache hits)
        existing_informalizations = await _load_existing_informalizations(
            search_session
        )

        logger.info("Building dependency layers for remaining declarations...")
        layers = _build_dependency_layers(remaining_declarations)
        logger.info(f"Built {len(layers)} dependency layers")

        processed = await _process_layers(
            session=search_session,
            layers=layers,
            client=client,
            model=model,
            prompt_template=prompt_template,
            existing_informalizations=existing_informalizations,
            cache=cache,
            semaphore=semaphore,
            commit_batch_size=commit_batch_size,
        )

        logger.info(
            f"Informalization complete. Processed {processed}/"
            f"{len(remaining_declarations)} remaining declarations via API"
        )
