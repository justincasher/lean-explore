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

from lean_explore.models import Declaration
from lean_explore.util import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class InformalizedDeclaration:
    """A declaration with its informalization."""

    name: str
    informalization: str


def _find_cycles_and_build_order(declarations: list[Declaration]) -> list[Declaration]:
    """Build processing order, breaking cycles by removing edges.

    Returns declarations in topological order where dependencies come first.
    """
    name_to_declaration = {
        declaration.name: declaration for declaration in declarations
    }

    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for declaration in declarations:
        in_degree[declaration.name] = 0

    # Build graph
    for declaration in declarations:
        if declaration.dependencies:
            if isinstance(declaration.dependencies, str):
                dependencies = json.loads(declaration.dependencies)
            else:
                dependencies = declaration.dependencies
            for dependency_name in dependencies:
                if dependency_name in name_to_declaration:
                    graph[dependency_name].append(declaration.name)
                    in_degree[declaration.name] += 1

    # Kahn's algorithm for topological sort (automatically breaks cycles)
    queue = [name for name in in_degree if in_degree[name] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(name_to_declaration[current])

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If there are nodes with non-zero in-degree, we have cycles
    # Add them anyway (cycle is broken by arbitrary order)
    remaining = [name_to_declaration[name] for name in in_degree if in_degree[name] > 0]
    if remaining:
        logger.warning(
            f"Found {len(remaining)} declarations in cycles, adding in arbitrary order"
        )
        result.extend(remaining)

    return result


async def _load_existing_informalizations(
    session: AsyncSession,
) -> list[InformalizedDeclaration]:
    """Load all existing informalizations from the database."""
    logger.info("Loading existing informalizations...")
    stmt = select(Declaration).where(Declaration.informalization.isnot(None))
    result = await session.execute(stmt)
    declarations = result.scalars().all()
    informalizations = [
        InformalizedDeclaration(
            name=declaration.name,
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


async def _process_one_declaration(
    *,
    declaration: Declaration,
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_by_name: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> tuple[int, str, str | None]:
    """Process a single declaration and generate its informalization.

    Args:
        declaration: Declaration to process
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_by_name: Map of declaration names to informalizations
        semaphore: Concurrency control semaphore

    Returns:
        Tuple of (declaration_id, declaration_name, informalization)
    """
    if declaration.informalization is not None:
        return declaration.id, declaration.name, None

    async with semaphore:
        try:
            dependencies_text = ""
            if declaration.dependencies:
                if isinstance(declaration.dependencies, str):
                    dependencies = json.loads(declaration.dependencies)
                else:
                    dependencies = declaration.dependencies
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
                max_tokens=500,
            )

            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                return declaration.id, declaration.name, result

            logger.warning(f"Empty response for declaration {declaration.name}")
            return declaration.id, declaration.name, None

        except Exception as e:
            logger.error(f"Failed to informalize {declaration.name}: {e}")
            return declaration.id, declaration.name, None


async def _process_declarations_in_batches(
    *,
    session: AsyncSession,
    declarations: list[Declaration],
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    existing_informalizations: list[InformalizedDeclaration],
    semaphore: asyncio.Semaphore,
    batch_size: int,
) -> int:
    """Process declarations in batches with progress tracking.

    Returns:
        Number of declarations processed
    """
    total = len(declarations)
    processed = 0
    pending_updates = []

    # Build lookup map once
    informalizations_by_name = {
        inf.name: inf.informalization for inf in existing_informalizations
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Informalizing declarations", total=total)

        for i in range(0, total, batch_size):
            chunk = declarations[i : i + batch_size]
            tasks = [
                _process_one_declaration(
                    declaration=declaration,
                    client=client,
                    model=model,
                    prompt_template=prompt_template,
                    informalizations_by_name=informalizations_by_name,
                    semaphore=semaphore,
                )
                for declaration in chunk
            ]
            results = await asyncio.gather(*tasks)

            for declaration_id, declaration_name, informalization in results:
                if informalization:
                    pending_updates.append(
                        {"id": declaration_id, "informalization": informalization}
                    )
                    # Add to lookup map for subsequent declarations
                    informalizations_by_name[declaration_name] = informalization
                processed += 1
                progress.update(task, advance=1)

            if pending_updates:
                await session.execute(update(Declaration), pending_updates)
                await session.commit()
                logger.info(f"Committed batch of {len(pending_updates)} updates")
                pending_updates.clear()

    return processed


async def informalize_declarations(
    engine: AsyncEngine,
    *,
    model: str = "google/gemini-2.5-flash",
    batch_size: int = 50,
    max_concurrent: int = 10,
    limit: int | None = None,
) -> None:
    """Generate informalizations for declarations missing them."""
    prompt_template = (Path(__file__).parent / "prompt.txt").read_text()
    logger.info("Starting informalization process...")
    logger.info(
        f"Model: {model}, Max concurrent: {max_concurrent}, Batch size: {batch_size}"
    )

    client = OpenRouterClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    async with AsyncSession(engine) as session:
        existing_informalizations = await _load_existing_informalizations(session)
        declarations = await _get_declarations_to_process(session, limit)

        logger.info(f"Found {len(declarations)} declarations needing informalization")
        if not declarations:
            logger.info("No declarations to process")
            return

        logger.info("Building dependency order and breaking cycles...")
        declarations = _find_cycles_and_build_order(declarations)
        logger.info("Dependency order established")

        processed = await _process_declarations_in_batches(
            session=session,
            declarations=declarations,
            client=client,
            model=model,
            prompt_template=prompt_template,
            existing_informalizations=existing_informalizations,
            semaphore=semaphore,
            batch_size=batch_size,
        )

        logger.info(
            f"Informalization complete. Processed {processed}/"
            f"{len(declarations)} declarations"
        )
