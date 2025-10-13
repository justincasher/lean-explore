"""Generate informal natural language descriptions for Lean declarations.

Reads declarations from the database, generates informal descriptions using
an LLM via OpenRouter, and updates the informalization field.
"""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from lean_explore.extract.schemas import Declaration
from lean_explore.util.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


def _find_cycles_and_build_order(declarations: list[Declaration]) -> list[Declaration]:
    """Build processing order, breaking cycles by removing edges.

    Returns declarations in topological order where dependencies come first.
    """
    name_to_decl = {decl.name: decl for decl in declarations}

    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for decl in declarations:
        in_degree[decl.name] = 0

    # Build graph
    for decl in declarations:
        if decl.dependencies:
            deps = json.loads(decl.dependencies) if isinstance(decl.dependencies, str) else decl.dependencies
            for dep_name in deps:
                if dep_name in name_to_decl: 
                    graph[dep_name].append(decl.name)
                    in_degree[decl.name] += 1

    # Kahn's algorithm for topological sort (automatically breaks cycles)
    queue = [name for name in in_degree if in_degree[name] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(name_to_decl[current])

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If there are nodes with non-zero in-degree, we have cycles
    # Add them anyway (cycle is broken by arbitrary order)
    remaining = [name_to_decl[name] for name in in_degree if in_degree[name] > 0]
    if remaining:
        logger.warning(f"Found {len(remaining)} declarations in cycles, adding in arbitrary order")
        result.extend(remaining)

    return result


async def _load_existing_informalizations(session: AsyncSession) -> dict[str, str]:
    """Load all existing informalizations from the database."""
    logger.info("Loading existing informalizations...")
    stmt = select(Declaration).where(Declaration.informalization.isnot(None))
    result = await session.execute(stmt)
    decls = result.scalars().all()
    informalizations_map = {decl.name: decl.informalization for decl in decls}
    logger.info(f"Loaded {len(informalizations_map)} existing informalizations")
    return informalizations_map


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
    decl: Declaration,
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_map: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> tuple[int, str, str | None]:
    """Process a single declaration and generate its informalization.

    Args:
        decl: Declaration to process
        client: OpenRouter client
        model: Model name to use
        prompt_template: Prompt template string
        informalizations_map: Map of existing informalizations
        semaphore: Concurrency control semaphore

    Returns:
        Tuple of (declaration_id, declaration_name, informalization)
    """
    if decl.informalization is not None:
        return decl.id, decl.name, None

    async with semaphore:
        try:
            dependencies_text = ""
            if decl.dependencies:
                deps = json.loads(decl.dependencies) if isinstance(decl.dependencies, str) else decl.dependencies
                dep_infos = []
                for dep_name in deps:
                    if dep_name in informalizations_map:
                        dep_infos.append(f"- {dep_name}: {informalizations_map[dep_name]}")

                if dep_infos:
                    dependencies_text = "Dependencies:\n" + "\n".join(dep_infos)

            prompt = prompt_template.format(
                name=decl.name,
                source_text=decl.source_text,
                docstring=decl.docstring or "No docstring available",
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
                return decl.id, decl.name, result

            logger.warning(f"Empty response for declaration {decl.name}")
            return decl.id, decl.name, None

        except Exception as e:
            logger.error(f"Failed to informalize {decl.name}: {e}")
            return decl.id, decl.name, None


async def _process_declarations_in_batches(
    session: AsyncSession,
    declarations: list[Declaration],
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_map: dict[str, str],
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
                    decl, client, model, prompt_template, informalizations_map, semaphore
                )
                for decl in chunk
            ]
            results = await asyncio.gather(*tasks)

            for decl_id, decl_name, informalization in results:
                if informalization:
                    pending_updates.append(
                        {"id": decl_id, "informalization": informalization}
                    )
                    informalizations_map[decl_name] = informalization
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
    model: str = "anthropic/claude-3.5-sonnet",
    batch_size: int = 50,
    max_concurrent: int = 10,
    limit: int | None = None,
) -> None:
    """Generate informalizations for declarations missing them."""
    prompt_template = (Path(__file__).parent / "prompt.txt").read_text()
    logger.info("Starting informalization process...")
    logger.info(f"Model: {model}, Max concurrent: {max_concurrent}, Batch size: {batch_size}")

    client = OpenRouterClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    async with AsyncSession(engine) as session:
        informalizations_map = await _load_existing_informalizations(session)
        declarations = await _get_declarations_to_process(session, limit)

        logger.info(f"Found {len(declarations)} declarations needing informalization")
        if not declarations:
            logger.info("No declarations to process")
            return

        logger.info("Building dependency order and breaking cycles...")
        declarations = _find_cycles_and_build_order(declarations)
        logger.info("Dependency order established")

        processed = await _process_declarations_in_batches(
            session, declarations, client, model, prompt_template,
            informalizations_map, semaphore, batch_size
        )

        logger.info(f"Informalization complete. Processed {processed}/{len(declarations)} declarations")


async def main():
    """Main entry point for running informalization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Example usage - you'll want to make these configurable
    db_url = "postgresql+asyncpg://user:pass@localhost/lean_explore"
    engine = create_async_engine(db_url, echo=False)

    await informalize_declarations(
        engine=engine,
        model="anthropic/claude-3.5-sonnet",
        batch_size=50,
        max_concurrent=10,
        limit=None,  # Set to a number for testing
    )


if __name__ == "__main__":
    asyncio.run(main())
