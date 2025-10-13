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
from lean_explore.util.openrouter import OpenRouterClient

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


async def generate_informalization(
    declaration: Declaration,
    client: OpenRouterClient,
    model: str,
    prompt_template: str,
    informalizations_map: dict[str, str],
) -> str | None:
    """Generate an informal description for a single declaration.

    Args:
        declaration: The Declaration to informalize
        client: OpenRouter client
        model: Model name to use
        prompt_template: Template for the prompt
        informalizations_map: Mapping of declaration names to their informalizations

    Returns:
        Generated description or None if failed
    """
    try:
        # Build dependencies section
        dependencies_text = ""
        if declaration.dependencies:
            deps = json.loads(declaration.dependencies) if isinstance(declaration.dependencies, str) else declaration.dependencies
            dep_infos = []
            for dep_name in deps:
                if dep_name in informalizations_map:
                    dep_infos.append(f"- {dep_name}: {informalizations_map[dep_name]}")

            if dep_infos:
                dependencies_text = "Dependencies:\n" + "\n".join(dep_infos)

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
            return response.choices[0].message.content.strip()

        logger.warning(f"Empty response for declaration {declaration.name}")
        return None

    except Exception as e:
        logger.error(f"Failed to informalize {declaration.name}: {e}")
        return None


async def informalize_declarations(
    engine: AsyncEngine,
    model: str = "anthropic/claude-3.5-sonnet",
    batch_size: int = 50,
    max_concurrent: int = 10,
    prompt_template: str | None = None,
    limit: int | None = None,
) -> None:
    """Generate informalizations for declarations missing them.

    Args:
        engine: Async SQLAlchemy engine
        model: OpenRouter model name
        batch_size: Number of updates to commit at once
        max_concurrent: Maximum concurrent LLM calls
        prompt_template: Custom prompt template (uses default if None)
        limit: Maximum number of declarations to process (for testing)
    """
    if prompt_template is None:
        prompt_file = Path(__file__).parent / "prompt.txt"
        prompt_template = prompt_file.read_text()

    logger.info("Starting informalization process...")
    logger.info(f"Model: {model}")
    logger.info(f"Max concurrent calls: {max_concurrent}")
    logger.info(f"Batch size: {batch_size}")

    client = OpenRouterClient()

    async with AsyncSession(engine) as session:
        # Load existing informalizations to avoid re-processing and to use as context
        logger.info("Loading existing informalizations...")
        existing_stmt = select(Declaration).where(Declaration.informalization.isnot(None))
        existing_result = await session.execute(existing_stmt)
        existing_decls = existing_result.scalars().all()
        informalizations_map = {decl.name: decl.informalization for decl in existing_decls}
        logger.info(f"Loaded {len(informalizations_map)} existing informalizations")

        # Find declarations needing informalization
        stmt = select(Declaration).where(Declaration.informalization.is_(None))
        if limit:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        declarations = list(result.scalars().all())

        total = len(declarations)
        logger.info(f"Found {total} declarations needing informalization")

        if total == 0:
            logger.info("No declarations to process")
            return

        # Sort declarations in topological order (dependencies first)
        logger.info("Building dependency order and breaking cycles...")
        declarations = _find_cycles_and_build_order(declarations)
        logger.info("Dependency order established")

        # Process with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        pending_updates = []
        processed = 0

        async def process_one(decl: Declaration) -> tuple[int, str, str | None]:
            # Double-check that this declaration hasn't been informalized
            # (could happen if processing resumed after partial completion)
            if decl.informalization is not None:
                return decl.id, decl.name, None

            async with semaphore:
                result = await generate_informalization(
                    decl, client, model, prompt_template, informalizations_map
                )
                return decl.id, decl.name, result

        # Create progress bar with time estimates
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Informalizing declarations", total=total)

            try:
                # Process in chunks to avoid memory issues
                for i in range(0, total, batch_size):
                    chunk = declarations[i : i + batch_size]
                    tasks = [process_one(decl) for decl in chunk]
                    results = await asyncio.gather(*tasks)

                    # Collect successful updates and build map for next batch
                    for decl_id, decl_name, informalization in results:
                        if informalization:
                            pending_updates.append(
                                {"id": decl_id, "informalization": informalization}
                            )
                            informalizations_map[decl_name] = informalization
                        processed += 1
                        progress.update(task, advance=1)

                    # Commit batch
                    if pending_updates:
                        await session.execute(update(Declaration), pending_updates)
                        await session.commit()
                        logger.info(f"Committed batch of {len(pending_updates)} updates")
                        pending_updates.clear()

            except Exception as e:
                logger.error(f"Error during informalization: {e}")
                raise

        logger.info(
            f"Informalization complete. Processed {processed}/{total} declarations"
        )


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
