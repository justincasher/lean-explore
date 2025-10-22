"""Calculate PageRank scores for Lean declarations based on dependencies."""

import json
import logging

import networkx as nx
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.models import Declaration

logger = logging.getLogger(__name__)


async def _build_dependency_graph(session: AsyncSession) -> nx.DiGraph:
    """Build a directed graph of declaration dependencies.

    Args:
        session: Async database session

    Returns:
        NetworkX directed graph where edges point from dependents to dependencies
    """
    stmt = select(Declaration.id, Declaration.name, Declaration.dependencies)
    result = await session.execute(stmt)
    declarations = list(result.all())

    # Build name to ID mapping
    name_to_id = {
        declaration.name: declaration.id for declaration in declarations
    }

    # Build graph
    graph = nx.DiGraph()
    for declaration in declarations:
        graph.add_node(declaration.id)

    # Add edges
    for declaration in declarations:
        if declaration.dependencies:
            if isinstance(declaration.dependencies, str):
                dependencies = json.loads(declaration.dependencies)
            else:
                dependencies = declaration.dependencies
            for dependency_name in dependencies:
                if dependency_name in name_to_id:
                    # Edge from dependent to dependency
                    graph.add_edge(declaration.id, name_to_id[dependency_name])

    logger.info(
        f"Built graph with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges"
    )
    return graph


async def calculate_pagerank(
    engine: AsyncEngine,
    alpha: float = 0.85,
    batch_size: int = 1000,
) -> None:
    """Calculate and store PageRank scores for declarations.

    Args:
        engine: Async database engine
        alpha: Damping parameter for PageRank algorithm
        batch_size: Number of updates to batch before committing
    """
    logger.info(f"Starting PageRank calculation with alpha={alpha}")

    async with AsyncSession(engine) as session:
        # Build dependency graph
        graph = await _build_dependency_graph(session)

        if graph.number_of_nodes() == 0:
            logger.warning("No declarations found, skipping PageRank")
            return

        # Calculate PageRank scores
        logger.info("Calculating PageRank scores...")
        scores = nx.pagerank(graph, alpha=alpha, max_iter=1000, tol=1e-8)

        # Update database in batches
        logger.info(f"Updating {len(scores)} PageRank scores in database...")
        updates = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Updating PageRank scores", total=len(scores))

            for declaration_id, score in scores.items():
                updates.append({"id": declaration_id, "pagerank": float(score)})

                if len(updates) >= batch_size:
                    await session.execute(update(Declaration), updates)
                    await session.commit()
                    progress.update(task, advance=len(updates))
                    updates.clear()

            # Commit remaining updates
            if updates:
                await session.execute(update(Declaration), updates)
                await session.commit()
                progress.update(task, advance=len(updates))

        logger.info("PageRank calculation complete")
