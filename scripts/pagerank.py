# File: scripts/pagerank.py

"""Calculates and stores PageRank scores for Declarations and StatementGroups.

Connects to a database, builds dependency graphs for both Declarations and
StatementGroups, calculates PageRank scores for each, and updates the respective
tables. For StatementGroups, it also calculates and stores a log-transformed,
min-max scaled version of their PageRank scores.
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# --- Dependency Imports ---
try:
    import networkx as nx
    import numpy as np
    from sqlalchemy import create_engine, func, select, update
    from sqlalchemy.exc import OperationalError, SQLAlchemyError
    from sqlalchemy.orm import Session, sessionmaker
    from tqdm import tqdm
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please run: pip install sqlalchemy networkx numpy tqdm",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Project Model & Config Imports ---
try:
    from .config import APP_CONFIG
    from lean_explore.shared.models.db import (
        Declaration,
        Dependency,
        StatementGroup,
        StatementGroupDependency,
    )
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Could not import project modules (models, config): {e}\n"
        "Ensure 'lean_explore' is installed (e.g., 'pip install -e .') "
        "and dependencies are met.",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:  # pylint: disable=broad-except
    print(
        f"An unexpected error occurred during project module import: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)  # Reduce SA verbosity


# --- Constants ---
DEFAULT_PAGERANK_ALPHA = 0.85
DB_UPDATE_BATCH_SIZE = 1000


def load_declaration_graph_from_db(
    session: Session,
) -> Tuple[Optional[nx.DiGraph], Set[int]]:
    """Loads the Declaration dependency graph from the database.

    An edge from Source -> Target implies Declaration Source depends on
    Declaration Target.

    Args:
        session: The SQLAlchemy session object.

    Returns:
        A tuple containing:
            - nx.DiGraph: The declaration dependency graph. None if no
              declarations are found.
            - Set[int]: A set of all unique Declaration IDs found.

    Raises:
        SQLAlchemyError: If there's an error querying the database.
    """
    logger.info("Loading all declaration IDs for Declaration Graph...")
    decl_ids_result = session.execute(select(Declaration.id)).fetchall()
    all_decl_ids: Set[int] = {row[0] for row in decl_ids_result}
    logger.info("Found %d unique declaration nodes.", len(all_decl_ids))

    if not all_decl_ids:
        logger.warning("No declarations found. Cannot build Declaration graph.")
        return None, set()

    logger.info("Loading dependency links for Declaration Graph...")
    dependencies_query = session.execute(
        select(Dependency.source_decl_id, Dependency.target_decl_id)
    )
    total_deps = (
        session.execute(select(func.count(Dependency.id))).scalar_one_or_none() or 0
    )

    graph_decl = nx.DiGraph()
    graph_decl.add_nodes_from(all_decl_ids)

    edge_count = 0
    invalid_edge_count = 0
    with tqdm(
        dependencies_query,
        desc="Processing Declaration Dependencies",
        unit="dep",
        total=total_deps,
    ) as pbar:
        for source_id, target_id in pbar:
            if source_id in all_decl_ids and target_id in all_decl_ids:
                graph_decl.add_edge(source_id, target_id)
                edge_count += 1
            else:
                invalid_edge_count += 1
                if invalid_edge_count % 10000 == 1:
                    logger.debug(
                        "Skipping invalid edge for DeclGraph (src/tgt ID not in "
                        "all_decl_ids): %s -> %s. Invalid count: %d",
                        source_id,
                        target_id,
                        invalid_edge_count,
                    )

    logger.info(
        "Built Declaration Graph with %d nodes and %d valid edges.",
        graph_decl.number_of_nodes(),
        edge_count,
    )
    if invalid_edge_count > 0:
        logger.warning(
            "Skipped %d invalid dependency edges for DeclGraph.", invalid_edge_count
        )
    return graph_decl, all_decl_ids


def load_statement_group_graph_from_db(
    session: Session,
) -> Optional[nx.DiGraph]:
    """Loads the StatementGroup dependency graph directly from the database.

    An edge from Group A -> Group B means StatementGroup A directly depends on
    StatementGroup B, as recorded in the 'statement_group_dependencies' table.

    Args:
        session: The SQLAlchemy session object.

    Returns:
        Optional[nx.DiGraph]: The statement group dependency graph. None if no
        statement groups are found or if there's an error.

    Raises:
        SQLAlchemyError: If there's an error querying the database.
    """
    logger.info("Loading StatementGroup nodes for graph construction...")
    sg_ids_result = session.execute(select(StatementGroup.id)).fetchall()
    all_sg_ids: Set[int] = {row[0] for row in sg_ids_result}
    logger.info("Found %d unique statement group nodes.", len(all_sg_ids))

    if not all_sg_ids:
        logger.warning("No statement groups found. Cannot build StatementGroup graph.")
        return None

    graph_sg = nx.DiGraph()
    graph_sg.add_nodes_from(all_sg_ids)

    logger.info(
        "Loading StatementGroup dependency links from "
        "'statement_group_dependencies' table..."
    )
    sg_deps_query = session.execute(
        select(
            StatementGroupDependency.source_statement_group_id,
            StatementGroupDependency.target_statement_group_id,
        )
    )
    total_sg_deps = (
        session.execute(
            select(func.count(StatementGroupDependency.id))
        ).scalar_one_or_none()
        or 0
    )

    edge_count = 0
    invalid_edge_count = 0
    with tqdm(
        sg_deps_query,
        desc="Processing StatementGroup Dependencies",
        unit="dep",
        total=total_sg_deps,
    ) as pbar:
        for source_sg_id, target_sg_id in pbar:
            if source_sg_id in all_sg_ids and target_sg_id in all_sg_ids:
                graph_sg.add_edge(source_sg_id, target_sg_id)
                edge_count += 1
            else:
                invalid_edge_count += 1
                if invalid_edge_count % 1000 == 1:
                    logger.debug(
                        "Skipping invalid edge for SG Graph (src/tgt SG ID not in "
                        "all_sg_ids): %s -> %s. Invalid count: %d",
                        source_sg_id,
                        target_sg_id,
                        invalid_edge_count,
                    )

    logger.info(
        "Built StatementGroup Graph with %d nodes and %d valid edges.",
        graph_sg.number_of_nodes(),
        edge_count,
    )
    if invalid_edge_count > 0:
        logger.warning(
            "Skipped %d invalid dependency edges for SG Graph (SG ID not found). "
            "This may indicate data inconsistency if 'statement_group_dependencies' "
            "table is out of sync.",
            invalid_edge_count,
        )
    return graph_sg


def calculate_pagerank(
    graph: nx.DiGraph, alpha: float, graph_type: str = "Generic"
) -> Dict[int, float]:
    """Calculates PageRank scores for the nodes in the given graph.

    Args:
        graph: The NetworkX DiGraph for which to calculate PageRank.
        alpha: The damping parameter for PageRank.
        graph_type: A string descriptor for the graph type (for logging).

    Returns:
        Dict[int, float]: A dictionary mapping node IDs to their PageRank scores.
        Returns an empty dictionary if the graph is empty, has no nodes, or if
        PageRank calculation fails.
    """
    if not graph or graph.number_of_nodes() == 0:
        logger.warning(
            "%s graph is empty or has no nodes; cannot calculate PageRank.", graph_type
        )
        return {}

    logger.info(
        "Calculating PageRank scores for %s Graph (alpha=%.2f)...", graph_type, alpha
    )
    try:
        pagerank_scores = nx.pagerank(graph, alpha=alpha, max_iter=1000, tol=1.0e-8)
        logger.info(
            "PageRank calculation for %s Graph complete. Found %d scores.",
            graph_type,
            len(pagerank_scores),
        )
        return pagerank_scores
    except nx.PowerIterationFailedConvergence as e_conv:
        logger.error(
            "PageRank power iteration failed to converge for %s Graph "
            "(alpha=%.2f, max_iter=1000, tol=1.0e-8): %s. "
            "Consider increasing max_iter, adjusting alpha, or checking graph "
            "structure.",
            graph_type,
            alpha,
            e_conv,
        )
        return {}
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Error during PageRank calculation for %s Graph: %s",
            graph_type,
            e,
            exc_info=True,
        )
        return {}


def update_pagerank_scores_in_db(
    session: Session,
    scores_map: Dict[int, float],
    model_class: type,
    column_name: str,
    batch_size: int,
    item_type_name: str,
) -> int:
    """Updates a score column in a given table for a batch of items.

    Args:
        session: The SQLAlchemy session.
        scores_map: Dictionary mapping item ID to its score.
        model_class: The SQLAlchemy model class (e.g., Declaration).
        column_name: The name of the column to update with the score
            (e.g., "pagerank_score").
        batch_size: Number of updates to batch before committing.
        item_type_name: Name of the item type for logging (e.g., "Declaration").

    Returns:
        int: The number of items successfully updated.

    Raises:
        SQLAlchemyError: If a database error occurs during commit.
        Exception: If an unexpected error occurs.
    """
    if not scores_map:
        logger.warning(
            "No %s '%s' scores provided, skipping DB update for this column.",
            item_type_name,
            column_name,
        )
        return 0

    logger.info(
        "Updating %d %s '%s' entries in the database...",
        len(scores_map),
        item_type_name,
        column_name,
    )
    updated_count = 0
    pending_updates_mappings: List[Dict[str, Any]] = []

    with tqdm(
        scores_map.items(),
        desc=f"Updating {item_type_name} {column_name}",
        unit="item",
        total=len(scores_map),
    ) as pbar:
        for item_id, score in pbar:
            pending_updates_mappings.append(
                {"id": int(item_id), column_name: float(score)}
            )
            if len(pending_updates_mappings) >= batch_size:
                session.execute(update(model_class), pending_updates_mappings)
                session.commit()
                updated_count += len(pending_updates_mappings)
                logger.debug(
                    "Committed batch of %d %s '%s' updates.",
                    len(pending_updates_mappings),
                    item_type_name,
                    column_name,
                )
                pending_updates_mappings.clear()

        if pending_updates_mappings:
            session.execute(update(model_class), pending_updates_mappings)
            session.commit()
            updated_count += len(pending_updates_mappings)
            logger.debug(
                "Committed final batch of %d %s '%s' updates.",
                len(pending_updates_mappings),
                item_type_name,
                column_name,
            )

    logger.info(
        "Successfully updated '%s' for %d %ss.",
        column_name,
        updated_count,
        item_type_name,
    )
    return updated_count


def calculate_and_store_scaled_statement_group_pagerank(
    session: Session, batch_size: int
) -> int:
    """Calculates and stores log-transformed, min-max scaled PageRank scores.

    This function applies scaling to the 'pagerank_score' column of the
    'statement_groups' table and updates the 'scaled_pagerank_score' column.
    It assumes raw PageRank scores are already populated.

    Args:
        session: The SQLAlchemy session object.
        batch_size: Number of database records to update before committing.

    Returns:
        int: The number of statement groups whose scaled PageRank scores
        were updated.

    Raises:
        SQLAlchemyError: If a database error occurs.
        Exception: If an unexpected error occurs during calculation or update.
    """
    logger.info("Calculating and storing scaled PageRank for StatementGroups...")

    sg_data_query = select(StatementGroup.id, StatementGroup.pagerank_score).where(
        StatementGroup.pagerank_score.isnot(None)
    )
    sg_data_result = session.execute(sg_data_query).fetchall()

    if not sg_data_result:
        logger.warning("No StatementGroups with PageRank scores found to scale.")
        return 0

    ids = np.array([item.id for item in sg_data_result])
    raw_scores = np.array(
        [
            item.pagerank_score if item.pagerank_score is not None else 0.0
            for item in sg_data_result
        ],
        dtype=np.float64,
    )
    logger.info("Processing %d raw PageRank scores for scaling.", len(raw_scores))

    epsilon = 1e-9
    log_scores = np.log(raw_scores + epsilon)

    min_log_score = np.min(log_scores)
    max_log_score = np.max(log_scores)
    logger.info(
        "Log-transformed scores range: [%.4f, %.4f]", min_log_score, max_log_score
    )

    scaled_scores_map: Dict[int, float] = {}
    if np.isclose(max_log_score, min_log_score):
        default_scaled_val = 0.0 if np.allclose(raw_scores, 0.0) else 0.5
        logger.warning(
            "All log-transformed PageRank scores are identical (%.4f). "
            "Scaled scores will be uniformly set to %.1f.",
            min_log_score,
            default_scaled_val,
        )
        for item_id in ids:
            scaled_scores_map[int(item_id)] = default_scaled_val
    else:
        scaled_values = (log_scores - min_log_score) / (max_log_score - min_log_score)
        for i, item_id in enumerate(ids):
            scaled_scores_map[int(item_id)] = float(scaled_values[i])

    return update_pagerank_scores_in_db(
        session,
        scaled_scores_map,
        StatementGroup,
        "scaled_pagerank_score",
        batch_size,
        "StatementGroup",
    )


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the PageRank calculation script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate PageRank for Declarations and StatementGroups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    db_url_default = APP_CONFIG.get("database", {}).get("url")

    parser.add_argument(
        "--db-url",
        type=str,
        default=db_url_default,
        help="SQLAlchemy database URL. Overrides config if provided.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_PAGERANK_ALPHA,
        help="Damping parameter (alpha) for PageRank.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DB_UPDATE_BATCH_SIZE,
        help="DB records to update before commit.",
    )
    parser.add_argument(
        "--skip-declarations",
        action="store_true",
        help="Skip PageRank for Declarations.",
    )
    parser.add_argument(
        "--skip-statement-groups",
        action="store_true",
        help="Skip PageRank and scaling for StatementGroups.",
    )

    args = parser.parse_args()
    if not args.db_url:
        logger.error(
            "Database URL is not configured. Provide via --db-url or in "
            "the application configuration file."
        )
        sys.exit(1)
    return args


def main():
    """Main execution function to orchestrate PageRank calculations and updates."""
    args = parse_arguments()
    db_url_display = f"...{args.db_url[-30:]}" if len(args.db_url) > 30 else args.db_url

    logger.info("--- Starting PageRank Calculations ---")
    logger.info("Using Database URL: %s", db_url_display)
    logger.info("PageRank Alpha: %.2f", args.alpha)
    logger.info("DB Update Batch Size: %d", args.batch_size)

    engine = None
    try:
        engine = create_engine(args.db_url, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with engine.connect():
            logger.info("Database connection successful.")

        if not args.skip_declarations:
            logger.info("--- Starting Declaration PageRank Phase ---")
            with SessionLocal() as session:
                try:
                    decl_graph, _ = load_declaration_graph_from_db(session)
                    if decl_graph and decl_graph.number_of_nodes() > 0:
                        decl_pr_scores = calculate_pagerank(
                            decl_graph, args.alpha, "Declaration"
                        )
                        if decl_pr_scores:
                            update_pagerank_scores_in_db(
                                session,
                                decl_pr_scores,
                                Declaration,
                                "pagerank_score",
                                args.batch_size,
                                "Declaration",
                            )
                        else:
                            logger.info(
                                "No PageRank scores calculated for declarations."
                            )
                    else:
                        logger.info(
                            "Skipping Declaration PageRank: graph empty/not built."
                        )
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
            logger.info("--- Declaration PageRank Phase Completed ---")
        else:
            logger.info("--- Skipping Declaration PageRank Phase ---")

        if not args.skip_statement_groups:
            logger.info("--- Starting StatementGroup PageRank Phase ---")
            with SessionLocal() as session:
                try:
                    sg_graph = load_statement_group_graph_from_db(session)
                    if sg_graph and sg_graph.number_of_nodes() > 0:
                        sg_pr_scores = calculate_pagerank(
                            sg_graph, args.alpha, "StatementGroup"
                        )
                        if sg_pr_scores:
                            update_pagerank_scores_in_db(
                                session,
                                sg_pr_scores,
                                StatementGroup,
                                "pagerank_score",
                                args.batch_size,
                                "StatementGroup",
                            )
                            session.commit()
                            calculate_and_store_scaled_statement_group_pagerank(
                                session, args.batch_size
                            )
                            session.commit()
                        else:
                            logger.info(
                                "No PageRank scores for statement groups to "
                                "update/scale."
                            )
                    else:
                        logger.info(
                            "Skipping StatementGroup PageRank: graph empty/not built."
                        )
                except Exception:
                    session.rollback()
                    raise
            logger.info("--- StatementGroup PageRank Phase Completed ---")
        else:
            logger.info("--- Skipping StatementGroup PageRank Phase ---")

        logger.info("--- All PageRank Operations Completed Successfully ---")

    except OperationalError as e:
        logger.error("Database connection/operational error: %s", e)
        logger.error(
            "Check DB URL, credentials, server status, and schema (tables/columns)."
        )
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error("A database error occurred: %s", e, exc_info=True)
        sys.exit(1)
    except ImportError as e:
        logger.error("Import error: %s. Ensure requirements are installed.", e)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.critical("An unexpected critical error occurred: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")


if __name__ == "__main__":
    main()
