# File: scripts/benchmark_search.py

"""Benchmarks the search functionality of the lean_explore package.

Measures queries per minute (QPM) and average query latency.
It loads assets once, then performs a configured number of benchmark passes
over a query list, and reports performance metrics.
"""

import argparse
import logging
import pathlib
import sys
import time
from typing import List, Optional

# Ensure the src directory is in the Python path
# This allows importing lean_explore modules
TRUE_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRUE_PROJECT_ROOT / "src"))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from tqdm import tqdm

    from lean_explore.local.search import (
        load_embedding_model,
        load_faiss_assets,
        perform_search,
    )

    from .config import APP_CONFIG
except ImportError as e:
    if "tqdm" in str(e).lower():
        print(
            "Warning: tqdm library not found. Progress bars will not be shown. "
            "Please install it: pip install tqdm",
            file=sys.stderr,
        )

        # Define a dummy tqdm if not found
        # pylint: disable=redefined-outer-name
        def tqdm(iterable, *args, **kwargs):  # type: ignore
            """Dummy tqdm function when tqdm is not installed."""
            return iterable
    else:
        print(
            f"Error: Could not import project modules or dependencies: {e}\n"
            "Ensure 'lean_explore' is structured correctly, all dependencies "
            "are installed (e.g., via 'pip install -r requirements.txt' or "
            "'pip install .'), and that this script is run from a context "
            "where 'src' is discoverable.",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Benchmark Configuration ---
DEFAULT_BENCHMARK_QUERIES_FILENAME = "benchmark_queries.txt"
# Number of full passes over the query list for actual benchmarking.
# Setting this to 1 means the benchmark phase will be one pass over all unique queries.
DEFAULT_NUM_BENCHMARK_RUNS = 1


def load_test_queries(file_path: pathlib.Path) -> List[str]:
    """Loads test queries from a text file, one query per line.

    Args:
        file_path: Path to the text file containing queries.

    Returns:
        A list of query strings. Returns an empty list if the file
        cannot be read or is empty.
    """
    if not file_path.exists():
        logger.error("Test queries file not found: %s", file_path)
        return []
    try:
        with open(file_path, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d test queries from %s.", len(queries), file_path)
        return queries
    except Exception as e:
        logger.error(
            "Failed to load test queries from %s: %s", file_path, e, exc_info=True
        )
        return []


def parse_benchmark_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the benchmark script.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark lean_explore search functionality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=DEFAULT_BENCHMARK_QUERIES_FILENAME,
        help="Filename of test queries (expected in scripts/ directory or "
        "provide full path).",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=DEFAULT_NUM_BENCHMARK_RUNS,
        help="Number of full passes over the query list for actual benchmarking.",
    )
    return parser.parse_args()


def main():
    """Main execution function for the benchmark script."""
    args = parse_benchmark_arguments()

    logger.info("--- Lean Explore Search Benchmark Initializing ---")
    logger.info("Using TRUE_PROJECT_ROOT: %s", TRUE_PROJECT_ROOT)

    try:
        search_config = APP_CONFIG.get("search", {})
        db_config = APP_CONFIG.get("database", {})
        raw_db_url_from_config = db_config.get("url")

        embedding_model_name = search_config.get("embedding_model_name")
        faiss_idx_path_str = search_config.get("faiss_index_path")
        faiss_map_path_str = search_config.get("faiss_map_path")

        faiss_k = int(search_config.get("faiss_k", 200))
        pagerank_weight = float(search_config.get("pagerank_weight", 0.3))
        text_relevance_weight = float(search_config.get("text_relevance_weight", 1.0))
        name_match_weight = float(search_config.get("name_match_weight", 0.0))

        if not all(
            [
                raw_db_url_from_config,
                embedding_model_name,
                faiss_idx_path_str,
                faiss_map_path_str,
            ]
        ):
            missing_configs = [
                k
                for k, v in {
                    "Database URL": raw_db_url_from_config,
                    "Embedding Model Name": embedding_model_name,
                    "FAISS Index Path": faiss_idx_path_str,
                    "FAISS Map Path": faiss_map_path_str,
                }.items()
                if not v
            ]
            raise ValueError(
                f"Missing critical configurations: {', '.join(missing_configs)}"
            )

        resolved_idx_path = TRUE_PROJECT_ROOT / faiss_idx_path_str
        resolved_map_path = TRUE_PROJECT_ROOT / faiss_map_path_str

        logger.info("Configuration loaded successfully.")
        logger.info("FAISS Index: %s", resolved_idx_path)
        logger.info("FAISS Map: %s", resolved_map_path)

        db_url_to_use = raw_db_url_from_config
        logger.info("Raw DB URL from config: %s", raw_db_url_from_config)
        if raw_db_url_from_config and raw_db_url_from_config.startswith("sqlite:///"):
            relative_db_file_path = raw_db_url_from_config[len("sqlite///") :]
            if not pathlib.Path(relative_db_file_path).is_absolute():
                absolute_db_path = TRUE_PROJECT_ROOT / relative_db_file_path
                db_url_to_use = f"sqlite:///{absolute_db_path.resolve()}"
        logger.info("Resolved DB URL to use: %s", db_url_to_use)

    except Exception as e:
        logger.error(
            "Failed to load or validate application configuration: %s", e, exc_info=True
        )
        sys.exit(1)

    logger.info("Loading shared assets...")
    s_transformer_model: Optional[SentenceTransformer] = None
    faiss_idx: Optional[faiss.Index] = None
    id_map: Optional[List[str]] = None
    engine = None

    try:
        s_transformer_model = load_embedding_model(embedding_model_name)
        if s_transformer_model is None:
            sys.exit(1)
        faiss_idx, id_map = load_faiss_assets(
            str(resolved_idx_path), str(resolved_map_path)
        )
        if faiss_idx is None or id_map is None:
            sys.exit(1)
        engine = create_engine(db_url_to_use, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("All shared assets loaded successfully.")
    except Exception as e:
        logger.error("An error occurred during asset loading: %s", e, exc_info=True)
        sys.exit(1)

    queries_file_path_str = args.queries_file
    queries_file_abs_path = (
        pathlib.Path(__file__).resolve().parent / queries_file_path_str
        if not pathlib.Path(queries_file_path_str).is_absolute()
        else pathlib.Path(queries_file_path_str)
    )
    test_queries = load_test_queries(queries_file_abs_path)
    if not test_queries:
        logger.error(
            "No test queries loaded. Ensure '%s' exists and is readable.",
            queries_file_abs_path,
        )
        if engine:
            engine.dispose()
        sys.exit(1)
    num_unique_queries = len(test_queries)

    search_module_logger = logging.getLogger("lean_explore.search")
    original_search_logger_level = search_module_logger.level

    try:
        search_module_logger.setLevel(logging.WARNING)

        # Warm-up phase has been removed.

        # Execution Phase
        if args.benchmark_runs <= 0 or num_unique_queries == 0:
            logger.info(
                "No benchmark runs specified or no queries loaded. Exiting "
                "benchmark execution."
            )
            if engine:
                engine.dispose()
            sys.exit(0)

        total_benchmark_queries_to_process = args.benchmark_runs * num_unique_queries
        logger.info(
            "Starting benchmark phase: %d passes over %d unique queries (%d "
            "total benchmark queries to process)...",
            args.benchmark_runs,
            num_unique_queries,
            total_benchmark_queries_to_process,
        )

        all_query_latencies: List[float] = []
        benchmark_total_queries_processed_successfully = 0
        benchmark_overall_start_time = time.perf_counter()

        with tqdm(
            total=total_benchmark_queries_to_process,
            desc="Benchmark Queries",
            unit="query",
            ncols=100,
        ) as pbar:
            for i in range(total_benchmark_queries_to_process):
                query = test_queries[i % num_unique_queries]
                query_start_time = time.perf_counter()
                try:
                    with SessionLocal() as session:
                        _results = perform_search(
                            session=session,
                            query_string=query,
                            model=s_transformer_model,
                            faiss_index=faiss_idx,
                            text_chunk_id_map=id_map,
                            faiss_k=faiss_k,
                            pagerank_weight=pagerank_weight,
                            text_relevance_weight=text_relevance_weight,
                            name_match_weight=name_match_weight,
                        )
                    query_end_time = time.perf_counter()
                    all_query_latencies.append(query_end_time - query_start_time)
                    benchmark_total_queries_processed_successfully += 1
                except Exception as e:
                    logger.error(
                        "[Benchmark] Error during query '%s': %s. Skipping.", query, e
                    )
                pbar.update(1)

        benchmark_overall_end_time = time.perf_counter()
        total_elapsed_time_seconds = (
            benchmark_overall_end_time - benchmark_overall_start_time
        )

        logger.info("Benchmark phase completed.")
        print("\n--- Benchmark Results ---")
        if benchmark_total_queries_processed_successfully == 0:
            print("No queries were successfully processed in the benchmark phase.")
        else:
            avg_latency_seconds = (
                sum(all_query_latencies)
                / benchmark_total_queries_processed_successfully
            )
            qps = (
                benchmark_total_queries_processed_successfully
                / total_elapsed_time_seconds
                if total_elapsed_time_seconds > 0
                else float("inf")
            )
            qpm = qps * 60
            print(
                f"Total Queries Attempted (Benchmark Phase):  "
                f"{total_benchmark_queries_to_process}"
            )
            print(
                f"Total Queries Processed Successfully:       "
                f"{benchmark_total_queries_processed_successfully}"
            )
            print(
                f"Total Time Taken (Benchmark Phase):         "
                f"{total_elapsed_time_seconds:.3f} seconds"
            )
            print(f"Queries Per Second (QPS, successful):       {qps:.2f}")
            print(f"Queries Per Minute (QPM, successful):       {qpm:.2f}")
            print(
                f"Average Latency per Query (successful):     "
                f"{avg_latency_seconds:.4f} seconds"
            )
            if all_query_latencies:
                all_query_latencies.sort()
                if len(all_query_latencies) > 0:
                    min_lat = all_query_latencies[0]
                    max_lat = all_query_latencies[-1]
                    median_lat = all_query_latencies[len(all_query_latencies) // 2]
                    p90_lat = all_query_latencies[
                        min(
                            int(len(all_query_latencies) * 0.90),
                            len(all_query_latencies) - 1,
                        )
                    ]
                    p99_lat = all_query_latencies[
                        min(
                            int(len(all_query_latencies) * 0.99),
                            len(all_query_latencies) - 1,
                        )
                    ]
                    print(f"Min Latency:                              {min_lat:.4f} s")
                    print(
                        f"Median Latency:                           {median_lat:.4f} s"
                    )
                    print(f"90th Percentile Latency:                  {p90_lat:.4f} s")
                    print(f"99th Percentile Latency:                  {p99_lat:.4f} s")
                    print(f"Max Latency:                              {max_lat:.4f} s")
        print("-------------------------\n")

    finally:
        search_module_logger.setLevel(original_search_logger_level)
        logger.debug(
            "Restored 'lean_explore.search' logger level to original: %s",
            original_search_logger_level,
        )
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")
    logger.info("Benchmark script finished.")


if __name__ == "__main__":
    main()
