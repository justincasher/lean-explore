# src/lean_explore/search.py

"""Performs semantic search and ranked retrieval of StatementGroups.

Combines semantic similarity from FAISS, pre-scaled PageRank scores, and
Lean name matching to rank StatementGroups. It loads necessary assets
(embedding model, FAISS index, ID map) using default configurations,
embeds the user query, performs FAISS search, filters based on a
similarity threshold, retrieves group details from the database, calculates
a name match score, normalizes semantic similarity scores, and then combines
these scores using configurable weights to produce a final ranked list.
It also logs search performance statistics to a dedicated JSONL file.
"""

import argparse
import datetime
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from filelock import Timeout, FileLock

# --- Dependency Imports ---
try:
    import faiss
    import numpy as np
    from rapidfuzz import fuzz # Using rapidfuzz for name matching
    from sentence_transformers import SentenceTransformer
    from sqlalchemy import create_engine, select, or_
    from sqlalchemy.exc import OperationalError, SQLAlchemyError
    from sqlalchemy.orm import Session, joinedload, sessionmaker
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please install them: pip install SQLAlchemy faiss-cpu "
        "sentence-transformers numpy rapidfuzz filelock",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Project Model & Default Config Imports ---
try:
    from lean_explore.shared.models.db import StatementGroup
    from lean_explore import defaults # Using the new defaults module
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Could not import project modules (StatementGroup, defaults): {e}\n"
        "Ensure 'lean_explore' is installed (e.g., 'pip install -e .') "
        "and all dependencies are met.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Constants ---
NEWLINE = os.linesep
EPSILON = 1e-9
# PROJECT_ROOT might be less relevant for asset paths if defaults.py provides absolute paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- Performance Logging Path Setup ---
# Logs will be stored in a user-writable directory, e.g., ~/.lean_explore/logs/
# defaults.LEAN_EXPLORE_USER_DATA_DIR is ~/.lean_explore/data/
# So, its parent is ~/.lean_explore/
_USER_LOGS_BASE_DIR = defaults.LEAN_EXPLORE_USER_DATA_DIR.parent / "logs"
PERFORMANCE_LOG_DIR = str(_USER_LOGS_BASE_DIR)
PERFORMANCE_LOG_FILENAME = "search_stats.jsonl"
PERFORMANCE_LOG_PATH = os.path.join(PERFORMANCE_LOG_DIR, PERFORMANCE_LOG_FILENAME)
LOCK_PATH = os.path.join(PERFORMANCE_LOG_DIR, f"{PERFORMANCE_LOG_FILENAME}.lock")


# --- Helper & Scoring Functions ---

def calculate_name_match_score(query_string: str, lean_name: str) -> float:
    """Calculates a name match score using fuzzy matching (WRatio), possibly mapped.

    Compares the lowercased query string and Lean name using `fuzz.WRatio`.
    The raw WRatio (0-100) is normalized to 0-1. Then, a threshold-based
    mapping is applied: scores below a threshold become 0, and scores above
    are linearly scaled to the 0-1 range.

    Args:
        query_string: The user's search query.
        lean_name: The Lean declaration name.

    Returns:
        A score between 0.0 and 1.0. Returns 0.0 if inputs are empty.
    """
    if not query_string or not lean_name:
        return 0.0

    norm_query = query_string.lower()
    norm_lean_name = lean_name.lower()

    score = fuzz.WRatio(norm_query, norm_lean_name) / 100.0

    threshold = 0.90
    if score < threshold:
        mapped_score = 0.0
    else:
        scale_factor = 1.0 / (1.0 - threshold + EPSILON)
        offset = -threshold * scale_factor
        mapped_score = (scale_factor * score) + offset
        mapped_score = min(1.0, max(0.0, mapped_score))

    return mapped_score

# --- Performance Logging Helper ---

def log_search_event_to_json(status: str, duration_ms: float, results_count: int, error_type: Optional[str] = None) -> None:
    """Logs a search event as a JSON line to a dedicated performance log file.

    Args:
        status: A string code indicating the outcome of the search.
        duration_ms: The total duration of the search processing in milliseconds.
        results_count: The number of search results returned.
        error_type: Optional. The type of error if the status indicates an error.
    """
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "search_processed",
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "results_count": results_count,
    }
    if error_type:
        log_entry["error_type"] = error_type

    try:
        os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)
    except OSError as e:
        # This error is critical for logging but should not stop the main search flow if possible.
        # The fallback print helps retain info if file logging fails.
        logger.error(
            "Performance logging error: Could not create log directory %s: %s. Log entry: %s",
            PERFORMANCE_LOG_DIR, e, log_entry, exc_info=False # Keep exc_info False to avoid spamming user console
        )
        print(f"FALLBACK_PERF_LOG (DIR_ERROR): {json.dumps(log_entry)}", file=sys.stderr)
        return

    lock = FileLock(LOCK_PATH, timeout=2)
    try:
        with lock:
            with open(PERFORMANCE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Timeout:
        logger.warning( # Changed to warning as it's a logging issue, not a core failure
            "Performance logging error: Timeout acquiring lock for %s. Log entry lost: %s",
            LOCK_PATH, log_entry
        )
        print(f"FALLBACK_PERF_LOG (LOCK_TIMEOUT): {json.dumps(log_entry)}", file=sys.stderr)
    except Exception as e:
        logger.error( # Keep as error for unexpected write issues
            "Performance logging error: Failed to write to %s: %s. Log entry: %s",
            PERFORMANCE_LOG_PATH, e, log_entry, exc_info=False
        )
        print(f"FALLBACK_PERF_LOG (WRITE_ERROR): {json.dumps(log_entry)}", file=sys.stderr)


# --- Asset Loading Functions ---
def load_faiss_assets(
    index_path_str: str, map_path_str: str
) -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
    """Loads the FAISS index and ID map from specified file paths.

    Args:
        index_path_str: String path to the FAISS index file.
        map_path_str: String path to the JSON ID map file.

    Returns:
        A tuple (faiss.Index or None, list_of_IDs or None).
    """
    index_path = pathlib.Path(index_path_str).resolve()
    map_path = pathlib.Path(map_path_str).resolve()

    if not index_path.exists():
        logger.error("FAISS index file not found: %s", index_path)
        return None, None
    if not map_path.exists():
        logger.error("FAISS ID map file not found: %s", map_path)
        return None, None

    faiss_index_obj: Optional[faiss.Index] = None
    id_map_list: Optional[List[str]] = None

    try:
        logger.info("Loading FAISS index from %s...", index_path)
        faiss_index_obj = faiss.read_index(str(index_path))
        logger.info(
            "Loaded FAISS index with %d vectors (Metric Type: %s).",
            faiss_index_obj.ntotal,
            faiss_index_obj.metric_type,
        )
    except Exception as e:
        logger.error("Failed to load FAISS index from %s: %s", index_path, e, exc_info=True)
        return None, id_map_list # Return None for index if loading failed

    try:
        logger.info("Loading ID map from %s...", map_path)
        with open(map_path, "r", encoding="utf-8") as f:
            id_map_list = json.load(f)
        if not isinstance(id_map_list, list):
            logger.error(
                "ID map file (%s) does not contain a valid JSON list.", map_path
            )
            return faiss_index_obj, None # Return None for map if parsing failed
        logger.info("Loaded ID map with %d entries.", len(id_map_list))
    except Exception as e:
        logger.error("Failed to load or parse ID map file %s: %s", map_path, e, exc_info=True)
        return faiss_index_obj, None # Return None for map if loading/parsing failed

    if (
        faiss_index_obj is not None
        and id_map_list is not None
        and faiss_index_obj.ntotal != len(id_map_list)
    ):
        logger.warning(
            "Mismatch: FAISS index size (%d) vs ID map size (%d). Results may be inconsistent.",
            faiss_index_obj.ntotal,
            len(id_map_list),
        )
    return faiss_index_obj, id_map_list


def load_embedding_model(model_name: str) -> Optional[SentenceTransformer]:
    """Loads the specified Sentence Transformer model.

    Args:
        model_name: The name or path of the sentence-transformer model.

    Returns:
        The loaded model, or None if loading fails.
    """
    logger.info("Loading sentence transformer model '%s'...", model_name)
    try:
        model = SentenceTransformer(model_name)
        logger.info(
            "Model '%s' loaded successfully. Max sequence length: %d.",
            model_name,
            model.max_seq_length,
        )
        return model
    except Exception as e: # Broad exception for any model loading issue
        logger.error("Failed to load sentence transformer model '%s': %s", model_name, e, exc_info=True)
        return None

# --- Main Search Function ---

def perform_search(
    session: Session,
    query_string: str,
    model: SentenceTransformer,
    faiss_index: faiss.Index,
    text_chunk_id_map: List[str],
    faiss_k: int,
    pagerank_weight: float,
    text_relevance_weight: float,
    name_match_weight: float,
    selected_packages: Optional[List[str]] = None,
    semantic_similarity_threshold: float = defaults.DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
    faiss_nprobe: int = defaults.DEFAULT_FAISS_NPROBE
) -> List[Tuple[StatementGroup, Dict[str, float]]]:
    """Performs semantic search and ranking.

    Args:
        session: SQLAlchemy session for database access.
        query_string: The user's search query string.
        model: The loaded SentenceTransformer embedding model.
        faiss_index: The loaded FAISS index for text chunks.
        text_chunk_id_map: A list mapping FAISS internal indices to text chunk IDs.
        faiss_k: The number of nearest neighbors to retrieve from FAISS.
        pagerank_weight: Weight for the pre-scaled PageRank score.
        text_relevance_weight: Weight for the normalized semantic similarity score.
        name_match_weight: Weight for the raw name match score.
        selected_packages: Optional list of package names to filter search by.
        semantic_similarity_threshold: Minimum similarity for a result to be considered.
        faiss_nprobe: Number of closest cells/clusters to search for IVF-type FAISS indexes.

    Returns:
        A list of tuples, sorted by final_score, containing a
        `StatementGroup` object and its scores.

    Raises:
        Exception: If critical errors like query embedding or FAISS search fail.
    """
    overall_start_time = time.time()

    logger.info("Search request event initiated.")
    if semantic_similarity_threshold > 0.0 + EPSILON:
        logger.info("Applying semantic similarity threshold: %.3f", semantic_similarity_threshold)

    if not query_string.strip():
        logger.warning("Empty query provided. Returning no results.")
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(status="EMPTY_QUERY_SUBMITTED", duration_ms=duration_ms, results_count=0)
        return []

    try:
        query_embedding = model.encode(
            [query_string.strip()], convert_to_numpy=True
        )[0].astype(np.float32)
        query_embedding_reshaped = np.expand_dims(query_embedding, axis=0)
        if faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT:
            logger.debug("Normalizing query embedding for Inner Product (cosine) search.")
            faiss.normalize_L2(query_embedding_reshaped)
    except Exception as e:
        logger.error("Failed to embed query: %s", e, exc_info=True)
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(status="EMBEDDING_ERROR", duration_ms=duration_ms, results_count=0, error_type=type(e).__name__)
        raise Exception(f"Query embedding failed: {e}") from e

    try:
        logger.debug("Searching FAISS index for top %d text chunk neighbors...", faiss_k)
        if hasattr(faiss_index, "nprobe") and isinstance(faiss_index.nprobe, int): # Check if index is IVF
            if faiss_nprobe > 0:
                faiss_index.nprobe = faiss_nprobe
                logger.debug(f"Set FAISS nprobe to: {faiss_index.nprobe}")
            else: # faiss_nprobe from config is invalid
                logger.warning(
                    f"Configured faiss_nprobe is {faiss_nprobe}. Must be > 0. "
                    "Using FAISS default or previously set nprobe for this IVF index."
                )
        distances, indices = faiss_index.search(query_embedding_reshaped, faiss_k)
    except Exception as e:
        logger.error("FAISS search failed: %s", e, exc_info=True)
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(status="FAISS_SEARCH_ERROR", duration_ms=duration_ms, results_count=0, error_type=type(e).__name__)
        raise Exception(f"FAISS search failed: {e}") from e

    sg_candidates_raw_similarity: Dict[int, float] = {}
    if indices.size > 0 and distances.size > 0:
        for i, faiss_internal_idx in enumerate(indices[0]):
            if faiss_internal_idx == -1: # FAISS can return -1 for no neighbor
                continue
            try:
                text_chunk_id_str = text_chunk_id_map[faiss_internal_idx]
                raw_faiss_score = distances[0][i]
                similarity_score: float

                if faiss_index.metric_type == faiss.METRIC_L2:
                    similarity_score = 1.0 / (1.0 + np.sqrt(max(0, raw_faiss_score)))
                elif faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    # Assuming normalized vectors, inner product is cosine similarity
                    similarity_score = raw_faiss_score
                else: # Default or unknown metric, treat score as distance-like
                    similarity_score = 1.0 / (1.0 + max(0, raw_faiss_score))
                    logger.warning(
                        "Unhandled FAISS metric type %d for text chunk. Using 1/(1+score) for similarity.",
                        faiss_index.metric_type
                    )
                similarity_score = max(0.0, min(1.0, similarity_score)) # Clamp to [0,1]

                parts = text_chunk_id_str.split("_")
                if len(parts) >= 2 and parts[0] == "sg":
                    try:
                        sg_id = int(parts[1])
                        # If multiple chunks from the same StatementGroup are retrieved,
                        # keep the one with the highest similarity to the query.
                        if sg_id not in sg_candidates_raw_similarity or \
                           similarity_score > sg_candidates_raw_similarity[sg_id]:
                            sg_candidates_raw_similarity[sg_id] = similarity_score
                    except ValueError:
                        logger.warning("Could not parse StatementGroup ID from chunk_id: %s", text_chunk_id_str)
                else:
                    logger.warning("Malformed text_chunk_id format: %s", text_chunk_id_str)
            except IndexError:
                logger.warning(
                    "FAISS internal index %d out of bounds for ID map (size %d). Possible data inconsistency.",
                    faiss_internal_idx, len(text_chunk_id_map)
                )
            except Exception as e: # Catch any other unexpected errors during result processing
                logger.warning(
                    "Error processing FAISS result for internal index %d (chunk_id '%s'): %s",
                    faiss_internal_idx, text_chunk_id_str if 'text_chunk_id_str' in locals() else "N/A", e
                )


    if not sg_candidates_raw_similarity:
        logger.info("No valid StatementGroup candidates found after FAISS search and parsing.")
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(status="NO_FAISS_CANDIDATES", duration_ms=duration_ms, results_count=0)
        return []
    logger.info(
        "Aggregated %d unique StatementGroup candidates from FAISS results.",
        len(sg_candidates_raw_similarity),
    )

    if semantic_similarity_threshold > 0.0 + EPSILON:
        initial_candidate_count = len(sg_candidates_raw_similarity)
        sg_candidates_raw_similarity = {
            sg_id: sim
            for sg_id, sim in sg_candidates_raw_similarity.items()
            if sim >= semantic_similarity_threshold
        }
        logger.info(
            "Post-thresholding: %d of %d candidates remaining (threshold: %.3f).",
            len(sg_candidates_raw_similarity),
            initial_candidate_count,
            semantic_similarity_threshold
        )

        if not sg_candidates_raw_similarity:
            logger.info(
                "No StatementGroup candidates met the semantic similarity threshold of %.3f.",
                semantic_similarity_threshold
            )
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(status="NO_CANDIDATES_POST_THRESHOLD", duration_ms=duration_ms, results_count=0)
            return []

    candidate_sg_ids = list(sg_candidates_raw_similarity.keys())
    sg_objects_map: Dict[int, StatementGroup] = {}
    try:
        logger.debug("Fetching StatementGroup details from DB for %d IDs...", len(candidate_sg_ids))
        stmt = select(StatementGroup).where(StatementGroup.id.in_(candidate_sg_ids))

        if selected_packages:
            logger.info("Filtering search by packages: %s", selected_packages)
            package_filters_sqla = []
            # Assuming package names in selected_packages are like "Mathlib", "Std"
            # And source_file in DB is like "Mathlib/CategoryTheory/Adjunction/Basic.lean"
            for pkg_name in selected_packages:
                # Ensure exact package match at the start of the file path component
                package_filters_sqla.append(StatementGroup.source_file.startswith(pkg_name + "/"))

            if package_filters_sqla:
                stmt = stmt.where(or_(*package_filters_sqla))

        # Eagerly load primary_declaration to avoid N+1 queries later if accessing lean_name
        stmt = stmt.options(joinedload(StatementGroup.primary_declaration))
        db_results = session.execute(stmt).scalars().unique().all()
        for sg_obj in db_results:
            sg_objects_map[sg_obj.id] = sg_obj

        logger.debug("Fetched details for %d StatementGroups from DB that matched filters.", len(sg_objects_map))
        # Log if some IDs from FAISS (post-threshold and package filter if applied) were not found in DB
        # This check is more informative if done *after* any package filtering logic in the query
        final_candidate_ids_after_db_match = set(sg_objects_map.keys())
        original_faiss_candidate_ids = set(candidate_sg_ids)

        if len(final_candidate_ids_after_db_match) < len(original_faiss_candidate_ids):
             missing_from_db_or_filtered_out = original_faiss_candidate_ids - final_candidate_ids_after_db_match
             logger.info(
                 "%d candidates from FAISS (post-threshold) were not found in DB "
                 "or excluded by package filters: (e.g., %s).",
                 len(missing_from_db_or_filtered_out),
                 list(missing_from_db_or_filtered_out)[:5]
             )

    except SQLAlchemyError as e:
        logger.error("Database query for StatementGroup details failed: %s", e, exc_info=True)
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(status="DB_FETCH_ERROR", duration_ms=duration_ms, results_count=0, error_type=type(e).__name__)
        raise # Re-raise to be handled by the caller

    results_with_scores: List[Tuple[StatementGroup, Dict[str, float]]] = []
    candidate_semantic_similarities: List[float] = [] # For normalization range
    processed_candidates_data: List[Dict[str, Any]] = [] # Temp store for data to be scored

    # Iterate over IDs that were confirmed to exist in the DB and match filters
    for sg_id in final_candidate_ids_after_db_match: # Use keys from sg_objects_map
        sg_obj = sg_objects_map[sg_id] # We know this exists
        raw_sem_sim = sg_candidates_raw_similarity[sg_id] # This ID came from FAISS initially
        raw_name_match = 0.0
        if sg_obj.primary_declaration and sg_obj.primary_declaration.lean_name:
            try:
                raw_name_match = calculate_name_match_score(
                    query_string, sg_obj.primary_declaration.lean_name
                )
            except Exception as e_nm: # Catch any error from name matching
                logger.warning(
                    "Error calculating name match score for SG %d ('%s'): %s",
                    sg_id, sg_obj.primary_declaration.lean_name, e_nm
                )
        else:
            logger.debug("Missing primary declaration or Lean name for SG ID %d, name score is 0.", sg_id)

        processed_candidates_data.append({
            "sg_obj": sg_obj,
            "raw_sem_sim": raw_sem_sim,
            "raw_name_match": raw_name_match,
        })
        candidate_semantic_similarities.append(raw_sem_sim)

    if not processed_candidates_data:
        logger.info("No candidates remaining after matching with DB data or other processing steps.")
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(status="NO_CANDIDATES_POST_PROCESSING", duration_ms=duration_ms, results_count=0)
        return []

    # Normalize semantic similarity scores for the retrieved candidates
    min_sem_sim = min(candidate_semantic_similarities) if candidate_semantic_similarities else 0.0
    max_sem_sim = max(candidate_semantic_similarities) if candidate_semantic_similarities else 0.0
    range_sem_sim = max_sem_sim - min_sem_sim
    logger.debug("Raw semantic similarity range for normalization: [%.4f, %.4f]", min_sem_sim, max_sem_sim)

    for candidate_data in processed_candidates_data:
        sg_obj = candidate_data["sg_obj"]
        current_raw_sem_sim = candidate_data["raw_sem_sim"]
        current_raw_name_match = candidate_data["raw_name_match"]

        # Normalize semantic similarity: scale to [0,1]
        norm_sem_sim = 0.5 # Default if range is zero (e.g., only one candidate)
        if range_sem_sim > EPSILON:
            norm_sem_sim = (current_raw_sem_sim - min_sem_sim) / range_sem_sim
        elif len(candidate_semantic_similarities) == 1 and candidate_semantic_similarities[0] > 0: # Single candidate
            norm_sem_sim = 1.0 # If only one candidate, its normalized score should be high if its raw score is non-zero.
        elif len(candidate_semantic_similarities) == 0 : # Should not happen given previous check
             norm_sem_sim = 0.0


        current_scaled_pagerank = sg_obj.scaled_pagerank_score if sg_obj.scaled_pagerank_score is not None else 0.0

        # Combine scores using weights
        weighted_norm_similarity = text_relevance_weight * norm_sem_sim
        weighted_scaled_pagerank = pagerank_weight * current_scaled_pagerank
        weighted_name_match = name_match_weight * current_raw_name_match
        final_score = weighted_norm_similarity + weighted_scaled_pagerank + weighted_name_match

        score_dict = {
            "final_score": final_score,
            "norm_similarity": norm_sem_sim,
            "scaled_pagerank": current_scaled_pagerank,
            "raw_name_match_score": current_raw_name_match,
            "weighted_norm_similarity": weighted_norm_similarity,
            "weighted_scaled_pagerank": weighted_scaled_pagerank,
            "weighted_name_match_score": weighted_name_match,
            "raw_similarity": current_raw_sem_sim, # Keep raw similarity for inspection
        }
        results_with_scores.append((sg_obj, score_dict))

    results_with_scores.sort(key=lambda item: item[1]["final_score"], reverse=True)

    final_status = "SUCCESS"
    results_count = len(results_with_scores)
    if not results_with_scores and processed_candidates_data: # Had candidates, but scoring/sorting yielded none (unlikely)
        final_status = "NO_RESULTS_FINAL_SCORED"
    elif not results_with_scores and not processed_candidates_data: # No candidates from the start essentially
        # This case should have been caught earlier, but as a safeguard for logging
        if not candidate_sg_ids: final_status = "NO_FAISS_CANDIDATES"
        elif not sg_candidates_raw_similarity : final_status = "NO_CANDIDATES_POST_THRESHOLD"


    duration_ms = (time.time() - overall_start_time) * 1000
    log_search_event_to_json(status=final_status, duration_ms=duration_ms, results_count=results_count)

    return results_with_scores

# --- Output Formatting ---

def print_results(results: List[Tuple[StatementGroup, Dict[str, float]]]) -> None:
    """Formats and prints the search results to the console.

    Args:
        results: A list of tuples, each containing a StatementGroup
            object and its scores, sorted by final_score.
    """
    if not results:
        print("\nNo results found.")
        return

    print(f"\n--- Top {len(results)} Search Results (StatementGroups) ---")
    for i, (sg_obj, scores) in enumerate(results):
        primary_decl_name = (
            sg_obj.primary_declaration.lean_name
            if sg_obj.primary_declaration and sg_obj.primary_declaration.lean_name
            else "N/A"
        )
        print(
            f"\n{i+1}. Lean Name: {primary_decl_name} (SG ID: {sg_obj.id})\n"
            f"   Final Score: {scores['final_score']:.4f} ("
            f"NormSim*W: {scores['weighted_norm_similarity']:.4f}, "
            f"ScaledPR*W: {scores['weighted_scaled_pagerank']:.4f}, "
            f"NameMatch*W: {scores['weighted_name_match_score']:.4f})"
        )
        print(
            f"   Scores: [NormSim: {scores['norm_similarity']:.4f}, "
            f"ScaledPR: {scores['scaled_pagerank']:.4f}, "
            f"RawNameMatch: {scores['raw_name_match_score']:.4f}, "
            f"RawSim: {scores['raw_similarity']:.4f}]"
        )

        lean_display = (
            sg_obj.display_statement_text or sg_obj.statement_text or "[No Lean code]"
        )
        lean_display_short = (
            (lean_display[:200] + "...") if len(lean_display) > 200 else lean_display
        )
        print(f"   Lean Code: {lean_display_short.replace(NEWLINE, ' ')}")

        desc_display = (
            sg_obj.informal_description or sg_obj.docstring or "[No description]"
        )
        desc_display_short = (
            (desc_display[:150] + "...") if len(desc_display) > 150 else desc_display
        )
        print(f"   Description: {desc_display_short.replace(NEWLINE, ' ')}")

        source_loc = sg_obj.source_file or "[No source file]"
        if source_loc.startswith("Mathlib/"): # Simplify Mathlib paths
            source_loc = source_loc[len("Mathlib/") :]
        print(f"   File: {source_loc}:{sg_obj.range_start_line}")

    print("\n---------------------------------------------------")


# --- Argument Parsing & Main Execution ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the search script.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Search Lean StatementGroups using combined scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="The search query string.")
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None, # Will use DEFAULT_RESULTS_LIMIT from defaults if None
        help="Maximum number of final results to display. "
             "Overrides default if set.",
    )
    parser.add_argument(
        "--packages",
        metavar="PKG",
        type=str,
        nargs="*", # Allows zero or more package names
        default=None, # No filter if not provided
        help="Filter search results by specific package names (e.g., Mathlib Std). "
             "If not provided, searches all packages."
    )
    return parser.parse_args()



def main():
    """Main execution function for the search script."""
    args = parse_arguments()

    logger.info("Using default configurations for paths and parameters from lean_explore.defaults.")

    # These now point to the versioned paths, e.g., .../toolchains/0.1.0/file.db
    db_url = defaults.DEFAULT_DB_URL
    embedding_model_name = defaults.DEFAULT_EMBEDDING_MODEL_NAME
    resolved_idx_path = str(defaults.DEFAULT_FAISS_INDEX_PATH.resolve())
    resolved_map_path = str(defaults.DEFAULT_FAISS_MAP_PATH.resolve())

    faiss_k_cand = defaults.DEFAULT_FAISS_K
    pr_weight = defaults.DEFAULT_PAGERANK_WEIGHT
    sem_sim_weight = defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT
    name_match_w = defaults.DEFAULT_NAME_MATCH_WEIGHT
    results_disp_limit = args.limit if args.limit is not None \
                         else defaults.DEFAULT_RESULTS_LIMIT
    semantic_sim_thresh = defaults.DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
    faiss_nprobe_val = defaults.DEFAULT_FAISS_NPROBE


    db_url_display = f"...{str(defaults.DEFAULT_DB_PATH.resolve())[-30:]}" if len(str(defaults.DEFAULT_DB_PATH.resolve())) > 30 else str(defaults.DEFAULT_DB_PATH.resolve())
    logger.info("--- Starting Search (Direct Script Execution) ---")
    logger.info("Query: '%s'", args.query)
    logger.info("Displaying Top: %d results", results_disp_limit)
    if args.packages:
        logger.info("Filtering by user-specified packages: %s", args.packages)
    else:
        logger.info("No package filter specified, searching all packages.")
    logger.info("FAISS k (candidates): %d", faiss_k_cand)
    logger.info("FAISS nprobe (from defaults): %d", faiss_nprobe_val)
    logger.info("Semantic Similarity Threshold (from defaults): %.3f", semantic_sim_thresh)
    logger.info(
        "Weights -> NormTextSim: %.2f, ScaledPR: %.2f, RawNameMatch: %.2f",
        sem_sim_weight, pr_weight, name_match_w
    )
    logger.info("Using FAISS index: %s", resolved_idx_path)
    logger.info("Using ID map: %s", resolved_map_path)
    logger.info("Database path: %s", db_url_display) # Changed from URL for clarity with file paths

    # Ensure user data directory and toolchain directory exist for logs etc.
    # The fetch command handles creation of the specific toolchain version dir.
    # Here, we ensure the base log directory can be created by performance logger.
    try:
        _USER_LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create user log directory {_USER_LOGS_BASE_DIR}: {e}")


    engine = None
    try:
        # Asset loading with improved error potential
        s_transformer_model = load_embedding_model(embedding_model_name)
        if s_transformer_model is None:
            # load_embedding_model already logs the error
            logger.error("Sentence transformer model loading failed. Cannot proceed with search.")
            sys.exit(1)

        faiss_idx, id_map = load_faiss_assets(resolved_idx_path, resolved_map_path)
        if faiss_idx is None or id_map is None:
            # load_faiss_assets already logs details
            logger.error(
                "Failed to load critical FAISS assets (index or ID map).\n"
                "Expected at:\n"
                f"  Index path: {resolved_idx_path}\n"
                f"  ID map path: {resolved_map_path}\n"
                "Please ensure these files exist or run 'leanexplore data fetch' to download the data toolchain."
            )
            sys.exit(1)

        # Database connection
        # Check for DB file existence before creating engine if it's a file-based SQLite DB
        is_file_db = db_url.startswith("sqlite:///")
        db_file_path = None
        if is_file_db:
            # Extract file path from sqlite:/// URL
            db_file_path_str = db_url[len("sqlite///"):]
            db_file_path = pathlib.Path(db_file_path_str)
            if not db_file_path.exists():
                logger.error(
                    f"Database file not found at the expected location: {db_file_path}\n"
                    "Please run 'leanexplore data fetch' to download the data toolchain."
                )
                sys.exit(1)

        engine = create_engine(db_url, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with SessionLocal() as session:
            ranked_results = perform_search(
                session=session,
                query_string=args.query,
                model=s_transformer_model,
                faiss_index=faiss_idx,
                text_chunk_id_map=id_map,
                faiss_k=faiss_k_cand,
                pagerank_weight=pr_weight,
                text_relevance_weight=sem_sim_weight,
                name_match_weight=name_match_w,
                selected_packages=args.packages,
                semantic_similarity_threshold=semantic_sim_thresh, # from defaults
                faiss_nprobe=faiss_nprobe_val # from defaults
            )

        print_results(ranked_results[:results_disp_limit])

    except FileNotFoundError as e: # Should be less common now with explicit checks
        logger.error(
            f"A required file was not found: {e.filename}.\n"
            "This could be an issue with configured paths or missing data.\n"
            "If this relates to core data assets, please try running 'leanexplore data fetch'."
        )
        sys.exit(1)
    except OperationalError as e_db:
        is_file_db_op_err = defaults.DEFAULT_DB_URL.startswith("sqlite:///")
        db_file_path_op_err = defaults.DEFAULT_DB_PATH
        if is_file_db_op_err and (
            "unable to open database file" in str(e_db).lower() or
            (db_file_path_op_err and not db_file_path_op_err.exists()) # Redundant check if previous one passed
        ):
            logger.error(
                f"Database connection failed: {e_db}\n"
                f"The database file appears to be missing or inaccessible at: {str(db_file_path_op_err.resolve()) if db_file_path_op_err else 'Unknown Path'}\n"
                "Please run 'leanexplore data fetch' to download or update the data toolchain."
            )
        else:
            logger.error(f"Database connection/operational error: {e_db}", exc_info=True)
        sys.exit(1)
    except SQLAlchemyError as e_sqla: # Catch other SQLAlchemy errors
        logger.error("A database error occurred during search: %s", e_sqla, exc_info=True)
        sys.exit(1)
    except Exception as e_general: # Catch-all for other unexpected critical errors
        logger.critical("An unexpected critical error occurred during search: %s", e_general, exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")


if __name__ == "__main__":
    main()