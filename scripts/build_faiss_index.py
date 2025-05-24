# File: scripts/build_faiss_index.py

"""Builds and saves a FAISS index from generated embeddings.

Reads an NPZ file containing 'ids' (string identifiers) and 'embeddings'
(NumPy array of float32 vectors). It then constructs a FAISS index from these
embeddings and saves the index to a specified file. Additionally, it saves the
'ids' array as a JSON list, which maps the FAISS index's internal sequential
IDs (0 to N-1) back to the original string identifiers.
"""

import argparse
import json
import logging
import pathlib
import sys
import time

# --- Dependency Imports ---
try:
    import faiss
    import numpy as np
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please install them by running: pip install numpy faiss-cpu\n"
        "(or faiss-gpu for CUDA-enabled GPU support, though this script"
        " primarily uses CPU for index building).",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_OUTPUT_INDEX_FILE = "main_faiss.index"
DEFAULT_OUTPUT_MAP_FILE = "faiss_ids_map.json"
DEFAULT_FAISS_INDEX_TYPE = "IndexFlatL2"


def build_and_save_index(
    input_npz_path: pathlib.Path,
    output_index_path: pathlib.Path,
    output_map_path: pathlib.Path,
    faiss_index_type: str = DEFAULT_FAISS_INDEX_TYPE,
    ivf_nlist: int = 100,
) -> None:
    """Loads embeddings and IDs, builds a FAISS index, and saves them.

    Args:
        input_npz_path: Path to the input NPZ file containing 'ids' and
            'embeddings' arrays.
        output_index_path: Path to save the serialized FAISS index.
        output_map_path: Path to save the JSON file mapping FAISS internal
            indices to original string IDs.
        faiss_index_type: Type of FAISS index to build (e.g., "IndexFlatL2",
            "IndexFlatIP", "IndexIVFFlat").
        ivf_nlist: Number of Voronoi cells if using an IVF-based index
            like "IndexIVFFlat". Ignored for other index types.

    Returns:
        None. The function will call sys.exit(1) on critical errors.
    """
    logger.info(
        "Starting FAISS index building process at %s",
        time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
    logger.info("Loading embeddings and IDs from: %s", input_npz_path.resolve())
    try:
        # allow_pickle=True is necessary if 'ids' array was saved with dtype=object
        data = np.load(input_npz_path, allow_pickle=True)
        ids_array = data.get("ids")
        embeddings_matrix = data.get("embeddings")

        if ids_array is None:
            logger.error("'ids' array not found in NPZ file: %s", input_npz_path)
            sys.exit(1)
        if embeddings_matrix is None:
            logger.error("'embeddings' array not found in NPZ file: %s", input_npz_path)
            sys.exit(1)

        if not ids_array.size:  # Check if array is empty
            logger.error(
                "'ids' array in NPZ file '%s' is empty. Cannot build index.",
                input_npz_path,
            )
            sys.exit(1)
        if embeddings_matrix.ndim != 2 or not embeddings_matrix.shape[0]:
            logger.error(
                "'embeddings' array in NPZ file '%s' is empty or not 2D. "
                "Shape: %s. Cannot build index.",
                input_npz_path,
                embeddings_matrix.shape,
            )
            sys.exit(1)

        if len(ids_array) != embeddings_matrix.shape[0]:
            logger.error(
                "Mismatch between number of IDs (%d) and "
                "number of embeddings (%d) in %s.",
                len(ids_array),
                embeddings_matrix.shape[0],
                input_npz_path,
            )
            sys.exit(1)

    except FileNotFoundError:
        logger.error("Input NPZ file not found: %s", input_npz_path.resolve())
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "Error loading NPZ file %s: %s", input_npz_path.resolve(), e, exc_info=True
        )
        sys.exit(1)

    num_embeddings, dimension = embeddings_matrix.shape
    logger.info(
        "Loaded %d IDs and %d embeddings, each with dimension %d.",
        len(ids_array),  # Use len(ids_array) for count of IDs
        num_embeddings,
        dimension,
    )

    # Ensure embeddings are float32, as FAISS typically expects this
    if embeddings_matrix.dtype != np.float32:
        logger.info(
            "Casting embeddings from %s to np.float32.", embeddings_matrix.dtype
        )
        embeddings_matrix = embeddings_matrix.astype(np.float32)

    logger.info("Building FAISS index of type: %s", faiss_index_type)
    faiss_index: faiss.Index  # Type hint for clarity

    try:
        if faiss_index_type == "IndexFlatL2":
            faiss_index = faiss.IndexFlatL2(dimension)
        elif faiss_index_type == "IndexFlatIP":
            # For cosine similarity; assumes vectors are L2 normalized.
            logger.info(
                "Using IndexFlatIP for inner product (cosine similarity on "
                "normalized vectors)."
            )
            faiss_index = faiss.IndexFlatIP(dimension)
        elif faiss_index_type == "IndexIVFFlat":
            nlist_actual = ivf_nlist
            # FAISS needs at least k_min (typically 39) points per cell for training.
            # Check if num_embeddings is sufficient for the chosen nlist.
            # This is a heuristic.
            min_points_for_train = nlist_actual * 39
            if num_embeddings > 0 and num_embeddings < min_points_for_train:
                # Heuristic for nlist suggestion if not enough data.
                nlist_suggested = max(1, int(np.sqrt(num_embeddings) / 2))
                if nlist_suggested < nlist_actual:
                    logger.warning(
                        "Number of embeddings (%d) is low for the specified "
                        "nlist (%d). Minimum recommended for training is ~%d. "
                        "Consider reducing nlist (e.g., to around %d) "
                        "or using IndexFlatL2/IndexFlatIP for better results.",
                        num_embeddings,
                        nlist_actual,
                        min_points_for_train,
                        nlist_suggested,
                    )
            # Using L2 for quantizer by default
            quantizer = faiss.IndexFlatL2(dimension)
            faiss_index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist_actual, faiss.METRIC_L2
            )

            if num_embeddings > 0:
                if not faiss_index.is_trained:
                    logger.info(
                        "Training %s index with %d vectors...",
                        faiss_index_type,
                        num_embeddings,
                    )
                    # For very large N, FAISS recommends training on a subset.
                    # Here, we train on all available embeddings.
                    faiss_index.train(embeddings_matrix)
                    logger.info("Index training complete.")
            else:  # num_embeddings == 0
                logger.warning(
                    "No embeddings provided; %s will be created untrained.",
                    faiss_index_type,
                )
        else:
            logger.error("Unsupported FAISS index type specified: %s", faiss_index_type)
            sys.exit(1)

        if num_embeddings > 0:
            logger.info("Adding embeddings to the FAISS index...")
            faiss_index.add(embeddings_matrix)
            logger.info(
                "Successfully added %d vectors to the index.", faiss_index.ntotal
            )
        else:
            # This case is mainly for IndexIVFFlat if created with 0 embeddings.
            # Other scenarios should have exited earlier if embeddings were empty.
            logger.info(
                "No embeddings to add to the index as the input matrix was "
                "empty or training failed to produce a usable index state."
            )

        logger.info("Saving FAISS index to: %s", output_index_path.resolve())
        output_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, str(output_index_path))

        logger.info(
            "Saving ID map (list of %d IDs) to: %s",
            len(ids_array),
            output_map_path.resolve(),
        )
        output_map_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert NumPy array of string objects to Python list of strings
        id_list = ids_array.tolist()
        with open(output_map_path, "w", encoding="utf-8") as f:
            json.dump(id_list, f, indent=2)  # Indent for readability

        logger.info("FAISS index and ID map created and saved successfully.")

    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            "An error occurred during FAISS index building or saving: %s",
            e,
            exc_info=True,
        )
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Builds a FAISS index from embeddings in an NPZ file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-npz-file",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Path to input NPZ file (e.g., 'data/generated_embeddings.npz') "
        "containing 'ids' and 'embeddings' arrays.",
    )
    parser.add_argument(
        "--output-index-file",
        type=pathlib.Path,
        default=pathlib.Path()
        / "data"
        / DEFAULT_OUTPUT_INDEX_FILE,  # Suggest saving to data/
        help="Path to save the serialized FAISS index file.",
    )
    parser.add_argument(
        "--output-map-file",
        type=pathlib.Path,
        default=pathlib.Path()
        / "data"
        / DEFAULT_OUTPUT_MAP_FILE,  # Suggest saving to data/
        help="Path to save JSON ID map file.",
    )
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        default=DEFAULT_FAISS_INDEX_TYPE,
        choices=["IndexFlatL2", "IndexFlatIP", "IndexIVFFlat"],
        help="Type of FAISS index: IndexFlatL2 (Euclidean), "
        "IndexFlatIP (cosine similarity - ensure normalized embeddings), "
        "IndexIVFFlat (for larger datasets, requires training).",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=100,
        help="Number of Voronoi cells (nlist) for IndexIVFFlat. "
        "Crucial for performance/accuracy. Typically sqrt(N) to 4*sqrt(N). "
        "Ignored for other index types.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    script_args = parse_arguments()

    # Use .resolve() for absolute paths, helpful for clarity in logs and operations
    build_and_save_index(
        input_npz_path=script_args.input_npz_file.resolve(),
        output_index_path=script_args.output_index_file.resolve(),
        output_map_path=script_args.output_map_file.resolve(),
        faiss_index_type=script_args.faiss_index_type,
        ivf_nlist=script_args.ivf_nlist,
    )
    logger.info(
        "--- FAISS index building process finished successfully at %s ---",
        time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
