# File: scripts/build_faiss_index.py

"""Builds and saves a FAISS index from generated embeddings.

Reads an NPZ file containing 'ids' (string identifiers) and 'embeddings'
(NumPy array of float32 vectors). It then constructs a FAISS index from these
embeddings and saves the index to a specified file. Additionally, it saves the
'ids' array as a JSON list, which maps the FAISS index's internal sequential
IDs (0 to N-1) back to the original string identifiers.

The script can optionally use a GPU for FAISS operations if faiss-gpu is
installed and a GPU is available, with a fallback to CPU operations.
"""

import argparse
import json
import logging
import pathlib
import sys
import time

try:
    import faiss
    import numpy as np
except ImportError as e:
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please install them by running: pip install numpy faiss-cpu\n"
        "(or faiss-gpu for CUDA-enabled GPU support).",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_INDEX_FILE = "main_faiss.index"
DEFAULT_OUTPUT_MAP_FILE = "faiss_ids_map.json"
DEFAULT_FAISS_INDEX_TYPE = "IndexFlatL2"
DEFAULT_IVF_NLIST = 100
DEFAULT_GPU_DEVICE = 0


def build_and_save_index(
    input_npz_path: pathlib.Path,
    output_index_path: pathlib.Path,
    output_map_path: pathlib.Path,
    faiss_index_type: str = DEFAULT_FAISS_INDEX_TYPE,
    ivf_nlist: int = DEFAULT_IVF_NLIST,
    force_cpu: bool = False,
    gpu_device: int = DEFAULT_GPU_DEVICE,
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
        force_cpu: If True, forces CPU usage even if a GPU is available.
        gpu_device: The ID of the GPU device to use if GPU is enabled.

    Returns:
        None. The function will call sys.exit(1) on critical errors.
    """
    logger.info(
        "Starting FAISS index building process at %s",
        time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
    logger.info("Loading embeddings and IDs from: %s", input_npz_path.resolve())
    try:
        data = np.load(input_npz_path, allow_pickle=True)
        ids_array = data.get("ids")
        embeddings_matrix = data.get("embeddings")

        if ids_array is None:
            logger.error("'ids' array not found in NPZ file: %s", input_npz_path)
            sys.exit(1)
        if embeddings_matrix is None:
            logger.error("'embeddings' array not found in NPZ file: %s", input_npz_path)
            sys.exit(1)

        if not ids_array.size:
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
    except Exception as e:
        logger.error(
            "Error loading NPZ file %s: %s", input_npz_path.resolve(), e, exc_info=True
        )
        sys.exit(1)

    num_embeddings, dimension = embeddings_matrix.shape
    logger.info(
        "Loaded %d IDs and %d embeddings, each with dimension %d.",
        len(ids_array),
        num_embeddings,
        dimension,
    )

    if embeddings_matrix.dtype != np.float32:
        logger.info(
            "Casting embeddings from %s to np.float32.", embeddings_matrix.dtype
        )
        embeddings_matrix = embeddings_matrix.astype(np.float32)

    gpu_available_count = 0
    try:
        gpu_available_count = faiss.get_num_gpus()
    except AttributeError:
        logger.info(
            "faiss.get_num_gpus() not found. "
            "Assuming faiss-cpu is installed or GPU support is unavailable."
        )
    except faiss.FaissException as e_faiss_gpu:
        logger.warning(
            "FAISS Exception while checking for GPUs: %s. Assuming no GPU available.",
            e_faiss_gpu,
        )

    effective_use_gpu = (gpu_available_count > 0) and (not force_cpu)

    if effective_use_gpu:
        if gpu_device >= gpu_available_count:
            logger.warning(
                "Selected GPU device ID %d is not available (found %d GPUs). "
                "Falling back to GPU device 0.",
                gpu_device,
                gpu_available_count,
            )
            actual_gpu_device = 0
        else:
            actual_gpu_device = gpu_device
        logger.info(
            "FAISS GPU support detected (%d GPU(s) available). "
            "Attempting to use GPU device %d.",
            gpu_available_count,
            actual_gpu_device,
        )
    elif force_cpu:
        logger.info("GPU usage forced off by user (--force-cpu). Using CPU.")
    else:
        logger.info("No FAISS-compatible GPU detected or available. Using CPU.")

    cpu_faiss_index: faiss.Index
    logger.info(
        "Initializing base CPU FAISS index structure of type: %s", faiss_index_type
    )
    try:
        if faiss_index_type == "IndexFlatL2":
            cpu_faiss_index = faiss.IndexFlatL2(dimension)
        elif faiss_index_type == "IndexFlatIP":
            logger.info(
                "Using IndexFlatIP for inner product. "
                "Ensure vectors are L2 normalized for cosine similarity."
            )
            cpu_faiss_index = faiss.IndexFlatIP(dimension)
        elif faiss_index_type == "IndexIVFFlat":
            nlist_actual = ivf_nlist
            min_points_for_train = nlist_actual * 39
            if num_embeddings > 0 and num_embeddings < min_points_for_train:
                nlist_suggested = max(1, int(np.sqrt(num_embeddings) / 2))
                if nlist_suggested < nlist_actual:
                    logger.warning(
                        "Number of embeddings (%d) is low for the specified "
                        "nlist (%d). Recommended minimum for training is ~%d. "
                        "Consider reducing nlist (e.g., to around %d) or using "
                        "a Flat index for potentially better results.",
                        num_embeddings,
                        nlist_actual,
                        min_points_for_train,
                        nlist_suggested,
                    )
            quantizer = faiss.IndexFlatL2(dimension)
            cpu_faiss_index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist_actual, faiss.METRIC_L2
            )
        else:
            logger.error("Unsupported FAISS index type specified: %s", faiss_index_type)
            sys.exit(1)
    except Exception as e_init:
        logger.error(
            "Error initializing FAISS index structure: %s", e_init, exc_info=True
        )
        sys.exit(1)

    final_index_to_save = cpu_faiss_index
    gpu_processing_successful = False

    if effective_use_gpu:
        gpu_resources = None
        cloned_gpu_index = None
        try:
            logger.info(
                "Attempting to use GPU device %d for FAISS operations.",
                actual_gpu_device,
            )
            gpu_resources = faiss.StandardGpuResources()

            logger.info("Cloning CPU index to GPU device %d...", actual_gpu_device)
            cloned_gpu_index = faiss.index_cpu_to_gpu(
                gpu_resources, actual_gpu_device, cpu_faiss_index
            )
            logger.info("Successfully cloned index to GPU.")

            if (
                hasattr(cloned_gpu_index, "is_trained")
                and not cloned_gpu_index.is_trained
            ):
                logger.info(
                    "Training index on GPU %d with %d vectors...",
                    actual_gpu_device,
                    num_embeddings,
                )
                cloned_gpu_index.train(embeddings_matrix)
                logger.info("GPU training complete.")

            if num_embeddings > 0:
                logger.info(
                    "Adding %d embeddings to index on GPU %d...",
                    num_embeddings,
                    actual_gpu_device,
                )
                cloned_gpu_index.add(embeddings_matrix)
                logger.info(
                    "GPU adding complete. Index ntotal on GPU: %d",
                    cloned_gpu_index.ntotal,
                )

            logger.info(
                "Moving index from GPU %d back to CPU for saving...",
                actual_gpu_device,
            )
            final_index_to_save = faiss.index_gpu_to_cpu(cloned_gpu_index)
            logger.info(
                "Index successfully processed on GPU and moved to CPU. "
                "Final ntotal: %d",
                final_index_to_save.ntotal,
            )
            gpu_processing_successful = True

        except Exception as e_gpu:
            logger.warning(
                "GPU operations failed: %s. Falling back to CPU operations.",
                e_gpu,
                exc_info=True,
            )
            if hasattr(cpu_faiss_index, "reset") and isinstance(
                cpu_faiss_index, faiss.IndexIVF
            ):
                logger.info("Resetting original CPU IVF index for clean CPU fallback.")
                cpu_faiss_index.reset()
            final_index_to_save = cpu_faiss_index
        finally:
            if cloned_gpu_index is not None:
                del cloned_gpu_index
            if gpu_resources is not None:
                del gpu_resources
            logger.debug("GPU resources (if any) have been released.")

    if not gpu_processing_successful:
        logger.info("Processing FAISS index on CPU...")
        final_index_to_save = cpu_faiss_index

        if (
            hasattr(final_index_to_save, "is_trained")
            and not final_index_to_save.is_trained
        ):
            if num_embeddings > 0:
                logger.info("Training index on CPU with %d vectors...", num_embeddings)
                final_index_to_save.train(embeddings_matrix)
                logger.info("CPU training complete.")
            else:
                logger.warning(
                    "No embeddings provided to train %s on CPU; index will be "
                    "untrained.",
                    faiss_index_type,
                )

        if num_embeddings > 0:
            if final_index_to_save.ntotal == 0:
                logger.info("Adding %d embeddings to index on CPU...", num_embeddings)
                final_index_to_save.add(embeddings_matrix)
                logger.info(
                    "CPU adding complete. Index ntotal on CPU: %d",
                    final_index_to_save.ntotal,
                )
            elif (
                final_index_to_save.ntotal != num_embeddings
                and hasattr(final_index_to_save, "is_trained")
                and final_index_to_save.is_trained
            ):
                logger.info(
                    "CPU index is trained but ntotal (%d) != num_embeddings (%d). "
                    "Attempting to add embeddings on CPU.",
                    final_index_to_save.ntotal,
                    num_embeddings,
                )

                logger.warning(
                    "This scenario (trained index, ntotal > 0 but != num_embeddings) "
                    "during CPU fallback might indicate complex partial GPU failure. "
                    "Proceeding with add, but verify index content if issues arise."
                )

        elif num_embeddings == 0:
            logger.info(
                "No embeddings to add to the index as the input matrix was empty."
            )

    try:
        logger.info("Saving FAISS index to: %s", output_index_path.resolve())
        output_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(final_index_to_save, str(output_index_path))

        logger.info(
            "Saving ID map (list of %d IDs) to: %s",
            len(ids_array),
            output_map_path.resolve(),
        )
        output_map_path.parent.mkdir(parents=True, exist_ok=True)
        id_list = ids_array.tolist()
        with open(output_map_path, "w", encoding="utf-8") as f:
            json.dump(id_list, f, indent=2)

        logger.info("FAISS index and ID map created and saved successfully.")

    except Exception as e:
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
        default=pathlib.Path("data") / "generated_embeddings.npz",
        help="Path to input NPZ file (e.g., 'data/generated_embeddings.npz') "
        "containing 'ids' and 'embeddings' arrays.",
    )
    parser.add_argument(
        "--output-index-file",
        type=pathlib.Path,
        default=pathlib.Path() / "data" / DEFAULT_OUTPUT_INDEX_FILE,
        help="Path to save the serialized FAISS index file.",
    )
    parser.add_argument(
        "--output-map-file",
        type=pathlib.Path,
        default=pathlib.Path() / "data" / DEFAULT_OUTPUT_MAP_FILE,
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
        default=DEFAULT_IVF_NLIST,
        help="Number of Voronoi cells (nlist) for IndexIVFFlat. "
        "Crucial for performance/accuracy. Typically sqrt(N) to 4*sqrt(N). "
        "Ignored for other index types.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if a GPU is available.",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=DEFAULT_GPU_DEVICE,
        help="GPU device ID to use if GPU processing is enabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    script_args = parse_arguments()

    build_and_save_index(
        input_npz_path=script_args.input_npz_file.resolve(),
        output_index_path=script_args.output_index_file.resolve(),
        output_map_path=script_args.output_map_file.resolve(),
        faiss_index_type=script_args.faiss_index_type,
        ivf_nlist=script_args.ivf_nlist,
        force_cpu=script_args.force_cpu,
        gpu_device=script_args.gpu_device,
    )
    logger.info(
        "--- FAISS index building process finished successfully at %s ---",
        time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
