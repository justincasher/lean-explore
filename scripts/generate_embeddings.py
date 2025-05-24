# File: scripts/generate_embeddings.py

"""Generates and saves embeddings from text items in a JSON file.

Reads a JSON file with 'id' and 'text' fields per entry, uses a
sentence-transformer model to compute embeddings for 'text' values,
and applies an optional processing limit.

Outputs an NPZ file with 'ids' (1D string array) and 'embeddings'
(2D float32 array), suitable for efficient storage and retrieval.
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

# --- Dependency Imports ---
try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please run: pip install torch sentence-transformers numpy tqdm",
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
# Reduce verbosity from sentence_transformers library
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# --- Constants ---
DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_OUTPUT_FILENAME = "generated_embeddings.npz"


def select_device(requested_device: Optional[str] = None) -> str:
    """Selects the most appropriate PyTorch device (mps, cuda, cpu).

    If a specific device is requested and available, it's used. Otherwise,
    it attempts to auto-detect CUDA, then MPS (for Apple Silicon), and
    finally falls back to CPU.

    Args:
        requested_device: The device explicitly requested by the user
            (e.g., "cuda", "mps", "cpu"). If None or empty,
            auto-detection is performed.

    Returns:
        str: A string representing the selected PyTorch device ("cuda",
            "mps", or "cpu").
    """
    if requested_device:
        device_lower = requested_device.lower()
        if device_lower == "cuda":
            if torch.cuda.is_available():
                logger.info("CUDA device explicitly requested and available.")
                return "cuda"
            logger.warning(
                "CUDA device requested but not available. Falling back to auto-detect."
            )
        elif device_lower == "mps":
            # Check for MPS (Apple Silicon GPU)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS device explicitly requested and available.")
                return "mps"
            logger.warning(
                "MPS device requested but not available. Falling back to auto-detect."
            )
        elif device_lower == "cpu":
            logger.info("CPU device explicitly requested.")
            return "cpu"
        else:
            logger.warning(
                "Unknown device '%s' requested. Falling back to auto-detection.",
                requested_device,
            )

    # Auto-detection if no specific device requested or if requested device
    # not available
    if torch.cuda.is_available():
        logger.info("CUDA device auto-detected and selected.")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS device auto-detected and selected.")
        return "mps"

    logger.info("No GPU acceleration (CUDA/MPS) detected. Using CPU.")
    return "cpu"


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """Loads a Sentence Transformer model onto the specified device.

    Args:
        model_name: The identifier of the sentence-transformer model to load
            (e.g., from Hugging Face Model Hub or a local path).
        device: The PyTorch device string ('cuda', 'mps', 'cpu') where the
            model will be loaded.

    Returns:
        SentenceTransformer: An instance of the loaded SentenceTransformer model.

    Raises:
        RuntimeError: If the model fails to load for any reason.
    """
    logger.info(
        "Loading sentence transformer model '%s' onto device '%s'...",
        model_name,
        device,
    )
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Model '%s' loaded successfully. Max sequence length: %d",
            model_name,
            model.max_seq_length,
        )
        return model
    except Exception as e:
        logger.error("Failed to load model '%s': %s", model_name, e, exc_info=True)
        raise RuntimeError(
            f"Model loading failed for '{model_name}' on device '{device}'"
        ) from e


def generate_embeddings_from_file(
    input_file_path: pathlib.Path,
    output_file_path: pathlib.Path,
    model_name: str,
    device: str,
    batch_size: int,
    limit: Optional[int] = None,
) -> None:
    """Loads text data, generates embeddings, and saves them to an NPZ file.

    Reads a JSON file (expected to be a list of objects, each with 'id'
    and 'text' fields), generates embeddings for each 'text' field using
    the specified sentence-transformer model, and writes a new NPZ file
    containing 'ids' and 'embeddings' arrays. An optional limit on the
    number of items processed can be applied.

    Args:
        input_file_path: Path to the input JSON file.
        output_file_path: Path to save the output NPZ file.
        model_name: Name of the sentence-transformer model.
        device: PyTorch device to use for embedding generation.
        batch_size: Batch size for the embedding model's encode function.
        limit: Optional maximum number of items from the input file to process.
            If None or non-positive, all items are processed.

    Returns:
        None. The function exits on critical errors or if no valid data is found.
    """
    logger.info("Attempting to read input data from: %s", input_file_path)
    try:
        with open(input_file_path, encoding="utf-8") as f:
            input_data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from input file %s: %s", input_file_path, e)
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Could not read input file %s: %s", input_file_path, e, exc_info=True
        )
        sys.exit(1)

    if not isinstance(input_data, list):
        logger.error(
            "Input JSON data is not a list as expected. Found type: %s",
            type(input_data).__name__,
        )
        sys.exit(1)

    if limit is not None and limit > 0:
        if len(input_data) > limit:
            logger.info(
                "Applying limit: processing only the first %d of %d items.",
                limit,
                len(input_data),
            )
            input_data = input_data[:limit]
        else:
            logger.info(
                "Limit of %d specified, but input only contains %d items. "
                "Processing all available items.",
                limit,
                len(input_data),
            )
    elif limit is not None:  # limit <= 0
        logger.warning("Limit value (%d) is not positive. Processing all items.", limit)

    if not input_data:
        logger.info(
            "Input file is empty or no data remains after applying limit. "
            "No output NPZ file will be created."
        )
        return

    original_ids: List[str] = []
    texts_to_embed: List[str] = []

    for i, record in enumerate(input_data):
        if not isinstance(record, dict):
            logger.warning(
                "Skipping item at index %d as it is not a dictionary: %s", i, record
            )
            continue
        record_id = record.get("id")
        record_text = record.get("text")

        if record_id is None or record_text is None:
            logger.warning(
                "Skipping record at index %d due to missing 'id' or 'text' field. "
                "Record: %s",
                i,
                record,
            )
            continue
        original_ids.append(str(record_id))
        texts_to_embed.append(str(record_text))

    if not texts_to_embed:
        logger.info(
            "No valid records with 'id' and 'text' found after filtering. "
            "No output NPZ file will be created."
        )
        return

    logger.info(
        "Successfully extracted %d text items for embedding.", len(texts_to_embed)
    )

    model = load_embedding_model(model_name, device)

    logger.info(
        "Generating embeddings for %d items (model batch size: %d)...",
        len(texts_to_embed),
        batch_size,
    )
    # Disabling type check for model.encode due to dynamic nature of SentenceTransformer
    embeddings_matrix: np.ndarray = model.encode(  # type: ignore
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,  # tqdm progress bar
        convert_to_numpy=True,
    )
    logger.info(
        "Generated %d embeddings, each with dimension %d.",
        embeddings_matrix.shape[0],
        embeddings_matrix.shape[1],
    )

    if len(original_ids) != embeddings_matrix.shape[0]:
        logger.error(
            "Critical mismatch after embedding: IDs count (%d) vs Embeddings count "
            "(%d). Aborting to prevent data corruption.",
            len(original_ids),
            embeddings_matrix.shape[0],
        )
        sys.exit(1)

    # 'object' dtype handles variable-length strings efficiently in NumPy.
    ids_array = np.array(original_ids, dtype=object)

    logger.info(
        "Attempting to write %d IDs and their embeddings to: %s (NPZ format)",
        len(ids_array),
        output_file_path,
    )
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_file_path, ids=ids_array, embeddings=embeddings_matrix
        )
        logger.info("Embeddings and IDs successfully written to NPZ output file.")
    except OSError as e:
        logger.error(
            "Failed to write NPZ file to %s: %s", output_file_path, e, exc_info=True
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "An unexpected error occurred during NPZ file writing: %s", e, exc_info=True
        )
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for embedding generation.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings from a JSON input and save to NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Path to input JSON file with texts (e.g., 'data/embedding_input.json').",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path()
        / "data"
        / DEFAULT_OUTPUT_FILENAME,  # Suggest saving to data/
        help="Path to save output NPZ file with IDs and embeddings.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformer model name (Hugging Face Hub or local path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,  # Auto-detects if not specified
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on ('cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Batch size for the embedding model's encode function.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Optional: Process only the first N items from the input file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    chosen_device = select_device(cli_args.device)

    effective_output_file = cli_args.output_file
    # Advise on .npz extension if not present; np.savez_compressed might append.
    if effective_output_file.suffix.lower() != ".npz":
        suggested_name = effective_output_file.with_suffix(".npz")
        logger.warning(
            "Output file '%s' does not have an .npz extension. "
            "NumPy will save in NPZ format, possibly as '%s' or '%s.npz'. "
            "Consider using an .npz extension explicitly for clarity (e.g., '%s').",
            effective_output_file,
            suggested_name.name,
            effective_output_file.name,
            suggested_name,
        )

    generate_embeddings_from_file(
        input_file_path=cli_args.input_file.resolve(),
        output_file_path=effective_output_file.resolve(),
        model_name=cli_args.model_name,
        device=chosen_device,
        batch_size=cli_args.batch_size,
        limit=cli_args.limit,
    )
    logger.info("--- Embedding generation process finished ---")
