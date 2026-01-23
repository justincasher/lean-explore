"""Build FAISS index from declaration embeddings.

This module creates a FAISS HNSW index for semantic search from embeddings
stored in the database. The index is built using PyTorch for efficient
device-accelerated processing.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session

from lean_explore.config import Config
from lean_explore.models import Declaration

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Detect if CUDA GPU is available for FAISS.

    Returns:
        Device string: 'cuda' if CUDA GPU available, otherwise 'cpu'.
        Note: FAISS doesn't support MPS, so Apple Silicon uses CPU.
    """
    if faiss.get_num_gpus() > 0:
        device = "cuda"
        logger.info("Using CUDA GPU for FAISS")
    else:
        device = "cpu"
        logger.info("Using CPU for FAISS")
    return device


def _load_embeddings_from_database(
    session: Session, embedding_field: str
) -> tuple[list[int], np.ndarray]:
    """Load embeddings and IDs from the database.

    Args:
        session: Sync database session.
        embedding_field: Name of the embedding field to load
            (e.g., 'informalization_embedding').

    Returns:
        Tuple of (declaration_ids, embeddings_array) where embeddings_array
        is a numpy array of shape (num_declarations, embedding_dimension).
    """
    stmt = select(Declaration.id, getattr(Declaration, embedding_field)).where(
        getattr(Declaration, embedding_field).isnot(None)
    )
    result = session.execute(stmt)
    rows = list(result.all())

    if not rows:
        logger.warning(f"No declarations found with {embedding_field}")
        return [], np.array([])

    declaration_ids = [row.id for row in rows]
    embeddings_list = [row[1] for row in rows]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)

    logger.info(
        f"Loaded {len(declaration_ids)} embeddings with dimension "
        f"{embeddings_array.shape[1]}"
    )

    return declaration_ids, embeddings_array


def _build_faiss_index(embeddings: np.ndarray, device: str) -> faiss.Index:
    """Build a FAISS HNSW index from embeddings.

    Args:
        embeddings: Numpy array of embeddings, shape (num_vectors, dimension).
        device: Device to use ('cuda', 'mps', or 'cpu').

    Returns:
        FAISS index with HNSW graph structure for fast similarity search.
    """
    num_vectors = embeddings.shape[0]
    dimension = embeddings.shape[1]

    logger.info(f"Building FAISS HNSW index for {num_vectors} vectors...")

    # M=32 connections per layer, efConstruction=40 for index quality
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 40

    if device == "cuda" and faiss.get_num_gpus() > 0:
        logger.info("Moving FAISS index to GPU")
        resource = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resource, 0, index)

    index.add(embeddings)

    logger.info("FAISS index built successfully")
    return index


async def build_faiss_indices(
    engine: AsyncEngine,
    output_directory: Path | None = None,
) -> None:
    """Build FAISS indices for all embedding types.

    This function creates FAISS HNSW indices for each embedding type
    (name, informalization, source_text, docstring) and saves them to disk
    along with ID mappings.

    Args:
        engine: Async database engine (URL extracted for sync access).
        output_directory: Directory to save indices. Defaults to active data path.
    """
    if output_directory is None:
        output_directory = Config.ACTIVE_DATA_PATH

    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving indices to {output_directory}")

    device = _get_device()

    embedding_fields = [
        "name_embedding",
        "informalization_embedding",
        "source_text_embedding",
        "docstring_embedding",
    ]

    # Use sync engine to avoid aiosqlite issues with binary data
    sync_url = str(engine.url).replace("sqlite+aiosqlite", "sqlite")
    sync_engine = create_engine(sync_url)

    with Session(sync_engine) as session:
        for i, embedding_field in enumerate(embedding_fields, 1):
            logger.info(
                f"Processing {embedding_field} ({i}/{len(embedding_fields)})..."
            )

            declaration_ids, embeddings = _load_embeddings_from_database(
                session, embedding_field
            )

            if len(declaration_ids) == 0:
                logger.warning(f"Skipping {embedding_field} (no data)")
                continue

            index = _build_faiss_index(embeddings, device)

            # Move GPU index back to CPU for serialization
            if device == "cuda" and isinstance(index, faiss.GpuIndex):
                index = faiss.index_gpu_to_cpu(index)

            index_filename = embedding_field.replace("_embedding", "_faiss.index")
            index_path = output_directory / index_filename
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")

            ids_map_filename = embedding_field.replace(
                "_embedding", "_faiss_ids_map.json"
            )
            ids_map_path = output_directory / ids_map_filename
            with open(ids_map_path, "w") as file:
                json.dump(declaration_ids, file)
            logger.info(f"Saved ID mapping to {ids_map_path}")

    sync_engine.dispose()
    logger.info("All FAISS indices built successfully")
