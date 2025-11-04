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
import torch
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from lean_explore.config import Config
from lean_explore.models import Declaration

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Detect the best available device for PyTorch operations.

    Returns:
        Device string: 'cuda' if CUDA GPU available, 'mps' if Apple Silicon,
        otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


async def _load_embeddings_from_database(
    session: AsyncSession, embedding_field: str
) -> tuple[list[int], np.ndarray]:
    """Load embeddings and IDs from the database.

    Args:
        session: Async database session.
        embedding_field: Name of the embedding field to load
            (e.g., 'informalization_embedding').

    Returns:
        Tuple of (declaration_ids, embeddings_array) where embeddings_array
        is a numpy array of shape (num_declarations, embedding_dimension).
    """
    stmt = select(Declaration.id, getattr(Declaration, embedding_field)).where(
        getattr(Declaration, embedding_field).isnot(None)
    )
    result = await session.execute(stmt)
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
        engine: Async database engine.
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

    async with AsyncSession(engine) as session:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                "Building FAISS indices", total=len(embedding_fields)
            )

            for embedding_field in embedding_fields:
                logger.info(f"Processing {embedding_field}...")

                declaration_ids, embeddings = await _load_embeddings_from_database(
                    session, embedding_field
                )

                if len(declaration_ids) == 0:
                    logger.warning(f"Skipping {embedding_field} (no data)")
                    progress.update(task, advance=1)
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

                progress.update(task, advance=1)

    logger.info("All FAISS indices built successfully")
