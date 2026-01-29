"""Build search indices from declaration data.

This module creates:
1. FAISS IVF index for semantic search from embeddings
2. BM25 indices for lexical search on declaration names

IVF (Inverted File) uses k-means clustering for efficient approximate
nearest neighbor search with controllable recall.
"""

import json
import logging
import re
from pathlib import Path

import bm25s
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
    """Build a FAISS IVF index from embeddings.

    Args:
        embeddings: Numpy array of embeddings, shape (num_vectors, dimension).
        device: Device to use ('cuda', 'mps', or 'cpu').

    Returns:
        FAISS IVF index for fast approximate nearest neighbor search.
    """
    num_vectors = embeddings.shape[0]
    dimension = embeddings.shape[1]

    # Number of clusters: sqrt(n) is a good heuristic, minimum 256
    nlist = max(256, int(np.sqrt(num_vectors)))

    logger.info(
        f"Building FAISS IVF index for {num_vectors} vectors with {nlist} clusters..."
    )

    # Use inner product (cosine similarity on normalized vectors)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    if device == "cuda" and faiss.get_num_gpus() > 0:
        logger.info("Training IVF index on GPU")
        resource = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resource, 0, index)
        gpu_index.train(embeddings)
        gpu_index.add(embeddings)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        logger.info("Training IVF index on CPU")
        index.train(embeddings)
        index.add(embeddings)

    logger.info("FAISS IVF index built successfully")
    return index


async def build_faiss_indices(
    engine: AsyncEngine,
    output_directory: Path | None = None,
) -> None:
    """Build FAISS index for informalization embeddings.

    This function creates a FAISS IVF index for informalization embeddings
    and saves it to disk along with ID mappings.

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
        "informalization_embedding",
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


def _tokenize_spaced(text: str) -> list[str]:
    """Tokenize text with spacing on dots, underscores, and camelCase.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase word tokens.
    """
    if not text:
        return []
    text = text.replace(".", " ").replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"\w+", text.lower())


def _tokenize_raw(text: str) -> list[str]:
    """Tokenize text as single token (preserves dots).

    Args:
        text: Input text to tokenize.

    Returns:
        List with the full text as a single lowercase token.
    """
    if not text:
        return []
    return [text.lower()]


def _load_declaration_names(session: Session) -> tuple[list[int], list[str]]:
    """Load all declaration IDs and names from the database.

    Args:
        session: Sync database session.

    Returns:
        Tuple of (declaration_ids, declaration_names).
    """
    stmt = select(Declaration.id, Declaration.name)
    result = session.execute(stmt)
    rows = list(result.all())

    declaration_ids = [row.id for row in rows]
    declaration_names = [row.name or "" for row in rows]

    logger.info(f"Loaded {len(declaration_ids)} declarations for BM25 indexing")
    return declaration_ids, declaration_names


def _build_bm25_indices(
    declaration_names: list[str],
) -> tuple[bm25s.BM25, bm25s.BM25]:
    """Build BM25 indices over declaration names.

    Creates two indices:
    1. Spaced tokenization (splits on dots, underscores, camelCase)
    2. Raw tokenization (full name as single token)

    Args:
        declaration_names: List of declaration names.

    Returns:
        Tuple of (bm25_spaced, bm25_raw) indices.
    """
    logger.info("Building BM25 indices over declaration names...")

    corpus_spaced = [list(set(_tokenize_spaced(n))) for n in declaration_names]
    corpus_raw = [list(set(_tokenize_raw(n))) for n in declaration_names]

    bm25_spaced = bm25s.BM25(method="bm25+")
    bm25_spaced.index(corpus_spaced)
    logger.info("Built BM25 spaced index")

    bm25_raw = bm25s.BM25(method="bm25+")
    bm25_raw.index(corpus_raw)
    logger.info("Built BM25 raw index")

    return bm25_spaced, bm25_raw


async def build_bm25_indices(
    engine: AsyncEngine,
    output_directory: Path | None = None,
) -> None:
    """Build BM25 indices for declaration name search.

    This function creates BM25 indices for lexical search on declaration
    names and saves them to disk along with ID mappings.

    Args:
        engine: Async database engine (URL extracted for sync access).
        output_directory: Directory to save indices. Defaults to active data path.
    """
    if output_directory is None:
        output_directory = Config.ACTIVE_DATA_PATH

    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving BM25 indices to {output_directory}")

    sync_url = str(engine.url).replace("sqlite+aiosqlite", "sqlite")
    sync_engine = create_engine(sync_url)

    with Session(sync_engine) as session:
        declaration_ids, declaration_names = _load_declaration_names(session)

    if not declaration_ids:
        logger.warning("No declarations found for BM25 indexing")
        sync_engine.dispose()
        return

    bm25_spaced, bm25_raw = _build_bm25_indices(declaration_names)

    # Save BM25 indices
    bm25_spaced_path = output_directory / "bm25_name_spaced"
    bm25_spaced.save(str(bm25_spaced_path))
    logger.info(f"Saved BM25 spaced index to {bm25_spaced_path}")

    bm25_raw_path = output_directory / "bm25_name_raw"
    bm25_raw.save(str(bm25_raw_path))
    logger.info(f"Saved BM25 raw index to {bm25_raw_path}")

    # Save ID mapping (shared by both indices)
    ids_map_path = output_directory / "bm25_ids_map.json"
    with open(ids_map_path, "w") as file:
        json.dump(declaration_ids, file)
    logger.info(f"Saved BM25 ID mapping to {ids_map_path}")

    sync_engine.dispose()
    logger.info("All BM25 indices built successfully")
