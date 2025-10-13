"""Embedding generation client using sentence transformers."""

import asyncio
import logging

import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingResponse(BaseModel):
    """Response from embedding generation."""

    texts: list[str]
    """Original input texts."""

    embeddings: list[list[float]]
    """List of embeddings (one per input text)."""

    model: str
    """Model name used for generation."""


class EmbeddingClient:
    """Client for generating text embeddings."""

    def __init__(self, model_name: str, device: str | None = None):
        """Initialize the embedding client.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ("cuda", "mps", "cpu"). Auto-detects if None.
        """
        self.model_name = model_name
        self.device = device or self._select_device()
        logger.info(f"Loading embedding model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def _select_device(self) -> str:
        """Select best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResponse with texts, embeddings, and model info
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.model.encode, texts, False, True
        )
        return EmbeddingResponse(
            texts=texts,
            embeddings=[emb.tolist() for emb in embeddings],
            model=self.model_name,
        )
