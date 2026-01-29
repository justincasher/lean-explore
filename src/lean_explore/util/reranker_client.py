"""Reranker client using Qwen3-Reranker for query-document scoring."""

import asyncio
import logging

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Find relevant Lean 4 math declarations"


class RerankerResponse(BaseModel):
    """Response from reranking operation."""

    query: str
    """The original query."""

    scores: list[float]
    """Relevance scores for each document (same order as input)."""

    model: str
    """Model name used for reranking."""


class RerankerClient:
    """Client for reranking query-document pairs using Qwen3-Reranker."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str | None = None,
        max_length: int = 512,
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        """Initialize the reranker client.

        Args:
            model_name: Name of the reranker model from HuggingFace.
            device: Device to use ("cuda", "mps", "cpu"). Auto-detects if None.
            max_length: Maximum sequence length for tokenization.
            instruction: Task instruction prepended to each query-document pair.
        """
        self.model_name = model_name
        self.device = device or self._select_device()
        self.max_length = max_length
        self.instruction = instruction

        logger.info(f"Loading reranker model {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )

        # Use float32 on CPU (faster than float16 which gets emulated)
        # Use float16 on GPU for memory efficiency
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Get token IDs for true/false classification
        self._token_true_id = self.tokenizer.convert_tokens_to_ids("true")
        self._token_false_id = self.tokenizer.convert_tokens_to_ids("false")

        logger.info("Reranker model loaded successfully")

    def _select_device(self) -> str:
        """Select best available device."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _format_pair(self, query: str, document: str) -> str:
        """Format a query-document pair with instruction.

        Args:
            query: The search query.
            document: The document text to score.

        Returns:
            Formatted string for the reranker model.
        """
        return (
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
        )

    @torch.no_grad()
    def _compute_scores_sync(self, pairs: list[str]) -> list[float]:
        """Compute relevance scores for formatted pairs synchronously.

        Args:
            pairs: List of formatted query-document strings.

        Returns:
            List of relevance scores in [0, 1].
        """
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Get logits for last token
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Extract true/false logits
        true_logits = logits[:, self._token_true_id]
        false_logits = logits[:, self._token_false_id]

        # Compute probability of "true" using softmax
        stacked = torch.stack([false_logits, true_logits], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp()

        return scores.cpu().tolist()

    def rerank_sync(
        self,
        query: str,
        documents: list[str],
    ) -> RerankerResponse:
        """Rerank documents synchronously (faster for small batches).

        Args:
            query: The search query.
            documents: List of document texts to rerank.

        Returns:
            RerankerResponse with scores for each document.
        """
        if not documents:
            return RerankerResponse(query=query, scores=[], model=self.model_name)

        pairs = [self._format_pair(query, doc) for doc in documents]
        scores = self._compute_scores_sync(pairs)
        return RerankerResponse(query=query, scores=scores, model=self.model_name)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        batch_size: int | None = None,
    ) -> RerankerResponse:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            batch_size: Number of pairs to process at once.

        Returns:
            RerankerResponse with scores for each document.
        """
        if not documents:
            return RerankerResponse(query=query, scores=[], model=self.model_name)

        # Default batch size: 16 on GPU (fits 8GB VRAM), 32 on CPU
        if batch_size is None:
            batch_size = 16 if self.device == "cuda" else 32

        # For small batches, run synchronously to avoid executor overhead
        if len(documents) <= batch_size:
            return self.rerank_sync(query, documents)

        # Format all pairs
        pairs = [self._format_pair(query, doc) for doc in documents]

        # Process in batches
        loop = asyncio.get_event_loop()
        all_scores: list[float] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_scores = await loop.run_in_executor(
                None, self._compute_scores_sync, batch
            )
            all_scores.extend(batch_scores)

        return RerankerResponse(query=query, scores=all_scores, model=self.model_name)
