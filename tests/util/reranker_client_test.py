"""Tests for the reranker client module.

These tests verify the RerankerClient class for reranking query-document pairs
using cross-encoder models.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lean_explore.util.reranker_client import (
    DEFAULT_INSTRUCTION,
    RerankerClient,
    RerankerResponse,
)


class TestRerankerResponse:
    """Tests for RerankerResponse model."""

    def test_reranker_response_fields(self):
        """Test RerankerResponse contains expected fields."""
        response = RerankerResponse(
            query="test query",
            scores=[0.9, 0.7, 0.3],
            model="test-model",
        )

        assert response.query == "test query"
        assert response.scores == [0.9, 0.7, 0.3]
        assert response.model == "test-model"


class TestRerankerClientInit:
    """Tests for RerankerClient initialization."""

    @pytest.fixture
    def mock_tokenizer_and_model(self):
        """Mock the tokenizer and model loading."""
        with patch(
            "lean_explore.util.reranker_client.AutoTokenizer"
        ) as mock_tokenizer_cls:
            mock_tokenizer = MagicMock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: (
                1 if x == "true" else 0
            )
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            with patch(
                "lean_explore.util.reranker_client.AutoModelForCausalLM"
            ) as mock_model_cls:
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model_cls.from_pretrained.return_value = mock_model

                with patch("lean_explore.util.reranker_client.torch") as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    mock_torch.float32 = torch.float32
                    mock_torch.float16 = torch.float16

                    yield mock_tokenizer, mock_model

    def test_select_device_cpu(self, mock_tokenizer_and_model):
        """Test device selection falls back to CPU."""
        client = RerankerClient(model_name="test-model")

        assert client.device == "cpu"

    def test_default_instruction(self, mock_tokenizer_and_model):
        """Test default instruction is set."""
        client = RerankerClient(model_name="test-model")

        assert client.instruction == DEFAULT_INSTRUCTION

    def test_custom_instruction(self, mock_tokenizer_and_model):
        """Test custom instruction can be provided."""
        client = RerankerClient(
            model_name="test-model",
            instruction="Custom instruction",
        )

        assert client.instruction == "Custom instruction"


class TestRerankerClientFormatPair:
    """Tests for query-document pair formatting."""

    @pytest.fixture
    def client(self):
        """Create a mock reranker client."""
        with patch(
            "lean_explore.util.reranker_client.AutoTokenizer"
        ) as mock_tokenizer_cls:
            mock_tokenizer = MagicMock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: (
                1 if x == "true" else 0
            )
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            with patch(
                "lean_explore.util.reranker_client.AutoModelForCausalLM"
            ) as mock_model_cls:
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                mock_model_cls.from_pretrained.return_value = mock_model

                with patch("lean_explore.util.reranker_client.torch") as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    mock_torch.float32 = torch.float32

                    yield RerankerClient(model_name="test-model")

    def test_format_pair(self, client):
        """Test query-document pair formatting."""
        result = client._format_pair("search query", "document text")

        assert "<Instruct>:" in result
        assert "<Query>: search query" in result
        assert "<Document>: document text" in result


class TestRerankerClientRerank:
    """Tests for RerankerClient.rerank method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock reranker client with mocked inference."""
        with patch(
            "lean_explore.util.reranker_client.AutoTokenizer"
        ) as mock_tokenizer_cls:
            mock_tokenizer = MagicMock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: (
                1 if x == "true" else 0
            )
            # Return mock tensor-like object
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            with patch(
                "lean_explore.util.reranker_client.AutoModelForCausalLM"
            ) as mock_model_cls:
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model

                # Mock model output
                mock_outputs = MagicMock()
                # Create fake logits tensor
                mock_logits = torch.tensor([[[0.0, 1.0]]])  # Shape: [1, 1, 2]
                mock_outputs.logits = mock_logits
                mock_model.return_value = mock_outputs

                mock_model_cls.from_pretrained.return_value = mock_model

                with patch("lean_explore.util.reranker_client.torch") as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    mock_torch.float32 = torch.float32
                    mock_torch.no_grad = MagicMock(
                        return_value=MagicMock(
                            __enter__=MagicMock(), __exit__=MagicMock()
                        )
                    )
                    mock_torch.stack = torch.stack
                    mock_torch.nn.functional.log_softmax = (
                        torch.nn.functional.log_softmax
                    )

                    yield RerankerClient(model_name="test-model")

    async def test_rerank_empty_documents(self, mock_client):
        """Test reranking with empty document list."""
        response = await mock_client.rerank("query", [])

        assert isinstance(response, RerankerResponse)
        assert response.scores == []
        assert response.query == "query"

    def test_rerank_sync_empty_documents(self, mock_client):
        """Test synchronous reranking with empty documents."""
        response = mock_client.rerank_sync("query", [])

        assert response.scores == []

    async def test_rerank_returns_response(self, mock_client):
        """Test that rerank returns RerankerResponse."""
        # This test is limited due to mocking complexity
        # The actual scoring logic is tested in integration tests
        response = await mock_client.rerank("query", [])

        assert isinstance(response, RerankerResponse)
        assert response.model == "test-model"


class TestRerankerClientIntegration:
    """Integration tests that load actual models."""

    @pytest.mark.external
    @pytest.mark.slow
    async def test_rerank_real_model(self):
        """Test reranking with a real model."""
        # Skip if no GPU and running in CI
        client = RerankerClient(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            max_length=256,
        )

        response = await client.rerank(
            query="natural number addition",
            documents=[
                "Nat.add: Adds two natural numbers",
                "String.length: Returns length of string",
                "List.map: Maps a function over a list",
            ],
        )

        assert len(response.scores) == 3
        # First document should score highest (most relevant)
        assert response.scores[0] > response.scores[1]
        assert response.scores[0] > response.scores[2]
