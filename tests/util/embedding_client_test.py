"""Tests for the embedding client module.

These tests verify the EmbeddingClient class for generating text embeddings
using sentence transformers.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lean_explore.util.embedding_client import EmbeddingClient, EmbeddingResponse


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse model."""

    def test_embedding_response_fields(self):
        """Test EmbeddingResponse contains expected fields."""
        response = EmbeddingResponse(
            texts=["hello", "world"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="test-model",
        )

        assert response.texts == ["hello", "world"]
        assert len(response.embeddings) == 2
        assert response.model == "test-model"


class TestEmbeddingClientInit:
    """Tests for EmbeddingClient initialization."""

    def test_select_device_cpu(self):
        """Test device selection falls back to CPU."""
        with patch("lean_explore.util.embedding_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            with patch(
                "lean_explore.util.embedding_client.SentenceTransformer"
            ) as mock_st:
                mock_st.return_value = MagicMock()
                client = EmbeddingClient(model_name="test-model")

                assert client.device == "cpu"

    def test_select_device_cuda(self):
        """Test device selection prefers CUDA when available."""
        with patch("lean_explore.util.embedding_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            with patch(
                "lean_explore.util.embedding_client.SentenceTransformer"
            ) as mock_st:
                mock_st.return_value = MagicMock()
                client = EmbeddingClient(model_name="test-model")

                assert client.device == "cuda"

    def test_max_length_setting(self):
        """Test that max_length is set on model."""
        with patch("lean_explore.util.embedding_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with patch(
                "lean_explore.util.embedding_client.SentenceTransformer"
            ) as mock_st:
                mock_model = MagicMock()
                mock_st.return_value = mock_model

                _client = EmbeddingClient(model_name="test-model", max_length=256)

                assert mock_model.max_seq_length == 256


class TestEmbeddingClientEmbed:
    """Tests for EmbeddingClient.embed method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock embedding client."""
        with patch("lean_explore.util.embedding_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with patch(
                "lean_explore.util.embedding_client.SentenceTransformer"
            ) as mock_st:
                mock_model = MagicMock()
                # Return numpy arrays like the real model
                mock_model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])
                mock_st.return_value = mock_model

                yield EmbeddingClient(model_name="test-model")

    async def test_embed_returns_response(self, mock_client):
        """Test that embed returns EmbeddingResponse."""
        response = await mock_client.embed(["hello", "world"])

        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 2
        assert response.model == "test-model"

    async def test_embed_with_query_flag(self, mock_client):
        """Test that is_query flag passes prompt_name."""
        await mock_client.embed(["query"], is_query=True)

        # Verify encode was called with prompt_name
        call_kwargs = mock_client.model.encode.call_args[1]
        assert call_kwargs.get("prompt_name") == "query"

    async def test_embed_without_query_flag(self, mock_client):
        """Test that documents don't use prompt_name."""
        await mock_client.embed(["document"], is_query=False)

        # Verify encode was called without prompt_name
        call_kwargs = mock_client.model.encode.call_args[1]
        assert "prompt_name" not in call_kwargs


class TestEmbeddingClientIntegration:
    """Integration tests that load actual models."""

    @pytest.mark.external
    @pytest.mark.slow
    async def test_embed_real_model(self):
        """Test embedding generation with a small real model."""
        # Use a small model for testing
        client = EmbeddingClient(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=128,
        )

        response = await client.embed(["Hello, world!", "Test sentence"])

        assert len(response.embeddings) == 2
        # MiniLM produces 384-dimensional embeddings
        assert len(response.embeddings[0]) == 384
        # Embeddings should be different
        assert response.embeddings[0] != response.embeddings[1]
