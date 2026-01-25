"""Tests for the OpenRouter client module.

These tests verify the OpenRouterClient class for interacting with the
OpenRouter API using OpenAI SDK types.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_explore.util.openrouter_client import OpenRouterClient


class TestOpenRouterClientInit:
    """Tests for OpenRouterClient initialization."""

    def test_init_requires_api_key(self):
        """Test that initialization fails without API key."""
        # Ensure no API key is set
        with patch.dict(os.environ, {}, clear=True):
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                OpenRouterClient()

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "lean_explore.util.openrouter_client.AsyncOpenAI"
            ) as mock_openai:
                client = OpenRouterClient()

                mock_openai.assert_called_once_with(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="test-key",
                )
                assert client.client is not None


class TestOpenRouterClientGenerate:
    """Tests for OpenRouterClient.generate method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenRouter client."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "lean_explore.util.openrouter_client.AsyncOpenAI"
            ) as mock_openai:
                mock_async_client = MagicMock()
                mock_openai.return_value = mock_async_client

                # Mock the chat completions
                mock_completion = MagicMock()
                mock_completion.choices = [
                    MagicMock(message=MagicMock(content="Generated response"))
                ]
                mock_async_client.chat.completions.create = AsyncMock(
                    return_value=mock_completion
                )

                yield OpenRouterClient()

    async def test_generate_basic(self, mock_client):
        """Test basic generation call."""
        messages = [{"role": "user", "content": "Hello"}]

        response = await mock_client.generate(
            model="test-model",
            messages=messages,
        )

        assert response is not None
        mock_client.client.chat.completions.create.assert_called_once()

    async def test_generate_with_parameters(self, mock_client):
        """Test generation with custom parameters."""
        messages = [{"role": "user", "content": "Hello"}]

        await mock_client.generate(
            model="anthropic/claude-3.5-sonnet",
            messages=messages,
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock_client.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    async def test_generate_default_temperature(self, mock_client):
        """Test that default temperature is 0.7."""
        messages = [{"role": "user", "content": "Hello"}]

        await mock_client.generate(
            model="test-model",
            messages=messages,
        )

        call_kwargs = mock_client.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7


class TestOpenRouterClientIntegration:
    """Integration tests that call actual API."""

    @pytest.mark.external
    async def test_generate_real_api(self):
        """Test generation with real OpenRouter API."""
        # Skip if no API key
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient()

        response = await client.generate(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )

        assert response is not None
        assert len(response.choices) > 0
        assert "hello" in response.choices[0].message.content.lower()
