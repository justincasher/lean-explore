"""Shared utilities for lean_explore.

Imports are lazy to avoid loading torch when not needed.
"""


def __getattr__(name: str):
    """Lazy import attributes to avoid loading torch unnecessarily."""
    if name == "EmbeddingClient":
        from lean_explore.util.embedding_client import EmbeddingClient

        return EmbeddingClient
    if name == "OpenRouterClient":
        from lean_explore.util.openrouter_client import OpenRouterClient

        return OpenRouterClient
    if name == "setup_logging":
        from lean_explore.util.logging import setup_logging

        return setup_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EmbeddingClient", "OpenRouterClient", "setup_logging"]
