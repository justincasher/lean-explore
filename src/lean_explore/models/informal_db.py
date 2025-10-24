"""SQLAlchemy ORM models for informalization cache database.

Provides a cache layer for source text to informalization mappings, allowing
reuse of informalizations across declarations with identical source text.
"""

import hashlib
from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from lean_explore.models.search_db import Base


class InformalizationCache(Base):
    """Cache entry mapping source text to its informalization.

    Stores a hash-indexed cache of source text to natural language descriptions,
    enabling reuse of informalizations across declarations with identical code.
    """

    __tablename__ = "informalization_cache"
    __table_args__ = (
        Index(
            "ix_informalization_cache_source_hash",
            "source_text_hash",
            unique=True,
        ),
        Index(
            "ix_informalization_cache_model",
            "model",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """Primary key identifier."""

    source_text_hash: Mapped[str] = mapped_column(
        Text, unique=True, index=True, nullable=False
    )
    """SHA-256 hash of the source text for fast lookup."""

    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    """The actual Lean source code that was informalized."""

    informalization: Mapped[str] = mapped_column(Text, nullable=False)
    """Natural language description generated for this source text."""

    model: Mapped[str] = mapped_column(Text, nullable=False)
    """LLM model name used to generate this informalization."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Timestamp when this cache entry was created."""

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    """Timestamp when this cache entry was last updated."""

    @staticmethod
    def compute_hash(source_text: str) -> str:
        """Compute SHA-256 hash of source text for cache lookup."""
        return hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls, source_text: str, informalization: str, model: str
    ) -> "InformalizationCache":
        """Create a new cache entry.

        Args:
            source_text: The Lean source code
            informalization: Natural language description
            model: LLM model name used for generation

        Returns:
            New InformalizationCache instance
        """
        return cls(
            source_text_hash=cls.compute_hash(source_text),
            source_text=source_text,
            informalization=informalization,
            model=model,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
