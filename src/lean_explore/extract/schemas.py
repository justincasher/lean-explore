# src/lean_explore/extract/models.py

"""SQLAlchemy ORM models for doc-gen4 data extraction.

Simple schema for a Lean declaration search engine.
Uses SQLAlchemy 2.0 syntax.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""

    pass


class Declaration(Base):
    """Represents a Lean declaration for search."""

    __tablename__ = "declarations"
    __table_args__ = (
        Index(
            "ix_declarations_name_embedding_hnsw",
            "name_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"name_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_declarations_informalization_embedding_hnsw",
            "informalization_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"informalization_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_declarations_source_text_embedding_hnsw",
            "source_text_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"source_text_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_declarations_docstring_embedding_hnsw",
            "docstring_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"docstring_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """Primary key identifier."""

    name: Mapped[str] = mapped_column(Text, unique=True, index=True, nullable=False)
    """Fully qualified Lean name (e.g., 'Nat.add')."""

    module: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    """Module name (e.g., 'Mathlib.Data.List.Basic')."""

    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Documentation string from the source code, if available."""

    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    """The actual Lean source code for this declaration."""

    source_link: Mapped[str] = mapped_column(Text, nullable=False)
    """GitHub URL to the declaration source code."""

    dependencies: Mapped[str | None] = mapped_column(Text, nullable=True)
    """JSON array of declaration names this declaration depends on."""

    informalization: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Natural language description of the declaration."""

    name_embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    """768-dimensional embedding of the declaration name."""

    informalization_embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    """768-dimensional embedding of the informalization text."""

    source_text_embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    """768-dimensional embedding of the source text."""

    docstring_embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    """768-dimensional embedding of the docstring."""

    pagerank: Mapped[float | None] = mapped_column(Float, nullable=True)
    """PageRank score based on dependency graph."""
