# src/lean_explore/extract/models.py

"""SQLAlchemy ORM models for doc-gen4 data extraction.

Simple schema for a Lean declaration search engine.
Uses SQLAlchemy 2.0 syntax.
"""

from sqlalchemy import Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""

    pass


class Declaration(Base):
    """Represents a Lean declaration for search."""

    __tablename__ = "declarations"

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
