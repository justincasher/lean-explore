# src/lean_explore/models/db.py

"""SQLAlchemy ORM models for the lean_explore database.

Defines 'declarations', 'dependencies', 'statement_groups', and
'statement_group_dependencies' tables representing Lean entities,
their dependency graphs at different granularities, and source code groupings.
Uses SQLAlchemy 2.0 syntax.
"""

import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Naming conventions for constraints and indexes for database consistency.
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""

    metadata = metadata_obj


class StatementGroup(Base):
    """Represents a unique block of source code text.

    Groups multiple `Declaration` entries from the same source code text and location.
    """

    __tablename__ = "statement_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """Primary key identifier for the statement group."""

    text_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True, unique=True
    )
    """SHA-256 hash of statement_text for unique identification."""

    statement_text: Mapped[str] = mapped_column(Text, nullable=False)
    """Canonical source code text for this group (full block)."""

    display_statement_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Optional, potentially truncated version optimized for display."""

    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Docstring associated with this code block."""

    informal_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Optional informal English description, potentially LLM-generated."""

    informal_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Optional informal English summary, potentially LLM-generated."""

    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    """Relative path to the .lean file containing this block."""

    range_start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    """Starting line number of the block in the source file."""

    range_start_col: Mapped[int] = mapped_column(Integer, nullable=False)
    """Starting column number of the block."""

    range_end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    """Ending line number of the block."""

    range_end_col: Mapped[int] = mapped_column(Integer, nullable=False)
    """Ending column number of the block."""

    pagerank_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, index=True
    )
    """PageRank score calculated for this statement group."""

    scaled_pagerank_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, index=True
    )
    """Log-transformed, min-max scaled PageRank score."""

    primary_decl_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("declarations.id"), nullable=False, index=True
    )
    """Foreign key to the primary or most representative declaration."""

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    """Timestamp of when the record was created."""

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )
    """Timestamp of the last update to the record."""

    # Relationships
    primary_declaration: Mapped["Declaration"] = relationship(
        "Declaration", foreign_keys=[primary_decl_id]
    )
    """SQLAlchemy relationship to the primary Declaration."""

    declarations: Mapped[List["Declaration"]] = relationship(
        "Declaration",
        foreign_keys="[Declaration.statement_group_id]",
        back_populates="statement_group",
    )
    """SQLAlchemy relationship to all Declarations in this group."""

    dependencies_as_source: Mapped[List["StatementGroupDependency"]] = relationship(
        foreign_keys="StatementGroupDependency.source_statement_group_id",
        back_populates="source_group",
        cascade="all, delete-orphan",
        lazy="select",
    )
    """Links to StatementGroupDependency where this group is the source."""

    dependencies_as_target: Mapped[List["StatementGroupDependency"]] = relationship(
        foreign_keys="StatementGroupDependency.target_statement_group_id",
        back_populates="target_group",
        cascade="all, delete-orphan",
        lazy="select",
    )
    """Links to StatementGroupDependency where this group is the target."""

    __table_args__ = (
        Index(
            "ix_statement_groups_location",
            "source_file",
            "range_start_line",
            "range_start_col",
        ),
    )

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation."""
        has_desc = "+" if self.informal_description else "-"
        return (
            f"<StatementGroup(id={self.id}, hash='{self.text_hash[:8]}...', "
            f"primary_decl_id='{self.primary_decl_id}', informal_desc='{has_desc}', "
            f"loc='{self.source_file}:{self.range_start_line}:{self.range_start_col}')>"
        )


class Declaration(Base):
    """Represents a Lean declaration, a node in the dependency graph."""

    __tablename__ = "declarations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """Primary key identifier."""

    lean_name: Mapped[str] = mapped_column(
        Text, unique=True, index=True, nullable=False
    )
    """Fully qualified Lean name (e.g., 'Nat.add'), unique and indexed."""

    decl_type: Mapped[str] = mapped_column(String(30), nullable=False)
    """Type of declaration (e.g., 'theorem', 'definition')."""

    source_file: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Relative path to the .lean source file."""

    module_name: Mapped[Optional[str]] = mapped_column(Text, index=True, nullable=True)
    """Lean module name (e.g., 'Mathlib.Data.Nat.Basic'), indexed."""

    is_internal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True if considered compiler-internal or auxiliary."""

    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Documentation string, if available."""

    is_protected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True if marked 'protected' in Lean."""

    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True if marked 'deprecated'."""

    is_projection: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True if it's a projection (e.g., from a class/structure)."""

    range_start_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    """Starting line number of the source block."""

    range_start_col: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    """Starting column number of the source block."""

    range_end_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    """Ending line number of the source block."""

    range_end_col: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    """Ending column number of the source block."""

    statement_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Full Lean code text of the originating source block."""

    declaration_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Extracted Lean signature text of the declaration."""

    statement_group_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("statement_groups.id"), nullable=True, index=True
    )
    """Optional foreign key to statement_groups.id."""

    pagerank_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, index=True
    )
    """PageRank score within the dependency graph, indexed."""

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    """Timestamp of record creation."""

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )
    """Timestamp of last record update."""

    statement_group: Mapped[Optional["StatementGroup"]] = relationship(
        "StatementGroup",
        foreign_keys=[statement_group_id],
        back_populates="declarations",
    )
    """SQLAlchemy relationship to the StatementGroup."""

    __table_args__ = (
        Index("ix_declarations_source_file", "source_file"),
        Index("ix_declarations_is_protected", "is_protected"),
        Index("ix_declarations_is_deprecated", "is_deprecated"),
        Index("ix_declarations_is_projection", "is_projection"),
        Index("ix_declarations_is_internal", "is_internal"),
    )

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation."""
        group_id_str = (
            f", group_id={self.statement_group_id}" if self.statement_group_id else ""
        )
        return (
            f"<Declaration(id={self.id}, lean_name='{self.lean_name}', "
            f"type='{self.decl_type}'{group_id_str})>"
        )


class Dependency(Base):
    """Represents a dependency link between two Lean declarations."""

    __tablename__ = "dependencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """Primary key identifier for the dependency link."""

    source_decl_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("declarations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    """Foreign key to the Declaration that depends on another."""

    target_decl_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("declarations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    """Foreign key to the Declaration that is depended upon."""

    dependency_type: Mapped[str] = mapped_column(String(30), nullable=False)
    """String describing the type of dependency (e.g., 'Direct')."""

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    """Timestamp of record creation."""

    __table_args__ = (
        UniqueConstraint(
            "source_decl_id",
            "target_decl_id",
            "dependency_type",
            name="uq_dependency_link",
        ),
        Index("ix_dependencies_source_target", "source_decl_id", "target_decl_id"),
    )

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation."""
        return (
            f"<Dependency(id={self.id}, source={self.source_decl_id}, "
            f"target={self.target_decl_id}, type='{self.dependency_type}')>"
        )


class StatementGroupDependency(Base):
    """Represents a dependency link between two StatementGroups."""

    __tablename__ = "statement_group_dependencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """Primary key identifier for the group dependency link."""

    source_statement_group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("statement_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    """Foreign key to the StatementGroup that depends on another."""

    target_statement_group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("statement_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    """Foreign key to the StatementGroup that is depended upon."""

    dependency_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="DerivedFromDecl"
    )
    """String describing the type of group dependency."""

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    """Timestamp of record creation."""

    source_group: Mapped["StatementGroup"] = relationship(
        foreign_keys=[source_statement_group_id],
        back_populates="dependencies_as_source",
    )
    """SQLAlchemy relationship to the source StatementGroup."""

    target_group: Mapped["StatementGroup"] = relationship(
        foreign_keys=[target_statement_group_id],
        back_populates="dependencies_as_target",
    )
    """SQLAlchemy relationship to the target StatementGroup."""

    __table_args__ = (
        UniqueConstraint(
            "source_statement_group_id",
            "target_statement_group_id",
            "dependency_type",
            name="uq_stmt_group_dependency_link",
        ),
        Index(
            "ix_stmt_group_deps_source_target",
            "source_statement_group_id",
            "target_statement_group_id",
        ),
    )

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation."""
        return (
            f"<StatementGroupDependency(id={self.id}, "
            f"source_sg_id={self.source_statement_group_id}, "
            f"target_sg_id={self.target_statement_group_id}, "
            f"type='{self.dependency_type}')>"
        )
