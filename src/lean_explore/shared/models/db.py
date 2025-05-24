# src/lean_explore/shared/models/db.py

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
    UniqueConstraint,  # Kept for potential standalone model testing
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
    """Base class for SQLAlchemy declarative models.

    Includes metadata with naming conventions for database constraints and indexes,
    ensuring consistency across the database schema.
    """

    metadata = metadata_obj


class StatementGroup(Base):
    """Represents a unique block of source code text.

    This table groups multiple `Declaration` entries that originate from the
    exact same source code text and location. This allows search results to
    show a single entry for a code block, while retaining all individual
    declarations for graph analysis and detailed views. It also tracks
    dependencies to and from other statement groups.

    Attributes:
        id: Primary key identifier for the statement group.
        text_hash: SHA-256 hash of `statement_text` for unique identification.
        statement_text: Canonical source code text for this group (full block).
        display_statement_text: Optional, potentially truncated version of the
            source code, optimized for display (e.g., omitting proofs).
        docstring: Docstring associated with this code block, typically from the
            primary declaration.
        informal_description: Optional informal English description, potentially
            LLM-generated.
        source_file: Relative path to the .lean file containing this block.
        range_start_line: Starting line number of the block in the source file.
        range_start_col: Starting column number of the block.
        range_end_line: Ending line number of the block.
        range_end_col: Ending column number of the block.
        pagerank_score: PageRank score calculated for this statement group.
        scaled_pagerank_score: Log-transformed, min-max scaled PageRank score.
        primary_decl_id: Foreign key to the 'declarations' table, identifying
            the primary or most representative declaration of this group.
        created_at: Timestamp of when the record was created.
        updated_at: Timestamp of the last update to the record.
        primary_declaration: SQLAlchemy relationship to the primary Declaration.
        declarations: SQLAlchemy relationship to all Declarations in this group.
        dependencies_as_source: Links to `StatementGroupDependency` where this
            group is the source (i.e., this group depends on others).
        dependencies_as_target: Links to `StatementGroupDependency` where this
            group is the target (i.e., other groups depend on this one).
    """

    __tablename__ = "statement_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True, unique=True
    )
    statement_text: Mapped[str] = mapped_column(Text, nullable=False)
    display_statement_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    informal_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    range_start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    range_start_col: Mapped[int] = mapped_column(Integer, nullable=False)
    range_end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    range_end_col: Mapped[int] = mapped_column(Integer, nullable=False)

    pagerank_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, index=True
    )
    scaled_pagerank_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, index=True
    )

    primary_decl_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("declarations.id"), nullable=False, index=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )

    # Relationships
    primary_declaration: Mapped["Declaration"] = relationship(
        "Declaration", foreign_keys=[primary_decl_id]
    )
    declarations: Mapped[List["Declaration"]] = relationship(
        "Declaration",
        foreign_keys="[Declaration.statement_group_id]",
        back_populates="statement_group",
    )

    dependencies_as_source: Mapped[List["StatementGroupDependency"]] = relationship(
        foreign_keys="StatementGroupDependency.source_statement_group_id",
        back_populates="source_group",
        cascade="all, delete-orphan",
        lazy="select",
    )
    dependencies_as_target: Mapped[List["StatementGroupDependency"]] = relationship(
        foreign_keys="StatementGroupDependency.target_statement_group_id",
        back_populates="target_group",
        cascade="all, delete-orphan",
        lazy="select",
    )

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
    """Represents a Lean declaration, a node in the dependency graph.

    Stores information about Lean declarations (definitions, theorems, axioms, etc.),
    including source location, Lean code, descriptions, and (potentially)
    embeddings. Declarations from the same source block can be grouped via
    `statement_group_id`.

    Attributes:
        id: Primary key identifier.
        lean_name: Fully qualified Lean name (e.g., 'Nat.add'), unique and indexed.
        decl_type: Type of declaration (e.g., 'theorem', 'definition').
        source_file: Relative path to the .lean source file.
        module_name: Lean module name (e.g., 'Mathlib.Data.Nat.Basic'), indexed.
        is_internal: True if considered compiler-internal or auxiliary.
        docstring: Documentation string, if available.
        is_protected: True if marked 'protected' in Lean.
        is_deprecated: True if marked 'deprecated'.
        is_projection: True if it's a projection (e.g., from a class/structure).
        range_start_line: Starting line number of the source block.
        range_start_col: Starting column number of the source block.
        range_end_line: Ending line number of the source block.
        range_end_col: Ending column number of the source block.
        statement_text: Full Lean code text of the originating source block.
        declaration_signature: Extracted Lean signature text of the declaration.
        statement_group_id: Optional foreign key to `statement_groups.id`.
        type_signature_text: Lean type signature (may be redundant with
            `declaration_signature`).
        informal_description: Informal English explanation.
        lean_embedding: Text representation (e.g., JSON) of the embedding for
            Lean code.
        informal_description_embedding: Text representation (e.g., JSON) of the
            embedding for the informal description.
        pagerank_score: PageRank score within the dependency graph, indexed.
        created_at: Timestamp of record creation.
        updated_at: Timestamp of last record update.
        statement_group: SQLAlchemy relationship to the StatementGroup.
    """

    __tablename__ = "declarations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lean_name: Mapped[str] = mapped_column(
        Text, unique=True, index=True, nullable=False
    )
    decl_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_file: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    module_name: Mapped[Optional[str]] = mapped_column(Text, index=True, nullable=True)
    is_internal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_protected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_projection: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    range_start_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    range_start_col: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    range_end_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    range_end_col: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    statement_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    declaration_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    statement_group_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("statement_groups.id"), nullable=True, index=True
    )

    type_signature_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    informal_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    lean_embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    informal_description_embedding: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    pagerank_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, index=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )

    statement_group: Mapped[Optional["StatementGroup"]] = relationship(
        "StatementGroup",
        foreign_keys=[statement_group_id],
        back_populates="declarations",
    )

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
    """Represents a dependency link between two Lean declarations.

    Each row signifies that a 'source' declaration depends on a 'target'
    declaration, forming an edge in the dependency graph. The nature of this
    dependency is described by `dependency_type`.

    Attributes:
        id: Primary key identifier for the dependency link.
        source_decl_id: Foreign key to the `Declaration` that depends on another.
        target_decl_id: Foreign key to the `Declaration` that is depended upon.
        dependency_type: String describing the type of dependency (e.g., 'Direct').
        context: Optional string providing context for the dependency.
        created_at: Timestamp of record creation.
    """

    __tablename__ = "dependencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_decl_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("declarations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_decl_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("declarations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dependency_type: Mapped[str] = mapped_column(String(30), nullable=False)
    context: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

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
    """Represents a dependency link between two StatementGroups.

    Each row signifies that a 'source' statement group depends on a 'target'
    statement group. This allows for a higher-level dependency graph.

    Attributes:
        id: Primary key identifier for the group dependency link.
        source_statement_group_id: Foreign key to the `StatementGroup` that
                                   depends on another.
        target_statement_group_id: Foreign key to the `StatementGroup` that
                                   is depended upon.
        dependency_type: String describing the type of group dependency
                         (e.g., 'DerivedFromDecl').
        created_at: Timestamp of record creation.
        source_group: SQLAlchemy relationship to the source StatementGroup.
        target_group: SQLAlchemy relationship to the target StatementGroup.
    """

    __tablename__ = "statement_group_dependencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_statement_group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("statement_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_statement_group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("statement_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dependency_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="DerivedFromDecl"
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationships back to StatementGroup
    source_group: Mapped["StatementGroup"] = relationship(
        foreign_keys=[source_statement_group_id],
        back_populates="dependencies_as_source",
    )
    target_group: Mapped["StatementGroup"] = relationship(
        foreign_keys=[target_statement_group_id],
        back_populates="dependencies_as_target",
    )

    __table_args__ = (
        UniqueConstraint(
            "source_statement_group_id",
            "target_statement_group_id",
            "dependency_type",  # Consider if type is part of uniqueness
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
