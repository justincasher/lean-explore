"""Tests for the search database models.

These tests verify the SQLAlchemy ORM models and custom types,
particularly the BinaryEmbedding type for storing embeddings.
"""

import struct

import pytest
from sqlalchemy import select

from lean_explore.models import Declaration
from lean_explore.models.search_db import BinaryEmbedding


class TestBinaryEmbedding:
    """Tests for BinaryEmbedding custom type."""

    def test_process_bind_param_converts_to_binary(self):
        """Test that list[float] is converted to binary."""
        embedding_type = BinaryEmbedding()
        values = [0.1, 0.2, 0.3, 0.4]

        result = embedding_type.process_bind_param(values, None)

        assert isinstance(result, bytes)
        # Should be 4 bytes per float
        assert len(result) == len(values) * 4

    def test_process_bind_param_none(self):
        """Test that None is preserved."""
        embedding_type = BinaryEmbedding()

        result = embedding_type.process_bind_param(None, None)

        assert result is None

    def test_process_result_value_converts_to_list(self):
        """Test that binary is converted back to list[float]."""
        embedding_type = BinaryEmbedding()
        original = [0.1, 0.2, 0.3, 0.4]
        binary = struct.pack(f"{len(original)}f", *original)

        result = embedding_type.process_result_value(binary, None)

        assert isinstance(result, list)
        assert len(result) == len(original)
        for i, val in enumerate(original):
            assert abs(result[i] - val) < 1e-6

    def test_process_result_value_none(self):
        """Test that None is preserved."""
        embedding_type = BinaryEmbedding()

        result = embedding_type.process_result_value(None, None)

        assert result is None

    def test_roundtrip(self):
        """Test that data survives a round-trip conversion."""
        embedding_type = BinaryEmbedding()
        original = [0.1, 0.2, 0.3, 0.4, 0.5]

        binary = embedding_type.process_bind_param(original, None)
        recovered = embedding_type.process_result_value(binary, None)

        assert len(recovered) == len(original)
        for i, val in enumerate(original):
            assert abs(recovered[i] - val) < 1e-6


class TestDeclarationModel:
    """Tests for Declaration ORM model."""

    async def test_create_declaration(self, async_db_session):
        """Test creating a basic declaration."""
        declaration = Declaration(
            name="Test.Declaration",
            module="Test.Module",
            source_text="def test := 1",
            source_link="https://example.com/test",
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Declaration).where(Declaration.name == "Test.Declaration")
        )
        loaded = result.scalar_one()

        assert loaded.name == "Test.Declaration"
        assert loaded.module == "Test.Module"
        assert loaded.id is not None

    async def test_declaration_with_embedding(self, async_db_session):
        """Test declaration with embedding storage."""
        embedding = [0.1] * 1024  # 1024-dimensional embedding
        declaration = Declaration(
            name="Test.WithEmbedding",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            informalization="A test declaration",
            informalization_embedding=embedding,
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Declaration).where(Declaration.name == "Test.WithEmbedding")
        )
        loaded = result.scalar_one()

        assert loaded.informalization_embedding is not None
        assert len(loaded.informalization_embedding) == 1024
        assert abs(loaded.informalization_embedding[0] - 0.1) < 1e-6

    async def test_declaration_unique_name(self, async_db_engine):
        """Test that declaration names must be unique."""
        from sqlalchemy.exc import IntegrityError
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(async_db_engine) as session:
            decl1 = Declaration(
                name="Unique.Name",
                module="Test",
                source_text="def test := 1",
                source_link="https://example.com",
            )
            session.add(decl1)
            await session.commit()

            decl2 = Declaration(
                name="Unique.Name",  # Same name
                module="Test2",
                source_text="def test := 2",
                source_link="https://example.com/2",
            )
            session.add(decl2)

            with pytest.raises(IntegrityError):
                await session.commit()

            await session.rollback()

    async def test_declaration_optional_fields(self, async_db_session):
        """Test declaration with optional fields as None."""
        declaration = Declaration(
            name="Minimal.Declaration",
            module="Test",
            source_text="def minimal := 1",
            source_link="https://example.com",
            docstring=None,
            dependencies=None,
            informalization=None,
            informalization_embedding=None,
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        result = await async_db_session.execute(
            select(Declaration).where(Declaration.name == "Minimal.Declaration")
        )
        loaded = result.scalar_one()

        assert loaded.docstring is None
        assert loaded.dependencies is None
        assert loaded.informalization is None
        assert loaded.informalization_embedding is None
