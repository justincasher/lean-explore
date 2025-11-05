"""Tests for embedding generation.

These tests verify the generation of vector embeddings for declaration fields
using sentence transformers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lean_explore.extract.embeddings import (
    _get_declarations_needing_embeddings,
    _process_batch,
    generate_embeddings,
)
from lean_explore.models import Declaration


class TestDeclarationQuerying:
    """Tests for finding declarations needing embeddings."""

    async def test_get_declarations_needing_embeddings_all_missing(
        self, async_db_session
    ):
        """Test getting declarations with no embeddings."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            informalization="Test declaration",
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        declarations = await _get_declarations_needing_embeddings(
            async_db_session, limit=None
        )

        assert len(declarations) == 1
        assert declarations[0].name == "Test"

    async def test_get_declarations_needing_embeddings_partial(self, async_db_session):
        """Test getting declarations with some embeddings missing."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            name_embedding=[0.1] * 768,  # Has this
            source_text_embedding=[0.3] * 768,  # Has this
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        declarations = await _get_declarations_needing_embeddings(
            async_db_session, limit=None
        )

        # Should still return declaration since some embeddings are missing
        assert len(declarations) == 1

    async def test_get_declarations_needing_embeddings_all_complete(
        self, async_db_session
    ):
        """Test that declarations with all embeddings are not returned."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            docstring="Test",
            informalization="Test declaration",
            name_embedding=[0.1] * 768,
            informalization_embedding=[0.2] * 768,
            source_text_embedding=[0.3] * 768,
            docstring_embedding=[0.4] * 768,
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        declarations = await _get_declarations_needing_embeddings(
            async_db_session, limit=None
        )

        # Should not return since all embeddings present
        assert len(declarations) == 0

    async def test_get_declarations_with_limit(self, async_db_session):
        """Test querying with a limit."""
        for i in range(10):
            declaration = Declaration(
                name=f"Test{i}",
                module="Test",
                source_text=f"def test{i} := {i}",
                source_link=f"https://example.com/{i}",
            )
            async_db_session.add(declaration)
        await async_db_session.commit()

        declarations = await _get_declarations_needing_embeddings(
            async_db_session, limit=5
        )

        assert len(declarations) == 5


class TestBatchProcessing:
    """Tests for batch embedding generation."""

    async def test_process_batch_all_fields(
        self, async_db_session, mock_embedding_client
    ):
        """Test processing batch with all fields needing embeddings."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            docstring="Test docstring",
            informalization="Test informalization",
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        count = await _process_batch(
            async_db_session, [declaration], mock_embedding_client
        )

        # Should generate embeddings for: name, informalization, source_text, docstring
        assert count == 4

        # Verify embeddings were set
        result = await async_db_session.execute(
            select(Declaration).where(Declaration.name == "Test")
        )
        updated = result.scalar_one()

        assert updated.name_embedding is not None
        assert updated.informalization_embedding is not None
        assert updated.source_text_embedding is not None
        assert updated.docstring_embedding is not None

    async def test_process_batch_partial_fields(
        self, async_db_session, mock_embedding_client
    ):
        """Test processing batch where some embeddings already exist."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            name_embedding=[0.1] * 768,  # Already exists
            informalization="Test",
            informalization_embedding=None,  # Missing
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        count = await _process_batch(
            async_db_session, [declaration], mock_embedding_client
        )

        # Should only generate for: informalization and source_text
        # (name already exists)
        assert count == 2

    async def test_process_batch_no_optional_fields(
        self, async_db_session, mock_embedding_client
    ):
        """Test processing declarations without optional fields."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            docstring=None,
            informalization=None,
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        count = await _process_batch(
            async_db_session, [declaration], mock_embedding_client
        )

        # Should only generate for: name and source_text
        assert count == 2

        result = await async_db_session.execute(
            select(Declaration).where(Declaration.name == "Test")
        )
        updated = result.scalar_one()

        assert updated.name_embedding is not None
        assert updated.source_text_embedding is not None
        assert updated.informalization_embedding is None
        assert updated.docstring_embedding is None

    async def test_process_batch_multiple_declarations(
        self, async_db_session, mock_embedding_client
    ):
        """Test processing multiple declarations in one batch."""
        declarations = []
        for i in range(3):
            decl = Declaration(
                name=f"Test{i}",
                module="Test",
                source_text=f"def test{i} := {i}",
                source_link=f"https://example.com/{i}",
            )
            async_db_session.add(decl)
            declarations.append(decl)
        await async_db_session.commit()

        count = await _process_batch(
            async_db_session, declarations, mock_embedding_client
        )

        # Each declaration has name and source_text: 3 * 2 = 6
        assert count == 6

        # Verify all were updated
        result = await async_db_session.execute(select(Declaration))
        all_declarations = result.scalars().all()
        for declaration in all_declarations:
            assert declaration.name_embedding is not None
            assert declaration.source_text_embedding is not None

    async def test_process_batch_empty(self, async_db_session, mock_embedding_client):
        """Test processing empty batch."""
        count = await _process_batch(async_db_session, [], mock_embedding_client)

        assert count == 0


class TestGenerateEmbeddingsE2E:
    """End-to-end embedding generation tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_generate_embeddings_full_pipeline(self, async_db_engine):
        """Test complete embedding generation pipeline."""
        # Add declarations to database
        async with AsyncSession(async_db_engine) as session:
            declarations = [
                Declaration(
                    name="Nat",
                    module="Init",
                    source_text="inductive Nat | zero | succ",
                    source_link="https://example.com/nat",
                    informalization="Natural numbers",
                ),
                Declaration(
                    name="Nat.add",
                    module="Init",
                    source_text="def add (n m : Nat) := n + m",
                    source_link="https://example.com/add",
                    informalization="Addition of natural numbers",
                ),
            ]
            for declaration in declarations:
                session.add(declaration)
            await session.commit()

        # Mock the EmbeddingClient
        with patch(
            "lean_explore.extract.embeddings.EmbeddingClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.model_name = "test-model"
            mock_client.device = "cpu"

            # Create mock response
            mock_response = MagicMock()
            mock_response.embeddings = [
                [0.1] * 768 for _ in range(10)
            ]  # Enough for all fields
            mock_client.embed = AsyncMock(return_value=mock_response)

            mock_client_cls.return_value = mock_client

            await generate_embeddings(
                async_db_engine,
                model_name="test-model",
                batch_size=10,
                limit=None,
            )

        # Verify embeddings were generated
        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            all_declarations = result.scalars().all()

            assert len(all_declarations) == 2
            for declaration in all_declarations:
                # All should have name and source_text embeddings
                assert declaration.name_embedding is not None
                assert declaration.source_text_embedding is not None
                assert declaration.informalization_embedding is not None

    @pytest.mark.integration
    async def test_generate_embeddings_with_batching(self, async_db_engine):
        """Test embedding generation with small batch size."""
        async with AsyncSession(async_db_engine) as session:
            # Create many declarations
            for i in range(10):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def decl{i} := {i}",
                    source_link=f"https://example.com/{i}",
                )
                session.add(declaration)
            await session.commit()

        with patch(
            "lean_explore.extract.embeddings.EmbeddingClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.model_name = "test-model"
            mock_client.device = "cpu"

            # Create mock response with enough embeddings
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1] * 768 for _ in range(100)]
            mock_client.embed = AsyncMock(return_value=mock_response)

            mock_client_cls.return_value = mock_client

            # Use small batch size to test batching
            await generate_embeddings(
                async_db_engine,
                model_name="test-model",
                batch_size=3,
                limit=None,
            )

        # Verify all were processed
        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            all_declarations = result.scalars().all()

            assert len(all_declarations) == 10
            for declaration in all_declarations:
                assert declaration.name_embedding is not None

    @pytest.mark.integration
    async def test_generate_embeddings_with_limit(self, async_db_engine):
        """Test embedding generation with a limit."""
        async with AsyncSession(async_db_engine) as session:
            for i in range(10):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def decl{i} := {i}",
                    source_link=f"https://example.com/{i}",
                )
                session.add(declaration)
            await session.commit()

        with patch(
            "lean_explore.extract.embeddings.EmbeddingClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.model_name = "test-model"
            mock_client.device = "cpu"

            mock_response = MagicMock()
            mock_response.embeddings = [[0.1] * 768 for _ in range(20)]
            mock_client.embed = AsyncMock(return_value=mock_response)

            mock_client_cls.return_value = mock_client

            # Only process 5 declarations
            await generate_embeddings(
                async_db_engine,
                model_name="test-model",
                batch_size=10,
                limit=5,
            )

        # Verify only 5 were processed
        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(
                select(Declaration).where(Declaration.name_embedding.isnot(None))
            )
            declarations_with_embeddings = result.scalars().all()

            assert len(declarations_with_embeddings) == 5

    @pytest.mark.integration
    async def test_generate_embeddings_empty_database(self, async_db_engine):
        """Test embedding generation with no declarations."""
        with patch(
            "lean_explore.extract.embeddings.EmbeddingClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.model_name = "test-model"
            mock_client.device = "cpu"
            mock_client_cls.return_value = mock_client

            # Should complete without errors
            await generate_embeddings(
                async_db_engine,
                model_name="test-model",
                batch_size=10,
                limit=None,
            )
