"""Tests for FAISS index building.

These tests verify the creation of FAISS HNSW indices from declaration embeddings.
"""

import json
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from lean_explore.extract.index import (
    _build_faiss_index,
    _get_device,
    _load_embeddings_from_database,
    build_faiss_indices,
)
from lean_explore.models import Declaration


class TestDeviceDetection:
    """Tests for device detection."""

    def test_get_device_cpu(self):
        """Test device detection defaulting to CPU."""
        with patch("lean_explore.extract.index.faiss") as mock_faiss:
            mock_faiss.get_num_gpus.return_value = 0

            device = _get_device()

            assert device == "cpu"

    def test_get_device_cuda(self):
        """Test device detection with CUDA available."""
        with patch("lean_explore.extract.index.faiss") as mock_faiss:
            mock_faiss.get_num_gpus.return_value = 1

            device = _get_device()

            assert device == "cuda"

    def test_get_device_mps(self):
        """Test device detection - MPS not supported by FAISS, falls back to CPU."""
        with patch("lean_explore.extract.index.faiss") as mock_faiss:
            # FAISS doesn't support MPS, so it will always report 0 GPUs on Mac
            mock_faiss.get_num_gpus.return_value = 0

            device = _get_device()

            # FAISS doesn't support MPS, so it uses CPU
            assert device == "cpu"


class TestEmbeddingLoading:
    """Tests for loading embeddings from database."""

    def test_load_embeddings_from_database(self, temp_directory):
        """Test loading embeddings for informalization field."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from lean_explore.models.search_db import Base

        db_path = temp_directory / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            # Add declarations with embeddings
            for i in range(3):
                declaration = Declaration(
                    name=f"Test{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    informalization=f"Test informalization {i}",
                    informalization_embedding=[float(i)] * 768,
                )
                session.add(declaration)
            session.commit()

            declaration_ids, embeddings = _load_embeddings_from_database(
                session, "informalization_embedding"
            )

        assert len(declaration_ids) == 3
        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == np.float32
        engine.dispose()

    def test_load_embeddings_filters_none(self, temp_directory):
        """Test that declarations without embeddings are filtered out."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from lean_explore.models.search_db import Base

        db_path = temp_directory / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            # Add declarations with and without embeddings
            decl_with = Declaration(
                name="HasEmbedding",
                module="Test",
                source_text="def test := 1",
                source_link="https://example.com",
                informalization="Has embedding",
                informalization_embedding=[0.1] * 768,
            )
            decl_without = Declaration(
                name="NoEmbedding",
                module="Test",
                source_text="def test2 := 2",
                source_link="https://example.com",
                informalization="No embedding",
            )
            session.add(decl_with)
            session.add(decl_without)
            session.commit()

            declaration_ids, embeddings = _load_embeddings_from_database(
                session, "informalization_embedding"
            )

        # Should only return the one with embedding
        assert len(declaration_ids) == 1
        assert embeddings.shape == (1, 768)
        engine.dispose()

    def test_load_embeddings_empty_database(self, temp_directory):
        """Test loading from database with no embeddings."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from lean_explore.models.search_db import Base

        db_path = temp_directory / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            declaration_ids, embeddings = _load_embeddings_from_database(
                session, "informalization_embedding"
            )

        assert declaration_ids == []
        assert embeddings.shape == (0,)
        engine.dispose()


class TestFAISSIndexBuilding:
    """Tests for FAISS IVF index construction.

    Note: These tests are marked as external because FAISS training causes
    segfaults on macOS due to OpenMP library conflicts between torch and FAISS.
    """

    @pytest.mark.external
    def test_build_faiss_index_cpu(self):
        """Test building FAISS IVF index on CPU."""
        # Need enough vectors to train IVF (at least 256 for default nlist)
        embeddings = np.random.rand(300, 768).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cpu")

        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.ntotal == 300
        assert index.d == 768

    @pytest.mark.external
    def test_build_faiss_index_small_dataset(self):
        """Test building index with smaller number of vectors."""
        # Still need minimum vectors for IVF training
        embeddings = np.random.rand(300, 768).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cpu")

        assert index.ntotal == 300

    @pytest.mark.external
    def test_build_faiss_index_search(self):
        """Test that built index can perform searches."""
        # Create embeddings with known structure, enough for IVF training
        num_vectors = 300
        dimension = 768
        embeddings = np.random.rand(num_vectors, dimension).astype(np.float32)
        # Make first vector distinctive
        embeddings[0] = np.array([1.0] + [0.0] * (dimension - 1), dtype=np.float32)

        index = _build_faiss_index(embeddings, device="cpu")

        # Set nprobe for better recall on IVF index
        index.nprobe = 10

        # Search for vector similar to first embedding
        query = np.array([[1.0] + [0.0] * (dimension - 1)]).astype(np.float32)
        distances, indices = index.search(query, k=1)

        # Should return first embedding as closest
        assert indices[0][0] == 0

    @pytest.mark.external
    def test_build_faiss_index_cuda(self):
        """Test building FAISS index with CUDA (if available)."""
        if faiss.get_num_gpus() == 0:
            pytest.skip("No CUDA GPUs available")

        # Need enough vectors for IVF training
        embeddings = np.random.rand(300, 768).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cuda")

        # Index is converted back to CPU after GPU training
        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.ntotal == 300


class TestBuildFAISSIndices:
    """Tests for building all FAISS indices.

    Note: These tests are marked as external because FAISS training causes
    segfaults on macOS due to OpenMP library conflicts between torch and FAISS.
    """

    @pytest.mark.external
    async def test_build_faiss_indices_full_pipeline(self, temp_directory):
        """Test building FAISS index from database."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from lean_explore.models.search_db import Base

        # Use file-based database (sync engine needs persistent connection)
        db_path = temp_directory / "test.db"
        async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Add declarations with embeddings (need 300+ for IVF training)
        num_declarations = 300
        async with AsyncSession(async_engine) as session:
            for i in range(num_declarations):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    informalization=f"Declaration number {i}",
                    informalization_embedding=[float(i % 100) / 100.0 + 0.1] * 768,
                )
                session.add(declaration)
            await session.commit()

        output_directory = temp_directory / "indices"

        with patch("lean_explore.extract.index._get_device") as mock_device:
            mock_device.return_value = "cpu"

            await build_faiss_indices(async_engine, output_directory)

        await async_engine.dispose()

        # Verify index file was created
        assert (output_directory / "informalization_faiss.index").exists()

        # Verify ID mapping file was created
        assert (output_directory / "informalization_faiss_ids_map.json").exists()

        # Verify index file can be loaded
        index = faiss.read_index(str(output_directory / "informalization_faiss.index"))
        assert index.ntotal == num_declarations

        # Verify ID mappings are correct
        with open(output_directory / "informalization_faiss_ids_map.json") as f:
            id_mapping = json.load(f)
        assert len(id_mapping) == num_declarations

    @pytest.mark.integration
    async def test_build_faiss_indices_empty_database(self, temp_directory):
        """Test building indices from empty database."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from lean_explore.models.search_db import Base

        # Use file-based database
        db_path = temp_directory / "test.db"
        async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        output_directory = temp_directory / "indices"

        with patch("lean_explore.extract.index._get_device") as mock_device:
            mock_device.return_value = "cpu"

            # Should complete without errors
            await build_faiss_indices(async_engine, output_directory)

        await async_engine.dispose()

        # No index files should be created
        index_files = list(output_directory.glob("*.index"))
        assert len(index_files) == 0

    @pytest.mark.external
    async def test_build_faiss_indices_default_output_path(self, temp_directory):
        """Test building indices with default output path."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from lean_explore.models.search_db import Base

        # Use file-based database
        db_path = temp_directory / "test.db"
        async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Add enough declarations for IVF training
        async with AsyncSession(async_engine) as session:
            for i in range(300):
                declaration = Declaration(
                    name=f"Test{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    informalization=f"Test informalization {i}",
                    informalization_embedding=[float(i % 100) / 100.0 + 0.1] * 768,
                )
                session.add(declaration)
            await session.commit()

        with patch("lean_explore.extract.index.Config") as mock_config:
            mock_output_path = Path("/tmp/test_output")
            mock_config.ACTIVE_DATA_PATH = mock_output_path

            with patch("lean_explore.extract.index._get_device") as mock_device:
                mock_device.return_value = "cpu"

                with patch(
                    "lean_explore.extract.index.faiss.write_index"
                ) as mock_write:
                    with patch("builtins.open", create=True):
                        await build_faiss_indices(async_engine, output_directory=None)

                        # Should use Config.ACTIVE_DATA_PATH
                        # Verify the path was created
                        mock_write.assert_called()

        await async_engine.dispose()

    @pytest.mark.external
    async def test_build_faiss_indices_correct_id_mapping(self, temp_directory):
        """Test that ID mappings correctly correspond to FAISS indices."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from lean_explore.models.search_db import Base

        # Use file-based database
        db_path = temp_directory / "test.db"
        async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        num_declarations = 300  # Need enough for IVF training
        async with AsyncSession(async_engine) as session:
            # Add declarations with known IDs
            declarations = []
            for i in range(num_declarations):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    informalization=f"Declaration number {i}",
                    informalization_embedding=[float(i % 100) / 100.0 + 0.1] * 768,
                )
                session.add(declaration)
                declarations.append(declaration)
            await session.commit()

            # Get the actual database IDs
            from sqlalchemy import select

            result = await session.execute(select(Declaration).order_by(Declaration.id))
            db_declarations = result.scalars().all()
            expected_ids = [d.id for d in db_declarations]

        output_directory = temp_directory / "indices"

        with patch("lean_explore.extract.index._get_device") as mock_device:
            mock_device.return_value = "cpu"

            await build_faiss_indices(async_engine, output_directory)

        await async_engine.dispose()

        # Load ID mapping and verify it matches database IDs
        with open(output_directory / "informalization_faiss_ids_map.json") as f:
            id_mapping = json.load(f)

        assert id_mapping == expected_ids
