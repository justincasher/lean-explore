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
        with patch("lean_explore.extract.index.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            device = _get_device()

            assert device == "cpu"

    def test_get_device_cuda(self):
        """Test device detection with CUDA available."""
        with patch("lean_explore.extract.index.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"

            device = _get_device()

            assert device == "cuda"

    def test_get_device_mps(self):
        """Test device detection with Apple Silicon."""
        with patch("lean_explore.extract.index.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True

            device = _get_device()

            assert device == "mps"


class TestEmbeddingLoading:
    """Tests for loading embeddings from database."""

    async def test_load_embeddings_from_database(self, async_db_session):
        """Test loading embeddings for a specific field."""
        # Add declarations with embeddings
        for i in range(3):
            declaration = Declaration(
                name=f"Test{i}",
                module="Test",
                source_text=f"def test{i} := {i}",
                source_link=f"https://example.com/{i}",
                name_embedding=[float(i)] * 768,
            )
            async_db_session.add(declaration)
        await async_db_session.commit()

        declaration_ids, embeddings = await _load_embeddings_from_database(
            async_db_session, "name_embedding"
        )

        assert len(declaration_ids) == 3
        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == np.float32

    async def test_load_embeddings_filters_none(self, async_db_session):
        """Test that declarations without embeddings are filtered out."""
        # Add declarations with and without embeddings
        decl_with = Declaration(
            name="HasEmbedding",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            name_embedding=[0.1] * 768,
        )
        decl_without = Declaration(
            name="NoEmbedding",
            module="Test",
            source_text="def test2 := 2",
            source_link="https://example.com",
        )
        async_db_session.add(decl_with)
        async_db_session.add(decl_without)
        await async_db_session.commit()

        declaration_ids, embeddings = await _load_embeddings_from_database(
            async_db_session, "name_embedding"
        )

        # Should only return the one with embedding
        assert len(declaration_ids) == 1
        assert embeddings.shape == (1, 768)

    async def test_load_embeddings_empty_database(self, async_db_session):
        """Test loading from database with no embeddings."""
        declaration_ids, embeddings = await _load_embeddings_from_database(
            async_db_session, "name_embedding"
        )

        assert declaration_ids == []
        assert embeddings.shape == (0,)

    async def test_load_embeddings_different_fields(self, async_db_session):
        """Test loading different embedding fields."""
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            name_embedding=[0.1] * 768,
            source_text_embedding=[0.2] * 768,
            informalization_embedding=[0.3] * 768,
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        # Load each field
        for field in [
            "name_embedding",
            "source_text_embedding",
            "informalization_embedding",
        ]:
            ids, embeddings = await _load_embeddings_from_database(
                async_db_session, field
            )
            assert len(ids) == 1
            assert embeddings.shape == (1, 768)


class TestFAISSIndexBuilding:
    """Tests for FAISS index construction."""

    def test_build_faiss_index_cpu(self):
        """Test building FAISS index on CPU."""
        embeddings = np.random.rand(100, 768).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cpu")

        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.ntotal == 100
        assert index.d == 768

    def test_build_faiss_index_small_dataset(self):
        """Test building index with small number of vectors."""
        embeddings = np.random.rand(5, 768).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cpu")

        assert index.ntotal == 5

    def test_build_faiss_index_search(self):
        """Test that built index can perform searches."""
        # Create embeddings with known structure
        embeddings = np.array(
            [
                [1.0] + [0.0] * 767,
                [0.0] + [1.0] + [0.0] * 766,
                [0.0] * 2 + [1.0] + [0.0] * 765,
            ]
        ).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cpu")

        # Search for vector similar to first embedding
        query = np.array([[1.0] + [0.0] * 767]).astype(np.float32)
        distances, indices = index.search(query, k=1)

        # Should return first embedding as closest
        assert indices[0][0] == 0

    @pytest.mark.external
    def test_build_faiss_index_cuda(self):
        """Test building FAISS index with CUDA (if available)."""
        if faiss.get_num_gpus() == 0:
            pytest.skip("No CUDA GPUs available")

        embeddings = np.random.rand(100, 768).astype(np.float32)

        index = _build_faiss_index(embeddings, device="cuda")

        # Should be GPU index
        assert isinstance(index, (faiss.GpuIndex, faiss.IndexHNSWFlat))
        assert index.ntotal == 100


class TestBuildFAISSIndices:
    """Tests for building all FAISS indices."""

    @pytest.mark.integration
    async def test_build_faiss_indices_full_pipeline(
        self, async_db_engine, temp_directory
    ):
        """Test building all FAISS indices from database."""
        # Add declarations with embeddings
        async with AsyncSession(async_db_engine) as session:
            for i in range(10):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    informalization=f"Declaration number {i}",
                    docstring=f"Doc {i}",
                    name_embedding=[float(i)] * 768,
                    source_text_embedding=[float(i) + 0.1] * 768,
                    informalization_embedding=[float(i) + 0.2] * 768,
                    docstring_embedding=[float(i) + 0.3] * 768,
                )
                session.add(declaration)
            await session.commit()

        output_directory = temp_directory / "indices"

        with patch("lean_explore.extract.index._get_device") as mock_device:
            mock_device.return_value = "cpu"

            await build_faiss_indices(async_db_engine, output_directory)

        # Verify index files were created
        assert (output_directory / "name_faiss.index").exists()
        assert (output_directory / "source_text_faiss.index").exists()
        assert (output_directory / "informalization_faiss.index").exists()
        assert (output_directory / "docstring_faiss.index").exists()

        # Verify ID mapping files were created
        assert (output_directory / "name_faiss_ids_map.json").exists()
        assert (output_directory / "source_text_faiss_ids_map.json").exists()
        assert (output_directory / "informalization_faiss_ids_map.json").exists()
        assert (output_directory / "docstring_faiss_ids_map.json").exists()

        # Verify index files can be loaded
        index = faiss.read_index(str(output_directory / "name_faiss.index"))
        assert index.ntotal == 10

        # Verify ID mappings are correct
        with open(output_directory / "name_faiss_ids_map.json") as f:
            id_mapping = json.load(f)
        assert len(id_mapping) == 10

    @pytest.mark.integration
    async def test_build_faiss_indices_partial_embeddings(
        self, async_db_engine, temp_directory
    ):
        """Test building indices when some embedding types are missing."""
        async with AsyncSession(async_db_engine) as session:
            # Add declarations with only some embeddings
            for i in range(5):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    name_embedding=[float(i)] * 768,
                    source_text_embedding=[float(i)] * 768,
                    # No informalization or docstring embeddings
                )
                session.add(declaration)
            await session.commit()

        output_directory = temp_directory / "indices"

        with patch("lean_explore.extract.index._get_device") as mock_device:
            mock_device.return_value = "cpu"

            await build_faiss_indices(async_db_engine, output_directory)

        # Should create indices for name and source_text only
        assert (output_directory / "name_faiss.index").exists()
        assert (output_directory / "source_text_faiss.index").exists()

        # Informalization and docstring indices should not be created
        # (or be skipped with warning)

    @pytest.mark.integration
    async def test_build_faiss_indices_empty_database(
        self, async_db_engine, temp_directory
    ):
        """Test building indices from empty database."""
        output_directory = temp_directory / "indices"

        with patch("lean_explore.extract.index._get_device") as mock_device:
            mock_device.return_value = "cpu"

            # Should complete without errors
            await build_faiss_indices(async_db_engine, output_directory)

        # No index files should be created
        index_files = list(output_directory.glob("*.index"))
        assert len(index_files) == 0

    @pytest.mark.integration
    async def test_build_faiss_indices_default_output_path(self, async_db_engine):
        """Test building indices with default output path."""
        async with AsyncSession(async_db_engine) as session:
            declaration = Declaration(
                name="Test",
                module="Test",
                source_text="def test := 1",
                source_link="https://example.com",
                name_embedding=[0.1] * 768,
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
                        await build_faiss_indices(
                            async_db_engine, output_directory=None
                        )

                        # Should use Config.ACTIVE_DATA_PATH
                        # Verify the path was created
                        mock_write.assert_called()

    @pytest.mark.integration
    async def test_build_faiss_indices_correct_id_mapping(
        self, async_db_engine, temp_directory
    ):
        """Test that ID mappings correctly correspond to FAISS indices."""
        async with AsyncSession(async_db_engine) as session:
            # Add declarations with known IDs
            declarations = []
            for i in range(5):
                declaration = Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    name_embedding=[float(i)] * 768,
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

            await build_faiss_indices(async_db_engine, output_directory)

        # Load ID mapping and verify it matches database IDs
        with open(output_directory / "name_faiss_ids_map.json") as f:
            id_mapping = json.load(f)

        assert id_mapping == expected_ids
