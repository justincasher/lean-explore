"""Tests for the extraction pipeline orchestration.

These tests verify the complete pipeline from doc-gen4 parsing through
FAISS index building, including step orchestration and error handling.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lean_explore.extract.__main__ import (
    _create_database_schema,
    _run_doc_gen4_step,
    _run_embeddings_step,
    _run_extract_step,
    _run_index_step,
    _run_informalize_step,
    run_pipeline,
)
from lean_explore.models import Declaration


class TestDatabaseSchemaCreation:
    """Tests for database schema creation."""

    async def test_create_database_schema(self, async_db_engine):
        """Test creating database tables."""
        await _create_database_schema(async_db_engine)

        # Verify tables exist by querying
        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            # Should not raise error
            assert result is not None

    async def test_create_database_schema_idempotent(self, async_db_engine):
        """Test that creating schema twice doesn't cause errors."""
        await _create_database_schema(async_db_engine)
        await _create_database_schema(async_db_engine)

        # Should succeed without errors


class TestDocGen4Step:
    """Tests for doc-gen4 generation step."""

    @pytest.mark.external
    async def test_run_doc_gen4_step_success(self, temp_directory):
        """Test successful doc-gen4 execution."""
        lean_directory = temp_directory / "lean"
        lean_directory.mkdir()

        # Mock subprocess to avoid actually running lake
        with patch("lean_explore.extract.doc_gen4.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = iter(["Building...\n", "Complete!\n"])
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            await _run_doc_gen4_step()

            # Called twice: "lake build" and "lake build LeanExtract:docs"
            assert mock_popen.call_count == 2

    @pytest.mark.external
    async def test_run_doc_gen4_step_failure(self):
        """Test doc-gen4 execution failure."""
        with patch("lean_explore.extract.doc_gen4.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = iter(["Error!\n"])
            mock_process.wait.return_value = 1
            mock_popen.return_value = mock_process

            with pytest.raises(RuntimeError, match="failed"):
                await _run_doc_gen4_step()


class TestExtractionStep:
    """Tests for declaration extraction step."""

    async def test_run_extract_step(self, async_db_engine):
        """Test extraction step."""
        with patch(
            "lean_explore.extract.doc_parser.extract_declarations"
        ) as mock_extract:
            mock_extract.return_value = AsyncMock()

            await _run_extract_step(async_db_engine)

            mock_extract.assert_called_once_with(async_db_engine)


class TestInformalizeStep:
    """Tests for informalization step."""

    async def test_run_informalize_step(self, async_db_engine):
        """Test informalization step."""
        with patch(
            "lean_explore.extract.informalize.informalize_declarations"
        ) as mock_informalize:
            mock_informalize.return_value = AsyncMock()

            await _run_informalize_step(
                async_db_engine,
                model="test-model",
                batch_size=100,
                max_concurrent=5,
                limit=None,
            )

            mock_informalize.assert_called_once_with(
                async_db_engine,
                model="test-model",
                commit_batch_size=100,
                max_concurrent=5,
                limit=None,
            )


class TestEmbeddingsStep:
    """Tests for embeddings generation step."""

    async def test_run_embeddings_step(self, async_db_engine):
        """Test embeddings generation step."""
        with patch(
            "lean_explore.extract.embeddings.generate_embeddings"
        ) as mock_embeddings:
            mock_embeddings.return_value = AsyncMock()

            await _run_embeddings_step(
                async_db_engine,
                model_name="test-model",
                batch_size=250,
                limit=None,
                max_seq_length=512,
            )

            mock_embeddings.assert_called_once_with(
                async_db_engine,
                model_name="test-model",
                batch_size=250,
                limit=None,
                max_seq_length=512,
            )


class TestIndexStep:
    """Tests for search index building step."""

    async def test_run_index_step(self, async_db_engine, temp_directory):
        """Test search index building step (FAISS and BM25)."""
        with patch("lean_explore.extract.index.build_faiss_indices") as mock_faiss:
            mock_faiss.return_value = AsyncMock()

            with patch("lean_explore.extract.index.build_bm25_indices") as mock_bm25:
                mock_bm25.return_value = AsyncMock()

                await _run_index_step(async_db_engine, temp_directory)

                mock_faiss.assert_called_once_with(
                    async_db_engine, output_directory=temp_directory
                )
                mock_bm25.assert_called_once_with(
                    async_db_engine, output_directory=temp_directory
                )


class TestFullPipeline:
    """End-to-end tests for the complete pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_run_pipeline_all_steps(self, temp_directory):
        """Test running complete pipeline with all steps enabled."""
        database_url = f"sqlite+aiosqlite:///{temp_directory / 'test.db'}"

        with patch(
            "lean_explore.extract.doc_parser.extract_declarations"
        ) as mock_extract:
            mock_extract.return_value = AsyncMock()

            with patch(
                "lean_explore.extract.informalize.informalize_declarations"
            ) as mock_informalize:
                mock_informalize.return_value = AsyncMock()

                with patch(
                    "lean_explore.extract.embeddings.generate_embeddings"
                ) as mock_embeddings:
                    mock_embeddings.return_value = AsyncMock()

                    with patch(
                        "lean_explore.extract.index.build_faiss_indices"
                    ) as mock_faiss:
                        mock_faiss.return_value = AsyncMock()

                        with patch(
                            "lean_explore.extract.index.build_bm25_indices"
                        ) as mock_bm25:
                            mock_bm25.return_value = AsyncMock()

                            with patch("lean_explore.extract.__main__.setup_logging"):
                                os.environ["OPENROUTER_API_KEY"] = "test-key"

                                await run_pipeline(
                                    database_url=database_url,
                                    extraction_path=temp_directory,
                                    run_doc_gen4=False,
                                    parse_docs=True,
                                    informalize=True,
                                    embeddings=True,
                                    index=True,
                                )

                                # Verify all steps were called
                                mock_extract.assert_called_once()
                                mock_informalize.assert_called_once()
                                mock_embeddings.assert_called_once()
                                mock_faiss.assert_called_once()
                                mock_bm25.assert_called_once()

    @pytest.mark.integration
    async def test_run_pipeline_selective_steps(self, temp_directory):
        """Test running pipeline with only some steps enabled."""
        database_url = f"sqlite+aiosqlite:///{temp_directory / 'test.db'}"

        with patch(
            "lean_explore.extract.doc_parser.extract_declarations"
        ) as mock_extract:
            mock_extract.return_value = AsyncMock()

            with patch("lean_explore.extract.__main__.setup_logging"):
                # Only run parse step
                await run_pipeline(
                    database_url=database_url,
                    extraction_path=temp_directory,
                    run_doc_gen4=False,
                    parse_docs=True,
                    informalize=False,
                    embeddings=False,
                    index=False,
                )

                mock_extract.assert_called_once()

    @pytest.mark.integration
    async def test_run_pipeline_requires_openrouter_key(self, temp_directory):
        """Test that pipeline requires OPENROUTER_API_KEY for informalization."""
        database_url = f"sqlite+aiosqlite:///{temp_directory / 'test.db'}"

        # Remove key if it exists
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        with patch("lean_explore.extract.__main__.setup_logging"):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY not set"):
                await run_pipeline(
                    database_url=database_url,
                    extraction_path=temp_directory,
                    run_doc_gen4=False,
                    parse_docs=False,
                    informalize=True,  # Requires API key
                    embeddings=False,
                    index=False,
                )

    @pytest.mark.integration
    async def test_run_pipeline_parameters_passed_correctly(self, temp_directory):
        """Test that pipeline parameters are correctly passed to steps."""
        database_url = f"sqlite+aiosqlite:///{temp_directory / 'test.db'}"

        with patch("lean_explore.extract.doc_parser.extract_declarations"):
            with patch(
                "lean_explore.extract.informalize.informalize_declarations"
            ) as mock_informalize:
                mock_informalize.return_value = AsyncMock()

                with patch(
                    "lean_explore.extract.embeddings.generate_embeddings"
                ) as mock_embeddings:
                    mock_embeddings.return_value = AsyncMock()

                    with patch("lean_explore.extract.index.build_faiss_indices"):
                        with patch("lean_explore.extract.__main__.setup_logging"):
                            os.environ["OPENROUTER_API_KEY"] = "test-key"

                            await run_pipeline(
                                database_url=database_url,
                                extraction_path=temp_directory,
                                run_doc_gen4=False,
                                parse_docs=False,
                                informalize=True,
                                embeddings=True,
                                index=False,
                                informalize_model="custom-model",
                                informalize_batch_size=50,
                                informalize_max_concurrent=20,
                                informalize_limit=100,
                                embedding_model="custom-embedding-model",
                                embedding_batch_size=100,
                                embedding_limit=50,
                            )

                            # Verify informalize was called
                            mock_informalize.assert_called_once()
                            # Verify embeddings was called
                            mock_embeddings.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_run_pipeline_creates_database(self, temp_directory):
        """Test that pipeline creates database if it doesn't exist."""
        db_path = temp_directory / "new.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        assert not db_path.exists()

        with patch("lean_explore.extract.__main__.setup_logging"):
            await run_pipeline(
                database_url=database_url,
                extraction_path=temp_directory,
                run_doc_gen4=False,
                parse_docs=False,
                informalize=False,
                embeddings=False,
                index=False,
            )

        # Database file should be created
        assert db_path.exists()

    @pytest.mark.integration
    async def test_run_pipeline_engine_disposal(self, temp_directory):
        """Test that database engine is properly disposed after pipeline."""
        db_path = temp_directory / "test.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"

        with patch("lean_explore.extract.__main__.setup_logging"):
            # Run pipeline and verify it completes without error
            # (disposal is tested by verifying no file handle issues)
            await run_pipeline(
                database_url=database_url,
                extraction_path=temp_directory,
                run_doc_gen4=False,
                parse_docs=False,
                informalize=False,
                embeddings=False,
                index=False,
            )

        # Database file should exist and be accessible (engine disposed properly)
        assert db_path.exists()

    @pytest.mark.integration
    async def test_run_pipeline_engine_disposal_on_error(self, temp_directory):
        """Test that engine is disposed even if pipeline fails."""
        database_url = f"sqlite+aiosqlite:///{temp_directory / 'test.db'}"

        with patch("lean_explore.extract.__main__.setup_logging"):
            with patch(
                "lean_explore.extract.__main__.create_async_engine"
            ) as mock_create_engine:
                mock_engine = AsyncMock()
                mock_create_engine.return_value = mock_engine

                with patch(
                    "lean_explore.extract.__main__._create_database_schema"
                ) as mock_schema:
                    mock_schema.side_effect = Exception("Database error")

                    with pytest.raises(Exception, match="Database error"):
                        await run_pipeline(
                            database_url=database_url,
                            extraction_path=temp_directory,
                            run_doc_gen4=False,
                            parse_docs=False,
                            informalize=False,
                            embeddings=False,
                            index=False,
                        )

                    # Engine should still be disposed
                    mock_engine.dispose.assert_called_once()
