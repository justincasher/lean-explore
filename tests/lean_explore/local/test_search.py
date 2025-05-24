# tests/local/test_search.py

"""Tests for core search logic and helper functions in `lean_explore.search`.

This module includes tests for performance logging,
asset loading (embedding models, FAISS indices), the main
`perform_search` algorithm, and the script's CLI execution.
"""

import argparse
import datetime
import json
import logging
import pathlib
import sys  # For mocking sys.argv and sys.exit
from typing import TYPE_CHECKING, Dict, Tuple
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from filelock import FileLock as FLock
from filelock import Timeout  # Alias FLock
from sentence_transformers import SentenceTransformer
from sqlalchemy.engine import Engine as SQLAlchemyEngine
from sqlalchemy.orm import Session as SQLAlchemySession

from lean_explore import defaults
from lean_explore.local import search as search_module
from lean_explore.shared.models.db import Declaration, StatementGroup

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from sqlalchemy.engine import Engine as SQLAlchemyEngine  # For specing mock engine


# --- Tests for Helper Functions (Assumed to be passing from previous steps) ---
class TestLogSearchEventToJson:
    """Tests for the `log_search_event_to_json` function."""

    @pytest.fixture
    def mock_log_paths(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        isolated_data_paths: pathlib.Path,
    ):
        """Mocks performance log paths to use a temporary directory.

        Leverages `isolated_data_paths` to ensure base paths are already mocked
        and then further ensures the log-specific paths within `search_module`
        point to the temporary directory.
        """
        temp_log_dir = (
            isolated_data_paths.parent / "logs_for_test"
        )  # Ensure a unique subdir under tmp_path
        temp_log_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(search_module, "_USER_LOGS_BASE_DIR", temp_log_dir)
        monkeypatch.setattr(search_module, "PERFORMANCE_LOG_DIR", str(temp_log_dir))

        temp_log_file = temp_log_dir / search_module.PERFORMANCE_LOG_FILENAME
        monkeypatch.setattr(search_module, "PERFORMANCE_LOG_PATH", str(temp_log_file))

        temp_lock_file = temp_log_dir / f"{search_module.PERFORMANCE_LOG_FILENAME}.lock"
        monkeypatch.setattr(search_module, "LOCK_PATH", str(temp_lock_file))

        # Ensure the log file is empty before each test that uses this fixture
        if temp_log_file.exists():
            temp_log_file.unlink()

        return temp_log_file, temp_lock_file

    def test_successful_log_write(
        self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path]
    ):
        """Verifies a successful log write with correct data format.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
        """
        log_file, _ = mock_log_paths
        timestamp_before = datetime.datetime.now(datetime.timezone.utc)
        search_module.log_search_event_to_json("SUCCESS", 123.456, 10)
        assert log_file.exists()
        log_content = json.loads(log_file.read_text(encoding="utf-8"))
        assert log_content["event"] == "search_processed"
        assert log_content["status"] == "SUCCESS"
        assert log_content["duration_ms"] == 123.46
        assert log_content["results_count"] == 10
        assert "error_type" not in log_content
        log_timestamp = datetime.datetime.fromisoformat(
            log_content["timestamp"].replace("Z", "+00:00")
        )
        timestamp_after = datetime.datetime.now(datetime.timezone.utc)
        assert timestamp_before <= log_timestamp <= timestamp_after

    def test_log_with_error_type(
        self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path]
    ):
        """Ensures `error_type` is included in the log when provided.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
        """
        log_file, _ = mock_log_paths
        search_module.log_search_event_to_json("ERROR", 50.0, 0, error_type="TestError")
        log_content = json.loads(log_file.read_text(encoding="utf-8"))
        assert log_content["error_type"] == "TestError"

    def test_log_directory_creation_failure(
        self,
        mock_log_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when log directory creation fails.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_print = mocker.patch("builtins.print")
        mocker.patch.object(
            search_module.os, "makedirs", side_effect=OSError("Cannot create dir")
        )
        with caplog.at_level(logging.ERROR):
            search_module.log_search_event_to_json("SUCCESS_DIR_FAIL", 10.0, 1)
        assert (
            "Performance logging error: Could not create log directory" in caplog.text
        )
        assert "Cannot create dir" in caplog.text
        mock_print.assert_called_once()
        assert "FALLBACK_PERF_LOG (DIR_ERROR)" in mock_print.call_args[0][0]

    def test_log_filelock_timeout(
        self,
        mock_log_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when acquiring the file lock times out.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        log_file, _ = mock_log_paths
        mock_print = mocker.patch("builtins.print")
        mock_file_lock_instance = mocker.MagicMock(spec=FLock)
        mock_file_lock_instance.__enter__.side_effect = Timeout("Lock timeout")
        mocker.patch.object(
            search_module, "FileLock", return_value=mock_file_lock_instance
        )
        with caplog.at_level(logging.WARNING):
            search_module.log_search_event_to_json("SUCCESS_LOCK_FAIL", 20.0, 2)
        assert not log_file.exists()
        assert "Performance logging error: Timeout acquiring lock" in caplog.text
        mock_print.assert_called_once()
        assert "FALLBACK_PERF_LOG (LOCK_TIMEOUT)" in mock_print.call_args[0][0]

    def test_log_file_write_failure(
        self,
        mock_log_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when writing to the log file fails.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_print = mocker.patch("builtins.print")
        mocker.patch("builtins.open", side_effect=OSError("Cannot write to file"))
        with caplog.at_level(logging.ERROR):
            search_module.log_search_event_to_json("SUCCESS_WRITE_FAIL", 30.0, 3)
        assert "Performance logging error: Failed to write to" in caplog.text
        assert "Cannot write to file" in caplog.text
        mock_print.assert_called_once()
        assert "FALLBACK_PERF_LOG (WRITE_ERROR)" in mock_print.call_args[0][0]


class TestLoadEmbeddingModel:
    """Tests for the `load_embedding_model` function."""

    def test_successful_load(
        self, mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
    ):
        """Verifies successful model loading simulation.

        Args:
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_st_constructor = mocker.patch.object(search_module, "SentenceTransformer")
        mock_model_instance = MagicMock(spec=SentenceTransformer)
        mock_model_instance.max_seq_length = 512
        mock_st_constructor.return_value = mock_model_instance
        model_name = "test-model"
        with caplog.at_level(logging.INFO):
            model = search_module.load_embedding_model(model_name)
        assert model is mock_model_instance
        mock_st_constructor.assert_called_once_with(model_name)
        assert f"Loading sentence transformer model '{model_name}'..." in caplog.text
        assert f"Model '{model_name}' loaded successfully." in caplog.text

    def test_load_failure(
        self, mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
    ):
        """Tests behavior when model loading raises an exception.

        Args:
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mocker.patch.object(
            search_module,
            "SentenceTransformer",
            side_effect=Exception("Model loading failed"),
        )
        model_name = "failing-model"
        with caplog.at_level(logging.ERROR):
            model = search_module.load_embedding_model(model_name)
        assert model is None
        assert (
            f"Failed to load sentence transformer model '{model_name}'" in caplog.text
        )
        assert "Model loading failed" in caplog.text


class TestLoadFaissAssets:
    """Tests for the `load_faiss_assets` function."""

    @pytest.fixture
    def faiss_asset_paths(
        self, tmp_path: pathlib.Path
    ) -> Tuple[pathlib.Path, pathlib.Path]:
        """Provides paths for temporary FAISS index and map files.

        Args:
            tmp_path: Pytest's built-in temporary directory fixture.

        Returns:
            A tuple containing paths to the temporary index file and map file.
        """
        index_file = tmp_path / "test.index"
        map_file = tmp_path / "test_map.json"
        return index_file, map_file

    def test_successful_load(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Verifies successful loading of FAISS index and ID map.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        index_file, map_file = faiss_asset_paths
        map_data = ["sg_1", "sg_2"]
        map_file.write_text(json.dumps(map_data), encoding="utf-8")
        index_file.touch()
        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.ntotal = len(map_data)
        mock_faiss_index.metric_type = faiss.METRIC_L2
        mocker.patch.object(
            search_module.faiss, "read_index", return_value=mock_faiss_index
        )
        with caplog.at_level(logging.INFO):
            index, id_map = search_module.load_faiss_assets(
                str(index_file), str(map_file)
            )
        assert index is mock_faiss_index
        assert id_map == map_data
        search_module.faiss.read_index.assert_called_once_with(str(index_file))
        assert (
            f"Loaded FAISS index with {mock_faiss_index.ntotal} vectors" in caplog.text
        )
        assert f"Loaded ID map with {len(map_data)} entries" in caplog.text
        assert "Mismatch: FAISS index size" not in caplog.text

    def test_index_file_not_found(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when the FAISS index file is not found.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            caplog: Pytest fixture to capture log output.
        """
        _, map_file = faiss_asset_paths
        map_file.write_text(json.dumps(["sg_1"]), encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(
                "non_existent.index", str(map_file)
            )
        assert index is None
        assert id_map is None
        assert "FAISS index file not found" in caplog.text
        assert "non_existent.index" in caplog.text

    def test_map_file_not_found(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when the FAISS ID map file is not found.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        index_file, _ = faiss_asset_paths
        index_file.touch()
        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.ntotal = 0
        mocker.patch.object(
            search_module.faiss, "read_index", return_value=mock_faiss_index
        )
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(
                str(index_file), "non_existent_map.json"
            )
        assert index is None
        assert id_map is None
        assert "FAISS ID map file not found" in caplog.text
        assert "non_existent_map.json" in caplog.text

    def test_faiss_read_index_error(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when `faiss.read_index` raises an exception.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        index_file, map_file = faiss_asset_paths
        index_file.touch()
        map_file.write_text(json.dumps(["id1"]), encoding="utf-8")
        mocker.patch.object(
            search_module.faiss,
            "read_index",
            side_effect=RuntimeError("FAISS load error"),
        )
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(
                str(index_file), str(map_file)
            )
        assert index is None
        assert id_map is None
        assert f"Failed to load FAISS index from {index_file}" in caplog.text
        assert "FAISS load error" in caplog.text

    def test_json_load_error_for_map(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when the ID map file contains invalid JSON.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        index_file, map_file = faiss_asset_paths
        index_file.touch()
        map_file.write_text("this is not json", encoding="utf-8")
        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.ntotal = 0
        mocker.patch.object(
            search_module.faiss, "read_index", return_value=mock_faiss_index
        )
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(
                str(index_file), str(map_file)
            )
        assert index is mock_faiss_index
        assert id_map is None
        assert f"Failed to load or parse ID map file {map_file}" in caplog.text
        assert "JSONDecodeError" in caplog.text or "Expecting value" in caplog.text

    def test_map_file_not_a_list(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests behavior when the ID map file's JSON content is not a list.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        index_file, map_file = faiss_asset_paths
        index_file.touch()
        map_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.ntotal = 0
        mocker.patch.object(
            search_module.faiss, "read_index", return_value=mock_faiss_index
        )
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(
                str(index_file), str(map_file)
            )
        assert index is mock_faiss_index
        assert id_map is None
        assert (
            f"ID map file ({map_file}) does not contain a valid JSON list."
            in caplog.text
        )

    def test_size_mismatch_warning(
        self,
        faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path],
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests that a warning is logged if FAISS index and ID map sizes differ.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        index_file, map_file = faiss_asset_paths
        map_data = ["sg_1", "sg_2"]
        map_file.write_text(json.dumps(map_data), encoding="utf-8")
        index_file.touch()
        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.ntotal = 3
        mock_faiss_index.metric_type = faiss.METRIC_L2
        mocker.patch.object(
            search_module.faiss, "read_index", return_value=mock_faiss_index
        )
        with caplog.at_level(logging.WARNING):
            search_module.load_faiss_assets(str(index_file), str(map_file))
        assert "Mismatch: FAISS index size (3) vs ID map size (2)." in caplog.text


# --- Tests for perform_search ---
class TestPerformSearch:
    """Tests for the core `perform_search` function."""

    @pytest.fixture
    def mock_search_dependencies(self, mocker: "MockerFixture") -> Dict[str, MagicMock]:
        """Provides common mocks for `perform_search` dependencies.

        Yields a dictionary of mocks for: model, faiss_index,
        and log_search_event_to_json.

        Args:
            mocker: Pytest-mock's mocker fixture.

        Returns:
            Dict[str, MagicMock]: A dictionary of named mock objects.
        """
        mock_model = MagicMock(spec=SentenceTransformer)
        mock_model.encode.return_value = [np.array([0.1, 0.2], dtype=np.float32)]

        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.metric_type = faiss.METRIC_L2
        mock_faiss_index.nprobe = 1
        mock_faiss_index.search.return_value = (np.array([[]]), np.array([[]]))

        mock_log_event = mocker.patch.object(search_module, "log_search_event_to_json")

        return {
            "model": mock_model,
            "faiss_index": mock_faiss_index,
            "log_event": mock_log_event,
        }

    def test_empty_query_string(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
    ):
        """Verifies behavior with an empty query string.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        results = search_module.perform_search(
            session=db_session,
            query_string="  ",
            model=mock_search_dependencies["model"],
            faiss_index=mock_search_dependencies["faiss_index"],
            text_chunk_id_map=[],
            faiss_k=10,
            pagerank_weight=0.1,
            text_relevance_weight=0.1,
            log_searches=True,
        )
        assert results == []
        mock_search_dependencies["log_event"].assert_called_once()
        assert (
            mock_search_dependencies["log_event"].call_args.kwargs["status"]
            == "EMPTY_QUERY_SUBMITTED"
        )

    def test_query_embedding_failure(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
    ):
        """Tests exception handling when model embedding fails.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        mock_search_dependencies["model"].encode.side_effect = Exception(
            "Embedding boom"
        )
        with pytest.raises(Exception, match="Query embedding failed: Embedding boom"):
            search_module.perform_search(
                session=db_session,
                query_string="test",
                model=mock_search_dependencies["model"],
                faiss_index=mock_search_dependencies["faiss_index"],
                text_chunk_id_map=[],
                faiss_k=10,
                pagerank_weight=0.1,
                text_relevance_weight=0.1,
                log_searches=True,
            )
        mock_search_dependencies["log_event"].assert_called_once()
        assert (
            mock_search_dependencies["log_event"].call_args.kwargs["status"]
            == "EMBEDDING_ERROR"
        )

    def test_faiss_search_failure(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
    ):
        """Tests exception handling when FAISS search fails.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        mock_search_dependencies["faiss_index"].search.side_effect = Exception(
            "FAISS boom"
        )
        with pytest.raises(Exception, match="FAISS search failed: FAISS boom"):
            search_module.perform_search(
                session=db_session,
                query_string="test",
                model=mock_search_dependencies["model"],
                faiss_index=mock_search_dependencies["faiss_index"],
                text_chunk_id_map=[],
                faiss_k=10,
                pagerank_weight=0.1,
                text_relevance_weight=0.1,
                log_searches=True,
            )
        mock_search_dependencies["log_event"].assert_called_once()
        assert (
            mock_search_dependencies["log_event"].call_args.kwargs["status"]
            == "FAISS_SEARCH_ERROR"
        )

    def test_no_faiss_candidates_after_parsing(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests when FAISS results don't parse to valid SG IDs.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
            caplog: Pytest fixture to capture log output.
        """
        mock_search_dependencies["faiss_index"].search.return_value = (
            np.array([[0.5]]),
            np.array([[0]]),
        )
        mock_text_chunk_id_map = ["malformed_id"]

        with caplog.at_level(logging.WARNING):
            results = search_module.perform_search(
                session=db_session,
                query_string="test",
                model=mock_search_dependencies["model"],
                faiss_index=mock_search_dependencies["faiss_index"],
                text_chunk_id_map=mock_text_chunk_id_map,
                faiss_k=10,
                pagerank_weight=0.1,
                text_relevance_weight=0.1,
                log_searches=True,
            )
        assert results == []
        assert "Malformed text_chunk_id format" in caplog.text
        assert any(
            call.kwargs["status"] == "NO_FAISS_CANDIDATES"
            for call in mock_search_dependencies["log_event"].call_args_list
        )

    def test_semantic_thresholding(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests that semantic similarity threshold correctly filters candidates.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
            caplog: Pytest fixture to capture log output.
        """
        mock_faiss_index = mock_search_dependencies["faiss_index"]
        mock_faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
        mock_faiss_index.search.return_value = (
            np.array([[0.8, 0.6, 0.4]]),
            np.array([[0, 1, 2]]),
        )
        text_chunk_id_map = ["sg_1", "sg_2", "sg_3"]

        decl1 = Declaration(id=1, lean_name="SG1.name", decl_type="def")
        sg1 = StatementGroup(
            id=1,
            primary_decl_id=1,
            primary_declaration=decl1,
            text_hash="h1",
            statement_text="text1",
            source_file="PkgA/F1.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.5,
        )
        decl2 = Declaration(id=2, lean_name="SG2.name", decl_type="def")
        sg2 = StatementGroup(
            id=2,
            primary_decl_id=2,
            primary_declaration=decl2,
            text_hash="h2",
            statement_text="text2",
            source_file="PkgA/F2.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.4,
        )
        decl3 = Declaration(id=3, lean_name="SG3.name", decl_type="def")
        sg3 = StatementGroup(
            id=3,
            primary_decl_id=3,
            primary_declaration=decl3,
            text_hash="h3",
            statement_text="text3",
            source_file="PkgA/F3.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.3,
        )

        db_session.add_all([decl1, sg1, decl2, sg2, decl3, sg3])
        db_session.commit()

        with caplog.at_level(logging.INFO):
            results = search_module.perform_search(
                session=db_session,
                query_string="find good ones",
                model=mock_search_dependencies["model"],
                faiss_index=mock_faiss_index,
                text_chunk_id_map=text_chunk_id_map,
                faiss_k=3,
                pagerank_weight=0.5,
                text_relevance_weight=0.5,
                semantic_similarity_threshold=0.5,
                log_searches=True,
            )

        assert len(results) == 2
        result_ids = {r[0].id for r in results}
        assert result_ids == {1, 2}
        assert (
            "Post-thresholding: 2 of 3 candidates remaining (threshold: 0.500)."
            in caplog.text
        )
        assert any(
            call.kwargs["status"] == "SUCCESS"
            for call in mock_search_dependencies["log_event"].call_args_list
        )

    def test_package_filtering(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests that package filtering correctly narrows down results.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
            caplog: Pytest fixture to capture log output.
        """
        mock_faiss_index = mock_search_dependencies["faiss_index"]
        mock_faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
        mock_faiss_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),
            np.array([[0, 1, 2]]),
        )
        text_chunk_id_map = ["sg_1", "sg_2", "sg_3"]

        decl1 = Declaration(id=1, lean_name="Mathlib.SG1.name", decl_type="def")
        sg1 = StatementGroup(
            id=1,
            primary_decl_id=1,
            primary_declaration=decl1,
            text_hash="h1",
            statement_text="text1",
            source_file="Mathlib/CategoryTheory/SG1.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.9,
        )
        decl2 = Declaration(id=2, lean_name="Std.SG2.name", decl_type="def")
        sg2 = StatementGroup(
            id=2,
            primary_decl_id=2,
            primary_declaration=decl2,
            text_hash="h2",
            statement_text="text2",
            source_file="Std/Data/SG2.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.8,
        )
        decl3 = Declaration(id=3, lean_name="MyProj.SG3.name", decl_type="def")
        sg3 = StatementGroup(
            id=3,
            primary_decl_id=3,
            primary_declaration=decl3,
            text_hash="h3",
            statement_text="text3",
            source_file="MyProj/Utils/SG3.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.7,
        )

        db_session.add_all([decl1, sg1, decl2, sg2, decl3, sg3])
        db_session.commit()

        with caplog.at_level(logging.INFO):
            results = search_module.perform_search(
                session=db_session,
                query_string="test query",
                model=mock_search_dependencies["model"],
                faiss_index=mock_faiss_index,
                text_chunk_id_map=text_chunk_id_map,
                faiss_k=3,
                pagerank_weight=0.1,
                text_relevance_weight=0.1,
                log_searches=True,
                selected_packages=["Mathlib"],
            )
        assert len(results) == 1
        assert results[0][0].id == 1
        assert "Filtering search by packages: ['Mathlib']" in caplog.text
        assert any(
            call.kwargs["status"] == "SUCCESS"
            for call in mock_search_dependencies["log_event"].call_args_list
        )

    def test_full_scoring_and_ranking(
        self,
        db_session: SQLAlchemySession,
        mock_search_dependencies: Dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests the combination of similarity and PageRank for ranking.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
            caplog: Pytest fixture to capture log output.
        """
        mock_faiss_index = mock_search_dependencies["faiss_index"]
        mock_faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
        mock_faiss_index.search.return_value = (
            np.array([[0.8, 0.7, 0.6]]),  # Raw similarities
            np.array([[0, 1, 2]]),  # FAISS indices
        )
        text_chunk_id_map = ["sg_1", "sg_2", "sg_3"]

        decl1 = Declaration(id=1, lean_name="Target1.def", decl_type="def")
        sg1 = StatementGroup(
            id=1,
            primary_decl_id=1,
            primary_declaration=decl1,
            text_hash="h1",
            statement_text="text1",
            source_file="Mathlib/T1.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.1,  # Low PR
        )
        decl2 = Declaration(id=2, lean_name="Something.Else.Target2", decl_type="def")
        sg2 = StatementGroup(
            id=2,
            primary_decl_id=2,
            primary_declaration=decl2,
            text_hash="h2",
            statement_text="text2",
            source_file="Mathlib/T2.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.8,  # High PR
        )
        decl3 = Declaration(id=3, lean_name="Generic.Name", decl_type="def")
        sg3 = StatementGroup(
            id=3,
            primary_decl_id=3,
            primary_declaration=decl3,
            text_hash="h3",
            statement_text="text3",
            source_file="Mathlib/T3.lean",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            scaled_pagerank_score=0.5,
        )

        db_session.add_all([decl1, sg1, decl2, sg2, decl3, sg3])
        db_session.commit()

        # Weights: text relevance (sem_sim) = 0.6, pagerank = 0.3
        results = search_module.perform_search(
            session=db_session,
            query_string="query Target1",
            model=mock_search_dependencies["model"],
            faiss_index=mock_faiss_index,
            text_chunk_id_map=text_chunk_id_map,
            faiss_k=3,
            pagerank_weight=0.3,
            text_relevance_weight=0.6,
            log_searches=True,
            semantic_similarity_threshold=0.0,
        )

        # Raw similarities from FAISS: [0.8, 0.7, 0.6] -> Normalized: [1.0, 0.5, 0.0]
        # PageRank: SG1=0.1, SG2=0.8, SG3=0.5

        # Expected scores for SG1 (raw_sim=0.8, norm_sim=1.0, pr=0.1)
        # Final = (0.6 * 1.0) + (0.3 * 0.1) = 0.6 + 0.03 = 0.63

        # Expected scores for SG2 (raw_sim=0.7, norm_sim=0.5, pr=0.8)
        # Final = (0.6 * 0.5) + (0.3 * 0.8) = 0.3 + 0.24 = 0.54

        # Expected scores for SG3 (raw_sim=0.6, norm_sim=0.0, pr=0.5)
        # Final = (0.6 * 0.0) + (0.3 * 0.5) = 0.0 + 0.15 = 0.15

        assert len(results) == 3

        # Check order and approximate scores
        assert results[0][0].id == 1
        assert results[0][1]["final_score"] == pytest.approx(0.63)
        assert results[0][1]["norm_similarity"] == pytest.approx(1.0)
        assert results[0][1]["scaled_pagerank"] == pytest.approx(0.1)
        assert "raw_name_match_score" not in results[0][1]

        assert results[1][0].id == 2
        assert results[1][1]["final_score"] == pytest.approx(0.54)
        assert results[1][1]["norm_similarity"] == pytest.approx(0.5)
        assert results[1][1]["scaled_pagerank"] == pytest.approx(0.8)
        assert "raw_name_match_score" not in results[1][1]

        assert results[2][0].id == 3
        assert results[2][1]["final_score"] == pytest.approx(0.15)
        assert results[2][1]["norm_similarity"] == pytest.approx(0.0)
        assert results[2][1]["scaled_pagerank"] == pytest.approx(0.5)
        assert "raw_name_match_score" not in results[2][1]

        assert any(
            call.kwargs["status"] == "SUCCESS"
            for call in mock_search_dependencies["log_event"].call_args_list
        )


# --- Tests for CLI Aspects (parse_arguments and main) ---
class TestSearchScriptCLI:
    """Tests for the CLI aspects of search.py (argument parsing and main execution)."""

    def test_parse_arguments_query_only(self, monkeypatch: pytest.MonkeyPatch):
        """Tests argument parsing with only a query provided.

        Args:
            monkeypatch: Pytest fixture for modifying object attributes.
        """
        monkeypatch.setattr(sys, "argv", ["search.py", "my test query"])
        args = search_module.parse_arguments()
        assert args.query == "my test query"
        assert args.limit is None
        assert args.packages is None

    def test_parse_arguments_all_options(self, monkeypatch: pytest.MonkeyPatch):
        """Tests argument parsing with all options specified.

        Args:
            monkeypatch: Pytest fixture for modifying object attributes.
        """
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "search.py",
                "another query",
                "--limit",
                "15",
                "--packages",
                "Mathlib",
                "Std",
            ],
        )
        args = search_module.parse_arguments()
        assert args.query == "another query"
        assert args.limit == 15
        assert args.packages == ["Mathlib", "Std"]

    @patch.object(search_module, "print_results")
    @patch.object(search_module, "perform_search")
    @patch.object(search_module, "sessionmaker")
    @patch.object(search_module, "create_engine")
    @patch.object(search_module, "load_faiss_assets")
    @patch.object(search_module, "load_embedding_model")
    @patch.object(search_module, "parse_arguments")
    def test_main_successful_run(
        self,
        mock_parse_args: MagicMock,
        mock_load_model: MagicMock,
        mock_load_faiss: MagicMock,
        mock_create_engine: MagicMock,
        mock_sessionmaker: MagicMock,
        mock_perform_search: MagicMock,
        mock_print_results: MagicMock,
        isolated_data_paths: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests the main execution flow for a successful search.

        Args:
            mock_parse_args: Mock for `parse_arguments`.
            mock_load_model: Mock for `load_embedding_model`.
            mock_load_faiss: Mock for `load_faiss_assets`.
            mock_create_engine: Mock for `create_engine`.
            mock_sessionmaker: Mock for `sessionmaker`.
            mock_perform_search: Mock for `perform_search`.
            mock_print_results: Mock for `print_results`.
            isolated_data_paths: Fixture to isolate default paths.
            caplog: Pytest fixture to capture log output.
        """
        mock_args = MagicMock(spec=argparse.Namespace)
        mock_args.query = "test query"
        mock_args.limit = 5
        mock_args.packages = ["TestPkg"]
        mock_parse_args.return_value = mock_args

        mock_s_transformer = MagicMock(spec=SentenceTransformer)
        mock_load_model.return_value = mock_s_transformer

        mock_faiss_idx = MagicMock(spec=faiss.Index)
        mock_id_map = ["id1"]
        mock_load_faiss.return_value = (mock_faiss_idx, mock_id_map)

        mock_engine_instance = MagicMock(spec=SQLAlchemyEngine)
        mock_create_engine.return_value = mock_engine_instance

        mock_session_instance = MagicMock(spec=SQLAlchemySession)
        mock_session_factory_instance = MagicMock()
        mock_session_factory_instance.return_value.__enter__.return_value = (
            mock_session_instance
        )
        mock_sessionmaker.return_value = mock_session_factory_instance

        ranked_results_fixture = [
            (MagicMock(spec=StatementGroup), {"final_score": 0.9})
        ]
        mock_perform_search.return_value = ranked_results_fixture

        defaults.DEFAULT_DB_PATH.touch()

        with caplog.at_level(logging.INFO):
            search_module.main()

        mock_parse_args.assert_called_once()
        mock_load_model.assert_called_once_with(defaults.DEFAULT_EMBEDDING_MODEL_NAME)
        mock_load_faiss.assert_called_once_with(
            str(defaults.DEFAULT_FAISS_INDEX_PATH.resolve()),
            str(defaults.DEFAULT_FAISS_MAP_PATH.resolve()),
        )
        mock_create_engine.assert_called_once_with(defaults.DEFAULT_DB_URL, echo=False)
        mock_sessionmaker.assert_called_once()
        mock_perform_search.assert_called_once_with(
            session=mock_session_instance,
            query_string=mock_args.query,
            model=mock_s_transformer,
            faiss_index=mock_faiss_idx,
            text_chunk_id_map=mock_id_map,
            faiss_k=defaults.DEFAULT_FAISS_K,
            pagerank_weight=defaults.DEFAULT_PAGERANK_WEIGHT,
            text_relevance_weight=defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT,
            log_searches=True,
            selected_packages=mock_args.packages,
            semantic_similarity_threshold=defaults.DEFAULT_SEM_SIM_THRESHOLD,
            faiss_nprobe=defaults.DEFAULT_FAISS_NPROBE,
        )
        mock_print_results.assert_called_once_with(
            ranked_results_fixture[: mock_args.limit]
        )

        assert "Starting Search (Direct Script Execution)" in caplog.text
        assert (search_module._USER_LOGS_BASE_DIR).exists()

    @patch.object(search_module, "sys")
    @patch.object(search_module, "load_embedding_model")
    @patch.object(search_module, "parse_arguments")
    def test_main_model_load_failure_exits(
        self,
        mock_parse_args: MagicMock,
        mock_load_model: MagicMock,
        mock_sys: MagicMock,
        isolated_data_paths: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests that main exits if embedding model loading fails.

        Args:
            mock_parse_args: Mock for `parse_arguments`.
            mock_load_model: Mock for `load_embedding_model`.
            mock_sys: Mock for the `sys` module.
            isolated_data_paths: Fixture to isolate default paths.
            caplog: Pytest fixture to capture log output.
        """
        mock_parse_args.return_value = MagicMock(query="q", limit=None, packages=None)
        mock_load_model.return_value = None
        mock_sys.exit.side_effect = SystemExit

        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
            search_module.main()

        assert excinfo.type is SystemExit
        mock_sys.exit.assert_called_once_with(1)
        assert "Sentence transformer model loading failed." in caplog.text
