# tests/local/test_search.py

"""Tests for core search logic and helper functions in `lean_explore.search`.

This module includes tests for name matching, performance logging,
asset loading (embedding models, FAISS indices), the main
`perform_search` algorithm, and the script's CLI execution.
"""

import pytest
import pathlib
import os
import json
import datetime
import logging
import argparse
import sys # For mocking sys.argv and sys.exit
from unittest.mock import MagicMock, patch, call
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Any
from sqlalchemy.engine import Engine as SQLAlchemyEngine

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from filelock import Timeout, FileLock as FLock # Alias FLock
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from lean_explore.local import search as search_module
from lean_explore import defaults
from lean_explore.shared.models.db import StatementGroup, Declaration

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from sqlalchemy.engine import Engine as SQLAlchemyEngine # For specing mock engine

# --- Tests for Helper Functions (Assumed to be passing from previous steps) ---
class TestCalculateNameMatchScore:
    """Tests for the `calculate_name_match_score` function."""

    def test_empty_inputs(self):
        """Ensures score is 0.0 if either query or name is empty."""
        assert search_module.calculate_name_match_score("", "Nat.add") == 0.0
        assert search_module.calculate_name_match_score("Nat.add", "") == 0.0
        assert search_module.calculate_name_match_score("", "") == 0.0

    def test_exact_match(self):
        """Ensures exact matches (case-insensitive) score approximately 1.0 after mapping."""
        assert search_module.calculate_name_match_score("Nat.add", "Nat.add") == pytest.approx(1.0)
        assert search_module.calculate_name_match_score("nat.add", "Nat.add") == pytest.approx(1.0)
        assert search_module.calculate_name_match_score("Nat.Add", "nat.ADD") == pytest.approx(1.0)

    def test_no_match(self):
        """Ensures completely different strings score 0.0."""
        assert search_module.calculate_name_match_score("Nat.mul", "Nat.add") == 0.0
        assert search_module.calculate_name_match_score("foo", "bar") == 0.0

    @pytest.mark.parametrize(
        "query, name, raw_score_ratio, expected_mapped_score",
        [
            ("Sim1", "Sim1", 0.90, 0.0),
            ("Sim2", "Sim2", 0.95, 0.5),
            ("Sim3", "Sim3", 0.99, 0.9),
            ("Sim4", "Sim4", 1.0, 1.0),
            ("Sim5", "Sim5", 0.89, 0.0),
            ("Sim6", "Sim6", 0.50, 0.0),
        ]
    )
    def test_score_mapping_logic(
        self, query: str, name: str, raw_score_ratio: float, expected_mapped_score: float, mocker: "MockerFixture"
    ):
        """Tests the thresholding and linear scaling logic for name matching.

        Args:
            query: The query string for the test.
            name: The Lean declaration name for the test.
            raw_score_ratio: The simulated raw fuzz.WRatio / 100.0.
            expected_mapped_score: The expected mapped score after processing.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(search_module.fuzz, "WRatio", return_value=raw_score_ratio * 100)
        assert search_module.calculate_name_match_score(query, name) == pytest.approx(expected_mapped_score, abs=1e-2)


class TestLogSearchEventToJson:
    """Tests for the `log_search_event_to_json` function."""

    @pytest.fixture
    def mock_log_paths(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, isolated_data_paths: pathlib.Path):
        """Mocks performance log paths to use a temporary directory.

        Leverages `isolated_data_paths` to ensure base paths are already mocked
        and then further ensures the log-specific paths within `search_module`
        point to the temporary directory.

        Args:
            tmp_path: Pytest's built-in temporary directory fixture.
            monkeypatch: Pytest fixture for modifying object attributes.
            isolated_data_paths: Fixture that mocks `defaults.LEAN_EXPLORE_USER_DATA_DIR`.
                                 This ensures `search_module._USER_LOGS_BASE_DIR`
                                 is derived from a temporary path.

        Returns:
            Tuple[pathlib.Path, pathlib.Path]: Paths to the temporary log file
                                               and lock file.
        """
        temp_log_dir = isolated_data_paths.parent / "logs_for_test" # Ensure a unique subdir under tmp_path
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

    def test_successful_log_write(self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path]):
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
        log_timestamp = datetime.datetime.fromisoformat(log_content["timestamp"].replace("Z", "+00:00"))
        timestamp_after = datetime.datetime.now(datetime.timezone.utc)
        assert timestamp_before <= log_timestamp <= timestamp_after

    def test_log_with_error_type(self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path]):
        """Ensures `error_type` is included in the log when provided.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
        """
        log_file, _ = mock_log_paths
        search_module.log_search_event_to_json("ERROR", 50.0, 0, error_type="TestError")
        log_content = json.loads(log_file.read_text(encoding="utf-8"))
        assert log_content["error_type"] == "TestError"

    def test_log_directory_creation_failure(
        self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
    ):
        """Tests behavior when log directory creation fails.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_print = mocker.patch("builtins.print")
        mocker.patch.object(search_module.os, "makedirs", side_effect=OSError("Cannot create dir"))
        with caplog.at_level(logging.ERROR):
            search_module.log_search_event_to_json("SUCCESS_DIR_FAIL", 10.0, 1)
        assert "Performance logging error: Could not create log directory" in caplog.text
        assert "Cannot create dir" in caplog.text
        mock_print.assert_called_once()
        assert "FALLBACK_PERF_LOG (DIR_ERROR)" in mock_print.call_args[0][0]

    def test_log_filelock_timeout(
        self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module, "FileLock", return_value=mock_file_lock_instance)
        with caplog.at_level(logging.WARNING):
            search_module.log_search_event_to_json("SUCCESS_LOCK_FAIL", 20.0, 2)
        assert not log_file.exists()
        assert "Performance logging error: Timeout acquiring lock" in caplog.text
        mock_print.assert_called_once()
        assert "FALLBACK_PERF_LOG (LOCK_TIMEOUT)" in mock_print.call_args[0][0]

    def test_log_file_write_failure(
        self, mock_log_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
    ):
        """Tests behavior when writing to the log file fails.

        Args:
            mock_log_paths: Fixture providing temporary log and lock file paths.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_print = mocker.patch("builtins.print")
        mocker.patch("builtins.open", side_effect=IOError("Cannot write to file"))
        with caplog.at_level(logging.ERROR):
            search_module.log_search_event_to_json("SUCCESS_WRITE_FAIL", 30.0, 3)
        assert "Performance logging error: Failed to write to" in caplog.text
        assert "Cannot write to file" in caplog.text
        mock_print.assert_called_once()
        assert "FALLBACK_PERF_LOG (WRITE_ERROR)" in mock_print.call_args[0][0]


class TestLoadEmbeddingModel:
    """Tests for the `load_embedding_model` function."""

    def test_successful_load(self, mocker: "MockerFixture", caplog: pytest.LogCaptureFixture):
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

    def test_load_failure(self, mocker: "MockerFixture", caplog: pytest.LogCaptureFixture):
        """Tests behavior when model loading raises an exception.

        Args:
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mocker.patch.object(
            search_module, "SentenceTransformer", side_effect=Exception("Model loading failed")
        )
        model_name = "failing-model"
        with caplog.at_level(logging.ERROR):
            model = search_module.load_embedding_model(model_name)
        assert model is None
        assert f"Failed to load sentence transformer model '{model_name}'" in caplog.text
        assert "Model loading failed" in caplog.text


class TestLoadFaissAssets:
    """Tests for the `load_faiss_assets` function."""

    @pytest.fixture
    def faiss_asset_paths(self, tmp_path: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
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
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module.faiss, "read_index", return_value=mock_faiss_index)
        with caplog.at_level(logging.INFO):
            index, id_map = search_module.load_faiss_assets(str(index_file), str(map_file))
        assert index is mock_faiss_index
        assert id_map == map_data
        search_module.faiss.read_index.assert_called_once_with(str(index_file))
        assert f"Loaded FAISS index with {mock_faiss_index.ntotal} vectors" in caplog.text
        assert f"Loaded ID map with {len(map_data)} entries" in caplog.text
        assert "Mismatch: FAISS index size" not in caplog.text

    def test_index_file_not_found(
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], caplog: pytest.LogCaptureFixture
    ):
        """Tests behavior when the FAISS index file is not found.

        Args:
            faiss_asset_paths: Tuple of paths to temp index and map files.
            caplog: Pytest fixture to capture log output.
        """
        _, map_file = faiss_asset_paths
        map_file.write_text(json.dumps(["sg_1"]), encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets("non_existent.index", str(map_file))
        assert index is None
        assert id_map is None
        assert "FAISS index file not found" in caplog.text
        assert "non_existent.index" in caplog.text

    def test_map_file_not_found(
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module.faiss, "read_index", return_value=mock_faiss_index)
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(str(index_file), "non_existent_map.json")
        assert index is None
        assert id_map is None
        assert "FAISS ID map file not found" in caplog.text
        assert "non_existent_map.json" in caplog.text

    def test_faiss_read_index_error(
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module.faiss, "read_index", side_effect=RuntimeError("FAISS load error"))
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(str(index_file), str(map_file))
        assert index is None
        assert id_map is None
        assert f"Failed to load FAISS index from {index_file}" in caplog.text
        assert "FAISS load error" in caplog.text

    def test_json_load_error_for_map(
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module.faiss, "read_index", return_value=mock_faiss_index)
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(str(index_file), str(map_file))
        assert index is mock_faiss_index
        assert id_map is None
        assert f"Failed to load or parse ID map file {map_file}" in caplog.text
        assert "JSONDecodeError" in caplog.text or "Expecting value" in caplog.text

    def test_map_file_not_a_list(
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module.faiss, "read_index", return_value=mock_faiss_index)
        with caplog.at_level(logging.ERROR):
            index, id_map = search_module.load_faiss_assets(str(index_file), str(map_file))
        assert index is mock_faiss_index
        assert id_map is None
        assert f"ID map file ({map_file}) does not contain a valid JSON list." in caplog.text

    def test_size_mismatch_warning(
        self, faiss_asset_paths: Tuple[pathlib.Path, pathlib.Path], mocker: "MockerFixture", caplog: pytest.LogCaptureFixture
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
        mocker.patch.object(search_module.faiss, "read_index", return_value=mock_faiss_index)
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
        log_search_event_to_json, and calculate_name_match_score.

        Args:
            mocker: Pytest-mock's mocker fixture.

        Returns:
            Dict[str, MagicMock]: A dictionary of named mock objects.
        """
        mock_model = MagicMock(spec=SentenceTransformer)
        # Default behavior for model.encode
        mock_model.encode.return_value = [np.array([0.1, 0.2], dtype=np.float32)]

        mock_faiss_index = MagicMock(spec=faiss.Index)
        mock_faiss_index.metric_type = faiss.METRIC_L2
        mock_faiss_index.nprobe = 1 # Default, can be asserted or changed
        # Default behavior for faiss_index.search (empty results)
        mock_faiss_index.search.return_value = (np.array([[]]), np.array([[]]))

        # Patch helpers within the search_module
        mock_log_event = mocker.patch.object(search_module, "log_search_event_to_json")
        mock_calc_name_score = mocker.patch.object(search_module, "calculate_name_match_score", return_value=0.0)

        return {
            "model": mock_model,
            "faiss_index": mock_faiss_index,
            "log_event": mock_log_event,
            "calc_name_score": mock_calc_name_score,
        }

    def test_empty_query_string(self, db_session: SQLAlchemySession, mock_search_dependencies: Dict[str, MagicMock]):
        """Verifies behavior with an empty query string.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        results = search_module.perform_search(
            session=db_session,
            query_string="  ", # Empty after strip
            model=mock_search_dependencies["model"],
            faiss_index=mock_search_dependencies["faiss_index"],
            text_chunk_id_map=[],
            faiss_k=10, pagerank_weight=0.1, text_relevance_weight=0.1, name_match_weight=0.1
        )
        assert results == []
        mock_search_dependencies["log_event"].assert_called_once()
        assert mock_search_dependencies["log_event"].call_args.kwargs['status'] == "EMPTY_QUERY_SUBMITTED"

    def test_query_embedding_failure(self, db_session: SQLAlchemySession, mock_search_dependencies: Dict[str, MagicMock]):
        """Tests exception handling when model embedding fails.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        mock_search_dependencies["model"].encode.side_effect = Exception("Embedding boom")
        with pytest.raises(Exception, match="Query embedding failed: Embedding boom"):
            search_module.perform_search(
                session=db_session, query_string="test", model=mock_search_dependencies["model"],
                faiss_index=mock_search_dependencies["faiss_index"], text_chunk_id_map=[],
                faiss_k=10, pagerank_weight=0.1, text_relevance_weight=0.1, name_match_weight=0.1
            )
        mock_search_dependencies["log_event"].assert_called_once()
        assert mock_search_dependencies["log_event"].call_args.kwargs['status'] == "EMBEDDING_ERROR"

    def test_faiss_search_failure(self, db_session: SQLAlchemySession, mock_search_dependencies: Dict[str, MagicMock]):
        """Tests exception handling when FAISS search fails.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        mock_search_dependencies["faiss_index"].search.side_effect = Exception("FAISS boom")
        with pytest.raises(Exception, match="FAISS search failed: FAISS boom"):
            search_module.perform_search(
                session=db_session, query_string="test", model=mock_search_dependencies["model"],
                faiss_index=mock_search_dependencies["faiss_index"], text_chunk_id_map=[],
                faiss_k=10, pagerank_weight=0.1, text_relevance_weight=0.1, name_match_weight=0.1
            )
        mock_search_dependencies["log_event"].assert_called_once()
        assert mock_search_dependencies["log_event"].call_args.kwargs['status'] == "FAISS_SEARCH_ERROR"

    def test_no_faiss_candidates_after_parsing(self, db_session: SQLAlchemySession, mock_search_dependencies: Dict[str, MagicMock]):
        """Tests behavior when FAISS returns results that don't parse to valid SG IDs.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        mock_search_dependencies["faiss_index"].search.return_value = (
            np.array([[0.5]]), np.array([[0]]) # One result
        )
        # text_chunk_id_map[0] is "malformed" -> no valid SG ID
        mock_text_chunk_id_map = ["malformed_id"]

        results = search_module.perform_search(
            session=db_session, query_string="test", model=mock_search_dependencies["model"],
            faiss_index=mock_search_dependencies["faiss_index"], text_chunk_id_map=mock_text_chunk_id_map,
            faiss_k=10, pagerank_weight=0.1, text_relevance_weight=0.1, name_match_weight=0.1
        )
        assert results == []
        # log_event will be called twice: once for "search_processed" and once inside for status
        assert any(call.kwargs['status'] == "NO_FAISS_CANDIDATES" for call in mock_search_dependencies["log_event"].call_args_list)


    def test_semantic_thresholding(self, db_session: SQLAlchemySession, mock_search_dependencies: Dict[str, MagicMock]):
        """Tests that semantic similarity threshold correctly filters candidates.

        Args:
            db_session: SQLAlchemy session fixture.
            mock_search_dependencies: Fixture providing common mocks.
        """
        mock_faiss_index = mock_search_dependencies["faiss_index"]
        mock_faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT # Cosine similarity
        # Distances are similarities for inner product
        mock_faiss_index.search.return_value = (
            np.array([[0.8, 0.6, 0.4]]), # Similarities
            np.array([[0, 1, 2]])        # Indices
        )
        text_chunk_id_map = ["sg_1", "sg_2", "sg_3"] # Link to DB IDs

        # DB entries (only sg_1 and sg_2 should be processed after thresholding)
        decl1 = Declaration(id=1, lean_name="SG1.name", decl_type="def")
        sg1 = StatementGroup(id=1, primary_decl_id=1, primary_declaration=decl1, text_hash="h1", statement_text="text1", source_file="PkgA/F1.lean", range_start_line=1, range_start_col=1, range_end_line=1, range_end_col=1, scaled_pagerank_score=0.5)
        decl2 = Declaration(id=2, lean_name="SG2.name", decl_type="def")
        sg2 = StatementGroup(id=2, primary_decl_id=2, primary_declaration=decl2, text_hash="h2", statement_text="text2", source_file="PkgA/F2.lean", range_start_line=1, range_start_col=1, range_end_line=1, range_end_col=1, scaled_pagerank_score=0.4)
        # SG3 should be filtered by threshold
        decl3 = Declaration(id=3, lean_name="SG3.name", decl_type="def")
        sg3 = StatementGroup(id=3, primary_decl_id=3, primary_declaration=decl3, text_hash="h3", statement_text="text3", source_file="PkgA/F3.lean", range_start_line=1, range_start_col=1, range_end_line=1, range_end_col=1, scaled_pagerank_score=0.3)

        db_session.add_all([decl1, sg1, decl2, sg2, decl3, sg3])
        db_session.commit()

        results = search_module.perform_search(
            session=db_session, query_string="find good ones", model=mock_search_dependencies["model"],
            faiss_index=mock_faiss_index, text_chunk_id_map=text_chunk_id_map,
            faiss_k=3, pagerank_weight=0.5, text_relevance_weight=0.5, name_match_weight=0.0,
            semantic_similarity_threshold=0.5 # sg_3 (sim 0.4) should be excluded
        )

        assert len(results) == 2
        result_ids = {r[0].id for r in results}
        assert result_ids == {1, 2}
        # Check that SG3 was not processed deeply (e.g. calculate_name_match_score not called for it)
        # This is harder to check without inspecting internal state or more complex mocking.
        # For now, the length of results is a good indicator.

    # More tests for perform_search: package_filtering, scoring logic, different FAISS metric_types, etc.
    # These would involve more complex data setup in db_session and mock configurations.


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
        assert args.limit is None # Default from argparse
        assert args.packages is None # Default from argparse

    def test_parse_arguments_all_options(self, monkeypatch: pytest.MonkeyPatch):
        """Tests argument parsing with all options specified.

        Args:
            monkeypatch: Pytest fixture for modifying object attributes.
        """
        monkeypatch.setattr(sys, "argv", [
            "search.py", "another query",
            "--limit", "15",
            "--packages", "Mathlib", "Std"
        ])
        args = search_module.parse_arguments()
        assert args.query == "another query"
        assert args.limit == 15
        assert args.packages == ["Mathlib", "Std"]

    @patch.object(search_module, "print_results")
    @patch.object(search_module, "perform_search")
    @patch.object(search_module, "sessionmaker") # Used within main for SessionLocal
    @patch.object(search_module, "create_engine")
    @patch.object(search_module, "load_faiss_assets")
    @patch.object(search_module, "load_embedding_model")
    @patch.object(search_module, "parse_arguments")
    def test_main_successful_run(
        self, mock_parse_args: MagicMock, mock_load_model: MagicMock, mock_load_faiss: MagicMock,
        mock_create_engine: MagicMock, mock_sessionmaker: MagicMock,
        mock_perform_search: MagicMock, mock_print_results: MagicMock,
        isolated_data_paths: pathlib.Path, # To ensure defaults point to temp
        caplog: pytest.LogCaptureFixture
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

        mock_engine_instance = MagicMock(spec=SQLAlchemyEngine) # Use a more specific spec if available
        mock_create_engine.return_value = mock_engine_instance

        mock_session_instance = MagicMock(spec=SQLAlchemySession)
        mock_session_factory_instance = MagicMock() # This is SessionLocal
        mock_session_factory_instance.return_value.__enter__.return_value = mock_session_instance # for 'with ... as session:'
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
            str(defaults.DEFAULT_FAISS_INDEX_PATH.resolve()), # main uses resolve
            str(defaults.DEFAULT_FAISS_MAP_PATH.resolve())
        )
        mock_create_engine.assert_called_once_with(defaults.DEFAULT_DB_URL, echo=False)
        mock_sessionmaker.assert_called_once()
        mock_perform_search.assert_called_once()
        mock_print_results.assert_called_once_with(ranked_results_fixture[:mock_args.limit])

        assert "Starting Search (Direct Script Execution)" in caplog.text
        # Ensure USER_LOGS_BASE_DIR was attempted to be created
        assert (search_module._USER_LOGS_BASE_DIR).exists()


    @patch.object(search_module, "sys")
    @patch.object(search_module, "load_embedding_model")
    @patch.object(search_module, "parse_arguments")
    def test_main_model_load_failure_exits(
        self, mock_parse_args: MagicMock, mock_load_model: MagicMock, mock_sys: MagicMock,
        isolated_data_paths: pathlib.Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.ERROR), \
             pytest.raises(SystemExit) as excinfo:
            search_module.main()

        assert excinfo.type == SystemExit
        mock_sys.exit.assert_called_once_with(1)
        assert "Sentence transformer model loading failed." in caplog.text