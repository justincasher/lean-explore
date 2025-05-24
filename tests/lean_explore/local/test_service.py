# tests/local/test_service.py
"""Tests for the lean_explore.local.service.Service class.

This module focuses on testing the initialization and core functionalities
of the Service class, which orchestrates local data asset loading and
provides methods for interacting with the local Lean explore data.
"""

import logging
import pathlib
from typing import TYPE_CHECKING, Dict, List, Tuple
from unittest.mock import MagicMock

import faiss
import pytest
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session as SQLAlchemySession

from lean_explore import defaults
from lean_explore.local.service import Service
from lean_explore.shared.models.api import (
    APISearchResponse,
    APISearchResultItem,
)

# Import your ORM and API models
from lean_explore.shared.models.db import (
    Declaration,
    StatementGroup,
    StatementGroupDependency,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestServiceInitialization:
    """Tests focused on the __init__ method of the Service class."""

    def test_successful_initialization(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
        mocker: "MockerFixture",
    ):
        """Verifies successful instantiation of Service when all assets load.

        This test ensures that when all underlying asset loading functions
        (mocked) succeed and prerequisite files (like the database file)
        are present in their isolated locations, the Service initializes
        correctly and sets its internal attributes as expected.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Fixture mocking embedding model loading.
            mock_load_faiss_assets: Fixture mocking FAISS asset loading.
            mock_sqlalchemy_engine_setup: Fixture mocking SQLAlchemy engine setup.
            mocker: Pytest-mock's mocker fixture.
        """
        defaults.DEFAULT_DB_PATH.touch()

        mock_model_instance = MagicMock(spec=SentenceTransformer)
        mock_faiss_index_instance = MagicMock(spec=faiss.Index)
        mock_id_map_instance: List[str] = ["id1", "id2"]

        mock_load_embedding_model.return_value = mock_model_instance
        mock_load_faiss_assets.return_value = (
            mock_faiss_index_instance,
            mock_id_map_instance,
        )

        service = Service()

        assert service.embedding_model is mock_model_instance
        assert service.faiss_index is mock_faiss_index_instance
        assert service.text_chunk_id_map is mock_id_map_instance
        assert service.engine is mock_sqlalchemy_engine_setup.return_value
        assert service.SessionLocal is not None

        assert service.default_faiss_k == defaults.DEFAULT_FAISS_K
        assert service.default_pagerank_weight == defaults.DEFAULT_PAGERANK_WEIGHT
        assert (
            service.default_text_relevance_weight
            == defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT
        )
        assert service.default_name_match_weight == defaults.DEFAULT_NAME_MATCH_WEIGHT
        assert (
            service.default_semantic_similarity_threshold
            == defaults.DEFAULT_SEM_SIM_THRESHOLD
        )
        assert service.default_results_limit == defaults.DEFAULT_RESULTS_LIMIT
        assert service.default_faiss_nprobe == defaults.DEFAULT_FAISS_NPROBE

        mock_load_embedding_model.assert_called_once_with(
            defaults.DEFAULT_EMBEDDING_MODEL_NAME
        )
        mock_load_faiss_assets.assert_called_once_with(
            str(defaults.DEFAULT_FAISS_INDEX_PATH), str(defaults.DEFAULT_FAISS_MAP_PATH)
        )
        mock_sqlalchemy_engine_setup.assert_called_once_with(defaults.DEFAULT_DB_URL)
        service.engine.connect.assert_called_once()

    def test_init_fails_if_embedding_model_load_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises RuntimeError if embedding model loading fails.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        defaults.DEFAULT_DB_PATH.touch()
        mock_load_embedding_model.return_value = None

        expected_msg = (
            f"Failed to load embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}"
        )
        with pytest.raises(RuntimeError, match=expected_msg):
            Service()

    def test_init_fails_if_faiss_assets_load_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises FileNotFoundError if FAISS assets are None.

        Covers cases where either the FAISS index or the ID map fails to load,
        signified by `mock_load_faiss_assets` returning `None` for one of
        its tuple elements.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        defaults.DEFAULT_DB_PATH.touch()
        base_error_message_regex = (
            r"Failed to load critical FAISS assets \(index or ID map\)"
        )

        mock_load_faiss_assets.return_value = (None, ["id1"])
        with pytest.raises(FileNotFoundError, match=base_error_message_regex):
            Service()

        mock_load_faiss_assets.return_value = (MagicMock(spec=faiss.Index), None)
        with pytest.raises(FileNotFoundError, match=base_error_message_regex):
            Service()

    def test_init_fails_if_db_file_does_not_exist(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises FileNotFoundError if the DB file is missing.

        This test ensures that the `Service` checks for the database file's
        existence at the path specified by `defaults.DEFAULT_DB_PATH` (which
        is an isolated path in this test context) and fails appropriately.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        if defaults.DEFAULT_DB_PATH.exists():
            defaults.DEFAULT_DB_PATH.unlink()

        expected_msg_regex = (
            "Database file not found at the expected location: "
            f"{str(defaults.DEFAULT_DB_PATH).replace('.', r'.')}"
        )
        with pytest.raises(FileNotFoundError, match=expected_msg_regex):
            Service()

    def test_init_fails_if_db_connection_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises RuntimeError if DB connection fails.

        Even if the database file exists, this test simulates a failure
        during the engine connection attempt.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        defaults.DEFAULT_DB_PATH.touch()

        mock_engine_instance = mock_sqlalchemy_engine_setup.return_value
        mock_dbapi_exception = Exception("DBAPI connection error detail")
        mock_engine_instance.connect.side_effect = OperationalError(
            "Test DB connection failed", {"param": 1}, mock_dbapi_exception
        )

        expected_match_str = (
            r"Database initialization failed: \(builtins\.Exception\) DBAPI connection "
            r"error detail\n\[SQL: Test DB connection failed\]\n"
            r"\[parameters: \{'param': 1\}\]\n"
            r"\(Background on this error at: https://sqlalche\.me/e/20/e3q8\)\. "
            r"The database file at .* might be corrupted"
        )
        with pytest.raises(RuntimeError, match=expected_match_str):
            Service()

    def test_init_proceeds_despite_toolchain_dir_OSError_if_exist_ok(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
        caplog: pytest.LogCaptureFixture,
        mocker: "MockerFixture",
    ):
        """Verifies Service init proceeds if toolchain base dir mkdir is handled.

        The `isolated_data_paths` fixture already creates the toolchain base
        directory. The `mkdir(parents=True, exist_ok=True)` call in
        `Service.__init__` should not raise an error. This test confirms
        that no error log regarding this directory creation appears.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
            caplog: Pytest fixture to capture log output.
            mocker: Pytest-mock's mocker fixture.
        """
        defaults.DEFAULT_DB_PATH.touch()
        mock_load_embedding_model.return_value = MagicMock(spec=SentenceTransformer)
        mock_load_faiss_assets.return_value = (MagicMock(spec=faiss.Index), ["id_map"])

        with caplog.at_level(logging.INFO):
            service = Service()
            assert service is not None

        assert (
            "User toolchains base directory ensured: "
            f"{str(defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR)}" in caplog.text
        )
        assert "Could not create user toolchains base directory" not in caplog.text


@pytest.fixture
def initialized_service(
    isolated_data_paths: pathlib.Path,
    mock_load_embedding_model: MagicMock,
    mock_load_faiss_assets: MagicMock,
    mock_sqlalchemy_engine_setup: MagicMock,
    mocker: "MockerFixture",
) -> Service:
    """Provides a successfully initialized Service instance for method testing.

    This fixture ensures that all dependencies for Service.__init__ are
    mocked for success, and dummy files (like the database file) are
    touched to satisfy existence checks.

    Args:
        isolated_data_paths: Fixture that redirects default data paths.
        mock_load_embedding_model: Fixture that mocks embedding model loading.
        mock_load_faiss_assets: Fixture that mocks FAISS asset loading.
        mock_sqlalchemy_engine_setup: Fixture that mocks SQLAlchemy engine setup.
        mocker: Pytest-mock's mocker fixture.

    Returns:
        Service: A successfully initialized Service instance.
    """
    defaults.DEFAULT_DB_PATH.touch()

    mock_model_instance = MagicMock(spec=SentenceTransformer)
    mock_faiss_index_instance = MagicMock(spec=faiss.Index)
    mock_id_map_instance: List[str] = ["id1", "id2"]

    mock_load_embedding_model.return_value = mock_model_instance
    mock_load_faiss_assets.return_value = (
        mock_faiss_index_instance,
        mock_id_map_instance,
    )
    return Service()


class TestServiceMethods:
    """Tests for the data access and utility methods of the Service class."""

    def test_serialize_sg_to_api_item(self, initialized_service: Service):
        """Tests StatementGroup ORM object to APISearchResultItem conversion.

        Args:
            initialized_service: A successfully initialized Service instance.
        """
        mock_sg_orm = MagicMock(spec=StatementGroup)
        mock_sg_orm.id = 1
        mock_sg_orm.source_file = "Test.lean"
        mock_sg_orm.range_start_line = 10
        mock_sg_orm.display_statement_text = "display text"
        mock_sg_orm.statement_text = "full statement text"
        mock_sg_orm.docstring = "This is a docstring."
        mock_sg_orm.informal_description = "An informal description."

        mock_primary_decl = MagicMock(spec=Declaration)
        mock_primary_decl.lean_name = "Nat.add"
        mock_sg_orm.primary_declaration = mock_primary_decl

        api_item = initialized_service._serialize_sg_to_api_item(mock_sg_orm)

        assert isinstance(api_item, APISearchResultItem)
        assert api_item.id == mock_sg_orm.id
        assert api_item.primary_declaration.lean_name == "Nat.add"
        assert api_item.source_file == "Test.lean"
        assert api_item.range_start_line == 10
        assert api_item.display_statement_text == "display text"
        assert api_item.statement_text == "full statement text"
        assert api_item.docstring == "This is a docstring."
        assert api_item.informal_description == "An informal description."

    def test_serialize_sg_to_api_item_no_primary_decl(
        self, initialized_service: Service
    ):
        """Tests serialization when the StatementGroup has no primary_declaration.

        Args:
            initialized_service: A successfully initialized Service instance.
        """
        mock_sg_orm = MagicMock(spec=StatementGroup)
        mock_sg_orm.id = 2
        mock_sg_orm.primary_declaration = None  # Simulate no primary declaration
        # Set other required fields for APISearchResultItem
        mock_sg_orm.source_file = "Another.lean"
        mock_sg_orm.range_start_line = 1
        mock_sg_orm.statement_text = "some text"
        mock_sg_orm.display_statement_text = None
        mock_sg_orm.docstring = None
        mock_sg_orm.informal_description = None

        api_item = initialized_service._serialize_sg_to_api_item(mock_sg_orm)
        assert api_item.primary_declaration.lean_name is None

    def test_get_by_id_found(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests retrieving a StatementGroup by ID when it exists.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session with schema created.
            mocker: Pytest-mock's mocker fixture.
        """
        # Populate the in-memory DB
        decl = Declaration(id=1, lean_name="Nat.zero", decl_type="axiom")
        sg = StatementGroup(
            id=101,
            text_hash="hash1",
            statement_text="def zero := 0",
            source_file="Init.Nat.Zero.lean",
            range_start_line=5,
            range_start_col=0,
            range_end_line=5,
            range_end_col=15,
            primary_decl_id=1,
            primary_declaration=decl,
        )
        db_session.add_all([decl, sg])
        db_session.commit()

        # Mock the service's SessionLocal to return our test db_session
        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )

        result = initialized_service.get_by_id(101)

        assert result is not None
        assert result.id == 101
        assert result.primary_declaration.lean_name == "Nat.zero"
        assert result.statement_text == "def zero := 0"

    def test_get_by_id_not_found(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests retrieving a StatementGroup by ID when it does not exist.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )
        result = initialized_service.get_by_id(999)  # Non-existent ID
        assert result is None

    def test_get_by_id_sqlalchemy_error(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests get_by_id when an SQLAlchemyError occurs.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session (unused directly here
                as SessionLocal is mocked).
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_session = mocker.MagicMock(spec=SQLAlchemySession)
        # Configure the chain of calls on the mock_session to raise the error
        # at .first()
        (
            mock_session.query.return_value.options.return_value.filter.return_value.first
        ).side_effect = SQLAlchemyError("DB query failed during get_by_id")

        # Mock the SessionLocal context manager on the service instance
        mock_session_local_cm = mocker.MagicMock()
        mock_session_local_cm.__enter__.return_value = mock_session
        mock_session_local_cm.__exit__.return_value = None

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=mock_session_local_cm
        )

        with caplog.at_level(logging.ERROR):
            result = initialized_service.get_by_id(101)

        assert result is None
        assert "Database error in get_by_id for group_id 101" in caplog.text
        assert "DB query failed during get_by_id" in caplog.text

    def test_get_dependencies_found(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests retrieving dependencies for a StatementGroup.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session.
            mocker: Pytest-mock's mocker fixture.
        """
        # Populate DB
        decl1 = Declaration(id=1, lean_name="Nat.succ", decl_type="def")
        decl2 = Declaration(id=2, lean_name="Nat.zero", decl_type="axiom")
        sg_source = StatementGroup(
            id=201,
            text_hash="hash_source",
            statement_text="def succ (n: Nat) := Nat.add n 1",
            primary_decl_id=1,
            primary_declaration=decl1,
            source_file="f1",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
        )
        sg_target = StatementGroup(
            id=202,
            text_hash="hash_target",
            statement_text="axiom zero : Nat",
            primary_decl_id=2,
            primary_declaration=decl2,
            source_file="f2",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
        )
        dependency = StatementGroupDependency(
            source_statement_group_id=201, target_statement_group_id=202
        )
        db_session.add_all([decl1, decl2, sg_source, sg_target, dependency])
        db_session.commit()

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )
        result = initialized_service.get_dependencies(201)

        assert result is not None
        assert result.source_group_id == 201
        assert result.count == 1
        assert len(result.citations) == 1
        assert result.citations[0].id == 202
        assert result.citations[0].primary_declaration.lean_name == "Nat.zero"

    def test_get_dependencies_source_not_found(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests get_dependencies when the source StatementGroup does not exist.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session.
            mocker: Pytest-mock's mocker fixture.
        """
        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )
        result = initialized_service.get_dependencies(999)  # Non-existent ID
        assert result is None

    def test_get_dependencies_no_dependencies(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests get_dependencies when the source group exists but has no dependencies.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session.
            mocker: Pytest-mock's mocker fixture.
        """
        decl1 = Declaration(id=1, lean_name="MyAxiom", decl_type="axiom")
        sg_source = StatementGroup(
            id=301,
            text_hash="hash_no_deps",
            statement_text="axiom lonely : Unit",
            primary_decl_id=1,
            primary_declaration=decl1,
            source_file="f",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
        )
        db_session.add_all([decl1, sg_source])
        db_session.commit()

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )
        result = initialized_service.get_dependencies(301)

        assert result is not None
        assert result.source_group_id == 301
        assert result.count == 0
        assert len(result.citations) == 0

    def test_search_successful(
        self, initialized_service: Service, mocker: "MockerFixture"
    ):
        """Tests the search method's orchestration and result serialization.

        This test mocks `perform_search` to focus on `Service.search`'s own
        logic: calling `perform_search`, serializing its results, applying limits,
        and wrapping the output in `APISearchResponse`.

        Args:
            mock_perform_search: Mock for `lean_explore.local.service.perform_search`.
            initialized_service: An initialized Service instance.
            mocker: Pytest-mock's mocker fixture.
        """
        mock_perform_search = mocker.patch("lean_explore.local.service.perform_search")
        mock_sg_orm1_decl = Declaration(id=1, lean_name="Res1", decl_type="def")
        mock_sg_orm1 = StatementGroup(
            id=10,
            text_hash="res1",
            statement_text="res1_text",
            primary_declaration=mock_sg_orm1_decl,
            source_file="F1",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
        )
        mock_sg_orm2_decl = Declaration(id=2, lean_name="Res2", decl_type="def")
        mock_sg_orm2 = StatementGroup(
            id=11,
            text_hash="res2",
            statement_text="res2_text",
            primary_declaration=mock_sg_orm2_decl,
            source_file="F2",
            range_start_line=2,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
        )

        mock_perform_search_results: List[Tuple[StatementGroup, Dict[str, float]]] = [
            (mock_sg_orm1, {"final_score": 0.9}),
            (mock_sg_orm2, {"final_score": 0.8}),
        ]
        mock_perform_search.return_value = mock_perform_search_results

        # Mock the SessionLocal context manager on the service instance
        mock_session = mocker.MagicMock(spec=SQLAlchemySession)
        mock_session_local_cm = mocker.MagicMock()
        mock_session_local_cm.__enter__.return_value = mock_session
        mock_session_local_cm.__exit__.return_value = None
        # Patch SessionLocal on the specific service instance for this test
        initialized_service.SessionLocal = lambda: mock_session_local_cm

        query_str = "test query"
        pkgs = ["Mathlib"]
        limit = 1
        response = initialized_service.search(
            query=query_str, package_filters=pkgs, limit=limit
        )

        assert isinstance(response, APISearchResponse)
        assert response.query == query_str
        assert response.packages_applied == pkgs
        assert response.count == limit  # Count after tool's limit
        assert len(response.results) == limit
        assert response.results[0].id == mock_sg_orm1.id
        assert response.results[0].primary_declaration.lean_name == "Res1"
        assert response.total_candidates_considered == len(
            mock_perform_search_results
        )  # Before limit
        assert response.processing_time_ms >= 0

        mock_perform_search.assert_called_once_with(
            session=mock_session,  # Ensure it received the session
            query_string=query_str,
            model=initialized_service.embedding_model,
            faiss_index=initialized_service.faiss_index,
            text_chunk_id_map=initialized_service.text_chunk_id_map,
            faiss_k=initialized_service.default_faiss_k,
            pagerank_weight=initialized_service.default_pagerank_weight,
            text_relevance_weight=initialized_service.default_text_relevance_weight,
            name_match_weight=initialized_service.default_name_match_weight,
            selected_packages=pkgs,
            semantic_similarity_threshold=initialized_service.default_semantic_similarity_threshold,
            faiss_nprobe=initialized_service.default_faiss_nprobe,
        )

    def test_search_perform_search_raises_exception(
        self, initialized_service: Service, mocker: "MockerFixture"
    ):
        """Tests that Service.search re-raises exceptions from perform_search.

        Args:
            mock_perform_search: Mock for `lean_explore.local.service.perform_search`.
            initialized_service: An initialized Service instance.
            mocker: Pytest-mock's mocker fixture.
        """
        mock_perform_search = mocker.patch("lean_explore.local.service.perform_search")
        mock_perform_search.side_effect = ValueError("perform_search failed")

        mock_session = mocker.MagicMock(spec=SQLAlchemySession)
        mock_session_local_cm = mocker.MagicMock()
        mock_session_local_cm.__enter__.return_value = mock_session
        mock_session_local_cm.__exit__.return_value = None
        initialized_service.SessionLocal = lambda: mock_session_local_cm

        with pytest.raises(ValueError, match="perform_search failed"):
            initialized_service.search(query="any query")

    def test_search_fails_if_assets_not_loaded(self, initialized_service: Service):
        """Tests that Service.search raises RuntimeError if core assets are missing.

        This tests the internal safeguard in the search method.

        Args:
            initialized_service: An initialized Service instance.
        """
        # Simulate a scenario where __init__ might have "succeeded" (due to mocks)
        # but a critical asset became None afterwards.
        initialized_service.embedding_model = (
            None  # Tamper with the initialized service
        )
        with pytest.raises(RuntimeError, match="Search service assets not loaded"):
            initialized_service.search(query="any")
