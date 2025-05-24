# tests/conftest.py

"""Shared fixtures for the lean-explore test suite.

This file provides fixtures that are automatically discovered by pytest
and can be used by any test function within the 'tests' directory
and its subdirectories. These fixtures help in setting up common
test preconditions, isolating tests from the actual user environment,
and providing mock objects.
"""

import pathlib
from typing import TYPE_CHECKING, Iterator, List
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import faiss
import sqlalchemy
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.orm import sessionmaker

import lean_explore.cli.config_utils
import lean_explore.defaults
from lean_explore.shared.models.db import Base as DBBase


@pytest.fixture
def isolated_config_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Redirects CLI configuration utilities to use a temporary directory.

    This fixture ensures that tests interacting with user configuration
    (e.g., API keys via `lean_explore.cli.config_utils`) do not affect
    the actual user's configuration files. It monkeypatches
    `lean_explore.cli.config_utils.get_config_file_path` to return a path
    within a test-specific temporary directory.

    The yielded path points to the application-specific configuration
    directory (e.g., `.../tmp_dir/.config/leanexplore/`).

    Args:
        tmp_path: Pytest's built-in fixture for a temporary directory.
        monkeypatch: Pytest's built-in fixture for modifying objects during tests.

    Yields:
        pathlib.Path: The path to the temporary isolated 'leanexplore'
                      configuration directory.
    """
    fake_config_app_dir_name = lean_explore.cli.config_utils._APP_CONFIG_DIR_NAME
    fake_config_filename = lean_explore.cli.config_utils._CONFIG_FILENAME

    fake_user_home_config_dir = tmp_path / ".config"
    fake_app_config_dir = fake_user_home_config_dir / fake_config_app_dir_name
    fake_app_config_dir.mkdir(parents=True, exist_ok=True)
    fake_config_file = fake_app_config_dir / fake_config_filename

    def mock_get_config_file_path() -> pathlib.Path:
        """Returns the path to the isolated, temporary config file."""
        return fake_config_file

    monkeypatch.setattr(
        lean_explore.cli.config_utils, "get_config_file_path", mock_get_config_file_path
    )
    yield fake_app_config_dir


@pytest.fixture
def isolated_data_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Redirects default data and toolchain paths to temporary locations.

    This fixture modifies constants in `lean_explore.defaults` to ensure
    that tests interacting with local data assets (database, FAISS index,
    ID maps, toolchain versions, active toolchain config) operate within a
    temporary directory structure created under `tmp_path`.

    The temporary structure resembles:
    `tmp_path_root/test_lean_explore_user_data/`
        `active_toolchain.txt`
        `toolchains/<DEFAULT_ACTIVE_TOOLCHAIN_VERSION>/`
            `lean_explore_data.db`
            `main_faiss.index`
            `faiss_ids_map.json`

    Args:
        tmp_path: Pytest's built-in fixture for a temporary directory.
        monkeypatch: Pytest's built-in fixture for modifying objects.

    Yields:
        pathlib.Path: The root path of the isolated user data directory structure.
    """
    fake_user_data_root = tmp_path / "test_lean_explore_user_data"
    fake_user_data_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        lean_explore.defaults, "LEAN_EXPLORE_USER_DATA_DIR", fake_user_data_root
    )

    fake_active_toolchain_file = (
        fake_user_data_root / lean_explore.defaults.ACTIVE_TOOLCHAIN_CONFIG_FILENAME
    )
    monkeypatch.setattr(
        lean_explore.defaults,
        "ACTIVE_TOOLCHAIN_CONFIG_FILE_PATH",
        fake_active_toolchain_file,
    )

    fake_toolchains_base_dir = fake_user_data_root / "toolchains"
    fake_toolchains_base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        lean_explore.defaults,
        "LEAN_EXPLORE_TOOLCHAINS_BASE_DIR",
        fake_toolchains_base_dir,
    )

    active_version_str = lean_explore.defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION
    fake_active_toolchain_version_data_path = (
        fake_toolchains_base_dir / active_version_str
    )
    fake_active_toolchain_version_data_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        lean_explore.defaults,
        "_ACTIVE_TOOLCHAIN_VERSION_DATA_PATH",
        fake_active_toolchain_version_data_path,
    )

    fake_db_path = (
        fake_active_toolchain_version_data_path
        / lean_explore.defaults.DEFAULT_DB_FILENAME
    )
    monkeypatch.setattr(lean_explore.defaults, "DEFAULT_DB_PATH", fake_db_path)
    monkeypatch.setattr(
        lean_explore.defaults, "DEFAULT_DB_URL", f"sqlite:///{fake_db_path.resolve()}"
    )

    fake_faiss_index_path = (
        fake_active_toolchain_version_data_path
        / lean_explore.defaults.DEFAULT_FAISS_INDEX_FILENAME
    )
    monkeypatch.setattr(
        lean_explore.defaults, "DEFAULT_FAISS_INDEX_PATH", fake_faiss_index_path
    )

    fake_faiss_map_path = (
        fake_active_toolchain_version_data_path
        / lean_explore.defaults.DEFAULT_FAISS_MAP_FILENAME
    )
    monkeypatch.setattr(
        lean_explore.defaults, "DEFAULT_FAISS_MAP_PATH", fake_faiss_map_path
    )

    yield fake_user_data_root


@pytest.fixture
def mock_load_embedding_model(mocker: "MockerFixture") -> MagicMock:
    """Mocks the embedding model loader used by the Service class.

    This fixture patches `load_embedding_model` as it is imported and used
    within `lean_explore.local.service`. It allows tests to control the
    behavior of embedding model loading during `Service` initialization.

    Args:
        mocker: Pytest-mock's mocker fixture. The type hint is a string
                literal to avoid import issues at conftest load time if
                `pytest_mock` isn't immediately discoverable, while still
                providing type information for static analysis.

    Yields:
        unittest.mock.MagicMock: The mock object replacing
                                 `lean_explore.local.service.load_embedding_model`.
    """
    mock_model_instance = mocker.MagicMock(spec=SentenceTransformer)
    mock_loader = mocker.patch(
        "lean_explore.local.service.load_embedding_model",
        return_value=mock_model_instance,
    )
    yield mock_loader


@pytest.fixture
def mock_load_faiss_assets(mocker: "MockerFixture") -> MagicMock:
    """Mocks the FAISS assets loader used by the Service class.

    This fixture patches `load_faiss_assets` as it is imported and used
    within `lean_explore.local.service`. Tests can use the yielded mock
    to control the outcome of FAISS asset loading, such as returning
    mocked index and ID map objects or simulating failures. By default,
    it returns mock FAISS index and a concrete list for ID map objects.

    Args:
        mocker: Pytest-mock's mocker fixture. (Type hinted as string literal).

    Yields:
        unittest.mock.MagicMock: The mock object replacing
                                 `lean_explore.local.service.load_faiss_assets`.
    """
    mock_faiss_index_instance = mocker.MagicMock(spec=faiss.Index)
    mock_id_map_instance: List[str] = ["id_map_val_1", "id_map_val_2"]
    mock_loader = mocker.patch(
        "lean_explore.local.service.load_faiss_assets",
        return_value=(mock_faiss_index_instance, mock_id_map_instance),
    )
    yield mock_loader


@pytest.fixture
def mock_sqlalchemy_engine_setup(mocker: "MockerFixture") -> MagicMock:
    """Mocks `sqlalchemy.create_engine` as used by the Service class.

    This fixture patches `create_engine` as it is imported and used within
    `lean_explore.local.service`. The mock engine returned by the patched
    `create_engine` has a `connect` method that is also a mock, configured
    by default to simulate a successful connection.

    Args:
        mocker: Pytest-mock's mocker fixture. (Type hinted as string literal).

    Yields:
        unittest.mock.MagicMock: The mock object replacing
                                 `lean_explore.local.service.create_engine`.
    """
    mock_engine = mocker.MagicMock(spec=sqlalchemy.engine.Engine)
    mock_connection = mocker.MagicMock(spec=sqlalchemy.engine.Connection)

    mock_connection.__enter__.return_value = mock_connection
    mock_connection.__exit__.return_value = None

    mock_engine.connect.return_value = mock_connection

    mock_create_engine = mocker.patch(
        "lean_explore.local.service.create_engine", return_value=mock_engine
    )

    yield mock_create_engine


@pytest.fixture(scope="function")
def db_session() -> Iterator[SQLAlchemySession]:
    """Provides an SQLAlchemy session with an in-memory SQLite database.

    All tables defined in `lean_explore.shared.models.db.Base.metadata`
    are created in this in-memory database before the session is yielded.
    After the test using this fixture completes, the session is closed,
    all tables are dropped, and the engine is disposed of, ensuring a
    clean database state for each test.

    Yields:
        sqlalchemy.orm.Session: An active SQLAlchemy session connected to a
                                fresh in-memory SQLite database with the
                                application's schema.
    """
    engine = create_engine("sqlite:///:memory:")
    DBBase.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        DBBase.metadata.drop_all(engine)
        engine.dispose()
