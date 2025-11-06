"""Shared test fixtures and configuration for lean-explore test suite."""

import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from lean_explore.models.search_db import Base, Declaration


@pytest.fixture
async def async_db_engine():
    """Create an in-memory SQLite database engine for testing.

    Returns:
        AsyncEngine: SQLAlchemy async engine connected to in-memory database.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_db_session(
    async_db_engine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing.

    Args:
        async_db_engine: The async database engine fixture.

    Yields:
        AsyncSession: SQLAlchemy async session for database operations.
    """
    async with AsyncSession(async_db_engine, expire_on_commit=False) as session:
        yield session
        await session.commit()
        await session.close()


@pytest.fixture
def sample_declaration() -> Declaration:
    """Create a sample Declaration object for testing.

    Returns:
        Declaration: A sample Lean declaration with typical field values.
    """
    return Declaration(
        name="Nat.add",
        module="Init.Data.Nat.Basic",
        docstring="Addition of natural numbers",
        source_text="def add (n m : Nat) : Nat := n + m",
        source_link="https://github.com/leanprover/lean4/blob/master/src/Init/Data/Nat/Basic.lean#L100-L101",
        dependencies='["Nat", "Nat.succ"]',
        informalization="Adds two natural numbers together",
        pagerank=0.5,
        name_embedding=[0.1] * 768,
        informalization_embedding=[0.4] * 768,
        source_text_embedding=[0.7] * 768,
        docstring_embedding=[1.0] * 768,
    )


@pytest.fixture
def sample_declarations() -> list[Declaration]:
    """Create multiple sample Declaration objects for testing.

    Returns:
        list[Declaration]: A list of sample declarations with dependencies.
    """
    return [
        Declaration(
            name="Nat",
            module="Init.Prelude",
            docstring="Natural numbers",
            source_text="inductive Nat | zero | succ (n : Nat)",
            source_link="https://github.com/leanprover/lean4/blob/master/src/Init/Prelude.lean#L50",
            dependencies="[]",
            informalization="The type of natural numbers",
            pagerank=0.8,
        ),
        Declaration(
            name="Nat.succ",
            module="Init.Prelude",
            docstring="Successor function",
            source_text="def succ (n : Nat) : Nat := Nat.succ n",
            source_link="https://github.com/leanprover/lean4/blob/master/src/Init/Prelude.lean#L51",
            dependencies='["Nat"]',
            informalization="Returns the successor of a natural number",
            pagerank=0.6,
        ),
        Declaration(
            name="Nat.add",
            module="Init.Data.Nat.Basic",
            docstring="Addition of natural numbers",
            source_text="def add (n m : Nat) : Nat := n + m",
            source_link="https://github.com/leanprover/lean4/blob/master/src/Init/Data/Nat/Basic.lean#L100",
            dependencies='["Nat", "Nat.succ"]',
            informalization="Adds two natural numbers together",
            pagerank=0.5,
        ),
    ]


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for file operations.

    Yields:
        Path: Path object pointing to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Create a mock OpenAI client for testing LLM interactions.

    Returns:
        AsyncMock: Mock client that simulates OpenAI API responses.
    """
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="This is a mock informalization"))
    ]
    mock_client.generate = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Create a mock embedding client for testing embedding generation.

    Returns:
        MagicMock: Mock client that simulates embedding API responses.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Each embedding is a 768-dimensional vector
    mock_response.embeddings = [[0.1] * 768 for _ in range(10)]
    mock_client.embed = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def empty_embedding_caches():
    """Create empty embedding caches for testing.

    Returns:
        EmbeddingCaches: Empty caches with no pre-loaded embeddings.
    """
    from lean_explore.extract.embeddings import EmbeddingCaches

    return EmbeddingCaches(
        by_name={},
        by_informalization={},
        by_source_text={},
        by_docstring={},
    )


@pytest.fixture
def sample_bmp_json_data() -> dict:
    """Create sample BMP (doc-gen4) JSON data for parser testing.

    Returns:
        dict: Sample JSON structure matching doc-gen4 output format.
    """
    return {
        "name": "Nat.add",
        "kind": "def",
        "doc": "Addition of natural numbers",
        "docLink": "https://github.com/leanprover/lean4/blob/master/src/Init/Data/Nat/Basic.lean#L100-L101",
    }


@pytest.fixture
def mock_faiss_index() -> MagicMock:
    """Create a mock FAISS index for testing search functionality.

    Returns:
        MagicMock: Mock FAISS index with search method.
    """
    mock_index = MagicMock()
    mock_index.search.return_value = (
        [[0.9, 0.8, 0.7]],  # Distances
        [[0, 1, 2]],  # Indices
    )
    mock_index.ntotal = 100
    return mock_index
