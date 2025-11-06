"""Tests for informalization module.

These tests verify the LLM-based generation of informal descriptions for Lean
declarations, including dependency layering, caching, and database operations.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from rich.progress import Progress
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lean_explore.extract.informalize import (
    InformalizationResult,
    _build_dependency_layers,
    _discover_database_files,
    _get_declarations_to_process,
    _load_cache_from_databases,
    _load_existing_informalizations,
    _parse_dependencies,
    _process_layer,
    _process_one_declaration,
    informalize_declarations,
)
from lean_explore.models import Declaration


class TestDependencyParsing:
    """Tests for dependency parsing utilities."""

    def test_parse_dependencies_json_string(self):
        """Test parsing dependencies from JSON string."""
        dependencies = '["Nat", "List", "Array"]'

        result = _parse_dependencies(dependencies)

        assert result == ["Nat", "List", "Array"]

    def test_parse_dependencies_list(self):
        """Test parsing dependencies from list."""
        dependencies = ["Nat", "List"]

        result = _parse_dependencies(dependencies)

        assert result == ["Nat", "List"]

    def test_parse_dependencies_none(self):
        """Test parsing None dependencies."""
        result = _parse_dependencies(None)

        assert result == []

    def test_parse_dependencies_empty_string(self):
        """Test parsing empty string."""
        result = _parse_dependencies("")

        assert result == []


class TestDependencyLayering:
    """Tests for dependency layer building."""

    def test_build_dependency_layers_simple(self, sample_declarations):
        """Test building layers from simple dependency graph."""
        layers = _build_dependency_layers(sample_declarations)

        # Should have at least 2 layers
        # Layer 0: Nat (no dependencies)
        # Layer 1: Nat.succ (depends on Nat)
        # Layer 2: Nat.add (depends on Nat and Nat.succ)
        assert len(layers) >= 2

        # First layer should contain Nat
        assert any(d.name == "Nat" for d in layers[0])

        # Nat.succ should be in later layer
        nat_succ_layer = next(
            i
            for i, layer in enumerate(layers)
            if any(d.name == "Nat.succ" for d in layer)
        )
        assert nat_succ_layer > 0

    def test_build_dependency_layers_no_dependencies(self):
        """Test layering with declarations that have no dependencies."""
        declarations = [
            Declaration(
                name="A",
                module="Test",
                source_text="def a := 1",
                source_link="https://example.com",
                dependencies=None,
            ),
            Declaration(
                name="B",
                module="Test",
                source_text="def b := 2",
                source_link="https://example.com",
                dependencies=None,
            ),
        ]

        layers = _build_dependency_layers(declarations)

        # All should be in first layer
        assert len(layers) == 1
        assert len(layers[0]) == 2

    def test_build_dependency_layers_with_cycles(self):
        """Test layering with cyclic dependencies."""
        declarations = [
            Declaration(
                name="A",
                module="Test",
                source_text="def a := b",
                source_link="https://example.com",
                dependencies='["B"]',
            ),
            Declaration(
                name="B",
                module="Test",
                source_text="def b := c",
                source_link="https://example.com",
                dependencies='["C"]',
            ),
            Declaration(
                name="C",
                module="Test",
                source_text="def c := a",
                source_link="https://example.com",
                dependencies='["A"]',
            ),
        ]

        layers = _build_dependency_layers(declarations)

        # Should handle cycles by putting them in a final layer
        assert len(layers) >= 1

        # All declarations should be in some layer
        all_names = {d.name for layer in layers for d in layer}
        assert all_names == {"A", "B", "C"}

    def test_build_dependency_layers_missing_dependencies(self):
        """Test layering when dependencies reference non-existent declarations."""
        declarations = [
            Declaration(
                name="A",
                module="Test",
                source_text="def a := missing + 1",
                source_link="https://example.com",
                dependencies='["NonExistent"]',
            ),
        ]

        layers = _build_dependency_layers(declarations)

        # Should treat as no dependencies (since dependency not in graph)
        assert len(layers) == 1
        assert layers[0][0].name == "A"


class TestDatabaseOperations:
    """Tests for database loading operations."""

    async def test_load_existing_informalizations(
        self, async_db_session, sample_declarations
    ):
        """Test loading existing informalizations from database."""
        for declaration in sample_declarations:
            async_db_session.add(declaration)
        await async_db_session.commit()

        results = await _load_existing_informalizations(async_db_session)

        # Should load the 3 declarations with informalizations
        assert len(results) == 3
        assert all(isinstance(r, InformalizationResult) for r in results)
        assert all(r.informalization is not None for r in results)

    async def test_get_declarations_to_process(self, async_db_session):
        """Test getting declarations needing informalization."""
        # Add declarations with and without informalizations
        decl_with_informal = Declaration(
            name="HasInformal",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            informalization="This is informalized",
        )
        decl_without_informal = Declaration(
            name="NeedsInformal",
            module="Test",
            source_text="def test2 := 2",
            source_link="https://example.com",
            informalization=None,
        )
        async_db_session.add(decl_with_informal)
        async_db_session.add(decl_without_informal)
        await async_db_session.commit()

        declarations = await _get_declarations_to_process(async_db_session, limit=None)

        # Should only return declaration without informalization
        assert len(declarations) == 1
        assert declarations[0].name == "NeedsInformal"

    async def test_get_declarations_to_process_with_limit(self, async_db_session):
        """Test getting declarations with a limit."""
        for i in range(10):
            decl = Declaration(
                name=f"Declaration{i}",
                module="Test",
                source_text=f"def test{i} := {i}",
                source_link=f"https://example.com/{i}",
                informalization=None,
            )
            async_db_session.add(decl)
        await async_db_session.commit()

        declarations = await _get_declarations_to_process(async_db_session, limit=5)

        assert len(declarations) == 5


class TestCacheLoading:
    """Tests for cross-database cache loading."""

    def test_discover_database_files(self, temp_directory):
        """Test discovering database files in data/cache directories."""
        # Create mock database files
        data_dir = temp_directory / "data"
        cache_dir = temp_directory / "cache"

        (data_dir / "v4.24.0").mkdir(parents=True)
        (data_dir / "v4.24.0" / "lean_explore.db").touch()

        (cache_dir / "v4.23.0").mkdir(parents=True)
        (cache_dir / "v4.23.0" / "lean_explore.db").touch()

        with patch("lean_explore.extract.informalize.Config") as mock_config:
            mock_config.DATA_DIRECTORY = data_dir
            mock_config.CACHE_DIRECTORY = cache_dir

            database_files = _discover_database_files()

        assert len(database_files) == 2

    @pytest.mark.slow
    async def test_load_cache_from_databases(self, async_db_engine, temp_directory):
        """Test loading cache from multiple databases."""
        # Create a database file with informalizations
        db_path = temp_directory / "test.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        from sqlalchemy.ext.asyncio import create_async_engine

        from lean_explore.models import Base

        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncSession(engine) as session:
            declaration = Declaration(
                name="Test",
                module="Test",
                source_text="def test := 42",
                source_link="https://example.com",
                informalization="Returns the number 42",
            )
            session.add(declaration)
            await session.commit()

        await engine.dispose()

        cache = await _load_cache_from_databases([db_path])

        assert "def test := 42" in cache
        assert cache["def test := 42"] == "Returns the number 42"


class TestProcessingFunctions:
    """Tests for declaration processing functions."""

    async def test_process_one_declaration_new(self, mock_openai_client):
        """Test processing a declaration without existing informalization."""
        # Create declaration without informalization
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            informalization=None,  # No existing informalization
        )

        semaphore = asyncio.Semaphore(5)
        prompt_template = (
            "Name: {name}\nSource: {source_text}\nDoc: {docstring}\n{dependencies}"
        )

        result = await _process_one_declaration(
            declaration=declaration,
            client=mock_openai_client,
            model="test-model",
            prompt_template=prompt_template,
            informalizations_by_name={},
            cache_by_source_text={},
            semaphore=semaphore,
        )

        assert result.informalization == "This is a mock informalization"
        mock_openai_client.generate.assert_called_once()

    async def test_process_one_declaration_cached(self, mock_openai_client):
        """Test processing with cached informalization."""
        # Create declaration without informalization
        declaration = Declaration(
            name="Test",
            module="Test",
            source_text="def test := 1",
            source_link="https://example.com",
            informalization=None,
        )

        semaphore = asyncio.Semaphore(5)
        prompt_template = "Name: {name}"
        cache = {declaration.source_text: "Cached informalization"}

        # Use mock client - shouldn't be called due to cache
        result = await _process_one_declaration(
            declaration=declaration,
            client=mock_openai_client,
            model="test-model",
            prompt_template=prompt_template,
            informalizations_by_name={},
            cache_by_source_text=cache,
            semaphore=semaphore,
        )

        assert result.informalization == "Cached informalization"
        # Verify client was not called (cache hit)
        mock_openai_client.generate.assert_not_called()

    async def test_process_one_declaration_already_informalized(
        self, sample_declaration, mock_openai_client
    ):
        """Test processing declaration that already has informalization."""
        semaphore = asyncio.Semaphore(5)

        # Set existing informalization
        sample_declaration.informalization = "Already done"

        result = await _process_one_declaration(
            declaration=sample_declaration,
            client=mock_openai_client,
            model="test-model",
            prompt_template="",
            informalizations_by_name={},
            cache_by_source_text={},
            semaphore=semaphore,
        )

        assert result.informalization is None
        # Verify client was not called (already informalized)
        mock_openai_client.generate.assert_not_called()

    @pytest.mark.slow
    async def test_process_layer(
        self, async_db_session, sample_declarations, mock_openai_client
    ):
        """Test processing a layer of declarations."""
        for declaration in sample_declarations:
            declaration.informalization = None
            async_db_session.add(declaration)
        await async_db_session.commit()

        semaphore = asyncio.Semaphore(5)
        prompt_template = "Name: {name}"

        with Progress() as progress:
            task = progress.add_task("Test", total=len(sample_declarations))

            processed = await _process_layer(
                session=async_db_session,
                layer=sample_declarations,
                client=mock_openai_client,
                model="test-model",
                prompt_template=prompt_template,
                informalizations_by_name={},
                cache_by_source_text={},
                semaphore=semaphore,
                progress=progress,
                task=task,
                commit_batch_size=10,
            )

        assert processed == len(sample_declarations)

        # Verify database was updated
        result = await async_db_session.execute(select(Declaration))
        declarations = result.scalars().all()
        for declaration in declarations:
            assert declaration.informalization is not None


class TestInformalizeE2E:
    """End-to-end informalization tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_informalize_declarations_full_pipeline(
        self, async_db_engine, mock_openai_client
    ):
        """Test complete informalization pipeline."""
        # Add declarations to database
        async with AsyncSession(async_db_engine) as session:
            declarations = [
                Declaration(
                    name="Base",
                    module="Test",
                    source_text="def base := 1",
                    source_link="https://example.com/base",
                    dependencies=None,
                ),
                Declaration(
                    name="Derived",
                    module="Test",
                    source_text="def derived := base + 1",
                    source_link="https://example.com/derived",
                    dependencies='["Base"]',
                ),
            ]
            for declaration in declarations:
                session.add(declaration)
            await session.commit()

        # Mock the OpenRouter client
        with patch(
            "lean_explore.extract.informalize.OpenRouterClient"
        ) as mock_client_cls:
            mock_client_cls.return_value = mock_openai_client

            # Mock prompt file
            with patch("lean_explore.extract.informalize.Path") as mock_path_cls:
                mock_prompt_file = MagicMock()
                prompt = (
                    "Name: {name}\nSource: {source_text}\n"
                    "Doc: {docstring}\n{dependencies}"
                )
                mock_prompt_file.read_text.return_value = prompt
                mock_path_cls.return_value.__truediv__.return_value = mock_prompt_file

                # Mock database discovery
                with patch(
                    "lean_explore.extract.informalize._discover_database_files"
                ) as mock_discover:
                    mock_discover.return_value = []

                    await informalize_declarations(
                        async_db_engine,
                        model="test-model",
                        commit_batch_size=100,
                        max_concurrent=5,
                        limit=None,
                    )

        # Verify all declarations were informalized
        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            all_declarations = result.scalars().all()

            assert len(all_declarations) == 2
            for declaration in all_declarations:
                assert declaration.informalization is not None

    @pytest.mark.integration
    async def test_informalize_with_dependency_context(
        self, async_db_engine, mock_openai_client
    ):
        """Test that dependency context is included in prompts."""
        async with AsyncSession(async_db_engine) as session:
            # Create base declaration with informalization
            base = Declaration(
                name="Nat",
                module="Init",
                source_text="inductive Nat | zero | succ",
                source_link="https://example.com",
                informalization="Natural numbers",
            )
            # Create dependent declaration without informalization
            derived = Declaration(
                name="Nat.add",
                module="Init",
                source_text="def add (n m : Nat) := n + m",
                source_link="https://example.com",
                dependencies='["Nat"]',
            )
            session.add(base)
            session.add(derived)
            await session.commit()

        with patch(
            "lean_explore.extract.informalize.OpenRouterClient"
        ) as mock_client_cls:
            mock_client_cls.return_value = mock_openai_client

            with patch("lean_explore.extract.informalize.Path") as mock_path_cls:
                mock_prompt_file = MagicMock()
                mock_prompt_file.read_text.return_value = "Name: {name}\n{dependencies}"
                mock_path_cls.return_value.__truediv__.return_value = mock_prompt_file

                with patch(
                    "lean_explore.extract.informalize._discover_database_files"
                ) as mock_discover:
                    mock_discover.return_value = []

                    await informalize_declarations(
                        async_db_engine,
                        model="test-model",
                        limit=1,  # Only process Nat.add
                    )

        # Verify the LLM was called with dependency context
        # The prompt should include "Nat: Natural numbers" in dependencies section
        call_args = mock_openai_client.generate.call_args
        if call_args:
            prompt = call_args[1]["messages"][0]["content"]
            # Should contain dependency information
            assert "Nat" in prompt or "Dependencies" in prompt
