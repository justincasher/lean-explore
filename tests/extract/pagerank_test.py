"""Tests for PageRank calculation.

These tests verify the dependency graph building and PageRank score calculation
for Lean declarations.
"""

import pytest
from sqlalchemy import select

from lean_explore.extract.pagerank import _build_dependency_graph, calculate_pagerank
from lean_explore.models import Declaration


class TestDependencyGraphBuilding:
    """Tests for building dependency graphs from declarations."""

    async def test_build_dependency_graph_simple(
        self, async_db_session, sample_declarations
    ):
        """Test building a simple dependency graph."""
        for declaration in sample_declarations:
            async_db_session.add(declaration)
        await async_db_session.commit()

        graph = await _build_dependency_graph(async_db_session)

        # Should have 3 nodes (Nat, Nat.succ, Nat.add)
        assert graph.number_of_nodes() == 3

        # Should have edges from dependents to dependencies
        # Nat.succ -> Nat
        # Nat.add -> Nat
        # Nat.add -> Nat.succ
        assert graph.number_of_edges() == 3

        # Verify specific edges exist
        nat_id = next(d.id for d in sample_declarations if d.name == "Nat")
        nat_succ_id = next(d.id for d in sample_declarations if d.name == "Nat.succ")
        nat_add_id = next(d.id for d in sample_declarations if d.name == "Nat.add")

        assert graph.has_edge(nat_succ_id, nat_id)
        assert graph.has_edge(nat_add_id, nat_id)
        assert graph.has_edge(nat_add_id, nat_succ_id)

    async def test_build_dependency_graph_no_dependencies(self, async_db_session):
        """Test graph building with declarations that have no dependencies."""
        declaration = Declaration(
            name="Standalone",
            module="Test",
            source_text="def standalone := 42",
            source_link="https://example.com",
            dependencies=None,
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        graph = await _build_dependency_graph(async_db_session)

        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

    async def test_build_dependency_graph_missing_dependencies(self, async_db_session):
        """Test that missing dependencies are silently ignored."""
        declaration = Declaration(
            name="Test.Declaration",
            module="Test",
            source_text="test",
            source_link="https://example.com",
            dependencies='["NonExistent", "AlsoMissing"]',
        )
        async_db_session.add(declaration)
        await async_db_session.commit()

        graph = await _build_dependency_graph(async_db_session)

        # Should have node for the declaration but no edges
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

    async def test_build_dependency_graph_json_list_dependencies(
        self, async_db_session
    ):
        """Test graph building with dependencies as JSON array string."""
        decl1 = Declaration(
            name="Base",
            module="Test",
            source_text="def base := 1",
            source_link="https://example.com",
            dependencies=None,
        )
        decl2 = Declaration(
            name="Derived",
            module="Test",
            source_text="def derived := base + 1",
            source_link="https://example.com",
            dependencies='["Base"]',  # JSON string
        )
        async_db_session.add(decl1)
        async_db_session.add(decl2)
        await async_db_session.commit()

        graph = await _build_dependency_graph(async_db_session)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1

    async def test_build_dependency_graph_cyclic(self, async_db_session):
        """Test graph building with cyclic dependencies."""
        decl1 = Declaration(
            name="A",
            module="Test",
            source_text="def a := b",
            source_link="https://example.com",
            dependencies='["B"]',
        )
        decl2 = Declaration(
            name="B",
            module="Test",
            source_text="def b := a",
            source_link="https://example.com",
            dependencies='["A"]',
        )
        async_db_session.add(decl1)
        async_db_session.add(decl2)
        await async_db_session.commit()

        graph = await _build_dependency_graph(async_db_session)

        # Should handle cycles gracefully
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 2


class TestPageRankCalculation:
    """Tests for PageRank score calculation."""

    async def test_calculate_pagerank_simple(
        self, async_db_engine, sample_declarations
    ):
        """Test PageRank calculation on a simple dependency graph."""
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(async_db_engine) as session:
            for declaration in sample_declarations:
                session.add(declaration)
            await session.commit()

        await calculate_pagerank(async_db_engine, alpha=0.85, batch_size=100)

        # Verify PageRank scores were calculated and stored
        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            declarations = result.scalars().all()

            for declaration in declarations:
                assert declaration.pagerank is not None
                assert declaration.pagerank > 0
                assert declaration.pagerank <= 1

            # Nat should have highest PageRank (most depended upon)
            nat_declaration = next(d for d in declarations if d.name == "Nat")
            assert nat_declaration.pagerank > 0.3

    async def test_calculate_pagerank_empty_database(self, async_db_engine):
        """Test PageRank calculation with no declarations."""
        await calculate_pagerank(async_db_engine, alpha=0.85, batch_size=100)

        # Should complete without errors (logged warning)

    async def test_calculate_pagerank_different_alpha(
        self, async_db_engine, sample_declarations
    ):
        """Test PageRank with different damping parameter."""
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(async_db_engine) as session:
            for declaration in sample_declarations:
                session.add(declaration)
            await session.commit()

        # Calculate with different alpha value
        await calculate_pagerank(async_db_engine, alpha=0.5, batch_size=100)

        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            declarations = result.scalars().all()

            for declaration in declarations:
                assert declaration.pagerank is not None

    async def test_calculate_pagerank_batching(self, async_db_engine):
        """Test PageRank calculation with small batch size."""
        from sqlalchemy.ext.asyncio import AsyncSession

        # Create many declarations to test batching
        declarations = []
        for i in range(50):
            declarations.append(
                Declaration(
                    name=f"Declaration{i}",
                    module="Test",
                    source_text=f"def decl{i} := {i}",
                    source_link=f"https://example.com/{i}",
                    dependencies='["Declaration0"]' if i > 0 else None,
                )
            )

        async with AsyncSession(async_db_engine) as session:
            for declaration in declarations:
                session.add(declaration)
            await session.commit()

        # Use small batch size to test batching logic
        await calculate_pagerank(async_db_engine, alpha=0.85, batch_size=10)

        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(select(Declaration))
            all_declarations = result.scalars().all()

            assert len(all_declarations) == 50
            for declaration in all_declarations:
                assert declaration.pagerank is not None


class TestPageRankE2E:
    """End-to-end PageRank tests."""

    @pytest.mark.integration
    async def test_pagerank_realistic_dependency_graph(self, async_db_engine):
        """Test PageRank on a realistic declaration dependency graph."""
        from sqlalchemy.ext.asyncio import AsyncSession

        # Create a realistic graph structure:
        # - Core types (high PageRank)
        # - Operations depending on core types (medium PageRank)
        # - Theorems depending on operations (lower PageRank)

        declarations = [
            # Core type
            Declaration(
                name="Nat",
                module="Init.Prelude",
                source_text="inductive Nat | zero | succ (n : Nat)",
                source_link="https://example.com/nat",
                dependencies=None,
            ),
            # Basic operations
            Declaration(
                name="Nat.add",
                module="Init.Data.Nat",
                source_text="def add (n m : Nat) : Nat := n + m",
                source_link="https://example.com/add",
                dependencies='["Nat"]',
            ),
            Declaration(
                name="Nat.mul",
                module="Init.Data.Nat",
                source_text="def mul (n m : Nat) : Nat := n * m",
                source_link="https://example.com/mul",
                dependencies='["Nat", "Nat.add"]',
            ),
            # Theorems
            Declaration(
                name="Nat.add_comm",
                module="Mathlib.Data.Nat.Basic",
                source_text="theorem add_comm (n m : Nat) : n + m = m + n",
                source_link="https://example.com/add_comm",
                dependencies='["Nat", "Nat.add"]',
            ),
            Declaration(
                name="Nat.mul_comm",
                module="Mathlib.Data.Nat.Basic",
                source_text="theorem mul_comm (n m : Nat) : n * m = m * n",
                source_link="https://example.com/mul_comm",
                dependencies='["Nat", "Nat.mul", "Nat.add"]',
            ),
        ]

        async with AsyncSession(async_db_engine) as session:
            for declaration in declarations:
                session.add(declaration)
            await session.commit()

        await calculate_pagerank(async_db_engine, alpha=0.85, batch_size=100)

        async with AsyncSession(async_db_engine) as session:
            result = await session.execute(
                select(Declaration).order_by(Declaration.pagerank.desc())
            )
            ranked_declarations = result.scalars().all()

            # Nat should have highest PageRank (most fundamental)
            assert ranked_declarations[0].name == "Nat"

            # Operations should have higher PageRank than theorems
            operation_names = {"Nat.add", "Nat.mul"}
            theorem_names = {"Nat.add_comm", "Nat.mul_comm"}

            operation_ranks = [
                d.pagerank for d in ranked_declarations if d.name in operation_names
            ]
            theorem_ranks = [
                d.pagerank for d in ranked_declarations if d.name in theorem_names
            ]

            # At least some operations should rank higher than theorems
            assert max(operation_ranks) > min(theorem_ranks)
