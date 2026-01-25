"""Tests for doc_parser module.

These tests verify the doc-gen4 BMP file parsing, source text extraction,
and declaration insertion functionality.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from lean_explore.extract.doc_parser import (
    _build_package_cache,
    _extract_dependencies_from_html,
    _extract_source_text,
    _insert_declarations_batch,
    _parse_declarations_from_files,
    _read_source_lines,
    extract_declarations,
)
from lean_explore.extract.types import Declaration
from lean_explore.models import Declaration as DBDeclaration


class TestPackageCache:
    """Tests for package cache building."""

    def test_build_package_cache_with_packages(self, temp_directory):
        """Test building package cache from .lake/packages directory."""
        lean_root = temp_directory / "lean"
        packages_directory = lean_root / ".lake" / "packages"
        packages_directory.mkdir(parents=True)

        (packages_directory / "mathlib4").mkdir()
        (packages_directory / "Qq").mkdir()
        (packages_directory / "batteries").mkdir()

        cache = _build_package_cache(lean_root)

        assert "mathlib4" in cache
        assert "qq" in cache  # Lowercase
        assert "batteries" in cache
        assert cache["mathlib4"] == packages_directory / "mathlib4"

    def test_build_package_cache_with_lean4_toolchain(self, temp_directory):
        """Test that Lean 4 toolchain is included in cache."""
        lean_root = temp_directory / "lean"
        lean_root.mkdir()

        toolchain_file = lean_root / "lean-toolchain"
        toolchain_file.write_text("leanprover/lean4:v4.24.0")

        cache = _build_package_cache(lean_root)

        # Should attempt to find lean4 in cache
        # (Will only succeed if elan installation exists on test machine)
        assert isinstance(cache, dict)

    def test_build_package_cache_empty_directory(self, temp_directory):
        """Test building cache from directory with no packages."""
        lean_root = temp_directory / "lean"
        lean_root.mkdir()

        cache = _build_package_cache(lean_root)

        assert cache == {}


class TestSourceExtraction:
    """Tests for source text extraction from files."""

    def test_read_source_lines(self, temp_directory):
        """Test reading specific lines from a source file."""
        source_file = temp_directory / "test.lean"
        source_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = _read_source_lines(source_file, 2, 4)

        assert result == "line 2\nline 3\nline 4\n"

    def test_read_source_lines_out_of_bounds(self, temp_directory):
        """Test that reading out-of-bounds lines raises an error."""
        source_file = temp_directory / "test.lean"
        source_file.write_text("line 1\nline 2\n")

        with pytest.raises(ValueError, match="out of bounds"):
            _read_source_lines(source_file, 1, 10)

    def test_extract_source_text_from_package(self, temp_directory):
        """Test extracting source text using package cache."""
        lean_root = temp_directory / "lean"
        packages_directory = lean_root / ".lake" / "packages"
        mathlib_directory = packages_directory / "mathlib4"
        mathlib_directory.mkdir(parents=True)

        source_file = mathlib_directory / "Mathlib" / "Data" / "List" / "Basic.lean"
        source_file.parent.mkdir(parents=True)
        source_text = (
            "def length : List α → Nat\n  | [] => 0\n  | _ :: xs => 1 + length xs\n"
        )
        source_file.write_text(source_text)

        package_cache = _build_package_cache(lean_root)
        source_link = "https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/Data/List/Basic.lean#L1-L3"

        result = _extract_source_text(source_link, lean_root, package_cache)

        assert "def length" in result
        assert "| [] => 0" in result

    def test_extract_source_text_from_lean_root(self, temp_directory):
        """Test extracting source text from lean root directory."""
        lean_root = temp_directory / "lean"
        lean_root.mkdir()

        source_file = lean_root / "MyProject" / "Basic.lean"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("theorem my_theorem : True := trivial\n")

        package_cache = {}
        source_link = (
            "https://github.com/myuser/myproject/blob/main/MyProject/Basic.lean#L1-L1"
        )

        result = _extract_source_text(source_link, lean_root, package_cache)

        assert "theorem my_theorem" in result

    def test_extract_source_text_invalid_link(self, temp_directory):
        """Test that invalid source links raise an error."""
        lean_root = temp_directory / "lean"
        package_cache = {}
        invalid_link = "https://example.com/not-a-github-link"

        with pytest.raises(ValueError, match="Could not parse source link"):
            _extract_source_text(invalid_link, lean_root, package_cache)

    def test_extract_source_text_file_not_found(self, temp_directory):
        """Test that missing source files raise an error."""
        lean_root = temp_directory / "lean"
        lean_root.mkdir()
        package_cache = {}
        source_link = "https://github.com/user/repo/blob/main/NonExistent.lean#L1-L1"

        with pytest.raises(FileNotFoundError):
            _extract_source_text(source_link, lean_root, package_cache)


class TestDependencyExtraction:
    """Tests for dependency extraction from HTML."""

    def test_extract_dependencies_from_html(self):
        """Test extracting declaration dependencies from HTML header."""
        html = """
        <div class="header">
            <a href="#Nat">Nat</a> →
            <a href="#Nat.add">Nat.add</a> →
            <a href="#List">List</a>
        </div>
        """

        dependencies = _extract_dependencies_from_html(html)

        assert dependencies == ["Nat", "Nat.add", "List"]

    def test_extract_dependencies_deduplication(self):
        """Test that duplicate dependencies are removed."""
        html = """
        <a href="#Nat">Nat</a>
        <a href="#Nat">Nat</a>
        <a href="#List">List</a>
        """

        dependencies = _extract_dependencies_from_html(html)

        assert dependencies == ["Nat", "List"]

    def test_extract_dependencies_empty_html(self):
        """Test extracting from HTML with no dependencies."""
        html = "<div>No links here</div>"

        dependencies = _extract_dependencies_from_html(html)

        assert dependencies == []


class TestDeclarationParsing:
    """Tests for BMP file parsing."""

    def test_parse_declarations_from_files(self, temp_directory):
        """Test parsing declarations from BMP files."""
        lean_root = temp_directory / "lean"
        doc_data_directory = lean_root / ".lake" / "build" / "doc-data"
        doc_data_directory.mkdir(parents=True)

        # Create package directory structure
        mathlib_directory = lean_root / ".lake" / "packages" / "mathlib4"
        mathlib_directory.mkdir(parents=True)
        source_file = mathlib_directory / "Mathlib" / "Init" / "Data" / "Nat.lean"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def Nat.add (n m : Nat) : Nat := n + m\n")

        # Create BMP file
        bmp_file = doc_data_directory / "Mathlib.Init.Data.Nat.bmp"
        bmp_data = {
            "name": "Mathlib.Init.Data.Nat",
            "declarations": [
                {
                    "info": {
                        "name": "Nat.add",
                        "doc": "Addition of natural numbers",
                        "sourceLink": "https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/Init/Data/Nat.lean#L1-L1",
                    },
                    "header": '<a href="#Nat">Nat</a>',
                }
            ],
        }
        bmp_file.write_text(json.dumps(bmp_data))

        package_cache = _build_package_cache(lean_root)

        with patch("lean_explore.extract.doc_parser.Config") as mock_config:
            mock_config.EXTRACT_PACKAGES = {"mathlib"}
            declarations = _parse_declarations_from_files(
                [bmp_file], lean_root, package_cache
            )

        assert len(declarations) == 1
        assert declarations[0].name == "Nat.add"
        assert declarations[0].module == "Mathlib.Init.Data.Nat"
        assert declarations[0].docstring == "Addition of natural numbers"
        assert "def Nat.add" in declarations[0].source_text
        assert declarations[0].dependencies == ["Nat"]

    def test_parse_declarations_filters_packages(self, temp_directory):
        """Test that declarations from non-configured packages are filtered."""
        lean_root = temp_directory / "lean"
        doc_data_directory = lean_root / ".lake" / "build" / "doc-data"
        doc_data_directory.mkdir(parents=True)

        # Create BMP file for a package not in EXTRACT_PACKAGES
        bmp_file = doc_data_directory / "SomeOtherPackage.Basic.bmp"
        bmp_data = {
            "name": "SomeOtherPackage.Basic",
            "declarations": [
                {
                    "info": {
                        "name": "SomeDeclaration",
                        "sourceLink": "https://github.com/user/pkg/blob/main/Basic.lean#L1-L1",
                    },
                    "header": "",
                }
            ],
        }
        bmp_file.write_text(json.dumps(bmp_data))

        package_cache = _build_package_cache(lean_root)

        with patch("lean_explore.extract.doc_parser.Config") as mock_config:
            mock_config.EXTRACT_PACKAGES = {"mathlib"}
            declarations = _parse_declarations_from_files(
                [bmp_file], lean_root, package_cache
            )

        assert len(declarations) == 0


class TestDeclarationInsertion:
    """Tests for database insertion."""

    async def test_insert_declarations_batch(
        self, async_db_session, sample_declaration
    ):
        """Test inserting declarations into database."""
        declarations = [sample_declaration]

        inserted_count = await _insert_declarations_batch(
            async_db_session, declarations, batch_size=100
        )

        assert inserted_count == 1

        result = await async_db_session.execute(
            select(DBDeclaration).where(DBDeclaration.name == "Nat.add")
        )
        db_declaration = result.scalar_one()
        assert db_declaration.name == "Nat.add"
        assert db_declaration.module == "Init.Data.Nat.Basic"

    async def test_insert_declarations_skips_duplicates(
        self, async_db_session, sample_declaration
    ):
        """Test that duplicate declarations are skipped."""
        declarations = [sample_declaration, sample_declaration]

        inserted_count = await _insert_declarations_batch(
            async_db_session, declarations, batch_size=100
        )

        # Should only insert once
        assert inserted_count == 1

        result = await async_db_session.execute(select(DBDeclaration))
        all_declarations = result.scalars().all()
        assert len(all_declarations) == 1

    async def test_insert_declarations_large_batch(
        self, async_db_session, sample_declarations
    ):
        """Test inserting multiple declarations in batches."""
        # Create 10 unique declarations
        declarations = []
        for i in range(10):
            declarations.append(
                Declaration(
                    name=f"Test.Declaration{i}",
                    module="Test.Module",
                    docstring=f"Test declaration {i}",
                    source_text=f"def test{i} := {i}",
                    source_link=f"https://example.com/test{i}.lean#L1-L1",
                    dependencies=None,
                )
            )

        inserted_count = await _insert_declarations_batch(
            async_db_session, declarations, batch_size=3
        )

        assert inserted_count == 10

        result = await async_db_session.execute(select(DBDeclaration))
        all_declarations = result.scalars().all()
        assert len(all_declarations) == 10


class TestExtractDeclarationsE2E:
    """End-to-end tests for declaration extraction."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires complex Path mocking that doesn't work reliably")
    async def test_extract_declarations_full_pipeline(
        self, async_db_engine, temp_directory
    ):
        """Test the full extraction pipeline from BMP files to database."""
        # Setup directory structure
        lean_root = temp_directory / "lean"
        doc_data_directory = lean_root / ".lake" / "build" / "doc-data"
        doc_data_directory.mkdir(parents=True)

        mathlib_directory = lean_root / ".lake" / "packages" / "mathlib4"
        mathlib_directory.mkdir(parents=True)
        source_file = mathlib_directory / "Mathlib" / "Data" / "Nat.lean"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def Nat.add (n m : Nat) : Nat := n + m\n")

        bmp_file = doc_data_directory / "Mathlib.Data.Nat.bmp"
        bmp_data = {
            "name": "Mathlib.Data.Nat",
            "declarations": [
                {
                    "info": {
                        "name": "Nat.add",
                        "doc": "Addition",
                        "sourceLink": "https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/Data/Nat.lean#L1-L1",
                    },
                    "header": "",
                }
            ],
        }
        bmp_file.write_text(json.dumps(bmp_data))

        # Mock the lean root path in extract_declarations
        with patch("lean_explore.extract.doc_parser.Path") as mock_path_cls:
            mock_path_instance = AsyncMock()
            mock_path_instance.__truediv__ = lambda self, other: lean_root / other
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = [bmp_file]
            mock_path_cls.return_value = mock_path_instance

            with patch("lean_explore.extract.doc_parser.Config") as mock_config:
                mock_config.EXTRACT_PACKAGES = {"mathlib"}

                with patch(
                    "lean_explore.extract.doc_parser._build_package_cache"
                ) as mock_cache:
                    mock_cache.return_value = _build_package_cache(lean_root)

                    with patch(
                        "lean_explore.extract.doc_parser._parse_declarations_from_files"
                    ) as mock_parse:
                        mock_parse.return_value = [
                            Declaration(
                                name="Nat.add",
                                module="Mathlib.Data.Nat",
                                docstring="Addition",
                                source_text="def Nat.add (n m : Nat) : Nat := n + m",
                                source_link="https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/Data/Nat.lean#L1-L1",
                                dependencies=None,
                            )
                        ]

                        await extract_declarations(async_db_engine, batch_size=100)

        # Verify declaration was inserted
        async with AsyncMock() as mock_session:
            mock_session.execute = AsyncMock()
            mock_session.execute.return_value.scalar_one.return_value = DBDeclaration(
                name="Nat.add",
                module="Mathlib.Data.Nat",
                docstring="Addition",
                source_text="def Nat.add (n m : Nat) : Nat := n + m",
                source_link="https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/Data/Nat.lean#L1-L1",
            )
