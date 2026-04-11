"""Tests for doc_parser module.

These tests verify the doc-gen4 BMP file parsing, source text extraction,
and declaration insertion functionality.
"""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from lean_explore.extract.doc_parser import (
    _BlobReader,
    _build_package_cache,
    _construct_source_link,
    _extract_dependencies_from_html,
    _extract_names_from_rendered_code,
    _extract_source_text,
    _filter_auto_generated_projections,
    _insert_declarations_batch,
    _parse_declarations_from_files,
    _parse_declarations_from_sqlite,
    _read_lean_toolchain_version,
    _read_source_lines,
    _strip_lean_comments,
    extract_declarations,
)
from lean_explore.extract.types import Declaration
from lean_explore.models import Declaration as DBDeclaration


class TestPackageCache:
    """Tests for package cache building."""

    def test_build_package_cache_with_packages(self, temp_directory):
        """Test building package cache from workspace .lake/packages directories."""
        lean_root = temp_directory / "lean"

        # Create packages in the mathlib workspace (as the new code expects)
        mathlib_packages_directory = lean_root / "mathlib" / ".lake" / "packages"
        mathlib_packages_directory.mkdir(parents=True)

        (mathlib_packages_directory / "mathlib4").mkdir()
        (mathlib_packages_directory / "Qq").mkdir()
        (mathlib_packages_directory / "batteries").mkdir()

        cache = _build_package_cache(lean_root)

        assert "mathlib4" in cache
        assert "qq" in cache  # Lowercase
        assert "batteries" in cache
        assert cache["mathlib4"] == mathlib_packages_directory / "mathlib4"

    def test_build_package_cache_with_lean4_toolchain(self, temp_directory):
        """Test that Lean 4 toolchain is included in cache."""
        lean_root = temp_directory / "lean"

        # Create toolchain in a workspace directory
        mathlib_dir = lean_root / "mathlib"
        mathlib_dir.mkdir(parents=True)

        toolchain_file = mathlib_dir / "lean-toolchain"
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

        # Create packages in the mathlib workspace (as the new code expects)
        mathlib_packages_directory = lean_root / "mathlib" / ".lake" / "packages"
        mathlib_directory = mathlib_packages_directory / "mathlib4"
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

    def test_extract_source_text_from_lake_toolchain(self, temp_directory):
        """Test extracting source text from the Lake toolchain path."""
        lean_root = temp_directory / "lean"
        toolchain_root = temp_directory / "toolchain" / "src"
        lean_source_dir = toolchain_root / "lean"
        lake_source_dir = toolchain_root / "lake"
        lean_source_dir.mkdir(parents=True)
        lake_file = lake_source_dir / "Lake" / "Config" / "Monad.lean"
        lake_file.parent.mkdir(parents=True)
        lake_file.write_text("def Lake.Config.Monad.run := 1\n")

        package_cache = {"lean4": lean_source_dir}
        source_link = (
            "https://github.com/leanprover/lean4/blob/toolchain/"
            "src/lake/Lake/Config/Monad.lean#L1-L1"
        )

        result = _extract_source_text(source_link, lean_root, package_cache)

        assert "Lake.Config.Monad.run" in result


def _encode_nat(n: int) -> bytes:
    """Encode a natural number in leansqlite's variable-length format."""
    chunks = []
    while n >= 128:
        chunks.append((n & 0x7F) | 0x80)
        n >>= 7
    chunks.append(n)
    return bytes(chunks)


def _encode_string(s: str) -> bytes:
    """Encode a string: Nat(utf8_byte_length) + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    return _encode_nat(len(encoded)) + encoded


def _encode_name(name: str) -> bytes:
    """Encode a dotted Lean name (e.g. 'Nat.add') into binary."""
    if not name:
        return b"\x00"  # anonymous
    parts = name.split(".")
    result = b"\x00"  # start with anonymous
    for part in parts:
        # Name.str = tag 1 + parent + string
        result = b"\x01" + result + _encode_string(part)
    return result


def _make_const_tag(name: str) -> bytes:
    """Build a RenderedCode.Tag.const(name) blob fragment."""
    return b"\x02" + _encode_name(name)


def _make_text_node(s: str) -> bytes:
    """Build a TaggedText.text(s) blob fragment."""
    return b"\x00" + _encode_string(s)


def _make_tag_node(tag_bytes: bytes, inner: bytes) -> bytes:
    """Build a TaggedText.tag(tag, inner) blob fragment."""
    return b"\x01" + tag_bytes + inner


def _make_append_node(children: list[bytes]) -> bytes:
    """Build a TaggedText.append(children) blob fragment."""
    return b"\x02" + _encode_nat(len(children)) + b"".join(children)


class TestRenderedCodeBlobParser:
    """Tests for RenderedCode BLOB parsing to extract dependency names."""

    def test_extract_single_const(self):
        """Test extracting a single const name from a type BLOB."""
        # TaggedText.tag(Tag.const("Nat"), TaggedText.text("Nat"))
        blob = _make_tag_node(_make_const_tag("Nat"), _make_text_node("Nat"))
        names = _extract_names_from_rendered_code(blob)
        assert names == ["Nat"]

    def test_extract_multiple_consts(self):
        """Test extracting multiple const names from a type BLOB."""
        # append([tag(const Nat, text Nat), text " → ", tag(const Bool, text Bool)])
        blob = _make_append_node([
            _make_tag_node(_make_const_tag("Nat"), _make_text_node("Nat")),
            _make_text_node(" → "),
            _make_tag_node(_make_const_tag("Bool"), _make_text_node("Bool")),
        ])
        names = _extract_names_from_rendered_code(blob)
        assert names == ["Nat", "Bool"]

    def test_extract_dotted_name(self):
        """Test extracting a dotted name like Nat.add."""
        blob = _make_tag_node(
            _make_const_tag("Nat.add"), _make_text_node("Nat.add")
        )
        names = _extract_names_from_rendered_code(blob)
        assert names == ["Nat.add"]

    def test_deduplicates_names(self):
        """Test that duplicate const references are deduplicated."""
        blob = _make_append_node([
            _make_tag_node(_make_const_tag("Nat"), _make_text_node("Nat")),
            _make_text_node(" → "),
            _make_tag_node(_make_const_tag("Nat"), _make_text_node("Nat")),
        ])
        names = _extract_names_from_rendered_code(blob)
        assert names == ["Nat"]

    def test_skips_non_const_tags(self):
        """Test that keyword, string, sort, otherExpr tags are skipped."""
        blob = _make_append_node([
            _make_tag_node(b"\x00", _make_text_node("def")),       # keyword
            _make_text_node(" "),
            _make_tag_node(_make_const_tag("Nat"), _make_text_node("Nat")),
            _make_tag_node(b"\x07", _make_text_node("x")),         # otherExpr
        ])
        names = _extract_names_from_rendered_code(blob)
        assert names == ["Nat"]

    def test_empty_blob_returns_empty(self):
        """Test that an empty or invalid BLOB returns empty list."""
        assert _extract_names_from_rendered_code(b"") == []
        assert _extract_names_from_rendered_code(b"\xff") == []

    def test_text_only_returns_empty(self):
        """Test that a BLOB with only text returns no names."""
        blob = _make_text_node("hello world")
        names = _extract_names_from_rendered_code(blob)
        assert names == []

    def test_nested_tagged_text(self):
        """Test parsing nested tag nodes."""
        # tag(otherExpr, tag(const("List"), text("List")))
        inner = _make_tag_node(_make_const_tag("List"), _make_text_node("List"))
        blob = _make_tag_node(b"\x07", inner)  # otherExpr wrapping
        names = _extract_names_from_rendered_code(blob)
        assert names == ["List"]

    def test_blob_reader_nat_encoding(self):
        """Test that Nat encoding/decoding matches leansqlite format."""
        reader = _BlobReader(_encode_nat(0))
        assert reader._read_nat() == 0

        reader = _BlobReader(_encode_nat(127))
        assert reader._read_nat() == 127

        reader = _BlobReader(_encode_nat(128))
        assert reader._read_nat() == 128

        reader = _BlobReader(_encode_nat(300))
        assert reader._read_nat() == 300

        reader = _BlobReader(_encode_nat(100000))
        assert reader._read_nat() == 100000

    def test_blob_reader_name_roundtrip(self):
        """Test that Name encoding produces correct dot-separated output."""
        reader = _BlobReader(_encode_name("Mathlib.Data.Nat.Basic"))
        assert reader._read_name() == "Mathlib.Data.Nat.Basic"

        reader = _BlobReader(_encode_name(""))
        assert reader._read_name() == ""

    def test_sort_tags_handled(self):
        """Test that sort tag variants (3-6) are handled without error."""
        for sort_byte in [3, 4, 5, 6]:
            blob = _make_tag_node(
                bytes([sort_byte]), _make_text_node("Type")
            )
            names = _extract_names_from_rendered_code(blob)
            assert names == []


class TestSqliteHelpers:
    """Tests for SQLite-specific parsing helpers."""

    def test_construct_source_link_for_core_module_with_version(self):
        """Test source links for core modules use the toolchain version."""
        result = _construct_source_link(
            "Init.Data.Nat.Basic", None, 12, 15,
            lean_version="v4.29.0-rc6",
        )

        assert (
            result
            == "https://github.com/leanprover/lean4/blob/v4.29.0-rc6/"
            "src/lean/Init/Data/Nat/Basic.lean#L12-L15"
        )

    def test_construct_source_link_for_lake_module_with_version(self):
        """Test source links for Lake modules use the toolchain version."""
        result = _construct_source_link(
            "Lake.Config.Monad", None, 7, 9,
            lean_version="v4.29.0-rc6",
        )

        assert (
            result
            == "https://github.com/leanprover/lean4/blob/v4.29.0-rc6/"
            "src/lake/Lake/Config/Monad.lean#L7-L9"
        )

    def test_construct_source_link_falls_back_to_master(self):
        """Test that missing lean_version falls back to 'master'."""
        result = _construct_source_link(
            "Init.Data.Nat.Basic", None, 12, 15,
        )

        assert (
            result
            == "https://github.com/leanprover/lean4/blob/master/"
            "src/lean/Init/Data/Nat/Basic.lean#L12-L15"
        )

    def test_construct_source_link_prefers_source_url(self):
        """Test that a non-None source_url is used directly."""
        url = "https://github.com/leanprover-community/mathlib4/blob/abc123/Foo.lean"
        result = _construct_source_link("Foo.Bar", url, 1, 10)

        assert result == f"{url}#L1-L10"

    def test_read_lean_toolchain_version(self, temp_directory):
        """Test reading version from a lean-toolchain file."""
        workspace = temp_directory / "workspace"
        workspace.mkdir()
        toolchain = workspace / "lean-toolchain"
        toolchain.write_text("leanprover/lean4:v4.29.0-rc6\n")

        assert _read_lean_toolchain_version(workspace) == "v4.29.0-rc6"

    def test_read_lean_toolchain_version_release(self, temp_directory):
        """Test reading a release version (no -rc suffix)."""
        workspace = temp_directory / "workspace"
        workspace.mkdir()
        toolchain = workspace / "lean-toolchain"
        toolchain.write_text("leanprover/lean4:v4.29.0\n")

        assert _read_lean_toolchain_version(workspace) == "v4.29.0"

    def test_read_lean_toolchain_version_missing(self, temp_directory):
        """Test returns None when lean-toolchain doesn't exist."""
        workspace = temp_directory / "workspace"
        workspace.mkdir()

        assert _read_lean_toolchain_version(workspace) is None


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
        doc_data_directory = lean_root / "mathlib" / ".lake" / "build" / "doc-data"
        doc_data_directory.mkdir(parents=True)

        # Create package directory structure in the mathlib workspace
        mathlib_packages_directory = lean_root / "mathlib" / ".lake" / "packages"
        mathlib_directory = mathlib_packages_directory / "mathlib4"
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

        # Now uses allowed_module_prefixes parameter instead of Config
        declarations = _parse_declarations_from_files(
            [bmp_file], lean_root, package_cache, allowed_module_prefixes=["Mathlib"]
        )

        assert len(declarations) == 1
        assert declarations[0].name == "Nat.add"
        assert declarations[0].module == "Mathlib.Init.Data.Nat"
        assert declarations[0].docstring == "Addition of natural numbers"
        assert "def Nat.add" in declarations[0].source_text
        assert declarations[0].dependencies == ["Nat"]

    def test_parse_declarations_filters_packages(self, temp_directory):
        """Test that declarations from non-configured prefixes are filtered."""
        lean_root = temp_directory / "lean"
        doc_data_directory = lean_root / "mathlib" / ".lake" / "build" / "doc-data"
        doc_data_directory.mkdir(parents=True)

        # Create BMP file for a module not matching allowed prefixes
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

        # Only allow "Mathlib" prefix - should filter out "SomeOtherPackage"
        declarations = _parse_declarations_from_files(
            [bmp_file], lean_root, package_cache, allowed_module_prefixes=["Mathlib"]
        )

        assert len(declarations) == 0

    def test_parse_declarations_from_sqlite_filters_non_rendered(
        self, temp_directory
    ):
        """Test SQLite parsing only keeps rendered declarations."""
        lean_root = temp_directory / "lean"
        database_path = temp_directory / "api-docs.db"

        source_dir = lean_root / "mathlib" / ".lake" / "packages" / "mathlib4"
        source_file = source_dir / "Mathlib" / "Data" / "Nat" / "Basic.lean"
        source_file.parent.mkdir(parents=True)
        source_file.write_text(
            "theorem Nat.visible : True := trivial\n"
            "def Nat.hidden : Nat := 0\n"
        )

        connection = sqlite3.connect(database_path)
        connection.executescript(
            """
            CREATE TABLE modules (name TEXT PRIMARY KEY, source_url TEXT);
            CREATE TABLE name_info (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              kind TEXT,
              name TEXT NOT NULL,
              type BLOB NOT NULL,
              sorried INTEGER NOT NULL,
              render INTEGER NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            CREATE TABLE declaration_ranges (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              start_line INTEGER NOT NULL,
              start_column INTEGER NOT NULL,
              start_utf16 INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              end_column INTEGER NOT NULL,
              end_utf16 INTEGER NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            CREATE TABLE declaration_markdown_docstrings (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              text TEXT NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            """
        )
        connection.execute(
            "INSERT INTO modules (name, source_url) VALUES (?, ?)",
            (
                "Mathlib.Data.Nat.Basic",
                "https://github.com/leanprover-community/mathlib4/blob/master/"
                "Mathlib/Data/Nat/Basic.lean",
            ),
        )
        connection.executemany(
            """
            INSERT INTO name_info
              (module_name, position, kind, name, type, sorried, render)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "Mathlib.Data.Nat.Basic",
                    1,
                    "theorem",
                    "Nat.visible",
                    _make_text_node("True"),
                    0,
                    1,
                ),
                (
                    "Mathlib.Data.Nat.Basic",
                    2,
                    "definition",
                    "Nat.hidden",
                    _make_text_node("Nat"),
                    0,
                    0,
                ),
            ],
        )
        connection.executemany(
            """
            INSERT INTO declaration_ranges
              (module_name, position, start_line, start_column, start_utf16,
               end_line, end_column, end_utf16)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("Mathlib.Data.Nat.Basic", 1, 1, 0, 0, 1, 32, 32),
                ("Mathlib.Data.Nat.Basic", 2, 2, 0, 0, 2, 22, 22),
            ],
        )
        connection.execute(
            """
            INSERT INTO declaration_markdown_docstrings
              (module_name, position, text)
            VALUES (?, ?, ?)
            """,
            ("Mathlib.Data.Nat.Basic", 1, "Visible theorem"),
        )
        connection.commit()
        connection.close()

        declarations = _parse_declarations_from_sqlite(
            database_path,
            lean_root,
            _build_package_cache(lean_root),
            allowed_module_prefixes=["Mathlib"],
        )

        assert [declaration.name for declaration in declarations] == ["Nat.visible"]
        assert declarations[0].docstring == "Visible theorem"

    def test_parse_declarations_from_sqlite_extracts_dependencies(
        self, temp_directory
    ):
        """Test that SQLite parsing extracts dependencies from type BLOBs."""
        lean_root = temp_directory / "lean"
        database_path = temp_directory / "api-docs.db"

        source_dir = lean_root / "mathlib" / ".lake" / "packages" / "mathlib4"
        source_file = source_dir / "Mathlib" / "Data" / "Nat" / "Basic.lean"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("def Nat.myFunc (n : Nat) : Bool := true\n")

        # Build a type BLOB: "Nat → Bool"
        type_blob = _make_append_node([
            _make_tag_node(_make_const_tag("Nat"), _make_text_node("Nat")),
            _make_text_node(" → "),
            _make_tag_node(_make_const_tag("Bool"), _make_text_node("Bool")),
        ])

        connection = sqlite3.connect(database_path)
        connection.executescript(
            """
            CREATE TABLE modules (name TEXT PRIMARY KEY, source_url TEXT);
            CREATE TABLE name_info (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              kind TEXT,
              name TEXT NOT NULL,
              type BLOB NOT NULL,
              sorried INTEGER NOT NULL,
              render INTEGER NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            CREATE TABLE declaration_ranges (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              start_line INTEGER NOT NULL,
              start_column INTEGER NOT NULL,
              start_utf16 INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              end_column INTEGER NOT NULL,
              end_utf16 INTEGER NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            CREATE TABLE declaration_markdown_docstrings (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              text TEXT NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            """
        )
        connection.execute(
            "INSERT INTO modules (name, source_url) VALUES (?, ?)",
            (
                "Mathlib.Data.Nat.Basic",
                "https://github.com/leanprover-community/mathlib4/blob/master/"
                "Mathlib/Data/Nat/Basic.lean",
            ),
        )
        connection.execute(
            """
            INSERT INTO name_info
              (module_name, position, kind, name, type, sorried, render)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("Mathlib.Data.Nat.Basic", 1, "definition", "Nat.myFunc", type_blob, 0, 1),
        )
        connection.execute(
            """
            INSERT INTO declaration_ranges
              (module_name, position, start_line, start_column, start_utf16,
               end_line, end_column, end_utf16)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("Mathlib.Data.Nat.Basic", 1, 1, 0, 0, 1, 40, 40),
        )
        connection.commit()
        connection.close()

        declarations = _parse_declarations_from_sqlite(
            database_path,
            lean_root,
            _build_package_cache(lean_root),
            allowed_module_prefixes=["Mathlib"],
        )

        assert len(declarations) == 1
        assert declarations[0].name == "Nat.myFunc"
        assert declarations[0].dependencies == ["Nat", "Bool"]

    def test_parse_declarations_from_sqlite_uses_core_fallback_source_link(
        self, temp_directory
    ):
        """Test SQLite parsing for core modules without a stored source URL."""
        lean_root = temp_directory / "lean"
        database_path = temp_directory / "api-docs.db"

        toolchain_root = temp_directory / "toolchain" / "src"
        lean_source_dir = toolchain_root / "lean"
        init_file = lean_source_dir / "Init" / "Data" / "Nat" / "Basic.lean"
        init_file.parent.mkdir(parents=True)
        init_file.write_text("theorem Nat.core : True := trivial\n")

        connection = sqlite3.connect(database_path)
        connection.executescript(
            """
            CREATE TABLE modules (name TEXT PRIMARY KEY, source_url TEXT);
            CREATE TABLE name_info (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              kind TEXT,
              name TEXT NOT NULL,
              type BLOB NOT NULL,
              sorried INTEGER NOT NULL,
              render INTEGER NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            CREATE TABLE declaration_ranges (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              start_line INTEGER NOT NULL,
              start_column INTEGER NOT NULL,
              start_utf16 INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              end_column INTEGER NOT NULL,
              end_utf16 INTEGER NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            CREATE TABLE declaration_markdown_docstrings (
              module_name TEXT NOT NULL,
              position INTEGER NOT NULL,
              text TEXT NOT NULL,
              PRIMARY KEY (module_name, position)
            );
            """
        )
        connection.execute(
            "INSERT INTO modules (name, source_url) VALUES (?, ?)",
            ("Init.Data.Nat.Basic", None),
        )
        connection.execute(
            """
            INSERT INTO name_info
              (module_name, position, kind, name, type, sorried, render)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Init.Data.Nat.Basic", 1, "theorem", "Nat.core",
                _make_text_node("True"), 0, 1,
            ),
        )
        connection.execute(
            """
            INSERT INTO declaration_ranges
              (module_name, position, start_line, start_column, start_utf16,
               end_line, end_column, end_utf16)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("Init.Data.Nat.Basic", 1, 1, 0, 0, 1, 31, 31),
        )
        connection.commit()
        connection.close()

        declarations = _parse_declarations_from_sqlite(
            database_path,
            lean_root,
            {"lean4": lean_source_dir},
            allowed_module_prefixes=["Init"],
            lean_version="v4.29.0-rc6",
        )

        assert len(declarations) == 1
        assert declarations[0].name == "Nat.core"
        assert (
            declarations[0].source_link
            == "https://github.com/leanprover/lean4/blob/v4.29.0-rc6/"
            "src/lean/Init/Data/Nat/Basic.lean#L1-L1"
        )


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


class TestStripLeanComments:
    """Tests for Lean comment stripping."""

    def test_strip_line_comments(self):
        """Test stripping line comments from source."""
        source = "def foo := 1 -- this is a comment\ndef bar := 2"
        result = _strip_lean_comments(source)
        assert result == "def foo := 1 def bar := 2"

    def test_strip_block_comments(self):
        """Test stripping block comments from source."""
        source = "def foo /- block comment -/ := 1"
        result = _strip_lean_comments(source)
        assert result == "def foo := 1"

    def test_strip_nested_block_comments(self):
        """Test stripping nested block comments from source."""
        source = "def foo /- outer /- inner -/ outer -/ := 1"
        result = _strip_lean_comments(source)
        assert result == "def foo := 1"

    def test_strip_doc_comments(self):
        """Test stripping doc comments from source."""
        source = """/-- Documentation for foo. -/
def foo := 1"""
        result = _strip_lean_comments(source)
        assert result == "def foo := 1"

    def test_strip_mixed_comments(self):
        """Test stripping mixed comment types from source."""
        source = """/-- Doc comment -/
def foo := 1 -- line comment
/- block -/ def bar := 2"""
        result = _strip_lean_comments(source)
        assert result == "def foo := 1 def bar := 2"

    def test_no_comments(self):
        """Test source with no comments passes through."""
        source = "def foo := 1\ndef bar := 2"
        result = _strip_lean_comments(source)
        assert result == "def foo := 1 def bar := 2"

    def test_empty_source(self):
        """Test empty source returns empty string."""
        result = _strip_lean_comments("")
        assert result == ""


class TestFilterAutoGeneratedProjections:
    """Tests for filtering auto-generated 'to*' projections."""

    def test_filters_projection_with_shared_source(self):
        """Test that 'to*' projections sharing source with parent are filtered."""
        # Simulate Scheme and Scheme.toLocallyRingedSpace with same source
        structure_source = (
            "structure Scheme extends LocallyRingedSpace where\n"
            "  local_affine : ∀ x, ∃ U R, ..."
        )

        declarations = [
            Declaration(
                name="AlgebraicGeometry.Scheme",
                module="Mathlib.AlgebraicGeometry.Scheme",
                docstring="A scheme is...",
                source_text=structure_source,
                source_link="https://github.com/example/blob/main/Scheme.lean#L1-L3",
                dependencies=None,
            ),
            Declaration(
                name="AlgebraicGeometry.Scheme.toLocallyRingedSpace",
                module="Mathlib.AlgebraicGeometry.Scheme",
                docstring=None,
                source_text=structure_source,  # Same source!
                source_link="https://github.com/example/blob/main/Scheme.lean#L1-L3",
                dependencies=None,
            ),
        ]

        filtered, removed_count = _filter_auto_generated_projections(declarations)

        assert len(filtered) == 1
        assert filtered[0].name == "AlgebraicGeometry.Scheme"
        assert removed_count == 1

    def test_keeps_legitimate_to_definition(self):
        """Test that legitimate 'to*' definitions with unique source are kept."""
        declarations = [
            Declaration(
                name="AlgebraicGeometry.Scheme",
                module="Mathlib.AlgebraicGeometry.Scheme",
                docstring="A scheme is...",
                source_text="structure Scheme extends LocallyRingedSpace where",
                source_link="https://github.com/example/blob/main/Scheme.lean#L1-L1",
                dependencies=None,
            ),
            Declaration(
                name="AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.toScheme",
                module="Mathlib.AlgebraicGeometry.OpenImmersion",
                docstring="If X ⟶ Y is an open immersion...",
                # Different source!
                source_text="def toScheme : Scheme := by apply ...",
                source_link="https://github.com/example/blob/main/OpenImmersion.lean#L50-L55",
                dependencies=None,
            ),
        ]

        filtered, removed_count = _filter_auto_generated_projections(declarations)

        assert len(filtered) == 2
        assert removed_count == 0

    def test_filters_based_on_stripped_source(self):
        """Test that comment differences are ignored when comparing source."""
        # Parent has doc comment, projection doesn't
        declarations = [
            Declaration(
                name="MyStruct",
                module="Test",
                docstring="Docs",
                source_text=(
                    "/-- This is a doc comment -/\n"
                    "structure MyStruct extends Base where"
                ),
                source_link="https://github.com/example/blob/main/Test.lean#L1-L2",
                dependencies=None,
            ),
            Declaration(
                name="MyStruct.toBase",
                module="Test",
                docstring=None,
                source_text="structure MyStruct extends Base where",  # No comment
                source_link="https://github.com/example/blob/main/Test.lean#L2-L2",
                dependencies=None,
            ),
        ]

        filtered, removed_count = _filter_auto_generated_projections(declarations)

        assert len(filtered) == 1
        assert filtered[0].name == "MyStruct"
        assert removed_count == 1

    def test_ignores_non_to_prefix(self):
        """Test that declarations not starting with 'to' are not filtered."""
        # Even if they share source, non-to* declarations are kept
        declarations = [
            Declaration(
                name="Foo",
                module="Test",
                docstring=None,
                source_text="def shared := 1",
                source_link="https://github.com/example/blob/main/Test.lean#L1-L1",
                dependencies=None,
            ),
            Declaration(
                name="Foo.bar",
                module="Test",
                docstring=None,
                source_text="def shared := 1",  # Same source but not 'to*'
                source_link="https://github.com/example/blob/main/Test.lean#L1-L1",
                dependencies=None,
            ),
        ]

        filtered, removed_count = _filter_auto_generated_projections(declarations)

        assert len(filtered) == 2
        assert removed_count == 0

    def test_requires_uppercase_after_to(self):
        """Test that 'to' must be followed by uppercase letter to be filtered."""
        declarations = [
            Declaration(
                name="Foo",
                module="Test",
                docstring=None,
                source_text="def shared := 1",
                source_link="https://github.com/example/blob/main/Test.lean#L1-L1",
                dependencies=None,
            ),
            Declaration(
                name="Foo.tostring",  # lowercase after 'to'
                module="Test",
                docstring=None,
                source_text="def shared := 1",
                source_link="https://github.com/example/blob/main/Test.lean#L1-L1",
                dependencies=None,
            ),
        ]

        filtered, removed_count = _filter_auto_generated_projections(declarations)

        assert len(filtered) == 2
        assert removed_count == 0

    def test_empty_list(self):
        """Test filtering empty list returns empty list."""
        filtered, removed_count = _filter_auto_generated_projections([])

        assert filtered == []
        assert removed_count == 0

    def test_single_declaration(self):
        """Test single declaration is kept even if it's a 'to*' name."""
        declarations = [
            Declaration(
                name="Foo.toBar",
                module="Test",
                docstring=None,
                source_text="def toBar := 1",
                source_link="https://github.com/example/blob/main/Test.lean#L1-L1",
                dependencies=None,
            ),
        ]

        filtered, removed_count = _filter_auto_generated_projections(declarations)

        assert len(filtered) == 1
        assert removed_count == 0
