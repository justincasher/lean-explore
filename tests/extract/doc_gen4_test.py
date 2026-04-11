"""Tests for doc-gen4 workspace orchestration."""

from pathlib import Path
from unittest.mock import patch

from lean_explore.extract.doc_gen4 import _uses_sqlite_docgen, run_doc_gen4


class TestDocGen4VersionDetection:
    """Tests for doc-gen4 format detection by Lean version."""

    def test_uses_sqlite_docgen_from_rc2(self):
        """Test the SQLite cutoff at v4.29.0-rc2."""
        assert _uses_sqlite_docgen("leanprover/lean4:v4.29.0-rc2")
        assert _uses_sqlite_docgen("leanprover/lean4:v4.29.0")
        assert not _uses_sqlite_docgen("leanprover/lean4:v4.29.0-rc1")
        assert not _uses_sqlite_docgen("leanprover/lean4:v4.28.0")


def _write_toolchain(version: str) -> None:
    """Write a lean-toolchain file for the mathlib workspace."""
    path = Path("lean/mathlib/lean-toolchain")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(version + "\n")


class TestRunDocGen4FreshHandling:
    """Tests for fresh-cache handling in run_doc_gen4."""

    async def test_fresh_legacy_workspace_clears_cache(self):
        """Test that legacy doc-gen4 workspaces still clear the cache."""
        _write_toolchain("leanprover/lean4:v4.29.0-rc1")
        with patch(
            "lean_explore.extract.doc_gen4.get_extraction_order",
            return_value=["mathlib"],
        ):
            with patch(
                "lean_explore.extract.doc_gen4._setup_workspace",
                return_value=("leanprover/lean4:v4.29.0-rc1", "v4.29.0-rc1"),
            ):
                with patch(
                    "lean_explore.extract.doc_gen4._clear_workspace_cache"
                ) as mock_clear:
                    with patch(
                        "lean_explore.extract.doc_gen4._run_lake_for_package"
                    ) as mock_run:
                        await run_doc_gen4(fresh=True)

        mock_clear.assert_called_once()
        mock_run.assert_called_once_with("mathlib", False)

    async def test_fresh_sqlite_workspace_skips_cache_clear(self):
        """Test that SQLite workspaces with unchanged toolchain skip clear."""
        _write_toolchain("leanprover/lean4:v4.29.0-rc2")
        with patch(
            "lean_explore.extract.doc_gen4.get_extraction_order",
            return_value=["mathlib"],
        ):
            with patch(
                "lean_explore.extract.doc_gen4._setup_workspace",
                return_value=("leanprover/lean4:v4.29.0-rc2", "v4.29.0-rc2"),
            ):
                with patch(
                    "lean_explore.extract.doc_gen4._clear_workspace_cache"
                ) as mock_clear:
                    with patch(
                        "lean_explore.extract.doc_gen4._run_lake_for_package"
                    ) as mock_run:
                        await run_doc_gen4(fresh=True)

        mock_clear.assert_not_called()
        mock_run.assert_called_once_with("mathlib", False)

    async def test_fresh_sqlite_workspace_clears_on_toolchain_change(self):
        """Test that a toolchain version change forces a cache clear."""
        _write_toolchain("leanprover/lean4:v4.29.0-rc2")
        with patch(
            "lean_explore.extract.doc_gen4.get_extraction_order",
            return_value=["mathlib"],
        ):
            with patch(
                "lean_explore.extract.doc_gen4._setup_workspace",
                return_value=("leanprover/lean4:v4.30.0-rc1", "v4.30.0-rc1"),
            ):
                with patch(
                    "lean_explore.extract.doc_gen4._clear_workspace_cache"
                ) as mock_clear:
                    with patch(
                        "lean_explore.extract.doc_gen4._run_lake_for_package"
                    ) as mock_run:
                        await run_doc_gen4(fresh=True)

        mock_clear.assert_called_once()
        mock_run.assert_called_once_with("mathlib", False)
