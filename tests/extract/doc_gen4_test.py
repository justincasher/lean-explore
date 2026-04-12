"""Tests for doc-gen4 workspace orchestration."""

from unittest.mock import MagicMock, patch

from lean_explore.extract.doc_gen4 import (
    _run_lake_for_package,
    _uses_sqlite_docgen,
    run_doc_gen4,
)
from lean_explore.extract.package_registry import PACKAGE_REGISTRY


class TestDocGen4VersionDetection:
    """Tests for doc-gen4 format detection by Lean version."""

    def test_uses_sqlite_docgen_from_rc2(self):
        """Test the SQLite cutoff at v4.29.0-rc2."""
        assert _uses_sqlite_docgen("leanprover/lean4:v4.29.0-rc2")
        assert _uses_sqlite_docgen("leanprover/lean4:v4.29.0")
        assert not _uses_sqlite_docgen("leanprover/lean4:v4.29.0-rc1")
        assert not _uses_sqlite_docgen("leanprover/lean4:v4.28.0")


class TestRunDocGen4FreshHandling:
    """Tests for fresh-cache handling in run_doc_gen4."""

    async def test_fresh_legacy_workspace_clears_cache(self):
        """Test that legacy doc-gen4 workspaces still clear the cache."""
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
        """Test that SQLite doc-gen4 workspaces keep the cache."""
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


class TestLakeBuildTargets:
    """Tests for explicit Lake build target selection."""

    def test_physlean_builds_wrapper_before_docs(self):
        """Test that PhysLean builds the wrapper target before doc generation."""
        with patch("lean_explore.extract.doc_gen4._run_lake_update_with_retry"):
            with patch("lean_explore.extract.doc_gen4.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
                with patch(
                    "lean_explore.extract.doc_gen4.subprocess.Popen"
                ) as mock_popen:
                    mock_process = MagicMock()
                    mock_process.stdout = iter([])
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process

                    _run_lake_for_package("physlean")

        assert mock_popen.call_args_list[0].args[0] == ["lake", "build", "PhysExtract"]
        assert mock_popen.call_args_list[1].args[0] == [
            "lake",
            "build",
            "PhysExtract:docs",
        ]


class TestPackageRegistry:
    """Tests for package registry metadata."""

    def test_physlean_uses_upstream_module_roots(self):
        """Test that PhysLean filters on the upstream module roots."""
        assert PACKAGE_REGISTRY["physlean"].module_prefixes == [
            "Physlib",
            "QuantumInfo",
        ]
