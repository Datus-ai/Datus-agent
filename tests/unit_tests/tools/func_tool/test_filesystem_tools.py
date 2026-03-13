# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CI-level tests for FilesystemFuncTool path resolution.

Tests _get_safe_path sandbox enforcement and absolute/relative path handling
in read_file and read_multiple_files.
"""

import os
import tempfile

import pytest

from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with test files."""
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested content", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fs_tool(workspace):
    """Create a FilesystemFuncTool rooted at workspace."""
    return FilesystemFuncTool(root_path=str(workspace))


# ── _get_safe_path sandbox tests ──


@pytest.mark.ci
class TestGetSafePath:
    """Test that _get_safe_path enforces sandbox boundaries."""

    def test_relative_path_inside_sandbox(self, fs_tool, workspace):
        result = fs_tool._get_safe_path("hello.txt")
        assert result == (workspace / "hello.txt").resolve()

    def test_relative_path_nested(self, fs_tool, workspace):
        result = fs_tool._get_safe_path("sub/nested.txt")
        assert result == (workspace / "sub" / "nested.txt").resolve()

    def test_traversal_blocked(self, fs_tool):
        result = fs_tool._get_safe_path("../../../etc/passwd")
        assert result is None

    def test_absolute_path_rejected(self, fs_tool):
        """Absolute paths should not resolve through _get_safe_path."""
        result = fs_tool._get_safe_path("/etc/hosts")
        assert result is None

    def test_prefix_collision_blocked(self, fs_tool, workspace):
        """A path whose prefix matches root but is a different directory must be rejected.

        e.g. root=/workspace/foo, target=/workspace/foobar should fail.
        """
        # Create a sibling directory with a name that is a prefix extension
        sibling = workspace.parent / (workspace.name + "bar")
        sibling.mkdir(exist_ok=True)
        try:
            (sibling / "secret.txt").write_text("secret", encoding="utf-8")
            # Attempt traversal to the sibling
            relative = os.path.relpath(sibling / "secret.txt", workspace)
            result = fs_tool._get_safe_path(relative)
            assert result is None
        finally:
            (sibling / "secret.txt").unlink(missing_ok=True)
            sibling.rmdir()

    def test_nonexistent_relative_path(self, fs_tool, workspace):
        """Non-existent relative path should still resolve (file may be created later)."""
        result = fs_tool._get_safe_path("new_file.txt")
        assert result == (workspace / "new_file.txt").resolve()


# ── read_file path handling tests ──


@pytest.mark.ci
class TestReadFilePathHandling:
    """Test read_file supports both absolute and relative paths."""

    def test_read_relative_path(self, fs_tool):
        result = fs_tool.read_file("hello.txt")
        assert result.success == 1
        assert result.result == "hello world"

    def test_read_absolute_path(self, fs_tool, workspace):
        abs_path = str(workspace / "hello.txt")
        result = fs_tool.read_file(abs_path)
        assert result.success == 1
        assert result.result == "hello world"

    def test_read_absolute_path_outside_sandbox(self):
        """Absolute paths outside sandbox are allowed for read (if they exist)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("external content")
            ext_path = f.name
        try:
            fs_tool = FilesystemFuncTool(root_path="/nonexistent_root")
            result = fs_tool.read_file(ext_path)
            assert result.success == 1
            assert result.result == "external content"
        finally:
            os.unlink(ext_path)

    def test_read_nonexistent_absolute_path(self, fs_tool):
        result = fs_tool.read_file("/tmp/nonexistent_file_xyz_12345.txt")
        assert result.success == 0
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_read_nonexistent_relative_path(self, fs_tool):
        result = fs_tool.read_file("no_such_file.txt")
        assert result.success == 0

    def test_read_traversal_blocked(self, fs_tool):
        result = fs_tool.read_file("../../../etc/passwd")
        assert result.success == 0


# ── read_multiple_files path handling tests ──


@pytest.mark.ci
class TestReadMultipleFilesPathHandling:
    """Test read_multiple_files supports both absolute and relative paths."""

    def test_read_mixed_paths(self, fs_tool, workspace):
        abs_path = str(workspace / "sub" / "nested.txt")
        result = fs_tool.read_multiple_files(["hello.txt", abs_path])
        assert result.success == 1
        assert result.result["hello.txt"] == "hello world"
        assert result.result[abs_path] == "nested content"

    def test_read_multiple_with_missing(self, fs_tool):
        result = fs_tool.read_multiple_files(["hello.txt", "missing.txt"])
        assert result.success == 1
        assert result.result["hello.txt"] == "hello world"
        assert "not found" in result.result["missing.txt"].lower()


# ── write_file sandbox enforcement tests ──


@pytest.mark.ci
class TestWriteFileSandbox:
    """Test that write_file always enforces sandbox."""

    def test_write_relative_path(self, fs_tool, workspace):
        result = fs_tool.write_file("new.txt", "new content")
        assert result.success == 1
        assert (workspace / "new.txt").read_text(encoding="utf-8") == "new content"

    def test_write_absolute_path_blocked(self, fs_tool):
        """Write to absolute path outside sandbox must be blocked."""
        result = fs_tool.write_file("/tmp/should_not_write.txt", "bad")
        assert result.success == 0

    def test_write_traversal_blocked(self, fs_tool):
        result = fs_tool.write_file("../../escape.txt", "bad")
        assert result.success == 0
