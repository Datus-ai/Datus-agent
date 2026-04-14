# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/filesystem_tools.py"""

import os
from pathlib import Path
from unittest.mock import patch

from datus.cli.generation_hooks import make_kb_path_normalizer
from datus.tools.func_tool.filesystem_tools import EditOperation, FilesystemConfig, FilesystemFuncTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(root_path: str) -> FilesystemFuncTool:
    return FilesystemFuncTool(root_path=root_path)


# ---------------------------------------------------------------------------
# FilesystemConfig
# ---------------------------------------------------------------------------


class TestFilesystemConfig:
    def test_default_root_path(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if set
            os.environ.pop("FILESYSTEM_MCP_PATH", None)
            cfg = FilesystemConfig()
        # os.path.expanduser("~") is the only portable way to get the home directory
        # at test time — it matches the same call in FilesystemConfig's default_factory.
        assert cfg.root_path == os.path.expanduser("~")

    def test_explicit_root_path(self, tmp_path):
        cfg = FilesystemConfig(root_path=str(tmp_path))
        assert cfg.root_path == str(tmp_path)

    def test_default_allowed_extensions(self):
        cfg = FilesystemConfig()
        assert ".py" in cfg.allowed_extensions
        assert ".txt" in cfg.allowed_extensions
        assert ".json" in cfg.allowed_extensions

    def test_custom_allowed_extensions(self):
        cfg = FilesystemConfig(allowed_extensions=[".py"])
        assert cfg.allowed_extensions == [".py"]

    def test_env_var_sets_root_path(self, tmp_path):
        with patch.dict(os.environ, {"FILESYSTEM_MCP_PATH": str(tmp_path)}):
            cfg = FilesystemConfig()
        assert cfg.root_path == str(tmp_path)


# ---------------------------------------------------------------------------
# FilesystemFuncTool - _get_safe_path
# ---------------------------------------------------------------------------


class TestGetSafePath:
    def test_valid_relative_path(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool._get_safe_path("subdir/file.txt")
        assert result is not None
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_path_traversal_blocked(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        # Try to escape root via ..
        result = tool._get_safe_path("../../etc/passwd")
        assert result is None

    def test_dot_path_returns_root(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool._get_safe_path(".")
        assert result is not None
        assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# FilesystemFuncTool - _is_allowed_file
# ---------------------------------------------------------------------------


class TestIsAllowedFile:
    def test_allowed_extension(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        assert tool._is_allowed_file(Path("file.py")) is True

    def test_disallowed_extension(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        assert tool._is_allowed_file(Path("file.exe")) is False

    def test_no_extension_filter(self, tmp_path):
        tool = FilesystemFuncTool(root_path=str(tmp_path))
        tool.config.allowed_extensions = []
        assert tool._is_allowed_file(Path("file.exe")) is True


# ---------------------------------------------------------------------------
# FilesystemFuncTool - read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("hello.txt")
        assert result.success == 1
        assert result.result == "hello world"

    def test_read_nonexistent_file(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("nonexistent.txt")
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_read_directory_as_file(self, tmp_path):
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("mydir")
        assert result.success == 0
        assert "not a file" in result.error.lower()

    def test_read_disallowed_extension(self, tmp_path):
        f = tmp_path / "binary.exe"
        f.write_bytes(b"\x00\x01\x02")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("binary.exe")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_read_file_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100)
        tool = _make_tool(str(tmp_path))
        tool.config.max_file_size = 10
        result = tool.read_file("big.txt")
        assert result.success == 0
        assert "too large" in result.error.lower()

    def test_read_file_unicode_error(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes(b"\xff\xfe\x00\x01")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("data.txt")
        assert result.success == 0

    def test_read_file_path_traversal_blocked(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("../../etc/passwd")
        assert result.success == 0


# ---------------------------------------------------------------------------
# FilesystemFuncTool - read_multiple_files
# ---------------------------------------------------------------------------


class TestReadMultipleFiles:
    def test_read_multiple_success(self, tmp_path):
        (tmp_path / "a.txt").write_text("content_a")
        (tmp_path / "b.txt").write_text("content_b")
        tool = _make_tool(str(tmp_path))
        result = tool.read_multiple_files(["a.txt", "b.txt"])
        assert result.success == 1
        assert result.result["a.txt"] == "content_a"
        assert result.result["b.txt"] == "content_b"

    def test_read_multiple_partial_missing(self, tmp_path):
        (tmp_path / "exists.txt").write_text("hi")
        tool = _make_tool(str(tmp_path))
        result = tool.read_multiple_files(["exists.txt", "missing.txt"])
        assert result.success == 1
        assert result.result["exists.txt"] == "hi"
        assert "not found" in result.result["missing.txt"].lower()

    def test_read_multiple_not_a_file(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        tool = _make_tool(str(tmp_path))
        result = tool.read_multiple_files(["mydir"])
        assert "not a file" in result.result["mydir"].lower()

    def test_read_multiple_disallowed_extension(self, tmp_path):
        (tmp_path / "file.exe").write_bytes(b"binary")
        tool = _make_tool(str(tmp_path))
        result = tool.read_multiple_files(["file.exe"])
        assert "not allowed" in result.result["file.exe"].lower()


# ---------------------------------------------------------------------------
# FilesystemFuncTool - write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_new_file(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.write_file("newfile.txt", "some content")
        assert result.success == 1
        assert (tmp_path / "newfile.txt").read_text() == "some content"

    def test_write_creates_parent_dirs(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.write_file("subdir/nested/file.txt", "data")
        assert result.success == 1
        assert (tmp_path / "subdir" / "nested" / "file.txt").exists()

    def test_write_disallowed_extension(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.write_file("file.exe", "binary")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_write_path_traversal_blocked(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.write_file("../../evil.txt", "evil")
        assert result.success == 0

    def test_write_overwrites_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        tool = _make_tool(str(tmp_path))
        result = tool.write_file("existing.txt", "new content")
        assert result.success == 1
        assert f.read_text() == "new content"


# ---------------------------------------------------------------------------
# FilesystemFuncTool - edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_edit_success(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello world\nfoo bar")
        tool = _make_tool(str(tmp_path))
        edits = [EditOperation(oldText="hello world", newText="goodbye world")]
        result = tool.edit_file("code.py", edits)
        assert result.success == 1
        assert "goodbye world" in f.read_text()

    def test_edit_with_dict_list(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("alpha beta gamma")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("code.py", [{"oldText": "alpha", "newText": "delta"}])
        assert result.success == 1
        assert "delta" in f.read_text()

    def test_edit_with_json_string(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo bar")
        tool = _make_tool(str(tmp_path))
        import json

        edits_json = json.dumps([{"oldText": "foo", "newText": "baz"}])
        result = tool.edit_file("code.py", edits_json)
        assert result.success == 1
        assert "baz" in f.read_text()

    def test_edit_invalid_json_string(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("code.py", "not valid json{{")
        assert result.success == 0
        assert "invalid json" in result.error.lower()

    def test_edit_text_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo bar")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("code.py", [EditOperation(oldText="nonexistent", newText="x")])
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_edit_file_not_found(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("missing.py", [EditOperation(oldText="x", newText="y")])
        assert result.success == 0

    def test_edit_disallowed_extension(self, tmp_path):
        f = tmp_path / "file.exe"
        f.write_bytes(b"binary")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("file.exe", [EditOperation(oldText="x", newText="y")])
        assert result.success == 0

    def test_edit_no_edits_applied(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("content")
        tool = _make_tool(str(tmp_path))
        # Empty oldText means condition not met
        result = tool.edit_file("code.py", [EditOperation(oldText="", newText="x")])
        assert result.success == 0
        assert "no edits" in result.error.lower()


# ---------------------------------------------------------------------------
# FilesystemFuncTool - create_directory
# ---------------------------------------------------------------------------


class TestCreateDirectory:
    def test_create_new_directory(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.create_directory("newdir")
        assert result.success == 1
        assert (tmp_path / "newdir").is_dir()

    def test_create_nested_directories(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.create_directory("a/b/c")
        assert result.success == 1
        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_create_existing_directory_ok(self, tmp_path):
        (tmp_path / "existing").mkdir()
        tool = _make_tool(str(tmp_path))
        result = tool.create_directory("existing")
        assert result.success == 1

    def test_create_directory_path_traversal(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.create_directory("../../evil_dir")
        assert result.success == 0


# ---------------------------------------------------------------------------
# FilesystemFuncTool - list_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_list_root_directory(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "subdir").mkdir()
        tool = _make_tool(str(tmp_path))
        result = tool.list_directory(".")
        assert result.success == 1
        names = [item["name"] for item in result.result]
        assert "file1.txt" in names
        assert "subdir" in names

    def test_list_directory_types(self, tmp_path):
        (tmp_path / "myfile.txt").write_text("x")
        (tmp_path / "mydir").mkdir()
        tool = _make_tool(str(tmp_path))
        result = tool.list_directory(".")
        items = {item["name"]: item["type"] for item in result.result}
        assert items["myfile.txt"] == "file"
        assert items["mydir"] == "directory"

    def test_list_nonexistent_directory(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.list_directory("nonexistent")
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_list_file_as_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.list_directory("file.txt")
        assert result.success == 0
        assert "not a directory" in result.error.lower()


# ---------------------------------------------------------------------------
# FilesystemFuncTool - directory_tree
# ---------------------------------------------------------------------------


class TestDirectoryTree:
    def test_tree_basic(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.py").write_text("code")
        tool = _make_tool(str(tmp_path))
        result = tool.directory_tree(".")
        assert result.success == 1
        assert "file.txt" in result.result
        assert "subdir" in result.result

    def test_tree_max_depth_zero(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "deep.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.directory_tree(".", max_depth=0)
        assert result.success == 1
        assert "max depth" in result.result

    def test_tree_nonexistent(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.directory_tree("nonexistent")
        assert result.success == 0

    def test_tree_file_as_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.directory_tree("file.txt")
        assert result.success == 0

    def test_tree_max_items_truncated(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.directory_tree(".", max_items=3)
        assert result.success == 1
        assert "truncated" in result.result.lower() or "max items" in result.result

    def test_tree_unlimited_depth(self, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        d.mkdir(parents=True)
        (d / "leaf.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.directory_tree(".", max_depth=-1)
        assert result.success == 1
        assert "leaf.txt" in result.result


# ---------------------------------------------------------------------------
# FilesystemFuncTool - move_file
# ---------------------------------------------------------------------------


class TestMoveFile:
    def test_move_file_success(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("content")
        tool = _make_tool(str(tmp_path))
        result = tool.move_file("source.txt", "dest.txt")
        assert result.success == 1
        assert not src.exists()
        assert (tmp_path / "dest.txt").exists()

    def test_move_source_not_found(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.move_file("nonexistent.txt", "dest.txt")
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_move_invalid_destination(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("content")
        tool = _make_tool(str(tmp_path))
        result = tool.move_file("source.txt", "../../evil.txt")
        assert result.success == 0

    def test_move_rename_file(self, tmp_path):
        src = tmp_path / "old_name.txt"
        src.write_text("data")
        tool = _make_tool(str(tmp_path))
        result = tool.move_file("old_name.txt", "new_name.txt")
        assert result.success == 1
        assert (tmp_path / "new_name.txt").read_text() == "data"


# ---------------------------------------------------------------------------
# FilesystemFuncTool - search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    def test_search_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("more code")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*.py")
        assert result.success == 1
        py_files = [Path(p).name for p in result.result["files"]]
        assert "a.py" in py_files

    def test_search_with_glob_star(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.py").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "**/*.py")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "deep.py" in names

    def test_search_nonexistent_directory(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.search_files("nonexistent", "*.py")
        assert result.success == 0

    def test_search_file_as_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files("file.txt", "*.py")
        assert result.success == 0

    def test_search_with_exclude_patterns(self, tmp_path):
        (tmp_path / "keep.py").write_text("x")
        (tmp_path / "exclude.py").write_text("y")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*.py", exclude_patterns=["exclude.py"])
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "keep.py" in names
        assert "exclude.py" not in names

    def test_search_ignores_git_directory(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / ".git" / "objects").mkdir()
        (tmp_path / ".git" / "objects" / "pack.py").write_text("x")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "**/*")
        assert result.success == 1
        paths = [Path(p).name for p in result.result["files"]]
        assert "main.py" in paths
        assert "config" not in paths
        assert "pack.py" not in paths

    def test_search_empty_results(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*.xyz_nonexistent")
        assert result.success == 1
        assert result.result["files"] == []
        assert result.result["truncated"] is False

    def test_search_max_results_truncates(self, tmp_path):
        """Results are truncated when exceeding max_results."""
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text(f"code {i}")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*.py", max_results=3)
        assert result.success == 1
        assert isinstance(result.result, dict)
        assert len(result.result["files"]) == 3
        assert result.result["truncated"] is True

    def test_search_max_results_no_truncate_when_under_limit(self, tmp_path):
        """No truncation when results fit within max_results."""
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.py").write_text("code")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*.py", max_results=10)
        assert result.success == 1
        assert isinstance(result.result, dict)
        assert len(result.result["files"]) == 2
        assert result.result["truncated"] is False

    def test_search_excludes_gitignore_patterns(self, tmp_path):
        """Patterns from .gitignore are excluded from search results."""
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n__pycache__\n")
        (tmp_path / "real.py").write_text("code")
        (tmp_path / "debug.log").write_text("log data")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.py").write_text("built")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("bytecode")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "real.py" in names
        assert "debug.log" not in names
        assert "output.py" not in names
        assert "cached.py" not in names
        # Trailing-slash entry should also exclude the directory itself
        assert "build" not in names

    def test_search_excludes_git_dir_always(self, tmp_path):
        """.git directory is always excluded even without .gitignore."""
        (tmp_path / "real.py").write_text("code")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "real.py" in names
        assert "config" not in names

    def test_search_fallback_excludes_without_gitignore(self, tmp_path):
        """Without .gitignore, fallback excludes (.git, __pycache__, node_modules) apply."""
        (tmp_path / "real.py").write_text("code")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("bytecode")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("js")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "real.py" in names
        assert "cached.py" not in names
        assert "pkg.js" not in names

    def test_search_unlimited_with_negative_max(self, tmp_path):
        """max_results=-1 returns all results without truncation."""
        for i in range(5):
            (tmp_path / f"file_{i}.txt").write_text(f"text {i}")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*.txt", max_results=-1)
        assert result.success == 1
        assert isinstance(result.result, dict)
        assert len(result.result["files"]) == 5
        assert result.result["truncated"] is False

    def test_search_negation_pattern_not_excluded(self, tmp_path):
        """Gitignore negation patterns (!) should not cause exclusion."""
        (tmp_path / ".gitignore").write_text("*.log\n!important.log\n")
        (tmp_path / "debug.log").write_text("debug")
        (tmp_path / "important.log").write_text("keep this")
        (tmp_path / "code.py").write_text("code")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "*")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "code.py" in names
        # important.log is excluded by *.log (negation not fully supported),
        # but crucially it must NOT be excluded by the ! line itself
        assert "debug.log" not in names

    def test_search_max_results_nested_dirs(self, tmp_path):
        """Early termination propagates correctly through nested directories."""
        for i in range(5):
            d = tmp_path / f"dir_{i}"
            d.mkdir()
            for j in range(5):
                (d / f"file_{j}.py").write_text(f"code {i}-{j}")
        tool = _make_tool(str(tmp_path))
        result = tool.search_files(".", "**/*.py", max_results=3)
        assert result.success == 1
        assert result.result["truncated"] is True
        # With propagation fix, should be exactly max_results
        assert len(result.result["files"]) == 3


# ---------------------------------------------------------------------------
# KB path_normalizer round-trip: write/read/edit with the silent prefix
# normalizer (covers the knowledge_base_home re-scope refactor contract).
# ---------------------------------------------------------------------------


class _StubCfg:
    def __init__(self, ns: str):
        self.current_namespace = ns


class TestFilesystemFuncToolKbNormalizerRoundTrip:
    """Contract tests for FilesystemFuncTool + KB path_normalizer."""

    def _make(self, tmp_path: Path, kind: str, namespace: str):
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        tool = FilesystemFuncTool(
            root_path=str(kb_root),
            path_normalizer=make_kb_path_normalizer(_StubCfg(namespace), default_kind=kind),
        )
        return tool, kb_root

    def test_naked_filename_lands_in_typed_subdir(self, tmp_path):
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        result = tool.write_file("orders.yml", "id: orders\n", file_type="semantic_model")
        assert result.success == 1
        on_disk = kb_root / "semantic_models" / "school_db" / "orders.yml"
        assert on_disk.is_file()
        assert on_disk.read_text() == "id: orders\n"

    def test_read_naked_filename_after_naked_write(self, tmp_path):
        """LLM forgets prefix on both write and subsequent read — both must succeed."""
        tool, _ = self._make(tmp_path, "semantic", "school_db")
        tool.write_file("orders.yml", "payload\n", file_type="semantic_model")
        assert tool.read_file("orders.yml").result == "payload\n"

    def test_read_with_full_prefix_after_naked_write(self, tmp_path):
        tool, _ = self._make(tmp_path, "semantic", "school_db")
        tool.write_file("orders.yml", "payload\n", file_type="semantic_model")
        assert tool.read_file("semantic_models/school_db/orders.yml").result == "payload\n"

    def test_read_multiple_files_mixed_naked_and_prefixed(self, tmp_path):
        """read_multiple_files applies the same normalizer to each path in the batch."""
        tool, _ = self._make(tmp_path, "semantic", "school_db")
        tool.write_file("orders.yml", "A\n", file_type="semantic_model")
        tool.write_file("customers.yml", "B\n", file_type="semantic_model")

        result = tool.read_multiple_files(["orders.yml", "semantic_models/school_db/customers.yml"])
        assert result.success == 1
        # Keyed by the caller's raw path so the LLM can correlate the response.
        assert result.result["orders.yml"] == "A\n"
        assert result.result["semantic_models/school_db/customers.yml"] == "B\n"

    def test_edit_file_after_naked_write(self, tmp_path):
        tool, _ = self._make(tmp_path, "sql_summary", "school_db")
        tool.write_file("q_001.yaml", "name: original\n", file_type="sql_summary")
        edit_result = tool.edit_file("q_001.yaml", [{"oldText": "original", "newText": "edited"}])
        assert edit_result.success == 1, edit_result.error
        assert tool.read_file("q_001.yaml").result == "name: edited\n"

    def test_write_with_full_prefix_does_not_double_prefix(self, tmp_path):
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        tool.write_file(
            "semantic_models/school_db/customers.yml",
            "id: customers\n",
            file_type="semantic_model",
        )
        assert (kb_root / "semantic_models" / "school_db" / "customers.yml").is_file()
        assert not (kb_root / "semantic_models" / "school_db" / "semantic_models").exists()

    def test_cross_kind_read_works(self, tmp_path):
        """A semantic-mode tool must still read peer sql_summaries by full path."""
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        peer = kb_root / "sql_summaries" / "school_db" / "q_001.yaml"
        peer.parent.mkdir(parents=True)
        peer.write_text("peer content\n")
        assert tool.read_file("sql_summaries/school_db/q_001.yaml").result == "peer content\n"

    def test_cross_kind_write_is_rejected_under_strict(self, tmp_path):
        """Writing to a peer kind's subdir must fail closed (sandbox enforcement)."""
        tool, _ = self._make(tmp_path, "semantic", "school_db")
        result = tool.write_file(
            "sql_summaries/school_db/q_001.yaml",
            "bad\n",
            file_type="semantic_model",
        )
        assert result.success == 0
        assert "not allowed" in (result.error or "").lower()

    def test_normalizer_exception_fails_write(self, tmp_path):
        """If the path_normalizer raises, write_file must fail instead of
        silently landing the mutation at an unnormalized path."""

        def _boom(path, file_type, *, strict_kind=False):
            raise RuntimeError("normalizer error")

        tool = FilesystemFuncTool(root_path=str(tmp_path), path_normalizer=_boom)
        result = tool.write_file("orders.yml", "data\n")
        assert result.success == 0
        assert "normalization failed" in (result.error or "").lower()
        # And no file should have been created.
        assert not any(tmp_path.rglob("orders.yml"))

    def test_normalizer_exception_does_not_fail_read(self, tmp_path):
        """Reads stay lax: on normalizer error, fall back to the original path
        and let the sandbox check fail naturally."""

        def _boom(path, file_type, *, strict_kind=False):
            raise RuntimeError("normalizer error")

        (tmp_path / "orders.yml").write_text("data\n")
        tool = FilesystemFuncTool(root_path=str(tmp_path), path_normalizer=_boom)
        result = tool.read_file("orders.yml")
        assert result.success == 1
        assert result.result == "data\n"
