# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/filesystem_tools.py"""

import os
from pathlib import Path
from unittest.mock import patch

from datus.cli.generation_hooks import make_kb_path_normalizer
from datus.tools.func_tool.filesystem_tools import FilesystemConfig, FilesystemFuncTool

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

    def test_read_file_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("lines.txt", offset=2, limit=2)
        assert result.success == 1
        assert "2: line2" in result.result
        assert "3: line3" in result.result
        assert "line1" not in result.result
        assert "line4" not in result.result

    def test_read_file_with_offset_only(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("lines.txt", offset=2)
        assert result.success == 1
        assert "2: line2" in result.result
        assert "3: line3" in result.result
        assert "1: line1" not in result.result

    def test_read_file_with_limit_only(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("lines.txt", limit=2)
        assert result.success == 1
        assert "1: line1" in result.result
        assert "2: line2" in result.result
        assert "line3" not in result.result

    def test_read_file_no_offset_limit_returns_full(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3")
        tool = _make_tool(str(tmp_path))
        result = tool.read_file("lines.txt")
        assert result.success == 1
        assert result.result == "line1\nline2\nline3"


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
        result = tool.edit_file("code.py", "hello world", "goodbye world")
        assert result.success == 1
        assert "goodbye world" in f.read_text()

    def test_edit_text_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo bar")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("code.py", "nonexistent", "x")
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_edit_multiple_matches_rejected(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello hello hello")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("code.py", "hello", "bye")
        assert result.success == 0
        assert "3 times" in result.error

    def test_edit_empty_old_string_rejected(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("content")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("code.py", "", "x")
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_edit_file_not_found(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("missing.py", "x", "y")
        assert result.success == 0

    def test_edit_disallowed_extension(self, tmp_path):
        f = tmp_path / "file.exe"
        f.write_bytes(b"binary")
        tool = _make_tool(str(tmp_path))
        result = tool.edit_file("file.exe", "x", "y")
        assert result.success == 0


# ---------------------------------------------------------------------------
# FilesystemFuncTool - glob
# ---------------------------------------------------------------------------


class TestGlobSearch:
    def test_glob_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("code")
        (tmp_path / "b.txt").write_text("text")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("more code")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*.py")
        assert result.success == 1
        py_files = [Path(p).name for p in result.result["files"]]
        assert "a.py" in py_files

    def test_glob_with_globstar(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.py").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("**/*.py")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "deep.py" in names

    def test_glob_nonexistent_directory(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*.py", "nonexistent")
        assert result.success == 0

    def test_glob_file_as_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*.py", "file.txt")
        assert result.success == 0

    def test_glob_ignores_git_directory(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("**/*")
        assert result.success == 1
        paths = [Path(p).name for p in result.result["files"]]
        assert "main.py" in paths
        assert "config" not in paths

    def test_glob_empty_results(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*.xyz_nonexistent")
        assert result.success == 1
        assert result.result["files"] == []
        assert result.result["truncated"] is False

    def test_glob_excludes_gitignore_patterns(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n__pycache__\n")
        (tmp_path / "real.py").write_text("code")
        (tmp_path / "debug.log").write_text("log data")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.py").write_text("built")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("bytecode")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "real.py" in names
        assert "debug.log" not in names
        assert "output.py" not in names
        assert "cached.py" not in names

    def test_glob_with_path_param(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.py").write_text("x")
        (tmp_path / "other.py").write_text("y")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*.py", "sub")
        assert result.success == 1
        names = [Path(p).name for p in result.result["files"]]
        assert "file.py" in names
        assert "other.py" not in names

    def test_glob_truncation(self, tmp_path):
        """Results are truncated when exceeding max_results (200)."""
        for i in range(5):
            (tmp_path / f"file_{i}.py").write_text(f"code {i}")
        tool = _make_tool(str(tmp_path))
        result = tool.glob("*.py")
        assert result.success == 1
        assert result.result["truncated"] is False
        assert len(result.result["files"]) == 5


# ---------------------------------------------------------------------------
# FilesystemFuncTool - grep
# ---------------------------------------------------------------------------


class TestGrep:
    def test_grep_finds_pattern(self, tmp_path):
        (tmp_path / "hello.py").write_text("def hello():\n    return 'world'\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("hello")
        assert result.success == 1
        assert len(result.result["matches"]) >= 1
        assert result.result["matches"][0]["content"] == "def hello():"

    def test_grep_regex_pattern(self, tmp_path):
        (tmp_path / "code.py").write_text("foo123\nbar456\nfoo789\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep(r"foo\d+")
        assert result.success == 1
        assert len(result.result["matches"]) == 2

    def test_grep_case_insensitive(self, tmp_path):
        (tmp_path / "data.txt").write_text("Hello\nhello\nHELLO\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("hello", case_sensitive=False)
        assert result.success == 1
        assert len(result.result["matches"]) == 3

    def test_grep_case_sensitive_default(self, tmp_path):
        (tmp_path / "data.txt").write_text("Hello\nhello\nHELLO\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("hello")
        assert result.success == 1
        assert len(result.result["matches"]) == 1

    def test_grep_with_include_filter(self, tmp_path):
        (tmp_path / "code.py").write_text("target line\n")
        (tmp_path / "data.txt").write_text("target line\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("target", include="*.py")
        assert result.success == 1
        files = [m["file"] for m in result.result["matches"]]
        assert any("code.py" in f for f in files)
        assert not any("data.txt" in f for f in files)

    def test_grep_skips_binary_files(self, tmp_path):
        (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00\x01target\x00")
        (tmp_path / "text.txt").write_text("target line\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("target")
        assert result.success == 1
        files = [m["file"] for m in result.result["matches"]]
        assert any("text.txt" in f for f in files)

    def test_grep_invalid_regex(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.grep("[invalid")
        assert result.success == 0
        assert "invalid regex" in result.error.lower()

    def test_grep_nonexistent_directory(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.grep("pattern", "nonexistent")
        assert result.success == 0

    def test_grep_returns_line_numbers(self, tmp_path):
        (tmp_path / "code.py").write_text("line1\ntarget\nline3\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("target")
        assert result.success == 1
        assert result.result["matches"][0]["line"] == 2

    def test_grep_respects_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "code.py").write_text("target\n")
        (tmp_path / "debug.log").write_text("target\n")
        tool = _make_tool(str(tmp_path))
        result = tool.grep("target")
        assert result.success == 1
        files = [m["file"] for m in result.result["matches"]]
        assert any("code.py" in f for f in files)
        assert not any("debug.log" in f for f in files)

    def test_grep_truncation(self, tmp_path):
        (tmp_path / "big.txt").write_text("\n".join([f"match line {i}" for i in range(150)]))
        tool = _make_tool(str(tmp_path))
        result = tool.grep("match")
        assert result.success == 1
        assert result.result["truncated"] is True
        assert len(result.result["matches"]) == 100


# ---------------------------------------------------------------------------
# FilesystemFuncTool - available_tools
# ---------------------------------------------------------------------------


class TestAvailableTools:
    def test_available_tools_returns_five(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        tools = tool.available_tools()
        names = [t.name for t in tools]
        assert set(names) == {"read_file", "write_file", "edit_file", "glob", "grep"}

    def test_available_tools_count(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        assert len(tool.available_tools()) == 5


# ---------------------------------------------------------------------------
# KB path_normalizer round-trip: write/read/edit with the silent prefix
# normalizer (covers the project-scoped ``subject/`` layout contract).
# ---------------------------------------------------------------------------


class TestFilesystemFuncToolKbNormalizerRoundTrip:
    """Contract tests for FilesystemFuncTool + KB path_normalizer."""

    def _make(self, tmp_path: Path, kind: str):
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        tool = FilesystemFuncTool(
            root_path=str(kb_root),
            path_normalizer=make_kb_path_normalizer(default_kind=kind),
        )
        return tool, kb_root

    def test_naked_filename_lands_in_typed_subdir(self, tmp_path):
        tool, kb_root = self._make(tmp_path, "semantic")
        result = tool.write_file("orders.yml", "id: orders\n", file_type="semantic_model")
        assert result.success == 1
        on_disk = kb_root / "semantic_models" / "orders.yml"
        assert on_disk.is_file()
        assert on_disk.read_text() == "id: orders\n"

    def test_read_naked_filename_after_naked_write(self, tmp_path):
        """LLM forgets prefix on both write and subsequent read — both must succeed."""
        tool, _ = self._make(tmp_path, "semantic")
        tool.write_file("orders.yml", "payload\n", file_type="semantic_model")
        assert tool.read_file("orders.yml").result == "payload\n"

    def test_read_with_full_prefix_after_naked_write(self, tmp_path):
        tool, _ = self._make(tmp_path, "semantic")
        tool.write_file("orders.yml", "payload\n", file_type="semantic_model")
        assert tool.read_file("semantic_models/orders.yml").result == "payload\n"

    def test_edit_file_after_naked_write(self, tmp_path):
        tool, _ = self._make(tmp_path, "sql_summary")
        tool.write_file("q_001.yaml", "name: original\n", file_type="sql_summary")
        edit_result = tool.edit_file("q_001.yaml", "original", "edited")
        assert edit_result.success == 1, edit_result.error
        assert tool.read_file("q_001.yaml").result == "name: edited\n"

    def test_write_with_full_prefix_does_not_double_prefix(self, tmp_path):
        tool, kb_root = self._make(tmp_path, "semantic")
        tool.write_file(
            "semantic_models/customers.yml",
            "id: customers\n",
            file_type="semantic_model",
        )
        assert (kb_root / "semantic_models" / "customers.yml").is_file()
        assert not (kb_root / "semantic_models" / "semantic_models").exists()

    def test_cross_kind_read_works(self, tmp_path):
        """A semantic-mode tool must still read peer sql_summaries by full path."""
        tool, kb_root = self._make(tmp_path, "semantic")
        peer = kb_root / "sql_summaries" / "q_001.yaml"
        peer.parent.mkdir(parents=True)
        peer.write_text("peer content\n")
        assert tool.read_file("sql_summaries/q_001.yaml").result == "peer content\n"

    def test_cross_kind_write_is_rejected_under_strict(self, tmp_path):
        """Writing to a peer kind's subdir must fail closed (sandbox enforcement)."""
        tool, _ = self._make(tmp_path, "semantic")
        result = tool.write_file(
            "sql_summaries/q_001.yaml",
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
