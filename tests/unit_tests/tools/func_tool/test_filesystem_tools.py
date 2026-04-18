# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/filesystem_tools.py"""

from pathlib import Path

from datus.tools.func_tool.filesystem_tools import FilesystemConfig, FilesystemFuncTool
from datus.tools.func_tool.fs_path_policy import PathZone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(root_path: str, *, current_node: str = "test_node") -> FilesystemFuncTool:
    return FilesystemFuncTool(root_path=root_path, current_node=current_node)


# ---------------------------------------------------------------------------
# FilesystemConfig
# ---------------------------------------------------------------------------


class TestFilesystemConfig:
    def test_default_root_path(self, tmp_path, monkeypatch):
        # Production call sites always pass an explicit ``root_path`` via
        # ``_make_filesystem_tool``; the ``os.getcwd()`` fallback only kicks
        # in for direct construction (tests, scripts). We chdir so the
        # assertion is deterministic regardless of where pytest was invoked.
        monkeypatch.chdir(tmp_path)
        cfg = FilesystemConfig()
        assert cfg.root_path == str(tmp_path)

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


# ---------------------------------------------------------------------------
# FilesystemFuncTool - _get_safe_path
# ---------------------------------------------------------------------------


class TestGetSafePath:
    """``_get_safe_path`` is the deprecated shim over ``classify_path``. It
    returns ``None`` for EXTERNAL/HIDDEN zones so legacy callers continue to
    see the original "out-of-sandbox" reject behavior.
    """

    def test_valid_relative_path(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool._get_safe_path("subdir/file.txt")
        assert result is not None
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_path_traversal_returns_none(self, tmp_path):
        # Escape via `..` lands in EXTERNAL → shim returns None for parity
        # with the pre-refactor contract. The tool itself would allow the
        # read (hook is responsible for the user prompt), but this shim
        # exists only for the handful of external callers.
        tool = _make_tool(str(tmp_path))
        assert tool._get_safe_path("../../etc/passwd") is None

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

    def test_read_file_path_traversal_classified_external(self, tmp_path):
        """``../`` escape lands in EXTERNAL. The tool does not hard-reject
        (the permission hook is responsible for asking the user), but the
        zone classifier must flag it so the hook can see it.
        """
        tool = _make_tool(str(tmp_path))
        resolved = tool._classify("../../etc/passwd")
        assert resolved.zone == PathZone.EXTERNAL

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

    def test_write_path_traversal_classified_external(self, tmp_path):
        """Write with ``../`` escape also classifies EXTERNAL — gating lives
        in the permission hook, not in the tool."""
        tool = _make_tool(str(tmp_path))
        resolved = tool._classify("../../evil.txt")
        assert resolved.zone == PathZone.EXTERNAL

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
# Zone-based visibility contract: .datus/ invisible except skills/memory
# whitelist; absolute-path reads allowed at the tool layer (permission hook
# is responsible for prompting the user outside the project).
# ---------------------------------------------------------------------------


class TestPathZones:
    """Covers the four classification zones on read/write/edit/glob."""

    def test_internal_relative_path_allowed(self, tmp_path):
        (tmp_path / "hello.md").write_text("hi")
        tool = _make_tool(str(tmp_path))
        resolved = tool._classify("hello.md")
        assert resolved.zone == PathZone.INTERNAL
        assert tool.read_file("hello.md").result == "hi"

    def test_hidden_dot_datus_invisible_on_read(self, tmp_path):
        secret = tmp_path / ".datus" / "sessions" / "x.db"
        secret.parent.mkdir(parents=True)
        secret.write_text("secret")
        tool = _make_tool(str(tmp_path))
        resolved = tool._classify(".datus/sessions/x.db")
        assert resolved.zone == PathZone.HIDDEN
        result = tool.read_file(".datus/sessions/x.db")
        # Same "File not found" whether the file exists or not.
        assert result.success == 0
        assert "file not found" in (result.error or "").lower()

    def test_hidden_dot_datus_invisible_on_write(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        result = tool.write_file(".datus/sessions/new.md", "payload")
        assert result.success == 0
        assert "file not found" in (result.error or "").lower()
        assert not (tmp_path / ".datus" / "sessions" / "new.md").exists()

    def test_whitelist_project_skills_readable_and_writable(self, tmp_path):
        tool = _make_tool(str(tmp_path))
        resolved = tool._classify(".datus/skills/my_skill/SKILL.md")
        assert resolved.zone == PathZone.WHITELIST
        assert tool.write_file(".datus/skills/my_skill/SKILL.md", "# skill\n").success == 1
        assert tool.read_file(".datus/skills/my_skill/SKILL.md").result == "# skill\n"

    def test_whitelist_per_node_memory_isolated(self, tmp_path):
        tool = _make_tool(str(tmp_path), current_node="gen_sql")
        own_zone = tool._classify(".datus/memory/gen_sql/MEMORY.md").zone
        other_zone = tool._classify(".datus/memory/chat/MEMORY.md").zone
        assert own_zone == PathZone.WHITELIST
        assert other_zone == PathZone.HIDDEN

    def test_current_node_none_degrades_memory_to_hidden(self, tmp_path):
        tool = FilesystemFuncTool(root_path=str(tmp_path), current_node=None)
        zone = tool._classify(".datus/memory/anything/MEMORY.md").zone
        assert zone == PathZone.HIDDEN

    def test_external_absolute_allowed_at_tool_layer(self, tmp_path):
        """Without a hook, the tool allows EXTERNAL reads. This documents the
        contract: the hook owns user confirmation, the tool owns visibility.
        """
        outside = tmp_path.parent / "outside.md"
        try:
            outside.write_text("out-of-project")
            # Fresh project root is an empty subdir so absolute access is external.
            project = tmp_path / "proj"
            project.mkdir()
            tool = _make_tool(str(project))
            zone = tool._classify(str(outside)).zone
            assert zone == PathZone.EXTERNAL
            assert tool.read_file(str(outside)).result == "out-of-project"
        finally:
            if outside.exists():
                outside.unlink()

    def test_glob_prunes_hidden_subtree(self, tmp_path):
        (tmp_path / "keep.md").write_text("visible")
        skill_dir = tmp_path / ".datus" / "skills" / "foo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("skill")
        hidden = tmp_path / ".datus" / "sessions"
        hidden.mkdir(parents=True)
        (hidden / "a.md").write_text("hidden")
        tool = _make_tool(str(tmp_path))
        names = {Path(p).name for p in tool.glob("**/*.md").result["files"]}
        assert "keep.md" in names
        assert "SKILL.md" in names
        assert "a.md" not in names

    def test_strict_rejects_external_read(self, tmp_path):
        """Strict mode fails closed on EXTERNAL: reads never touch the host fs.

        Used by API / claw where there's no broker to ask the user. We do
        not fall through to an ``exists()`` probe either, so a missing
        external path and an existing one produce the same rejection.
        """
        outside = tmp_path.parent / "strict_outside.md"
        outside.write_text("secret")
        project = tmp_path / "proj"
        project.mkdir()
        tool = FilesystemFuncTool(root_path=str(project), current_node="chat", strict=True)
        try:
            result = tool.read_file(str(outside))
            assert result.success == 0
            assert "not allowed in strict mode" in (result.error or "").lower()
            assert str(outside) in (result.error or "")
        finally:
            if outside.exists():
                outside.unlink()

    def test_strict_rejects_external_write(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        target = tmp_path / "elsewhere.md"
        tool = FilesystemFuncTool(root_path=str(project), current_node="chat", strict=True)
        result = tool.write_file(str(target), "payload")
        assert result.success == 0
        assert "not allowed in strict mode" in (result.error or "").lower()
        assert not target.exists()

    def test_strict_rejects_external_glob(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        (other / "x.md").write_text("x")
        project = tmp_path / "proj"
        project.mkdir()
        tool = FilesystemFuncTool(root_path=str(project), current_node="chat", strict=True)
        result = tool.glob("*.md", str(other))
        assert result.success == 0
        assert "not allowed in strict mode" in (result.error or "").lower()

    def test_strict_allows_internal_and_whitelist(self, tmp_path):
        """Strict mode must NOT break project-internal / whitelist access —
        otherwise the API surface could not read its own files."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "hello.md").write_text("hi")
        tool = FilesystemFuncTool(root_path=str(project), current_node="chat", strict=True)
        assert tool.read_file("hello.md").result == "hi"
        assert tool.write_file(".datus/skills/x/SKILL.md", "# skill\n").success == 1

    def test_glob_external_seed_returns_absolute_paths(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        (other / "x.md").write_text("body")
        project = tmp_path / "proj"
        project.mkdir()
        tool = _make_tool(str(project))
        result = tool.glob("*.md", str(other))
        assert result.success == 1
        assert result.result.get("external") is True
        files = result.result["files"]
        assert files
        assert all(Path(p).is_absolute() for p in files)
