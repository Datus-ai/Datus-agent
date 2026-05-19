# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.tools.func_tool.memory_filesystem_tools``.

Pins the contract that :class:`MemoryFilesystemFuncTool` mirrors the LLM-facing surface of
:class:`FilesystemFuncTool` for the read operations the ``ask_*`` follow-up
subagents actually use:

* ``read_file`` returns the same shape (success/error/result, optional line
  slicing in ``"N: line"`` form, size cap parity).
* ``glob`` and ``grep`` return the same result-dict shape so the prompt
  template / LLM tool surface doesn't need to branch on backend.
* The bundle is read-only; write/edit/delete are deliberately absent.
"""

from __future__ import annotations

from datus.tools.func_tool.memory_filesystem_tools import MemoryFilesystemFuncTool


def _bundle() -> dict:
    """Return a small mixed-content bundle resembling a published report."""
    return {
        "manifest.json": '{"slug": "demo", "name": "Demo"}',
        "analysis/intent.md": "## intent\ninvestigate Q3 anomalies\n",
        "analysis/insights.json": '{"items": []}',
        "queries/total_sales.sql": "SELECT SUM(amount) FROM orders;",
        "queries/total_sales.json": '{"rows": [[42]]}',
        "queries/by_region.sql": "SELECT region, SUM(amount) FROM orders GROUP BY 1;",
        "render/app.jsx": "export default function App() { return <div/>; }",
    }


# --------------------------------------------------------------------------- #
# read_file                                                                   #
# --------------------------------------------------------------------------- #


class TestReadFile:
    def test_returns_full_content(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.read_file("queries/total_sales.sql")
        assert res.success == 1
        assert res.result == "SELECT SUM(amount) FROM orders;"

    def test_path_normalization_dot_slash(self):
        """``./prefix`` is normalized away — matches what the LLM sometimes emits."""
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.read_file("./manifest.json")
        assert res.success == 1
        assert "Demo" in res.result

    def test_path_normalization_leading_slash(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.read_file("/manifest.json")
        assert res.success == 1
        assert "Demo" in res.result

    def test_missing_file_returns_error(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.read_file("queries/does_not_exist.sql")
        assert res.success == 0
        assert "not found" in (res.error or "").lower()

    def test_offset_limit_returns_numbered_slice(self):
        fs = MemoryFilesystemFuncTool(
            {
                "notes.md": "line1\nline2\nline3\nline4\nline5\n",
            }
        )
        res = fs.read_file("notes.md", offset=2, limit=2)
        assert res.success == 1
        # Same "N: content" shape as the disk-backed tool.
        assert res.result == "2: line2\n3: line3"

    def test_offset_only_reads_to_end(self):
        fs = MemoryFilesystemFuncTool({"x.md": "a\nb\nc\n"})
        res = fs.read_file("x.md", offset=2)
        assert res.success == 1
        assert res.result == "2: b\n3: c\n4: "

    def test_full_read_size_cap(self):
        """Files over the per-read cap fail with a directive to use slicing."""
        big = "x" * (250 * 1024)  # > 200 KiB cap
        fs = MemoryFilesystemFuncTool({"big.txt": big})
        res = fs.read_file("big.txt")
        assert res.success == 0
        assert "too large" in (res.error or "").lower()
        # The error message must point the LLM at the workaround so it
        # doesn't just give up.
        assert "offset" in (res.error or "")


# --------------------------------------------------------------------------- #
# glob                                                                        #
# --------------------------------------------------------------------------- #


class TestGlob:
    def test_top_level_pattern(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.glob("manifest.json")
        assert res.success == 1
        assert res.result["files"] == ["manifest.json"]
        assert res.result["truncated"] is False

    def test_directory_scoped(self):
        """Pattern is applied relative to ``path`` when ``path`` is a subdir."""
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.glob("*.sql", path="queries")
        # Files are reported in slug-relative form, not path-relative — matches
        # the disk-backed tool so the LLM gets a stable id it can pass back to
        # ``read_file`` without rewriting.
        assert set(res.result["files"]) == {"queries/total_sales.sql", "queries/by_region.sql"}

    def test_recursive_double_star(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.glob("**/*.json")
        assert set(res.result["files"]) == {
            "manifest.json",
            "analysis/insights.json",
            "queries/total_sales.json",
        }

    def test_no_match_returns_empty(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.glob("**/*.toml")
        assert res.success == 1
        assert res.result["files"] == []
        assert res.result["truncated"] is False


# --------------------------------------------------------------------------- #
# grep                                                                        #
# --------------------------------------------------------------------------- #


class TestGrep:
    def test_finds_match_with_line_number(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.grep(r"SUM\(amount\)", path="queries", include="*.sql")
        assert res.success == 1
        matches = res.result["matches"]
        # Both .sql files reference SUM(amount).
        files_hit = {m["file"] for m in matches}
        assert files_hit == {"queries/total_sales.sql", "queries/by_region.sql"}
        # Each match carries a 1-based line number and the line content.
        for m in matches:
            assert m["line"] >= 1
            assert "SUM(amount)" in m["content"]

    def test_case_insensitive(self):
        fs = MemoryFilesystemFuncTool({"a.md": "Hello world\nGOODBYE\n"})
        res = fs.grep("hello", case_sensitive=False)
        assert res.success == 1
        assert len(res.result["matches"]) == 1
        assert res.result["matches"][0]["content"] == "Hello world"

    def test_invalid_regex_returns_error(self):
        fs = MemoryFilesystemFuncTool({"a.md": "x"})
        res = fs.grep("[unclosed")
        assert res.success == 0
        assert "regex" in (res.error or "").lower()

    def test_no_match_returns_empty_list(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        res = fs.grep("never_appears_anywhere")
        assert res.success == 1
        assert res.result["matches"] == []
        assert res.result["truncated"] is False


# --------------------------------------------------------------------------- #
# Surface contract                                                            #
# --------------------------------------------------------------------------- #


class TestTruncationCap:
    """Off-by-one regression guard for the truncation flag.

    Earlier code broke when ``len(matches) >= _MAX_*`` and then declared
    truncation from the same condition, so a bundle with EXACTLY the cap
    was flagged truncated even though no further matches were skipped.
    Fix: overshoot by one, set truncated from ``>``, trim back to cap.
    """

    def test_glob_at_exact_cap_not_marked_truncated(self, monkeypatch):
        import datus.tools.func_tool.memory_filesystem_tools as mod

        monkeypatch.setattr(mod, "_MAX_GLOB_RESULTS", 3)
        fs = MemoryFilesystemFuncTool({f"f{i}.md": "x" for i in range(3)})
        res = fs.glob("*.md")
        assert len(res.result["files"]) == 3
        assert res.result["truncated"] is False
        # The truncation message is only attached when truncation actually
        # happened — its absence is part of the "not truncated" contract.
        assert "message" not in res.result

    def test_glob_over_cap_marks_truncated_and_caps_results(self, monkeypatch):
        import datus.tools.func_tool.memory_filesystem_tools as mod

        monkeypatch.setattr(mod, "_MAX_GLOB_RESULTS", 3)
        fs = MemoryFilesystemFuncTool({f"f{i}.md": "x" for i in range(5)})
        res = fs.glob("*.md")
        assert len(res.result["files"]) == 3
        assert res.result["truncated"] is True
        assert "Results truncated" in res.result["message"]

    def test_grep_at_exact_cap_not_marked_truncated(self, monkeypatch):
        import datus.tools.func_tool.memory_filesystem_tools as mod

        monkeypatch.setattr(mod, "_MAX_GREP_MATCHES", 3)
        # One file, three matching lines — hits the cap exactly.
        fs = MemoryFilesystemFuncTool({"a.md": "hit\nhit\nhit\n"})
        res = fs.grep("hit")
        assert len(res.result["matches"]) == 3
        assert res.result["truncated"] is False

    def test_grep_over_cap_marks_truncated_and_caps_results(self, monkeypatch):
        import datus.tools.func_tool.memory_filesystem_tools as mod

        monkeypatch.setattr(mod, "_MAX_GREP_MATCHES", 3)
        # Five matching lines — must report 3 and truncated=True.
        fs = MemoryFilesystemFuncTool({"a.md": "hit\nhit\nhit\nhit\nhit\n"})
        res = fs.grep("hit")
        assert len(res.result["matches"]) == 3
        assert res.result["truncated"] is True


class TestSurfaceContract:
    def test_all_tools_name_is_read_only(self):
        """MemoryFilesystemFuncTool intentionally exposes only the three read operations."""
        assert MemoryFilesystemFuncTool.all_tools_name() == ["read_file", "glob", "grep"]

    def test_available_tools_count_matches_all_tools_name(self):
        fs = MemoryFilesystemFuncTool(_bundle())
        tools = fs.available_tools()
        # The LLM surface and ``all_tools_name`` must agree — otherwise
        # the agent_service tool catalog drifts from what's actually
        # callable at runtime.
        assert len(tools) == len(MemoryFilesystemFuncTool.all_tools_name())

    def test_root_path_is_label_not_filesystem(self):
        """``root_path`` is read by ChatAgenticNode for a debug log only.

        It must be a recognizable label (not a real path) so logs make it
        obvious the artifact is in-memory rather than on disk.
        """
        fs = MemoryFilesystemFuncTool({}, root_label="in-memory:demo")
        assert fs.root_path == "in-memory:demo"

    def test_no_write_methods_in_surface(self):
        """Defence-in-depth: writing must not be possible from the LLM."""
        fs = MemoryFilesystemFuncTool(_bundle())
        names = {getattr(t, "name", None) for t in fs.available_tools()}
        assert "write_file" not in names
        assert "edit_file" not in names
        assert "delete_file" not in names
