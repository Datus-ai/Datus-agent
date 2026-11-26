# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/memory_loader.py"""

import os
import stat
from unittest.mock import patch

import pytest

from datus.utils.memory_loader import (
    MEMORY_BASE_DIR,
    MEMORY_BYTE_LIMIT,
    MEMORY_CONTEXT_BYTE_LIMIT,
    MEMORY_FILENAME,
    apply_single_replacement,
    format_memory_context,
    get_memory_dir,
    get_memory_file_path,
    has_memory,
    load_memory_context,
    read_memory_raw,
    write_memory_raw,
)


class TestHasMemory:
    """Tests for has_memory()."""

    def test_chat_has_memory(self):
        assert has_memory("chat") is True

    def test_custom_agent_has_memory(self):
        assert has_memory("my_custom_agent") is True

    def test_gen_sql_no_memory(self):
        assert has_memory("gen_sql") is False

    def test_gen_visual_report_no_memory(self):
        assert has_memory("gen_visual_report") is False

    def test_semantic_modeling_no_memory(self):
        assert has_memory("semantic_modeling") is False

    def test_gen_sql_summary_no_memory(self):
        assert has_memory("gen_sql_summary") is False

    def test_explore_no_memory(self):
        assert has_memory("explore") is False

    def test_compare_no_memory(self):
        assert has_memory("compare") is False


class TestByteLimit:
    """The soft limit is expressed in UTF-8 bytes."""

    def test_limit_value(self):
        assert MEMORY_BYTE_LIMIT == 20_000


class TestPathHelpers:
    """Tests for get_memory_file_path / read_memory_raw / write_memory_raw."""

    def test_file_path_layout(self, tmp_path):
        path = get_memory_file_path(str(tmp_path), "chat")
        assert path == tmp_path / MEMORY_BASE_DIR / "chat" / MEMORY_FILENAME

    def test_read_missing_returns_empty(self, tmp_path):
        assert read_memory_raw(str(tmp_path), "chat") == ""

    def test_write_creates_parents_and_roundtrips(self, tmp_path):
        write_memory_raw(str(tmp_path), "chat", "hello memory")
        assert get_memory_file_path(str(tmp_path), "chat").exists()
        assert read_memory_raw(str(tmp_path), "chat") == "hello memory"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
    def test_new_memory_file_is_not_private_to_the_owner(self, tmp_path):
        """Other local readers (web UI, backups) keep the access a plain file write gave them."""
        write_memory_raw(str(tmp_path), "chat", "hello memory")
        mode = stat.S_IMODE(get_memory_file_path(str(tmp_path), "chat").stat().st_mode)
        assert mode & 0o044 == 0o044

    def test_write_overwrites(self, tmp_path):
        write_memory_raw(str(tmp_path), "chat", "first")
        write_memory_raw(str(tmp_path), "chat", "second")
        assert read_memory_raw(str(tmp_path), "chat") == "second"

    def test_read_unicode_error_returns_empty(self, tmp_path):
        memory_file = get_memory_file_path(str(tmp_path), "chat")
        memory_file.parent.mkdir(parents=True)
        memory_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
        result = read_memory_raw(str(tmp_path), "chat")
        assert result == ""

    def test_read_os_error_returns_empty(self, tmp_path):
        memory_file = get_memory_file_path(str(tmp_path), "chat")
        memory_file.parent.mkdir(parents=True)
        memory_file.write_text("content", encoding="utf-8")
        with patch("pathlib.Path.open", side_effect=OSError("Permission denied")):
            assert read_memory_raw(str(tmp_path), "chat") == ""


class TestApplySingleReplacement:
    """Tests for the shared pure replacement core."""

    def test_unique_replacement(self):
        new, err = apply_single_replacement("a b c", "b", "X")
        assert err is None
        assert new == "a X c"

    def test_empty_old_string_rejected(self):
        new, err = apply_single_replacement("abc", "", "X")
        assert new is None
        assert "must not be empty" in err

    def test_not_found(self):
        new, err = apply_single_replacement("abc", "zzz", "X")
        assert new is None
        assert "not found" in err

    def test_multiple_matches_rejected(self):
        new, err = apply_single_replacement("a a a", "a", "X")
        assert new is None
        assert "exactly once" in err

    def test_empty_new_string_deletes(self):
        new, err = apply_single_replacement("keep DROP", " DROP", "")
        assert err is None
        assert new == "keep"

    def test_long_old_string_preview_truncated(self):
        new, err = apply_single_replacement("abc", "z" * 200, "X")
        assert new is None
        assert "..." in err


class TestLoadMemoryContext:
    """Tests for load_memory_context()."""

    def test_file_not_found_returns_empty(self, tmp_path):
        assert load_memory_context(str(tmp_path), "chat") == ""

    def test_normal_content(self, tmp_path):
        write_memory_raw(str(tmp_path), "chat", "# Memory\n\n- item 1\n- item 2\n")
        result = load_memory_context(str(tmp_path), "chat")
        assert "# Memory" in result
        assert "- item 1" in result
        assert "- item 2" in result

    def test_empty_file(self, tmp_path):
        write_memory_raw(str(tmp_path), "chat", "")
        assert load_memory_context(str(tmp_path), "chat") == ""

    @pytest.mark.parametrize(
        "content",
        ["- one fact\n- another fact", "a" * MEMORY_BYTE_LIMIT, "a" * MEMORY_CONTEXT_BYTE_LIMIT + "\n- last fact\n"],
        ids=["under_limit", "at_limit", "over_prompt_limit"],
    )
    def test_load_preserves_full_content(self, tmp_path, content):
        write_memory_raw(str(tmp_path), "chat", content)
        assert load_memory_context(str(tmp_path), "chat") == content

    def test_custom_agent_memory(self, tmp_path):
        write_memory_raw(str(tmp_path), "my_agent", "# Custom Agent Memory\n")
        result = load_memory_context(str(tmp_path), "my_agent")
        assert "Custom Agent Memory" in result

    @pytest.mark.acceptance
    def test_workspace_subagent_isolation_and_update(self, tmp_path):
        """Memory is loaded from the requested workspace/subagent and reflects file updates."""
        write_memory_raw(str(tmp_path), "chat", "- prefer explicit joins")
        write_memory_raw(str(tmp_path), "finance_agent", "- revenue uses net_amount")

        assert "explicit joins" in load_memory_context(str(tmp_path), "chat")
        assert "net_amount" not in load_memory_context(str(tmp_path), "chat")
        assert "net_amount" in load_memory_context(str(tmp_path), "finance_agent")

        write_memory_raw(str(tmp_path), "chat", "- prefer CTEs for multi-step SQL")
        updated = load_memory_context(str(tmp_path), "chat")
        assert "prefer CTEs" in updated
        assert "explicit joins" not in updated


class TestFormatMemoryContext:
    """The prompt excerpt and the notice that goes with it."""

    @staticmethod
    def _lines(count: int) -> list[str]:
        return [f"- fact {index:04d}: " + "y" * 80 for index in range(count)]

    def test_small_memory_is_unchanged_without_notice(self):
        assert format_memory_context("- one\n- two") == ("- one\n- two", "")

    def test_over_soft_limit_asks_to_consolidate_below_it(self):
        content, notice = format_memory_context("x" * (MEMORY_BYTE_LIMIT + 1))
        assert content == "x" * (MEMORY_BYTE_LIMIT + 1)
        assert f"below {MEMORY_BYTE_LIMIT} bytes" in notice
        assert "edit_memory" in notice
        assert "partial" not in notice

    def test_excerpt_never_slices_an_entry(self):
        lines = self._lines(300)
        content = "\n".join(lines)
        excerpt, notice = format_memory_context(content)
        visible = excerpt.split("\n")
        assert visible == lines[: len(visible)]
        assert len(excerpt.encode("utf-8")) <= MEMORY_CONTEXT_BYTE_LIMIT
        assert f"first {len(excerpt.encode('utf-8'))} of {len(content.encode('utf-8'))} bytes" in notice
        assert "partial" in notice

    def test_line_ending_exactly_at_the_limit_is_kept(self):
        first = "a" * (MEMORY_CONTEXT_BYTE_LIMIT - 1)
        excerpt, _ = format_memory_context(first + "b\n- hidden")
        assert excerpt == first + "b"

    def test_single_oversized_line_falls_back_to_a_character_boundary(self):
        excerpt, notice = format_memory_context("记" * MEMORY_CONTEXT_BYTE_LIMIT)
        assert excerpt == "记" * (MEMORY_CONTEXT_BYTE_LIMIT // 3)
        assert "partial" in notice

    @pytest.mark.parametrize("lf_bytes", [MEMORY_BYTE_LIMIT - 50, MEMORY_CONTEXT_BYTE_LIMIT])
    def test_crlf_is_sized_like_the_memory_tools_see_it(self, lf_bytes):
        lines = [f"- {index:03d} " + "z" * 90 for index in range(lf_bytes // 97 + 1)]
        lf = "\n".join(lines)[:lf_bytes]
        crlf = lf.replace("\n", "\r\n")
        assert len(crlf.encode("utf-8")) > len(lf.encode("utf-8"))
        assert format_memory_context(crlf) == format_memory_context(lf)

    def test_read_only_gets_only_the_partial_notice(self):
        assert format_memory_context("x" * (MEMORY_BYTE_LIMIT + 1), read_only=True) == (
            "x" * (MEMORY_BYTE_LIMIT + 1),
            "",
        )
        _, notice = format_memory_context("\n".join(self._lines(300)), read_only=True)
        assert "partial" in notice
        assert "edit_memory" not in notice


class TestGetMemoryDir:
    """Tests for get_memory_dir()."""

    def test_chat_dir(self):
        assert get_memory_dir(".", "chat") == f"{MEMORY_BASE_DIR}/chat"

    def test_custom_agent_dir(self):
        result = get_memory_dir("/workspace", "my_agent")
        assert result == f"{MEMORY_BASE_DIR}/my_agent"

    def test_dir_is_relative(self):
        result = get_memory_dir("/any/path", "chat")
        assert not result.startswith("/")
