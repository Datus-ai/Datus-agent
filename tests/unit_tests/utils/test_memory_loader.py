# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/memory_loader.py"""

from unittest.mock import patch

import pytest

from datus.utils.memory_loader import (
    MEMORY_BASE_DIR,
    MEMORY_BYTE_LIMIT,
    MEMORY_FILENAME,
    apply_single_replacement,
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

    def test_gen_report_no_memory(self):
        assert has_memory("gen_report") is False

    def test_gen_semantic_model_no_memory(self):
        assert has_memory("gen_semantic_model") is False

    def test_gen_metrics_no_memory(self):
        assert has_memory("gen_metrics") is False

    def test_gen_sql_summary_no_memory(self):
        assert has_memory("gen_sql_summary") is False

    def test_explore_no_memory(self):
        assert has_memory("explore") is False

    def test_compare_no_memory(self):
        assert has_memory("compare") is False


class TestByteLimit:
    """The single hard cap is 2000 bytes."""

    def test_limit_value(self):
        assert MEMORY_BYTE_LIMIT == 2000


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
        with patch("pathlib.Path.read_text", side_effect=OSError("Permission denied")):
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

    def test_under_limit_not_truncated(self, tmp_path):
        write_memory_raw(str(tmp_path), "chat", "- one fact\n- another fact")
        result = load_memory_context(str(tmp_path), "chat")
        assert "truncated" not in result

    def test_over_byte_limit_truncated_on_newline(self, tmp_path):
        # 30 lines × ~100 bytes each ≈ 3 KB, over the 2000-byte cap.
        lines = [f"- fact line number {i} " + ("x" * 80) for i in range(30)]
        write_memory_raw(str(tmp_path), "chat", "\n".join(lines))

        result = load_memory_context(str(tmp_path), "chat")
        assert "truncated" in result
        # The whole returned string — body + appended warning — stays within the
        # hard cap; the warning bytes are reserved before the content is sliced.
        assert len(result.encode("utf-8")) <= MEMORY_BYTE_LIMIT
        body = result.split("> WARNING", 1)[0]
        assert len(body.encode("utf-8")) <= MEMORY_BYTE_LIMIT

    def test_exact_limit_not_truncated(self, tmp_path):
        # Build content that is exactly MEMORY_BYTE_LIMIT bytes.
        content = "a" * MEMORY_BYTE_LIMIT
        write_memory_raw(str(tmp_path), "chat", content)
        result = load_memory_context(str(tmp_path), "chat")
        assert "truncated" not in result
        assert result == content

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
