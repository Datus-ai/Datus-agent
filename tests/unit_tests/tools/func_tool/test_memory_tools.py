# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/tools/func_tool/memory_tools.py (MemoryFuncTool)."""

import pytest

from datus.tools.func_tool.memory_tools import MemoryFuncTool
from datus.utils.memory_loader import MEMORY_BYTE_LIMIT, get_memory_file_path, read_memory_raw


@pytest.fixture
def tool(tmp_path):
    return MemoryFuncTool(root_path=str(tmp_path), memory_node="chat")


class TestSurface:
    def test_tool_names(self):
        assert MemoryFuncTool.all_tools_name() == ["add_memory", "edit_memory"]

    def test_available_tools_exposes_two(self, tool):
        names = {t.name for t in tool.available_tools()}
        assert names == {"add_memory", "edit_memory"}

    def test_memory_node_property(self, tool):
        assert tool.memory_node == "chat"


class TestAddMemory:
    def test_first_add_creates_file(self, tool, tmp_path):
        r = tool.add_memory("User prefers DuckDB")
        assert r.success == 1
        assert get_memory_file_path(str(tmp_path), "chat").exists()
        assert read_memory_raw(str(tmp_path), "chat") == "User prefers DuckDB"
        assert r.result["memory"] == "User prefers DuckDB"
        assert r.result["used_bytes"] == len("User prefers DuckDB")
        assert r.result["remaining_budget"] == MEMORY_BYTE_LIMIT - len("User prefers DuckDB")

    def test_append_adds_newline_between_entries(self, tool, tmp_path):
        tool.add_memory("first fact")
        r = tool.add_memory("second fact")
        assert r.success == 1
        assert read_memory_raw(str(tmp_path), "chat") == "first fact\nsecond fact"

    def test_trailing_newlines_stripped_from_addition(self, tool, tmp_path):
        tool.add_memory("alpha")
        tool.add_memory("beta\n\n")
        assert read_memory_raw(str(tmp_path), "chat") == "alpha\nbeta"

    def test_empty_content_rejected(self, tool, tmp_path):
        r = tool.add_memory("   ")
        assert r.success == 0
        assert "must not be empty" in r.error
        assert not get_memory_file_path(str(tmp_path), "chat").exists()

    def test_over_limit_rejected_and_not_written(self, tool, tmp_path):
        tool.add_memory("a" * 1900)
        before = read_memory_raw(str(tmp_path), "chat")
        r = tool.add_memory("b" * 200)
        assert r.success == 0
        # The guidance names the byte limit and how much to free.
        assert str(MEMORY_BYTE_LIMIT) in r.error
        assert "edit_memory" in r.error
        # Nothing was written: file content unchanged.
        assert read_memory_raw(str(tmp_path), "chat") == before

    def test_add_exactly_at_limit_succeeds(self, tool, tmp_path):
        r = tool.add_memory("a" * MEMORY_BYTE_LIMIT)
        assert r.success == 1
        assert r.result["used_bytes"] == MEMORY_BYTE_LIMIT
        assert r.result["remaining_budget"] == 0

    def test_add_one_over_limit_rejected(self, tool, tmp_path):
        r = tool.add_memory("a" * (MEMORY_BYTE_LIMIT + 1))
        assert r.success == 0
        assert read_memory_raw(str(tmp_path), "chat") == ""


class TestEditMemory:
    def test_unique_replacement(self, tool, tmp_path):
        tool.add_memory("User prefers DuckDB")
        r = tool.edit_memory("DuckDB", "PostgreSQL")
        assert r.success == 1
        assert read_memory_raw(str(tmp_path), "chat") == "User prefers PostgreSQL"

    def test_delete_with_empty_new_string(self, tool, tmp_path):
        tool.add_memory("keep this")
        tool.add_memory("drop this")
        r = tool.edit_memory("\ndrop this", "")
        assert r.success == 1
        assert read_memory_raw(str(tmp_path), "chat") == "keep this"

    def test_empty_old_string_rejected(self, tool):
        tool.add_memory("anything")
        r = tool.edit_memory("", "x")
        assert r.success == 0
        assert "must not be empty" in r.error

    def test_edit_empty_memory_rejected(self, tool):
        r = tool.edit_memory("anything", "x")
        assert r.success == 0
        assert "empty" in r.error.lower()

    def test_not_found_rejected(self, tool):
        tool.add_memory("hello world")
        r = tool.edit_memory("nonexistent", "x")
        assert r.success == 0
        assert "not found" in r.error

    def test_multiple_matches_rejected(self, tool):
        tool.add_memory("dup")
        tool.add_memory("dup")
        r = tool.edit_memory("dup", "x")
        assert r.success == 0
        assert "exactly once" in r.error

    def test_edit_exceeding_limit_rejected(self, tool, tmp_path):
        tool.add_memory("a" * 1990 + "MARK")
        before = read_memory_raw(str(tmp_path), "chat")
        r = tool.edit_memory("MARK", "b" * 50)
        assert r.success == 0
        assert str(MEMORY_BYTE_LIMIT) in r.error
        assert read_memory_raw(str(tmp_path), "chat") == before

    def test_edit_result_payload(self, tool):
        tool.add_memory("alpha")
        r = tool.edit_memory("alpha", "beta")
        assert r.result["memory"] == "beta"
        assert r.result["used_bytes"] == len("beta")
        assert r.result["remaining_budget"] == MEMORY_BYTE_LIMIT - len("beta")


class TestFullToPrunedToRetry:
    """The end-to-end loop the over-limit guidance describes."""

    def test_full_then_prune_then_add(self, tool, tmp_path):
        stale = "STALE " + "x" * 1980
        tool.add_memory(stale)
        rejected = tool.add_memory("new important fact")
        assert rejected.success == 0

        # Free space by deleting the stale entry, then retry.
        pruned = tool.edit_memory(stale, "")
        assert pruned.success == 1
        retry = tool.add_memory("new important fact")
        assert retry.success == 1
        assert "new important fact" in read_memory_raw(str(tmp_path), "chat")
