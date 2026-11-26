# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/tools/func_tool/memory_tools.py (MemoryFuncTool)."""

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError
from unittest.mock import patch

import pytest

from datus.tools.func_tool.memory_tools import MemoryFuncTool
from datus.utils.memory_loader import (
    MEMORY_BYTE_LIMIT,
    MEMORY_CONTEXT_BYTE_LIMIT,
    get_memory_file_path,
    read_memory_raw,
    write_memory_raw,
)


def _utf8_text(size: int) -> str:
    """Multi-byte text of exactly ``size`` UTF-8 bytes."""
    return "记" * (size // 3) + "x" * (size % 3)


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

    def test_add_exactly_at_limit_succeeds(self, tool, tmp_path):
        content = _utf8_text(MEMORY_BYTE_LIMIT)
        r = tool.add_memory(content)
        assert r.success == 1
        assert r.result["used_bytes"] == MEMORY_BYTE_LIMIT
        assert r.result["remaining_budget"] == 0
        assert "warning" not in r.result
        assert get_memory_file_path(str(tmp_path), "chat").read_bytes() == content.encode("utf-8")

    def test_append_one_byte_over_limit_warns_and_saves(self, tool, tmp_path):
        original = "a" * (MEMORY_BYTE_LIMIT - 3)
        assert tool.add_memory(original).success == 1
        r = tool.add_memory("忆")
        assert r.success == 1
        assert r.result["used_bytes"] == MEMORY_BYTE_LIMIT + 1
        assert r.result["remaining_budget"] == 0
        # The warning tells the agent to consolidate back under the soft limit.
        assert f"below {MEMORY_BYTE_LIMIT} bytes" in r.result["warning"]
        assert "edit_memory" in r.result["warning"]
        assert read_memory_raw(str(tmp_path), "chat") == original + "\n忆"

    def test_add_up_to_hard_limit_succeeds(self, tool, tmp_path):
        content = _utf8_text(MEMORY_CONTEXT_BYTE_LIMIT)
        r = tool.add_memory(content)
        assert r.success == 1
        assert r.result["used_bytes"] == MEMORY_CONTEXT_BYTE_LIMIT
        assert get_memory_file_path(str(tmp_path), "chat").read_bytes() == content.encode("utf-8")

    @pytest.mark.parametrize(
        "existing_bytes, addition",
        [
            (0, _utf8_text(MEMORY_CONTEXT_BYTE_LIMIT + 1)),
            (MEMORY_CONTEXT_BYTE_LIMIT - 3, "忆"),
            (MEMORY_CONTEXT_BYTE_LIMIT, "- newest fact"),
        ],
        ids=["single_oversized_write", "one_byte_over", "already_full"],
    )
    def test_add_past_hard_limit_is_rejected_and_file_unchanged(self, tool, tmp_path, existing_bytes, addition):
        if existing_bytes:
            write_memory_raw(str(tmp_path), "chat", _utf8_text(existing_bytes))
        memory_file = get_memory_file_path(str(tmp_path), "chat")
        before = memory_file.read_bytes() if memory_file.exists() else None

        r = tool.add_memory(addition)

        assert r.success == 0
        assert str(MEMORY_CONTEXT_BYTE_LIMIT) in r.error
        assert "edit_memory" in r.error
        assert (memory_file.read_bytes() if memory_file.exists() else None) == before
        # Whatever the agent manages to save stays within what the prompt can show.
        assert len(read_memory_raw(str(tmp_path), "chat").encode("utf-8")) <= MEMORY_CONTEXT_BYTE_LIMIT

    def test_result_size_does_not_grow_with_the_file(self, tool, tmp_path):
        entries = [f"- fact {index:04d}: " + "y" * 80 for index in range(250)]
        write_memory_raw(str(tmp_path), "chat", "\n".join(entries))

        results = [tool.add_memory("- newest fact"), tool.edit_memory("- newest fact", "- edited fact")]

        for result in results:
            assert result.success == 1
            payload = json.dumps(result.result, ensure_ascii=False)
            assert len(payload) < 1_000
            assert not any(entry in payload for entry in entries)


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

    def test_edit_exceeding_limit_warns_and_saves(self, tool, tmp_path):
        prefix = "a" * (MEMORY_BYTE_LIMIT - 4)
        assert tool.add_memory(prefix + "MARK").success == 1
        r = tool.edit_memory("MARK", "b" * 50)
        assert r.success == 1
        assert str(MEMORY_BYTE_LIMIT) in r.result["warning"]
        assert read_memory_raw(str(tmp_path), "chat") == prefix + "b" * 50

    def test_edit_result_payload(self, tool):
        tool.add_memory("alpha")
        r = tool.edit_memory("alpha", "beta")
        assert r.result["used_bytes"] == len("beta")
        assert r.result["remaining_budget"] == MEMORY_BYTE_LIMIT - len("beta")

    def test_edit_growing_past_hard_limit_is_rejected(self, tool, tmp_path):
        original = "a" * (MEMORY_CONTEXT_BYTE_LIMIT - 4) + "MARK"
        assert tool.add_memory(original).success == 1

        r = tool.edit_memory("MARK", "MARK!")

        assert r.success == 0
        assert str(MEMORY_CONTEXT_BYTE_LIMIT) in r.error
        assert read_memory_raw(str(tmp_path), "chat") == original

    def test_oversized_hand_edited_file_can_still_be_shrunk(self, tool, tmp_path):
        stale = "- STALE " + "x" * 2_000
        oversized = "y" * MEMORY_CONTEXT_BYTE_LIMIT + "\n" + stale + "\n- keep"
        write_memory_raw(str(tmp_path), "chat", oversized)

        shrunk = tool.edit_memory(stale, "- short")
        assert shrunk.success == 1
        assert shrunk.result["used_bytes"] > MEMORY_CONTEXT_BYTE_LIMIT
        # Growing it again stays rejected, and so does appending.
        assert tool.edit_memory("- short", "- short" + "z" * 10).success == 0
        assert tool.add_memory("- more").success == 0
        assert read_memory_raw(str(tmp_path), "chat") == "y" * MEMORY_CONTEXT_BYTE_LIMIT + "\n- short\n- keep"

    def test_pruning_clears_warning_and_preserves_new_entry(self, tool, tmp_path):
        stale = "STALE " + "x" * (MEMORY_BYTE_LIMIT - 6)
        assert tool.add_memory(stale).success == 1
        appended = tool.add_memory("new important fact")
        assert appended.success == 1
        assert "warning" in appended.result

        pruned = tool.edit_memory(stale + "\n", "")
        assert pruned.success == 1
        assert "warning" not in pruned.result
        assert read_memory_raw(str(tmp_path), "chat") == "new important fact"


def _pause_parallel_reads(writers):
    snapshots = Barrier(writers)

    def read(*args, **kwargs):
        content = read_memory_raw(*args, **kwargs)
        try:
            snapshots.wait(timeout=1)
        except BrokenBarrierError:
            # With the mutation lock, only the first writer can reach this barrier.
            pass
        return content

    return patch("datus.tools.func_tool.memory_tools.read_memory_raw", side_effect=read)


def _append_from_process(root, connection):
    from datus.tools.func_tool.memory_tools import write_memory_raw

    def pause_after_read(*args, **kwargs):
        connection.send("ready to write")
        # An unlocked parent finishes its own write before this snapshot commits.
        connection.poll(1)
        write_memory_raw(*args, **kwargs)

    with connection, patch("datus.tools.func_tool.memory_tools.write_memory_raw", side_effect=pause_after_read):
        tool = MemoryFuncTool(root_path=root, memory_node="chat")
        connection.send(tool.add_memory("child").success)
        connection.poll(5)


class TestConcurrentMemory:
    def test_parallel_appends_preserve_every_successful_entry(self, tmp_path):
        entries = [f"entry {index}" for index in range(8)]

        def append(entry):
            writer = MemoryFuncTool(root_path=str(tmp_path), memory_node="chat")
            return writer.add_memory(entry).success

        with _pause_parallel_reads(len(entries)):
            with ThreadPoolExecutor(max_workers=len(entries)) as executor:
                assert list(executor.map(append, entries)) == [1] * len(entries)

        assert sorted(read_memory_raw(str(tmp_path), "chat").splitlines()) == sorted(entries)

    def test_competing_edits_reject_the_stale_replacement(self, tool, tmp_path):
        assert tool.add_memory("original").success == 1

        def edit(replacement):
            return tool.edit_memory("original", replacement)

        replacements = ["first choice", "second choice"]
        with _pause_parallel_reads(len(replacements)):
            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(edit, replacements))

        saved = [replacement for replacement, outcome in zip(replacements, outcomes, strict=True) if outcome.success]
        rejected = [outcome for outcome in outcomes if not outcome.success]
        assert len(saved) == len(rejected) == 1
        assert "not found" in rejected[0].error
        assert read_memory_raw(str(tmp_path), "chat") == saved[0]

    def test_parallel_process_appends_preserve_both_entries(self, tool, tmp_path):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_append_from_process, args=(str(tmp_path), child))
        process.start()
        child.close()
        try:
            with parent:
                assert parent.poll(15)
                assert parent.recv() == "ready to write"
                assert tool.add_memory("parent").success == 1
                parent.send("finished")
                assert parent.poll(5)
                assert parent.recv() == 1
            process.join(5)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
        assert sorted(read_memory_raw(str(tmp_path), "chat").splitlines()) == ["child", "parent"]


class TestMemoryWriteFailures:
    def test_invalid_utf8_is_not_overwritten(self, tool, tmp_path):
        memory_file = get_memory_file_path(str(tmp_path), "chat")
        memory_file.parent.mkdir(parents=True)
        original = b"\xffinvalid UTF-8"
        memory_file.write_bytes(original)

        assert tool.add_memory("new entry").success == 0
        assert memory_file.read_bytes() == original

    def test_failed_replace_preserves_original_and_allows_retry(self, tool, tmp_path):
        assert tool.add_memory("original").success == 1
        with patch("datus.storage.semantic_model.artifact_file.os.replace", side_effect=OSError("replace failed")):
            failed = tool.add_memory("new entry")
        assert failed.success == 0
        assert "replace failed" in failed.error
        assert read_memory_raw(str(tmp_path), "chat") == "original"

        assert tool.add_memory("new entry").success == 1
        assert read_memory_raw(str(tmp_path), "chat") == "original\nnew entry"
