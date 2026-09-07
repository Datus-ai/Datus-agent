# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for compact_archive: ToolArchive + maybe_truncate_item.

CI-tier: zero network, zero LLM, hermetic via tmp_path.
"""

import json
from pathlib import Path

import pytest

from datus.agent.node.compact_archive import (
    ARCHIVED_MARKER,
    ToolArchive,
    is_archived_output,
    maybe_truncate_item,
    parse_archived_marker,
)
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture
def archive(tmp_path):
    """Hermetic archive rooted under tmp_path — bypasses path_manager.

    Uses ``preview_chars=100`` so the implicit error-output 2× multiplier
    (200) is clearly distinguishable in tests.
    """
    return ToolArchive(
        project_name="proj",
        session_id="sid1",
        base_dir=tmp_path / "data",
        preview_chars=100,
    )


class TestToolArchive:
    def test_creates_directory(self, archive, tmp_path):
        assert archive.dir == tmp_path / "data"
        assert archive.dir.exists()

    def test_archive_args_returns_marker_with_path_and_preview(self, archive):
        content = "x" * 5000
        marker = archive.archive(content, message_idx=42, kind="args")
        # Plain-text marker, not a dict: prefix is fixed, then `path=` and `preview=`.
        assert marker.startswith(ARCHIVED_MARKER + " ")
        assert "path=" in marker and "preview=" in marker
        # Filename pattern stable: zero-padded idx + kind + 8-char hash + extension.
        head, _, _ = marker.partition(" preview=")
        archived = Path(head.split("path=", 1)[1])
        assert archived.suffix == ".json"
        assert "000042_args" in archived.name
        # The file on disk holds the original bytes verbatim.
        assert archived.read_text() == content
        assert "x" * 100 in marker  # preview present

    def test_archive_output_writes_txt_file(self, archive):
        content = "hello " * 1000
        marker = archive.archive(content, message_idx=7, kind="output")
        archived = Path(marker.split("path=", 1)[1].split(" preview=", 1)[0])
        assert archived.suffix == ".txt"
        assert "000007_output" in archived.name
        assert archived.read_text() == content

    def test_archive_idempotent_same_hash(self, archive):
        content = "identical content"
        m1 = archive.archive(content, 1, "args")
        m2 = archive.archive(content, 1, "args")
        # Same idx + content → same hash prefix → identical filename and marker.
        assert m1 == m2

    def test_error_output_uses_long_preview(self, archive):
        err = json.dumps({"success": 0, "error": "boom", "result": None}) + "X" * 500
        marker = archive.archive(err, message_idx=10, kind="output")
        preview = marker.split("preview=", 1)[1]
        # Error outputs get preview_chars * 2 (= 200) so the full failure
        # context is visible inline. Allow ≤ because newline/space
        # normalization could shorten by a few chars at the boundary.
        assert 150 < len(preview) <= 200
        assert "boom" in preview

    def test_non_error_output_uses_short_preview(self, archive):
        ok = json.dumps({"success": 1, "result": "ok"}) + "X" * 500
        marker = archive.archive(ok, message_idx=11, kind="output")
        preview = marker.split("preview=", 1)[1]
        # 100 = preview_chars (fixture).
        assert len(preview) <= 100

    def test_archive_rejects_invalid_kind(self, archive):
        with pytest.raises(DatusException) as excinfo:
            archive.archive("content", 0, "bogus")
        assert excinfo.value.code == ErrorCode.COMMON_FIELD_INVALID

    def test_preview_is_single_line(self, archive):
        # Multi-line content must be flattened — newlines in the marker would
        # break any log parser or grep-based debugging.
        content = "line1\nline2\nline3" + "X" * 500
        marker = archive.archive(content, message_idx=0, kind="output")
        assert "\n" not in marker
        assert "\r" not in marker


class TestIsError:
    def test_funcresult_success_zero(self):
        assert ToolArchive._is_error(json.dumps({"success": 0, "error": "x", "result": None})) is True

    def test_funcresult_success_one_no_error(self):
        assert ToolArchive._is_error(json.dumps({"success": 1, "result": "ok"})) is False

    def test_non_json_with_error_marker(self):
        assert ToolArchive._is_error("Traceback: line 1\nValueError") is True
        assert ToolArchive._is_error('something "error": "x"') is True

    def test_non_json_clean(self):
        assert ToolArchive._is_error("just a plain string output") is False

    def test_empty_error_field_not_treated_as_error(self):
        assert ToolArchive._is_error(json.dumps({"success": 1, "error": "", "result": "x"})) is False
        assert ToolArchive._is_error(json.dumps({"success": 1, "error": None, "result": "x"})) is False


class TestArchivedMarkerDetection:
    """Idempotency helpers: detect already-archived items in re-scan."""

    def test_output_marker_detected_directly(self):
        assert is_archived_output(f"{ARCHIVED_MARKER} path=/tmp/y preview=...") is True

    def test_output_marker_not_detected_for_real_output(self):
        assert is_archived_output(json.dumps({"success": 1, "result": "ok"})) is False

    def test_output_marker_not_detected_for_non_string(self):
        assert is_archived_output(None) is False
        assert is_archived_output({"result": "ok"}) is False


class TestParseArchivedMarker:
    """Display-side parser for ``[DATUS_ARCHIVED] path=... preview=...``."""

    def test_parses_well_formed_marker(self):
        text = f"{ARCHIVED_MARKER} path=/abs/000003_args_abc12345.json preview=hello world"
        result = parse_archived_marker(text)
        assert result == {"path": "/abs/000003_args_abc12345.json", "preview": "hello world"}

    def test_preview_keeps_internal_spaces(self):
        # ``preview=`` is the split delimiter and appears exactly once, so the
        # rest of the marker (which may include spaces, ``=``, brackets) is
        # captured verbatim in the preview field.
        text = f"{ARCHIVED_MARKER} path=/tmp/x.json preview=SELECT a, b FROM t WHERE x = 1"
        result = parse_archived_marker(text)
        assert result["preview"] == "SELECT a, b FROM t WHERE x = 1"

    def test_handles_inline_truncated_fallback_path(self):
        # ``_inline_truncated`` writes ``path=<unavailable: archive write failed>``
        # — the angle-bracketed token contains a space but no ``preview=``
        # substring, so the parser still splits correctly.
        text = f"{ARCHIVED_MARKER} path=<unavailable: archive write failed> preview=oops"
        result = parse_archived_marker(text)
        assert result == {"path": "<unavailable: archive write failed>", "preview": "oops"}

    def test_returns_none_for_non_marker_string(self):
        assert parse_archived_marker("just a regular tool output") is None

    def test_returns_none_for_non_string(self):
        assert parse_archived_marker(None) is None
        assert parse_archived_marker(123) is None
        assert parse_archived_marker({"path": "/x"}) is None
        assert parse_archived_marker(["a", "b"]) is None

    def test_marker_with_empty_preview(self):
        text = f"{ARCHIVED_MARKER} path=/abs/x.json preview="
        result = parse_archived_marker(text)
        assert result == {"path": "/abs/x.json", "preview": ""}

    def test_marker_without_preview_segment(self):
        # Defensive: callers should not produce markers without ``preview=``,
        # but the parser still returns a usable dict so display surfaces don't
        # crash. Path is captured; preview defaults to empty.
        text = f"{ARCHIVED_MARKER} path=/abs/x.json"
        result = parse_archived_marker(text)
        assert result == {"path": "/abs/x.json", "preview": ""}


class TestMaybeTruncateItem:
    def test_long_args_now_pass_through(self, archive):
        # ``function_call.arguments`` is never archived — a marker string in
        # ``arguments`` would break the tool-call payload sent to the LLM. Even
        # a 2000-char payload must come back byte-identical, with no file
        # written to the archive dir.
        item = {"type": "function_call", "name": "f", "arguments": "z" * 2000, "call_id": "c1"}
        out = maybe_truncate_item(item, archive, threshold=1000, idx=3)
        assert out is item  # identity preserved → caller sees "no change"
        assert out["arguments"] == "z" * 2000
        assert list(archive.dir.iterdir()) == []

    def test_long_output_get_archived(self, archive):
        item = {"type": "function_call_output", "output": "y" * 2000, "call_id": "c1"}
        out = maybe_truncate_item(item, archive, threshold=1000, idx=4)
        assert out is not item
        assert out["output"].startswith(ARCHIVED_MARKER)

    def test_short_args_pass_through(self, archive):
        item = {"type": "function_call", "name": "f", "arguments": "tiny", "call_id": "c1"}
        out = maybe_truncate_item(item, archive, threshold=1000, idx=5)
        # Identity preserved → caller can detect "no change" via ``is``.
        assert out is item

    def test_short_output_pass_through(self, archive):
        item = {"type": "function_call_output", "output": "ok", "call_id": "c1"}
        assert maybe_truncate_item(item, archive, threshold=1000, idx=6) is item

    def test_non_tool_items_pass_through(self, archive):
        for item in ({"type": "message", "content": "z" * 5000}, {"type": "reasoning", "content": "z" * 5000}):
            assert maybe_truncate_item(item, archive, threshold=10, idx=7) is item

    def test_idempotent_already_archived_output_skipped(self, archive):
        item = {"type": "function_call_output", "output": "y" * 5000}
        first = maybe_truncate_item(item, archive, threshold=1000, idx=10)
        before = sorted(p.name for p in archive.dir.iterdir())
        second = maybe_truncate_item(first, archive, threshold=1000, idx=10)
        after = sorted(p.name for p in archive.dir.iterdir())
        assert second is first
        assert before == after

    def test_archive_failure_falls_back_to_inline_marker(self, archive, monkeypatch):
        # Force the underlying write to raise → caller degrades to an inline
        # marker pointing at ``<unavailable>``. Information is lost but the
        # session can continue.
        def raise_oserror(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(archive, "archive", raise_oserror)
        item = {"type": "function_call_output", "output": "x" * 5000, "call_id": "c1"}
        out = maybe_truncate_item(item, archive, threshold=1000, idx=8)
        assert out["output"].startswith(ARCHIVED_MARKER)
        assert "<unavailable" in out["output"]
        # New dict so the rest of the compact pass continues.
        assert out is not item


class TestArchiveOldToolOutputs:
    """Mid-turn archive stage: everything but the newest K tool outputs."""

    @staticmethod
    def _responses_items(n_outputs: int, size: int = 1200):  # above the 1000-char default threshold
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ask"}]}]
        for i in range(n_outputs):
            items.append({"type": "function_call", "call_id": f"c{i}", "name": "execute_sql", "arguments": "{}"})
            items.append({"type": "function_call_output", "call_id": f"c{i}", "output": f"{i}:" + "x" * size})
        return items

    def test_keeps_the_newest_k_outputs_and_archives_the_rest(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._responses_items(7)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="responses", archive=archive, threshold=100, keep_recent=5
        )
        assert archived == 2
        outputs = [it for it in rewritten if it.get("type") == "function_call_output"]
        assert is_archived_output(outputs[0]["output"]) and is_archived_output(outputs[1]["output"])
        for kept in outputs[2:]:
            assert not is_archived_output(kept["output"])
        # Untouched items are the very same objects; calls are never rewritten.
        assert rewritten[0] is items[0]
        assert all(rewritten[i] is items[i] for i in range(len(items)) if items[i].get("type") == "function_call")

    def test_archived_content_is_recoverable_from_disk(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._responses_items(2)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="responses", archive=archive, threshold=100, keep_recent=1
        )
        assert archived == 1
        marker = parse_archived_marker(rewritten[2]["output"])
        assert Path(marker["path"]).read_text() == items[2]["output"]

    def test_short_outputs_below_threshold_are_left_alone(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._responses_items(4, size=10)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="responses", archive=archive, threshold=100, keep_recent=1
        )
        assert archived == 0
        assert rewritten == items

    def test_second_pass_is_idempotent(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._responses_items(6)
        once, first = archive_old_tool_outputs(
            items, item_format="responses", archive=archive, threshold=100, keep_recent=3
        )
        twice, second = archive_old_tool_outputs(
            once, item_format="responses", archive=archive, threshold=100, keep_recent=3
        )
        assert first == 3 and second == 0
        assert twice == once

    def test_keep_recent_zero_archives_everything_eligible(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._responses_items(3)
        _, archived = archive_old_tool_outputs(
            items, item_format="responses", archive=archive, threshold=100, keep_recent=0
        )
        assert archived == 3

    @staticmethod
    def _anthropic_items(n_results: int, size: int = 1200, as_blocks: bool = False):
        items = [{"role": "user", "content": [{"type": "text", "text": "ask"}]}]
        for i in range(n_results):
            items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": f"t{i}", "name": "execute_sql", "input": {}}],
                }
            )
            body = f"{i}:" + "y" * size
            content = [{"type": "text", "text": body}] if as_blocks else body
            items.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": content}]}
            )
        return items

    def test_anthropic_string_tool_results_are_archived_in_place(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._anthropic_items(4)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="anthropic", archive=archive, threshold=100, keep_recent=2
        )
        assert archived == 2
        results = [m for m in rewritten if m["role"] == "user" and m["content"][0].get("type") == "tool_result"]
        assert is_archived_output(results[0]["content"][0]["content"])
        assert is_archived_output(results[1]["content"][0]["content"])
        assert not is_archived_output(results[2]["content"][0]["content"])
        # The tool_use pairing is untouched: same ids, same assistant messages.
        assert results[0]["content"][0]["tool_use_id"] == "t0"
        assert all(rewritten[i] is items[i] for i in range(len(items)) if items[i]["role"] == "assistant")

    def test_anthropic_text_block_tool_results_are_archived(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._anthropic_items(2, as_blocks=True)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="anthropic", archive=archive, threshold=100, keep_recent=1
        )
        assert archived == 1
        marker = rewritten[2]["content"][0]["content"]
        assert is_archived_output(marker)
        assert Path(parse_archived_marker(marker)["path"]).read_text() == items[2]["content"][0]["content"][0]["text"]

    def test_anthropic_pass_is_idempotent(self, archive):
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._anthropic_items(3)
        once, first = archive_old_tool_outputs(
            items, item_format="anthropic", archive=archive, threshold=100, keep_recent=1
        )
        twice, second = archive_old_tool_outputs(
            once, item_format="anthropic", archive=archive, threshold=100, keep_recent=1
        )
        assert first == 2 and second == 0
        assert twice == once

    def test_outputs_not_longer_than_the_preview_are_left_alone(self, archive):
        """Archiving only pays off when the marker is shorter than the text.

        With ``preview_chars=100`` a 120-char output would become a ~220-char
        marker, so it must neither be rewritten nor counted (both formats).
        """
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        items = self._responses_items(3, size=120)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="responses", archive=archive, threshold=100, keep_recent=0
        )
        assert archived == 0
        assert all(new is old for new, old in zip(rewritten, items))
        assert not any("[DATUS_ARCHIVED]" in it.get("output", "") for it in rewritten if isinstance(it, dict))

        anthropic_items = [
            {"role": "user", "content": [{"type": "text", "text": "ask"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "r" * 120}]},
        ]
        rewritten, archived = archive_old_tool_outputs(
            anthropic_items, item_format="anthropic", archive=archive, threshold=100, keep_recent=0
        )
        assert archived == 0
        assert rewritten[2] is anthropic_items[2]

    def test_a_marker_longer_than_the_text_is_discarded(self, archive, tmp_path):
        """A very long archive path can make even a >preview output grow; keep the original then."""
        from datus.agent.node.compact_archive import archive_old_tool_outputs

        long_dir = tmp_path / ("d" * 200) / ("e" * 200)
        long_dir.mkdir(parents=True)
        big_path_archive = ToolArchive(project_name="proj", session_id="sid1", base_dir=long_dir, preview_chars=100)
        items = self._responses_items(1, size=150)
        rewritten, archived = archive_old_tool_outputs(
            items, item_format="responses", archive=big_path_archive, threshold=100, keep_recent=0
        )
        assert archived == 0
        assert rewritten[-1] is items[-1]
