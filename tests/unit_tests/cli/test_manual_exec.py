# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/manual_exec.py.

Tests cover:
- encode/decode round-trip for the sentinel + JSON message format (SQL table /
  SQL message / SQL error / bash), including the caps and cell coercion that
  keep a record small enough to re-render on resume
- is_exec_message / exec_preview
- render_exec_block producing a bordered block whose text carries the command,
  result and meta (the single renderer shared by the live turn and resume)
- payload builders' truncation contracts
"""

import io
import json

from rich.console import Console

from datus.cli.manual_exec import (
    EXEC_SENTINEL,
    MAX_OUTPUT_CHARS,
    MAX_ROWS,
    build_bash_payload,
    build_sql_error_payload,
    build_sql_message_payload,
    build_sql_payload,
    build_tool_payload,
    decode_exec_message,
    encode_exec_message,
    exec_preview,
    exec_to_markdown,
    is_exec_message,
    render_exec_block,
)


def _render_text(payload) -> str:
    console = Console(file=io.StringIO(), width=200, no_color=True)
    console.print(render_exec_block(payload))
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# Tests: encode / decode round-trip
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    def test_sql_table_round_trip(self):
        payload = build_sql_payload(
            "SELECT id, name FROM users LIMIT 2",
            ["id", "name"],
            [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}],
            2,
            0.12,
        )
        message = encode_exec_message(payload)
        assert message.startswith(EXEC_SENTINEL)
        decoded = decode_exec_message(message)
        assert decoded["kind"] == "sql"
        assert decoded["command"] == "SELECT id, name FROM users LIMIT 2"
        assert decoded["columns"] == ["id", "name"]
        assert decoded["rows"] == [["1", "alice"], ["2", "bob"]]
        assert decoded["truncated"] is False
        assert decoded["meta"] == "2 rows in 0.12s"

    def test_bash_round_trip_preserves_stderr_marker(self):
        payload = build_bash_payload("git status", True, "out\n\n[stderr]\nwarn", None, 0.05)
        decoded = decode_exec_message(encode_exec_message(payload))
        assert decoded["kind"] == "bash"
        assert decoded["command"] == "git status"
        assert "\n[stderr]\n" in decoded["output"]
        assert decoded["meta"] == "exit 0 in 0.05s"

    def test_sql_error_round_trip(self):
        payload = build_sql_error_payload("SELECT bad", "syntax error")
        decoded = decode_exec_message(encode_exec_message(payload))
        assert decoded["success"] is False
        assert decoded["error"] == "syntax error"
        assert decoded["meta"] == "failed"

    def test_tool_round_trip(self):
        payload = build_tool_payload("search_table foo", True, '[{"table": "t1"}]', None, 0.12)
        decoded = decode_exec_message(encode_exec_message(payload))
        assert decoded["kind"] == "tool"
        assert decoded["command"] == "search_table foo"
        assert decoded["success"] is True
        assert decoded["output"] == '[{"table": "t1"}]'
        assert decoded["meta"] == "ok in 0.12s"

    def test_tool_failure_round_trip(self):
        payload = build_tool_payload("execute_sql", False, "", "denied", 0.0)
        decoded = decode_exec_message(encode_exec_message(payload))
        assert decoded["kind"] == "tool"
        assert decoded["success"] is False
        assert decoded["error"] == "denied"
        assert decoded["meta"] == "failed in 0.00s"

    def test_tool_output_capped(self):
        payload = build_tool_payload("dump", True, "x" * (MAX_OUTPUT_CHARS + 500), None, 0.01)
        assert payload["output"].endswith("… [truncated]")
        assert len(payload["output"]) < MAX_OUTPUT_CHARS + 100

    def test_unicode_command_survives(self):
        payload = build_sql_message_payload("SELECT '名前' AS 列", True, "success in 0.01s")
        decoded = decode_exec_message(encode_exec_message(payload))
        assert decoded["command"] == "SELECT '名前' AS 列"

    def test_command_with_at_and_brackets_survives(self):
        # ``@`` / ``[`` in the command must round-trip verbatim — the whole
        # point of the sentinel format is surviving @-parsing and Rich markup.
        payload = build_sql_error_payload("SELECT * FROM t WHERE u='@bob' AND a='[x]'", "boom")
        decoded = decode_exec_message(encode_exec_message(payload))
        assert decoded["command"] == "SELECT * FROM t WHERE u='@bob' AND a='[x]'"


class TestIsExecAndDecodeGuards:
    def test_plain_message_is_not_exec(self):
        assert is_exec_message("SELECT 1") is False
        assert decode_exec_message("SELECT 1") is None

    def test_non_string_is_not_exec(self):
        assert is_exec_message(None) is False
        assert decode_exec_message(None) is None

    def test_malformed_json_decodes_to_none(self):
        assert decode_exec_message(EXEC_SENTINEL + "{not json") is None

    def test_unknown_kind_decodes_to_none(self):
        assert decode_exec_message(EXEC_SENTINEL + '{"kind":"python","command":"x"}') is None


class TestDecodeValidation:
    """A persisted record is validated against its typed payload model, so a
    corrupted row degrades to an ordinary message instead of crashing renderers."""

    def test_malformed_rows_decodes_to_none(self):
        # ``rows`` must be a list of string lists; ``[1]`` is not.
        bad = EXEC_SENTINEL + json.dumps({"kind": "sql", "command": "x", "success": True, "rows": [1]})
        assert decode_exec_message(bad) is None

    def test_missing_required_field_decodes_to_none(self):
        # ``success`` has no default — a record without it is not a valid payload.
        bad = EXEC_SENTINEL + json.dumps({"kind": "sql", "command": "x"})
        assert decode_exec_message(bad) is None

    def test_kind_as_non_string_decodes_to_none(self):
        bad = EXEC_SENTINEL + json.dumps({"kind": ["sql"], "command": "x", "success": True})
        assert decode_exec_message(bad) is None


class TestExecToMarkdownSafety:
    """``exec_to_markdown`` must render arbitrary command/result content without
    letting it break out of the Markdown structure (web/SSE consumers)."""

    def test_plain_message_passthrough(self):
        assert exec_to_markdown("hello world") == "hello world"

    def test_command_with_triple_backticks_uses_collision_safe_fence(self):
        payload = build_sql_error_payload("SELECT '```x```'", "e")
        md = exec_to_markdown(encode_exec_message(payload))
        # A 4-backtick fence keeps the command's own ``` runs inside the block.
        assert "````sql\nSELECT '```x```'\n````" in md

    def test_output_with_backticks_uses_collision_safe_fence(self):
        payload = build_bash_payload("echo x", True, "```code```", None, 0.01)
        md = exec_to_markdown(encode_exec_message(payload))
        assert "````\n```code```\n````" in md

    def test_table_cells_escape_pipes(self):
        payload = build_sql_payload("SELECT a", ["a|b"], [{"a|b": "x|y"}], 1, 0.01)
        md = exec_to_markdown(encode_exec_message(payload))
        # Pipes in header and cell are escaped so they don't spawn phantom columns.
        assert "a\\|b" in md
        assert "x\\|y" in md

    def test_error_content_markdown_is_neutralised(self):
        payload = build_sql_error_payload("SELECT 1", "boom **bold** | pipe")
        md = exec_to_markdown(encode_exec_message(payload))
        assert "**Error:**" in md  # our own label stays
        assert "\\*\\*bold\\*\\*" in md  # the error's markdown is escaped
        assert "\\| pipe" in md


class TestExecPreview:
    def test_sql_preview(self):
        message = encode_exec_message(build_sql_error_payload("SELECT\n  1", "e"))
        # Newlines collapsed to a single space, prompt-prefixed.
        assert exec_preview(message) == "sql> SELECT 1"

    def test_bash_preview(self):
        message = encode_exec_message(build_bash_payload("ls -la", True, "", None, 0.01))
        assert exec_preview(message) == "bash> ls -la"

    def test_tool_preview(self):
        message = encode_exec_message(build_tool_payload("search_table foo", True, "out", None, 0.01))
        assert exec_preview(message) == "! search_table foo"

    def test_plain_message_preview_passthrough(self):
        assert exec_preview("hello world") == "hello world"

    def test_tool_render_shows_command_and_output(self):
        payload = build_tool_payload("list_tables", True, "t1\nt2", None, 0.02)
        rendered = _render_text(payload)
        assert "! list_tables" in rendered
        assert "t1" in rendered
        assert "ok in 0.02s" in rendered

    def test_tool_markdown_renders_output_fence(self):
        message = encode_exec_message(build_tool_payload("list_tables", True, "t1", None, 0.02))
        md = exec_to_markdown(message)
        assert md.startswith("`!`")
        assert "t1" in md


# ---------------------------------------------------------------------------
# Tests: caps / cell coercion
# ---------------------------------------------------------------------------


class TestCaps:
    def test_sql_rows_capped_and_marked_truncated(self):
        rows = [{"id": i} for i in range(MAX_ROWS + 10)]
        payload = build_sql_payload("SELECT id FROM t", ["id"], rows, MAX_ROWS + 10, 0.4)
        assert len(payload["rows"]) == MAX_ROWS
        assert payload["truncated"] is True
        assert payload["row_count"] == MAX_ROWS + 10

    def test_sql_cell_coercion_handles_none_and_types(self):
        rows = [{"a": None, "b": 3.5}]
        payload = build_sql_payload("SELECT a, b FROM t", ["a", "b"], rows, 1, 0.01)
        assert payload["rows"] == [["", "3.5"]]

    def test_bash_output_capped(self):
        payload = build_bash_payload("cat big", True, "x" * (MAX_OUTPUT_CHARS + 500), None, 0.1)
        assert payload["output"].endswith("… [truncated]")
        assert len(payload["output"]) <= MAX_OUTPUT_CHARS + len("\n… [truncated]")

    def test_bash_failure_carries_error(self):
        payload = build_bash_payload("false", False, "partial", "Command exited with code 1", 0.02)
        assert payload["success"] is False
        assert payload["error"] == "Command exited with code 1"
        assert payload["meta"] == "failed in 0.02s"


# ---------------------------------------------------------------------------
# Tests: render_exec_block
# ---------------------------------------------------------------------------


class TestRenderExecBlock:
    def test_sql_block_shows_command_table_and_meta(self):
        payload = build_sql_payload(
            "SELECT id, name FROM users",
            ["id", "name"],
            [{"id": 1, "name": "alice"}],
            1,
            0.12,
        )
        text = _render_text(payload)
        assert "sql>" in text
        assert "SELECT id, name FROM users" in text
        assert "alice" in text
        assert "1 rows in 0.12s" in text

    def test_sql_truncation_note_rendered(self):
        rows = [{"id": i} for i in range(MAX_ROWS + 5)]
        payload = build_sql_payload("SELECT id FROM t", ["id"], rows, MAX_ROWS + 5, 0.3)
        text = _render_text(payload)
        assert f"showing first {MAX_ROWS} of {MAX_ROWS + 5} rows" in text

    def test_bash_block_shows_output_and_stderr(self):
        payload = build_bash_payload("make", True, "building\n\n[stderr]\nwarn: x", None, 0.2)
        text = _render_text(payload)
        assert "bash>" in text
        assert "building" in text
        assert "warn: x" in text
        assert "exit 0 in 0.20s" in text

    def test_error_payload_shows_error_line(self):
        payload = build_sql_error_payload("SELECT bad", "syntax error near BAD")
        text = _render_text(payload)
        assert "sql>" in text
        assert "syntax error near BAD" in text

    def test_output_with_markup_is_not_interpreted(self):
        # Command output containing ``[...]`` must not be parsed as Rich markup.
        payload = build_bash_payload("echo", True, "value=[red]not-a-tag[/red]", None, 0.01)
        text = _render_text(payload)
        assert "[red]not-a-tag[/red]" in text

    def test_block_is_framed_like_user_message_in_mode_colour(self):
        """Top/bottom rules (HORIZONTALS) like a user message, coloured per mode."""
        from rich.box import HORIZONTALS
        from rich.panel import Panel

        sql_panel = render_exec_block(build_sql_error_payload("SELECT 1", "e"))
        assert isinstance(sql_panel, Panel)
        assert sql_panel.box is HORIZONTALS
        assert str(sql_panel.border_style) == "red"

        bash_panel = render_exec_block(build_bash_payload("ls", True, "", None, 0.01))
        assert isinstance(bash_panel, Panel)
        assert bash_panel.box is HORIZONTALS
        assert str(bash_panel.border_style) == "yellow"
