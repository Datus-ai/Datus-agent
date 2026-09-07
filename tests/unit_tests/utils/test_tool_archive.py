# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the shared disk-archive primitive.

The archive helpers moved from ``datus.agent.node.compact_archive`` to
``datus.utils.tool_archive`` so both the compact pass and the bash tool can
share them. These tests cover the relocated primitives and the round-trip
between :func:`build_archived_marker` and :func:`parse_archived_marker`.
"""

import json

import pytest

from datus.utils.tool_archive import (
    ARCHIVED_MARKER,
    ToolArchive,
    build_archived_marker,
    is_archived_output,
    is_error_output,
    make_single_line_preview,
    parse_archived_marker,
)


class TestMarkerRoundTrip:
    def test_build_then_parse(self):
        marker = build_archived_marker("/tmp/000001_output_ab.txt", "the preview text")
        assert marker.startswith(ARCHIVED_MARKER)
        parsed = parse_archived_marker(marker)
        assert parsed == {"path": "/tmp/000001_output_ab.txt", "preview": "the preview text"}

    def test_is_archived_output(self):
        assert is_archived_output(build_archived_marker("/tmp/x.txt", "p"))
        assert not is_archived_output('{"success": 1, "result": "hi"}')
        assert not is_archived_output(None)

    def test_parse_non_marker_returns_none(self):
        assert parse_archived_marker("not a marker") is None
        assert parse_archived_marker(None) is None

    def test_parse_write_failure_fallback_path(self):
        marker = build_archived_marker("<unavailable: archive write failed>", "preview here")
        parsed = parse_archived_marker(marker)
        assert parsed["path"] == "<unavailable: archive write failed>"
        assert parsed["preview"] == "preview here"


class TestSingleLinePreview:
    def test_flattens_newlines_and_truncates(self):
        assert make_single_line_preview("a\nb\r\nc", 100) == "a b  c"
        assert make_single_line_preview("x" * 50, 10) == "x" * 10


_SUCCESS_ENVELOPE = {"success": 1, "error": None, "result": {"original_rows": 60, "compressed_data": "a,b\n1,2"}}
_FAILURE_ENVELOPE = {"success": 0, "error": "no such table: frpm", "result": None}


class TestIsErrorOutput:
    def test_funcresult_success_zero_is_error(self):
        assert is_error_output('{"success": 0, "error": "boom"}')

    def test_success_one_not_error(self):
        assert not is_error_output('{"success": 1, "result": "ok"}')

    def test_non_json_traceback_is_error(self):
        assert is_error_output("Traceback (most recent call last): ...")

    @pytest.mark.parametrize("serialize", [json.dumps, str, repr], ids=["json", "str", "repr"])
    def test_successful_envelope_is_never_an_error_whatever_the_serialization(self, serialize):
        """The SDK stores tool outputs as ``str(dict)`` (single quotes, ``None``);
        ``'error': None`` must not be mistaken for an error marker."""
        assert not is_error_output(serialize(_SUCCESS_ENVELOPE))

    @pytest.mark.parametrize("serialize", [json.dumps, str, repr], ids=["json", "str", "repr"])
    def test_failed_envelope_is_an_error_whatever_the_serialization(self, serialize):
        assert is_error_output(serialize(_FAILURE_ENVELOPE))
        assert is_error_output(serialize({"success": 1, "error": "partial failure", "result": None}))

    def test_repr_error_message_with_a_single_quote(self):
        # Python switches to double quotes for such strings: ``'error': "can't open"``.
        assert is_error_output(str({"success": 0, "error": "can't open db", "result": None}))
        assert is_error_output(str({"success": 1, "error": "can't open db", "result": None}))

    def test_repr_success_with_error_looking_data_inside_result(self):
        # An ``error`` column in the *result* payload is data, not an error envelope.
        payload = {"success": 1, "error": None, "result": {"columns": ["error"], "rows": [["Traceback seen"]]}}
        assert not is_error_output(str(payload))

    def test_unparseable_text_only_matches_string_valued_error_markers(self):
        assert not is_error_output("prefix 'error': None suffix")
        assert not is_error_output('prefix "error": null suffix')
        assert is_error_output("prefix 'error': 'boom' suffix")
        assert is_error_output('prefix "error": "boom" suffix')
        assert is_error_output("prefix 'success': 0 suffix")

    def test_non_string_input_is_not_an_error(self):
        assert not is_error_output(None)
        assert not is_error_output({"success": 0})


class TestToolArchivePrimitives:
    def test_repr_success_output_gets_the_normal_preview_width(self, tmp_path):
        """A successful ``str(FuncToolResult)`` output is archived with ``preview_chars``,
        not the doubled error width, so the marker really is shorter than the text."""
        archive = ToolArchive("proj", "sess", base_dir=tmp_path, preview_chars=40)
        text = str({"success": 1, "error": None, "result": {"compressed_data": "x" * 400}})
        marker = archive.archive(text, 1, "output")
        preview = parse_archived_marker(marker)["preview"]
        # ``make_single_line_preview`` strips trailing whitespace, so compare by bound.
        assert text.startswith(preview) and 30 < len(preview) <= 40
        failed = str({"success": 0, "error": "boom " * 50, "result": None})
        failed_preview = parse_archived_marker(archive.archive(failed, 2, "output"))["preview"]
        assert 40 < len(failed_preview) <= 80

    def test_archive_writes_file_and_returns_marker(self, tmp_path):
        archive = ToolArchive("proj", "sess", base_dir=tmp_path, preview_chars=20)
        marker = archive.archive("full content here that is fairly long", 3, "output")
        files = list(tmp_path.glob("000003_output_*.txt"))
        assert len(files) == 1
        assert files[0].read_text() == "full content here that is fairly long"
        # Marker points at the written file and carries a bounded preview.
        parsed = parse_archived_marker(marker)
        assert parsed["path"] == str(files[0])
        assert parsed["preview"] == "full content here th"  # preview_chars=20

    def test_archive_rejects_bad_kind(self, tmp_path):
        from datus.utils.exceptions import DatusException

        archive = ToolArchive("proj", "sess", base_dir=tmp_path)
        with pytest.raises(DatusException):
            archive.archive("x", 0, "bogus")
