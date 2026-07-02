# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the shared disk-archive primitive.

The archive helpers moved from ``datus.agent.node.compact_archive`` to
``datus.utils.tool_archive`` so both the compact pass and the bash tool can
share them. These tests cover the relocated primitives and the round-trip
between :func:`build_archived_marker` and :func:`parse_archived_marker`.
"""

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


class TestIsErrorOutput:
    def test_funcresult_success_zero_is_error(self):
        assert is_error_output('{"success": 0, "error": "boom"}')

    def test_success_one_not_error(self):
        assert not is_error_output('{"success": 1, "result": "ok"}')

    def test_non_json_traceback_is_error(self):
        assert is_error_output("Traceback (most recent call last): ...")


class TestToolArchivePrimitives:
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
