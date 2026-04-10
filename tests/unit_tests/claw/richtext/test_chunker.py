# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.claw.richtext.chunker."""

from datus.claw.richtext.chunker import chunk_text


class TestChunkText:
    def test_short_text_no_split(self):
        text = "hello world"
        assert chunk_text(text) == [text]

    def test_paragraph_split(self):
        para1 = "a" * 40
        para2 = "b" * 40
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text(text, max_length=50)
        assert len(chunks) == 2
        assert chunks[0] == para1
        assert chunks[1] == para2

    def test_line_split(self):
        line1 = "a" * 30
        line2 = "b" * 30
        text = f"{line1}\n{line2}"
        chunks = chunk_text(text, max_length=35)
        assert len(chunks) == 2
        assert chunks[0] == line1
        assert chunks[1] == line2

    def test_hard_split(self):
        text = "x" * 100
        chunks = chunk_text(text, max_length=30)
        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks) == text

    def test_custom_limit(self):
        text = "a" * 10
        assert chunk_text(text, max_length=5) == ["aaaaa", "aaaaa"]

    def test_empty_string(self):
        assert chunk_text("") == [""]

    def test_exact_limit(self):
        text = "a" * 50
        assert chunk_text(text, max_length=50) == [text]

    def test_multiple_paragraphs_accumulate(self):
        text = "short\n\nalso short\n\nstill short"
        chunks = chunk_text(text, max_length=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_mixed_paragraph_and_line_split(self):
        long_para = "line1\nline2\nline3"
        short_para = "ok"
        text = f"{long_para}\n\n{short_para}"
        chunks = chunk_text(text, max_length=10)
        assert all(len(c) <= 10 for c in chunks)
