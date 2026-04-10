# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.claw.richtext.ir data models."""

from datus.claw.richtext.ir import LinkSpan, MarkdownIR, StyleSpan, StyleType


class TestStyleType:
    def test_enum_values(self):
        assert StyleType.BOLD == "bold"
        assert StyleType.ITALIC == "italic"
        assert StyleType.STRIKETHROUGH == "strikethrough"
        assert StyleType.CODE == "code"
        assert StyleType.CODE_BLOCK == "code_block"
        assert StyleType.BLOCKQUOTE == "blockquote"
        assert StyleType.HEADING == "heading"
        assert StyleType.TABLE == "table"

    def test_is_string_enum(self):
        assert isinstance(StyleType.BOLD, str)


class TestStyleSpan:
    def test_basic(self):
        span = StyleSpan(start=0, end=5, style=StyleType.BOLD)
        assert span.start == 0
        assert span.end == 5
        assert span.style == StyleType.BOLD
        assert span.meta is None

    def test_with_meta(self):
        span = StyleSpan(start=0, end=5, style=StyleType.HEADING, meta={"level": 2})
        assert span.meta == {"level": 2}


class TestLinkSpan:
    def test_basic(self):
        link = LinkSpan(start=0, end=6, href="https://example.com")
        assert link.start == 0
        assert link.end == 6
        assert link.href == "https://example.com"


class TestMarkdownIR:
    def test_empty(self):
        ir = MarkdownIR(text="")
        assert ir.text == ""
        assert ir.styles == []
        assert ir.links == []

    def test_with_styles_and_links(self):
        ir = MarkdownIR(
            text="hello world",
            styles=[StyleSpan(start=0, end=5, style=StyleType.BOLD)],
            links=[LinkSpan(start=6, end=11, href="https://example.com")],
        )
        assert len(ir.styles) == 1
        assert len(ir.links) == 1
        assert ir.text == "hello world"

    def test_serialization_roundtrip(self):
        ir = MarkdownIR(
            text="test",
            styles=[StyleSpan(start=0, end=4, style=StyleType.CODE, meta={"language": "py"})],
        )
        data = ir.model_dump()
        ir2 = MarkdownIR.model_validate(data)
        assert ir2.text == ir.text
        assert ir2.styles[0].style == StyleType.CODE
        assert ir2.styles[0].meta == {"language": "py"}
