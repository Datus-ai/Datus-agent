# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.gateway.richtext.render."""

from datus.gateway.richtext.ir import LinkSpan, MarkdownIR, StyleSpan, StyleType
from datus.gateway.richtext.render import RenderOptions, StyleMarker, render_ir


def _slack_options() -> RenderOptions:
    return RenderOptions(
        style_markers={
            StyleType.BOLD: StyleMarker("*", "*"),
            StyleType.ITALIC: StyleMarker("_", "_"),
            StyleType.STRIKETHROUGH: StyleMarker("~", "~"),
            StyleType.CODE: StyleMarker("`", "`"),
            StyleType.CODE_BLOCK: StyleMarker("```\n", "\n```"),
            StyleType.BLOCKQUOTE: StyleMarker("> ", "\n"),
        },
        escape_fn=lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
        link_builder=lambda text, href: f"<{href}|{text}>",
    )


def _slack_options_legacy() -> RenderOptions:
    """Options including HEADING and TABLE for backward compat tests."""
    return RenderOptions(
        style_markers={
            StyleType.BOLD: StyleMarker("*", "*"),
            StyleType.ITALIC: StyleMarker("_", "_"),
            StyleType.STRIKETHROUGH: StyleMarker("~", "~"),
            StyleType.CODE: StyleMarker("`", "`"),
            StyleType.CODE_BLOCK: StyleMarker("```\n", "\n```"),
            StyleType.HEADING: StyleMarker("*", "*"),
            StyleType.TABLE: StyleMarker("```\n", "\n```"),
        },
        escape_fn=lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
        link_builder=lambda text, href: f"<{href}|{text}>",
    )


class TestRenderIrBasic:
    def test_empty_ir(self):
        ir = MarkdownIR(text="")
        assert render_ir(ir, RenderOptions()) == ""

    def test_plain_text_no_styles(self):
        ir = MarkdownIR(text="hello world")
        assert render_ir(ir, RenderOptions()) == "hello world"

    def test_bold(self):
        ir = MarkdownIR(
            text="hello bold world",
            styles=[StyleSpan(start=6, end=10, style=StyleType.BOLD)],
        )
        result = render_ir(ir, _slack_options())
        assert "*bold*" in result
        assert result == "hello *bold* world"

    def test_escape_fn(self):
        ir = MarkdownIR(text="a < b & c > d")
        opts = _slack_options()
        result = render_ir(ir, opts)
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_multiple_styles(self):
        ir = MarkdownIR(
            text="bold and italic",
            styles=[
                StyleSpan(start=0, end=4, style=StyleType.BOLD),
                StyleSpan(start=9, end=15, style=StyleType.ITALIC),
            ],
        )
        result = render_ir(ir, _slack_options())
        assert "*bold*" in result
        assert "_italic_" in result

    def test_unknown_style_ignored(self):
        ir = MarkdownIR(
            text="hello",
            styles=[StyleSpan(start=0, end=5, style=StyleType.BLOCKQUOTE)],
        )
        # Slack options don't have BLOCKQUOTE in this test helper — update to test skipping
        opts = RenderOptions(style_markers={StyleType.BOLD: StyleMarker("*", "*")})
        result = render_ir(ir, opts)
        assert result == "hello"


class TestRenderIrLinks:
    def test_link_with_builder(self):
        ir = MarkdownIR(
            text="click here please",
            links=[LinkSpan(start=0, end=10, href="https://example.com")],
        )
        opts = _slack_options()
        result = render_ir(ir, opts)
        assert "<https://example.com|click here>" in result

    def test_link_without_builder(self):
        ir = MarkdownIR(
            text="click here",
            links=[LinkSpan(start=0, end=10, href="https://example.com")],
        )
        opts = RenderOptions()
        result = render_ir(ir, opts)
        assert result == "click here"

    def test_link_with_styles(self):
        ir = MarkdownIR(
            text="bold link text",
            styles=[StyleSpan(start=5, end=9, style=StyleType.BOLD)],
            links=[LinkSpan(start=5, end=9, href="https://example.com")],
        )
        opts = _slack_options()
        result = render_ir(ir, opts)
        # The link builder wraps the rendered text (including style markers)
        assert "<https://example.com|" in result
        assert "link" in result


class TestRenderIrCodeBlock:
    def test_code_block(self):
        ir = MarkdownIR(
            text="print('hi')",
            styles=[StyleSpan(start=0, end=11, style=StyleType.CODE_BLOCK, meta={"language": "python"})],
        )
        opts = _slack_options()
        result = render_ir(ir, opts)
        assert result == "```\nprint('hi')\n```"


class TestRenderIrTable:
    def test_table_wrapped_in_code_block(self):
        table_text = "| A | B |\n| 1 | 2 |\n"
        ir = MarkdownIR(
            text=table_text,
            styles=[StyleSpan(start=0, end=len(table_text), style=StyleType.TABLE)],
        )
        result = render_ir(ir, _slack_options_legacy())
        assert result.startswith("```\n")
        assert result.endswith("\n```")
        assert "| A | B |" in result

    def test_table_roundtrip_markdown_to_slack_legacy(self):
        from datus.gateway.richtext.parser import markdown_to_ir

        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
        ir = markdown_to_ir(md)
        result = render_ir(ir, _slack_options_legacy())
        assert result.startswith("```\n")
        assert result.rstrip().endswith("```")
        assert "Name" in result
        assert "Alice" in result

    def test_table_bullets_roundtrip(self):
        from datus.gateway.richtext.parser import markdown_to_ir

        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        ir = markdown_to_ir(md, table_mode="bullets")
        result = render_ir(ir, _slack_options())
        assert "*Alice*" in result
        assert "*Bob*" in result
        assert "\u2022 Age: 30" in result
        assert "\u2022 Age: 25" in result


class TestRenderIrRoundTrip:
    def test_markdown_to_slack(self):
        """End-to-end: markdown -> IR -> Slack mrkdwn."""
        from datus.gateway.richtext.parser import markdown_to_ir

        ir = markdown_to_ir("**hello** world")
        result = render_ir(ir, _slack_options())
        assert "*hello*" in result
        assert "world" in result

    def test_markdown_code_to_slack(self):
        from datus.gateway.richtext.parser import markdown_to_ir

        ir = markdown_to_ir("Use `SELECT *` query")
        result = render_ir(ir, _slack_options())
        assert "`SELECT *`" in result
