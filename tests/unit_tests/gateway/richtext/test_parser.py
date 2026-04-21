# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.gateway.richtext.parser."""

from datus.gateway.richtext.ir import StyleType
from datus.gateway.richtext.parser import markdown_to_ir


class TestMarkdownToIrBasic:
    def test_empty_string(self):
        ir = markdown_to_ir("")
        assert ir.text == ""
        assert ir.styles == []
        assert ir.links == []

    def test_plain_text(self):
        ir = markdown_to_ir("hello world")
        assert ir.text.strip() == "hello world"
        assert ir.styles == []

    def test_bold(self):
        ir = markdown_to_ir("**bold**")
        assert "bold" in ir.text
        bold_spans = [s for s in ir.styles if s.style == StyleType.BOLD]
        assert len(bold_spans) == 1
        span = bold_spans[0]
        assert ir.text[span.start : span.end] == "bold"

    def test_italic(self):
        ir = markdown_to_ir("*italic*")
        italic_spans = [s for s in ir.styles if s.style == StyleType.ITALIC]
        assert len(italic_spans) == 1
        assert ir.text[italic_spans[0].start : italic_spans[0].end] == "italic"

    def test_strikethrough(self):
        ir = markdown_to_ir("~~deleted~~")
        strike_spans = [s for s in ir.styles if s.style == StyleType.STRIKETHROUGH]
        assert len(strike_spans) == 1
        assert ir.text[strike_spans[0].start : strike_spans[0].end] == "deleted"

    def test_inline_code(self):
        ir = markdown_to_ir("`code`")
        code_spans = [s for s in ir.styles if s.style == StyleType.CODE]
        assert len(code_spans) == 1
        assert ir.text[code_spans[0].start : code_spans[0].end] == "code"

    def test_code_block(self):
        ir = markdown_to_ir("```python\nprint('hi')\n```")
        cb_spans = [s for s in ir.styles if s.style == StyleType.CODE_BLOCK]
        assert len(cb_spans) == 1
        span = cb_spans[0]
        assert "print('hi')" in ir.text[span.start : span.end]
        assert span.meta == {"language": "python"}

    def test_code_block_no_language(self):
        ir = markdown_to_ir("```\nhello\n```")
        cb_spans = [s for s in ir.styles if s.style == StyleType.CODE_BLOCK]
        assert len(cb_spans) == 1
        assert cb_spans[0].meta is None

    def test_heading(self):
        ir = markdown_to_ir("# Title")
        h_spans = [s for s in ir.styles if s.style == StyleType.HEADING]
        assert len(h_spans) == 1
        assert ir.text[h_spans[0].start : h_spans[0].end] == "Title"
        assert h_spans[0].meta == {"level": 1}

    def test_heading_levels(self):
        ir = markdown_to_ir("## Sub\n### SubSub")
        h_spans = sorted([s for s in ir.styles if s.style == StyleType.HEADING], key=lambda s: s.start)
        assert len(h_spans) == 2
        assert h_spans[0].meta == {"level": 2}
        assert h_spans[1].meta == {"level": 3}

    def test_blockquote(self):
        ir = markdown_to_ir("> quoted text")
        bq_spans = [s for s in ir.styles if s.style == StyleType.BLOCKQUOTE]
        assert len(bq_spans) == 1
        assert "quoted text" in ir.text[bq_spans[0].start : bq_spans[0].end]

    def test_link(self):
        ir = markdown_to_ir("[click here](https://example.com)")
        assert len(ir.links) == 1
        link = ir.links[0]
        assert ir.text[link.start : link.end] == "click here"
        assert link.href == "https://example.com"


class TestMarkdownToIrNested:
    def test_bold_italic(self):
        ir = markdown_to_ir("***bold italic***")
        bold = [s for s in ir.styles if s.style == StyleType.BOLD]
        italic = [s for s in ir.styles if s.style == StyleType.ITALIC]
        assert len(bold) == 1
        assert len(italic) == 1
        # Both should cover the same text
        assert ir.text[bold[0].start : bold[0].end] == ir.text[italic[0].start : italic[0].end]

    def test_bold_with_inline_code(self):
        ir = markdown_to_ir("**bold `code` more**")
        bold = [s for s in ir.styles if s.style == StyleType.BOLD]
        code = [s for s in ir.styles if s.style == StyleType.CODE]
        assert len(bold) == 1
        assert len(code) == 1
        # code span should be within bold span
        assert bold[0].start <= code[0].start
        assert code[0].end <= bold[0].end

    def test_link_with_bold(self):
        ir = markdown_to_ir("[**bold link**](https://example.com)")
        bold = [s for s in ir.styles if s.style == StyleType.BOLD]
        assert len(bold) == 1
        assert len(ir.links) == 1
        assert ir.links[0].href == "https://example.com"


class TestMarkdownToIrTable:
    def test_simple_table(self):
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        ir = markdown_to_ir(md)
        assert "Name" in ir.text
        assert "Age" in ir.text
        assert "Alice" in ir.text
        assert "|" in ir.text

    def test_table_pipe_format(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        ir = markdown_to_ir(md)
        lines = ir.text.strip().split("\n")
        # header, separator, data row
        assert len(lines) == 3
        assert lines[0].startswith("|")
        assert "-" in lines[1] and "|" in lines[1]
        assert "1" in lines[2]
        assert "2" in lines[2]

    def test_table_preserves_content(self):
        md = "| Col1 | Col2 | Col3 |\n|------|------|------|\n| a | b | c |"
        ir = markdown_to_ir(md)
        assert "Col1" in ir.text
        assert "Col2" in ir.text
        assert "Col3" in ir.text
        assert "a" in ir.text

    def test_table_produces_table_style_span(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        ir = markdown_to_ir(md)
        table_spans = [s for s in ir.styles if s.style == StyleType.TABLE]
        assert len(table_spans) == 1
        span = table_spans[0]
        covered = ir.text[span.start : span.end]
        assert "|" in covered
        assert "A" in covered
        assert "1" in covered

    def test_table_with_surrounding_text(self):
        md = "Before\n\n| X | Y |\n|---|---|\n| 1 | 2 |\n\nAfter"
        ir = markdown_to_ir(md)
        assert "Before" in ir.text
        assert "After" in ir.text
        assert "| X" in ir.text
        table_spans = [s for s in ir.styles if s.style == StyleType.TABLE]
        assert len(table_spans) == 1


class TestMarkdownToIrEdgeCases:
    def test_multiple_paragraphs(self):
        ir = markdown_to_ir("para one\n\npara two")
        assert "para one" in ir.text
        assert "para two" in ir.text

    def test_list_items(self):
        ir = markdown_to_ir("- item one\n- item two")
        assert "item one" in ir.text
        assert "item two" in ir.text

    def test_horizontal_rule(self):
        ir = markdown_to_ir("above\n\n---\n\nbelow")
        assert "---" in ir.text

    def test_mixed_content(self):
        md = "# Hello\n\nThis is **bold** and *italic* with `code`.\n\n```sql\nSELECT 1\n```"
        ir = markdown_to_ir(md)
        assert "Hello" in ir.text
        assert "bold" in ir.text
        assert "italic" in ir.text
        assert "code" in ir.text
        assert "SELECT 1" in ir.text
        heading = [s for s in ir.styles if s.style == StyleType.HEADING]
        bold = [s for s in ir.styles if s.style == StyleType.BOLD]
        italic = [s for s in ir.styles if s.style == StyleType.ITALIC]
        code_inline = [s for s in ir.styles if s.style == StyleType.CODE]
        code_block = [s for s in ir.styles if s.style == StyleType.CODE_BLOCK]
        assert len(heading) == 1
        assert len(bold) == 1
        assert len(italic) == 1
        assert len(code_inline) == 1
        assert len(code_block) == 1


class TestHeadingStyle:
    def test_heading_style_bold(self):
        ir = markdown_to_ir("# Title", heading_style="bold")
        bold_spans = [s for s in ir.styles if s.style == StyleType.BOLD]
        heading_spans = [s for s in ir.styles if s.style == StyleType.HEADING]
        assert len(bold_spans) == 1
        assert len(heading_spans) == 0
        assert ir.text[bold_spans[0].start : bold_spans[0].end] == "Title"

    def test_heading_style_none(self):
        ir = markdown_to_ir("# Title", heading_style="none")
        heading_spans = [s for s in ir.styles if s.style == StyleType.HEADING]
        bold_spans = [s for s in ir.styles if s.style == StyleType.BOLD]
        assert len(heading_spans) == 0
        assert len(bold_spans) == 0
        assert "Title" in ir.text

    def test_heading_style_heading_default(self):
        ir = markdown_to_ir("# Title")
        heading_spans = [s for s in ir.styles if s.style == StyleType.HEADING]
        assert len(heading_spans) == 1
        assert heading_spans[0].meta == {"level": 1}

    def test_heading_style_bold_multiple_levels(self):
        ir = markdown_to_ir("# H1\n## H2", heading_style="bold")
        bold_spans = [s for s in ir.styles if s.style == StyleType.BOLD]
        assert len(bold_spans) == 2


class TestTableModes:
    def test_table_mode_off_default(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        ir = markdown_to_ir(md)
        table_spans = [s for s in ir.styles if s.style == StyleType.TABLE]
        assert len(table_spans) == 1
        assert "|" in ir.text

    def test_table_mode_bullets(self):
        md = "| Name | Age | City |\n|------|-----|------|\n| Alice | 30 | NYC |\n| Bob | 25 | LA |"
        ir = markdown_to_ir(md, table_mode="bullets")
        table_spans = [s for s in ir.styles if s.style == StyleType.TABLE]
        assert len(table_spans) == 0
        bold_spans = [s for s in ir.styles if s.style == StyleType.BOLD]
        assert len(bold_spans) >= 2  # Alice and Bob bolded
        assert "\u2022 Age:" in ir.text
        assert "\u2022 City:" in ir.text
        assert "Alice" in ir.text
        assert "Bob" in ir.text

    def test_table_mode_code(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        ir = markdown_to_ir(md, table_mode="code")
        code_block_spans = [s for s in ir.styles if s.style == StyleType.CODE_BLOCK]
        table_spans = [s for s in ir.styles if s.style == StyleType.TABLE]
        assert len(code_block_spans) == 1
        assert len(table_spans) == 0
        assert "|" in ir.text

    def test_table_mode_bullets_single_row(self):
        md = "| X | Y |\n|---|---|\n| a | b |"
        ir = markdown_to_ir(md, table_mode="bullets")
        assert "a" in ir.text
        assert "\u2022 Y: b" in ir.text


class TestBulletSymbol:
    def test_bullet_uses_unicode(self):
        ir = markdown_to_ir("- item one\n- item two")
        assert "\u2022 item one" in ir.text
        assert "\u2022 item two" in ir.text

    def test_nested_bullets(self):
        md = "- outer\n  - inner"
        ir = markdown_to_ir(md)
        assert "\u2022 outer" in ir.text
        assert "inner" in ir.text


class TestOrderedList:
    def test_ordered_list(self):
        ir = markdown_to_ir("1. first\n2. second\n3. third")
        assert "1. first" in ir.text
        assert "2. second" in ir.text
        assert "3. third" in ir.text

    def test_ordered_list_custom_start(self):
        ir = markdown_to_ir("3. alpha\n4. beta")
        assert "3. alpha" in ir.text
        assert "4. beta" in ir.text

    def test_mixed_list_types(self):
        md = "1. ordered\n\n- bullet"
        ir = markdown_to_ir(md)
        assert "1. ordered" in ir.text
        assert "\u2022 bullet" in ir.text
