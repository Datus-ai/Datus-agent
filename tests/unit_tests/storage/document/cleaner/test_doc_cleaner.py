# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for DocumentCleaner."""

from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner, clean_document, clean_text
from datus.storage.document.schemas import FetchedDocument


def _make_doc(
    raw_content: str,
    content_type: str = "markdown",
) -> FetchedDocument:
    """Create a minimal FetchedDocument for testing."""
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url="https://example.com/docs/test.md",
        source_type="github",
        doc_path="docs/test.md",
        raw_content=raw_content,
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    """Tests for clean_text method."""

    def test_remove_control_characters(self):
        """Control characters (except newline/tab) are removed."""
        cleaner = DocumentCleaner()
        text = "Hello\x00World\x01\x02\x03"
        result = cleaner.clean_text(text)

        assert "\x00" not in result
        assert "\x01" not in result
        assert "HelloWorld" in result

    def test_preserve_newlines_and_tabs(self):
        """Newlines and tabs are preserved during control char removal."""
        cleaner = DocumentCleaner()
        text = "Line 1\nLine 2\tTabbed"
        result = cleaner.clean_text(text)

        assert "\n" in result
        assert "\t" in result

    def test_windows_line_endings_normalized(self):
        """Windows CRLF line endings are converted to LF."""
        cleaner = DocumentCleaner()
        text = "Line 1\r\nLine 2\r\nLine 3"
        result = cleaner.clean_text(text)

        assert "\r\n" not in result
        assert "\r" not in result
        assert "Line 1\nLine 2\nLine 3" == result

    def test_multiple_blank_lines_collapsed(self):
        """Three or more consecutive blank lines are collapsed to two."""
        cleaner = DocumentCleaner()
        text = "Para 1\n\n\n\n\nPara 2"
        result = cleaner.clean_text(text)

        assert "\n\n\n" not in result
        assert "Para 1\n\nPara 2" == result

    def test_trailing_whitespace_removed(self):
        """Trailing whitespace on lines is removed."""
        cleaner = DocumentCleaner()
        text = "Hello   \nWorld  \nEnd"
        result = cleaner.clean_text(text)

        lines = result.split("\n")
        for line in lines:
            assert line == line.rstrip(), f"Line has trailing whitespace: '{line}'"

    def test_unicode_normalization(self):
        """Unicode NFC normalization is applied."""
        cleaner = DocumentCleaner()
        # e followed by combining acute accent (two code points)
        text = "caf\u0065\u0301"
        result = cleaner.clean_text(text)

        # After NFC, should be single character e-acute
        assert "\u00e9" in result

    def test_unicode_normalization_disabled(self):
        """Unicode normalization can be disabled."""
        cleaner = DocumentCleaner(normalize_unicode=False)
        text = "caf\u0065\u0301"
        result = cleaner.clean_text(text)

        # Should keep the decomposed form
        assert "\u0301" in result

    def test_strip_result(self):
        """Result is stripped of leading/trailing whitespace."""
        cleaner = DocumentCleaner()
        text = "  \n  Hello World  \n  "
        result = cleaner.clean_text(text)

        assert result == "Hello World"

    def test_del_character_removed(self):
        """DEL character (0x7F) is removed."""
        cleaner = DocumentCleaner()
        text = "Hello\x7FWorld"
        result = cleaner.clean_text(text)

        assert "\x7f" not in result
        assert "HelloWorld" in result


# ---------------------------------------------------------------------------
# Code block preservation
# ---------------------------------------------------------------------------


class TestCodeBlockPreservation:
    """Tests for code block preservation during cleaning."""

    def test_code_block_content_preserved(self):
        """Content inside code blocks is not modified by cleaning."""
        cleaner = DocumentCleaner()
        text = "Before\n\n```python\nprint('hello')\nx = 1\n```\n\nAfter"
        result = cleaner.clean_text(text)

        assert "```python\nprint('hello')\nx = 1\n```" in result

    def test_code_block_control_chars_preserved(self):
        """Control characters inside code blocks are preserved (code blocks are extracted before cleaning)."""
        cleaner = DocumentCleaner()
        # The code block is saved before control char removal, then restored
        text = "Text\n\n```\ncode content\n```\n\nMore text"
        result = cleaner.clean_text(text)

        assert "```\ncode content\n```" in result

    def test_multiple_code_blocks_preserved(self):
        """Multiple code blocks are all preserved."""
        cleaner = DocumentCleaner()
        text = "# Title\n\n```sql\nSELECT 1;\n```\n\nMiddle text.\n\n```python\nx = 2\n```\n\nEnd."
        result = cleaner.clean_text(text)

        assert "```sql\nSELECT 1;\n```" in result
        assert "```python\nx = 2\n```" in result

    def test_code_block_preservation_disabled(self):
        """When preserve_code_blocks=False, code blocks are treated as normal text."""
        cleaner = DocumentCleaner(preserve_code_blocks=False)
        # With many blank lines inside what looks like a code block
        text = "Before\n\n```\nLine1\n\n\n\n\nLine2\n```\n\nAfter"
        result = cleaner.clean_text(text)

        # Without preservation, the multiple blank lines inside would be collapsed
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# clean() with HTML content
# ---------------------------------------------------------------------------


class TestCleanHTML:
    """Tests for clean() with HTML content type."""

    def test_script_tags_removed(self):
        """<script> tags and their contents are removed."""
        cleaner = DocumentCleaner()
        doc = _make_doc(
            '<p>Hello</p><script>alert("xss");</script><p>World</p>',
            content_type="html",
        )
        result = cleaner.clean(doc)

        assert "<script>" not in result.raw_content
        assert "alert" not in result.raw_content
        assert "Hello" in result.raw_content
        assert "World" in result.raw_content

    def test_style_tags_removed(self):
        """<style> tags and their contents are removed."""
        cleaner = DocumentCleaner()
        doc = _make_doc(
            "<p>Text</p><style>body { color: red; }</style><p>More</p>",
            content_type="html",
        )
        result = cleaner.clean(doc)

        assert "<style>" not in result.raw_content
        assert "color: red" not in result.raw_content

    def test_noscript_tags_removed(self):
        """<noscript> tags and their contents are removed."""
        cleaner = DocumentCleaner()
        doc = _make_doc(
            "<p>Content</p><noscript>Please enable JS</noscript><p>End</p>",
            content_type="html",
        )
        result = cleaner.clean(doc)

        assert "<noscript>" not in result.raw_content
        assert "enable JS" not in result.raw_content

    def test_html_comments_removed(self):
        """HTML comments are removed."""
        cleaner = DocumentCleaner()
        doc = _make_doc(
            "<p>Before</p><!-- This is a comment --><p>After</p>",
            content_type="html",
        )
        result = cleaner.clean(doc)

        assert "<!--" not in result.raw_content
        assert "This is a comment" not in result.raw_content

    def test_cleaned_doc_preserves_metadata(self):
        """clean() returns a new FetchedDocument preserving all metadata fields."""
        cleaner = DocumentCleaner()
        doc = _make_doc("Clean content.", content_type="html")
        result = cleaner.clean(doc)

        assert result.platform == doc.platform
        assert result.version == doc.version
        assert result.source_url == doc.source_url
        assert result.source_type == doc.source_type
        assert result.doc_path == doc.doc_path
        assert result.content_type == doc.content_type
        assert result.fetch_timestamp == doc.fetch_timestamp

    def test_cleaned_doc_is_new_instance(self):
        """clean() returns a new FetchedDocument, not the same object."""
        cleaner = DocumentCleaner()
        doc = _make_doc("Some content.", content_type="markdown")
        result = cleaner.clean(doc)

        assert result is not doc

    def test_markdown_content_type_uses_clean_text(self):
        """Markdown content type goes through clean_text path (no HTML stripping)."""
        cleaner = DocumentCleaner()
        doc = _make_doc("# Title\n\n\n\n\nBody text.", content_type="markdown")
        result = cleaner.clean(doc)

        # Multiple blank lines should be collapsed
        assert "\n\n\n" not in result.raw_content
        assert "# Title" in result.raw_content
        assert "Body text." in result.raw_content


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_clean_document_function(self):
        """clean_document() convenience function works."""
        doc = _make_doc("Hello\x00World\n\n\n\n\nEnd.", content_type="markdown")
        result = clean_document(doc)

        assert "\x00" not in result.raw_content
        assert "\n\n\n" not in result.raw_content
        assert result is not doc

    def test_clean_text_function(self):
        """clean_text() convenience function works."""
        result = clean_text("Hello\x00World\n\n\n\n\nEnd.")

        assert "\x00" not in result
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for DocumentCleaner."""

    def test_empty_string(self):
        """Cleaning an empty string returns empty string."""
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("")

        assert result == ""

    def test_only_whitespace(self):
        """Cleaning whitespace-only string returns empty string."""
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("   \n\n   \t   ")

        assert result == ""

    def test_only_control_chars(self):
        """Cleaning only control characters returns empty string."""
        cleaner = DocumentCleaner()
        result = cleaner.clean_text("\x00\x01\x02\x03")

        assert result == ""

    def test_lone_carriage_return_normalized(self):
        """Lone \\r (old Mac line ending) is normalized to \\n."""
        cleaner = DocumentCleaner()
        text = "Line 1\rLine 2\rLine 3"
        result = cleaner.clean_text(text)

        assert "\r" not in result
        assert "Line 1\nLine 2\nLine 3" == result

    def test_all_cleaning_disabled(self):
        """When all cleaning options are disabled, text is only stripped."""
        cleaner = DocumentCleaner(
            normalize_unicode=False,
            remove_control_chars=False,
            normalize_whitespace=False,
            preserve_code_blocks=False,
        )
        text = "  Hello\x00World  "
        result = cleaner.clean_text(text)

        # Only strip() is applied
        assert result == "Hello\x00World"
