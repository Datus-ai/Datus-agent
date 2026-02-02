# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Cleaner

Cleans and normalizes document content for better processing and storage.
"""

import re
import unicodedata

from datus.storage.document.schemas import FetchedDocument
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DocumentCleaner:
    """Cleans document content for processing.

    Features:
    - Unicode normalization (NFC)
    - Control character removal
    - Whitespace normalization
    - Code block preservation
    - Navigation/boilerplate removal (for HTML)

    Example:
        >>> cleaner = DocumentCleaner()
        >>> cleaned_doc = cleaner.clean(fetched_doc)
    """

    # Patterns for content to remove
    REMOVE_PATTERNS = [
        # Multiple consecutive blank lines
        (r"\n{3,}", "\n\n"),
        # Trailing whitespace on lines
        (r"[ \t]+$", "", re.MULTILINE),
        # Multiple spaces (not in code blocks)
        (r"(?<!\n)  +(?!\n)", " "),
    ]

    # HTML boilerplate patterns
    HTML_BOILERPLATE_PATTERNS = [
        r"<!-- .*? -->",  # HTML comments
        r"<script[\s\S]*?</script>",  # Script tags
        r"<style[\s\S]*?</style>",  # Style tags
        r"<noscript[\s\S]*?</noscript>",  # NoScript tags
    ]

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True,
        preserve_code_blocks: bool = True,
    ):
        """Initialize the cleaner.

        Args:
            normalize_unicode: Apply NFC Unicode normalization
            remove_control_chars: Remove control characters
            normalize_whitespace: Normalize excessive whitespace
            preserve_code_blocks: Protect code blocks from modification
        """
        self.normalize_unicode = normalize_unicode
        self.remove_control_chars = remove_control_chars
        self.normalize_whitespace = normalize_whitespace
        self.preserve_code_blocks = preserve_code_blocks

        # Compile patterns - each pattern is (regex, replacement, [flags])
        self._remove_patterns = []
        for item in self.REMOVE_PATTERNS:
            pattern = item[0]
            replacement = item[1]
            flags = item[2] if len(item) > 2 else 0
            self._remove_patterns.append((re.compile(pattern, flags), replacement))

        self._html_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.HTML_BOILERPLATE_PATTERNS]

    def clean(self, doc: FetchedDocument) -> FetchedDocument:
        """Clean a fetched document.

        Args:
            doc: Document to clean

        Returns:
            Cleaned document (new instance)
        """
        content = doc.raw_content

        # Clean based on content type
        if doc.content_type == "html":
            content = self._clean_html(content)
        else:
            content = self._clean_markdown(content)

        # Create new document with cleaned content
        return FetchedDocument(
            platform=doc.platform,
            version=doc.version,
            source_url=doc.source_url,
            source_type=doc.source_type,
            doc_path=doc.doc_path,
            raw_content=content,
            content_type=doc.content_type,
            metadata=doc.metadata,
            fetch_timestamp=doc.fetch_timestamp,
        )

    def clean_text(self, text: str) -> str:
        """Clean raw text content.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        code_blocks = []

        if self.preserve_code_blocks:
            # Extract and preserve code blocks using a unique placeholder
            # that won't be affected by control char removal
            code_pattern = re.compile(r"```[\s\S]*?```")

            def save_code(match):
                code_blocks.append(match.group(0))
                return f"__DATUS_CODE_BLOCK_{len(code_blocks) - 1}__"

            text = code_pattern.sub(save_code, text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Remove control characters (except newlines and tabs)
        if self.remove_control_chars:
            text = self._remove_control_chars(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Restore code blocks
        if self.preserve_code_blocks:
            for i, code in enumerate(code_blocks):
                text = text.replace(f"__DATUS_CODE_BLOCK_{i}__", code)

        return text.strip()

    def _clean_markdown(self, content: str) -> str:
        """Clean Markdown content.

        Args:
            content: Markdown content

        Returns:
            Cleaned content
        """
        return self.clean_text(content)

    def _clean_html(self, content: str) -> str:
        """Clean HTML content.

        Removes boilerplate before general cleaning.

        Args:
            content: HTML content

        Returns:
            Cleaned content
        """
        # Remove HTML boilerplate
        for pattern in self._html_patterns:
            content = pattern.sub("", content)

        # Apply general text cleaning
        return self.clean_text(content)

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs.

        Args:
            text: Input text

        Returns:
            Text with control characters removed
        """
        # Keep: newline (0x0A), carriage return (0x0D), tab (0x09)
        # Remove: other control characters (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F)
        return "".join(char for char in text if char in "\n\r\t" or (ord(char) >= 32 and ord(char) != 127))

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Convert Windows line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove trailing whitespace on each line
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)

        # Collapse multiple blank lines to maximum of 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text


def clean_document(doc: FetchedDocument) -> FetchedDocument:
    """Convenience function to clean a document.

    Args:
        doc: Document to clean

    Returns:
        Cleaned document
    """
    cleaner = DocumentCleaner()
    return cleaner.clean(doc)


def clean_text(text: str) -> str:
    """Convenience function to clean text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    cleaner = DocumentCleaner()
    return cleaner.clean_text(text)
