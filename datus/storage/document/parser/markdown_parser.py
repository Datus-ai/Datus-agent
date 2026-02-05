# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Markdown Document Parser

Parses Markdown documents into a structured format with hierarchical sections.
Uses markdown-it-py for accurate AST parsing.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from datus.storage.document.schemas import FetchedDocument, ParsedDocument, ParsedSection
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Check if markdown-it-py is available
try:
    from markdown_it import MarkdownIt
    from markdown_it.tree import SyntaxTreeNode

    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    MarkdownIt = None
    SyntaxTreeNode = None


class MarkdownParser:
    """Parser for Markdown documents.

    Converts Markdown content into a hierarchical structure of sections,
    preserving heading levels and code blocks.

    Features:
    - Hierarchical section extraction based on headings
    - Code block preservation
    - Metadata extraction from frontmatter
    - Link and image reference collection

    Example:
        >>> parser = MarkdownParser()
        >>> parsed = parser.parse(fetched_doc)
        >>> print(parsed.title)
        >>> for section in parsed.sections:
        ...     print(f"{section.level}: {section.title}")
    """

    def __init__(self):
        """Initialize the Markdown parser."""
        if MARKDOWN_IT_AVAILABLE:
            self._md = MarkdownIt("commonmark", {"breaks": True, "html": True})
            self._md.enable("table")
        else:
            self._md = None
            logger.warning(
                "markdown-it-py not available. Using fallback regex parser. " "Install with: pip install markdown-it-py"
            )

    def parse(self, doc: FetchedDocument) -> ParsedDocument:
        """Parse a fetched Markdown document.

        Args:
            doc: Fetched document with Markdown content

        Returns:
            Parsed document with hierarchical sections
        """
        content = doc.raw_content

        # Extract frontmatter if present
        metadata, content = self._extract_frontmatter(content)

        # Parse into sections
        if MARKDOWN_IT_AVAILABLE:
            sections = self._parse_with_markdown_it(content)
        else:
            sections = self._parse_with_regex(content)

        # Extract title (first h1 or from frontmatter)
        title = metadata.get("title", "")
        if not title and sections:
            for section in sections:
                if section.level == 1 and section.title:
                    title = section.title
                    break

        # If still no title, use doc_path
        if not title:
            title = doc.doc_path.split("/")[-1].replace(".md", "").replace("-", " ").title()

        return ParsedDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            source_doc=doc,
        )

    def _extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from Markdown.

        Args:
            content: Markdown content

        Returns:
            Tuple of (metadata dict, remaining content)
        """
        metadata = {}

        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                content = parts[2].strip()

                # Simple YAML parsing (key: value)
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        metadata[key] = value

                # Extract nav-relevant hints for framework resolvers
                _NAV_HINT_KEYS = {"sidebar_position", "sidebar_label", "weight", "slug", "linkTitle"}
                nav_hints = {k: metadata[k] for k in _NAV_HINT_KEYS if k in metadata}
                if nav_hints:
                    metadata["_nav_hints"] = nav_hints

        return metadata, content

    def _parse_with_markdown_it(self, content: str) -> List[ParsedSection]:
        """Parse Markdown using markdown-it-py.

        Args:
            content: Markdown content

        Returns:
            List of parsed sections
        """
        tokens = self._md.parse(content)

        sections = []
        current_section: Optional[ParsedSection] = None
        current_content_parts: List[str] = []
        section_stack: List[ParsedSection] = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                # Save current section content
                if current_section is not None:
                    current_section.content = "\n\n".join(current_content_parts).strip()
                    current_content_parts = []

                # Get heading level and text
                level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.

                # Get heading text from next token
                title = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    title = tokens[i + 1].content

                # Create new section
                new_section = ParsedSection(
                    level=level,
                    title=title,
                    content="",
                    children=[],
                )

                # Determine parent
                if not section_stack:
                    # Top-level section
                    sections.append(new_section)
                    section_stack.append(new_section)
                else:
                    # Find appropriate parent (section with lower level)
                    while section_stack and section_stack[-1].level >= level:
                        section_stack.pop()

                    if section_stack:
                        section_stack[-1].children.append(new_section)
                    else:
                        sections.append(new_section)

                    section_stack.append(new_section)

                current_section = new_section

            elif token.type == "fence":
                # Code block - preserve with markers
                lang = token.info or ""
                code = token.content
                code_block = f"```{lang}\n{code}```"
                current_content_parts.append(code_block)

            elif token.type == "code_block":
                # Indented code block
                code_block = f"```\n{token.content}```"
                current_content_parts.append(code_block)

            elif token.type == "paragraph_open":
                # Collect paragraph content
                para_content = []
                i += 1
                while i < len(tokens) and tokens[i].type != "paragraph_close":
                    if tokens[i].type == "inline":
                        para_content.append(tokens[i].content)
                    i += 1
                if para_content:
                    current_content_parts.append(" ".join(para_content))

            elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
                # Collect list items
                list_content = self._extract_list_content(tokens, i)
                if list_content:
                    current_content_parts.append(list_content)
                # Skip to end of list
                depth = 1
                while i < len(tokens) and depth > 0:
                    i += 1
                    if i >= len(tokens):
                        break
                    if tokens[i].type.endswith("_list_open"):
                        depth += 1
                    elif tokens[i].type.endswith("_list_close"):
                        depth -= 1

            elif token.type == "table_open":
                # Extract table as markdown
                table_content = self._extract_table_content(tokens, i)
                if table_content:
                    current_content_parts.append(table_content)
                # Skip to end of table
                while i < len(tokens) and tokens[i].type != "table_close":
                    i += 1

            elif token.type == "blockquote_open":
                # Extract blockquote
                quote_content = []
                i += 1
                while i < len(tokens) and tokens[i].type != "blockquote_close":
                    if tokens[i].type == "inline":
                        quote_content.append(tokens[i].content)
                    i += 1
                if quote_content:
                    quoted = "\n".join(f"> {line}" for line in quote_content)
                    current_content_parts.append(quoted)

            i += 1

        # Save final section content
        if current_section is not None:
            current_section.content = "\n\n".join(current_content_parts).strip()
        elif current_content_parts and not sections:
            # Content before any heading
            sections.insert(
                0,
                ParsedSection(
                    level=0,
                    title="",
                    content="\n\n".join(current_content_parts).strip(),
                    children=[],
                ),
            )

        return sections

    def _extract_list_content(self, tokens, start_idx: int) -> str:
        """Extract list content as markdown text."""
        items = []
        is_ordered = tokens[start_idx].type == "ordered_list_open"
        item_num = 1
        depth = 0

        i = start_idx + 1
        while i < len(tokens):
            token = tokens[i]

            if token.type == "list_item_open":
                depth += 1
            elif token.type == "list_item_close":
                depth -= 1
            elif token.type.endswith("_list_close") and depth == 0:
                break
            elif token.type == "inline" and depth == 1:
                prefix = f"{item_num}. " if is_ordered else "- "
                items.append(f"{prefix}{token.content}")
                if is_ordered:
                    item_num += 1

            i += 1

        return "\n".join(items)

    def _extract_table_content(self, tokens, start_idx: int) -> str:
        """Extract table content as markdown."""
        rows = []
        current_row = []
        is_header = True

        i = start_idx + 1
        while i < len(tokens) and tokens[i].type != "table_close":
            token = tokens[i]

            if token.type == "tr_open":
                current_row = []
            elif token.type == "tr_close":
                if current_row:
                    rows.append("| " + " | ".join(current_row) + " |")
                    if is_header and len(rows) == 1:
                        # Add separator after header
                        rows.append("|" + "|".join(["---"] * len(current_row)) + "|")
                        is_header = False
            elif token.type == "inline":
                current_row.append(token.content)

            i += 1

        return "\n".join(rows)

    def _parse_with_regex(self, content: str) -> List[ParsedSection]:
        """Fallback parser using regex when markdown-it-py is not available.

        Args:
            content: Markdown content

        Returns:
            List of parsed sections
        """
        # Split by headings
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections = []
        last_end = 0
        section_stack: List[ParsedSection] = []

        for match in heading_pattern.finditer(content):
            # Content before this heading
            pre_content = content[last_end : match.start()].strip()

            # Add pre-content to current section or create intro section
            if pre_content:
                if section_stack:
                    section_stack[-1].content += "\n\n" + pre_content
                elif not sections:
                    # Content before first heading
                    sections.append(ParsedSection(level=0, title="", content=pre_content, children=[]))

            # Create new section
            level = len(match.group(1))
            title = match.group(2).strip()

            new_section = ParsedSection(level=level, title=title, content="", children=[])

            # Find parent
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()

            if section_stack:
                section_stack[-1].children.append(new_section)
            else:
                sections.append(new_section)

            section_stack.append(new_section)
            last_end = match.end()

        # Remaining content
        if last_end < len(content):
            remaining = content[last_end:].strip()
            if remaining and section_stack:
                section_stack[-1].content += "\n\n" + remaining

        return sections
