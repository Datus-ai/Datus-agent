# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Intermediate Representation (IR) data models for rich text."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class StyleType(str, Enum):
    """Supported rich text style types."""

    BOLD = "bold"
    ITALIC = "italic"
    STRIKETHROUGH = "strikethrough"
    CODE = "code"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    HEADING = "heading"
    TABLE = "table"


class StyleSpan(BaseModel):
    """A style annotation on a range of plain text."""

    start: int
    end: int
    style: StyleType
    meta: Optional[dict] = None


class LinkSpan(BaseModel):
    """A hyperlink annotation on a range of plain text."""

    start: int
    end: int
    href: str


class MarkdownIR(BaseModel):
    """Span-based intermediate representation of rich text.

    ``text`` contains the plain text with all formatting stripped.
    ``styles`` and ``links`` annotate ranges of that plain text.
    """

    text: str
    styles: list[StyleSpan] = []
    links: list[LinkSpan] = []
