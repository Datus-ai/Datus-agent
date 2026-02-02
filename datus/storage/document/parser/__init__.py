# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Parser Module

Provides parsers for different document formats:
- Markdown (using markdown-it-py)
- HTML (using BeautifulSoup4)
"""

from datus.storage.document.parser.html_parser import HTMLParser
from datus.storage.document.parser.markdown_parser import MarkdownParser
from datus.storage.document.parser.metadata_extractor import MetadataExtractor

__all__ = [
    "MarkdownParser",
    "HTMLParser",
    "MetadataExtractor",
]
