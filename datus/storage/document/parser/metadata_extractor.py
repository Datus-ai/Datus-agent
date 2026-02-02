# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Metadata Extractor

Extracts metadata from documents including version, keywords, and language.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from datus.storage.document.schemas import ParsedDocument
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """Extracts metadata from documents.

    Features:
    - Version detection from content and URLs
    - Keyword extraction
    - Language detection
    - Platform-specific metadata

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(parsed_doc)
        >>> print(metadata["version"], metadata["keywords"])
    """

    # Common SQL/Database keywords for extraction
    SQL_KEYWORDS = {
        "select",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "table",
        "index",
        "view",
        "function",
        "procedure",
        "trigger",
        "database",
        "schema",
        "column",
        "primary",
        "foreign",
        "key",
        "constraint",
        "join",
        "inner",
        "outer",
        "left",
        "right",
        "where",
        "group",
        "order",
        "having",
        "limit",
        "offset",
        "union",
        "intersect",
        "except",
        "case",
        "when",
        "then",
        "else",
        "end",
        "cast",
        "convert",
        "null",
        "not",
        "and",
        "or",
        "in",
        "between",
        "like",
        "exists",
        "distinct",
        "aggregate",
        "window",
        "partition",
        "over",
        "rank",
        "row_number",
        "lag",
        "lead",
        "first_value",
        "last_value",
    }

    # Platform-specific keywords
    PLATFORM_KEYWORDS = {
        "snowflake": {
            "warehouse",
            "stage",
            "pipe",
            "stream",
            "task",
            "snowpipe",
            "time_travel",
            "clone",
            "share",
            "secure",
            "external",
            "variant",
            "object",
            "array",
            "flatten",
            "lateral",
        },
        "duckdb": {
            "parquet",
            "csv",
            "json",
            "arrow",
            "httpfs",
            "s3",
            "extension",
            "pragma",
            "copy",
            "export",
            "import",
        },
        "postgresql": {
            "vacuum",
            "analyze",
            "explain",
            "toast",
            "tablespace",
            "sequence",
            "serial",
            "array",
            "jsonb",
            "hstore",
            "citext",
            "uuid",
            "inet",
            "macaddr",
            "range",
            "composite",
        },
        "bigquery": {
            "dataset",
            "project",
            "partition",
            "clustering",
            "unnest",
            "struct",
            "array",
            "repeated",
            "nested",
            "standard_sql",
            "legacy_sql",
        },
    }

    # Version patterns
    VERSION_PATTERNS = [
        # Semantic version: v1.2.3, 1.2.3
        (r"v?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)", 1.0),
        # Date version: 2024-01, 2024.01
        (r"(20\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?)", 0.8),
        # Major version only: v15, version 15
        (r"(?:v|version\s*)(\d+)", 0.5),
    ]

    def __init__(self):
        """Initialize the metadata extractor."""
        self._version_patterns = [
            (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in self.VERSION_PATTERNS
        ]

    def extract(
        self,
        doc: ParsedDocument,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract metadata from a parsed document.

        Args:
            doc: Parsed document
            platform: Platform name for platform-specific extraction

        Returns:
            Metadata dictionary with version, keywords, language, etc.
        """
        metadata = dict(doc.metadata)

        # Get full text content
        full_text = self._get_full_text(doc)

        # Extract version
        if "version" not in metadata:
            source_doc = doc.source_doc
            if source_doc:
                metadata["version"] = source_doc.version
            else:
                detected_version = self._detect_version(full_text)
                if detected_version:
                    metadata["version"] = detected_version

        # Extract keywords
        keywords = self._extract_keywords(full_text, platform)
        metadata["keywords"] = keywords

        # Detect language
        metadata["language"] = self._detect_language(full_text)

        # Extract additional metadata
        metadata["word_count"] = len(full_text.split())
        metadata["has_code_blocks"] = "```" in full_text
        metadata["has_tables"] = "|---|" in full_text or "| " in full_text

        return metadata

    def extract_keywords(
        self,
        text: str,
        platform: Optional[str] = None,
        max_keywords: int = 10,
    ) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Text content
            platform: Platform for platform-specific keywords
            max_keywords: Maximum number of keywords to return

        Returns:
            List of extracted keywords
        """
        return self._extract_keywords(text, platform, max_keywords)

    def _get_full_text(self, doc: ParsedDocument) -> str:
        """Get full text content from parsed document.

        Args:
            doc: Parsed document

        Returns:
            Combined text content
        """
        parts = [doc.title]

        def collect_content(section):
            if section.title:
                parts.append(section.title)
            if section.content:
                parts.append(section.content)
            for child in section.children:
                collect_content(child)

        for section in doc.sections:
            collect_content(section)

        return "\n\n".join(parts)

    def _detect_version(self, text: str) -> Optional[str]:
        """Detect version from text content.

        Args:
            text: Text content

        Returns:
            Detected version or None
        """
        candidates = []

        for pattern, weight in self._version_patterns:
            matches = pattern.findall(text)
            for match in matches:
                version = match if isinstance(match, str) else match[0]
                candidates.append((version, weight))

        if not candidates:
            return None

        # Return highest weighted match
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _extract_keywords(
        self,
        text: str,
        platform: Optional[str] = None,
        max_keywords: int = 10,
    ) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Text content
            platform: Platform for platform-specific keywords
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        # Lowercase and tokenize
        text_lower = text.lower()
        words = re.findall(r"\b[a-z_][a-z0-9_]*\b", text_lower)

        # Count word frequencies
        word_counts = Counter(words)

        # Get target keywords based on SQL and platform-specific terms
        target_keywords: Set[str] = set(self.SQL_KEYWORDS)
        if platform and platform in self.PLATFORM_KEYWORDS:
            target_keywords.update(self.PLATFORM_KEYWORDS[platform])

        # Find matching keywords
        found_keywords = []
        for word, count in word_counts.most_common():
            if word in target_keywords and count >= 2:
                found_keywords.append(word)
                if len(found_keywords) >= max_keywords:
                    break

        # Also check for compound terms
        compound_terms = self._find_compound_terms(text_lower)
        for term in compound_terms:
            if len(found_keywords) < max_keywords and term not in found_keywords:
                found_keywords.append(term)

        return found_keywords[:max_keywords]

    def _find_compound_terms(self, text: str) -> List[str]:
        """Find compound technical terms.

        Args:
            text: Lowercase text

        Returns:
            List of compound terms found
        """
        compound_patterns = [
            r"create\s+table",
            r"primary\s+key",
            r"foreign\s+key",
            r"inner\s+join",
            r"left\s+join",
            r"right\s+join",
            r"full\s+outer\s+join",
            r"group\s+by",
            r"order\s+by",
            r"window\s+function",
            r"common\s+table\s+expression",
            r"materialized\s+view",
            r"stored\s+procedure",
            r"user\s+defined\s+function",
            r"data\s+type",
            r"time\s+travel",
        ]

        found = []
        for pattern in compound_patterns:
            if re.search(pattern, text):
                # Convert to underscore format
                term = re.sub(r"\s+", "_", pattern.replace(r"\s+", " "))
                found.append(term)

        return found

    def _detect_language(self, text: str) -> str:
        """Detect language of text.

        Simple heuristic based on character frequency.

        Args:
            text: Text content

        Returns:
            Language code ('en', 'zh', etc.)
        """
        # Count CJK characters
        cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))

        # If more than 10% CJK, likely Chinese
        if cjk_count > len(text) * 0.1:
            return "zh"

        # Check for other scripts
        cyrillic_count = len(re.findall(r"[\u0400-\u04ff]", text))
        if cyrillic_count > len(text) * 0.1:
            return "ru"

        # Default to English
        return "en"
