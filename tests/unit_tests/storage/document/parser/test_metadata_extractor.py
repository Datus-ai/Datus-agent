# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for MetadataExtractor."""

from datus.storage.document.parser.metadata_extractor import MetadataExtractor
from datus.storage.document.schemas import FetchedDocument, ParsedDocument, ParsedSection


def _make_parsed_doc(
    title: str = "Test Doc",
    content: str = "Some content.",
    sections: list | None = None,
    source_doc: FetchedDocument | None = None,
    metadata: dict | None = None,
) -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
    if sections is None:
        sections = [ParsedSection(level=1, title=title, content=content, children=[])]
    return ParsedDocument(
        title=title,
        sections=sections,
        metadata=metadata or {},
        source_doc=source_doc,
    )


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    """Tests for extract_keywords (public API)."""

    def test_extract_sql_keywords(self):
        """SQL keywords mentioned >= 2 times are extracted."""
        extractor = MetadataExtractor()
        text = (
            "Use SELECT to query data. The SELECT statement supports WHERE clauses. "
            "You can also use JOIN to combine tables. The JOIN operation is powerful. "
            "Use GROUP BY for aggregation. GROUP BY works with aggregate functions."
        )
        keywords = extractor.extract_keywords(text)

        assert "select" in keywords
        assert "join" in keywords
        assert "group" in keywords

    def test_extract_keywords_snowflake_platform(self):
        """Snowflake-specific keywords are recognized when platform='snowflake'."""
        extractor = MetadataExtractor()
        text = (
            "A warehouse is a compute resource. Configure your warehouse for optimal performance. "
            "Use a stage to load data. The stage can be internal or external. "
            "Snowpipe automates data loading. Snowpipe is easy to set up."
        )
        keywords = extractor.extract_keywords(text, platform="snowflake")

        assert "warehouse" in keywords
        assert "stage" in keywords

    def test_extract_keywords_duckdb_platform(self):
        """DuckDB-specific keywords are recognized when platform='duckdb'."""
        extractor = MetadataExtractor()
        text = (
            "Read parquet files directly. Parquet is columnar. "
            "Use extensions like httpfs. The httpfs extension enables S3 access. "
            "Export data to csv format. CSV export is fast."
        )
        keywords = extractor.extract_keywords(text, platform="duckdb")

        assert "parquet" in keywords
        assert "httpfs" in keywords

    def test_extract_keywords_postgresql_platform(self):
        """PostgreSQL-specific keywords are recognized when platform='postgresql'."""
        extractor = MetadataExtractor()
        text = (
            "Use VACUUM to reclaim storage. VACUUM is an important maintenance task. "
            "The JSONB type stores JSON data. JSONB supports indexing. "
            "Create a sequence for auto-increment. The sequence generates unique IDs."
        )
        keywords = extractor.extract_keywords(text, platform="postgresql")

        assert "vacuum" in keywords
        assert "jsonb" in keywords
        assert "sequence" in keywords

    def test_extract_keywords_max_limit(self):
        """Number of extracted keywords does not exceed max_keywords."""
        extractor = MetadataExtractor()
        # Text with many SQL keywords each appearing >= 2 times
        text = " ".join(
            f"{kw} {kw} " for kw in ["select", "insert", "update", "delete", "create", "drop", "alter", "table"]
        )
        keywords = extractor.extract_keywords(text, max_keywords=3)

        assert len(keywords) <= 3

    def test_extract_keywords_minimum_frequency(self):
        """Keywords appearing only once are not included (need >= 2 occurrences)."""
        extractor = MetadataExtractor()
        text = "Use SELECT once. Use INSERT once. Nothing repeats here."
        keywords = extractor.extract_keywords(text)

        # select/insert each appear only once, should not be extracted
        assert "select" not in keywords
        assert "insert" not in keywords

    def test_extract_keywords_no_platform(self):
        """Without platform, only SQL keywords are extracted."""
        extractor = MetadataExtractor()
        text = "The warehouse is ready. The warehouse is large. Use SELECT to query. SELECT from multiple tables."
        keywords = extractor.extract_keywords(text, platform=None)

        assert "select" in keywords
        # 'warehouse' is snowflake-specific, should not appear without platform
        assert "warehouse" not in keywords


# ---------------------------------------------------------------------------
# _detect_version
# ---------------------------------------------------------------------------


class TestDetectVersion:
    """Tests for _detect_version."""

    def test_semantic_version(self):
        """Semantic version like v1.2.3 is detected."""
        extractor = MetadataExtractor()
        result = extractor._detect_version("This document is for version v1.2.3 of the software.")

        assert result == "1.2.3"

    def test_semantic_version_no_v_prefix(self):
        """Semantic version without v prefix like 2.5.0 is detected."""
        extractor = MetadataExtractor()
        result = extractor._detect_version("Compatible with 2.5.0 release.")

        assert result == "2.5.0"

    def test_date_version(self):
        """Date-based version like 2024-01 is detected."""
        extractor = MetadataExtractor()
        # No semantic version present, so date version should be picked
        result = extractor._detect_version("Released in 2024-01 update cycle.")

        assert result is not None
        assert "2024" in result

    def test_major_version_only(self):
        """Major version like 'version 15' is detected when no better match."""
        extractor = MetadataExtractor()
        result = extractor._detect_version("Supports version 15 features only.")

        assert result == "15"

    def test_no_version_found(self):
        """Returns None when no version pattern matches."""
        extractor = MetadataExtractor()
        result = extractor._detect_version("No version information in this text at all.")

        assert result is None

    def test_highest_weight_wins(self):
        """Semantic version (weight 1.0) is preferred over date version (weight 0.8)."""
        extractor = MetadataExtractor()
        result = extractor._detect_version("Version 3.2.1 released in 2024-06.")

        assert result == "3.2.1"

    def test_version_with_prerelease(self):
        """Semantic version with pre-release suffix like 1.0.0-beta is detected."""
        extractor = MetadataExtractor()
        result = extractor._detect_version("Testing version 1.0.0-beta for new features.")

        assert result == "1.0.0-beta"


# ---------------------------------------------------------------------------
# _detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Tests for _detect_language."""

    def test_english_text(self):
        """English text is detected as 'en'."""
        extractor = MetadataExtractor()
        result = extractor._detect_language("This is a regular English text about SQL databases.")

        assert result == "en"

    def test_chinese_text(self):
        """Chinese text with > 10% CJK characters is detected as 'zh'."""
        extractor = MetadataExtractor()
        # Create text that is mostly Chinese characters
        result = extractor._detect_language("这是一段中文文本关于数据库查询和SQL语法的介绍")

        assert result == "zh"

    def test_cyrillic_text(self):
        """Cyrillic text with > 10% Cyrillic characters is detected as 'ru'."""
        extractor = MetadataExtractor()
        result = extractor._detect_language("Это русский текст о базах данных и запросах SQL для аналитики")

        assert result == "ru"

    def test_mixed_text_below_threshold(self):
        """Mixed text with < 10% CJK defaults to 'en'."""
        extractor = MetadataExtractor()
        # Mostly English with one CJK char
        result = extractor._detect_language("This is English text with one character 中 in the middle of many words.")

        assert result == "en"

    def test_empty_text(self):
        """Empty text defaults to 'en'."""
        extractor = MetadataExtractor()
        result = extractor._detect_language("")

        assert result == "en"


# ---------------------------------------------------------------------------
# _find_compound_terms
# ---------------------------------------------------------------------------


class TestFindCompoundTerms:
    """Tests for _find_compound_terms."""

    def test_create_table_detected(self):
        """'create table' compound term is detected."""
        extractor = MetadataExtractor()
        result = extractor._find_compound_terms("use create table to define schema")

        assert any("create" in t and "table" in t for t in result)

    def test_primary_key_detected(self):
        """'primary key' compound term is detected."""
        extractor = MetadataExtractor()
        result = extractor._find_compound_terms("the primary key is required for each table")

        assert any("primary" in t and "key" in t for t in result)

    def test_group_by_detected(self):
        """'group by' compound term is detected."""
        extractor = MetadataExtractor()
        result = extractor._find_compound_terms("use group by to aggregate rows by a column")

        assert any("group" in t and "by" in t for t in result)

    def test_no_compound_terms(self):
        """Returns empty list when no compound terms are present."""
        extractor = MetadataExtractor()
        result = extractor._find_compound_terms("just a simple plain text with no sql patterns")

        assert result == []

    def test_multiple_compound_terms(self):
        """Multiple compound terms in one text are all detected."""
        extractor = MetadataExtractor()
        text = "use create table with a primary key and then order by id"
        result = extractor._find_compound_terms(text)

        assert len(result) >= 3


# ---------------------------------------------------------------------------
# extract() full pipeline
# ---------------------------------------------------------------------------


class TestExtractPipeline:
    """Tests for extract() full pipeline."""

    def test_extract_populates_keywords(self):
        """extract() produces 'keywords' in output metadata."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(
            title="SQL Guide",
            content="Use SELECT to query. SELECT from table. JOIN tables together. JOIN is useful.",
        )
        result = extractor.extract(doc)

        assert "keywords" in result
        assert isinstance(result["keywords"], list)

    def test_extract_populates_language(self):
        """extract() detects language."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(title="English Doc", content="A regular English document about databases.")
        result = extractor.extract(doc)

        assert result["language"] == "en"

    def test_extract_populates_word_count(self):
        """extract() includes word_count."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(title="Title", content="One two three four five.")
        result = extractor.extract(doc)

        assert "word_count" in result
        assert result["word_count"] > 0

    def test_extract_detects_code_blocks(self):
        """extract() detects code block presence."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(title="Code", content="Example:\n```sql\nSELECT 1;\n```\n")
        result = extractor.extract(doc)

        assert result["has_code_blocks"] is True

    def test_extract_detects_tables(self):
        """extract() detects table presence."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(title="Tables", content="| Name | Value |\n|---|\n| a | 1 |")
        result = extractor.extract(doc)

        assert result["has_tables"] is True

    def test_extract_version_from_source_doc(self):
        """extract() uses version from source_doc when available."""
        extractor = MetadataExtractor()
        source = FetchedDocument(
            platform="snowflake",
            version="3.5.0",
            source_url="https://example.com",
            source_type="github",
            doc_path="test.md",
            raw_content="raw",
            content_type="markdown",
        )
        doc = _make_parsed_doc(title="Title", content="Content.", source_doc=source)
        result = extractor.extract(doc)

        assert result["version"] == "3.5.0"

    def test_extract_preserves_existing_metadata(self):
        """extract() preserves metadata already in ParsedDocument."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(
            title="Title",
            content="Content about SELECT twice. SELECT again.",
            metadata={"custom_key": "custom_value"},
        )
        result = extractor.extract(doc)

        assert result["custom_key"] == "custom_value"

    def test_extract_with_platform(self):
        """extract() with platform adds platform-specific keywords."""
        extractor = MetadataExtractor()
        doc = _make_parsed_doc(
            title="Snowflake Guide",
            content=(
                "Configure a warehouse for your queries. The warehouse provides compute. "
                "Load data via a stage. The stage stores files temporarily."
            ),
        )
        result = extractor.extract(doc, platform="snowflake")

        assert "keywords" in result
        # warehouse and stage should be detected with snowflake platform
        assert "warehouse" in result["keywords"] or "stage" in result["keywords"]
