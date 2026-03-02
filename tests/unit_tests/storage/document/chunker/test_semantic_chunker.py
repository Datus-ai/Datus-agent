# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for SemanticChunker and ChunkingConfig."""

import pytest

from datus.storage.document.chunker.semantic_chunker import ChunkingConfig, SemanticChunker
from datus.storage.document.schemas import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_SIZE,
    ParsedDocument,
    ParsedSection,
    PlatformDocChunk,
)

# =============================================================================
# Helpers
# =============================================================================

BASE_METADATA = {
    "platform": "test",
    "version": "1.0",
    "source_type": "github",
    "source_url": "https://example.com/doc.md",
    "doc_path": "docs/test.md",
    "language": "en",
    "content_hash": "abc123",
    "nav_path": [],
    "group_name": "",
}


def _make_section(level: int, title: str, content: str, children=None) -> ParsedSection:
    """Create a ParsedSection with optional children."""
    return ParsedSection(level=level, title=title, content=content, children=children or [])


def _make_doc(title: str, sections: list, metadata=None) -> ParsedDocument:
    """Create a ParsedDocument."""
    return ParsedDocument(title=title, sections=sections, metadata=metadata or {})


# =============================================================================
# ChunkingConfig Tests
# =============================================================================


class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass defaults and custom values."""

    def test_config_defaults_match_schema_constants(self):
        """Default config values should match the module-level constants from schemas."""
        config = ChunkingConfig()
        assert config.chunk_size == DEFAULT_CHUNK_SIZE
        assert config.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert config.min_chunk_size == DEFAULT_MIN_CHUNK_SIZE
        assert config.max_chunk_size == DEFAULT_MAX_CHUNK_SIZE

    def test_config_defaults_boolean_flags(self):
        """Boolean flags default to True for preserve_code_blocks, preserve_paragraphs, add_context_prefix."""
        config = ChunkingConfig()
        assert config.preserve_code_blocks is True
        assert config.preserve_paragraphs is True
        assert config.add_context_prefix is True

    def test_config_defaults_heading_depth_and_merge_buffer(self):
        """max_heading_depth defaults to 3, section_merge_buffer to 1.2."""
        config = ChunkingConfig()
        assert config.max_heading_depth == 3
        assert config.section_merge_buffer == 1.2

    def test_config_custom_values(self):
        """Custom values override all defaults."""
        config = ChunkingConfig(
            chunk_size=512,
            chunk_overlap=64,
            min_chunk_size=100,
            max_chunk_size=1024,
            preserve_code_blocks=False,
            preserve_paragraphs=False,
            add_context_prefix=False,
            max_heading_depth=2,
            section_merge_buffer=1.5,
        )
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 1024
        assert config.preserve_code_blocks is False
        assert config.preserve_paragraphs is False
        assert config.add_context_prefix is False
        assert config.max_heading_depth == 2
        assert config.section_merge_buffer == 1.5


# =============================================================================
# _flatten_section_content Tests
# =============================================================================


class TestFlattenSectionContent:
    """Tests for SemanticChunker._flatten_section_content."""

    def test_flatten_leaf_section_content_only(self):
        """A leaf section with content only returns that content."""
        section = _make_section(2, "Title", "Some paragraph content.")
        result = SemanticChunker._flatten_section_content(section)
        assert result == "Some paragraph content."

    def test_flatten_section_with_no_content(self):
        """A section with empty content and no children returns empty string."""
        section = _make_section(2, "Title", "")
        result = SemanticChunker._flatten_section_content(section)
        assert result == ""

    def test_flatten_section_with_single_child(self):
        """A section with one child includes the child's heading marker and content."""
        child = _make_section(3, "Child Title", "Child content.")
        section = _make_section(2, "Parent Title", "Parent content.", children=[child])
        result = SemanticChunker._flatten_section_content(section)
        assert "Parent content." in result
        assert "### Child Title" in result
        assert "Child content." in result

    def test_flatten_section_with_nested_children(self):
        """Deeply nested children are all flattened with correct heading markers."""
        grandchild = _make_section(4, "Grandchild", "GC content.")
        child = _make_section(3, "Child", "Child content.", children=[grandchild])
        section = _make_section(2, "Parent", "Parent content.", children=[child])
        result = SemanticChunker._flatten_section_content(section)

        # Verify all content and heading markers
        assert "Parent content." in result
        assert "### Child" in result
        assert "Child content." in result
        assert "#### Grandchild" in result
        assert "GC content." in result

    def test_flatten_section_child_without_title(self):
        """A child with no title still has its content included but no heading marker."""
        child = _make_section(3, "", "Untitled child content.")
        section = _make_section(2, "Parent", "Parent content.", children=[child])
        result = SemanticChunker._flatten_section_content(section)
        assert "Untitled child content." in result
        # No heading marker for empty title child
        assert "###" not in result

    def test_flatten_section_multiple_children(self):
        """Multiple children at the same level are all included."""
        child1 = _make_section(3, "First", "First content.")
        child2 = _make_section(3, "Second", "Second content.")
        section = _make_section(2, "Parent", "Parent content.", children=[child1, child2])
        result = SemanticChunker._flatten_section_content(section)
        assert "### First" in result
        assert "First content." in result
        assert "### Second" in result
        assert "Second content." in result

    def test_flatten_section_whitespace_only_children_excluded(self):
        """Children with whitespace-only content are excluded from the joined result."""
        child = _make_section(3, "Empty", "   ")
        section = _make_section(2, "Parent", "Parent content.", children=[child])
        result = SemanticChunker._flatten_section_content(section)
        # The heading marker "### Empty" is included because title is non-empty,
        # but the whitespace-only content itself is filtered out by the join condition
        assert "Parent content." in result


# =============================================================================
# _chunk_section Tests
# =============================================================================


class TestChunkSection:
    """Tests for SemanticChunker._chunk_section."""

    def test_chunk_section_deep_nesting_flattens(self):
        """Sections at max_heading_depth with children get flattened into one text block."""
        config = ChunkingConfig(chunk_size=2000, max_chunk_size=4000, max_heading_depth=3)
        chunker = SemanticChunker(config=config)

        # Section at level 3 (== max_heading_depth) with children
        child = _make_section(4, "Deep Child", "Deep content here.")
        section = _make_section(3, "Level3", "Level3 content.", children=[child])

        chunks = chunker._chunk_section(section, titles=["Doc"], base_metadata=BASE_METADATA, start_index=0)

        assert len(chunks) >= 1
        combined_text = " ".join(c.chunk_text for c in chunks)
        assert "Level3 content." in combined_text
        assert "Deep content here." in combined_text

    def test_chunk_section_small_merged_section(self):
        """Small sections (within merge_buffer * chunk_size) are kept as one chunk."""
        config = ChunkingConfig(chunk_size=500, max_chunk_size=1000, section_merge_buffer=1.2)
        chunker = SemanticChunker(config=config)

        # Total content well under 500 * 1.2 = 600 chars
        child = _make_section(2, "Sub", "Short child content.")
        section = _make_section(1, "Main", "Short parent.", children=[child])

        chunks = chunker._chunk_section(section, titles=[], base_metadata=BASE_METADATA, start_index=0)

        # Should produce a single chunk with all content merged
        assert len(chunks) == 1
        assert "Short parent." in chunks[0].chunk_text
        assert "Short child content." in chunks[0].chunk_text

    def test_chunk_section_recursive_children(self):
        """Large sections with children recurse into children separately."""
        config = ChunkingConfig(chunk_size=100, max_chunk_size=200, section_merge_buffer=1.2)
        chunker = SemanticChunker(config=config)

        # Create content that exceeds merge threshold when combined
        parent_content = "A" * 80
        child_content = "B" * 80
        child = _make_section(2, "Child", child_content)
        section = _make_section(1, "Parent", parent_content, children=[child])

        chunks = chunker._chunk_section(section, titles=[], base_metadata=BASE_METADATA, start_index=0)

        # Should produce multiple chunks (parent content + child content separately)
        assert len(chunks) >= 2

    def test_chunk_section_no_content_no_children(self):
        """A section with no content and no children yields no chunks."""
        config = ChunkingConfig(chunk_size=500)
        chunker = SemanticChunker(config=config)

        section = _make_section(1, "Empty", "")
        chunks = chunker._chunk_section(section, titles=[], base_metadata=BASE_METADATA, start_index=0)
        assert len(chunks) == 0

    def test_chunk_section_level0_skips_merge_check(self):
        """Level 0 sections skip the small-merge check (only level >= 1 merges)."""
        config = ChunkingConfig(chunk_size=5000, section_merge_buffer=1.2)
        chunker = SemanticChunker(config=config)

        child = _make_section(1, "Child", "Child text.")
        section = _make_section(0, "Root", "Root text.", children=[child])

        chunks = chunker._chunk_section(section, titles=[], base_metadata=BASE_METADATA, start_index=0)
        # Should recurse (not merge) because level is 0
        assert len(chunks) >= 1

    def test_chunk_section_builds_title_hierarchy(self):
        """Titles accumulate correctly through the hierarchy."""
        config = ChunkingConfig(chunk_size=5000, add_context_prefix=False)
        chunker = SemanticChunker(config=config)

        section = _make_section(1, "Getting Started", "Intro text here.")
        chunks = chunker._chunk_section(section, titles=["DuckDB Guide"], base_metadata=BASE_METADATA, start_index=0)

        assert len(chunks) == 1
        assert chunks[0].titles == ["DuckDB Guide", "Getting Started"]


# =============================================================================
# _split_content Tests
# =============================================================================


class TestSplitContent:
    """Tests for SemanticChunker._split_content."""

    def test_split_content_empty_returns_empty(self):
        """Empty or whitespace-only content returns no chunks."""
        chunker = SemanticChunker(ChunkingConfig(chunk_size=500))
        result = chunker._split_content("", titles=[], base_metadata=BASE_METADATA, start_index=0)
        assert result == []

    def test_split_content_whitespace_only_returns_empty(self):
        """Whitespace-only content returns no chunks."""
        chunker = SemanticChunker(ChunkingConfig(chunk_size=500))
        result = chunker._split_content("   \n  \n  ", titles=[], base_metadata=BASE_METADATA, start_index=0)
        assert result == []

    def test_split_content_fits_in_one_chunk(self):
        """Content smaller than chunk_size returns a single chunk."""
        config = ChunkingConfig(chunk_size=500, add_context_prefix=False)
        chunker = SemanticChunker(config)
        content = "This is a short paragraph."
        result = chunker._split_content(content, titles=["Test"], base_metadata=BASE_METADATA, start_index=0)

        assert len(result) == 1
        assert result[0].chunk_text == content
        assert result[0].chunk_index == 0

    def test_split_content_multiple_paragraphs(self):
        """Content with multiple paragraphs exceeding chunk_size splits at paragraph boundaries."""
        config = ChunkingConfig(chunk_size=100, max_chunk_size=200, add_context_prefix=False)
        chunker = SemanticChunker(config)

        para1 = "A" * 60
        para2 = "B" * 60
        para3 = "C" * 60
        content = f"{para1}\n\n{para2}\n\n{para3}"

        result = chunker._split_content(content, titles=["Test"], base_metadata=BASE_METADATA, start_index=0)
        assert len(result) >= 2
        # Each chunk should contain at least one paragraph
        all_text = "\n\n".join(c.chunk_text for c in result)
        assert para1 in all_text
        assert para2 in all_text
        assert para3 in all_text

    def test_split_content_oversized_paragraph_splits_further(self):
        """A paragraph exceeding max_chunk_size is split using hierarchical splitting."""
        config = ChunkingConfig(chunk_size=50, max_chunk_size=100, add_context_prefix=False)
        chunker = SemanticChunker(config)

        # Single paragraph of 150 chars with word boundaries
        words = ["word"] * 30  # 30 * 5 = 150 chars with spaces
        content = " ".join(words)

        result = chunker._split_content(content, titles=["Test"], base_metadata=BASE_METADATA, start_index=0)
        assert len(result) >= 2

    def test_split_content_preserves_code_blocks_intact(self):
        """Code blocks are kept as single units when preserve_code_blocks is True."""
        config = ChunkingConfig(chunk_size=100, max_chunk_size=500, preserve_code_blocks=True, add_context_prefix=False)
        chunker = SemanticChunker(config)

        code = "```python\nfor i in range(10):\n    print(i)\n```"
        content = f"Intro text here.\n\n{code}"

        result = chunker._split_content(content, titles=["Test"], base_metadata=BASE_METADATA, start_index=0)
        # The code block should appear intact in one of the chunks
        all_texts = [c.chunk_text for c in result]
        code_found = any("```python" in t and "```" in t[t.index("```python") + 1 :] for t in all_texts)
        assert code_found, "Code block should be preserved intact"

    def test_split_content_code_block_exceeds_max_chunk_size(self):
        """Oversized code block creates its own separate chunk."""
        config = ChunkingConfig(chunk_size=50, max_chunk_size=100, preserve_code_blocks=True, add_context_prefix=False)
        chunker = SemanticChunker(config)

        # Code block larger than max_chunk_size
        long_code = "x = 1\n" * 25  # ~150 chars
        code = f"```python\n{long_code}```"
        content = f"Intro.\n\n{code}"

        result = chunker._split_content(content, titles=["Test"], base_metadata=BASE_METADATA, start_index=0)
        # Should have at least 2 chunks (intro + code block)
        assert len(result) >= 2

    def test_split_content_start_index_propagated(self):
        """Chunk indices start from the given start_index."""
        config = ChunkingConfig(chunk_size=50, max_chunk_size=200, add_context_prefix=False)
        chunker = SemanticChunker(config)

        content = "First paragraph.\n\nSecond paragraph."
        result = chunker._split_content(content, titles=["Test"], base_metadata=BASE_METADATA, start_index=5)

        assert result[0].chunk_index == 5


# =============================================================================
# _split_text_hierarchically Tests
# =============================================================================


class TestSplitTextHierarchically:
    """Tests for SemanticChunker._split_text_hierarchically."""

    def test_split_text_within_max_size_returns_as_is(self):
        """Text fitting in max_size returns a single-element list."""
        chunker = SemanticChunker()
        result = chunker._split_text_hierarchically("short text", max_size=100)
        assert result == ["short text"]

    def test_split_text_at_paragraph_boundaries(self):
        """Text with paragraph breaks splits at paragraph boundaries first."""
        chunker = SemanticChunker()
        para1 = "A" * 30
        para2 = "B" * 30
        text = f"{para1}\n\n{para2}"
        result = chunker._split_text_hierarchically(text, max_size=40)
        assert len(result) == 2
        assert result[0].strip() == para1
        assert result[1].strip() == para2

    def test_split_text_at_sentence_boundaries(self):
        """Text without paragraph breaks falls back to sentence splitting."""
        chunker = SemanticChunker()
        # Two sentences joined without a blank line
        text = "First sentence here. Second sentence here."
        result = chunker._split_text_hierarchically(text, max_size=25)
        assert len(result) >= 2
        combined = "".join(result)
        assert "First sentence here." in combined

    @pytest.mark.parametrize(
        "text,max_size,expected_min_pieces",
        [
            # Comma-separated list
            ("one, two, three, four, five, six, seven, eight", 15, 2),
            # Semicolon-separated
            ("clause one; clause two; clause three; clause four", 15, 2),
        ],
        ids=["comma_split", "semicolon_split"],
    )
    def test_split_text_at_clause_boundaries(self, text, max_size, expected_min_pieces):
        """Text falls back to clause separators when no sentence boundary works."""
        chunker = SemanticChunker()
        result = chunker._split_text_hierarchically(text, max_size=max_size)
        assert len(result) >= expected_min_pieces

    def test_split_text_at_word_boundaries(self):
        """Text without punctuation falls back to whitespace/word splitting."""
        chunker = SemanticChunker()
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        result = chunker._split_text_hierarchically(text, max_size=30)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 30

    def test_split_text_cjk_character_fallback(self):
        """CJK text without natural boundaries falls back to character-based splitting."""
        chunker = SemanticChunker()
        # 50 CJK characters with no punctuation or spaces
        text = "\u4e00" * 50
        result = chunker._split_text_hierarchically(text, max_size=20)
        assert len(result) >= 3
        for piece in result:
            assert len(piece) <= 20

    def test_split_text_respects_max_size_constraint(self):
        """Every resulting piece should be at most max_size characters."""
        chunker = SemanticChunker()
        text = "This is a fairly long sentence. And another one follows. " * 10
        max_size = 50
        result = chunker._split_text_hierarchically(text, max_size=max_size)
        for piece in result:
            assert len(piece) <= max_size, f"Piece exceeds max_size: {len(piece)} > {max_size}"

    def test_split_text_preserves_all_content(self):
        """Splitting should preserve all characters from the original text."""
        chunker = SemanticChunker()
        text = "Hello world. This is a test. Another sentence here."
        result = chunker._split_text_hierarchically(text, max_size=20)
        # All original words should be present
        combined = "".join(result)
        for word in ["Hello", "world", "test", "Another", "sentence"]:
            assert word in combined


# =============================================================================
# _create_chunk Tests
# =============================================================================


class TestCreateChunk:
    """Tests for SemanticChunker._create_chunk."""

    def test_create_chunk_basic_fields(self):
        """Created chunk has correct basic fields from metadata."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        chunk = chunker._create_chunk(
            "Some content", titles=["Doc", "Section"], base_metadata=BASE_METADATA, chunk_index=3
        )

        assert chunk.chunk_text == "Some content"
        assert chunk.chunk_index == 3
        assert chunk.title == "Section"
        assert chunk.titles == ["Doc", "Section"]
        assert chunk.version == "1.0"
        assert chunk.source_type == "github"
        assert chunk.source_url == "https://example.com/doc.md"
        assert chunk.doc_path == "docs/test.md"
        assert chunk.language == "en"
        assert chunk.content_hash == "abc123"

    def test_create_chunk_hierarchy_from_titles_and_nav_path(self):
        """Hierarchy combines nav_path and titles with deduplication."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        metadata = {**BASE_METADATA, "nav_path": ["Guides", "SQL"]}
        chunk = chunker._create_chunk("Content", titles=["SQL", "Functions"], base_metadata=metadata, chunk_index=0)

        # "SQL" appears in both nav_path and titles, should be deduplicated
        assert chunk.hierarchy == "Guides > SQL > Functions"
        assert chunk.nav_path == ["Guides", "SQL"]

    def test_create_chunk_hierarchy_no_duplicates(self):
        """When nav_path last element matches first title, it is not duplicated."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        metadata = {**BASE_METADATA, "nav_path": ["Guides", "Overview"]}
        chunk = chunker._create_chunk("Content", titles=["Overview", "Details"], base_metadata=metadata, chunk_index=0)

        assert chunk.hierarchy == "Guides > Overview > Details"

    def test_create_chunk_context_prefix_added(self):
        """When add_context_prefix=True, hierarchy is prepended to content."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=True))
        metadata = {**BASE_METADATA, "nav_path": ["Guides"]}
        chunk = chunker._create_chunk("Plain content here.", titles=["Topic"], base_metadata=metadata, chunk_index=0)

        assert chunk.chunk_text.startswith("[Guides > Topic]")
        assert "Plain content here." in chunk.chunk_text

    def test_create_chunk_context_prefix_skipped_for_heading_content(self):
        """When content starts with #, context prefix is not added."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=True))
        metadata = {**BASE_METADATA, "nav_path": ["Guides"]}
        chunk = chunker._create_chunk(
            "# Already a heading\nSome content.", titles=["Topic"], base_metadata=metadata, chunk_index=0
        )

        assert not chunk.chunk_text.startswith("[")

    def test_create_chunk_context_prefix_skipped_when_no_hierarchy(self):
        """When there is no hierarchy, no context prefix is added."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=True))
        chunk = chunker._create_chunk("Content.", titles=[], base_metadata=BASE_METADATA, chunk_index=0)

        assert not chunk.chunk_text.startswith("[")

    def test_create_chunk_empty_titles_yields_empty_title(self):
        """When titles list is empty, title field is an empty string."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        chunk = chunker._create_chunk("Content.", titles=[], base_metadata=BASE_METADATA, chunk_index=0)

        assert chunk.title == ""
        assert chunk.titles == []

    def test_create_chunk_generates_deterministic_id(self):
        """Chunk ID is deterministic based on doc_path, index, and version."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        chunk1 = chunker._create_chunk("Content.", titles=[], base_metadata=BASE_METADATA, chunk_index=0)
        chunk2 = chunker._create_chunk("Different content.", titles=[], base_metadata=BASE_METADATA, chunk_index=0)

        expected_id = PlatformDocChunk.generate_chunk_id(doc_path="docs/test.md", chunk_index=0, version="1.0")
        assert chunk1.chunk_id == expected_id
        assert chunk2.chunk_id == expected_id

    def test_create_chunk_group_name_from_metadata(self):
        """group_name is extracted from base_metadata."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        metadata = {**BASE_METADATA, "group_name": "Getting Started"}
        chunk = chunker._create_chunk("Content.", titles=["Intro"], base_metadata=metadata, chunk_index=0)

        assert chunk.group_name == "Getting Started"

    def test_create_chunk_timestamps_set(self):
        """created_at and updated_at are set to non-empty ISO strings."""
        chunker = SemanticChunker(ChunkingConfig(add_context_prefix=False))
        chunk = chunker._create_chunk("Content.", titles=[], base_metadata=BASE_METADATA, chunk_index=0)

        assert chunk.created_at != ""
        assert chunk.updated_at != ""
        # Should look like ISO format
        assert "T" in chunk.created_at


# =============================================================================
# _merge_small_chunks Tests
# =============================================================================


class TestMergeSmallChunks:
    """Tests for SemanticChunker._merge_small_chunks."""

    def _make_chunk(self, text, hierarchy="h", chunk_index=0):
        """Helper to create a PlatformDocChunk for merge testing."""
        return PlatformDocChunk(
            chunk_id=f"id_{chunk_index}",
            chunk_text=text,
            chunk_index=chunk_index,
            title="T",
            titles=["T"],
            nav_path=[],
            group_name="",
            hierarchy=hierarchy,
            version="1.0",
            source_type="github",
            source_url="https://example.com",
            doc_path="test.md",
            keywords=[],
        )

    def test_merge_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        chunker = SemanticChunker()
        result = chunker._merge_small_chunks([])
        assert result == []

    def test_merge_single_chunk_unchanged(self):
        """A single chunk is returned as-is."""
        chunker = SemanticChunker(ChunkingConfig(min_chunk_size=100))
        chunk = self._make_chunk("Short.")
        result = chunker._merge_small_chunks([chunk])
        assert len(result) == 1
        assert result[0].chunk_text == "Short."

    def test_merge_two_small_same_hierarchy_merged(self):
        """Two small chunks with the same hierarchy are merged."""
        config = ChunkingConfig(min_chunk_size=100, chunk_size=500)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("Short A.", hierarchy="same", chunk_index=0)
        c2 = self._make_chunk("Short B.", hierarchy="same", chunk_index=1)

        result = chunker._merge_small_chunks([c1, c2])
        assert len(result) == 1
        assert "Short A." in result[0].chunk_text
        assert "Short B." in result[0].chunk_text

    def test_merge_different_hierarchy_not_merged(self):
        """Chunks with different hierarchy values are not merged."""
        config = ChunkingConfig(min_chunk_size=100, chunk_size=500)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("Short A.", hierarchy="path1", chunk_index=0)
        c2 = self._make_chunk("Short B.", hierarchy="path2", chunk_index=1)

        result = chunker._merge_small_chunks([c1, c2])
        assert len(result) == 2

    def test_merge_large_chunk_not_merged(self):
        """A chunk at or above min_chunk_size is not merged with the next."""
        config = ChunkingConfig(min_chunk_size=10, chunk_size=500)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("A" * 20, hierarchy="same", chunk_index=0)
        c2 = self._make_chunk("B" * 20, hierarchy="same", chunk_index=1)

        result = chunker._merge_small_chunks([c1, c2])
        assert len(result) == 2

    def test_merge_combined_would_exceed_chunk_size(self):
        """If combined text exceeds chunk_size, chunks are not merged even if small."""
        config = ChunkingConfig(min_chunk_size=100, chunk_size=50)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("A" * 30, hierarchy="same", chunk_index=0)
        c2 = self._make_chunk("B" * 30, hierarchy="same", chunk_index=1)

        result = chunker._merge_small_chunks([c1, c2])
        assert len(result) == 2

    def test_merge_preserves_first_chunk_metadata(self):
        """Merged chunk retains the first chunk's ID, index, and title."""
        config = ChunkingConfig(min_chunk_size=100, chunk_size=500)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("Part A.", hierarchy="same", chunk_index=5)
        c2 = self._make_chunk("Part B.", hierarchy="same", chunk_index=6)

        result = chunker._merge_small_chunks([c1, c2])
        assert len(result) == 1
        assert result[0].chunk_id == "id_5"
        assert result[0].chunk_index == 5

    def test_merge_combines_keywords(self):
        """Merged chunk has deduplicated keywords from both chunks."""
        config = ChunkingConfig(min_chunk_size=100, chunk_size=500)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("Part A.", hierarchy="same", chunk_index=0)
        c1.keywords = ["select", "join"]
        c2 = self._make_chunk("Part B.", hierarchy="same", chunk_index=1)
        c2.keywords = ["join", "table"]

        result = chunker._merge_small_chunks([c1, c2])
        assert len(result) == 1
        assert set(result[0].keywords) == {"select", "join", "table"}

    def test_merge_chain_of_three_small_chunks(self):
        """Three consecutive small chunks: first two merge, third may or may not merge."""
        config = ChunkingConfig(min_chunk_size=100, chunk_size=500)
        chunker = SemanticChunker(config)

        c1 = self._make_chunk("A.", hierarchy="same", chunk_index=0)
        c2 = self._make_chunk("B.", hierarchy="same", chunk_index=1)
        c3 = self._make_chunk("C.", hierarchy="same", chunk_index=2)

        result = chunker._merge_small_chunks([c1, c2, c3])
        # After merging c1+c2, the merged chunk is still < min_chunk_size
        # so it can merge with c3 too
        assert len(result) <= 2
        combined = "\n\n".join(c.chunk_text for c in result)
        assert "A." in combined
        assert "B." in combined
        assert "C." in combined


# =============================================================================
# chunk() End-to-End Tests
# =============================================================================


class TestChunkEndToEnd:
    """Tests for the full SemanticChunker.chunk() method."""

    def test_chunk_single_section_small_doc(self):
        """A small single-section document produces one chunk."""
        config = ChunkingConfig(chunk_size=2000, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="Test Doc",
            sections=[_make_section(1, "Intro", "This is a brief introduction.")],
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "docs/intro.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) == 1
        assert "brief introduction" in chunks[0].chunk_text
        assert chunks[0].chunk_index == 0
        assert chunks[0].title == "Intro"
        assert chunks[0].titles == ["Test Doc", "Intro"]

    def test_chunk_multiple_sections(self):
        """A document with multiple sections produces multiple chunks."""
        config = ChunkingConfig(chunk_size=100, max_chunk_size=200, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="Guide",
            sections=[
                _make_section(1, "Section A", "A" * 80),
                _make_section(1, "Section B", "B" * 80),
                _make_section(1, "Section C", "C" * 80),
            ],
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "docs/guide.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 3
        # Verify sequential indexing after re-index
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_with_nav_path_in_metadata(self):
        """Document metadata nav_path flows through to chunk hierarchy."""
        config = ChunkingConfig(chunk_size=2000, add_context_prefix=True)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="Loading Data",
            sections=[_make_section(1, "Overview", "How to load data into the database.")],
            metadata={"nav_path": ["Guides", "Data Management"]},
        )
        metadata = {
            "platform": "snowflake",
            "version": "2024",
            "source_type": "website",
            "source_url": "https://docs.snowflake.com/loading",
            "doc_path": "loading.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 1
        # Hierarchy should combine nav_path + titles
        assert "Guides" in chunks[0].hierarchy
        assert "Data Management" in chunks[0].hierarchy

    def test_chunk_group_name_from_metadata(self):
        """group_name is extracted from doc metadata."""
        config = ChunkingConfig(chunk_size=2000, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="SQL Reference",
            sections=[_make_section(1, "SELECT", "The SELECT statement.")],
            metadata={"group_name": "Reference", "nav_path": ["Reference", "SQL"]},
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "sql.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 1
        assert chunks[0].group_name == "Reference"

    def test_chunk_group_name_fallback_to_nav_path(self):
        """When group_name is not in metadata, it falls back to first nav_path element."""
        config = ChunkingConfig(chunk_size=2000, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="Functions",
            sections=[_make_section(1, "Aggregate", "Aggregate functions.")],
            metadata={"nav_path": ["SQL Reference", "Functions"]},
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "funcs.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 1
        assert chunks[0].group_name == "SQL Reference"

    def test_chunk_reindex_after_merge(self):
        """After merging, chunk indices are sequential starting from 0."""
        config = ChunkingConfig(chunk_size=500, min_chunk_size=200, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="Doc",
            sections=[
                _make_section(1, "A", "Content A."),
                _make_section(1, "B", "Content B."),
                _make_section(1, "C", "Content C."),
            ],
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "test.md",
        }

        chunks = chunker.chunk(doc, metadata)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            expected_id = PlatformDocChunk.generate_chunk_id(
                doc_path="test.md",
                chunk_index=i,
                version="1.0",
            )
            assert chunk.chunk_id == expected_id

    def test_chunk_empty_document(self):
        """A document with no sections returns no chunks."""
        config = ChunkingConfig(chunk_size=500)
        chunker = SemanticChunker(config)

        doc = _make_doc(title="Empty", sections=[])
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "empty.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert chunks == []

    def test_chunk_doc_title_included_in_titles(self):
        """The document title is included as the first element in chunk titles."""
        config = ChunkingConfig(chunk_size=2000, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="My Document",
            sections=[_make_section(1, "Section One", "Content here.")],
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "doc.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 1
        assert chunks[0].titles[0] == "My Document"

    def test_chunk_doc_without_title(self):
        """When document title is empty, titles start from the section title."""
        config = ChunkingConfig(chunk_size=2000, add_context_prefix=False)
        chunker = SemanticChunker(config)

        doc = _make_doc(
            title="",
            sections=[_make_section(1, "Only Section", "Some content.")],
        )
        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "notitle.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 1
        assert chunks[0].titles == ["Only Section"]


# =============================================================================
# _split_large_paragraph Tests
# =============================================================================


class TestSplitLargeParagraph:
    """Tests for SemanticChunker._split_large_paragraph."""

    def test_split_large_paragraph_with_sentences(self):
        """Large paragraph with sentence boundaries splits at sentences."""
        config = ChunkingConfig(chunk_size=50, add_context_prefix=False)
        chunker = SemanticChunker(config)

        paragraph = "This is sentence one. This is sentence two. This is sentence three. And this is four."
        chunks = chunker._split_large_paragraph(paragraph, titles=["T"], base_metadata=BASE_METADATA, start_index=0)
        assert len(chunks) >= 2

    def test_split_large_paragraph_preserves_content(self):
        """All content from the original paragraph is present in the chunks."""
        config = ChunkingConfig(chunk_size=30, add_context_prefix=False)
        chunker = SemanticChunker(config)

        paragraph = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
        chunks = chunker._split_large_paragraph(paragraph, titles=["T"], base_metadata=BASE_METADATA, start_index=0)
        combined = " ".join(c.chunk_text for c in chunks)
        for word in ["Alpha", "bravo", "charlie", "delta", "echo"]:
            assert word in combined

    def test_split_large_paragraph_indices_sequential(self):
        """Chunk indices are sequential starting from start_index."""
        config = ChunkingConfig(chunk_size=30, add_context_prefix=False)
        chunker = SemanticChunker(config)

        paragraph = "Word " * 30
        chunks = chunker._split_large_paragraph(paragraph, titles=["T"], base_metadata=BASE_METADATA, start_index=10)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == 10 + i


# =============================================================================
# Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Edge cases and special scenarios."""

    def test_default_config_used_when_none(self):
        """When no config is provided, default ChunkingConfig is used."""
        chunker = SemanticChunker(config=None)
        assert chunker.config.chunk_size == DEFAULT_CHUNK_SIZE
        assert chunker.config.max_chunk_size == DEFAULT_MAX_CHUNK_SIZE

    def test_code_block_pattern_matches_multiline(self):
        """The code block regex correctly matches multiline code blocks."""
        text = "before\n```python\nline1\nline2\n```\nafter"
        matches = SemanticChunker.CODE_BLOCK_PATTERN.findall(text)
        assert len(matches) == 1
        assert "line1" in matches[0]
        assert "line2" in matches[0]

    def test_paragraph_pattern_matches_blank_lines(self):
        """The paragraph regex matches various blank line patterns."""
        text = "para1\n\npara2\n  \npara3"
        parts = SemanticChunker.PARAGRAPH_PATTERN.split(text)
        assert len(parts) == 3

    def test_split_content_no_preserve_code_blocks(self):
        """When preserve_code_blocks is False, code blocks are treated as regular content."""
        config = ChunkingConfig(chunk_size=50, max_chunk_size=200, preserve_code_blocks=False, add_context_prefix=False)
        chunker = SemanticChunker(config)

        code = "```python\nprint('hello')\n```"
        content = f"Intro text.\n\n{code}"
        result = chunker._split_content(content, titles=["T"], base_metadata=BASE_METADATA, start_index=0)
        assert len(result) >= 1

    def test_chunk_with_deeply_nested_sections(self):
        """A document with h1 > h2 > h3 > h4 nesting is handled correctly."""
        config = ChunkingConfig(chunk_size=2000, max_heading_depth=3, add_context_prefix=False)
        chunker = SemanticChunker(config)

        h4 = _make_section(4, "H4 Title", "H4 content.")
        h3 = _make_section(3, "H3 Title", "H3 content.", children=[h4])
        h2 = _make_section(2, "H2 Title", "H2 content.", children=[h3])
        doc = _make_doc(title="Deep Doc", sections=[h2])

        metadata = {
            "platform": "test",
            "version": "1.0",
            "source_type": "github",
            "source_url": "https://example.com",
            "doc_path": "deep.md",
        }

        chunks = chunker.chunk(doc, metadata)
        assert len(chunks) >= 1
        combined = " ".join(c.chunk_text for c in chunks)
        # All content should be present somewhere
        assert "H2 content." in combined
        assert "H3 content." in combined
        assert "H4 content." in combined
