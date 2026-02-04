# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Document Chunker

Splits documents into chunks while preserving semantic coherence.
Respects document structure (headings, code blocks) and maintains context.

Design inspired by:
- semchunk: Hierarchical splitting with punctuation-based boundaries
- Chonkie: Structure preservation and multi-language support
- LangChain: Configurable thresholds and overlap handling
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from datus.storage.document.parser.metadata_extractor import MetadataExtractor
from datus.storage.document.schemas import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_SIZE,
    ParsedDocument,
    ParsedSection,
    PlatformDocChunk,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# =============================================================================
# Hierarchical Splitters (inspired by semchunk)
# =============================================================================

# Splitting hierarchy - ordered by preference (most desirable first)
# Reference: semchunk's _NON_WHITESPACE_SEMANTIC_SPLITTERS
SPLIT_HIERARCHY: List[Tuple[str, str]] = [
    # Level 1: Paragraph boundaries (most preferred)
    (r"\n\s*\n", "paragraph"),
    # Level 2: Sentence terminators
    (r"(?<=[.!?。！？])\s+", "sentence_en"),
    (r"(?<=[.!?。！？])(?=[^\s])", "sentence_cjk"),  # CJK: no space after punctuation
    # Level 3: Clause separators
    (r"(?<=[;；])\s*", "semicolon"),
    (r"(?<=[,，、])\s*", "comma"),
    (r"(?<=[:])\s+", "colon"),
    # Level 4: Other boundaries
    (r"(?<=[)）\]】」』])\s*", "bracket_close"),
    (r"\s+(?=[(（\[【「『])", "bracket_open"),
    # Level 5: Word boundaries (least preferred for text)
    (r"\s+", "whitespace"),
]


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Target chunk size in characters (soft limit).
            Chunks may exceed this if needed to preserve paragraph integrity.
        chunk_overlap: Number of characters to overlap between chunks.
            Helps maintain context across chunk boundaries.
        min_chunk_size: Minimum chunk size. Smaller chunks are merged with neighbors.
        max_chunk_size: Maximum chunk size (hard limit).
            Paragraphs exceeding this will be split at sentence/clause boundaries.
        preserve_code_blocks: Keep code blocks intact even if oversized.
        preserve_paragraphs: Prioritize keeping paragraphs whole.
            When True, allows chunks between chunk_size and max_chunk_size
            to keep paragraphs intact.
        add_context_prefix: Add hierarchy context (e.g., "[Section > Subsection]")
            to the beginning of each chunk for better retrieval.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
    preserve_code_blocks: bool = True
    preserve_paragraphs: bool = True  # New: prioritize paragraph integrity
    add_context_prefix: bool = True
    max_heading_depth: int = 3  # Max heading level for splitting (h4+ content is flattened into parent)
    section_merge_buffer: float = 1.2  # Buffer ratio for min section merge (1.2 = 120% of chunk_size)


class SemanticChunker:
    """Semantic-aware document chunker with hierarchical splitting.

    Splits documents into chunks while preserving semantic coherence using
    a multi-level splitting strategy inspired by semchunk and Chonkie:

    Splitting Hierarchy (most to least preferred):
    1. Paragraph boundaries (\\n\\n)
    2. Sentence terminators (.!?。！？)
    3. Clause separators (;,;，、:)
    4. Bracket boundaries
    5. Whitespace (word boundaries)
    6. Character boundaries (CJK fallback)

    Features:
    - Respects heading hierarchy from parsed documents
    - Keeps code blocks intact (configurable)
    - Prioritizes paragraph integrity (configurable)
    - Multi-language support (English, Chinese, Japanese, Korean)
    - Maintains context through hierarchy prefixes
    - Merges small chunks to avoid fragmentation

    Example:
        >>> config = ChunkingConfig(chunk_size=512, preserve_paragraphs=True)
        >>> chunker = SemanticChunker(config=config)
        >>> chunks = chunker.chunk(parsed_doc, metadata)
        >>> for chunk in chunks:
        ...     print(f"{chunk.hierarchy}: {len(chunk.chunk_text)} chars")
    """

    # Regex patterns
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    # Compiled split patterns (module-level, initialized once)
    _split_patterns: List[Tuple[re.Pattern, str]] = [(re.compile(pattern), name) for pattern, name in SPLIT_HIERARCHY]

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or ChunkingConfig()
        self._metadata_extractor = MetadataExtractor()

    def chunk(
        self,
        doc: ParsedDocument,
        base_metadata: Dict[str, Any],
    ) -> List[PlatformDocChunk]:
        """Chunk a parsed document.

        Args:
            doc: Parsed document with sections
            base_metadata: Base metadata (platform, version, source_url, etc.)

        Returns:
            List of document chunks
        """
        chunks = []
        chunk_index = 0

        # Extract nav_path from parsed document metadata (site navigation)
        nav_path = doc.metadata.get("nav_path", []) if doc.metadata else []

        # Extract group_name (prefer from metadata, fallback to first nav_path element)
        group_name = ""
        if doc.metadata:
            group_name = doc.metadata.get("group_name", "")
        if not group_name and nav_path:
            group_name = nav_path[0]

        # Store nav_path and group_name in base_metadata for chunk creation
        base_metadata = base_metadata.copy()
        base_metadata["nav_path"] = nav_path
        base_metadata["group_name"] = group_name

        logger.debug(f"nav_path: {nav_path}, group_name: {group_name}")

        # Process each top-level section with page-internal titles starting fresh
        # titles will only contain page-internal headings (h1, h2, h3...)
        initial_titles = [doc.title] if doc.title else []

        for section in doc.sections:
            section_chunks = self._chunk_section(
                section=section,
                titles=initial_titles.copy(),
                base_metadata=base_metadata,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        # Re-index after merging
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            # Regenerate chunk_id with new index
            chunk.chunk_id = PlatformDocChunk.generate_chunk_id(
                doc_path=chunk.doc_path,
                chunk_index=i,
                version=chunk.version,
            )

        logger.debug(f"Created {len(chunks)} chunks from '{doc.title}'")
        return chunks

    @staticmethod
    def _flatten_section_content(section: ParsedSection) -> str:
        """Flatten a section and all its children into a single text block.

        Preserves heading markers for children so the structure is readable
        within a single chunk.

        Args:
            section: Section to flatten

        Returns:
            Combined text content
        """
        parts: List[str] = []
        if section.content:
            parts.append(section.content)
        for child in section.children:
            if child.title:
                prefix = "#" * child.level
                parts.append(f"{prefix} {child.title}")
            parts.append(SemanticChunker._flatten_section_content(child))
        return "\n\n".join(p for p in parts if p.strip())

    def _chunk_section(
        self,
        section: ParsedSection,
        titles: List[str],
        base_metadata: Dict[str, Any],
        start_index: int,
    ) -> List[PlatformDocChunk]:
        """Recursively chunk a section.

        Splitting stops at ``max_heading_depth`` (default 3).  Sections at
        deeper levels have all their children flattened into a single text
        block so that h4/h5/h6 headings never cause additional splits.

        For h1/h2 sections whose total content (including children) is within
        ``chunk_size * section_merge_buffer``, the entire section is kept as
        one chunk instead of being split by child headings.

        Args:
            section: Section to chunk
            titles: Page-internal heading hierarchy (h1, h2, h3...)
            base_metadata: Base metadata for chunks (includes nav_path, group_name)
            start_index: Starting chunk index

        Returns:
            List of chunks from this section
        """
        # Build current title hierarchy (page-internal only)
        current_titles = titles.copy()
        if section.title:
            current_titles.append(section.title)

        # --- Rule 1: flatten content for sections at or beyond max heading depth ---
        if section.level >= self.config.max_heading_depth and section.children:
            flat_text = self._flatten_section_content(section)
            if flat_text.strip():
                return self._split_content(
                    content=flat_text,
                    titles=current_titles,
                    base_metadata=base_metadata,
                    start_index=start_index,
                )
            return []

        # --- Rule 2: if total section size is small enough, keep as one chunk ---
        merge_threshold = int(self.config.chunk_size * self.config.section_merge_buffer)
        if section.children and section.level >= 1:
            total_text = self._flatten_section_content(section)
            if len(total_text) <= merge_threshold:
                if total_text.strip():
                    return self._split_content(
                        content=total_text,
                        titles=current_titles,
                        base_metadata=base_metadata,
                        start_index=start_index,
                    )
                return []

        # --- Default: process content then recurse into children ---
        chunks = []
        current_index = start_index

        # Process section content
        if section.content:
            content_chunks = self._split_content(
                content=section.content,
                titles=current_titles,
                base_metadata=base_metadata,
                start_index=current_index,
            )
            chunks.extend(content_chunks)
            current_index += len(content_chunks)

        # Process children recursively
        for child in section.children:
            child_chunks = self._chunk_section(
                section=child,
                titles=current_titles,
                base_metadata=base_metadata,
                start_index=current_index,
            )
            chunks.extend(child_chunks)
            current_index += len(child_chunks)

        return chunks

    def _split_content(
        self,
        content: str,
        titles: List[str],
        base_metadata: Dict[str, Any],
        start_index: int,
    ) -> List[PlatformDocChunk]:
        """Split content into chunks.

        Args:
            content: Text content to split
            titles: Title hierarchy
            base_metadata: Base metadata
            start_index: Starting chunk index

        Returns:
            List of chunks
        """
        content = content.strip()
        if not content:
            return []

        # If content fits in one chunk, return it
        if len(content) <= self.config.chunk_size:
            return [self._create_chunk(content, titles, base_metadata, start_index)]

        # Extract code blocks for special handling
        code_blocks = self.CODE_BLOCK_PATTERN.findall(content)
        text_parts = self.CODE_BLOCK_PATTERN.split(content)

        chunks = []
        current_chunk_parts = []
        current_size = 0
        chunk_index = start_index

        def flush_chunk():
            """Create chunk from accumulated parts."""
            nonlocal current_chunk_parts, current_size, chunk_index

            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts)
                if chunk_text.strip():
                    chunk = self._create_chunk(chunk_text, titles, base_metadata, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                current_chunk_parts = []
                current_size = 0

        # Interleave text and code blocks
        code_iter = iter(code_blocks)
        for _i, text_part in enumerate(text_parts):
            # Process text part
            if text_part.strip():
                paragraphs = self.PARAGRAPH_PATTERN.split(text_part)

                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue

                    para_size = len(para)

                    # If paragraph alone exceeds max size, split it
                    if para_size > self.config.max_chunk_size:
                        flush_chunk()
                        sub_chunks = self._split_large_paragraph(para, titles, base_metadata, chunk_index)
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                        continue

                    # Paragraph integrity priority:
                    # - If adding paragraph exceeds max_chunk_size: must flush
                    # - If adding paragraph exceeds chunk_size but within max_chunk_size:
                    #   allow if current chunk is empty (keep paragraph intact)
                    #   or flush if current chunk already has content
                    combined_size = current_size + para_size

                    if combined_size > self.config.max_chunk_size:
                        # Exceeds max limit, must flush first
                        flush_chunk()
                    elif combined_size > self.config.chunk_size and current_chunk_parts:
                        # Exceeds soft limit and we have existing content
                        # Flush current content, then add paragraph to new chunk
                        flush_chunk()
                    # else: either within chunk_size, or empty chunk can accept larger paragraph

                    current_chunk_parts.append(para)
                    current_size += para_size

            # Add corresponding code block
            try:
                code_block = next(code_iter)
                code_size = len(code_block)

                if self.config.preserve_code_blocks:
                    # Always keep code blocks intact
                    combined_size = current_size + code_size

                    if code_size > self.config.max_chunk_size:
                        # Code block exceeds max size - create separate chunk
                        flush_chunk()
                        chunk = self._create_chunk(code_block, titles, base_metadata, chunk_index)
                        chunks.append(chunk)
                        chunk_index += 1
                    elif combined_size > self.config.max_chunk_size:
                        # Adding code block would exceed max, flush first
                        flush_chunk()
                        current_chunk_parts.append(code_block)
                        current_size = code_size
                    elif combined_size > self.config.chunk_size and current_chunk_parts:
                        # Exceeds soft limit with existing content, flush first
                        flush_chunk()
                        current_chunk_parts.append(code_block)
                        current_size = code_size
                    else:
                        # Within limits, add to current chunk
                        current_chunk_parts.append(code_block)
                        current_size += code_size
                else:
                    # Treat code blocks like regular content
                    if current_size + code_size > self.config.chunk_size:
                        flush_chunk()
                    current_chunk_parts.append(code_block)
                    current_size += code_size

            except StopIteration:
                pass

        # Flush remaining content
        flush_chunk()

        return chunks

    def _split_text_hierarchically(
        self,
        text: str,
        max_size: int,
    ) -> List[str]:
        """Split text using hierarchical boundaries (semchunk-inspired).

        Tries splitting at increasingly less desirable boundaries:
        1. Sentences (. ! ? 。 ！ ？)
        2. Clauses (; , ; ， 、 :)
        3. Brackets
        4. Whitespace (words)
        5. Characters (CJK fallback)

        Args:
            text: Text to split
            max_size: Maximum size for each piece

        Returns:
            List of text pieces, each <= max_size
        """
        if len(text) <= max_size:
            return [text]

        # Try each split pattern in order of preference
        for pattern, _pattern_name in self._split_patterns:
            pieces = pattern.split(text)
            pieces = [p for p in pieces if p.strip()]

            if len(pieces) <= 1:
                continue

            # Check if this split helps
            if all(len(p) <= max_size for p in pieces):
                return pieces

            # Merge small pieces and recursively split large ones
            result = []
            current = ""

            for piece in pieces:
                if len(piece) > max_size:
                    # Flush current accumulator
                    if current:
                        result.append(current)
                        current = ""
                    # Recursively split large piece with next pattern level
                    result.extend(self._split_text_hierarchically(piece, max_size))
                elif len(current) + len(piece) <= max_size:
                    # Accumulate
                    current = current + piece if current else piece
                else:
                    # Flush and start new
                    if current:
                        result.append(current)
                    current = piece

            if current:
                result.append(current)

            if result:
                return result

        # Final fallback: character-based split for CJK or very long words
        result = []
        for i in range(0, len(text), max_size):
            result.append(text[i : i + max_size])
        return result

    def _split_large_paragraph(
        self,
        paragraph: str,
        titles: List[str],
        base_metadata: Dict[str, Any],
        start_index: int,
    ) -> List[PlatformDocChunk]:
        """Split a large paragraph that exceeds max chunk size.

        Uses hierarchical splitting strategy for better semantic preservation.
        Supports multiple languages including Chinese, Japanese, Korean.

        Args:
            paragraph: Large paragraph to split
            titles: Title hierarchy
            base_metadata: Base metadata
            start_index: Starting chunk index

        Returns:
            List of chunks
        """
        # Use hierarchical splitting to break down the paragraph
        pieces = self._split_text_hierarchically(paragraph, self.config.chunk_size)

        chunks = []
        chunk_index = start_index

        # Group pieces to maximize chunk utilization
        current_parts = []
        current_size = 0

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            piece_size = len(piece)

            # If adding this piece would exceed chunk_size, flush current
            if current_size + piece_size > self.config.chunk_size and current_parts:
                chunk_text = " ".join(current_parts)
                chunk = self._create_chunk(chunk_text, titles, base_metadata, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                current_parts = []
                current_size = 0

            current_parts.append(piece)
            current_size += piece_size

        # Handle remaining content
        if current_parts:
            chunk_text = " ".join(current_parts)
            chunk = self._create_chunk(chunk_text, titles, base_metadata, chunk_index)
            chunks.append(chunk)

        return chunks

    def _split_long_sentence(
        self,
        sentence: str,
        titles: List[str],
        base_metadata: Dict[str, Any],
        start_index: int,
    ) -> List[PlatformDocChunk]:
        """Split a very long sentence that exceeds chunk size.

        Uses hierarchical splitting for better semantic preservation.
        Automatically handles both word-based and character-based languages.

        Args:
            sentence: Long sentence to split
            titles: Title hierarchy
            base_metadata: Base metadata
            start_index: Starting chunk index

        Returns:
            List of chunks
        """
        # Use hierarchical splitting - it handles both CJK and word-based languages
        pieces = self._split_text_hierarchically(sentence, self.config.chunk_size)

        chunks = []
        chunk_index = start_index

        for piece in pieces:
            piece = piece.strip()
            if piece:
                chunk = self._create_chunk(piece, titles, base_metadata, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _create_chunk(
        self,
        content: str,
        titles: List[str],
        base_metadata: Dict[str, Any],
        chunk_index: int,
    ) -> PlatformDocChunk:
        """Create a chunk object.

        Args:
            content: Chunk text content
            titles: Page-internal heading hierarchy (h1, h2, h3...)
            base_metadata: Base metadata (includes nav_path, group_name)
            chunk_index: Chunk index

        Returns:
            PlatformDocChunk object
        """
        # Extract nav_path and group_name from metadata
        nav_path = base_metadata.get("nav_path", [])
        group_name = base_metadata.get("group_name", "")

        # Build full hierarchy: nav_path + page-internal titles (deduplicated)
        # Avoid duplicating if the last nav_path item equals the first title
        full_path = list(nav_path)
        for t in titles:
            if not full_path or t != full_path[-1]:
                full_path.append(t)

        hierarchy = " > ".join(full_path) if full_path else ""

        # Add context prefix if configured
        if self.config.add_context_prefix and hierarchy:
            # Don't add prefix if content already starts with a heading
            if not content.strip().startswith("#"):
                content = f"[{hierarchy}]\n\n{content}"

        # Extract keywords
        keywords = self._metadata_extractor.extract_keywords(content, base_metadata.get("platform", ""), max_keywords=5)

        # Generate chunk ID
        chunk_id = PlatformDocChunk.generate_chunk_id(
            doc_path=base_metadata.get("doc_path", ""),
            chunk_index=chunk_index,
            version=base_metadata.get("version", ""),
        )

        now = datetime.now(timezone.utc).isoformat()

        return PlatformDocChunk(
            chunk_id=chunk_id,
            chunk_text=content.strip(),
            chunk_index=chunk_index,
            title=titles[-1] if titles else "",
            titles=titles,  # Page-internal headings only
            nav_path=nav_path,  # Site navigation path
            group_name=group_name,  # Top-level group
            hierarchy=hierarchy,  # Full combined path
            version=base_metadata.get("version", ""),
            source_type=base_metadata.get("source_type", ""),
            source_url=base_metadata.get("source_url", ""),
            doc_path=base_metadata.get("doc_path", ""),
            keywords=keywords,
            language=base_metadata.get("language", "en"),
            content_hash=base_metadata.get("content_hash", ""),
            created_at=now,
            updated_at=now,
        )

    def _merge_small_chunks(
        self,
        chunks: List[PlatformDocChunk],
    ) -> List[PlatformDocChunk]:
        """Merge small consecutive chunks.

        Args:
            chunks: List of chunks

        Returns:
            List with small chunks merged
        """
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if chunks can be merged
            can_merge = (
                len(current.chunk_text) < self.config.min_chunk_size
                and current.hierarchy == next_chunk.hierarchy
                and len(current.chunk_text) + len(next_chunk.chunk_text) <= self.config.chunk_size
            )

            if can_merge:
                # Merge chunks
                current = PlatformDocChunk(
                    chunk_id=current.chunk_id,
                    chunk_text=current.chunk_text + "\n\n" + next_chunk.chunk_text,
                    chunk_index=current.chunk_index,
                    title=current.title,
                    titles=current.titles,
                    nav_path=current.nav_path,
                    group_name=current.group_name,
                    hierarchy=current.hierarchy,
                    version=current.version,
                    source_type=current.source_type,
                    source_url=current.source_url,
                    doc_path=current.doc_path,
                    keywords=list(set(current.keywords + next_chunk.keywords)),
                    language=current.language,
                    content_hash=current.content_hash,
                    created_at=current.created_at,
                    updated_at=current.updated_at,
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged
