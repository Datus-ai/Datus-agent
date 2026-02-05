# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Storage Module

Provides comprehensive document storage and processing with full-featured schema:
- Version tracking (each platform has its own store)
- Navigation path (titles, nav_path, group_name, hierarchy)
- Keywords extraction
- Deduplication via chunk_id

Storage:
- DocumentStore: Full-featured document storage

Data Models:
- PlatformDocChunk, FetchedDocument, ParsedDocument, ParsedSection

Fetchers:
- LocalFetcher: Local file system
- GitHubFetcher: GitHub repositories
- WebFetcher: Official websites

Initialization:
- init_platform_docs: Full pipeline for platform documentation
- import_documents: Import local documents

Note: Search functionality is provided by datus.tools.search_tools.SearchTool

Usage:
    from datus.storage.document import (
        DocumentStore,
        init_platform_docs,
        import_documents,
        SOURCE_TYPE_LOCAL,
    )

    # Initialize from local directory
    from datus.configuration.agent_config import DocumentConfig

    cfg = DocumentConfig(type="local", source="/path/to/docs")
    result = init_platform_docs(
        db_path="/path/to/db",
        platform="custom",
        cfg=cfg,
    )

    # Access store for custom operations
    store = DocumentStore(db_path, embedding_model)
"""

# Chunker
from datus.storage.document.chunker import SemanticChunker

# Cleaner
from datus.storage.document.cleaner import DocumentCleaner

# Initialization functions
from datus.storage.document.doc_init import InitResult, import_documents, init_platform_docs

# Fetchers
from datus.storage.document.fetcher import BaseFetcher, GitHubFetcher, LocalFetcher, RateLimiter, WebFetcher

# Parsers
from datus.storage.document.parser import HTMLParser, MarkdownParser, MetadataExtractor

# Data models
from datus.storage.document.schemas import (  # Constants
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_RST,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_SIZE,
    SOURCE_TYPE_GITHUB,
    SOURCE_TYPE_LOCAL,
    SOURCE_TYPE_WEBSITE,
    FetchedDocument,
    ParsedDocument,
    ParsedSection,
    PlatformDocChunk,
)

# Store classes
from datus.storage.document.store import DocumentStore, document_store, get_platform_doc_schema

# Streaming processor
from datus.storage.document.streaming_processor import ProcessingStats, StreamingDocProcessor

__all__ = [
    # Store classes
    "DocumentStore",
    "document_store",
    # Data models
    "PlatformDocChunk",
    "FetchedDocument",
    "ParsedDocument",
    "ParsedSection",
    "get_platform_doc_schema",
    # Constants
    "SOURCE_TYPE_GITHUB",
    "SOURCE_TYPE_WEBSITE",
    "SOURCE_TYPE_LOCAL",
    "CONTENT_TYPE_MARKDOWN",
    "CONTENT_TYPE_HTML",
    "CONTENT_TYPE_RST",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MIN_CHUNK_SIZE",
    "DEFAULT_MAX_CHUNK_SIZE",
    # Fetchers
    "BaseFetcher",
    "LocalFetcher",
    "GitHubFetcher",
    "WebFetcher",
    "RateLimiter",
    # Parsers
    "MarkdownParser",
    "HTMLParser",
    "MetadataExtractor",
    # Chunker
    "SemanticChunker",
    # Cleaner
    "DocumentCleaner",
    # Init functions
    "init_platform_docs",
    "import_documents",
    "InitResult",
    # Streaming processor
    "StreamingDocProcessor",
    "ProcessingStats",
]
