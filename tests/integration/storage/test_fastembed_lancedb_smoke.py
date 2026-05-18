# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Nightly smoke for the real FastEmbed + LanceDB document-store path."""

from __future__ import annotations

import pytest
from datus_storage_base.backend_config import StorageBackendConfig

from datus.storage.backend_holder import create_vector_connection, init_backends, reset_backends
from datus.storage.document.schemas import PlatformDocChunk
from datus.storage.document.store import DocumentStore
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.constants import EmbeddingProvider

pytestmark = [pytest.mark.nightly, pytest.mark.timeout(90)]


def _chunk(doc_path: str, chunk_index: int, text: str) -> PlatformDocChunk:
    return PlatformDocChunk(
        chunk_id=PlatformDocChunk.generate_chunk_id(doc_path, chunk_index, "nightly"),
        chunk_text=text,
        chunk_index=chunk_index,
        title="FastEmbed LanceDB Smoke",
        titles=["FastEmbed LanceDB Smoke"],
        nav_path=["Storage", "Smoke"],
        group_name="Storage",
        hierarchy="Storage > Smoke > FastEmbed LanceDB Smoke",
        version="nightly",
        source_type="local",
        source_url="",
        doc_path=doc_path,
        keywords=["fastembed", "lancedb"],
        language="en",
    )


def test_real_fastembed_lancedb_document_store_round_trip(tmp_path):
    """The real FastEmbed adapter should write and query LanceDB without long retry stalls."""
    store = None
    try:
        init_backends(StorageBackendConfig(), data_dir=str(tmp_path / "data"))
        model = EmbeddingModel(
            model_name="all-MiniLM-L6-v2",
            dim_size=384,
            registry_name=EmbeddingProvider.FASTEMBED,
            batch_size=2,
        )
        store = DocumentStore(embedding_model=model, db=create_vector_connection("fastembed_lancedb_smoke"))
        stored = store.store_chunks(
            [
                _chunk("docs/storage/fastembed.md", 0, "FastEmbed writes document vectors into LanceDB."),
                _chunk("docs/storage/lancedb.md", 1, "LanceDB searches document chunks using vector embeddings."),
            ]
        )

        assert stored == 2
        assert store.table.count_rows() == 2
        results = store.search_docs("document vector search", version="nightly", top_n=1)
        assert len(results) == 1
        assert "chunk_text" in results[0]
    finally:
        try:
            if store is not None:
                store.delete_docs(version=None)
        finally:
            reset_backends()
