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


def _cold_model_resource_probe(wrapper_count: int) -> dict:
    """Run only in a fresh interpreter: observe native construction without replacing it."""
    import gc
    import os
    import subprocess
    import sys
    import threading
    import weakref
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np
    from fastembed import TextEmbedding

    from datus.storage.fastembed_embeddings import FastEmbedEmbeddings, check_snapshot

    def resident_bytes():
        if sys.platform.startswith("linux"):
            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if sys.platform == "darwin":
            return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024
        return None

    # Warm downloaded assets, not an ONNX session. The measured cold path
    # therefore exercises model loading rather than network availability.
    settings = FastEmbedEmbeddings()
    check_snapshot(settings.name, settings.cache_dir)
    del settings
    original_init = TextEmbedding.__init__
    native_models = []
    construction_lock = threading.Lock()
    rounds = []

    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        with construction_lock:
            native_models.append(weakref.ref(self))

    before = resident_bytes()
    with patch.object(TextEmbedding, "__init__", observed_init):
        for round_index in range(3):
            wrappers = [FastEmbedEmbeddings() for _ in range(wrapper_count)]
            wrapper_refs = [weakref.ref(wrapper) for wrapper in wrappers]
            start = threading.Barrier(8)

            def embed(index, start=start, wrappers=wrappers):
                start.wait(timeout=10)
                wrapper = wrappers[index % wrapper_count]
                model = wrapper.embedding_model
                vector = wrapper.generate_embeddings(["Total order amount is 350 dollars."])[0]
                return id(model), vector

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(embed, range(8)))
            assert len({identity for identity, _ in results}) == 1
            vectors = np.asarray([vector for _, vector in results])
            assert vectors.shape == (8, 384)
            assert np.isfinite(vectors).all()
            assert np.linalg.norm(vectors[0]) > 0
            np.testing.assert_allclose(vectors, np.repeat(vectors[:1], 8, axis=0), atol=1e-6)
            del wrappers, results, vectors, embed
            gc.collect()
            assert all(reference() is None for reference in wrapper_refs), "Warm cache retained an embedding wrapper"
            live_models = sum(reference() is not None for reference in native_models)
            assert live_models == 1
            assert len(native_models) == 1, "Equivalent projects/rounds reconstructed a native model"
            rounds.append(
                {
                    "round": round_index,
                    "constructions": len(native_models),
                    "live_models": live_models,
                    "rss_bytes": resident_bytes(),
                }
            )
    return {"pid": os.getpid(), "wrappers": wrapper_count, "before_rss_bytes": before, "rounds": rounds}


@pytest.mark.parametrize("wrapper_count", [1, 8], ids=["shared-wrapper", "equivalent-wrappers"])
def test_real_fastembed_cold_concurrency_and_repeated_resource_cycles(wrapper_count, record_property):
    """Real ONNX inference shares one cold session and stays bounded over repeated rounds."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import json,sys; "
        "from tests.integration.storage.test_fastembed_lancedb_smoke import _cold_model_resource_probe; "
        "print('RESOURCE_REPORT:' + json.dumps(_cold_model_resource_probe(int(sys.argv[1]))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(wrapper_count)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=75,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = [
        line.removeprefix("RESOURCE_REPORT:")
        for line in completed.stdout.splitlines()
        if line.startswith("RESOURCE_REPORT:")
    ]
    assert len(reports) == 1, completed.stdout
    report = json.loads(reports[0])
    assert report["pid"] != os.getpid()
    record_property("fastembed_resource_cycles", json.dumps(report))
