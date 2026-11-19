# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import gc
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from datus.storage import fastembed_embeddings
from datus.storage.fastembed_embeddings import FastEmbedEmbeddings


@pytest.mark.parametrize(
    "description",
    [
        # fastembed < 0.5
        {"dim": 384, "sources": {"hf": "qdrant/all-MiniLM-L6-v2-onnx"}},
        # fastembed >= 0.5: DenseModelDescription
        SimpleNamespace(dim=384, sources=SimpleNamespace(hf="qdrant/all-MiniLM-L6-v2-onnx")),
    ],
    ids=["dict", "model-description"],
)
def test_ndims_falls_back_to_the_model_description(monkeypatch, description):
    """Without ``embedding_size`` on the model, the dim comes from its description, in either shape."""
    monkeypatch.setattr(
        fastembed_embeddings.TextEmbedding,
        "_get_model_description",
        classmethod(lambda _cls, _name: description),
    )
    embeddings = FastEmbedEmbeddings()
    embeddings._model_instance = SimpleNamespace()

    assert embeddings.ndims() == 384


@pytest.fixture
def tracked_embedding_factory(monkeypatch):
    """Replace model allocation/I/O, retaining the production sharing logic."""
    if hasattr(fastembed_embeddings, "_model_cache"):
        monkeypatch.setattr(fastembed_embeddings, "_model_cache", fastembed_embeddings._ModelCache())
    else:
        FastEmbedEmbeddings._create_embedding_model.cache_clear()
    created = []

    class Model:
        def __init__(self, **kwargs):
            created.append(self)
            self.options = kwargs

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", Model)
    monkeypatch.setattr(fastembed_embeddings, "check_snapshot", lambda *args: None)
    return created


@pytest.fixture
def cold_model_barrier(monkeypatch):
    """Rendezvous one constructor with seven real single-flight waiters.

    With broken sharing, eight constructors instead reach the same barrier,
    so the construction-count assertion still detects the regression.
    """
    barrier = threading.Barrier(8)

    class WaitingFuture(Future):
        def result(self, timeout=None):
            barrier.wait(timeout=5)
            return super().result(timeout)

    monkeypatch.setattr(fastembed_embeddings, "Future", WaitingFuture, raising=False)
    return barrier


@pytest.mark.parametrize("wrapper_count", [1, 8])
def test_concurrent_cold_model_is_shared(tracked_embedding_factory, cold_model_barrier, monkeypatch, wrapper_count):
    """Both one wrapper and equivalent wrappers construct one underlying session."""
    wrappers = [FastEmbedEmbeddings(batch_size=16 + i) for i in range(wrapper_count)]
    start = threading.Barrier(8)
    factory = fastembed_embeddings.TextEmbedding

    def held_constructor(**kwargs):
        cold_model_barrier.wait(timeout=5)
        return factory(**kwargs)

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", held_constructor)

    def load(index):
        start.wait(timeout=5)
        return wrappers[index % wrapper_count].embedding_model

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(load, range(8)))
    assert len(tracked_embedding_factory) == 1
    assert all(model is models[0] for model in models)


def test_live_model_survives_registry_lru_eviction(tracked_embedding_factory):
    """An in-use model stays discoverable after a different model warms the LRU."""
    original = FastEmbedEmbeddings(name="first")
    live = original.embedding_model
    other = FastEmbedEmbeddings(name="second").embedding_model
    assert other is not live
    assert FastEmbedEmbeddings(name="first").embedding_model is live
    assert len(tracked_embedding_factory) == 2


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_model_session_options_are_isolated(tracked_embedding_factory, monkeypatch, default_device):
    """The same model on another device/cache cannot accidentally reuse a session."""
    monkeypatch.setattr(fastembed_embeddings, "get_embedding_device", lambda: default_device)
    first, second, third = FastEmbedEmbeddings(), FastEmbedEmbeddings(), FastEmbedEmbeddings()
    first.device = third.device = "cpu"
    second.device = "cuda"
    third.cache_dir = "/different-cache"
    assert len({id(x.embedding_model) for x in (first, second, third)}) == 3
    assert len(tracked_embedding_factory) == 3


def test_model_cache_does_not_retain_wrapper(tracked_embedding_factory):
    """The warm cache owns a model, never the wrapper and its incidental fields."""
    wrapper = FastEmbedEmbeddings()
    _ = wrapper.embedding_model
    reference = weakref.ref(wrapper)
    del wrapper
    gc.collect()
    assert reference() is None


def test_failed_model_initialization_can_retry(tracked_embedding_factory, monkeypatch):
    """A failed allocation must not leave a failed/incomplete registry entry."""
    factory = fastembed_embeddings.TextEmbedding
    attempts = 0

    def flaky(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected allocation failure")
        return factory(**kwargs)

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", flaky)
    wrapper = FastEmbedEmbeddings()
    with pytest.raises(Exception, match="injected allocation failure"):
        _ = wrapper.embedding_model
    assert wrapper.embedding_model is FastEmbedEmbeddings().embedding_model
    assert attempts == 2


def test_concurrent_model_failure_wakes_waiters_and_allows_retry(
    tracked_embedding_factory, cold_model_barrier, monkeypatch
):
    """One failed native allocation reaches every waiter; the key can load again."""
    original = fastembed_embeddings.TextEmbedding
    attempts = []
    start = threading.Barrier(8)

    def failed(**kwargs):
        attempts.append(kwargs)
        cold_model_barrier.wait(timeout=5)
        raise RuntimeError("injected concurrent allocation failure")

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", failed)
    wrappers = [FastEmbedEmbeddings() for _ in range(8)]

    def load(index):
        start.wait(timeout=5)
        return wrappers[index].embedding_model

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load, index) for index in range(8)]
        errors = [future.exception(timeout=5) for future in futures]
    assert len(attempts) == 1
    assert all(error is errors[0] for error in errors)
    assert "injected concurrent allocation failure" in str(errors[0])
    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", original)
    assert wrappers[0].embedding_model is wrappers[1].embedding_model
